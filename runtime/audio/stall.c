// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The audio stall detector (stall.h). One watchdog thread, mostly asleep, polls every 50 ms on its own clock (a test can
// drive it), because the renderer's own thread may be the thing that stalled. It reports, once per episode and then once
// every 10 s while the episode lasts, never silence and never a stop:
//   1. a guest thread that waits on the renderer's frame event for longer than the bound (AUDIO_STALL_BOUND_NS, 1 s: 200
//      frames of 5 ms; a healthy wait is a frame plus scheduler lateness plus the longest host hitch, and the audio path
//      never waits for the GPU or a pipeline build). The kernel's waiter view (rt_kernel_waiters) gives who waits and since
//      when; a thread that polls with a timeout is one episode for its whole run of timeouts.
//   2. a wave buffer of a started voice that was submitted and is not released after its own duration (the samples left over
//      the voice's rate x pitch), plus the buffers queued ahead of it, plus the bound, counted from the later of its
//      submission and the voice's start. The bound is wall time, so it keeps running when the renderer stops rendering.
//   3. a running renderer that has had updates and gets none for the bound: the client cannot see a release without one.
//
// What the host can see of a thread that polls a wave buffer's state in guest memory without waiting: not the address (guest
// loads are plain loads), only that it runs (its CPU time against its core time, the spin gate's passes, its SleepThread
// calls). Its stall is reported from the renderer's side, by report 2 with the thread that submitted the buffer and by
// report 3 when nobody calls RequestUpdate; it is not claimed more precisely than that.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "stall.h"
#include "oot2/kernel.h"
#include "oot2/runtime.h"
#include "plat.h"

#define POLL_NS 50000000ull
#define REPEAT_NS 10000000000ull
#define MAX_WATCHED 8
#define MAX_EPISODES 256
#define CANDIDATES 32
#define LOCK_TRIES 40               // 40 tries 100 us apart: a frame or an update holds the lock for far less than 4 ms
#define LOCK_TRY_NS 100000ull

enum { EP_EVENT = 1, EP_BUFFER, EP_UPDATE, EP_LOCK };
typedef struct {
    int used;
    int kind;
    uint64_t k1, k2, k3;
    uint64_t first_seen, reported_at, mark;
    int reported;
} Episode;

static PlatMutex g_lock = PLAT_MUTEX_INIT;   // the watched set, the episodes, and the poll
static AudioRenderer *g_watched[MAX_WATCHED];
static int g_refs;
// One watchdog thread while a renderer is watched. Its stop and done flags are its own (atomic), so that a renderer opened
// while the last one is closing starts a new thread and cannot disturb the one that is ending.
typedef struct Watchdog { PlatThread *thread; int stop, done; } Watchdog;
static Watchdog *g_dog;
static Episode g_ep[MAX_EPISODES];
static uint64_t g_poll_no, g_stalls, g_wait_max, g_polled_ns, g_polls;
static int g_table_full_logged;

// ---- the clock
static PlatMutex g_drv_lock = PLAT_MUTEX_INIT;
static AudioClockDriver g_drv;
static int g_driven;

void audio_watchdog_drive(const AudioClockDriver *d) {
    plat_mutex_lock(&g_drv_lock);
    if (d) {
        if (!d->now || !d->wait_until) {
            plat_mutex_unlock(&g_drv_lock);
            rt_fatal("audio_watchdog_drive: a driver needs now and wait_until");
        }
        g_drv = *d;
    }
    __atomic_store_n(&g_driven, d != NULL, __ATOMIC_RELEASE);
    plat_mutex_unlock(&g_drv_lock);
}

uint64_t stall_now_ns(void) { return __atomic_load_n(&g_driven, __ATOMIC_ACQUIRE) ? g_drv.now(g_drv.ctx) : plat_now_ns(); }

static void wait_until(uint64_t due, const int *stop) {
    if (__atomic_load_n(&g_driven, __ATOMIC_ACQUIRE)) {
        g_drv.wait_until(g_drv.ctx, due);
        return;
    }
    for (uint64_t now = plat_now_ns(); now < due && !__atomic_load_n(stop, __ATOMIC_ACQUIRE); now = plat_now_ns()) {
        uint64_t left = due - now;
        plat_sleep_ns(left < 10000000ull ? left : 10000000ull);
    }
}

// ---- names
static void thread_text(uint32_t handle, uint64_t entry, int have_entry, char *out, size_t n) {
    char name[96] = "unknown";
    if (!handle) {
        snprintf(out, n, "none (a host thread)");
        return;
    }
    if (!have_entry) {
        RtThreadTime t[64];
        int cnt = rt_kernel_thread_times(t, 64);
        for (int i = 0; i < cnt && i < 64; i++)
            if (t[i].handle == handle) {
                entry = t[i].entry;
                have_entry = 1;
            }
    }
    if (have_entry) rt_guest_name(entry, name, sizeof name);
    snprintf(out, n, "%#x (entry %s)", handle, name);
}

// ---- episodes
static Episode *episode(int kind, uint64_t k1, uint64_t k2, uint64_t k3, uint64_t now) {
    Episode *free_slot = NULL;
    for (int i = 0; i < MAX_EPISODES; i++) {
        Episode *e = &g_ep[i];
        if (!e->used) {
            if (!free_slot) free_slot = e;
        } else if (e->kind == kind && e->k1 == k1 && e->k2 == k2 && e->k3 == k3) {
            e->mark = g_poll_no;
            return e;
        }
    }
    if (!free_slot) {
        if (!g_table_full_logged) {
            g_table_full_logged = 1;
            rt_log("audio: stall: more than %d episodes at once; the rest are not tracked", MAX_EPISODES);
        }
        return NULL;
    }
    *free_slot = (Episode){.used = 1, .kind = kind, .k1 = k1, .k2 = k2, .k3 = k3, .first_seen = now, .mark = g_poll_no};
    return free_slot;
}

// 1 when the line is due: not reported yet, or reported REPEAT_NS ago
static int due(Episode *e, uint64_t now) {
    if (e->reported && now - e->reported_at < REPEAT_NS) return 0;
    if (!e->reported) g_stalls++;
    e->reported = 1;
    e->reported_at = now;
    return 1;
}

// ---- one renderer
typedef struct {
    int voice, slot;
    uint32_t samples, thread;
    uint64_t sub_ns, frames_since;
    const char *vstate;
} Late;

static const char *vstate_name(int s) { return s == SV_STARTED ? "started" : s == SV_PAUSED ? "paused" : "stopped"; }

static uint64_t duration_ns(uint64_t samples, const ServerVoice *v) {
    double rate = (double)v->in.sample_rate * voice_pitch(v);
    return rate > 0 ? (uint64_t)((double)samples * 1e9 / rate) : 0;
}

// now less t, or 0 when t is later than now (a stamp made after the poll read its clock)
static uint64_t since(uint64_t now, uint64_t t) { return now > t ? now - t : 0; }

// The renderer's lock, tried for a few ms and never waited for: the watchdog is a thread of its own so that it can say the
// renderer's thread is the one that stalled, and a wait on that thread's lock would silence it. 1 when it holds the lock.
static int renderer_lock(AudioRenderer *r) {
    for (int i = 0; i < LOCK_TRIES; i++) {
        if (plat_mutex_trylock(&r->lock)) return 1;
        plat_sleep_ns(LOCK_TRY_NS);
    }
    return 0;
}

// The wave buffers of started voices that are late, with the renderer's lock held: at most CANDIDATES of them.
static uint64_t length_of(const AudrenWaveBuffer *b) { return b->end > b->start ? (uint64_t)((int64_t)b->end - b->start) : 0; }

static int scan_voices(const AudioRenderer *r, uint64_t now, Late *late) {
    const uint64_t frames = r->elapsed_frames;
    int nlate = 0;
    for (int k = 0; k < r->cfg.voice_count && nlate < CANDIDATES; k++) {
        const ServerVoice *v = &r->voices[k];
        if (!v->in.in_use || v->state != SV_STARTED || !v->in.sample_rate) continue;
        uint64_t ahead = 0;
        for (unsigned n = 0; n < 4 && nlate < CANDIDATES; n++) {
            unsigned slot = (v->index + n) & 3;
            if (!v->valid[slot] && !v->pending[slot]) continue;
            const AudrenWaveBuffer *b = &v->buffers[slot];
            if (b->loop && b->loop_count < 0) break;   // an endless loop is never released, and holds everything behind it
            uint64_t length = length_of(b);
            if (n == 0 && v->valid[slot] && v->offset < length) length -= v->offset;
            length *= b->loop ? (uint64_t)b->loop_count + 1 : 1;
            uint64_t own = duration_ns(length, v);
            uint64_t from = v->sub_ns[slot] > v->start_ns ? v->sub_ns[slot] : v->start_ns;
            uint64_t limit = from + ahead + own + AUDIO_STALL_BOUND_NS;
            ahead += own;
            if (now >= limit)
                late[nlate++] = (Late){k, (int)slot, (uint32_t)(length_of(b)), v->sub_thread[slot],
                                       v->sub_ns[slot], frames - v->sub_frame[slot], vstate_name(v->state)};
        }
    }
    return nlate;
}

static void poll_renderer(AudioRenderer *r, uint64_t now) {
    Late late[CANDIDATES];
    int nlate = 0;
    // what the reports below need besides the voices is read without the lock (atomic loads of what the renderer stores)
    const int running = __atomic_load_n(&r->running, __ATOMIC_RELAXED);
    const uint64_t frames = __atomic_load_n(&r->elapsed_frames, __ATOMIC_RELAXED),
                   last_frame = __atomic_load_n(&r->last_frame_wd, __ATOMIC_RELAXED),
                   last_update = __atomic_load_n(&r->last_update_wd, __ATOMIC_RELAXED),
                   update_thread = __atomic_load_n(&r->last_update_thread, __ATOMIC_RELAXED);
    char who[160];
    if (!renderer_lock(r)) {
        // 4. the renderer's thread holds the lock (a frame, an update, a sink write that does not return): the voices cannot be
        // read, and the buffers already reported are kept as they are, not forgotten and reported again when the lock is back
        for (int i = 0; i < MAX_EPISODES; i++)
            if (g_ep[i].used && g_ep[i].kind == EP_BUFFER && g_ep[i].k1 == (uint64_t)(uintptr_t)r) g_ep[i].mark = g_poll_no;
        Episode *e = episode(EP_LOCK, (uint64_t)(uintptr_t)r, 0, 0, now);
        if (e) {
            uint64_t age = since(now, e->first_seen);
            if (age >= AUDIO_STALL_BOUND_NS && due(e, now)) {
                char ago[64] = "no frame yet";
                if (last_frame) snprintf(ago, sizeof ago, "last frame %llu ms ago", (unsigned long long)(since(now, last_frame) / 1000000));
                rt_log("audio: stall: the renderer's thread has not released the renderer's lock for %llu ms (a frame, an update or "
                       "a sink write does not return); renderer %s, %llu frames rendered, %s", (unsigned long long)(age / 1000000),
                       running ? "running" : "stopped", (unsigned long long)frames, ago);
            }
        }
    } else {
        nlate = scan_voices(r, now, late);
        plat_mutex_unlock(&r->lock);
    }

    const char *state = running ? "running" : "stopped";
    // 1. a guest thread waiting on the frame event
    if (r->readable) {
        RtWaiter w[16];
        int n = rt_kernel_waiters(r->readable, w, 16);
        for (int i = 0; i < n; i++) {
            Episode *e = episode(EP_EVENT, (uint64_t)(uintptr_t)r, w[i].thread, w[i].since_ns, now);
            if (!e) continue;
            uint64_t age = since(now, e->first_seen);
            if (age > g_wait_max) g_wait_max = age;
            if (age < AUDIO_STALL_BOUND_NS || !due(e, now)) continue;
            thread_text(w[i].thread, w[i].entry, 1, who, sizeof who);
            char ago[64] = "no frame yet";
            if (last_frame) snprintf(ago, sizeof ago, "last frame %llu ms ago", (unsigned long long)(since(now, last_frame) / 1000000));
            rt_log("audio: stall: guest thread %s has waited %llu ms on the renderer event; renderer %s, %llu frames rendered, %s",
                   who, (unsigned long long)(age / 1000000), state, (unsigned long long)frames, ago);
        }
    }
    // 2. wave buffers submitted and not released
    for (int i = 0; i < nlate; i++) {
        Episode *e = episode(EP_BUFFER, (uint64_t)(uintptr_t)r, (uint64_t)late[i].voice << 8 | (uint64_t)late[i].slot, late[i].sub_ns, now);
        if (!e || !due(e, now)) continue;
        thread_text(late[i].thread, 0, 0, who, sizeof who);
        rt_log("audio: stall: voice %d wave buffer %d (%u samples) submitted by guest thread %s %llu ms ago (%llu frames "
               "rendered since), not released; voice %s, renderer %s", late[i].voice, late[i].slot, late[i].samples, who,
               (unsigned long long)(since(now, late[i].sub_ns) / 1000000), (unsigned long long)late[i].frames_since, late[i].vstate, state);
    }
    // 3. a running renderer with no update
    if (running && last_update && since(now, last_update) >= AUDIO_STALL_BOUND_NS) {
        Episode *e = episode(EP_UPDATE, (uint64_t)(uintptr_t)r, last_update, 0, now);
        if (e && due(e, now)) {
            thread_text((uint32_t)update_thread, 0, 0, who, sizeof who);
            rt_log("audio: stall: no update for %llu ms; renderer running, %llu frames rendered; the last update came from guest "
                   "thread %s", (unsigned long long)(since(now, last_update) / 1000000), (unsigned long long)frames, who);
        }
    }
}

static void poll(uint64_t now) {
    plat_mutex_lock(&g_lock);
    if (now < g_polled_ns) {   // time went backwards (a driven clock, a poll from a test): that poll is stale
        plat_mutex_unlock(&g_lock);
        return;
    }
    g_poll_no++;
    for (int i = 0; i < MAX_WATCHED; i++)
        if (g_watched[i]) poll_renderer(g_watched[i], now);
    for (int i = 0; i < MAX_EPISODES; i++)   // what was not seen this time is over (a wait that ended, a buffer released)
        if (g_ep[i].used && g_ep[i].mark != g_poll_no) g_ep[i].used = 0;
    g_polled_ns = now;
    g_polls++;
    plat_mutex_unlock(&g_lock);
}

static void watchdog_main(void *arg) {
    Watchdog *w = arg;
    while (!__atomic_load_n(&w->stop, __ATOMIC_ACQUIRE)) {
        uint64_t now = stall_now_ns();
        poll(now);
        wait_until(now + POLL_NS, &w->stop);
    }
    __atomic_store_n(&w->done, 1, __ATOMIC_RELEASE);   // the last touch: stall_unregister frees the handle after this
}

void stall_register(AudioRenderer *r) {
    plat_mutex_lock(&g_lock);
    int i = 0;
    while (i < MAX_WATCHED && g_watched[i]) i++;
    if (i == MAX_WATCHED) {
        plat_mutex_unlock(&g_lock);
        return;   // more renderers than the watchdog follows (the renderer table's own limit is checked at its open)
    }
    g_watched[i] = r;
    if (g_refs++ == 0) {
        g_dog = calloc(1, sizeof *g_dog);
        if (g_dog) g_dog->thread = plat_thread_start(watchdog_main, g_dog, 1ull << 20);
        if (!g_dog || !g_dog->thread) {
            rt_log("audio: cannot start the stall watchdog's thread; stalls will not be reported");
            free(g_dog);
            g_dog = NULL;
        }
    }
    plat_mutex_unlock(&g_lock);
}

void stall_unregister(AudioRenderer *r) {
    plat_mutex_lock(&g_lock);
    int watched = 0;   // a renderer past MAX_WATCHED was never counted
    for (int i = 0; i < MAX_WATCHED; i++)
        if (g_watched[i] == r) g_watched[i] = NULL, watched = 1;
    for (int i = 0; i < MAX_EPISODES; i++)
        if (g_ep[i].used && g_ep[i].k1 == (uint64_t)(uintptr_t)r) g_ep[i].used = 0;
    Watchdog *dog = NULL;
    if (watched && --g_refs == 0) {
        dog = g_dog;
        g_dog = NULL;
    }
    plat_mutex_unlock(&g_lock);
    if (!dog) return;
    __atomic_store_n(&dog->stop, 1, __ATOMIC_RELEASE);
    for (;;) {
        plat_mutex_lock(&g_drv_lock);
        if (g_driven && g_drv.wake) g_drv.wake(g_drv.ctx);
        plat_mutex_unlock(&g_drv_lock);
        if (__atomic_load_n(&dog->done, __ATOMIC_ACQUIRE)) break;
        plat_sleep_ns(50000);
    }
    plat_thread_free(dog->thread);
    free(dog);
}

uint64_t stall_count(void) {
    plat_mutex_lock(&g_lock);
    uint64_t v = g_stalls;
    plat_mutex_unlock(&g_lock);
    return v;
}

uint64_t stall_event_wait_max_ns(void) {
    plat_mutex_lock(&g_lock);
    uint64_t v = g_wait_max;
    plat_mutex_unlock(&g_lock);
    return v;
}

void audio_watchdog_poll_at(uint64_t now) { poll(now); }
