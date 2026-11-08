// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The renderer's frame clock (frameclock.h; the grid rules are the virtual display clock's, runtime/core/vsync.c: due
// times from the origin and the tick number, a whole-tick-late wake-up signals once and counts what it passed, and a
// test can drive the time source and the wait).
//
// The thread is stopped by a flag the wait honours: the host's wait sleeps in slices of at most one tick, and a driven
// wait is woken through the driver's wake function. A stopped clock is joined by watching the thread's last store.
#include <stdlib.h>

#include "frameclock.h"
#include "oot2/audio.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CLOCK_STACK (1ull << 20)

static PlatMutex g_driver_lock = PLAT_MUTEX_INIT;
static AudioClockDriver g_drv;
static int g_driven;

struct FrameClock {
    uint32_t sample_count, sample_rate;
    uint64_t origin_ns;
    FrameTick tick;
    void *ctx;
    PlatThread *thread;
    int stop, done;   // atomic
    uint64_t ticks, skipped, last, late_max;   // written by the thread, read through __atomic
};

void audio_clock_drive(const AudioClockDriver *d) {
    plat_mutex_lock(&g_driver_lock);
    if (d) {
        if (!d->now || !d->wait_until) {
            plat_mutex_unlock(&g_driver_lock);
            rt_fatal("audio_clock_drive: a driver needs now and wait_until");
        }
        g_drv = *d;
    }
    __atomic_store_n(&g_driven, d != NULL, __ATOMIC_RELEASE);
    plat_mutex_unlock(&g_driver_lock);
}

static int driven(void) { return __atomic_load_n(&g_driven, __ATOMIC_ACQUIRE); }

uint64_t frameclock_now_ns(void) { return driven() ? g_drv.now(g_drv.ctx) : plat_now_ns(); }

// k x sample_count x 1e9 / sample_rate, rounded up: never earlier than the exact instant
static uint64_t offset_ns(const FrameClock *fc, uint64_t k) {
    unsigned __int128 n = (unsigned __int128)k * fc->sample_count * 1000000000ull + fc->sample_rate - 1;
    return (uint64_t)(n / fc->sample_rate);
}

uint64_t frameclock_due_ns(const FrameClock *fc, uint64_t k) { return fc->origin_ns + offset_ns(fc, k); }

// the tick in progress at time now: the k whose due time has come and whose successor's has not (0 before tick 1)
static uint64_t tick_at(const FrameClock *fc, uint64_t now) {
    if (now <= fc->origin_ns) return 0;
    unsigned __int128 n = (unsigned __int128)(now - fc->origin_ns) * fc->sample_rate;
    uint64_t k = (uint64_t)(n / ((unsigned __int128)fc->sample_count * 1000000000ull));
    while (k && offset_ns(fc, k) > now - fc->origin_ns) k--;   // the due time rounds up: step back if it is not yet due
    while (offset_ns(fc, k + 1) <= now - fc->origin_ns) k++;
    return k;
}

static void wait_until(FrameClock *fc, uint64_t due) {
    if (driven()) {
        g_drv.wait_until(g_drv.ctx, due);
        return;
    }
    for (uint64_t now = plat_now_ns(); now < due && !__atomic_load_n(&fc->stop, __ATOMIC_ACQUIRE); now = plat_now_ns()) {
        uint64_t left = due - now, slice = offset_ns(fc, 1);
        plat_sleep_ns(left < slice ? left : slice);
    }
}

static void clock_main(void *arg) {
    FrameClock *fc = arg;
    plat_thread_raise_priority();
    uint64_t k = 1, ticks = 0, skipped = 0, late_max = 0;
    while (!__atomic_load_n(&fc->stop, __ATOMIC_ACQUIRE)) {
        wait_until(fc, frameclock_due_ns(fc, k));
        if (__atomic_load_n(&fc->stop, __ATOMIC_ACQUIRE)) break;
        uint64_t now = frameclock_now_ns();
        if (now < frameclock_due_ns(fc, k)) continue;   // woken early (a driver's wake without a stop): wait again
        uint64_t current = tick_at(fc, now), passed = 0;
        if (current > k) {   // a whole tick late or more: one call for the tick in progress, and the grid stays
            passed = current - k;
            k = current;
        }
        skipped += passed;
        fc->tick(fc->ctx, k, passed);
        uint64_t sent = frameclock_now_ns(), due = frameclock_due_ns(fc, k), late = sent > due ? sent - due : 0;
        if (late > late_max) late_max = late;
        ticks++;
        __atomic_store_n(&fc->skipped, skipped, __ATOMIC_RELAXED);
        __atomic_store_n(&fc->late_max, late_max, __ATOMIC_RELAXED);
        __atomic_store_n(&fc->last, k, __ATOMIC_RELAXED);
        __atomic_store_n(&fc->ticks, ticks, __ATOMIC_RELEASE);
        k++;
    }
    __atomic_store_n(&fc->done, 1, __ATOMIC_RELEASE);   // the last touch of fc: frameclock_stop frees it after this
}

FrameClock *frameclock_start(uint32_t sample_count, uint32_t sample_rate, FrameTick tick, void *ctx) {
    FrameClock *fc = calloc(1, sizeof *fc);
    if (!fc) rt_fatal("audio: frame clock allocation");
    fc->sample_count = sample_count;
    fc->sample_rate = sample_rate;
    fc->tick = tick;
    fc->ctx = ctx;
    fc->origin_ns = frameclock_now_ns();
    PlatThread *t = plat_thread_start(clock_main, fc, CLOCK_STACK);
    if (!t) {
        free(fc);
        rt_log("audio: cannot start the frame clock's thread");
        return NULL;
    }
    fc->thread = t;   // nobody joins it: the thread's own last store says it is done, and the handle is freed then
    return fc;
}

void frameclock_stop(FrameClock *fc) {
    __atomic_store_n(&fc->stop, 1, __ATOMIC_RELEASE);
    for (;;) {
        if (driven() && g_drv.wake) g_drv.wake(g_drv.ctx);
        if (__atomic_load_n(&fc->done, __ATOMIC_ACQUIRE)) break;
        plat_sleep_ns(50000);
    }
    plat_thread_free(fc->thread);   // the thread touches neither its handle nor fc after its last store
    free(fc);
}

struct AudioTicker {
    FrameClock *fc;
    void (*fn)(void *ctx);
    void *ctx;
};

static void ticker_tick(void *ctx, uint64_t tick, uint64_t skipped) {
    (void)tick, (void)skipped;
    struct AudioTicker *t = ctx;
    t->fn(t->ctx);
}

AudioTicker *audio_ticker_start(uint32_t count, uint32_t rate, void (*fn)(void *ctx), void *ctx) {
    struct AudioTicker *t = calloc(1, sizeof *t);
    if (!t) rt_fatal("audio: ticker allocation");
    t->fn = fn;
    t->ctx = ctx;
    t->fc = frameclock_start(count, rate, ticker_tick, t);
    if (!t->fc) {
        free(t);
        return NULL;
    }
    return t;
}

void audio_ticker_stop(AudioTicker *t) {
    frameclock_stop(t->fc);
    free(t);
}

FrameClockStats frameclock_stats(const FrameClock *fc) {
    FrameClockStats s;
    s.ticks = __atomic_load_n(&fc->ticks, __ATOMIC_ACQUIRE);
    s.skipped = __atomic_load_n(&fc->skipped, __ATOMIC_RELAXED);
    s.last = __atomic_load_n(&fc->last, __ATOMIC_RELAXED);
    s.late_max_ns = __atomic_load_n(&fc->late_max, __ATOMIC_RELAXED);
    return s;
}
