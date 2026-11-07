// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The virtual 60 Hz display clock (oot2/vsync.h; architecture D22). Its schedule is a function of the guest system
// counter alone; a host thread, started by the first bind, signals the bound events at every tick.
//
// Adapted from oot-pc-recomp runtime/gfx_present.c at 8e09aaa39c2cff5c3ca7c2a4ee36345745eebe17, section "vsync"
// (vsync_main, vsync_ensure, gfx_vsync_bind, gfx_vsync_ticks, gfx_vsync_time_ns; no licence header there: the
// repository's GPL-3.0-or-later; its file header takes the vsync behaviour from MacNativeGames
// games/totk/runtime/frontend.m at b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026 MacNativeGames
// contributors). Kept: a thread above normal priority that waits for the due time of tick n, computed from the start
// and n, then signals every bound event; the thread starts with the first bind. Changed here: the schedule is counted
// in system-counter counts from the counter's origin, so a tick number means one instant for every reader; a wake-up a
// whole tick late or more signals once for the tick in progress and counts the ticks it passed, where the donor
// restarted its schedule after a 100 ms stall (its grid moved); the events are kernel events signalled through
// rt_kernel_event_signal, at most 8, and there are no vsync counters; a test can drive the time source and the wait;
// the ticks, skips and lateness are counted.
#include "core.h"
#include "oot2/kernel.h"
#include "oot2/vsync.h"
#include "plat.h"

#define MAX_EVENTS 8
#define CLOCK_STACK (1ull << 20)

static PlatMutex g_lock = PLAT_MUTEX_INIT;   // g_ev, g_nev, g_started, and the driver while it is set
static uint32_t g_ev[MAX_EVENTS];
static int g_nev, g_started;
static RtVsyncDriver g_drv;                  // written once, before g_driven is set and before the clock starts
static int g_driven;
static uint64_t g_ticks, g_skipped, g_last, g_late_max;   // written by the clock thread only

static int driven(void) { return __atomic_load_n(&g_driven, __ATOMIC_ACQUIRE); }
static uint64_t origin_ns(void) { return driven() ? g_drv.origin_ns : counter_origin_ns(); }
static uint64_t now_ns(void) { return driven() ? g_drv.now(g_drv.ctx) : plat_now_ns(); }

// ceil(n * COUNTER_HZ / RT_VSYNC_HZ): from the origin and n alone, never a sum of rounded periods
uint64_t rt_vsync_due_counter(uint64_t n) {
    return (uint64_t)(((unsigned __int128)n * COUNTER_HZ + RT_VSYNC_HZ - 1) / RT_VSYNC_HZ);
}

uint64_t rt_vsync_due_ns(uint64_t n) { return origin_ns() + rt_vsync_due_counter(n) * COUNTER_NS; }

// floor(counter * RT_VSYNC_HZ / COUNTER_HZ) is the n with due_counter(n) <= counter < due_counter(n + 1)
uint64_t rt_vsync_tick(void) {
    uint64_t now = now_ns(), origin = origin_ns();
    if (now < origin) return 0;
    uint64_t counter = (now - origin) / COUNTER_NS;
    return (uint64_t)((unsigned __int128)counter * RT_VSYNC_HZ / COUNTER_HZ);
}

// The host's wait: absolute deadline, never early (plat_sleep_ns is late by up to PLAT_SLEEP_LATE_NS, never early, and
// the loop covers a wake-up that comes back sooner anyway).
static void host_wait_until(uint64_t due) {
    for (uint64_t now = plat_now_ns(); now < due; now = plat_now_ns()) plat_sleep_ns(due - now);
}

static void wait_until(uint64_t due) {
    if (driven()) g_drv.wait_until(g_drv.ctx, due);
    else host_wait_until(due);
}

// Driven, the time moves only when the test moves it: poll rt_vsync_tick(), yielding for the first stretch and then
// sleeping, so a lockstep driver finds the waiter quickly and an idle one costs no processor.
void rt_vsync_wait_tick(uint64_t n) {
    if (!driven()) {
        while (rt_vsync_tick() < n) host_wait_until(rt_vsync_due_ns(n));
        return;
    }
    uint64_t start = plat_now_ns();
    while (rt_vsync_tick() < n) {
        if (plat_now_ns() - start < 50000000ull) plat_yield();
        else plat_sleep_ns(1000000ull);
    }
}

uint64_t rt_vsync_now_ns(void) { return now_ns(); }

// With a timeout the wait is measured on the clock the tick is (now_ns: the driver's when driven), so a test's time and
// the deadline agree; undriven, the sleep is cut at the deadline and never runs past the due time of tick n.
int rt_vsync_wait_tick_for(uint64_t n, uint64_t timeout_ns) {
    if (rt_vsync_tick() >= n) return 1;
    if (timeout_ns == UINT64_MAX) {
        rt_vsync_wait_tick(n);
        return 1;
    }
    uint64_t deadline = now_ns() + timeout_ns;
    if (deadline < timeout_ns) deadline = UINT64_MAX;   // saturate
    uint64_t start = plat_now_ns();
    while (rt_vsync_tick() < n) {
        uint64_t now = now_ns();
        if (now >= deadline) return 0;
        if (!driven()) {
            uint64_t due = rt_vsync_due_ns(n);
            host_wait_until(due < deadline ? due : deadline);
        } else if (plat_now_ns() - start < 50000000ull) {
            plat_yield();
        } else {
            plat_sleep_ns(1000000ull);
        }
    }
    return 1;
}

static void clock_main(void *arg) {
    (void)arg;
    plat_thread_raise_priority();   // where the platform allows it: a wake-up on time matters more than fairness here
    uint64_t n = rt_vsync_tick() + 1, ticks = 0, skipped = 0, late_max = 0;
    uint32_t ev[MAX_EVENTS];
    for (;;) {
        wait_until(rt_vsync_due_ns(n));
        // A wake-up a whole tick late or more: one signal, for the tick in progress. The ticks passed are counted and
        // never signalled in a burst, and the next deadline stays on the grid.
        uint64_t current = rt_vsync_tick();
        if (current > n) {
            skipped += current - n;
            n = current;
        }
        plat_mutex_lock(&g_lock);
        int k = g_nev;
        for (int i = 0; i < k; i++) ev[i] = g_ev[i];
        plat_mutex_unlock(&g_lock);
        for (int i = 0; i < k; i++) rt_kernel_event_signal(ev[i]);
        uint64_t sent = now_ns(), due = rt_vsync_due_ns(n), late = sent > due ? sent - due : 0;
        if (late > late_max) late_max = late;
        ticks++;
        __atomic_store_n(&g_skipped, skipped, __ATOMIC_RELAXED);
        __atomic_store_n(&g_late_max, late_max, __ATOMIC_RELAXED);
        __atomic_store_n(&g_last, n, __ATOMIC_RELAXED);
        __atomic_store_n(&g_ticks, ticks, __ATOMIC_RELEASE);
        n++;
    }
}

void rt_vsync_bind(uint32_t writable) {
    if (!writable) rt_fatal("rt_vsync_bind: handle 0 is no event");
    int known = 0, full = 0, start = 0;
    plat_mutex_lock(&g_lock);
    for (int i = 0; i < g_nev; i++) known |= g_ev[i] == writable;
    full = !known && g_nev == MAX_EVENTS;
    if (!known && !full) g_ev[g_nev++] = writable;
    if (!full && !g_started) g_started = start = 1;
    plat_mutex_unlock(&g_lock);
    if (full) rt_fatal("vsync: more than %d events bound to the 60 Hz clock", MAX_EVENTS);
    if (start && !plat_thread_start(clock_main, NULL, CLOCK_STACK)) rt_fatal("vsync: cannot start the clock thread");
}

void rt_vsync_drive(const RtVsyncDriver *d) {
    if (!d || !d->now || !d->wait_until) rt_fatal("rt_vsync_drive: a driver needs now and wait_until");
    plat_mutex_lock(&g_lock);
    int refused = g_started || driven();
    if (!refused) {
        g_drv = *d;
        __atomic_store_n(&g_driven, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&g_lock);
    if (refused) rt_fatal("rt_vsync_drive: the clock is already %s", g_started ? "running" : "driven");
}

RtVsyncStats rt_vsync_stats(void) {
    RtVsyncStats s = {.ticks = __atomic_load_n(&g_ticks, __ATOMIC_ACQUIRE),
                      .skipped = __atomic_load_n(&g_skipped, __ATOMIC_RELAXED),
                      .last = __atomic_load_n(&g_last, __ATOMIC_RELAXED),
                      .late_max_ns = __atomic_load_n(&g_late_max, __ATOMIC_RELAXED)};
    plat_mutex_lock(&g_lock);
    s.events = g_nev;
    plat_mutex_unlock(&g_lock);
    return s;
}
