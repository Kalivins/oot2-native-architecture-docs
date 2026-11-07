// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The virtual 60 Hz display clock (architecture D22; research 15, D2 and D3). The docked console's display refreshes
// at 60 Hz, and the game paces itself on its vsync events and on the present interval it asks for. On the host that
// display is a virtual clock taken from the host's monotonic clock through the guest system counter (rt_ticks, 31.25
// MHz), never from the monitor, whose rate may be 100 Hz (research 15, F13): no present mode, swapchain or display
// mode feeds it.
//
// Tick n is due when the system counter reaches ceil(n * 31,250,000 / 60). One tick is 520,833 1/3 counts, so every due
// time is computed from the origin and n, never by adding a rounded period: an hour is exactly 216,000 ticks. Tick 0
// is the counter's origin (process start), so a tick number means the same instant for every reader.
//
// A host thread that holds no virtual core signals every bound event once per tick, through rt_kernel_event_signal, at
// the tick's due time or later, never earlier. It starts with the first rt_vsync_bind. A wake-up a whole tick late or
// more (the host stalled) signals once, for the tick in progress, and counts the ticks it passed as skipped: no burst
// of signals to catch up, and the grid never moves.
#pragma once
#include <stdint.h>

#define RT_VSYNC_HZ 60u

// Tick n's due time: the system-counter value (rt_ticks), and the same instant on plat_now_ns's clock (on the driver's
// clock when the clock is driven).
uint64_t rt_vsync_due_counter(uint64_t n);
uint64_t rt_vsync_due_ns(uint64_t n);
// The current tick: the n whose due time has come and whose successor's has not (0 before tick 1 is due).
uint64_t rt_vsync_tick(void);

// Block the calling host thread until the current tick is n or later (it returns at once when it already is). The wait is
// on the clock alone, never on the clock thread or its events, and never early: undriven it sleeps until the due time
// of tick n on the host's clock, driven it follows the driver's time (a test moves it). The caller gives its virtual
// core up around the call when it is a guest thread (rt_kernel_block_begin); the NVN2 front end's SyncWait on a window's
// acquire sync (runtime/nvn/nvn_queue.c) and, under the old placement, its AcquireTexture (runtime/nvn/nvn_window.c) are
// the callers.
void rt_vsync_wait_tick(uint64_t n);

// The clock's own time in nanoseconds: the driver's when the clock is driven, plat_now_ns otherwise. A timeout that
// belongs with a tick is measured on this clock.
uint64_t rt_vsync_now_ns(void);
// rt_vsync_wait_tick with a timeout (relative, on rt_vsync_now_ns's clock; UINT64_MAX: none): 1 when the current tick
// is n or later, 0 when the timeout ran out first. A timeout of 0 never waits. The NVN2 front end's SyncWait waits on a
// window's acquire sync this way (runtime/nvn/nvn_queue.c).
int rt_vsync_wait_tick_for(uint64_t n, uint64_t timeout_ns);

// Signal the event whose writable end this is at every tick from the next one (rt_kernel_event_create, or a system
// event the service layer bound). The same event bound twice is signalled once per tick. At most 8 events.
void rt_vsync_bind(uint32_t writable);

typedef struct RtVsyncStats {
    uint64_t ticks;          // ticks signalled
    uint64_t skipped;        // ticks passed by wake-ups a whole tick late or more
    uint64_t last;           // the last tick signalled; 0 before the first
    uint64_t late_max_ns;    // the latest a tick's signals were all sent after its due time
    int events;              // events bound
} RtVsyncStats;
RtVsyncStats rt_vsync_stats(void);

// Driven mode, for tests: the clock reads its time from now() and waits through wait_until() instead of the host's
// clock and sleep, so a test runs an hour of ticks in seconds and puts every wake-up where it wants. origin_ns is the
// time of tick 0 on now()'s clock. Set once, before the first rt_vsync_bind (a fatal after it). The clock thread is
// the only caller of wait_until, which returns at deadline_ns or later on now()'s clock.
typedef struct RtVsyncDriver {
    uint64_t origin_ns;
    uint64_t (*now)(void *ctx);
    void (*wait_until)(void *ctx, uint64_t deadline_ns);
    void *ctx;
} RtVsyncDriver;
void rt_vsync_drive(const RtVsyncDriver *d);
