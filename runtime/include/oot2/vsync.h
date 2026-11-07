// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The virtual 60 Hz display clock (architecture D22; the console frame rate study). The docked console's display refreshes
// at 60 Hz, and the game paces itself on its vsync events and on the present interval it asks for. On the host that
// display is a virtual clock taken from the host's monotonic clock through the guest system counter (rt_ticks, 31.25
// MHz), never from the monitor, whose rate may be 100 Hz (the console frame rate study): no present mode, swapchain or display
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

// Other schedules on the same origin (the fixed frame rate gate; architecture D22's fixed frame rate). A schedule of `hz` slots a second
// has its slot m due when the system counter reaches ceil(m * 31,250,000 / hz), from the origin and m alone, never by
// adding a rounded period, so an hour is exactly 3600 * hz slots whatever hz is. At RT_VSYNC_HZ its slots are the
// clock's ticks, and at a divisor of it (30, 20) slot m is tick m * 60 / hz exactly. Nothing here moves the clock
// thread, whose events stay on the 60 Hz ticks. hz is 1 or more (0 is a fatal).
uint64_t rt_sched_due_counter(uint64_t m, uint32_t hz);
// floor(counter * hz / 31,250,000): the slot of the schedule that the system-counter value `counter` falls in, whose due
// count is at or before it and whose successor's is after it.
uint64_t rt_sched_slot_of(uint64_t counter, uint32_t hz);
// The system counter now, in counts from the origin (the clock's own time: the driver's when driven), and the host time
// (rt_vsync_now_ns's clock) at which it reads `count`.
uint64_t rt_vsync_counter(void);
uint64_t rt_vsync_counter_ns(uint64_t count);
// The slot of the schedule in progress now: rt_sched_slot_of(rt_vsync_counter(), hz); at RT_VSYNC_HZ the current tick.
uint64_t rt_sched_slot(uint32_t hz);

// Block the calling host thread until the current tick is n or later (it returns at once when it already is). The wait is
// on the clock alone, never on the clock thread or its events, and never early: undriven it sleeps until the due time
// of tick n on the host's clock, driven it follows the driver's time (a test moves it). The caller gives its virtual
// core up around the call when it is a guest thread (rt_kernel_block_begin); the NVN2 front end's SyncWait on a window's
// acquire sync (runtime/nvn/nvn_queue.c) and, under the old placement, its AcquireTexture (runtime/nvn/nvn_window.c) are
// the callers. rt_vsync_wait_counter is the same wait on a due count instead of a tick, which is what a slot of any
// other schedule is: the thread sleeps until that instant (one timed sleep, no spin, no thread of its own), so a fixed
// frame rate adds no waiter that burns a processor.
void rt_vsync_wait_tick(uint64_t n);
void rt_vsync_wait_counter(uint64_t count);

// The clock's own time in nanoseconds: the driver's when the clock is driven, plat_now_ns otherwise. A timeout that
// belongs with a tick is measured on this clock.
uint64_t rt_vsync_now_ns(void);
// rt_vsync_wait_tick with a timeout (relative, on rt_vsync_now_ns's clock; UINT64_MAX: none): 1 when the current tick
// is n or later, 0 when the timeout ran out first. A timeout of 0 never waits. The NVN2 front end's SyncWait waits on a
// window's acquire sync this way (runtime/nvn/nvn_queue.c). rt_vsync_wait_counter_for waits for a due count.
int rt_vsync_wait_tick_for(uint64_t n, uint64_t timeout_ns);
int rt_vsync_wait_counter_for(uint64_t count, uint64_t timeout_ns);

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
