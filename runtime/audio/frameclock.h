// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The renderer's frame clock: a grid of ticks, tick k due at origin + k x sample_count x 1e9 / sample_rate (rounded up),
// computed from the origin and k, never by adding a rounded period. A host thread that holds no virtual core calls the
// tick function at each due time or later, never earlier. A wake-up a whole tick late or more calls it once for the
// tick in progress and counts the ticks it passed: no burst to catch up, and the grid never moves.
#pragma once
#include <stdint.h>

typedef struct FrameClock FrameClock;
// tick is the tick number (1, 2, ...), skipped the ticks passed since the previous call (0 when on time).
typedef void (*FrameTick)(void *ctx, uint64_t tick, uint64_t skipped);

typedef struct FrameClockStats {
    uint64_t ticks, skipped, last;
    uint64_t late_max_ns;   // the latest a tick's function returned after the tick's due time
} FrameClockStats;

// The clock's time source (the driver's when one is set, else the host's monotonic clock), for the code that stamps
// events on the same clock.
uint64_t frameclock_now_ns(void);
// The origin is taken now. NULL (and a log line) when the thread cannot start.
FrameClock *frameclock_start(uint32_t sample_count, uint32_t sample_rate, FrameTick tick, void *ctx);
// Ends the thread and waits for it: when this returns the tick function is not running and will not run again. Frees.
void frameclock_stop(FrameClock *fc);
FrameClockStats frameclock_stats(const FrameClock *fc);
// Tick k's due time on frameclock_now_ns's clock.
uint64_t frameclock_due_ns(const FrameClock *fc, uint64_t k);
