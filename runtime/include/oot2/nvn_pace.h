// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The acquire gate's records and its experiment setting (architecture D22; runtime/nvn/nvn_window.c). The gate itself
// has no interface: it is nvnWindowAcquireTexture, which blocks on the virtual 60 Hz clock (oot2/vsync.h).
//
// The rule, in ticks of that clock. A present takes the tick in progress when QueuePresentTexture is called (never
// earlier than the window's previous present): its due tick. The window's next AcquireTexture returns once the clock
// has reached the due tick plus the present interval, read from the window when that acquire is called. The first
// acquire of a window, and any acquire while no present has happened since the window was initialised, waits for
// nothing. Interval 0 or less waits for nothing. A present that arrives late takes the tick it arrives in; the next
// acquire then waits for that tick plus the interval, so ticks keep the clock's own phase (the grid never moves) and
// there is no burst of immediate acquires to catch up.
#pragma once
#include <stdint.h>

// What one present left behind. requested is the interval the game had set when it presented, as written; held is the
// experiment hold in effect then (0: none), which the gate used for the next acquire and which never enters requested.
typedef struct NvnPresentRecord {
    uint64_t seq;        // 1 for the first present of the process, across windows
    uint64_t due;        // the tick the present took
    uint64_t shown;      // the tick the back end says it was shown at; 0 when unknown (the null back end, not yet)
    int32_t requested;   // the window's present interval at the call
    int32_t held;        // the hold in effect at the call
    int32_t texture;     // the window texture index presented
    int32_t pad;
} NvnPresentRecord;

#define NVN_PRESENT_RING 512   // records kept: the newest this many presents

uint64_t nvn_present_count(void);   // presents so far
// The record of present `seq` into *out: 1, or 0 when seq is 0, not yet made, or older than the ring. A shown tick not
// known yet is asked of the back end (GfxBackend.present_shown, which never blocks) on each read until it answers.
int nvn_present_record(uint64_t seq, NvnPresentRecord *out);

// Experiment E2 (research 15): hold every present for `ticks` ticks while the game still asks for its own interval. The
// next acquire waits for max(interval, ticks). Off (0) unless this is called with a positive count, or the environment
// variable OOT2_HOLD_TICKS names one at the first acquire, present or nvn_present_hold_ticks call, if this was never called. Never a default;
// the records keep what the game asked for.
void nvn_present_hold(int ticks);
int nvn_present_hold_ticks(void);
