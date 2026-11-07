// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The acquire gate's records and its settings (architecture D22; runtime/nvn/nvn_window.c, runtime/nvn/nvn_queue.c). The
// gate has no interface of its own: it is nvnWindowAcquireTexture and the sync that call returns, on the virtual 60 Hz
// clock (oot2/vsync.h).
//
// The rule, in ticks of that clock. A present takes the tick in progress when QueuePresentTexture is called (never
// earlier than the window's previous present): its due tick. The next AcquireTexture of the window is released at the due
// tick plus the present interval, read from the window when that acquire is called (the gate tick). The first acquire of
// a window, and any acquire while no present has happened since the window was initialised, is released at once.
// Interval 0 or less releases at once. A present that arrives late takes the tick it arrives in; the next acquire is then
// released at that tick plus the interval, so ticks keep the clock's own phase (the grid never moves) and there is no
// burst of immediate acquires to catch up.
//
// Where the release is (ticket 95, research 52 Q1). The default, NVN_ACQUIRE_WAIT_SYNC, is the console's: the call
// chooses the index and returns at once, and the sync it returns is signalled at the gate tick (the clock has reached it
// and the texture's last present is done); a guest thread that waits on the sync (nvnSyncWait) gives its virtual core up
// for the wait, with no lock held. A game that passes no sync is held for nothing. A present that comes before the wait
// on the sync it acquired takes the gate tick of that acquire if the clock is not there yet (the later of the tick in
// progress and the acquire's gate tick), so a frame still takes one tick per interval whichever side of the wait the
// game presents on. The old placement, NVN_ACQUIRE_WAIT_ACQUIRE, blocks inside the acquire until the gate tick (the
// guest thread's virtual core given up) and signals the sync at once; there the present's due tick is the tick in
// progress, as before. Both placements use the same gate tick, so the E2 hold below works under both.
//
// A fixed frame rate (ticket 111, research 94 F11). The settings store's key frame_rate_target N (0, the default: off;
// 20 to 120; live, read at every acquire and present) gives the gate a second schedule of N slots a second on the same
// origin as the clock's ticks (oot2/vsync.h, rt_sched_*): slot m is due at the system-counter count ceil(m * 31,250,000 /
// N), computed from m, never as a sum of periods. A present takes the slot in progress (its due count is when that slot
// began, or the gate count of its acquire if that is later), the next acquire is released at the slot after the one the
// due instant falls in, and the game's interval is ignored while a target is set: the target's period replaces it, and the
// record keeps what the game asked for (requested) beside the target (target). The E2 hold, in ticks of the 60 Hz clock,
// floors the period in whole slots, rounded up, so the gate stays on the schedule and is never shorter than the hold. The
// gate counts, win_t.pace and target, and the sync's ready are system-counter counts, not ticks, so a change of N moves
// the gate to the new schedule at once: the slot is taken from the due count by the rate in force at the acquire. With no
// target the count of a tick is its due count and everything above holds tick for tick. At 30 and 20 the schedule is every
// second and third tick, and at 60 it is the ticks themselves. The game's vsync events stay on the clock thread's 60 Hz
// ticks whatever N is; the wait on a count is one timed sleep on the waiting thread, never a thread of its own.
#pragma once
#include <stdint.h>

// What one present left behind. requested is the interval the game had set when it presented, as written; held is the
// experiment hold in effect then (0: none), which the gate used for the next acquire and which never enters requested;
// placement is the NvnAcquireWait in effect at the present.
typedef struct NvnPresentRecord {
    uint64_t seq;        // 1 for the first present of the process, across windows
    uint64_t due;        // the tick the present took (under the default placement it can be a tick still to come: see above)
    uint64_t due_count;  // the same instant exactly, in system-counter counts from the clock's origin (oot2/vsync.h)
    uint64_t shown;      // the tick the back end says it was shown at; 0 when unknown (the null back end, not yet)
    int32_t requested;   // the window's present interval at the call
    int32_t held;        // the hold in effect at the call
    int32_t texture;     // the window texture index presented
    int32_t placement;   // NvnAcquireWait in effect at the call
    int32_t target;      // frame_rate_target in force at the call (0: none)
    int32_t pad;         // zero; the setting-events record is to take it
} NvnPresentRecord;

#define NVN_PRESENT_RING 512   // records kept: the newest this many presents

uint64_t nvn_present_count(void);   // presents so far
// The record of present `seq` into *out: 1, or 0 when seq is 0, not yet made, or older than the ring. A shown tick not
// known yet is asked of the back end (GfxBackend.present_shown, which never blocks) on each read until it answers.
int nvn_present_record(uint64_t seq, NvnPresentRecord *out);

// Where the pacing hold of AcquireTexture is (ticket 95): the settings store's key hold_placement (oot2/settings.h; the
// environment variable OOT2_HOLD_PLACEMENT, the file, --set; values sync and acquire, live), read at every acquire and
// present, so a change applies from the next one; a sync an earlier acquire returned keeps the tick it was given. The
// enum's values are the key's indexes (nvn_settings.c checks).
typedef enum NvnAcquireWait {
    NVN_ACQUIRE_WAIT_SYNC = 0,      // the default: the call returns at once, its sync is signalled at the gate tick
    NVN_ACQUIRE_WAIT_ACQUIRE = 1,   // the old placement: the call blocks until the gate tick
} NvnAcquireWait;
// Override the store's key: a program or a test that wants one placement whatever the layers say. Anything else than the
// two values is a fatal naming it. Once called, the key is no longer read.
void nvn_acquire_wait(int mode);
// The placement in effect: nvn_acquire_wait's mode if it was called, else the store's key (a program that never loads
// the store reads OOT2_HOLD_PLACEMENT and the default; a value the key does not take ends the process by name with
// exit code 2).
int nvn_acquire_wait_mode(void);

// Experiment E2 (research 15): hold every present for `ticks` ticks while the game still asks for its own interval. The
// next acquire waits for max(interval, ticks). Off (0) unless this is called with a positive count, or the environment
// variable OOT2_HOLD_TICKS names one at the first acquire, present or nvn_present_hold_ticks call, if this was never called. Never a default;
// the records keep what the game asked for.
void nvn_present_hold(int ticks);
int nvn_present_hold_ticks(void);
