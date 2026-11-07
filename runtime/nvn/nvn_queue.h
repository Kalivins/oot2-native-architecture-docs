// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Internal header of the queue and window files: nvn_queue.c (queues, syncs, events, timestamps) and nvn_window.c
// (windows and QueuePresentTexture).
#pragma once
#include "nvn_api.h"

// The queue or sync argument i of entry point fn, checked: a named stop when it was never initialised (or was
// finalized).
queue_t *nvn_queue_arg(Cpu *c, int i, const char *fn);
sync_t *nvn_sync_arg(Cpu *c, int i, const char *fn);
// Give sync s the fence value `value`: how is 1 for a queue's fence, 2 for a window's acquire. Wakes the SyncWaits that
// wait for a sync to be fenced.
void nvn_sync_fence(sync_t *s, uint32_t how, uint64_t value);
// A window's acquire (how 2) gives the sync its fence value and, with it, the count `ready` of the system counter (from
// the virtual clock's origin, oot2/vsync.h) the sync is signalled at: a tick of the 60 Hz clock, or a slot of the fixed
// rate's schedule (0: no such condition). Fence value and count are set together, so a wait sees one acquire's pair.
// nvn_sync_fence clears the count.
void nvn_sync_acquire(sync_t *s, uint64_t value, uint64_t ready);

// The pacing trace (nvn_trace.c): off unless OOT2_PACE_TRACE names a file. nvn_trace writes one line (event, wall
// nanoseconds, the clock's tick, then the formatted body, then the probes' values when probes is not 0).
int nvn_trace_on(void);
void nvn_trace(const char *event, int probes, const char *fmt, ...) __attribute__((format(printf, 3, 4)));
