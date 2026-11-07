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
