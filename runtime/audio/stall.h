// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The audio stall detector's face for the renderer (stall.c): a watchdog thread of its own that reports, by name and never by
// silence, a guest thread that waits on the renderer's frame event for longer than a bound, a wave buffer that was submitted
// and not released after a bound, and a running renderer that gets no update for a bound.
#pragma once
#include <stdint.h>

#include "audren_state.h"

// The watchdog's clock (the driver's when a test set one, else plat_now_ns): the renderer stamps submissions and updates
// with it, and the bound is read against it.
uint64_t stall_now_ns(void);
// A renderer that is open is watched; the thread runs while at least one is.
void stall_register(AudioRenderer *r);
void stall_unregister(AudioRenderer *r);
// The reports made so far (first lines of episodes, not repeats), and the longest wait on a renderer event the watchdog has seen
// (a lower bound: it looks every poll).
uint64_t stall_count(void);
uint64_t stall_event_wait_max_ns(void);
