// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The guest system counter. (The spin-wait gate, rt_spin_gate, belongs to the thread scheduler: kernel.c; the virtual
// 60 Hz display clock built on this counter is vsync.c.)
#include "core.h"
#include "plat.h"

_Static_assert(1000000000u % COUNTER_HZ == 0 && 1000000000u / COUNTER_HZ == COUNTER_NS,
               "a count is a whole number of nanoseconds");

static uint64_t g_start_ns;

__attribute__((constructor)) static void clock_start(void) { g_start_ns = plat_now_ns(); }

uint64_t counter_origin_ns(void) { return g_start_ns; }

uint64_t rt_ticks(void) { return (plat_now_ns() - g_start_ns) / COUNTER_NS; }

uint32_t rt_tick_hz(void) { return COUNTER_HZ; }
