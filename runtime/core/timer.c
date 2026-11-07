// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The policy of the system timer resolution (the timer resolution hold): whether to hold 0.5 ms while the game runs, which the setting
// timer_resolution answers (oot2/settings.h; on by default; OOT2_TIMER_RESOLUTION=0 or --set timer_resolution=0 leaves
// the system's), and the one log line that says what happened. The request itself and its release on every exit path are
// the platform layer's (plat.h, plat_timer_hold); this file only decides and reports.
//
// The vsync clock study measured on the desktop that the p99 of the vsync events and of the acquire gate falls from 1.0 to 0.51 ms
// when the process holds 0.5 ms (NtSetTimerResolution(5000)), against whatever another process happens to hold. The cost
// is a higher timer interrupt rate, system-wide, for as long as the game runs: hence a setting.
#include <stdio.h>

#include "core.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "plat.h"

#define TIMER_WANT_NS 500000ull   // 0.5 ms

static void ms(char out[16], uint64_t ns) { snprintf(out, 16, "%.3f ms", (double)ns / 1e6); }

void rt_timer_hold_start(void) {
    const char *from;
    rt_setting_source(RT_SET_timer_resolution, &from);
    if (!rt_setting(RT_SET_timer_resolution)) {
        rt_log("timer: the system's resolution is left as it is, not requested (timer_resolution = 0, %s)", from);
        return;
    }
    uint64_t granted = 0;
    char want[16], got[16], now[16];
    ms(want, TIMER_WANT_NS);
    if (!plat_timer_hold(TIMER_WANT_NS, &granted)) {
        rt_log("timer: the system would not hold %s; its resolution stays as it is", want);
        return;
    }
    if (!granted) {   // the platform has no resolution to report (POSIX): nothing was held
        rt_log("timer: this platform has no system timer resolution to hold (timer_resolution = 1, %s)", from);
        return;
    }
    ms(got, granted);
    ms(now, plat_timer_resolution_ns());
    rt_log("timer: holding %s (the system granted %s, %s in force now; timer_resolution = 1, %s)", want, got, now, from);
}
