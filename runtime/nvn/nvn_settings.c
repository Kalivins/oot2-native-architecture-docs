// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The NVN2 front end's live keys in the settings store (oot2/settings.h), applied by hooks. hold_ticks is experiment
// E2's hold (ticket 88), the store's first live key: its hook calls nvn_present_hold on whichever thread changed it, and
// the next acquire waits for the new hold. nvn_present_hold's own reading of OOT2_HOLD_TICKS (nvn_window.c) is left for
// a program that never binds the store, such as ticket 88's tests: once bound, the call wins over that reading, and the
// store, which reads the same variable, holds the value.
#include "oot2/nvn_pace.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"

static void log_hold(int64_t ticks, const char *from) {
    if (ticks) rt_log("nvn: experiment E2: every present is held for %lld ticks, whatever the game asks for (%s)",
                      (long long)ticks, from);
    else rt_log("nvn: experiment E2 is off (%s)", from);
}

static void apply_hold(const RtSettingChange *c, void *ctx) {
    (void)ctx;
    if (c->pending) return;   // a live key is never pending
    nvn_present_hold((int)c->value);
    log_hold(c->value, c->from);
}

void nvn_settings_bind(void) {
    rt_settings_hook(RT_SET_hold_ticks, apply_hold, NULL);   // first, so no change falls between the read and the hook
    const char *from;
    rt_setting_source(RT_SET_hold_ticks, &from);
    int64_t hold = rt_setting(RT_SET_hold_ticks);
    nvn_present_hold((int)hold);
    if (hold) log_hold(hold, from);
}
