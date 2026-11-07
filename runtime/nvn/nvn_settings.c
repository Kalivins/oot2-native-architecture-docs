// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The NVN2 front end's live keys in the settings store (oot2/settings.h), applied by hooks. hold_ticks is experiment
// E2's hold (ticket 88), the store's first live key: its hook calls nvn_present_hold on whichever thread changed it, and
// the next acquire waits for the new hold. nvn_present_hold's own reading of OOT2_HOLD_TICKS (nvn_window.c) is left for
// a program that never binds the store, such as ticket 88's tests: once bound, the call wins over that reading, and the
// store, which reads the same variable, holds the value. hold_placement (ticket 95) is read by nvn_window.c from the
// store at every acquire and present, so its hook only says when it changed; nvn_acquire_wait overrides it.
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

// The key's names are the placement enum's values.
_Static_assert(NVN_ACQUIRE_WAIT_SYNC == 0 && NVN_ACQUIRE_WAIT_ACQUIRE == 1, "hold_placement's names are sync|acquire");

static void log_placement(int64_t placement, const char *from) {
    if (placement == NVN_ACQUIRE_WAIT_ACQUIRE)
        rt_log("nvn: AcquireTexture holds inside the call (hold_placement = acquire, %s)", from);
    else rt_log("nvn: AcquireTexture returns at once and its sync is signalled at the due tick (hold_placement = sync, %s)",
                from);
}

static void apply_placement(const RtSettingChange *c, void *ctx) {
    (void)ctx;
    if (c->pending) return;   // a live key is never pending
    log_placement(c->value, c->from);
}

void nvn_settings_bind(void) {
    // the hooks first, so no change falls between a read and its hook
    rt_settings_hook(RT_SET_hold_ticks, apply_hold, NULL);
    rt_settings_hook(RT_SET_hold_placement, apply_placement, NULL);
    const char *from;
    rt_setting_source(RT_SET_hold_ticks, &from);
    int64_t hold = rt_setting(RT_SET_hold_ticks);
    nvn_present_hold((int)hold);
    if (hold) log_hold(hold, from);
    int64_t placement = rt_setting(RT_SET_hold_placement);
    rt_setting_source(RT_SET_hold_placement, &from);
    if (placement) log_placement(placement, from);
}
