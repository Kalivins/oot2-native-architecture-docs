// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The NVN2 front end's live keys in the settings store (oot2/settings.h), applied by hooks. hold_ticks is experiment
// E2's hold (the acquire gate), the store's first live key: its hook calls nvn_present_hold on whichever thread changed it, and
// the next acquire waits for the new hold. nvn_present_hold's own reading of OOT2_HOLD_TICKS (nvn_window.c) is left for
// a program that never binds the store, such as the acquire gate's tests: once bound, the call wins over that reading, and the
// store, which reads the same variable, holds the value. hold_placement (the acquire sync hold) is read by nvn_window.c from the
// store at every acquire and present, so its hook only says when it changed; nvn_acquire_wait overrides it.
// frame_rate_target (the fixed frame rate gate) is read the same way, and its hook only says when it changed.
#include <stdio.h>

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

static void log_rate(int64_t rate, const char *from) {
    if (rate) rt_log("nvn: fixed frame rate: presents are paced at %lld a second on a schedule of their own, whatever "
                     "interval the game asks for (frame_rate_target = %lld, %s)", (long long)rate, (long long)rate, from);
    else rt_log("nvn: no fixed frame rate: the game's own interval on the 60 Hz clock (frame_rate_target = 0, %s)", from);
}

static void apply_rate(const RtSettingChange *c, void *ctx) {
    (void)ctx;
    if (c->pending) return;   // a live key is never pending
    log_rate(c->value, c->from);
}

// The run header: one line in every run, default or not, naming where the pacing hold is, the E2 hold and the fixed
// frame rate. The fixed frame rate is the step override: while it is set the game's own present interval is ignored
// and presents are paced on the target's schedule (nvn_window.c). Read through the store at bind time, so it cannot
// race the hold's own first read in nvn_window.c.
static void log_header(void) {
    // each source text is copied at once: a live key's source can be rewritten by another thread's change
    const char *src;
    char pfrom[160], hfrom[160], rfrom[160];
    rt_setting_source(RT_SET_hold_placement, &src);
    int64_t placement = rt_setting(RT_SET_hold_placement);
    snprintf(pfrom, sizeof pfrom, "%s", src);
    rt_setting_source(RT_SET_hold_ticks, &src);
    int64_t hold = rt_setting(RT_SET_hold_ticks);
    snprintf(hfrom, sizeof hfrom, "%s", src);
    rt_setting_source(RT_SET_frame_rate_target, &src);
    int64_t rate = rt_setting(RT_SET_frame_rate_target);
    snprintf(rfrom, sizeof rfrom, "%s", src);
    char hold_text[48], rate_text[48];
    if (hold) snprintf(hold_text, sizeof hold_text, "%lld ticks", (long long)hold);
    else snprintf(hold_text, sizeof hold_text, "none");
    if (rate) snprintf(rate_text, sizeof rate_text, "%lld a second", (long long)rate);
    else snprintf(rate_text, sizeof rate_text, "none");
    rt_log("nvn: pacing: placement %s (%s), E2 hold %s (%s), frame rate target %s (%s)",
           placement == NVN_ACQUIRE_WAIT_ACQUIRE ? "acquire" : "sync", pfrom, hold_text, hfrom, rate_text, rfrom);
}

void nvn_settings_bind(void) {
    // the hooks first, so no change falls between a read and its hook
    rt_settings_hook(RT_SET_hold_ticks, apply_hold, NULL);
    rt_settings_hook(RT_SET_hold_placement, apply_placement, NULL);
    rt_settings_hook(RT_SET_frame_rate_target, apply_rate, NULL);
    const char *from;
    rt_setting_source(RT_SET_hold_ticks, &from);
    int64_t hold = rt_setting(RT_SET_hold_ticks);
    nvn_present_hold((int)hold);
    if (hold) log_hold(hold, from);
    int64_t placement = rt_setting(RT_SET_hold_placement);
    rt_setting_source(RT_SET_hold_placement, &from);
    if (placement) log_placement(placement, from);
    int64_t rate = rt_setting(RT_SET_frame_rate_target);
    rt_setting_source(RT_SET_frame_rate_target, &from);
    if (rate) log_rate(rate, from);
    log_header();
}
