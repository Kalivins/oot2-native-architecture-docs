// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The boot driver (architecture D9). rtld is not loaded, so the main thread starts in a host function that does what
// rtld does on the console once the modules are linked: it calls the SDK's nn::init::Start with the main thread's
// handle and three callbacks. Start initialises nn::os, calls the first callback, initialises the SDK module, runs the
// game's nninitStartup (heap and allocators), calls the second callback, which runs each module's constructors, then
// nnMain, and the third callback when nnMain returns (the services study).
//
// Adapted from oot-pc-recomp runtime/boot.c at 8e09aaa (GPL-3.0-or-later repository; adapted there from MacNativeGames
// games/totk/runtime/loader.c and services.c at b75be8c, MIT, Copyright (c) 2026 MacNativeGames contributors, and LAS
// Recomp runtime/main.c at da35b83, MIT, Copyright (c) 2026 Peiikai): Start's arguments, the three callbacks, DT_INIT of
// each module in load order from the second. Changed here: no table of game versions, init_array sizes or return
// addresses (the runtime knows no address of the game; the census names where each override was first called, which
// gives the checkpoints of nnMain and of the graphics bring-up); a module without DT_INIT runs its init_array entries;
// the third callback is the game ending, a guest stop.
#include <string.h>

#include "hos.h"
#include "plat.h"

#define SYM_INIT_START "_ZN2nn4init5StartEmmPFvvES2_S2_"

static uint64_t g_entry, g_cb[3];
static int g_inits_requested;

static double ms_since(uint64_t t0) { return (double)(plat_now_ns() - t0) / 1e6; }

// ---- the three callbacks of nn::init::Start (rtld's on the console)
static void cb_loader_ready(Cpu *c) {
    (void)c;
    rt_log("boot: rtld callback 1: the modules are linked");
}

// Each module's constructors, in load order (main, subsdk0..9, sdk), as rtld runs them: DT_INIT registers the module
// with the C library and runs its init_array.
static void cb_module_inits(Cpu *c) {
    if (__atomic_exchange_n(&g_inits_requested, 1, __ATOMIC_ACQ_REL))
        hos_gap(c, "boot", "nn::init::Start asked for the module constructors a second time");
    for (int i = 0; i < rt_loaded_count(); i++) {
        const RtLoaded *m = rt_loaded(i);
        uint64_t entries = m->init_array_size / 8, t0 = plat_now_ns();
        if (m->init) {
            hos_call(c, m->init, 0);
        } else {
            for (uint64_t k = 0; k < entries; k++) {
                uint64_t f = LD64(m->init_array + 8 * k);
                if (f && f != ~0ull) hos_call(c, f, 0);
            }
        }
        rt_log("boot: %s constructors (%s, %llu init_array entries) ran in %.1f ms", m->name,
               m->init ? "DT_INIT" : "init_array only", (unsigned long long)entries, ms_since(t0));
    }
}

static void cb_module_finis(Cpu *c) {
    rt_guest_stop(c, "nnMain returned and nn::init::Start asked for the module finalizers: the game ended");
}

// ---- the main thread's entry: x0 = 0, x1 = its own handle (rt_kernel_run)
static void rtld_start(Cpu *c) {
    uint32_t handle = (uint32_t)c->x[1];
    uint64_t start = hos_sdk(c, SYM_INIT_START);
    char at[96];
    rt_log("boot: nn::init::Start at %s, main thread %#x", rt_guest_name(start, at, sizeof at), handle);
    // nn::init::Start(main thread handle, arguments region, loader ready, module constructors, module finalizers)
    hos_call(c, start, 5, (uint64_t)handle, (uint64_t)0, g_cb[0], g_cb[1], g_cb[2]);
    hos_gap(c, "boot", "nn::init::Start returned to rtld, which the SDK never does");
}

void boot_register(void) {
    g_entry = rt_hostfn_register("<rtld>", rtld_start);
    g_cb[0] = rt_hostfn_register("<rtld loader_ready>", cb_loader_ready);
    g_cb[1] = rt_hostfn_register("<rtld module_inits>", cb_module_inits);
    g_cb[2] = rt_hostfn_register("<rtld module_finis>", cb_module_finis);
}

uint64_t hos_boot_entry(void) {
    if (!g_entry) rt_fatal("hos_boot_entry before hos_init");
    return g_entry;
}
