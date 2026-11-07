// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The NVN2 entry-point table (architecture D10). Every name of nvn_procs.inc gets a synthetic host address before
// rt_dispatch_init seals the host-function registry, and nvnBootstrapLoader and nvnDeviceGetProcAddress hand them out.
// A name with an NVN_API body gets that body; every other listed name stops as the named fatal
// "unimplemented nvn: <name>" (exit RT_EXIT_UNIMPLEMENTED), so a run that reaches one is a census line. A name that is
// not on the list returns 0, as the console's driver returns NULL, and is logged once.
//
// The game asks for its entry points once: its NVN2 loader imports nvnBootstrapLoader from sdk, asks it for
// nvnDeviceGetProcAddress, then asks that for every name it uses and keeps the answers in a table of its own.
//
// Adapted from TotK Recomp runtime/nvn_procs.c and runtime/ov_gfx.c (the nvnBootstrapLoader override) (MIT, Copyright
// (c) 2026 Valiantsin Dzerakh; licenses/totk-MIT.txt), as DKB Recomp at 9d8fa02 and oot-pc-recomp at 8e09aaa carry
// them (GPL-3.0-or-later). Changed here: bodies are found through the
// HOST_OVERRIDE table, not through weak definitions (a weak definition in a static library is not replaced by a strong
// one the link never pulls in, and weak symbols on PE/COFF are fragile); every unbodied name shares one stand-in that
// reads its own name from the host-function registry; asking for a name before the table is registered is a fatal.
#include <stdlib.h>
#include <string.h>

#include "nvn_api.h"

static const char *const g_names[NVN_PROC_COUNT] = {
#define NVN_PROC(n) #n,
#include "nvn_procs.inc"
#undef NVN_PROC
};

typedef struct {
    const char *name;
    uint64_t addr;
    int body;
} Proc;
static Proc g_procs[NVN_PROC_COUNT];   // in list order, which is strcmp order (checked at registration)
static int g_registered;

// Every listed name without a body: on entry c->pc is the entry point's own synthetic address (runtime.h).
static void nvn_unbodied(Cpu *c) {
    const char *name = rt_hostfn_name(c->pc);
    if (rt_census_on()) {   // census mode only: list the gap once, answer 0 and go on (the services study)
        if (rt_hostfn_calls(c->pc) == 1) {
            char at[96];
            rt_log("unbodied nvn: %s, answered 0, at %s", name ? name : "(unnamed)",
                   rt_guest_name(c->x[30], at, sizeof at));
        }
        ret_x(c, 0);
        return;
    }
    rt_unimplemented(c, "nvn", "%s", name ? name : "(an entry point the registry does not name)");
}

void nvn_procs_register(void) {
    if (g_registered) return;
    if (rt_sealed()) rt_fatal("nvn: nvn_procs_register after rt_dispatch_init: the entry points have no addresses");
    for (int i = 0; i < NVN_PROC_COUNT; i++) {
        if (i && strcmp(g_names[i - 1], g_names[i]) >= 0)
            rt_fatal("nvn: nvn_procs.inc is not sorted or has a duplicate at %s", g_names[i]);
        uint64_t body = rt_hostfn_bind(g_names[i]);
        g_procs[i] = (Proc){g_names[i], body ? body : rt_hostfn_register(g_names[i], nvn_unbodied), body != 0};
    }
    g_registered = 1;
}

int nvn_proc_count(void) { return NVN_PROC_COUNT; }
const char *nvn_proc_name(int i) { return i >= 0 && i < NVN_PROC_COUNT ? g_names[i] : NULL; }
int nvn_proc_has_body(int i) { return i >= 0 && i < NVN_PROC_COUNT && g_registered ? g_procs[i].body : 0; }

// A name not on the list: logged the first time it is asked for. The names come from the game, so each is cut to the
// length of a slot, and after MAX_UNKNOWN distinct ones the rest go unlogged, which is said once.
#define MAX_UNKNOWN 32
#define NAME_BYTES 96
static void unknown(const char *name) {
    static int lock;
    static char seen[MAX_UNKNOWN][NAME_BYTES];
    static int nseen, full_said;
    char cut[NAME_BYTES];
    strncpy(cut, name, sizeof cut - 1);
    cut[sizeof cut - 1] = 0;
    while (__atomic_exchange_n(&lock, 1, __ATOMIC_ACQUIRE)) __builtin_ia32_pause();
    int fresh = 1;
    for (int i = 0; i < nseen && fresh; i++) fresh = strcmp(seen[i], cut) != 0;
    int say_full = 0;
    if (fresh && nseen == MAX_UNKNOWN) {
        fresh = 0;
        say_full = !full_said;
        full_said = 1;
    } else if (fresh) {
        memcpy(seen[nseen++], cut, sizeof cut);
    }
    __atomic_store_n(&lock, 0, __ATOMIC_RELEASE);
    if (fresh) rt_log("nvn: no entry point named \"%s\"; returning 0", cut);
    if (say_full) rt_log("nvn: more than %d unknown entry-point names; the rest are not logged", MAX_UNKNOWN);
}

static int by_name(const void *key, const void *entry) { return strcmp(key, ((const Proc *)entry)->name); }

uint64_t nvn_proc_address(const char *name) {
    if (!g_registered)
        rt_fatal("nvn: an entry point was asked for before nvn_procs_register (start-up calls it before "
                 "rt_dispatch_init)");
    if (!name) {
        unknown("(null)");
        return 0;
    }
    const Proc *p = bsearch(name, g_procs, NVN_PROC_COUNT, sizeof *g_procs, by_name);
    if (p) return p->addr;
    unknown(name);
    return 0;
}

// void *nvnBootstrapLoader(const char *name). main imports it from sdk, so the loader binds the import here, and it
// replaces sdk's own definition too.
static void nvn_bootstrap_loader(Cpu *c) { ret_x(c, nvn_proc_address(arg_p(c, 0))); }
HOST_OVERRIDE("nvnBootstrapLoader", nvn_bootstrap_loader);

// PFNNVNGENERICFUNCPTRPROC nvnDeviceGetProcAddress(const NVNdevice *device, const char *name). One device, one set of
// entry points: the answer does not depend on device.
NVN_API(nvnDeviceGetProcAddress) { ret_x(c, nvn_proc_address(arg_p(c, 1))); }
