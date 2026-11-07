// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Host functions at synthetic guest addresses: the HOST_OVERRIDE linker table, the registry behind it, and host code
// calling guest code (rt_call). Adapted from DKB Recomp runtime/hostfn.c at 9d8fa02 (MIT, adapted there from LAS
// Recomp runtime/hle.h and runtime/dispatch.c, Copyright (c) 2026 Peiikai): the linker-section table and the
// registry of synthetic addresses. Changed here: a slot without a function is the named fatal of a missing import,
// and the window is an array that rt_dispatch reaches without a hash.
//
// Every registered address dispatches through hostfn_entry, which counts the call and tail-calls the function. The
// registry is fixed when rt_dispatch_init seals it; guest threads start after that, so lookups take no lock.
#include <stdio.h>
#include <string.h>

#include "core.h"
#include "oot2/settings.h"
#include "plat.h"

#define MAX_HOSTFN 16384   // under RT_HOSTFN_SIZE / 16, which also holds RT_RET_HOST

#ifdef _WIN32
// PE/COFF: the linker sorts .ovr$a < .ovr$m (the entries, runtime.h) < .ovr$z by the text after '$'. The two markers
// hold no name, and any padding the linker puts between the entries of two objects reads as an empty entry.
__attribute__((section(".ovr$a"), used)) static const RtHostEntry ovr_start[1];
__attribute__((section(".ovr$z"), used)) static const RtHostEntry ovr_stop[1];
#else
// ELF: the linker defines __start_ and __stop_ around a section named like a C identifier. Weak: a program without
// any override has no such section, and both bounds are then NULL.
extern const RtHostEntry __start_oot2_ovr[] __attribute__((weak)), __stop_oot2_ovr[] __attribute__((weak));
#define ovr_start __start_oot2_ovr
#define ovr_stop __stop_oot2_ovr
#endif

typedef struct {
    const char *name;
    GuestFn fn;        // NULL: a missing import, whose call is a named fatal
    uint64_t calls;
    uint64_t first_lr; // x30 of the first call, set once (the census names where each host function was first met)
} Slot;
static Slot slots[MAX_HOSTFN];
static int nslots;
static int sealed;

static uint64_t slot_addr(int i) { return RT_HOSTFN_BASE + 16ull * (uint64_t)i; }

static int slot_of(uint64_t addr) {
    uint64_t i = (addr - RT_HOSTFN_BASE) >> 4;
    if (addr - RT_HOSTFN_BASE >= RT_HOSTFN_SIZE || (addr & 15) || i >= (uint64_t)nslots) return -1;
    return (int)i;
}

static void table_bounds(const RtHostEntry **start, const RtHostEntry **stop) {
    const RtHostEntry *p = ovr_start, *e = ovr_stop;
    __asm__("" : "+r"(p), "+r"(e));   // separate objects: keep the compiler from reasoning about the bounds
    *start = p;
    *stop = e;
}

int rt_hostfn_override_count(void) {
    const RtHostEntry *p, *e;
    table_bounds(&p, &e);
    int n = 0;
    for (; p < e; p++) n += p->name != NULL;
    return n;
}

static GuestFn override_of(const char *symbol) {
    const RtHostEntry *p, *e;
    table_bounds(&p, &e);
    GuestFn found = NULL;
    for (; p < e; p++) {
        if (!p->name || strcmp(p->name, symbol)) continue;
        if (found && found != p->fn) rt_fatal("host functions: two different overrides of %s", symbol);
        found = p->fn;
    }
    return found;
}

static uint64_t add_slot(const char *name, GuestFn fn) {
    if (sealed) rt_fatal("host functions: %s registered after rt_dispatch_init", name);
    if (!name || !*name) rt_fatal("host functions: an entry without a name");
    for (int i = 0; i < nslots; i++)
        if (slots[i].fn == fn && !strcmp(slots[i].name, name)) return slot_addr(i);
    if (nslots == MAX_HOSTFN) rt_fatal("host functions: more than %d", MAX_HOSTFN);
    slots[nslots] = (Slot){.name = name, .fn = fn};
    return slot_addr(nslots++);
}

uint64_t rt_hostfn_register(const char *name, GuestFn fn) {
    if (!fn) rt_fatal("rt_hostfn_register(%s): no function", name ? name : "(null)");
    return add_slot(name, fn);
}

uint64_t rt_hostfn_bind(const char *symbol) {
    GuestFn fn = symbol ? override_of(symbol) : NULL;
    return fn ? add_slot(symbol, fn) : 0;
}

uint64_t rt_hostfn_missing(const char *symbol) { return add_slot(symbol, NULL); }

const char *rt_hostfn_name(uint64_t addr) {
    if (addr == RT_RET_HOST) return "<return to host>";
    int i = slot_of(addr);
    return i < 0 ? NULL : slots[i].name;
}

uint64_t rt_hostfn_calls(uint64_t addr) {
    int i = slot_of(addr);
    return i < 0 ? 0 : __atomic_load_n(&slots[i].calls, __ATOMIC_RELAXED);
}

int rt_hostfn_count(void) { return nslots; }

const char *rt_hostfn_at(int i, uint64_t *calls, uint64_t *first_lr, int *missing) {
    if (i < 0 || i >= nslots) return NULL;
    *calls = __atomic_load_n(&slots[i].calls, __ATOMIC_RELAXED);
    *first_lr = __atomic_load_n(&slots[i].first_lr, __ATOMIC_RELAXED);
    *missing = !slots[i].fn;
    return slots[i].name;
}

int rt_hostfn_snapshot(uint64_t *calls, int max) {
    for (int i = 0; i < nslots && i < max; i++) calls[i] = __atomic_load_n(&slots[i].calls, __ATOMIC_RELAXED);
    return nslots;
}

// ---- census mode (runtime.h)
static int g_census = -1;
static uint64_t g_census_frames;   // stored before g_census is published, read after
static void (*g_census_flush)(void);

int rt_census_on(void) {
    int v = __atomic_load_n(&g_census, __ATOMIC_RELAXED);
    if (v >= 0) return v;
    v = rt_setting(RT_SET_census) != 0;   // OOT2_CENSUS, or another layer of the settings store
    uint64_t frames = v ? (uint64_t)rt_setting(RT_SET_census_frames) : 0;
    __atomic_store_n(&g_census_frames, frames, __ATOMIC_RELAXED);   // every racing thread stores the same value
    int none = -1;
    if (__atomic_compare_exchange_n(&g_census, &none, v, 0, __ATOMIC_ACQ_REL, __ATOMIC_RELAXED) && v)
        rt_log("OOT2_CENSUS=1: gaps in NVN2 are listed and passed (answer 0), not fatal; this is not a default run%s",
               frames ? "; it ends at its frame limit" : "");   // rt_log marks it like every census-mode line
    return v;
}

uint64_t rt_census_frames(void) { return rt_census_on() ? __atomic_load_n(&g_census_frames, __ATOMIC_RELAXED) : 0; }
void rt_census_set_flush(void (*flush)(void)) { g_census_flush = flush; }
void rt_census_flush(void) {
    if (g_census_flush) g_census_flush();
}

void rt_census_end(int code, const char *line) {
    rt_run_stop_hook(code, line);
    plat_exit(code);
}

const char *rt_hostfn_unbound(int index) {
    const RtHostEntry *p, *e;
    table_bounds(&p, &e);
    for (; p < e; p++) {
        if (!p->name) continue;
        int bound = 0;
        for (int i = 0; i < nslots && !bound; i++) bound = slots[i].fn == p->fn && !strcmp(slots[i].name, p->name);
        if (!bound && index-- == 0) return p->name;
    }
    return NULL;
}

void hostfn_seal(void) { sealed = 1; }

void rt_replaced(Cpu *c, const char *symbol) {
    GuestFn fn = override_of(symbol);
    if (!fn) rt_unimplemented(c, "replaced", "%s", symbol);
    fn(c);
}

GuestFn hostfn_lookup(uint64_t addr) { return slot_of(addr) < 0 ? NULL : hostfn_entry; }

void hostfn_entry(Cpu *c) {
    int i = slot_of(c->pc);
    if (i < 0) {
        char at[96];
        rt_fatal("host function call at %s: no host function registered there", rt_guest_name(c->pc, at, sizeof at));
    }
    __atomic_fetch_add(&slots[i].calls, 1, __ATOMIC_RELAXED);
    if (!__atomic_load_n(&slots[i].first_lr, __ATOMIC_RELAXED)) {
        uint64_t none = 0;
        __atomic_compare_exchange_n(&slots[i].first_lr, &none, c->x[30] ? c->x[30] : 1, 0, __ATOMIC_RELAXED,
                                    __ATOMIC_RELAXED);
    }
    GuestFn fn = slots[i].fn;
    if (!fn) rt_unimplemented(c, "import", "%s", slots[i].name);
    __attribute__((musttail)) return fn(c);
}

void rt_call(Cpu *c, uint64_t addr) {
    c->x[30] = RT_RET_HOST;
    c->pc = addr;
    rt_fpcr_sync(c);
    rt_dispatch(c);
}
