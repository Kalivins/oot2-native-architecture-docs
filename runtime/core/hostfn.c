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

#include <stdlib.h>

#include "core.h"
#include "oot2/kernel.h"
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

// ---- the census timer (runtime.h)
// A frame is a nested call: its time in, taken from the thread's CPU clock on the way in, its children's sums in the
// thread's table (not on the C stack, so a frame a longjmp or a fibre switch abandoned can never be written through).
// On the way out a frame's own time is (out - in) - children - overhead * nchildren, it is a child of the frame below
// it, and the depth goes back to where it was at the way in, so a frame skipped by a guest longjmp is dropped.
#define TIMER_DEPTH 256
#define TIMER_GUEST_CELL MAX_HOSTFN   // the cell after the last slot: guest callbacks
typedef struct { uint64_t calls, ns; } TimerCell;
typedef struct TimerThread {
    struct TimerThread *next;
    uint32_t id, handle, depth, pad;
    uint64_t calls[RT_TIMER_CLASSES], ns[RT_TIMER_CLASSES];
    uint64_t child[TIMER_DEPTH];
    uint32_t nchild[TIMER_DEPTH];
    TimerCell cell[MAX_HOSTFN + 1];
} TimerThread;
static int g_timer = -1;                    // -1: not read yet (rt_dispatch_init); 0: off; 1: on
static uint64_t g_overhead, g_floor, g_overflows;   // calibrated: the timer's cost outside a child, and inside a frame
static uint8_t g_cell_class[MAX_HOSTFN + 1];
static TimerThread *g_timer_head;           // every thread's table, newest first; never freed
static uint32_t g_timer_ids;
static _Thread_local TimerThread *t_timer;

int rt_timer_on(void) { return __atomic_load_n(&g_timer, __ATOMIC_RELAXED) > 0; }
uint64_t rt_timer_overhead_ns(void) { return __atomic_load_n(&g_overhead, __ATOMIC_RELAXED); }
uint64_t rt_timer_floor_ns(void) { return __atomic_load_n(&g_floor, __ATOMIC_RELAXED); }
uint64_t rt_timer_overflows(void) { return __atomic_load_n(&g_overflows, __ATOMIC_RELAXED); }

static TimerThread *timer_table(void) {
    TimerThread *t = t_timer;
    if (t) return t;
    t = calloc(1, sizeof *t);
    if (!t) rt_fatal("census timer: out of memory for a thread's table (%zu bytes)", sizeof *t);
    t->id = __atomic_fetch_add(&g_timer_ids, 1, __ATOMIC_RELAXED);
    t->handle = rt_kernel_current();
    t->next = __atomic_load_n(&g_timer_head, __ATOMIC_RELAXED);
    while (!__atomic_compare_exchange_n(&g_timer_head, &t->next, t, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
    }
    t_timer = t;
    return t;
}

// The way out of the frame at depth d that began at t_in; cell and its class say where its own time goes.
static void timer_pop(TimerThread *t, uint32_t d, int cell, uint64_t t_in, uint64_t t_out) {
    uint64_t total = t_out - t_in, nested = t->child[d] + g_overhead * t->nchild[d] + g_floor;
    uint64_t self = total > nested ? total - nested : 0;
    t->depth = d;
    if (d) {
        t->child[d - 1] += total;
        t->nchild[d - 1]++;
    }
    int cls = g_cell_class[cell];
    __atomic_fetch_add(&t->cell[cell].calls, 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&t->cell[cell].ns, self, __ATOMIC_RELAXED);
    __atomic_fetch_add(&t->calls[cls], 1, __ATOMIC_RELAXED);
    __atomic_fetch_add(&t->ns[cls], self, __ATOMIC_RELAXED);
}

static int timer_push(TimerThread *t, uint32_t *d) {
    *d = t->depth;
    if (*d >= TIMER_DEPTH) {
        __atomic_fetch_add(&g_overflows, 1, __ATOMIC_RELAXED);
        return 0;
    }
    t->child[*d] = 0;
    t->nchild[*d] = 0;
    t->depth = *d + 1;
    return 1;
}

__attribute__((noinline)) static void timed_call(Cpu *c, GuestFn fn, int slot) {
    TimerThread *t = timer_table();
    uint32_t d;
    if (!timer_push(t, &d)) return fn(c);
    uint64_t t_in = plat_thread_cpu_ns(NULL);
    fn(c);
    timer_pop(t, d, slot, t_in, plat_thread_cpu_ns(NULL));
}

// A guest callback inside a timed host function: its own time is the guest bucket, its whole time a child of the host
// function that called it. A call with no timed frame below it (a thread's entry) is not timed.
__attribute__((noinline)) static void timed_guest(Cpu *c) {
    TimerThread *t = t_timer;
    uint32_t d;
    if (!t || !t->depth || !timer_push(t, &d)) return rt_dispatch(c);
    uint64_t t_in = plat_thread_cpu_ns(NULL);
    rt_dispatch(c);
    timer_pop(t, d, TIMER_GUEST_CELL, t_in, plat_thread_cpu_ns(NULL));
}

int rt_timer_threads(RtTimerThread *out, int max) {
    int n = 0;
    for (TimerThread *t = __atomic_load_n(&g_timer_head, __ATOMIC_ACQUIRE); t; t = t->next, n++) {
        if (n >= max) continue;
        out[n] = (RtTimerThread){.id = t->id, .handle = t->handle};
        for (int k = 0; k < RT_TIMER_CLASSES; k++) {
            out[n].calls[k] = __atomic_load_n(&t->calls[k], __ATOMIC_RELAXED);
            out[n].ns[k] = __atomic_load_n(&t->ns[k], __ATOMIC_RELAXED);
        }
    }
    return n;
}

void rt_timer_slots(uint64_t *calls, uint64_t *ns, int n) {
    for (int i = 0; i <= n; i++) calls[i] = ns[i] = 0;
    for (TimerThread *t = __atomic_load_n(&g_timer_head, __ATOMIC_ACQUIRE); t; t = t->next)
        for (int i = 0; i <= n; i++) {
            int cell = i == n ? TIMER_GUEST_CELL : i;
            calls[i] += __atomic_load_n(&t->cell[cell].calls, __ATOMIC_RELAXED);
            ns[i] += __atomic_load_n(&t->cell[cell].ns, __ATOMIC_RELAXED);
        }
}

// What the timer costs a parent per nested call: a frame, then N empty frames nested in it, popped at once, on a table of
// its own that is dropped. The parent's whole time less its children's, per child, is the cost of a push and a pop that
// falls outside the child's own reading, which is what timer_pop subtracts for every child.
#define TIMER_CALIBRATE_N 2000
static uint64_t timer_calibrate(uint64_t *floor) {
    TimerThread *t = calloc(1, sizeof *t);
    if (!t) rt_fatal("census timer: out of memory for the calibration");
    uint64_t best = UINT64_MAX, least = UINT64_MAX;
    for (int rep = 0; rep < 5; rep++) {
        uint32_t d0, d1;
        t->depth = 0;
        timer_push(t, &d0);
        uint64_t in0 = plat_thread_cpu_ns(NULL);
        for (int k = 0; k < TIMER_CALIBRATE_N; k++) {
            timer_push(t, &d1);
            uint64_t in1 = plat_thread_cpu_ns(NULL);
            timer_pop(t, d1, 0, in1, plat_thread_cpu_ns(NULL));
        }
        uint64_t out0 = plat_thread_cpu_ns(NULL), whole = out0 - in0, kids = t->child[d0];
        uint64_t per = whole > kids ? (whole - kids) / TIMER_CALIBRATE_N : 0;
        if (per < best) best = per;
        if (kids / TIMER_CALIBRATE_N < least) least = kids / TIMER_CALIBRATE_N;   // an empty frame's own reading
        timer_pop(t, d0, 0, in0, out0);
    }
    free(t);
    *floor = least;
    return best;
}

// Read at rt_dispatch_init, when every slot is registered: the timer is on only in census mode.
static void timer_resolve(void) {
    int want = rt_setting(RT_SET_census_timer) != 0, on = want && rt_census_on();
    if (want && !on) rt_log("OOT2_CENSUS_TIMER is set without OOT2_CENSUS=1: the timer stays off");
    if (on) {
        for (int i = 0; i < nslots; i++) {
            const char *n = slots[i].name;
            g_cell_class[i] = !strcmp(n, "nvnQueueSubmitCommands") ? RT_TIMER_SUBMIT
                              : !strncmp(n, "nvn", 3)             ? RT_TIMER_NVN
                                                                  : RT_TIMER_OTHER;
        }
        g_cell_class[TIMER_GUEST_CELL] = RT_TIMER_GUEST;
        uint64_t floor;
        uint64_t overhead = timer_calibrate(&floor);
        __atomic_store_n(&g_floor, floor, __ATOMIC_RELAXED);
        __atomic_store_n(&g_overhead, overhead, __ATOMIC_RELAXED);
        rt_log("OOT2_CENSUS_TIMER=1: every host function call is timed by its thread's CPU time (%llu ns a nested call "
               "subtracted from its parent, %llu ns from the call itself); this is not a default run",
               (unsigned long long)g_overhead, (unsigned long long)g_floor);
    }
    __atomic_store_n(&g_timer, on, __ATOMIC_RELEASE);
}

void hostfn_seal(void) {
    sealed = 1;
    timer_resolve();
}

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
    if (__builtin_expect(__atomic_load_n(&g_timer, __ATOMIC_RELAXED) <= 0, 1)) __attribute__((musttail)) return fn(c);
    timed_call(c, fn, i);
}

void rt_call(Cpu *c, uint64_t addr) {
    c->x[30] = RT_RET_HOST;
    c->pc = addr;
    rt_fpcr_sync(c);
    if (__builtin_expect(__atomic_load_n(&g_timer, __ATOMIC_RELAXED) <= 0, 1)) rt_dispatch(c);
    else timed_guest(c);
}
