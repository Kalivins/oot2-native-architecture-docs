// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// nn::os fibres on host fibres. The SDK switches a fibre by swapping callee-saved registers and the stack pointer
// (nn::os::detail::DispatchFiberContext). Lifted code keeps callee-saved registers in host locals (architecture D3), so
// a guest fibre needs a host stack of its own as well as a guest one: each runs on a host fibre with its own Cpu, and
// a suspended fibre's host frames and registers stay untouched while others run. Any thread may resume a fibre that is
// not running, as on the console. The four public entry points are host functions bound by symbol; every caller in
// OoT's modules reaches them through a PLT slot (main's imports, the sdk's own preemptible calls), which the loader
// binds. The sdk's internal switch, its fibre trampoline and its alias stacks then have no caller left; runtime/hos
// binds them to named fatals.
//
// What follows a fibre and what stays with the thread. DispatchFiberContext saves x19 to x30, sp and q8 to q15, nothing
// else; here every register follows the fibre (its own Cpu, a superset the AAPCS64 allows: SwitchToFiber is a call).
// TPIDR_EL0, TPIDRRO_EL0 (the thread-local region, the thread's own handle at TLR+0x110, the preemption-disable words),
// FPCR and FPSR belong to the thread and are carried into the Cpu of the context it enters. So do the kernel's thread
// object, its virtual core, its priority and its host CPU time (D26: a fibre's run is charged to the thread that ran
// it). The exclusive monitor is cleared at every switch.
//
// The SDK's own records, kept as its SwitchToFiber keeps them (SDK 22's layouts, read from its nn::os fibre functions):
// the state byte of each FiberType (1 ready, 2 running, 4 finished, 0 after FinalizeFiber), its function, argument,
// stack and size, and the ThreadType's current fibre (0 on the thread's own context), which
// nn::os::detail::DebugImplByHorizon::GetCurrentStackInfo reads to report the fibre's stack instead of the thread's. The
// FiberType's saved context (+0x50) is not written: only the sdk's DispatchFiberContext and nn::diag's backtrace of a
// suspended fibre read it, and both are named fatals.
//
// Adapted from TotK Recomp runtime/kernel.c at 0ae47a5 (MIT, Copyright (c) 2026 Valiantsin Dzerakh), as DKB Recomp
// carries it at 9d8fa02 ("fibers"): a host fiber and a Cpu per guest fiber, the thread state carried over at a switch,
// the context left behind released by whatever runs next, ExitThread on a fiber unwound from the thread's own context.
// From oot-pc-recomp runtime/kernel.c at 8e09aaa (GPL-3.0-or-later): SwitchToFiber(nullptr) goes back to the thread's
// own context. Changed here: the SDK's state bytes and the ThreadType's current fibre are kept; a switch to a fibre that
// another thread left a moment ago waits for its host context to be saved instead of failing; what the SDK aborts on
// (a switch to a running or finished fibre, a null fibre on the thread's own context) is a guest stop; fibres are found
// by a hash of their guest address; the frame that resumes after a switch names its thread from its own record, never
// from thread-local storage.
#include <stdlib.h>
#include <string.h>

#include "kern.h"

#define HOST_STACK (256ull << 20)   // as a guest thread's: every guest call is a host call
#define TLR_THREAD_OBJ 0x1F8        // the ThreadType of the thread (TPIDRRO_EL0 + 0x1F8)
#define TT_FIBRE 0x70               // ThreadType: the current fibre, 0 on the thread's own context
#define FT_STATE 0x20               // FiberType, u8
#define FT_ALIAS 0x21               // u8: 1 when the stack is mapped at an alias (the SDK does when flags bit 0 is clear)
#define FT_FUNC 0x28
#define FT_ARG 0x30
#define FT_STACK 0x38
#define FT_STACK_USED 0x40          // the alias, else the stack
#define FT_SIZE 0x48
enum { FS_NONE = 0, FS_READY = 1, FS_RUNNING = 2, FS_DONE = 4 };   // the SDK's state byte

struct KFibre {
    Cpu cpu;                     // first: Cpu wants 16-byte alignment, which calloc gives
    uint64_t addr, func, arg, stack, size;
    PlatFibre *host;
    int state;                   // FS_*: a switch claims a ready fibre by compare-exchange
    int parked;                  // its host context is saved, so a thread may enter it (atomic)
    int started, abandoned;      // entered once; left running by ExitThread
    RtThread *on;                // the thread that entered it last, written by that thread before it enters
};

// ---- fibres by guest address: open addressing, a reader-writer lock; TOMB marks a removed entry
#define TOMB ((KFibre *)(uintptr_t)1)
static PlatLock g_lock = PLAT_LOCK_INIT;
static KFibre **g_slot;
static uint32_t g_cap, g_used;   // used counts live entries and tombstones

static uint32_t slot_of(uint64_t addr, uint32_t cap) { return (uint32_t)((addr >> 3) * 0x9E3779B97F4A7C15ull >> 32) & (cap - 1); }

static KFibre *find_locked(uint64_t addr) {
    if (!g_cap) return NULL;
    for (uint32_t i = slot_of(addr, g_cap), n = 0; n < g_cap && g_slot[i]; i = (i + 1) & (g_cap - 1), n++)
        if (g_slot[i] != TOMB && g_slot[i]->addr == addr) return g_slot[i];
    return NULL;
}

static KFibre *fibre_find(uint64_t addr) {
    plat_lock_shared(&g_lock);
    KFibre *f = find_locked(addr);
    plat_unlock_shared(&g_lock);
    return f;
}

static void put_locked(KFibre *f) {
    if ((g_used + 1) * 2 > g_cap) {   // grow (or sweep the tombstones) past half full
        uint32_t live = 0, cap = g_cap ? g_cap : 64;
        for (uint32_t i = 0; i < g_cap; i++) live += g_slot[i] && g_slot[i] != TOMB;
        while ((live + 1) * 4 > cap) cap *= 2;
        KFibre **s = calloc(cap, sizeof *s);
        if (!s) rt_fatal("kernel: fibre table allocation");
        for (uint32_t i = 0; i < g_cap; i++)
            if (g_slot[i] && g_slot[i] != TOMB) {
                uint32_t j = slot_of(g_slot[i]->addr, cap);
                while (s[j]) j = (j + 1) & (cap - 1);
                s[j] = g_slot[i];
            }
        free(g_slot);
        g_slot = s;
        g_cap = cap;
        g_used = live;
    }
    uint32_t i = slot_of(f->addr, g_cap);
    while (g_slot[i] && g_slot[i] != TOMB) i = (i + 1) & (g_cap - 1);
    if (!g_slot[i]) g_used++;
    g_slot[i] = f;
}

static void del_locked(KFibre *f) {
    for (uint32_t i = slot_of(f->addr, g_cap);; i = (i + 1) & (g_cap - 1))
        if (g_slot[i] == f) {
            g_slot[i] = TOMB;
            return;
        }
}

// ---- switching
// After a host fibre switch, on the context that was entered, with u the thread that entered it: park the context u
// left (a thread waiting to enter it may go on), set the host MXCSR from the guest FPCR (each host fibre keeps its own),
// and finish an ExitThread issued on a fibre once back on the thread's own context.
static void resumed(RtThread *u) {
    KFibre *l = u->leaving;
    u->leaving = NULL;
    if (l) __atomic_store_n(&l->parked, 1, __ATOMIC_RELEASE);
    rt_fpcr_sync(u->cur);
    if (u->exit_pending && !u->fibre) rt_host_longjmp(u->exit_buf, 1);
}

// t leaves the context it runs (left: what that context becomes when it is a fibre: FS_READY, FS_DONE, or FS_RUNNING
// for one abandoned by ExitThread) for to, a fibre this thread claimed, or its own context when to is NULL. Returns when
// the context it left is entered again, on whichever thread enters it; t is not used after the switch.
static void enter(RtThread *t, KFibre *to, int left) {
    KFibre *from = t->fibre;
    Cpu *src = t->cur, *dst = to ? &to->cpu : &t->cpu;
    if (!t->own && !(t->own = plat_fibre_self())) rt_fatal("kernel: thread %#x cannot become a host fibre", t->handle);
    if (to) {
        while (!__atomic_load_n(&to->parked, __ATOMIC_ACQUIRE)) plat_pause();   // left a moment ago on another thread
        __atomic_store_n(&to->parked, 0, __ATOMIC_RELAXED);
        to->on = t;
        if (!to->started) {
            to->started = 1;
            dst->sp = (to->stack + to->size) & ~15ull;
        }
    }
    dst->tpidr = src->tpidr;
    dst->tpidrro = src->tpidrro;
    dst->fpcr = src->fpcr;
    dst->fpsr = src->fpsr;
    dst->excl_addr = ~0ull;
    dst->thread = t;
    // The SDK's records, before the context left can be claimed by another thread.
    if (to) ST8(to->addr + FT_STATE, FS_RUNNING);
    ST64(LD64(src->tpidrro + TLR_THREAD_OBJ) + TT_FIBRE, to ? to->addr : 0);
    if (from && left != FS_RUNNING) {
        ST8(from->addr + FT_STATE, left);
        __atomic_store_n(&from->state, left, __ATOMIC_RELEASE);
    }
    t->leaving = from;
    t->fibre = to;
    klock();   // kicks (kkick) write the pending flag of t->cur under the lock: move it with the context
    t->cur = dst;
    if (__atomic_exchange_n(&src->pending, 0, __ATOMIC_ACQ_REL)) __atomic_store_n(&dst->pending, 1, __ATOMIC_RELEASE);
    K.st.fibre_switches++;
    kunlock();
    plat_fibre_switch(from ? from->host : t->own, to ? to->host : t->own);
    resumed(from ? from->on : t);   // only its own thread enters a thread's own context
}

// The fibre at addr, claimed for a switch: guest stops where the SDK's SwitchToFiber aborts.
static KFibre *claim(Cpu *c, uint64_t addr, const char *what) {
    KFibre *f = fibre_find(addr);
    if (!f) rt_guest_stop(c, "%s %#llx, which is not an initialized fibre: the SDK aborts", what, (unsigned long long)addr);
    int s = FS_READY;
    if (!__atomic_compare_exchange_n(&f->state, &s, FS_RUNNING, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
        rt_guest_stop(c, "%s %#llx, which is %s: the SDK aborts", what, (unsigned long long)addr,
                      s == FS_RUNNING ? "running" : "finished");
    return f;
}

// A host fibre's start: run the guest fibre's function; what it returns is the fibre to switch to (NULL: the thread's
// own context), as the SDK's trampoline does.
static void fibre_main(void *p) {
    KFibre *f = p;
    resumed(f->on);
    Cpu *c = &f->cpu;
    c->x[0] = f->arg;
    rt_call(c, f->func);
    uint64_t next = c->x[0];
    KFibre *n = next ? claim(c, next, "a fibre's function returned") : NULL;
    enter(f->on, n, FS_DONE);   // f->on: the fibre may have moved to another thread while its function ran
    rt_fatal("kernel: fibre %#llx entered after it finished", (unsigned long long)f->addr);
}

// ---- the four public entry points
// The first InitializeFiber and the first SwitchToFiber of a run are logged with their caller: a log then shows where
// fibres come in among the other start-up lines.
static int g_said_init, g_said_switch;

static void say_first(int *said, const Cpu *c, const char *what, uint64_t addr) {
    if (__atomic_exchange_n(said, 1, __ATOMIC_RELAXED)) return;
    char at[96];
    rt_log("kernel: first %s(%#llx), from %s", what, (unsigned long long)addr, rt_guest_name(c->x[30] - 4, at, sizeof at));
}

// nn::os::InitializeFiber(x0 FiberType*, x1 function, x2 argument, x3 stack, x4 stack size, w5 flags). The fibre runs
// on the stack it was given: the SDK maps the stack at an alias first when flags bit 0 is clear, for guard pages; the
// guest sees the stack as the SDK leaves it with bit 0 set.
static void fibre_initialize(Cpu *c) {
    uint64_t addr = c->x[0];
    say_first(&g_said_init, c, "InitializeFiber", addr);
    plat_lock(&g_lock);
    KFibre *f = find_locked(addr);
    if (f && __atomic_load_n(&f->state, __ATOMIC_ACQUIRE) == FS_RUNNING && !f->abandoned) {
        plat_unlock(&g_lock);
        rt_guest_stop(c, "InitializeFiber(%#llx): the fibre is running, and the SDK would write its first frame over "
                         "the stack it runs on", (unsigned long long)addr);
    }
    int fresh = !f;
    if (f) {   // initialized again without FinalizeFiber, as the SDK allows
        while (!__atomic_load_n(&f->parked, __ATOMIC_ACQUIRE)) plat_pause();
        plat_fibre_free(f->host);
        memset(&f->cpu, 0, sizeof f->cpu);
    } else {
        if (!(f = calloc(1, sizeof *f))) rt_fatal("kernel: fibre allocation");
        f->addr = addr;
        put_locked(f);
    }
    f->cpu.excl_addr = ~0ull;
    f->func = c->x[1];
    f->arg = c->x[2];
    f->stack = c->x[3];
    f->size = c->x[4];
    f->started = f->abandoned = 0;
    f->on = NULL;
    f->host = plat_fibre_new(HOST_STACK, fibre_main, f);
    if (!f->host) rt_fatal("kernel: no host fibre for fibre %#llx", (unsigned long long)addr);
    __atomic_store_n(&f->parked, 1, __ATOMIC_RELEASE);
    __atomic_store_n(&f->state, FS_READY, __ATOMIC_RELEASE);
    plat_unlock(&g_lock);
    ST8(addr + FT_STATE, FS_READY);
    ST8(addr + FT_ALIAS, 0);
    ST64(addr + FT_FUNC, f->func);
    ST64(addr + FT_ARG, f->arg);
    ST64(addr + FT_STACK, f->stack);
    ST64(addr + FT_STACK_USED, f->stack);
    ST64(addr + FT_SIZE, f->size);
    klock();
    K.st.fibres_created++;   // every InitializeFiber; live counts the records
    K.st.fibres_live += (uint64_t)fresh;
    kunlock();
}

// nn::os::FinalizeFiber(x0 FiberType*). The SDK checks nothing and clears the state byte; a fibre that runs cannot be
// freed here, since its host stack is in use.
static void fibre_finalize(Cpu *c) {
    uint64_t addr = c->x[0];
    plat_lock(&g_lock);
    KFibre *f = find_locked(addr);
    if (f && __atomic_load_n(&f->state, __ATOMIC_ACQUIRE) == FS_RUNNING && !f->abandoned) {
        plat_unlock(&g_lock);
        rt_guest_stop(c, "FinalizeFiber(%#llx): the fibre is running on thread %#x", (unsigned long long)addr,
                      f->on ? f->on->handle : 0);
    }
    if (f) {
        del_locked(f);
        while (!__atomic_load_n(&f->parked, __ATOMIC_ACQUIRE)) plat_pause();   // left a moment ago
    }
    plat_unlock(&g_lock);
    ST8(addr + FT_STATE, FS_NONE);
    if (!f) return;
    plat_fibre_free(f->host);
    free(f);
    klock();
    K.st.fibres_live--;
    kunlock();
}

// nn::os::SwitchToFiber(x0 FiberType*): to that fibre, or with nullptr back to the thread's own context.
static void fibre_switch(Cpu *c) {
    RtThread *t = kcurrent();
    if (!t) rt_fatal("kernel: SwitchToFiber on a host thread, which is no guest thread");
    uint64_t addr = c->x[0];
    say_first(&g_said_switch, c, "SwitchToFiber", addr);
    if (!addr && !t->fibre)
        rt_guest_stop(c, "SwitchToFiber(nullptr) on the thread's own context: the SDK writes through a null fibre there");
    enter(t, addr ? claim(c, addr, "SwitchToFiber to") : NULL, FS_READY);
}

// nn::os::GetCurrentFiber() -> x0: the fibre the thread runs, 0 on its own context.
static void fibre_current(Cpu *c) {
    RtThread *t = kcurrent();
    c->x[0] = t && t->fibre ? t->fibre->addr : 0;
}

HOST_OVERRIDE("_ZN2nn2os15InitializeFiberEPNS0_9FiberTypeEPFS2_PvES3_S3_mi", fibre_initialize);
HOST_OVERRIDE("_ZN2nn2os13FinalizeFiberEPNS0_9FiberTypeE", fibre_finalize);
HOST_OVERRIDE("_ZN2nn2os13SwitchToFiberEPNS0_9FiberTypeE", fibre_switch);
HOST_OVERRIDE("_ZN2nn2os15GetCurrentFiberEv", fibre_current);

// ---- the kernel's side (kern.h)
void kfibre_exit_thread(RtThread *t) {
    t->exit_pending = 1;
    t->fibre->abandoned = 1;   // left running, as on the console; FinalizeFiber may free it once its context is saved
    enter(t, NULL, FS_RUNNING);
    rt_fatal("kernel: ExitThread on a fibre came back");
}

void kfibre_thread_end(RtThread *t) {
    if (!t->own) return;
    plat_fibre_unself(t->own);
    t->own = NULL;
}

uint64_t kfibre_addr(const KFibre *f) { return f ? f->addr : 0; }
