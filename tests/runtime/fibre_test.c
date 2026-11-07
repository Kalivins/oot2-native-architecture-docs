// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of nn::os fibres on host fibres (runtime/core/fibre.c): synthetic guest programs written by hand in the
// lifter's shape call the four public fibre entry points through the host-function addresses their symbols bind to, as
// the game's PLT calls do (x30 written, then rt_dispatch). Each guest function keeps a C local across its switches,
// which the compiler holds in a callee-saved host register or the host stack: the D3 case a fibre's host stack exists
// for. No game code is involved: the module is a synthetic layout written by tools/layout.py.
//
//   fibre_test CASE    runs one case; prints "ok CASE" and exits 0, or exits 1 with a FAIL line. The cases that end in
//                      a fatal error exit with the runtime's code (tests/test_runtime_fibre.py).
//
// The guest functions come first and together, then one (guest_end) that no thread runs: asynchronous preemption takes
// the module's range, up to its end marker (endmark.c), as lifted code (see kernel_test.c).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"   // hostfn_entry: a call without rt_dispatch's safe point (case pending_moves)
#include "layout.h"
#include "endmark.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "plat.h"   // the watchdog's and the event helper's host threads

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))   // the guest address of guest function i
#define PC 0x7000000040ull           // the guest pc the test's calls and SVCs report
#define TLR_THREAD_OBJ 0x1F8         // the ThreadType pointer in the thread-local region
#define TT_FIBRE 0x70                // ThreadType: the current fibre (SDK 22)

static const char SYM_INIT[] = "_ZN2nn2os15InitializeFiberEPNS0_9FiberTypeEPFS2_PvES3_S3_mi";
static const char SYM_FINAL[] = "_ZN2nn2os13FinalizeFiberEPNS0_9FiberTypeE";
static const char SYM_SWITCH[] = "_ZN2nn2os13SwitchToFiberEPNS0_9FiberTypeE";
static const char SYM_CURRENT[] = "_ZN2nn2os15GetCurrentFiberEv";

// Guest heap words, from the heap base the host sets before the run.
static uint64_t HEAP;
enum { FLAG = 0x0, STOP = 0x8, GO = 0x10, SUM = 0x18, ORDER_N = 0x40, RESULTS = 0x200, HANDLES = 0x300, ORDER = 0x400,
       FT = 0x1000, KEEP = 0x8000, STACKS = 0x100000, FSTACKS = 0x800000 };
#define W(off) (HEAP + (off))
#define FIB(i) W(FT + 0x100ull * (uint64_t)(i))           // FiberType records, zero until initialized
#define FSTACK(i) W(FSTACKS + 0x10000ull * (uint64_t)(i))  // guest stacks of the fibres
#define FSIZE 0x10000ull
#define RING 100                                           // fibres in the ring case
#define ROUNDS 5
static uint64_t f_init, f_final, f_switch, f_current;     // bound before rt_dispatch_init
static uint32_t g_event_w, g_event_r;                      // made by the host before the run

// ---- helpers the guest functions use: inline, no locks
static inline uint64_t sv(Cpu *c, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3;
    rt_svc(c, imm, PC);
    return c->x[0];
}
// A call through a PLT slot, as the lifter writes one: x30, then the dispatch of the bound address.
static inline uint64_t gcall(Cpu *c, uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3, uint64_t a4,
                             uint64_t a5) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3, c->x[4] = a4, c->x[5] = a5;
    c->x[30] = PC + 4;
    c->pc = fn;
    rt_dispatch(c);
    return c->x[0];
}
static inline void fib_init(Cpu *c, int i, uint64_t fn, uint64_t arg) { gcall(c, f_init, FIB(i), fn, arg, FSTACK(i), FSIZE, 1); }
static inline void fib_final(Cpu *c, int i) { gcall(c, f_final, FIB(i), 0, 0, 0, 0, 0); }
static inline void fib_switch(Cpu *c, uint64_t ft) { gcall(c, f_switch, ft, 0, 0, 0, 0, 0); }
static inline uint64_t fib_current(Cpu *c) { return gcall(c, f_current, 0, 0, 0, 0, 0, 0); }
static inline uint64_t tt_fibre(const Cpu *c) { return LD64(LD64(c->tpidrro + TLR_THREAD_OBJ) + TT_FIBRE); }
static inline uint8_t ft_state(int i) { return LD8(FIB(i) + 0x20); }
// As the SDK's GetCurrentStackInfo decides: on a fibre, the stack its FiberType records must hold sp.
static inline int sp_on_fibre_stack(const Cpu *c) {
    uint64_t f = tt_fibre(c);
    return f && c->sp > LD64(f + 0x40) && c->sp <= LD64(f + 0x40) + LD64(f + 0x48);
}
static inline uint32_t core_now(Cpu *c) { return (uint32_t)sv(c, 0x10, 0, 0, 0, 0); }
static inline uint32_t create(Cpu *c, uint64_t entry, int slot, int prio, int core) {
    c->x[1] = entry, c->x[2] = 0, c->x[3] = W(STACKS) + 0x10000ull * (uint64_t)(slot + 1), c->x[4] = (uint32_t)prio,
    c->x[5] = (uint32_t)core;
    rt_svc(c, 0x08, PC);
    CHECK(c->x[0] == 0);
    return (uint32_t)c->x[1];
}
static inline void start(Cpu *c, uint32_t h) { CHECK(sv(c, 0x09, h, 0, 0, 0) == 0); }
static inline void join(Cpu *c, uint32_t h) {
    ST32(W(HANDLES), h);
    CHECK(sv(c, 0x18, 0, W(HANDLES), 1, ~0ull) == 0);
    CHECK(sv(c, 0x16, h, 0, 0, 0) == 0);
}
static inline void exit_thread(Cpu *c) { sv(c, 0x0A, 0, 0, 0, 0); }
static inline uint32_t self_handle(const Cpu *c) { return LD32(c->tpidrro + 0x110); }
static inline void record(uint32_t v) {
    uint64_t i = __atomic_fetch_add((uint64_t *)(uintptr_t)W(ORDER_N), 1, __ATOMIC_SEQ_CST);
    ST32(W(ORDER) + 4 * i, v);
}
static inline int order_is(int n, const uint32_t *want) {
    if (LD64(W(ORDER_N)) != (uint64_t)n) return 0;
    for (int i = 0; i < n; i++)
        if (LD32(W(ORDER) + 4ull * (uint64_t)i) != want[i]) return 0;
    return 1;
}
static inline uint64_t opaque(void) { return __builtin_ia32_rdtsc() | 1; }   // a value no compiler can fold

// ---- guest functions, in address order (the module table below lists them)
enum { FN_BASIC_A, FN_MAIN_BASIC, FN_RET_A, FN_RET_B, FN_RET_AGAIN, FN_MAIN_RETURNS, FN_MOVING, FN_RELAY,
       FN_RESUMER, FN_MAIN_OTHER, FN_HOG, FN_SLEEPER, FN_MAIN_MIGRATE, FN_BETTER, FN_KICKED, FN_MAIN_KICK, FN_WAITER,
       FN_PENDING, FN_MAIN_PENDING, FN_EXITING, FN_EXITER, FN_MAIN_EXIT, FN_RING, FN_MAIN_RING, FN_SELF,
       FN_RETURNS0, FN_MAIN_FATAL, FN_END };
static void guest_end(Cpu *c);

// basic: created, switched to, back to the thread, switched to again, returns to the thread
static void g_basic_a(Cpu *c) {
    CHECK(c->x[0] == 0x1234);   // the argument
    CHECK(fib_current(c) == FIB(0) && tt_fibre(c) == FIB(0) && ft_state(0) == 2 && sp_on_fibre_stack(c));
    CHECK(self_handle(c) == LD32(W(RESULTS)) && c->tpidr == LD64(W(RESULTS) + 0x10));   // the thread's, carried in
    CHECK(c->fpcr == 0x03C00000);                                                         // so is FPCR
    uint64_t keep = opaque();
    ST64(W(RESULTS) + 0x18, keep);
    c->x[19] = 0xF1B0;   // the fibre's own registers
    record(2);
    fib_switch(c, 0);
    CHECK(keep == LD64(W(RESULTS) + 0x18) && c->x[19] == 0xF1B0);
    CHECK(fib_current(c) == FIB(0) && tt_fibre(c) == FIB(0) && ft_state(0) == 2);
    record(4);
    c->x[0] = 0;   // returns: back to the thread's own context
}
static void g_main_basic(Cpu *c) {
    ST32(W(RESULTS), self_handle(c));
    ST64(W(RESULTS) + 0x10, c->tpidr);
    fib_init(c, 0, G(FN_BASIC_A), 0x1234);
    CHECK(ft_state(0) == 1 && LD8(FIB(0) + 0x21) == 0 && LD64(FIB(0) + 0x28) == G(FN_BASIC_A));
    CHECK(LD64(FIB(0) + 0x30) == 0x1234 && LD64(FIB(0) + 0x38) == FSTACK(0) && LD64(FIB(0) + 0x40) == FSTACK(0));
    CHECK(LD64(FIB(0) + 0x48) == FSIZE && fib_current(c) == 0 && tt_fibre(c) == 0);
    for (int i = 19; i <= 28; i++) c->x[i] = 0x1900 + (uint64_t)i;
    c->fpcr = 0x03C00000;
    uint64_t keep = opaque(), sp = c->sp;
    ST64(W(RESULTS) + 8, keep);
    record(1);
    fib_switch(c, FIB(0));
    CHECK(keep == LD64(W(RESULTS) + 8) && c->sp == sp);
    for (int i = 19; i <= 28; i++) CHECK(c->x[i] == 0x1900 + (uint64_t)i);
    CHECK(fib_current(c) == 0 && tt_fibre(c) == 0 && ft_state(0) == 1);
    record(3);
    fib_switch(c, FIB(0));
    record(5);
    CHECK(ft_state(0) == 4 && fib_current(c) == 0 && tt_fibre(c) == 0);
    fib_final(c, 0);
    CHECK(ft_state(0) == 0);
    static const uint32_t want[] = {1, 2, 3, 4, 5};
    CHECK(order_is(5, want));
    c->fpcr = 0;
    exit_thread(c);
}

// returns: a fibre's function returns the next fibre; a finished fibre is initialized again
static void g_ret_a(Cpu *c) {
    record((uint32_t)c->x[0]);
    c->x[0] = FIB(1);
}
static void g_ret_b(Cpu *c) {
    uint32_t arg = (uint32_t)c->x[0];
    CHECK(ft_state(0) == 4 && ft_state(1) == 2 && fib_current(c) == FIB(1));
    record(arg);
    c->x[0] = 0;
}
static void g_ret_again(Cpu *c) {
    record((uint32_t)c->x[0]);
    c->x[0] = 0;
}
static void g_main_returns(Cpu *c) {
    fib_init(c, 0, G(FN_RET_A), 1);
    fib_init(c, 1, G(FN_RET_B), 2);
    fib_switch(c, FIB(0));
    record(3);
    CHECK(ft_state(0) == 4 && ft_state(1) == 4 && tt_fibre(c) == 0);
    fib_init(c, 0, G(FN_RET_AGAIN), 4);   // without FinalizeFiber, as the SDK allows
    CHECK(ft_state(0) == 1);
    fib_switch(c, FIB(0));
    static const uint32_t want[] = {1, 2, 3, 4};
    CHECK(order_is(4, want));
    fib_final(c, 0);
    fib_final(c, 1);
    exit_thread(c);
}

// other_thread: a fibre left on one thread is resumed by another thread from a fibre of that thread's, takes its thread
// state, and hands back to that fibre, which returns to the second thread's own context
static void g_moving(Cpu *c) {
    uint64_t keep = opaque();
    ST64(W(RESULTS) + 8, keep);
    CHECK(self_handle(c) == LD32(W(RESULTS)));
    c->x[20] = 0x2020;
    record(1);
    fib_switch(c, 0);   // back to the main thread's own context; a fibre of the second thread enters it again
    CHECK(keep == LD64(W(RESULTS) + 8) && c->x[20] == 0x2020);
    CHECK(self_handle(c) == LD32(W(RESULTS) + 4) && c->tpidrro == LD64(W(RESULTS) + 0x10));
    CHECK(fib_current(c) == FIB(0) && tt_fibre(c) == FIB(0) && ft_state(1) == 1 && sp_on_fibre_stack(c));
    record(4);
    fib_switch(c, FIB(1));
    CHECK(!"a fibre left for good was entered again");
}
static void g_relay(Cpu *c) {   // fibre 1, entered by the second thread
    CHECK(self_handle(c) == LD32(W(RESULTS) + 4) && ft_state(0) == 1);
    record(3);
    fib_switch(c, FIB(0));
    CHECK(fib_current(c) == FIB(1) && tt_fibre(c) == FIB(1) && ft_state(0) == 1);
    record(5);
    fib_switch(c, 0);   // back to the second thread's own context
    CHECK(!"a fibre left for good was entered again");
}
static void g_resumer(Cpu *c) {
    ST32(W(RESULTS) + 4, self_handle(c));
    ST64(W(RESULTS) + 0x10, c->tpidrro);
    CHECK(tt_fibre(c) == 0 && ft_state(0) == 1);
    record(2);
    fib_switch(c, FIB(1));
    record(6);
    CHECK(fib_current(c) == 0 && tt_fibre(c) == 0 && ft_state(0) == 1 && ft_state(1) == 1);
    exit_thread(c);
}
static void g_main_other(Cpu *c) {
    ST32(W(RESULTS), self_handle(c));
    fib_init(c, 0, G(FN_MOVING), 0);
    fib_init(c, 1, G(FN_RELAY), 0);
    fib_switch(c, FIB(0));
    CHECK(tt_fibre(c) == 0 && ft_state(0) == 1);
    uint32_t t = create(c, G(FN_RESUMER), 0, 44, 1);
    start(c, t);
    join(c, t);
    static const uint32_t want[] = {1, 2, 3, 4, 5, 6};
    CHECK(order_is(6, want) && tt_fibre(c) == 0);
    fib_final(c, 0);
    fib_final(c, 1);
    exit_thread(c);
}

// migrate_core: a fibre's thread sleeps on the fibre and wakes on another core; then the fibre, left there, is resumed
// after its thread slept on its own context and woke back on the first core
static void g_hog(Cpu *c) {   // worse than main, pinned to core 0: takes core 0 only while main sleeps
    ST32(W(GO), 1);
    while (!__atomic_load_n((uint32_t *)(uintptr_t)W(STOP), __ATOMIC_ACQUIRE)) __builtin_ia32_pause();
    exit_thread(c);
}
static void g_sleeper(Cpu *c) {
    uint64_t keep = opaque();
    ST64(W(RESULTS) + 8, keep);
    c->x[21] = 0x2121;
    CHECK(core_now(c) == 0);
    sv(c, 0x0B, 5000000, 0, 0, 0);   // sleeps 5 ms on the fibre: the hog takes core 0, so the thread wakes on core 1
    CHECK(core_now(c) == 1);
    CHECK(keep == LD64(W(RESULTS) + 8) && c->x[21] == 0x2121 && sp_on_fibre_stack(c));
    CHECK(fib_current(c) == FIB(0) && tt_fibre(c) == FIB(0) && self_handle(c) == LD32(W(RESULTS)));
    record(1);
    fib_switch(c, 0);
    CHECK(core_now(c) == 0);         // resumed on core 0
    CHECK(keep == LD64(W(RESULTS) + 8) && c->x[21] == 0x2121 && fib_current(c) == FIB(0) && tt_fibre(c) == FIB(0));
    record(3);
    c->x[0] = 0;
}
static void g_main_migrate(Cpu *c) {
    ST32(W(RESULTS), self_handle(c));
    CHECK(sv(c, 0x0F, 0xFFFF8000u, 0, 3, 0) == 0);   // ideal core 0, may run on cores 0 and 1
    uint32_t h = create(c, G(FN_HOG), 0, 50, 0);
    start(c, h);
    CHECK(core_now(c) == 0 && LD32(W(GO)) == 0);     // worse: it waits for core 0
    fib_init(c, 0, G(FN_SLEEPER), 0);
    fib_switch(c, FIB(0));
    CHECK(core_now(c) == 1);
    record(2);
    ST32(W(STOP), 1);
    join(c, h);                      // the hog ends and frees core 0 (the join may not block: it may have ended)
    sv(c, 0x0B, 1000000, 0, 0, 0);   // sleeps 1 ms on its own context and wakes on core 0, its ideal core, now free
    CHECK(core_now(c) == 0);
    fib_switch(c, FIB(0));
    static const uint32_t want[] = {1, 2, 3};
    CHECK(order_is(3, want) && ft_state(0) == 4);
    fib_final(c, 0);
    exit_thread(c);
}

// kick_on_fibre: a better thread started from a fibre takes the core at the SVC's exit, through the fibre's Cpu
static void g_better(Cpu *c) {
    record(1);
    exit_thread(c);
}
static void g_kicked(Cpu *c) {
    uint32_t h = create(c, G(FN_BETTER), 0, 40, 0);
    ST32(W(HANDLES) + 8, h);
    start(c, h);
    record(2);
    c->x[0] = 0;
}
static void g_main_kick(Cpu *c) {
    fib_init(c, 0, G(FN_KICKED), 0);
    fib_switch(c, FIB(0));
    join(c, LD32(W(HANDLES) + 8));
    static const uint32_t want[] = {1, 2};
    CHECK(order_is(2, want));
    fib_final(c, 0);
    exit_thread(c);
}

// pending_moves: a kick that lands on the thread's own Cpu and is not yet acted on follows it onto the fibre it enters
static void g_waiter(Cpu *c) {   // better than main: waits on the host's event, then records first
    ST32(W(HANDLES) + 0x10, g_event_r);
    CHECK(sv(c, 0x18, 0, W(HANDLES) + 0x10, 1, ~0ull) == 0);
    record(1);
    exit_thread(c);
}
static void g_pending(Cpu *c) {
    record(2);
    c->x[0] = 0;
}
static void g_main_pending(Cpu *c) {
    fib_init(c, 0, G(FN_PENDING), 0);
    uint32_t b = create(c, G(FN_WAITER), 0, 40, 0);
    start(c, b);                     // it runs at once and blocks on the event
    ST32(W(GO), 1);                  // the host helper signals the event: the waiter is ready and kicks this thread
    while (!__atomic_load_n((uint32_t *)(uintptr_t)W(FLAG), __ATOMIC_ACQUIRE)) __builtin_ia32_pause();
    CHECK(__atomic_load_n(&c->pending, __ATOMIC_ACQUIRE));   // on this context's Cpu, not acted on (no safe point)
    c->x[0] = FIB(0);
    c->x[30] = PC + 4;
    c->pc = f_switch;
    hostfn_entry(c);                 // the switch, without the safe point of rt_dispatch's entry
    join(c, b);
    static const uint32_t want[] = {1, 2};
    CHECK(order_is(2, want));
    fib_final(c, 0);
    exit_thread(c);
}

// exit_on_fibre: ExitThread on a fibre ends the thread; the fibre stays running, and FinalizeFiber may free it
static void g_exiting(Cpu *c) {
    record(2);
    exit_thread(c);
    CHECK(!"ExitThread returned on a fibre");
}
static void g_exiter(Cpu *c) {
    record(1);
    fib_switch(c, FIB(0));
    CHECK(!"the thread's own context came back after ExitThread");
}
static void g_main_exit(Cpu *c) {
    fib_init(c, 0, G(FN_EXITING), 0);
    uint32_t t = create(c, G(FN_EXITER), 0, 44, 1);
    start(c, t);
    join(c, t);
    static const uint32_t want[] = {1, 2};
    CHECK(order_is(2, want) && ft_state(0) == 2);
    fib_final(c, 0);
    CHECK(ft_state(0) == 0);
    exit_thread(c);
}

// ring: RING fibres pass control round ROUNDS times, each checking the order; the first one's return ends the ring
static void g_ring(Cpu *c) {
    uint64_t me = c->x[0], keep = opaque();
    ST64(W(KEEP) + 8 * me, keep);
    for (uint64_t k = 0;; k++) {
        if (me == 0 && k == ROUNDS) break;
        CHECK(LD64(W(SUM)) == k * RING + me && fib_current(c) == FIB(me) && tt_fibre(c) == FIB(me));
        ST64(W(SUM), LD64(W(SUM)) + 1);
        fib_switch(c, FIB((me + 1) % RING));
        CHECK(keep == LD64(W(KEEP) + 8 * me));
    }
    c->x[0] = 0;
}
static void g_main_ring(Cpu *c) {
    for (int i = 0; i < RING; i++) fib_init(c, i, G(FN_RING), (uint64_t)i);
    fib_switch(c, FIB(0));
    CHECK(LD64(W(SUM)) == (uint64_t)ROUNDS * RING && ft_state(0) == 4);
    for (int i = 1; i < RING; i++) CHECK(ft_state(i) == 1);
    for (int i = 0; i < RING; i++) fib_final(c, i);
    exit_thread(c);
}

// fatal cases
static void g_self(Cpu *c) {
    if (LD32(W(RESULTS) + 0x100) == 1) fib_switch(c, FIB(0));   // switch_running
    else fib_final(c, 0);                                     // finalize_running
    CHECK(!"the SDK aborts here");
}
static void g_returns0(Cpu *c) { c->x[0] = 0; }
static void g_main_fatal(Cpu *c) {
    uint32_t which = LD32(W(RESULTS) + 0x100);
    if (which == 1 || which == 2) {
        fib_init(c, 0, G(FN_SELF), 0);
        fib_switch(c, FIB(0));
    } else if (which == 3) {   // switch_finished
        fib_init(c, 0, G(FN_RETURNS0), 0);
        fib_switch(c, FIB(0));
        fib_switch(c, FIB(0));
    } else if (which == 4) {   // switch_unknown
        fib_switch(c, FIB(5));
    } else {                   // null_own
        fib_switch(c, 0);
    }
    CHECK(!"the case went on");
}

static void guest_end(Cpu *c) { (void)c; }

static const GuestFn g_fns[FN_END + 1] = {
    g_basic_a, g_main_basic, g_ret_a, g_ret_b, g_ret_again, g_main_returns, g_moving, g_relay, g_resumer,
    g_main_other, g_hog, g_sleeper, g_main_migrate, g_better, g_kicked, g_main_kick, g_waiter, g_pending,
    g_main_pending, g_exiting, g_exiter, g_main_exit, g_ring, g_main_ring, g_self, g_returns0, g_main_fatal, guest_end};
static uint64_t g_addrs[FN_END + 1];

// ---- the host side
static const uint64_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static void boot(uint64_t cores, int no_async) {
    RtKernelConfig k = {.core_mask = cores, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                        .program_id = 0x0100000000001000ull, .is_application = 1, .no_async_preempt = no_async};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++) k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    rt_kernel_init(&k);
    // what the loader does for the sdk's definitions and main's imports of these symbols
    f_init = rt_hostfn_bind(SYM_INIT);
    f_final = rt_hostfn_bind(SYM_FINAL);
    f_switch = rt_hostfn_bind(SYM_SWITCH);
    f_current = rt_hostfn_bind(SYM_CURRENT);
    CHECK(f_init && f_final && f_switch && f_current);
    uint32_t r;
    g_event_w = rt_kernel_event_create(&r);
    g_event_r = r;
    for (int i = 0; i <= FN_END; i++) g_addrs[i] = G(i);
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, g_addrs, g_fns, (uint64_t)FN_END + 1, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 64ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
}

static void run(uint64_t cores, int fn) {
    boot(cores, 0);
    rt_kernel_run(G(fn), 0);
}

// pending_moves: once the main thread says GO, signal the event its waiter blocks on, then let the main thread go on.
static void event_helper(void *arg) {
    (void)arg;
    while (!__atomic_load_n(&HEAP, __ATOMIC_ACQUIRE) || !__atomic_load_n((uint32_t *)(uintptr_t)W(GO), __ATOMIC_ACQUIRE))
        plat_sleep_ns(100000);
    rt_kernel_event_signal(g_event_w);
    __atomic_store_n((uint32_t *)(uintptr_t)W(FLAG), 1, __ATOMIC_RELEASE);
}

static void print_stats(void) {
    RtKernelStats s = rt_kernel_stats();
    printf("fibres: created %llu, live %llu, switches %llu\n", (unsigned long long)s.fibres_created,
           (unsigned long long)s.fibres_live, (unsigned long long)s.fibre_switches);
}

// A case that has not ended after HANG_NS logs the kernel's state, then exits 3 (see kernel_test.c).
#define HANG_NS 20000000000ull
static void watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(HANG_NS);
    uint64_t h = __atomic_load_n(&HEAP, __ATOMIC_RELAXED);
    if (h)
        rt_log("HUNG: order count %llu, flag %u, go %u, stop %u, sum %llu", (unsigned long long)LD64(h + ORDER_N),
               LD32(h + FLAG), LD32(h + GO), LD32(h + STOP), (unsigned long long)LD64(h + SUM));
    rt_kernel_dump();
    _Exit(3);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: fibre_test <case>\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    const char *cs = argv[1];
    static const char *const fatal[] = {"", "switch_running", "finalize_running", "switch_finished", "switch_unknown",
                                        "null_own"};
    int which = 0;
    for (int i = 1; i < 6; i++)
        if (!strcmp(cs, fatal[i])) which = i;
    if (!strcmp(cs, "basic")) {
        run(0x1, FN_MAIN_BASIC);
        RtKernelStats s = rt_kernel_stats();
        CHECK(s.fibres_created == 1 && s.fibres_live == 0 && s.fibre_switches == 4);
    } else if (!strcmp(cs, "returns")) {
        run(0x1, FN_MAIN_RETURNS);
        RtKernelStats s = rt_kernel_stats();
        CHECK(s.fibres_created == 3 && s.fibres_live == 0 && s.fibre_switches == 5);
    } else if (!strcmp(cs, "other_thread")) run(0x3, FN_MAIN_OTHER);
    else if (!strcmp(cs, "migrate_core")) run(0x3, FN_MAIN_MIGRATE);
    else if (!strcmp(cs, "kick_on_fibre")) run(0x1, FN_MAIN_KICK);
    else if (!strcmp(cs, "pending_moves")) {
        CHECK(plat_thread_start(event_helper, NULL, 1ull << 20));
        boot(0x1, 1);   // no asynchronous preemption: the kick waits for a safe point
        rt_kernel_run(G(FN_MAIN_PENDING), 0);
    } else if (!strcmp(cs, "exit_on_fibre")) {
        run(0x3, FN_MAIN_EXIT);
        CHECK(rt_kernel_stats().fibres_live == 0);
    } else if (!strcmp(cs, "ring")) {
        run(0x1, FN_MAIN_RING);
        RtKernelStats s = rt_kernel_stats();
        CHECK(s.fibres_created == RING && s.fibres_live == 0 && s.fibre_switches == (uint64_t)ROUNDS * RING + 2);
    } else if (which) {
        boot(0x1, 0);
        ST32(W(RESULTS) + 0x100, (uint32_t)which);
        rt_kernel_run(G(FN_MAIN_FATAL), 0);
    } else if (!strcmp(cs, "host_switch")) {
        boot(0x1, 0);
        Cpu h;
        memset(&h, 0, sizeof h);
        h.pc = f_switch;
        rt_dispatch(&h);
    } else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    print_stats();
    printf("ok %s\n", cs);
    return 0;
}
