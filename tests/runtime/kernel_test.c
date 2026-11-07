// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the kernel (runtime/core/kernel.c, ksync.c, kinfo.c): synthetic guest programs written by hand in
// the lifter's shape run on guest threads and virtual cores. The guest mutex and condition variable below follow the
// SDK's public protocol (a mutex word of owner handle | waiters bit, SVCs only on contention). No game code is
// involved: the module is a synthetic layout written by tools/layout.py.
//
//   kernel_test CASE    runs one case; prints "ok CASE" and exits 0, or exits 1 with a FAIL line. The cases that end
//                       in a fatal error exit with the runtime's code (tests/test_runtime_kernel.py).
//
// The guest functions come first and together, then a last one (guest_end) that no thread runs: asynchronous
// preemption takes the module's range, from its lowest function to its end marker (endmark.c, linked after this file),
// as lifted code, so everything in this file after the first guest function counts; nothing in that range may take a
// lock of the C library while a guest thread runs (printing happens on the host thread, after rt_kernel_run).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "plat.h"   // the watchdog's thread and sleep, the busy thread's CPU time

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))   // the guest address of guest function i
#define PC 0x7000000040ull           // the guest pc the test's SVCs report
#define WAITERS 0x40000000u
#define R_TIMED_OUT 0xEA01u
#define R_CANCELLED 0xEC01u
#define R_INVALID_STATE 0xFA01u
#define R_INVALID_HANDLE 0xE401u
#define R_INVALID_CORE 0xE201u
#define R_INVALID_COMBINATION 0xE801u
#define R_INVALID_PRIORITY 0xE001u
#define R_BUSY 0xF401u

// Guest heap words, from the heap base the host sets before the run.
static uint64_t HEAP;
enum { FLAG = 0x0, COUNTER = 0x8, MUTEX = 0x10, KEY = 0x14, SLOT = 0x18, SUM = 0x20, ADDR = 0x28, ADDR64 = 0x30,
       STOP = 0x38, ORDER_N = 0x40, SPINNING = 0x48, EPOCH = 0x4C, ACK = 0x50, ORDER = 0x100, RESULTS = 0x200,
       HANDLES = 0x300, STACKS = 0x100000 };
#define W(off) (HEAP + (off))
static uint32_t g_event_w, g_event_r;   // made by the host before the run (rt_kernel_event_create)
static uint64_t g_hostfn_block;         // a host function that blocks (rt_kernel_block_begin)

// ---- helpers the guest functions use: inline, no locks
static inline uint64_t sv(Cpu *c, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3;
    rt_svc(c, imm, PC);
    return c->x[0];
}
static inline uint32_t create(Cpu *c, uint64_t entry, uint64_t arg, int prio, int core) {
    c->x[1] = entry, c->x[2] = arg, c->x[3] = W(STACKS) + 0x10000 * (arg + 1), c->x[4] = (uint32_t)prio,
    c->x[5] = (uint32_t)core;
    rt_svc(c, 0x08, PC);
    CHECK(c->x[0] == 0);
    return (uint32_t)c->x[1];
}
static inline void start(Cpu *c, uint32_t h) { CHECK(sv(c, 0x09, h, 0, 0, 0) == 0); }
static inline void wait_one(Cpu *c, uint32_t h) {
    ST32(W(HANDLES), h);
    CHECK(sv(c, 0x18, 0, W(HANDLES), 1, ~0ull) == 0);
}
static inline void join(Cpu *c, uint32_t h) {
    wait_one(c, h);
    CHECK(sv(c, 0x16, h, 0, 0, 0) == 0);
}
static inline void exit_thread(Cpu *c) { sv(c, 0x0A, 0, 0, 0, 0); }
static inline uint32_t self_handle(const Cpu *c) { return LD32(c->tpidrro + 0x110); }
static inline void record(uint32_t v) {
    uint64_t i = __atomic_fetch_add((uint64_t *)(uintptr_t)W(ORDER_N), 1, __ATOMIC_SEQ_CST);
    ST32(W(ORDER) + 4 * i, v);
}
static inline uint32_t order(int i) { return LD32(W(ORDER) + 4ull * (uint64_t)i); }

// The SDK's mutex protocol over the kernel's ArbitrateLock and ArbitrateUnlock.
static inline void mutex_lock(Cpu *c, uint64_t m) {
    uint32_t self = self_handle(c);
    for (;;) {
        uint32_t v = 0;
        if (__atomic_compare_exchange_n((uint32_t *)(uintptr_t)m, &v, self, 0, __ATOMIC_ACQUIRE, __ATOMIC_RELAXED)) return;
        if (!(v & WAITERS) &&
            !__atomic_compare_exchange_n((uint32_t *)(uintptr_t)m, &v, v | WAITERS, 0, __ATOMIC_RELAXED, __ATOMIC_RELAXED))
            continue;
        CHECK(sv(c, 0x1A, v & ~WAITERS, m, self, 0) == 0);
        if ((__atomic_load_n((uint32_t *)(uintptr_t)m, __ATOMIC_ACQUIRE) & ~WAITERS) == self) return;
    }
}
static inline void mutex_unlock(Cpu *c, uint64_t m) {
    uint32_t v = self_handle(c);
    if (__atomic_compare_exchange_n((uint32_t *)(uintptr_t)m, &v, 0, 0, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) return;
    CHECK(sv(c, 0x1B, m, 0, 0, 0) == 0);
}
// The SDK's condition wait: the kernel releases the mutex, and hands it back on a signal; after a timeout the caller
// takes it again.
static inline uint32_t cond_wait(Cpu *c, uint64_t m, uint64_t key, int64_t ns) {
    uint32_t r = (uint32_t)sv(c, 0x1C, m, key, self_handle(c), (uint64_t)ns);
    if (r == R_TIMED_OUT) mutex_lock(c, m);
    return r;
}
// The SDK's signal reads the key in user mode and calls the kernel only when a waiter set it.
static inline void cond_signal(Cpu *c, uint64_t key) {
    if (__atomic_load_n((uint32_t *)(uintptr_t)key, __ATOMIC_ACQUIRE)) sv(c, 0x1D, key, 0, 0, 0);
}

// ---- guest functions, in address order (the module table below lists them)
static void guest_end(Cpu *c);

// threads: exit from three frames down; the frames above are never returned to
static void g_exit_deep(Cpu *c) {
    record(2);
    exit_thread(c);
    CHECK(!"ExitThread returned");
}
static void g_exit_mid(Cpu *c) {
    record(1);
    c->x[30] = G(1) + 8;
    c->pc = G(0);
    rt_dispatch(c);
    CHECK(!"the thread came back from ExitThread");
}
static void g_exit_top(Cpu *c) {
    CHECK(c->x[0] == 7);   // the argument
    CHECK(c->tpidrro && LD64(c->tpidrro + 0x1F8) == c->tpidrro + 0x200 && self_handle(c) != 0);
    CHECK(LD32(c->tpidrro + 0x100) == 0);
    record(0);
    c->x[30] = G(2) + 8;
    c->pc = G(1);
    rt_dispatch(c);
}
static void g_main_threads(Cpu *c) {
    uint32_t me = (uint32_t)c->x[1];
    CHECK(me == self_handle(c));
    uint32_t t = create(c, G(2), 7, 44, 0);
    CHECK(sv(c, 0x25, 0, t, 0, 0) == 0 && c->x[1] >= 0x51);   // GetThreadId
    CHECK(sv(c, 0x0C, 0, t, 0, 0) == 0 && c->x[1] == 44);     // GetThreadPriority
    start(c, t);
    CHECK(sv(c, 0x09, t, 0, 0, 0) == R_INVALID_STATE);        // started twice
    wait_one(c, t);
    CHECK(LD32(W(ORDER_N)) == 3 && order(0) == 0 && order(1) == 1 && order(2) == 2);
    CHECK(sv(c, 0x0C, 0, t, 0, 0) == 0);                       // the handle stays valid after the end
    CHECK(sv(c, 0x16, t, 0, 0, 0) == 0);
    CHECK(sv(c, 0x0C, 0, t, 0, 0) == R_INVALID_HANDLE);        // closed: stale
    CHECK(sv(c, 0x16, 0xFFFF8000u, 0, 0, 0) == R_INVALID_HANDLE);
    // the thread-local region lies in the kernel region and QueryMemory reports it ThreadLocal
    CHECK(sv(c, 0x06, W(RESULTS), 0, c->tpidrro, 0) == 0 && LD32(W(RESULTS) + 16) == 0xC && LD32(W(RESULTS) + 24) == 3);
    exit_thread(c);
}

// priority: one core; threads started while the main thread holds it run in priority order
static void g_record_arg(Cpu *c) {
    record((uint32_t)c->x[0]);
    exit_thread(c);
}
static void g_main_priority(Cpu *c) {
    uint32_t h[4];
    static const int prio[4] = {50, 40, 45, 40};
    static const uint32_t ran[4] = {0, 1, 1, 2};   // a better thread runs as StartThread returns, a worse one waits
    for (int i = 0; i < 4; i++) {
        h[i] = create(c, G(5), (uint64_t)i + 1, prio[i], 0);
        start(c, h[i]);
        CHECK(LD32(W(ORDER_N)) == ran[i]);
    }
    for (int i = 0; i < 4; i++) join(c, h[i]);
    CHECK(LD32(W(ORDER_N)) == 4);
    CHECK(order(0) == 2 && order(1) == 4 && order(2) == 3 && order(3) == 1);
    c->x[4] = 27, c->x[5] = 0, c->x[1] = G(5), c->x[3] = W(STACKS);   // a priority outside 28..59
    rt_svc(c, 0x08, PC);
    CHECK(c->x[0] == R_INVALID_PRIORITY);
    c->x[4] = 44, c->x[5] = 3;                                          // a core outside the mask
    rt_svc(c, 0x08, PC);
    CHECK(c->x[0] == R_INVALID_CORE);
    exit_thread(c);
}

// mutex: counting under the mutex on three cores; hand-off order on one core in g_main_mutex_order
#define COUNT_ITER 3000
static void g_mutex_count(Cpu *c) {
    for (int i = 0; i < COUNT_ITER; i++) {
        mutex_lock(c, W(MUTEX));
        uint64_t v = LD64(W(COUNTER));
        if ((i & 63) == 0) sv(c, 0x0B, 0, 0, 0, 0);   // yield inside the section: others contend
        ST64(W(COUNTER), v + 1);
        mutex_unlock(c, W(MUTEX));
    }
    exit_thread(c);
}
static void g_main_mutex(Cpu *c) {
    uint32_t h[4];
    for (int i = 0; i < 4; i++) {
        h[i] = create(c, G(7), (uint64_t)i, 44, i % 3);
        CHECK(sv(c, 0x0F, h[i], (uint64_t)-1, 7, 0) == 0);   // any of the three cores
        start(c, h[i]);
    }
    for (int i = 0; i < 4; i++) join(c, h[i]);
    CHECK(LD64(W(COUNTER)) == 4 * COUNT_ITER);
    CHECK(LD32(W(MUTEX)) == 0);
    // ArbitrateLock reads the word before the handle (ksync.c). h[0] is closed now: no error while the word names
    // nobody, as for a waiter that reaches the SVC after its owner has unlocked and ended; an invalid handle once it
    // does.
    CHECK(sv(c, 0x1A, h[0], W(MUTEX), self_handle(c), 0) == 0);
    ST32(W(MUTEX), h[0] | WAITERS);
    CHECK(sv(c, 0x1A, h[0], W(MUTEX), self_handle(c), 0) == R_INVALID_HANDLE);
    ST32(W(MUTEX), 0);
    exit_thread(c);
}
static void g_mutex_order(Cpu *c) {
    uint32_t id = (uint32_t)c->x[0];
    mutex_lock(c, W(MUTEX));
    record(id);
    mutex_unlock(c, W(MUTEX));
    exit_thread(c);
}
static void g_main_mutex_order(Cpu *c) {
    uint32_t h[3];
    static const int prio[3] = {40, 35, 45};
    mutex_lock(c, W(MUTEX));
    for (int i = 0; i < 3; i++) {
        h[i] = create(c, G(9), (uint64_t)i + 1, prio[i], 0);
        start(c, h[i]);
    }
    mutex_unlock(c, W(MUTEX));
    for (int i = 0; i < 3; i++) join(c, h[i]);
    CHECK(LD32(W(ORDER_N)) == 3 && order(0) == 2 && order(1) == 1 && order(2) == 3);
    exit_thread(c);
}

// condition variable: one-slot producer and consumer on two cores, then a timed-out wait
#define ITEMS 500
static void g_producer(Cpu *c) {
    for (uint64_t i = 1; i <= ITEMS; i++) {
        mutex_lock(c, W(MUTEX));
        while (LD64(W(SLOT))) CHECK(cond_wait(c, W(MUTEX), W(KEY), -1) == 0);
        ST64(W(SLOT), i);
        cond_signal(c, W(KEY));
        mutex_unlock(c, W(MUTEX));
    }
    exit_thread(c);
}
static void g_main_condvar(Cpu *c) {
    uint32_t p = create(c, G(11), 1, 44, 1);
    start(c, p);
    for (int n = 0; n < ITEMS; n++) {
        mutex_lock(c, W(MUTEX));
        while (!LD64(W(SLOT))) CHECK(cond_wait(c, W(MUTEX), W(KEY), -1) == 0);
        ST64(W(SUM), LD64(W(SUM)) + LD64(W(SLOT)));
        ST64(W(SLOT), 0);
        cond_signal(c, W(KEY));
        mutex_unlock(c, W(MUTEX));
    }
    join(c, p);
    CHECK(LD64(W(SUM)) == (uint64_t)ITEMS * (ITEMS + 1) / 2);
    mutex_lock(c, W(MUTEX));
    CHECK(cond_wait(c, W(MUTEX), W(KEY), 1000000) == R_TIMED_OUT);   // nobody signals: 1 ms, then the mutex again
    CHECK(LD32(W(MUTEX)) == self_handle(c));
    mutex_unlock(c, W(MUTEX));
    exit_thread(c);
}

// address arbiter: wait if equal, decrement and wait if less, 64-bit wait if equal; signal kinds 0, 1 and 2
static void g_arb_waiter(Cpu *c) {
    if (c->x[0] == 1) CHECK(sv(c, 0x34, W(ADDR), 2, 0, ~0ull) == 0);               // wait while ADDR == 0
    else CHECK(sv(c, 0x34, W(ADDR64), 3, 0x1122334455667788ull, ~0ull) == 0);   // 64 bits
    record((uint32_t)c->x[0]);
    exit_thread(c);
}
static void g_main_arbiter(Cpu *c) {
    CHECK(sv(c, 0x34, W(ADDR), 2, 5, 0) == R_INVALID_STATE);   // not equal: no wait
    CHECK(sv(c, 0x34, W(ADDR), 2, 0, 0) == R_TIMED_OUT);       // equal, timeout 0
    ST32(W(ADDR) + 4, 3);
    CHECK(sv(c, 0x34, W(ADDR) + 4, 1, 5, 0) == R_TIMED_OUT && LD32(W(ADDR) + 4) == 2);   // decremented, then timed out
    ST64(W(ADDR64), 0x1122334455667788ull);
    uint32_t a = create(c, G(13), 1, 40, 0), b = create(c, G(13), 2, 40, 0);
    start(c, a);
    start(c, b);   // both better: they ran to their waits
    CHECK(LD32(W(ORDER_N)) == 0);
    CHECK(sv(c, 0x35, W(ADDR), 1, 7, 1) == R_INVALID_STATE);   // value is 0, not 7
    CHECK(sv(c, 0x35, W(ADDR), 1, 0, 1) == 0 && LD32(W(ADDR)) == 1);   // increment if equal, wake one
    ST64(W(ADDR64), 0);
    CHECK(sv(c, 0x35, W(ADDR64), 0, 0, -1) == 0);              // wake all at the 64-bit word
    join(c, a);
    join(c, b);
    CHECK(LD32(W(ORDER_N)) == 2);
    ST32(W(ADDR) + 8, 9);
    CHECK(sv(c, 0x35, W(ADDR) + 8, 2, 9, 1) == 0 && LD32(W(ADDR) + 8) == 10);   // no waiter: value + 1
    exit_thread(c);
}

// events: a waiter on the readable end, the writable end signalled, ResetSignal, CancelSynchronization
static void g_event_waiter(Cpu *c) {
    ST32(W(HANDLES) + 0x40, g_event_r);
    CHECK(sv(c, 0x18, 0, W(HANDLES) + 0x40, 1, ~0ull) == 0 && c->x[1] == 0);
    record(1);
    CHECK(sv(c, 0x18, 0, W(HANDLES) + 0x40, 0, ~0ull) == R_CANCELLED);   // no handle: only the cancel ends it
    record(2);
    exit_thread(c);
}
static void g_main_events(Cpu *c) {
    CHECK(sv(c, 0x18, 0, W(HANDLES), 0, 1000000) == R_TIMED_OUT);   // count 0: a 1 ms sleep
    uint32_t w = create(c, G(15), 0, 40, 0);
    start(c, w);
    CHECK(LD32(W(ORDER_N)) == 0);
    CHECK(sv(c, 0x17, g_event_r, 0, 0, 0) == R_INVALID_STATE);      // not signalled
    CHECK(sv(c, 0x11, g_event_r, 0, 0, 0) == R_INVALID_HANDLE);     // signal takes the writable end
    CHECK(sv(c, 0x11, g_event_w, 0, 0, 0) == 0);
    CHECK(LD32(W(ORDER_N)) == 1);                                    // the better waiter ran at once
    CHECK(sv(c, 0x19, w, 0, 0, 0) == 0);
    join(c, w);
    CHECK(LD32(W(ORDER_N)) == 2);
    CHECK(sv(c, 0x17, g_event_r, 0, 0, 0) == 0 && sv(c, 0x17, g_event_r, 0, 0, 0) == R_INVALID_STATE);
    CHECK(sv(c, 0x11, g_event_w, 0, 0, 0) == 0 && sv(c, 0x12, g_event_w, 0, 0, 0) == 0);
    CHECK(sv(c, 0x17, g_event_r, 0, 0, 0) == R_INVALID_STATE);       // ClearEvent cleared it
    exit_thread(c);
}

// spin gate: one core, two threads of one priority; the spinner hands its core over at a gate pass
static void g_spinner(Cpu *c) {
    while (!__atomic_load_n((uint32_t *)(uintptr_t)W(FLAG), __ATOMIC_ACQUIRE)) rt_spin_gate(c, G(17));
    record(1);
    exit_thread(c);
}
static void g_setter(Cpu *c) {
    record(2);
    __atomic_store_n((uint32_t *)(uintptr_t)W(FLAG), 1, __ATOMIC_RELEASE);
    exit_thread(c);
}
static void g_main_spin(Cpu *c) {
    uint32_t s = create(c, G(17), 1, 44, 0), t = create(c, G(18), 2, 44, 0);
    start(c, s);
    start(c, t);
    join(c, s);
    join(c, t);
    CHECK(order(0) == 2 && order(1) == 1);
    exit_thread(c);
}

// asynchronous preemption: one core; a worse thread spins in lifted code with no safe point, and a better thread that
// wakes from a sleep takes the core from it. The better thread sleeps again until it wakes to find the worse one
// spinning: on a loaded host the main thread can still hold the core when a sleep ends, and the better thread then
// takes the core at the main thread's next safe point, with no preemption to test.
static void g_tight_spin(Cpu *c) {
    __atomic_store_n((uint32_t *)(uintptr_t)W(SPINNING), 1, __ATOMIC_RELEASE);
    while (!__atomic_load_n((uint32_t *)(uintptr_t)W(FLAG), __ATOMIC_ACQUIRE)) {
    }
    record(1);
    exit_thread(c);
}
static void g_sleeper(Cpu *c) {
    do sv(c, 0x0B, 5000000, 0, 0, 0);   // 5 ms
    while (!__atomic_load_n((uint32_t *)(uintptr_t)W(SPINNING), __ATOMIC_ACQUIRE));
    record(2);
    __atomic_store_n((uint32_t *)(uintptr_t)W(FLAG), 1, __ATOMIC_RELEASE);
    exit_thread(c);
}
static void g_main_async(Cpu *c) {
    uint32_t hi = create(c, G(21), 2, 30, 0), lo = create(c, G(20), 1, 50, 0);
    start(c, hi);   // runs at once, then sleeps
    start(c, lo);
    join(c, hi);
    join(c, lo);
    CHECK(order(0) == 2 && order(1) == 1);
    exit_thread(c);
}

// pause and core mask: a worker counts with safe points on two cores; the main thread pauses it and moves it
static void g_counter(Cpu *c) {
    while (!LD32(W(STOP))) {
        __atomic_fetch_add((uint64_t *)(uintptr_t)W(COUNTER), 1, __ATOMIC_RELAXED);
        c->x[30] = G(23) + 8;
        c->pc = G(0) + 0x100 * 25;   // g_nop: an indirect call, a safe point
        rt_dispatch(c);
        ST32(W(RESULTS) + 0x80, (uint32_t)sv(c, 0x10, 0, 0, 0, 0));   // the core it runs on
    }
    exit_thread(c);
}
static void g_main_activity(Cpu *c) {
    // priority 45: once on the main thread's core it gives the core back at its next safe point, as Horizon runs a
    // thread of equal priority only when the running one yields
    uint32_t w = create(c, G(23), 0, 45, 1);
    start(c, w);
    while (LD64(W(COUNTER)) < 1000) sv(c, 0x0B, 100000, 0, 0, 0);
    CHECK(sv(c, 0x32, 0xFFFF8000u, 1, 0, 0) == R_BUSY);
    CHECK(sv(c, 0x32, w, 1, 0, 0) == 0 && sv(c, 0x32, w, 1, 0, 0) == R_INVALID_STATE);
    sv(c, 0x0B, 2000000, 0, 0, 0);
    uint64_t a = LD64(W(COUNTER));
    sv(c, 0x0B, 5000000, 0, 0, 0);
    CHECK(LD64(W(COUNTER)) == a);   // paused
    CHECK(sv(c, 0x32, w, 0, 0, 0) == 0);
    while (LD64(W(COUNTER)) < a + 1000) sv(c, 0x0B, 100000, 0, 0, 0);
    CHECK(LD32(W(RESULTS) + 0x80) == 1);
    CHECK(sv(c, 0x0F, w, 0, 0x2, 0) == R_INVALID_COMBINATION);   // ideal core outside the mask
    CHECK(sv(c, 0x0F, w, 0, 0x100, 0) == R_INVALID_CORE);        // a core the program has not
    CHECK(sv(c, 0x0F, w, 0, 0x1, 0) == 0);                       // moved to core 0, where the main thread runs
    sv(c, 0x0B, 0, 0, 0, 0);
    a = LD64(W(COUNTER));
    while (LD64(W(COUNTER)) < a + 1000) sv(c, 0x0B, (uint64_t)-2, 0, 0, 0);   // yield to it
    CHECK(LD32(W(RESULTS) + 0x80) == 0);
    CHECK(sv(c, 0x0E, 0, 0, w, 0) == 0 && c->x[1] == 0 && c->x[2] == 1);
    ST32(W(STOP), 1);
    join(c, w);
    exit_thread(c);
}

static void g_nop(Cpu *c) { (void)c; }

// a host function that blocks gives its core up: on one core, the thread that sets the flag runs meanwhile
static void g_blocker(Cpu *c) {
    c->x[30] = G(26) + 8;
    c->pc = g_hostfn_block;
    rt_dispatch(c);
    record(1);
    exit_thread(c);
}
static void g_main_block(Cpu *c) {
    uint32_t b = create(c, G(26), 1, 44, 0), s = create(c, G(18), 2, 44, 0);
    start(c, b);
    start(c, s);
    join(c, b);
    join(c, s);
    CHECK(order(0) == 2 && order(1) == 1);
    exit_thread(c);
}

// time: a thread that runs 30 ms of its own CPU time and one that sleeps 30 ms between two frame marks. The busy thread
// counts its host CPU time, not wall time: on a loaded host a 30 ms wall-clock spin can get less than half of it on a
// CPU. It gives up after 10 s of wall time, and the host's check then fails. The sleeper is charged, between its marks,
// at most the wall time around them less the sleep, since it neither holds a core nor runs while it sleeps. A loaded
// host stretches both sides of that bound alike (a thread granted a core holds it until its host thread runs), so the
// bound holds where a fixed one did not.
static void g_busy(Cpu *c) {
    uint64_t from = plat_thread_cpu_ns(NULL), cap = plat_now_ns() + 10000000000ull;
    while (plat_thread_cpu_ns(NULL) - from < 30000000 && plat_now_ns() < cap) {
    }
    exit_thread(c);
}
static void own_frame(Cpu *c, uint64_t *cpu, uint64_t *core) {   // marks a frame; this thread's time over the last one
    rt_kernel_frame();
    RtThreadTime t[16];
    int n = rt_kernel_thread_times(t, 16);
    for (int i = 0; i < n && i < 16; i++)
        if (t[i].handle == self_handle(c)) {
            *cpu = t[i].cpu_ns, *core = t[i].core_ns;
            return;
        }
    CHECK(!"the thread is missing from the time table");
}
static void g_rest(Cpu *c) {
    uint64_t cpu, core, from = plat_now_ns();
    own_frame(c, &cpu, &core);
    sv(c, 0x0B, 30000000, 0, 0, 0);
    own_frame(c, &cpu, &core);
    ST64(W(RESULTS) + 0x8, plat_now_ns() - from);
    ST64(W(RESULTS) + 0x10, cpu);
    ST64(W(RESULTS) + 0x18, core);
    exit_thread(c);
}
static void g_main_times(Cpu *c) {
    uint32_t b = create(c, G(28), 1, 44, 1), r = create(c, G(29), 2, 44, 0);
    ST32(W(RESULTS), b);
    ST32(W(RESULTS) + 4, r);
    start(c, b);
    start(c, r);
    wait_one(c, b);
    wait_one(c, r);
    exit_thread(c);   // the host reads the times after the run
}

// information SVCs
static void g_main_info(Cpu *c) {
    CHECK(sv(c, 0x29, 0, 0, 0xFFFF8001u, 0) == 0 && c->x[1] == 0x3);   // CoreMask
    CHECK(sv(c, 0x29, 0, 1, 0xFFFF8001u, 0) == 0 && c->x[1] == 0x0FFFFFFFF0000000ull);
    CHECK(sv(c, 0x29, 0, 5, 0xFFFF8001u, 0) == 0 && c->x[1] == 0x20000000ull);   // HeapRegionSize, from rt_mem_info
    CHECK(sv(c, 0x29, 0, 18, 0xFFFF8001u, 0) == 0 && c->x[1] == 0x0100000000001000ull);
    CHECK(sv(c, 0x29, 0, 20, 0xFFFF8001u, 0) == 0 && c->x[1] != 0);
    CHECK(sv(c, 0x29, 0, 23, 0xFFFF8001u, 0) == 0 && c->x[1] == 1);
    CHECK(sv(c, 0x29, 0, 33, 0xFFFF8001u, 0) == 0 && c->x[1] == 0x1000);
    uint64_t r1 = (sv(c, 0x29, 0, 11, 0, 0), c->x[1]), r2 = (sv(c, 0x29, 0, 11, 0, 1), c->x[1]);
    CHECK(r1 != r2);
    CHECK(sv(c, 0x24, 0, 0xFFFF8001u, 0, 0) == 0 && c->x[1] == 0x81);
    uint64_t t0 = sv(c, 0x1E, 0, 0, 0, 0);
    CHECK(t0 > 0);
    memcpy((void *)(uintptr_t)W(RESULTS), "hello from the guest\n", 21);
    CHECK(sv(c, 0x27, W(RESULTS), 21, 0, 0) == 0);
    CHECK(sv(c, 0x26, 0x80000000ull, 0, 0, 0) == 0);   // a notification returns
    exit_thread(c);
}

// fatal cases
static void g_returns(Cpu *c) { c->x[0] = 1; }
static void g_main_returns(Cpu *c) {
    start(c, create(c, G(32), 0, 40, 0));
    sv(c, 0x0B, 50000000, 0, 0, 0);
    exit_thread(c);
}
static void g_main_svc_arg(Cpu *c) { sv(c, (uint32_t)LD32(W(RESULTS) + 0x100), 0, LD32(W(RESULTS) + 0x104), 0, 0); }
static void g_main_touch(Cpu *c) { c->x[0] = LD32(c->tpidrro + 0x3000 * 40); }   // stored: a discarded load is dropped

// the key before the mutex (0x1C): the main thread waits with a timeout of 0 again and again, so each wait sets the key,
// releases the mutex and returns, while a thread on the other core reads the mutex word and the key. It must never find
// the mutex released by a wait while the key is still clear: the SDK's signal would then skip the kernel and the wake
// would be lost (ksync.c). EPOCH is odd while the main thread sets the words up, and a look counts only when EPOCH is
// even and the same before and after it, so it saw one wait's stores; x86 shows another core a thread's stores in order.
// Every KEY_HOLD-th wait, the main thread keeps the mutex released until the watcher has looked (ACK): a host that
// never runs the two threads at once, as a loaded CI runner may, still gives the check something to see. Only looks
// taken during a wait can catch the wrong order, so the mutation run needs a host with two free cores.
#define KEY_WAITS 100000
#define KEY_HOLD 1000
static void g_key_watch(Cpu *c) {
    uint64_t seen = 0, bad = 0;
    ST32(W(FLAG), 1);
    while (!__atomic_load_n((uint32_t *)(uintptr_t)W(STOP), __ATOMIC_ACQUIRE)) {
        uint32_t e = __atomic_load_n((uint32_t *)(uintptr_t)W(EPOCH), __ATOMIC_ACQUIRE);
        uint32_t m = __atomic_load_n((uint32_t *)(uintptr_t)W(MUTEX), __ATOMIC_ACQUIRE);
        uint32_t k = __atomic_load_n((uint32_t *)(uintptr_t)W(KEY), __ATOMIC_ACQUIRE);
        if (!e || (e & 1) || m || __atomic_load_n((uint32_t *)(uintptr_t)W(EPOCH), __ATOMIC_ACQUIRE) != e) continue;
        seen++;
        bad += !k;
        __atomic_store_n((uint32_t *)(uintptr_t)W(ACK), e, __ATOMIC_RELEASE);
    }
    ST64(W(RESULTS) + 0x20, seen);
    ST64(W(RESULTS) + 0x28, bad);
    exit_thread(c);
}
static void g_main_key_order(Cpu *c) {
    uint32_t w = create(c, G(36), 0, 44, 1), self = self_handle(c);
    start(c, w);
    while (!LD32(W(FLAG))) sv(c, 0x0B, 100000, 0, 0, 0);   // the watcher runs
    for (int n = 0; n < KEY_WAITS; n++) {
        __atomic_fetch_add((uint32_t *)(uintptr_t)W(EPOCH), 1, __ATOMIC_SEQ_CST);   // odd: setting up
        __atomic_store_n((uint32_t *)(uintptr_t)W(MUTEX), self, __ATOMIC_SEQ_CST);  // nobody else takes it
        __atomic_store_n((uint32_t *)(uintptr_t)W(KEY), 0, __ATOMIC_SEQ_CST);
        __atomic_fetch_add((uint32_t *)(uintptr_t)W(EPOCH), 1, __ATOMIC_SEQ_CST);   // even: the wait is watched
        CHECK(sv(c, 0x1C, W(MUTEX), W(KEY), self, 0) == R_TIMED_OUT);
        if (n % KEY_HOLD == 0) {
            uint32_t e = __atomic_load_n((uint32_t *)(uintptr_t)W(EPOCH), __ATOMIC_ACQUIRE);
            while (__atomic_load_n((uint32_t *)(uintptr_t)W(ACK), __ATOMIC_ACQUIRE) != e) sv(c, 0x0B, 20000, 0, 0, 0);
        }
    }
    ST32(W(STOP), 1);
    join(c, w);
    exit_thread(c);
}

static void guest_end(Cpu *c) { (void)c; }

static const GuestFn g_fns[] = {g_exit_deep, g_exit_mid, g_exit_top, g_main_threads, NULL, g_record_arg,
                                g_main_priority, g_mutex_count, g_main_mutex, g_mutex_order, g_main_mutex_order,
                                g_producer, g_main_condvar, g_arb_waiter, g_main_arbiter, g_event_waiter,
                                g_main_events, g_spinner, g_setter, g_main_spin, g_tight_spin, g_sleeper,
                                g_main_async, g_counter, g_main_activity, g_nop, g_blocker, g_main_block, g_busy,
                                g_rest, g_main_times, g_main_info, g_returns, g_main_returns, g_main_svc_arg,
                                g_main_touch, g_key_watch, g_main_key_order, guest_end};
#define NFN ((int)(sizeof g_fns / sizeof *g_fns))
static uint64_t g_addrs[NFN];
static GuestFn g_used[NFN];

// ---- the host side
static void host_block(Cpu *c) {   // waits for FLAG with the core given up
    (void)c;
    int token = rt_kernel_block_begin();
    while (!__atomic_load_n((uint32_t *)(uintptr_t)W(FLAG), __ATOMIC_ACQUIRE)) {
    }
    rt_kernel_block_end(token);
}

static const uint64_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static RtKernelConfig kernel_config(uint64_t cores) {
    RtKernelConfig k = {.core_mask = cores, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                        .program_id = 0x0100000000001000ull, .is_application = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++) k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    return k;
}

static void boot(const RtKernelConfig *k) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    rt_kernel_init(k);
    g_hostfn_block = rt_hostfn_register("test host_block", host_block);
    uint32_t r;
    g_event_w = rt_kernel_event_create(&r);
    g_event_r = r;
    int n = 0;
    for (int i = 0; i < NFN; i++)
        if (g_fns[i]) g_addrs[n] = G(i), g_used[n++] = g_fns[i];
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, g_addrs, g_used, (uint64_t)n, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 64ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
}

static void run(uint64_t cores, int fn) {
    RtKernelConfig k = kernel_config(cores);
    boot(&k);
    rt_kernel_run(G(fn), 0);
}

static void case_times(void) {
    run(0x3, 30);
    rt_kernel_frame();
    RtThreadTime t[16];
    int n = rt_kernel_thread_times(t, 16);
    uint32_t busy = LD32(W(RESULTS)), rest = LD32(W(RESULTS) + 4);
    uint64_t around = LD64(W(RESULTS) + 0x8), cpu = LD64(W(RESULTS) + 0x10), core = LD64(W(RESULTS) + 0x18);
    int seen = 0;
    for (int i = 0; i < n && i < 16; i++) {
        if (t[i].handle == busy) {   // totals: the sleeper's marks split the busy thread's run across frames
            seen++;
            CHECK(t[i].cpu_total_ns > 15000000 && t[i].core_total_ns > 15000000 && t[i].exited);
        }
        if (t[i].handle == rest) {
            seen++;
            CHECK(t[i].exited);
        }
    }
    CHECK(seen == 2);
    printf("times: a 30 ms sleep inside %llu us of wall time, charged %llu us of CPU and %llu us on a core\n",
           (unsigned long long)(around / 1000), (unsigned long long)(cpu / 1000), (unsigned long long)(core / 1000));
    CHECK(around >= 30000000 && cpu + 15000000 < around && core + 15000000 < around);
}

static void case_npdm(void) {
    uint8_t d[0x200];
    memset(d, 0, sizeof d);
    memcpy(d, "META", 4);
    d[0xE] = 44, d[0xF] = 0;
    uint32_t v = 0x100000, aoff = 0x80, asize = 0x100, sysres = 0x3000000;
    memcpy(d + 0x1C, &v, 4);
    memcpy(d + 0x14, &sysres, 4);
    memcpy(d + 0x70, &aoff, 4);
    memcpy(d + 0x74, &asize, 4);
    uint8_t *a = d + aoff;
    memcpy(a, "ACI0", 4);
    uint64_t pid = 0x0100000000001000ull;
    memcpy(a + 0x10, &pid, 8);
    uint32_t koff = 0x40, ksize = 6 * 4;
    memcpy(a + 0x30, &koff, 4);
    memcpy(a + 0x34, &ksize, 4);
    // ThreadInfo (priorities 28..59, cores 0..5), two SVC words (0x01-0x17; 0x80), application, 512 handles, padding
    const uint32_t caps[6] = {0x050073B7u, 0x1FFFFFCFu, 0xA000200Fu, 0x00005FFFu, 0x02007FFFu, 0xFFFFFFFFu};
    memcpy(a + koff, caps, sizeof caps);
    RtKernelConfig k;
    const char *why;
    CHECK(rt_npdm_parse(d, sizeof d, &k, &why));
    CHECK(k.core_mask == 0x3F && k.highest_priority == 28 && k.lowest_priority == 59);
    CHECK(k.main_priority == 44 && k.main_core == 0 && k.main_stack_size == 0x100000 && k.system_resource_size == sysres);
    CHECK(k.program_id == pid && k.is_application == 1 && k.handle_table_size == 512);
    CHECK(k.svc_mask[0] == 0x00FFFFFEull && k.svc_mask[2] == 1);   // 0x01..0x17 (bit 0 is SVC 0); 0x80
    d[0x80] = 'X';
    CHECK(!rt_npdm_parse(d, sizeof d, &k, &why) && strstr(why, "ACI0"));
    CHECK(!rt_npdm_parse(d, 0x40, &k, &why) && strstr(why, "META"));
}

// A case that has not ended after HANG_NS logs the heap words the cases count in and the kernel's state, then exits 3:
// a hang reports where it stopped instead of running into the driver's timeout.
#define HANG_NS 20000000000ull
static void watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(HANG_NS);
    uint64_t h = __atomic_load_n(&HEAP, __ATOMIC_RELAXED);
    if (h)
        rt_log("HUNG: counter %llu, order count %u, flag %u, stop %u, mutex %#x, slot %llu", (unsigned long long)LD64(h + COUNTER),
               LD32(h + ORDER_N), LD32(h + FLAG), LD32(h + STOP), LD32(h + MUTEX), (unsigned long long)LD64(h + SLOT));
    else
        rt_log("HUNG before the guest heap was set up");
    rt_kernel_dump();
    _Exit(3);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: kernel_test <case>\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    const char *cs = argv[1];
    if (!strcmp(cs, "threads")) run(0x3, 3);
    else if (!strcmp(cs, "priority")) run(0x1, 6);
    else if (!strcmp(cs, "mutex")) run(0x7, 8);
    else if (!strcmp(cs, "mutex_order")) run(0x1, 10);
    else if (!strcmp(cs, "condvar")) run(0x3, 12);
    else if (!strcmp(cs, "arbiter")) run(0x1, 14);
    else if (!strcmp(cs, "events")) run(0x1, 16);
    else if (!strcmp(cs, "spin_gate")) {
        run(0x1, 19);
        CHECK(rt_kernel_stats().spin_handoffs >= 1);
    } else if (!strcmp(cs, "async")) {
        run(0x1, 22);
        RtKernelStats s = rt_kernel_stats();
        CHECK(s.async_on && s.async_sent >= 1 && s.async_taken >= 1);
        printf("async sent %llu taken %llu elsewhere %llu\n", (unsigned long long)s.async_sent,
               (unsigned long long)s.async_taken, (unsigned long long)s.async_elsewhere);
    } else if (!strcmp(cs, "key_order")) {
        run(0x3, 37);
        uint64_t seen = LD64(W(RESULTS) + 0x20), bad = LD64(W(RESULTS) + 0x28);
        printf("key order: %d waits; the released mutex seen %llu times, with the key clear %llu times\n", KEY_WAITS,
               (unsigned long long)seen, (unsigned long long)bad);
        CHECK(seen >= KEY_WAITS / KEY_HOLD && bad == 0);
    } else if (!strcmp(cs, "activity")) run(0x3, 24);
    else if (!strcmp(cs, "block")) run(0x1, 27);
    else if (!strcmp(cs, "times")) case_times();
    else if (!strcmp(cs, "info")) {
        run(0x3, 31);
        rt_svc_census();
    } else if (!strcmp(cs, "npdm")) case_npdm();
    // fatal cases
    else if (!strcmp(cs, "thread_returns")) run(0x1, 33);
    else if (!strcmp(cs, "svc_outside_mask") || !strcmp(cs, "svc_masked") || !strcmp(cs, "svc_deferred") ||
             !strcmp(cs, "getinfo_unknown") || !strcmp(cs, "arbiter_type") || !strcmp(cs, "break")) {
        RtKernelConfig k = kernel_config(0x1);
        if (!strcmp(cs, "svc_masked")) k.svc_mask[0] &= ~(1ull << 0x27);   // an SVC the runtime implements
        boot(&k);
        uint32_t imm = 0x45, a1 = 0;
        if (!strcmp(cs, "svc_masked")) imm = 0x27;
        else if (!strcmp(cs, "svc_deferred")) imm = 0x21;
        else if (!strcmp(cs, "getinfo_unknown")) imm = 0x29, a1 = 9;
        else if (!strcmp(cs, "arbiter_type")) imm = 0x34, a1 = 7;
        else if (!strcmp(cs, "break")) imm = 0x26;
        ST32(W(RESULTS) + 0x100, imm);
        ST32(W(RESULTS) + 0x104, a1);
        if (!strcmp(cs, "break")) ST32(W(RESULTS) + 0x104, 0);
        rt_kernel_run(G(34), 0);
    } else if (!strcmp(cs, "touch_kernel")) run(0x1, 35);
    else if (!strcmp(cs, "host_svc")) {
        RtKernelConfig k = kernel_config(0x1);
        boot(&k);
        Cpu h;
        memset(&h, 0, sizeof h);
        rt_svc(&h, 0x0B, PC);
    } else if (!strcmp(cs, "init_order")) {
        RtKernelConfig k = kernel_config(0x1);
        rt_kernel_init(&k);
    } else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
