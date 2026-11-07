// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the boundary between generated code and the runtime (cpu.h, runtime.h). The guest functions
// below are written by hand in the shape the lifter emits: they take the Cpu, keep guest registers in it (or, with
// callee-saved trust, in host locals), branch indirectly through rt_dispatch and call host functions by address.
// No game code is involved: the module is a synthetic layout written by tools/layout.py.
//
//   boundary_test CASE    runs one case; prints "ok CASE" and exits 0, or exits 1 with a FAIL line. The cases
//                         that end in a fatal error exit with the runtime's code (tests/test_runtime_boundary.py).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/runtime.h"

_Static_assert(MOD_LAYOUT_LIMIT == RT_MODULE_LIMIT, "tools/layout.py MODULE_LIMIT and runtime.h RT_MODULE_LIMIT");
_Static_assert(sizeof(Cpu) % 16 == 0, "Cpu keeps its 16-byte alignment in arrays");

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// Guest addresses of the synthetic functions.
#define G_ADD3 (MOD_MAIN_BASE + 0x100)
#define G_CALLER (MOD_MAIN_BASE + 0x200)
#define G_LOOP (MOD_MAIN_BASE + 0x300)
#define G_SETJMP (MOD_MAIN_BASE + 0x400)
#define G_DEEP_G (MOD_MAIN_BASE + 0x500)
#define G_DEEP_H (MOD_MAIN_BASE + 0x600)
#define G_SETJMP_RETURNS (MOD_MAIN_BASE + 0x700)
#define G_SDK_DOUBLE (MOD_SDK_BASE + 0x40)
#define SP0 0x7100000000ull

static Cpu cpu;
static uint64_t a_longjmp, a_setjmp, a_throw, a_test_add, a_missing;
static int safepoints;

// ---- synthetic guest functions

static void guest_add3(Cpu *c) { c->x[0] += 3; }

static void guest_sdk_double(Cpu *c) { c->x[0] *= 2; }

// An indirect call (blr x1) with callee-saved trust: x19 lives in a host local across the call.
static void guest_caller(Cpu *c) {
    uint64_t x19 = c->x[19] + 1;
    c->x[30] = G_CALLER + 8;
    c->pc = c->x[1];
    rt_dispatch(c);
    c->x[0] = c->x[0] * 2 + (x19 - c->x[19] - 1);
}

// A loop of indirect branches (br): each pass is a tail call, so the host stack must stay flat.
static void guest_loop(Cpu *c) {
    if (!c->x[2]) return;
    c->x[2]--;
    c->x[3]++;
    c->pc = G_LOOP;
    __attribute__((musttail)) return rt_dispatch(c);
}

// setjmp, as the lifter compiles it (D3): every register in the Cpu, RT_SETJMP at the call, RT_JMP_LEAVE on return.
static uint64_t jmp_buf_guest[24];   // guest memory: under the identity map any host memory is guest memory
static uint64_t landings[4];
static int nlandings;

static void set_saved_state(Cpu *c) {
    for (int r = 19; r <= 29; r++) c->x[r] = 0x1900 + (uint64_t)r;
    c->sp = SP0;
    for (int r = 8; r <= 15; r++) c->v[r] = ((__uint128_t)0xFFFF << 64) | (0x800 + (uint64_t)r);
}

static void check_saved_state(const Cpu *c) {
    for (int r = 19; r <= 29; r++) CHECK(c->x[r] == 0x1900 + (uint64_t)r);
    CHECK(c->x[30] == G_SETJMP + 0x20);
    CHECK(c->sp == SP0);
    for (int r = 8; r <= 15; r++) CHECK(c->v[r] == 0x800 + (uint64_t)r);   // upper halves cleared, as ldp d8 does
}

static void guest_setjmp(Cpu *c) {
    set_saved_state(c);
    c->x[0] = (uint64_t)(uintptr_t)jmp_buf_guest;
    c->x[30] = G_SETJMP + 0x20;   // what the BL to setjmp writes
    c->x[0] = RT_SETJMP(c, c->x[0]);
    landings[nlandings++] = c->x[0];
    if (c->x[0] != 0) check_saved_state(c);
    if (c->x[0] == 0 || c->x[0] == 7) {
        // Change every callee-saved register, then go down two frames that longjmp: with 7 the first time, 0 next.
        for (int r = 19; r <= 30; r++) c->x[r] = 0xBAD;
        c->sp = SP0 - 0x1000;
        for (int r = 8; r <= 15; r++) c->v[r] = 0xBAD;
        c->x[1] = c->x[0] == 0 ? 7 : 0;
        c->x[30] = G_SETJMP + 0x40;
        c->pc = G_DEEP_G;
        rt_dispatch(c);
        CHECK(!"longjmp returned");
    }
    RT_JMP_LEAVE(c);
}

// Callee-saved trust in the frames between: x19 in a host local, which the longjmp abandons.
static void guest_deep_g(Cpu *c) {
    uint64_t x19 = c->x[19];
    c->x[19] = 0x6666;
    c->sp -= 0x200;
    c->x[0] = (uint64_t)(uintptr_t)jmp_buf_guest;
    c->x[30] = G_DEEP_G + 0x10;
    c->pc = G_DEEP_H;
    rt_dispatch(c);
    c->x[19] = x19;
}

static void guest_deep_h(Cpu *c) {
    c->x[20] = 0x7777;
    c->x[30] = G_DEEP_H + 0x10;
    c->pc = a_longjmp;   // the import, bound to the host override of "longjmp"
    rt_dispatch(c);
}

static void guest_setjmp_returns(Cpu *c) {
    c->x[0] = (uint64_t)(uintptr_t)jmp_buf_guest;
    c->x[30] = G_SETJMP_RETURNS + 0x10;
    c->x[0] = RT_SETJMP(c, c->x[0]);
    RT_JMP_LEAVE(c);
}

static const uint64_t main_fn_addrs[] = {G_ADD3, G_CALLER, G_LOOP, G_SETJMP, G_DEEP_G, G_DEEP_H, G_SETJMP_RETURNS};
static const GuestFn main_fn_ptrs[] = {guest_add3, guest_caller, guest_loop, guest_setjmp,
                                       guest_deep_g, guest_deep_h, guest_setjmp_returns};
static const uint64_t sdk_fn_addrs[] = {G_SDK_DOUBLE};
static const GuestFn sdk_fn_ptrs[] = {guest_sdk_double};
static const RtModule modules[] = {
    {"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, main_fn_addrs, main_fn_ptrs, sizeof main_fn_addrs / sizeof main_fn_addrs[0],
     TEST_END},
    {"sdk", MOD_SDK_BASE, MOD_SDK_SIZE, sdk_fn_addrs, sdk_fn_ptrs, 1, TEST_END},
};

// ---- host functions and SVC handlers

static void host_test_add(Cpu *c) {
    CHECK(c->pc == a_test_add);
    c->x[0] += 100;
}
HOST_OVERRIDE("nnTestAdd", host_test_add);

static uint64_t slept_ns;
static void svc_sleep_thread(Cpu *c, uint64_t pc) {
    (void)pc;
    slept_ns = c->x[0];
    c->x[0] = 0;
}

static void safepoint_hook(Cpu *c) {
    safepoints++;
    c->pending = 0;
}

static void setup(void) {
    a_longjmp = rt_hostfn_bind("longjmp");
    a_setjmp = rt_hostfn_bind("setjmp");
    a_throw = rt_hostfn_bind("__cxa_throw");
    a_test_add = rt_hostfn_bind("nnTestAdd");
    a_missing = rt_hostfn_missing("_ZN2nn2os12NotThereYetEv");
    rt_svc_register(0x0B, svc_sleep_thread);
    rt_set_safepoint_hook(safepoint_hook);
    rt_dispatch_init(modules, 2);
    cpu.excl_addr = ~0ull;
}

// ---- cases

static void case_dispatch(void) {
    Cpu *c = &cpu;
    c->x[0] = 5;
    c->x[1] = G_ADD3;
    c->x[19] = 40;
    rt_call(c, G_CALLER);
    CHECK(c->x[0] == 16);
    CHECK(c->x[30] == G_CALLER + 8);   // rt_call wrote RT_RET_HOST; the guest's own call wrote this
    c->x[0] = 21;
    rt_call(c, G_SDK_DOUBLE);
    CHECK(c->x[0] == 42);
    CHECK(rt_dispatch_lookup(G_ADD3) == guest_add3);
    CHECK(rt_dispatch_lookup(G_SDK_DOUBLE) == guest_sdk_double);
    CHECK(rt_dispatch_lookup(G_ADD3 + 4) == NULL);
    CHECK(rt_dispatch_lookup(G_ADD3 + 1) == NULL);
    CHECK(rt_dispatch_lookup(0x6000000000ull) == NULL);
    RtDispatchStats s = rt_dispatch_stats();
    CHECK(s.functions == 8);
    CHECK(s.lo == G_ADD3 && s.hi == G_SDK_DOUBLE + 4);
    // One million indirect branches: without tail calls this overflows the host stack.
    c->x[2] = 1000000;
    c->x[3] = 0;
    rt_call(c, G_LOOP);
    CHECK(c->x[2] == 0 && c->x[3] == 1000000);
    // A safe-point request is served at the next dispatch, once.
    c->pending = 1;
    c->x[0] = 1;
    rt_call(c, G_ADD3);
    CHECK(safepoints == 1 && c->pending == 0 && c->x[0] == 4);
    char name[96];
    CHECK(strstr(rt_guest_name(G_ADD3, name, sizeof name), "(main+0x100)") != NULL);
    CHECK(strstr(rt_guest_name(G_SDK_DOUBLE, name, sizeof name), "(sdk+0x40)") != NULL);
}

static void case_hostfn(void) {
    Cpu *c = &cpu;
    CHECK(a_test_add - RT_HOSTFN_BASE < RT_HOSTFN_SIZE && (a_test_add & 15) == 0);
    CHECK(rt_sealed());
    c->x[0] = 1;
    c->x[1] = a_test_add;
    c->x[19] = 0;
    rt_call(c, G_CALLER);   // the guest calls the host function through an indirect call
    CHECK(c->x[0] == 202);
    CHECK(rt_hostfn_calls(a_test_add) == 1);
    CHECK(strcmp(rt_hostfn_name(a_test_add), "nnTestAdd") == 0);
    CHECK(rt_hostfn_name(a_test_add + 16 * 64) == NULL);
    CHECK(rt_hostfn_override_count() >= 8);   // jmp.c's seven and this file's one
    CHECK(rt_dispatch_lookup(a_test_add) != NULL);
    char name[96];
    CHECK(strstr(rt_guest_name(a_test_add, name, sizeof name), "(host nnTestAdd)") != NULL);
}

static void case_svc(void) {
    Cpu *c = &cpu;
    c->x[0] = 123456;
    rt_svc(c, 0x0B, G_ADD3);
    CHECK(c->x[0] == 0 && slept_ns == 123456);
    CHECK(rt_svc_calls(0x0B) == 1);
    CHECK(strcmp(rt_svc_name(0x01), "SetHeapSize") == 0);
    CHECK(strcmp(rt_svc_name(0x29), "GetInfo") == 0);
    CHECK(rt_svc_name(0x38) == NULL && rt_svc_name(0x100) == NULL);
}

static void case_setjmp(void) {
    Cpu *c = &cpu;
    rt_call(c, G_SETJMP);
    CHECK(nlandings == 3);
    CHECK(landings[0] == 0 && landings[1] == 7 && landings[2] == 1);
    CHECK(rt_hostfn_calls(a_longjmp) == 2);
    for (int i = 0; i < RT_JMP_SLOTS; i++) CHECK(c->jmp[i].env == 0);   // RT_JMP_LEAVE freed the record
}

static void case_exclusive(void) {
    Cpu *c = &cpu;
    static uint64_t word = 5;
    static __uint128_t pair __attribute__((aligned(16)));
    uint64_t a = (uint64_t)(uintptr_t)&word;
    CHECK(ldx64(c, a) == 5);
    CHECK(stx64(c, a, 6) == 0 && word == 6);
    CHECK(stx64(c, a, 7) == 1 && word == 6);   // the monitor was cleared by the first store
    ldx64(c, a);
    word = 9;   // another writer
    CHECK(stx64(c, a, 10) == 1 && word == 9);
    uint64_t p = (uint64_t)(uintptr_t)&pair;
    ldxp128(c, p);
    CHECK(stxp128(c, p, 1, 2) == 0 && pair == (((__uint128_t)2 << 64) | 1));
}

static void case_ticks(void) {
    CHECK(rt_tick_hz() == 31250000u);
    uint64_t t0 = rt_ticks(), t1 = t0;
    for (long i = 0; i < 2000000000L && t1 - t0 < 31250; i++) t1 = rt_ticks();   // one millisecond
    CHECK(t1 - t0 >= 31250);
}

// ---- cases that end in a fatal error (the driver checks the exit code and the FATAL line)

static void case_svc_missing(void) { rt_svc(&cpu, 0x29, G_ADD3); }
static void case_missing_import(void) { rt_call(&cpu, a_missing); }
static void case_dispatch_not_start(void) { rt_call(&cpu, G_ADD3 + 4); }
static void case_dispatch_outside(void) { rt_call(&cpu, 0x6000000000ull); }
static void case_setjmp_indirect(void) { rt_call(&cpu, a_setjmp); }
static void case_brk(void) { rt_brk(&cpu, 1, G_ADD3); }

static void case_longjmp_stale(void) {
    Cpu *c = &cpu;
    rt_call(c, G_SETJMP_RETURNS);   // setjmp, then the function returns
    c->x[0] = (uint64_t)(uintptr_t)jmp_buf_guest;
    c->x[1] = 1;
    rt_call(c, G_DEEP_H);
}

static void case_throw(void) {
    static const char type_name[] = "N4test9TestErrorE";
    static uint64_t type_info[2];
    type_info[1] = (uint64_t)(uintptr_t)type_name;
    cpu.x[0] = 0x1234;
    cpu.x[1] = (uint64_t)(uintptr_t)type_info;
    cpu.x[29] = 0;
    rt_call(&cpu, a_throw);
}

static void case_late_register(void) { rt_hostfn_register("late", guest_add3); }
static void case_late_svc(void) { rt_svc_register(0x01, svc_sleep_thread); }

// ---- refusals of rt_dispatch_init (no setup)

static void refuse(const RtModule *m, int n) {
    rt_dispatch_init(m, n);
    CHECK(!"rt_dispatch_init accepted a malformed table");
}

static void case_refuse_unsorted(void) {
    static const uint64_t a[] = {G_CALLER, G_ADD3};
    static const GuestFn f[] = {guest_caller, guest_add3};
    RtModule m = {"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, a, f, 2, TEST_END};
    refuse(&m, 1);
}

static void case_refuse_outside(void) {
    static const uint64_t a[] = {MOD_MAIN_BASE + MOD_MAIN_SIZE};
    static const GuestFn f[] = {guest_add3};
    RtModule m = {"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, a, f, 1, TEST_END};
    refuse(&m, 1);
}

static void case_refuse_overlap(void) {
    RtModule m[2] = {modules[0], modules[1]};
    m[1].base = MOD_MAIN_BASE + MOD_MAIN_SIZE - 0x1000;
    refuse(m, 2);
}

static void case_refuse_limit(void) {
    static const uint64_t a[] = {RT_MODULE_LIMIT - 0x1000};
    static const GuestFn f[] = {guest_add3};
    RtModule m = {"main", RT_MODULE_LIMIT - 0x1000, 0x2000, a, f, 1, TEST_END};
    refuse(&m, 1);
}

static void case_refuse_host_in_module(void) {
    rt_register_host(MOD_MAIN_BASE + 0x800, guest_add3);
    refuse(modules, 2);
}

typedef struct { const char *name; void (*run)(void); int needs_setup; } Case;
static const Case cases[] = {
    {"dispatch", case_dispatch, 1},
    {"hostfn", case_hostfn, 1},
    {"svc", case_svc, 1},
    {"setjmp", case_setjmp, 1},
    {"exclusive", case_exclusive, 1},
    {"ticks", case_ticks, 1},
    {"svc_missing", case_svc_missing, 1},
    {"missing_import", case_missing_import, 1},
    {"dispatch_not_start", case_dispatch_not_start, 1},
    {"dispatch_outside", case_dispatch_outside, 1},
    {"setjmp_indirect", case_setjmp_indirect, 1},
    {"brk", case_brk, 1},
    {"longjmp_stale", case_longjmp_stale, 1},
    {"throw", case_throw, 1},
    {"late_register", case_late_register, 1},
    {"late_svc", case_late_svc, 1},
    {"refuse_unsorted", case_refuse_unsorted, 0},
    {"refuse_outside", case_refuse_outside, 0},
    {"refuse_overlap", case_refuse_overlap, 0},
    {"refuse_limit", case_refuse_limit, 0},
    {"refuse_host_in_module", case_refuse_host_in_module, 0},
};

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: boundary_test CASE\n");
        return 2;
    }
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        if (strcmp(argv[1], cases[i].name)) continue;
        if (cases[i].needs_setup) setup();
        cases[i].run();
        printf("ok %s\n", cases[i].name);
        return 0;
    }
    fprintf(stderr, "boundary_test: no case %s\n", argv[1]);
    return 2;
}
