// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the census timer (runtime.h, hostfn.c), driven by tests/test_runtime_hostfn_timer.py:
// `hostfn_timer_test <case>`. The host functions below burn thread CPU time (plat_thread_cpu_ns, as kernel_test.c does)
// or sleep, and call each other the way guest code calls a host function (rt_dispatch at a synthetic address), or call
// guest code (rt_call). The census timer must charge each its own time: not a nested host call's, not a guest callback's,
// not the time a thread spends blocked. The driver sets OOT2_CENSUS=1 and OOT2_CENSUS_TIMER=1 for the "timed" case, and
// leaves them out (or sets only the timer) for "default"; nothing here comes from a game.
//
//   A  burns 20 ms of its own, then dispatches B, which burns 10 ms
//   C  rt_call into a guest function that burns 10 ms and dispatches D, which burns 5 ms
//   S  sleeps 30 ms (the thread gives its core up: no CPU time)
//   Z  returns at once; called 20000 times by one host function in the case "nested"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

typedef unsigned long long ull;
#define G_CB (MOD_MAIN_BASE + 0x100)

static Cpu cpu;
static uint64_t a_A, a_B, a_C, a_D, a_S, a_Z;

static void burn_ns(uint64_t ns) {
    uint64_t from = plat_thread_cpu_ns(NULL), cap = plat_now_ns() + 20000000000ull;
    while (plat_thread_cpu_ns(NULL) - from < ns && plat_now_ns() < cap) {
    }
}

static void host_a(Cpu *c) {
    burn_ns(20000000);
    c->pc = a_B;
    rt_dispatch(c);
}
static void host_b(Cpu *c) {
    (void)c;
    burn_ns(10000000);
}
static void host_c(Cpu *c) { rt_call(c, G_CB); }
static void host_d(Cpu *c) {
    (void)c;
    burn_ns(5000000);
}
static void host_s(Cpu *c) {
    (void)c;
    plat_sleep_ns(30000000);
}
static void host_z(Cpu *c) { (void)c; }
HOST_OVERRIDE("timerA", host_a);
HOST_OVERRIDE("timerB", host_b);
HOST_OVERRIDE("timerC", host_c);
HOST_OVERRIDE("timerD", host_d);
HOST_OVERRIDE("timerS", host_s);
HOST_OVERRIDE("timerZ", host_z);

static void guest_cb(Cpu *c) {   // a guest function: burns, then calls the host function D
    burn_ns(10000000);
    c->x[30] = G_CB + 8;
    c->pc = a_D;
    rt_dispatch(c);
}
static const uint64_t main_addrs[] = {G_CB};
static const GuestFn main_fns[] = {guest_cb};
static const RtModule modules[] = {{"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, main_addrs, main_fns, 1, TEST_END}};

static void setup(void) {
    a_A = rt_hostfn_bind("timerA"), a_B = rt_hostfn_bind("timerB"), a_C = rt_hostfn_bind("timerC");
    a_D = rt_hostfn_bind("timerD"), a_S = rt_hostfn_bind("timerS"), a_Z = rt_hostfn_bind("timerZ");
    CHECK(a_A && a_B && a_C && a_D && a_S && a_Z);
    rt_dispatch_init(modules, 1);
    cpu.excl_addr = ~0ull;
}

static void call(uint64_t addr) { rt_call(&cpu, addr); }

#define MAXSLOTS 64
static uint64_t g_calls[MAXSLOTS + 1], g_ns[MAXSLOTS + 1];
static int slot_of(uint64_t a) { return (int)((a - RT_HOSTFN_BASE) >> 4); }
static void read_slots(void) {
    int n = rt_hostfn_count();
    CHECK(n < MAXSLOTS);
    rt_timer_slots(g_calls, g_ns, n);
}
static double ms(uint64_t a) { return (double)g_ns[slot_of(a)] / 1e6; }
static void print_slot(const char *name, uint64_t a) {
    printf("%s calls %llu excl_ms %.3f\n", name, (ull)g_calls[slot_of(a)], ms(a));
}

static void case_timed(void) {
    setup();
    CHECK(rt_timer_on());
    call(a_A);
    call(a_C);
    call(a_S);
    read_slots();
    print_slot("A", a_A);
    print_slot("B", a_B);
    print_slot("C", a_C);
    print_slot("D", a_D);
    print_slot("S", a_S);
    printf("guest calls %llu excl_ms %.3f\n", (ull)g_calls[rt_hostfn_count()], (double)g_ns[rt_hostfn_count()] / 1e6);
    RtTimerThread t[4];
    CHECK(rt_timer_threads(t, 4) == 1);
    printf("classes nvn %llu submit %llu other %llu guest %llu\n", (ull)t[0].calls[RT_TIMER_NVN],
           (ull)t[0].calls[RT_TIMER_SUBMIT], (ull)t[0].calls[RT_TIMER_OTHER], (ull)t[0].calls[RT_TIMER_GUEST]);
    printf("overhead_ns %llu floor_ns %llu overflows %llu\n", (ull)rt_timer_overhead_ns(), (ull)rt_timer_floor_ns(),
           (ull)rt_timer_overflows());
    printf("ok timed\n");
}

// The calibration at work: Z, which does nothing, is called 20000 times from one parent frame that burns nothing either.
// The parent's own time must be far under what the timer itself costs for 20000 children.
static void host_many(Cpu *c) {
    for (int i = 0; i < 20000; i++) {
        c->pc = a_Z;
        rt_dispatch(c);
    }
}
HOST_OVERRIDE("timerMany", host_many);
static void case_nested(void) {
    uint64_t a_many = rt_hostfn_bind("timerMany");
    setup();
    CHECK(rt_timer_on());
    // Five calls, each read on its own: a loaded machine only ever adds to a reading, so the test takes the smallest.
    double last_many = 0, last_z = 0;
    for (int i = 0; i < 5; i++) {
        call(a_many);
        read_slots();
        printf("run %d many_ms %.3f z_ms %.3f\n", i, ms(a_many) - last_many, ms(a_Z) - last_z);
        last_many = ms(a_many), last_z = ms(a_Z);
    }
    print_slot("many", a_many);
    print_slot("Z", a_Z);
    printf("overhead_ns %llu floor_ns %llu\n", (ull)rt_timer_overhead_ns(), (ull)rt_timer_floor_ns());
    printf("ok nested\n");
}

// Not timed: the same program without the flag (or with only the timer variable): no table, no thread, no total.
static void case_default(void) {
    setup();
    CHECK(!rt_timer_on());
    call(a_A);
    call(a_C);
    call(a_S);
    read_slots();
    for (int i = 0; i <= rt_hostfn_count(); i++) CHECK(g_calls[i] == 0 && g_ns[i] == 0);
    RtTimerThread t[4];
    CHECK(rt_timer_threads(t, 4) == 0);
    CHECK(rt_timer_overhead_ns() == 0);
    CHECK(rt_hostfn_calls(a_A) == 1 && rt_hostfn_calls(a_B) == 1);   // the registry's own count is unchanged
    printf("ok default\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: hostfn_timer_test <case>\n");
        return 2;
    }
    if (!strcmp(argv[1], "timed")) case_timed();
    else if (!strcmp(argv[1], "nested")) case_nested();
    else if (!strcmp(argv[1], "default")) case_default();
    else {
        fprintf(stderr, "unknown case %s\n", argv[1]);
        return 2;
    }
    return 0;
}
