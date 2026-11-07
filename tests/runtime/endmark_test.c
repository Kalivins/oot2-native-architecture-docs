// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the end marker of a module's lifted code (the lifted-code end marker), driven by tests/test_runtime_endmark.py:
// `endmark_test <case>`. A passing case prints "ok <case>"; a refusal case must end in one named fatal.
//
// Two synthetic modules stand for lifted code. Their bytes are laid out by hand, in this order, with no padding, so the
// address one byte past a function is known exactly:
//
//   main: em_a_first (1 byte), em_a_mid (2 bytes), em_a_last (3 bytes), then em_a_end, the module's end marker
//   host code that the linker placed after main (2 bytes)
//   sdk:  em_b_first (1 byte), em_b_last (4 bytes), then em_b_end, its end marker
//   host code after sdk (2 bytes)
//
// Three consumers ask whether a host address is lifted code: rt_host_pc_module (the lookup over the lifter's tables),
// kern_pc_lifted (the asynchronous preemption range check) and dispatch_host_pc_guest (the "who read" index of the page
// ranges). Each must say yes for the last byte of the last function of a module and no for the next byte, for the host
// code after the module, and for the host code between the two modules. Nothing here comes from a game.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "kern.h"

__asm__(".text\n"
        ".globl em_a_first\nem_a_first:\n .byte 0xc3\n"
        ".globl em_a_mid\nem_a_mid:\n .byte 0x90,0xc3\n"
        ".globl em_a_last\nem_a_last:\n .byte 0x90,0x90,0xc3\n"
        ".globl em_a_end\nem_a_end:\n .byte 0xcc\n"
        ".globl em_host_a\nem_host_a:\n .byte 0xc3,0xc3\n"
        ".globl em_b_first\nem_b_first:\n .byte 0xc3\n"
        ".globl em_b_last\nem_b_last:\n .byte 0x90,0x90,0x90,0xc3\n"
        ".globl em_b_end\nem_b_end:\n .byte 0xcc\n"
        ".globl em_host_b\nem_host_b:\n .byte 0xc3,0xc3\n");

void em_a_first(Cpu *c);
void em_a_mid(Cpu *c);
void em_a_last(Cpu *c);
void em_a_end(Cpu *c);
void em_host_a(Cpu *c);
void em_b_first(Cpu *c);
void em_b_last(Cpu *c);
void em_b_end(Cpu *c);
void em_host_b(Cpu *c);

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MA MOD_MAIN_BASE
#define MBASE MOD_SDK_BASE
#define A(f) ((uintptr_t)(f))

static const uint64_t a_addrs[] = {MA + 0x100, MA + 0x180, MA + 0x200};
static const GuestFn a_fns[] = {em_a_first, em_a_mid, em_a_last};
static const uint64_t b_addrs[] = {MBASE + 0x100, MBASE + 0x300};
static const GuestFn b_fns[] = {em_b_first, em_b_last};

static void init_with(const void *a_end, const void *b_end) {
    static RtModule mods[2];
    mods[0] = (RtModule){"main", MA, MOD_MAIN_SIZE, a_addrs, a_fns, 3, a_end};
    mods[1] = (RtModule){"sdk", MBASE, MOD_SDK_SIZE, b_addrs, b_fns, 2, b_end};
    rt_dispatch_init(mods, 2);
}

static void init_good(void) { init_with((const void *)em_a_end, (const void *)em_b_end); }

// ---- the three consumers, on the same addresses
static void case_lookup(void) {
    init_good();
    CHECK(rt_host_pc_module(A(em_a_first)) && !strcmp(rt_host_pc_module(A(em_a_first)), "main"));
    CHECK(rt_host_pc_module(A(em_a_last) + 2) && !strcmp(rt_host_pc_module(A(em_a_last) + 2), "main"));   // its last byte
    CHECK(A(em_a_end) == A(em_a_last) + 3);
    CHECK(!rt_host_pc_module(A(em_a_end)));        // one byte past the last function
    CHECK(!rt_host_pc_module(A(em_host_a)));       // host code linked after the module
    CHECK(!rt_host_pc_module(A(em_host_a) + 1));
    CHECK(rt_host_pc_module(A(em_b_first)) && !strcmp(rt_host_pc_module(A(em_b_first)), "sdk"));
    CHECK(rt_host_pc_module(A(em_b_last) + 3) && !strcmp(rt_host_pc_module(A(em_b_last) + 3), "sdk"));
    CHECK(!rt_host_pc_module(A(em_b_end)));
    CHECK(!rt_host_pc_module(A(em_host_b)));
}

static void case_preempt(void) {
    init_good();
    kern_load_code_ranges();
    CHECK(kern_pc_lifted(A(em_a_first)));
    CHECK(kern_pc_lifted(A(em_a_mid) + 1));
    CHECK(kern_pc_lifted(A(em_a_last) + 2));       // the last function's own last byte: guest
    CHECK(!kern_pc_lifted(A(em_a_end)));           // one byte past it: host
    CHECK(!kern_pc_lifted(A(em_host_a)));          // host code between the two modules
    CHECK(!kern_pc_lifted(A(em_host_a) + 1));
    CHECK(kern_pc_lifted(A(em_b_first)));
    CHECK(kern_pc_lifted(A(em_b_last) + 3));
    CHECK(!kern_pc_lifted(A(em_b_end)));
    CHECK(!kern_pc_lifted(A(em_host_b)));
}

static void case_whoread(void) {
    init_good();
    uint64_t g = 0;
    CHECK(dispatch_host_pc_guest(A(em_a_first), &g) && g == MA + 0x100);
    CHECK(dispatch_host_pc_guest(A(em_a_mid) + 1, &g) && g == MA + 0x180);
    CHECK(dispatch_host_pc_guest(A(em_a_last), &g) && g == MA + 0x200);
    CHECK(dispatch_host_pc_guest(A(em_a_last) + 2, &g) && g == MA + 0x200);   // the last function's own last byte
    CHECK(!dispatch_host_pc_guest(A(em_a_end), &g));                          // one byte past it
    CHECK(!dispatch_host_pc_guest(A(em_host_a), &g));
    CHECK(!dispatch_host_pc_guest(A(em_host_a) + 1, &g));                     // between the two modules
    CHECK(dispatch_host_pc_guest(A(em_b_last) + 3, &g) && g == MBASE + 0x300);
    CHECK(!dispatch_host_pc_guest(A(em_b_end), &g));
    CHECK(!dispatch_host_pc_guest(A(em_host_b) + 1, &g));
}

// ---- refusals of rt_dispatch_init
static void case_no_marker(void) { init_with((const void *)em_a_end, NULL); }
static void case_marker_at_last(void) { init_with((const void *)em_a_last, (const void *)em_b_end); }
static void case_marker_below(void) { init_with((const void *)em_a_end, (const void *)em_b_first); }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: endmark_test <case>\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "lookup")) case_lookup();
    else if (!strcmp(cs, "preempt")) case_preempt();
    else if (!strcmp(cs, "whoread")) case_whoread();
    else if (!strcmp(cs, "no_marker")) case_no_marker();
    else if (!strcmp(cs, "marker_at_last")) case_marker_at_last();
    else if (!strcmp(cs, "marker_below")) case_marker_below();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
