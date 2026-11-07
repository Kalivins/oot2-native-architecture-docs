// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Runs lifted synthetic modules for tests/test_lift_diff.py. No game code: the test assembles the modules, lifts
// them with tools/lift/gen.py, and links the generated C with this file and runtime/core.
//
//   driver cases IN OUT   for each case in IN (a function address and an initial state), runs the function and
//                         writes the final state to OUT
//   driver call ADDR X1   runs the function at ADDR with x0 = a 1 KiB buffer and x1 = X1, prints "x0=<hex> sp=<n>"
//
// x20 holds the buffer's address and x21 its middle; both, and sp, are reported as offsets so that the reference
// model, which cannot know host addresses, can compare them.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "oot2/runtime.h"

#define DECL(name, NAME)                         \
    extern const uint64_t name##_fn_addrs[];     \
    extern const GuestFn name##_fn_ptrs[];       \
    extern const uint64_t name##_fn_count;       \
    extern void name##_lifted_end(Cpu *);
MOD_LIST(DECL)
#define ENTRY(name, NAME) {#name, MOD_##NAME##_BASE, MOD_##NAME##_SIZE, name##_fn_addrs, name##_fn_ptrs, 0, (const void *)name##_lifted_end},
static RtModule mods[] = {MOD_LIST(ENTRY)};
#define COUNT(name, NAME) mods[i++].count = name##_fn_count;

#define BUF_SIZE 1024
#define STACK_SIZE 65536
static Cpu cpu;
static uint8_t buf[BUF_SIZE] __attribute__((aligned(64)));
static uint8_t stack[STACK_SIZE] __attribute__((aligned(16)));

static void init(void) {
    int i = 0;
    MOD_LIST(COUNT)
    rt_dispatch_init(mods, MOD_COUNT);
}

static uint64_t stack_top(void) { return (uint64_t)(uintptr_t)(stack + STACK_SIZE); }

static void fresh(Cpu *c) {
    memset(c, 0, sizeof *c);
    c->excl_addr = ~0ull;
    c->sp = stack_top();
}

static int get(FILE *f, void *p, size_t n) { return fread(p, 1, n, f) == n; }

static int run_cases(const char *in_path, const char *out_path) {
    FILE *in = fopen(in_path, "rb"), *out = fopen(out_path, "wb");
    uint32_t n;
    if (!in || !out || !get(in, &n, 4)) {
        fprintf(stderr, "driver: cannot read %s or write %s\n", in_path, out_path);
        return 1;
    }
    for (uint32_t k = 0; k < n; k++) {
        Cpu *c = &cpu;
        uint64_t addr, nzcv;
        fresh(c);
        if (!get(in, &addr, 8) || !get(in, c->x, 8 * 31) || !get(in, &nzcv, 8) || !get(in, c->v, 16 * 8) ||
            !get(in, buf, BUF_SIZE)) {
            fprintf(stderr, "driver: case %u is truncated\n", k);
            return 1;
        }
        c->nzcv = nzcv;
        c->x[20] = (uint64_t)(uintptr_t)buf;
        c->x[21] = (uint64_t)(uintptr_t)buf + 512;
        rt_call(c, addr);
        int64_t sp = (int64_t)(c->sp - stack_top());
        c->x[20] -= (uint64_t)(uintptr_t)buf;
        c->x[21] -= (uint64_t)(uintptr_t)buf;
        if (fwrite(c->x, 8, 31, out) != 31 || fwrite(&sp, 8, 1, out) != 1 || fwrite(c->v, 16, 8, out) != 8 ||
            fwrite(buf, 1, BUF_SIZE, out) != BUF_SIZE) {
            fprintf(stderr, "driver: cannot write %s\n", out_path);
            return 1;
        }
    }
    fclose(in);
    return fclose(out) != 0;
}

int main(int argc, char **argv) {
    if (argc == 4 && !strcmp(argv[1], "cases")) {
        init();
        return run_cases(argv[2], argv[3]);
    }
    if (argc == 4 && !strcmp(argv[1], "call")) {
        init();
        Cpu *c = &cpu;
        fresh(c);
        c->x[0] = (uint64_t)(uintptr_t)buf;
        c->x[1] = strtoull(argv[3], NULL, 0);
        rt_call(c, strtoull(argv[2], NULL, 0));
        printf("x0=%llx sp=%lld\n", (unsigned long long)c->x[0], (long long)(c->sp - stack_top()));
        return 0;
    }
    fprintf(stderr, "usage: driver cases IN OUT | driver call ADDR X1\n");
    return 2;
}
