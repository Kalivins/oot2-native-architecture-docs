// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The page table's builder (runtime/gpu/raw_table.c) on its own, for tests/test_raw_table.py, which walks what it
// writes with the translator's own raw-access helpers.
//
//   raw_table_test RUNS TABLE
//
// RUNS is a text file, one run a line: lo hi address guest kind, the first four in hex, kind a BACKING_* number. The
// program lays the runs out (raw_layout) and, when they fit, writes the table's words to TABLE (little-endian 32-bit
// words) and prints
//   layout <span_base> <span_count> <blocks> <words> <pages> <mirror_pages>
//   mirror <slot> <address> <guest>        (one line per mirrored page, in the builder's order; hex)
// When they do not fit it prints "refused <why>" and writes nothing. Exit 0 either way; 2 for bad input.
#include <inttypes.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>

#include "oot2/runtime.h"
#include "oot2/vk_raw.h"

// The builder's only call into the runtime: out of memory.
void rt_fatal(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fputs("FATAL ", stderr);
    vfprintf(stderr, fmt, ap);
    fputc('\n', stderr);
    va_end(ap);
    exit(70);
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: raw_table_test RUNS TABLE\n");
        return 2;
    }
    FILE *f = fopen(argv[1], "r");
    if (!f) {
        fprintf(stderr, "cannot read %s\n", argv[1]);
        return 2;
    }
    int n = 0, cap = 16;
    BackingRun *runs = malloc(cap * sizeof *runs);
    unsigned long long lo, hi, address, guest;
    int kind;
    while (fscanf(f, "%llx %llx %llx %llx %d", &lo, &hi, &address, &guest, &kind) == 5) {
        if (n == cap) runs = realloc(runs, (cap *= 2) * sizeof *runs);
        if (!runs) return 2;
        runs[n++] = (BackingRun){lo, hi, address, guest, kind};
    }
    fclose(f);

    RawLayout l;
    char why[256];
    if (!raw_layout(runs, n, &l, why, sizeof why)) {
        printf("refused %s\n", why);
        return 0;
    }
    uint32_t *words = malloc((size_t)l.words * 4);
    RawMirrorPage *pages = malloc((size_t)(l.mirror_pages + 1) * sizeof *pages);
    uint64_t *guests = malloc((size_t)(l.mirror_pages + 1) * sizeof *guests);
    if (!words || !pages || !guests) return 2;
    raw_write(runs, n, &l, words, pages, guests);

    FILE *out = fopen(argv[2], "wb");
    if (!out || fwrite(words, 4, (size_t)l.words, out) != l.words || fclose(out)) {
        fprintf(stderr, "cannot write %s\n", argv[2]);
        return 2;
    }
    printf("layout %u %u %u %" PRIu64 " %" PRIu64 " %" PRIu64 "\n", l.span_base, l.span_count, l.blocks, l.words,
           l.pages, l.mirror_pages);
    for (uint64_t i = 0; i < l.mirror_pages; i++)
        printf("mirror %x %" PRIx64 " %" PRIx64 "\n", pages[i].slot, pages[i].address, guests[i]);
    return 0;
}
