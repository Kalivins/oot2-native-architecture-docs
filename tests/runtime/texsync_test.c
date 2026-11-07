// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Checks of which regions of a CPU-written texture are stale (runtime/gpu/texsync.c, oot2/texsync.h) without a device
// and without guest memory: the write watch stands in as the page generations' source (a mark per page that a query with
// reset clears, as the operating system's does), and the texture's storage is a range of addresses nobody reads. Driven by
// tests/test_runtime_texsync.py: `texsync_test <case>`, one case per process; a passing case prints "ok <case>".
//
//   first     a texture never listed gets every region; those no page of which was ever written are zeros, the others not
//   again     a second use with nothing written lists nothing; a write into one page lists exactly the regions over it (a
//             page shared by two regions lists both), once; a write during the copy (after the query) is listed next time
//   layers    a write into one layer of an array lists that layer's regions only
//   untracked a range the write watch does not cover lists every region at every use, none of them zeros
//   linear    a linear texture is one region, listed by any write into its storage
//   stops     the shapes with no region say so by name
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/nvn_tex.h"
#include "oot2/pagegen.h"
#include "oot2/texsync.h"
#include "oot2/untile.h"
#include "untile_table.h"

#define BASE 0x80000000ull
#define LIMIT (BASE + (256ull << 20))
#define PG 0x1000ull

static uint8_t g_marks[(256 << 20) / 0x1000];
static uint8_t g_touched[(256 << 20) / 0x1000];   // committed: every page the CPU wrote since the start
static int g_untracked;

static uint64_t source(uint64_t addr, uint64_t size, uint64_t *pages, uint64_t max, int reset) {
    if (g_untracked || addr < BASE || addr + size > LIMIT) return UINT64_MAX;
    uint64_t n = 0;
    for (uint64_t a = addr & ~(PG - 1); a < addr + size && n < max; a += PG) {
        uint64_t i = (a - BASE) / PG;
        if (!g_marks[i]) continue;
        pages[n++] = a;
        if (reset) g_marks[i] = 0;
    }
    return n;
}

static void write_at(uint64_t a, uint64_t n) {   // the CPU writes [a, a + n)
    for (uint64_t p = a & ~(PG - 1); p < a + n; p += PG) g_marks[(p - BASE) / PG] = 1, g_touched[(p - BASE) / PG] = 1;
}

// The commit state of the stand-in's pages: the run of pages in one state from addr, up to size.
static uint64_t probe(uint64_t addr, uint64_t size, int *committed) {
    if (addr < BASE || addr >= LIMIT) return UINT64_MAX;
    uint8_t state = g_touched[(addr - BASE) / PG];
    uint64_t n = 0;
    for (uint64_t a = addr; a < addr + size && a < LIMIT; a = (a & ~(PG - 1)) + PG) {
        if (g_touched[(a - BASE) / PG] != state) break;
        n = (a & ~(PG - 1)) + PG - addr;
    }
    *committed = state;
    return n > size ? size : n;
}

// Pages that hold committed memory that the write watch never saw: host I/O into guest pages.
static void host_io_at(uint64_t a, uint64_t n) {
    for (uint64_t p = a & ~(PG - 1); p < a + n; p += PG) g_touched[(p - BASE) / PG] = 1;
}

static void fresh(void) {
    memset(g_marks, 0, sizeof g_marks);
    memset(g_touched, 0, sizeof g_touched);
    g_untracked = 0;
    pagegen_reset();
    pagegen_set_source(source);
    texsync_set_commit_probe(probe);
    texsync_reset_stats();
}

static nvn_texdesc desc(int target, int format, uint32_t w, uint32_t h, uint32_t depth, uint32_t levels, int flags) {
    Spec s = {target, format, flags, w, h, depth, levels, 0};
    return desc_of(&s);
}

static int has(const TexRegion *r, int n, unsigned level, unsigned layer) {
    for (int i = 0; i < n; i++)
        if (r[i].level == level && r[i].layer == layer) return 1;
    return 0;
}

static unsigned regions_over(const nvn_texdesc *d, uint64_t storage, uint64_t lo, uint64_t hi) {
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    unsigned n = 0;
    for (unsigned layer = 0; layer < L.layers; layer++)
        for (unsigned level = 0; level < (d->levels ? d->levels : 1); level++) {
            UntileGeom g;
            const char *why;
            CHECK(untile_geom(d, level, layer, &g, &why));
            uint64_t a = (storage + g.base) & ~(PG - 1), b = (storage + g.base + g.span + PG - 1) & ~(PG - 1);
            n += a < hi && lo < b;
        }
    return n;
}

static void case_first(void) {
    fresh();
    nvn_texdesc d = desc(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 256, 256, 1, 9, 0);
    nvn_texlayout L;
    nvn_tex_layout(&d, &L);
    uint64_t storage = BASE + (1 << 20);
    TexSync st = {0};
    TexRegion r[64];
    const char *why = NULL;
    // too little room for the regions: they are counted, the state is not consumed, and asking again lists them all
    TexSync few = {0};
    CHECK(texsync_regions(&d, storage, &few, r, 2, &why) == 9 && !few.defined && few.gen == 0);
    CHECK(texsync_regions(&d, storage, &few, r, 64, &why) == 9 && few.defined);
    // nothing written: every region, all zeros
    int n = texsync_regions(&d, storage, &st, r, 64, &why);
    CHECK(n == 9 && st.defined);
    for (int i = 0; i < n; i++) CHECK(r[i].zero);
    // another texture, whose level 0 the CPU wrote before: that region is read, the rest are zeros
    TexSync st2 = {0};
    uint64_t at2 = BASE + (8 << 20);
    UntileGeom g0;
    CHECK(untile_geom(&d, 0, 0, &g0, &why));
    write_at(at2 + 100, 8);
    n = texsync_regions(&d, at2, &st2, r, 64, &why);
    CHECK(n == 9);
    uint64_t plo = (at2 + 100) & ~(PG - 1), phi = plo + PG;
    unsigned zeros = 0;
    for (int i = 0; i < n; i++) {
        UntileGeom g;
        CHECK(untile_geom(&d, r[i].level, r[i].layer, &g, &why));
        uint64_t a = (at2 + g.base) & ~(PG - 1), b = (at2 + g.base + g.span + PG - 1) & ~(PG - 1);
        CHECK(r[i].zero == !(a < phi && plo < b));
        zeros += (unsigned)r[i].zero;
    }
    CHECK(zeros < 9 && zeros >= 5 && has(r, n, 0, 0) && !r[0].zero);
    (void)g0;
    // host I/O wrote level 5's storage with no mark: committed counts as written, so that region is read, not zeros
    TexSync st3 = {0};
    uint64_t at3 = BASE + (16 << 20);
    UntileGeom g5;
    CHECK(untile_geom(&d, 5, 0, &g5, &why));
    host_io_at(at3 + g5.base, g5.span);
    n = texsync_regions(&d, at3, &st3, r, 64, &why);
    CHECK(n == 9 && !r[5].zero && r[5].level == 5);
    printf("first: all regions, zeros where nothing was written, a region committed without a mark is read\n");
}

static void case_again(void) {
    fresh();
    nvn_texdesc d = desc(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 256, 256, 1, 9, 0);
    uint64_t storage = BASE + (1 << 20);
    TexSync st = {0};
    TexRegion r[64];
    const char *why = NULL;
    int n = texsync_regions(&d, storage, &st, r, 64, &why);
    CHECK(n == 9);
    // nothing written: nothing stale
    CHECK(texsync_regions(&d, storage, &st, r, 64, &why) == 0);
    // one write
    UntileGeom g3;
    CHECK(untile_geom(&d, 3, 0, &g3, &why));
    uint64_t w = storage + g3.base + 16;
    write_at(w, 16);
    unsigned want = regions_over(&d, storage, w & ~(PG - 1), (w & ~(PG - 1)) + PG);
    n = texsync_regions(&d, storage, &st, r, 64, &why);
    CHECK((unsigned)n == want && has(r, n, 3, 0));
    for (int i = 0; i < n; i++) CHECK(!r[i].zero);
    CHECK(texsync_regions(&d, storage, &st, r, 64, &why) == 0);   // once
    // a write that lands after a query (the CPU writing while the copy runs) is listed by the next
    n = texsync_regions(&d, storage, &st, r, 64, &why);
    CHECK(n == 0);
    write_at(w, 16);
    n = texsync_regions(&d, storage, &st, r, 64, &why);
    CHECK(n >= 1 && has(r, n, 3, 0));
    TexSyncStats ss;
    texsync_stats(&ss);
    CHECK(ss.queries == 6 && ss.untracked == 0);
    printf("again: nothing stale without a write, the regions over a written page with one, once\n");
}

static void case_layers(void) {
    fresh();
    nvn_texdesc d = desc(NVN_TEXTURE_TARGET_2D_ARRAY, NVN_FORMAT_RGBA8, 128, 128, 4, 8, 0);
    nvn_texlayout L;
    nvn_tex_layout(&d, &L);
    uint64_t storage = BASE + (1 << 20);
    TexSync st = {0};
    TexRegion r[64];
    const char *why = NULL;
    CHECK(texsync_regions(&d, storage, &st, r, 64, &why) == 32);
    UntileGeom g;
    CHECK(untile_geom(&d, 0, 2, &g, &why));
    uint64_t w = storage + g.base + g.span / 2;   // the middle of layer 2's first level
    write_at(w, 4);
    int n = texsync_regions(&d, storage, &st, r, 64, &why);
    CHECK(n >= 1);
    for (int i = 0; i < n; i++) CHECK(r[i].layer == 2);
    CHECK(has(r, n, 0, 2));
    printf("layers: a write into layer 2 lists layer 2's regions only\n");
}

static void case_untracked(void) {
    fresh();
    g_untracked = 1;
    nvn_texdesc d = desc(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 64, 1, 4, 0);
    TexSync st = {0};
    TexRegion r[16];
    const char *why = NULL;
    for (int use = 0; use < 3; use++) {
        int n = texsync_regions(&d, BASE + (1 << 20), &st, r, 16, &why);
        CHECK(n == 4);
        for (int i = 0; i < n; i++) CHECK(!r[i].zero);
    }
    TexSyncStats ss;
    texsync_stats(&ss);
    CHECK(ss.untracked == 3);
    printf("untracked: every region at every use, never zeros\n");
}

static void case_linear(void) {
    fresh();
    nvn_texdesc d = desc(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 100, 50, 1, 1, NVN_TEXTURE_FLAGS_LINEAR);
    uint64_t storage = BASE + (1 << 20);
    TexSync st = {0};
    TexRegion r[4];
    const char *why = NULL;
    CHECK(texsync_regions(&d, storage, &st, r, 4, &why) == 1 && r[0].zero);
    CHECK(texsync_regions(&d, storage, &st, r, 4, &why) == 0);
    write_at(storage + 5000, 1);
    int n = texsync_regions(&d, storage, &st, r, 4, &why);
    CHECK(n == 1 && !r[0].zero && r[0].level == 0 && r[0].layer == 0);
    printf("linear: one region\n");
}

static void case_stops(void) {
    fresh();
    TexSync st = {0};
    TexRegion r[4];
    const char *why = NULL;
    nvn_texdesc d = desc(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 64, 1, 1, NVN_TEXTURE_FLAGS_SPARSE);
    CHECK(texsync_regions(&d, BASE, &st, r, 4, &why) < 0 && strstr(why, "sparse"));
    d = desc(NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 64, 64, 1, 1, 0);
    d.samples = 4;
    CHECK(texsync_regions(&d, BASE, &st, r, 4, &why) < 0 && strstr(why, "multisampled"));
    CHECK(!st.defined);
    printf("stops: sparse and multisampled name their reason\n");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: texsync_test <first|again|layers|untracked|linear|stops>\n");
        return 2;
    }
    if (!strcmp(argv[1], "first")) case_first();
    else if (!strcmp(argv[1], "again")) case_again();
    else if (!strcmp(argv[1], "layers")) case_layers();
    else if (!strcmp(argv[1], "untracked")) case_untracked();
    else if (!strcmp(argv[1], "linear")) case_linear();
    else if (!strcmp(argv[1], "stops")) case_stops();
    else {
        fprintf(stderr, "unknown case %s\n", argv[1]);
        return 2;
    }
    printf("ok %s\n", argv[1]);
    return 0;
}
