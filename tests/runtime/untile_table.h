// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The table of texture descriptions and the walk that writes texels into block-linear storage with no formula, shared by
// the checks of the untilers: tests/runtime/untile_test.c (the CPU untiler against the layout functions) and
// tests/runtime/vk_texture_test.c (the GPU untiler and the uploads against the same table).
#pragma once
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/nvn_tex.h"
#include "oot2/untile.h"

#ifndef CHECK
#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)
#endif

// The offset / 16 of the 16-byte piece at row y, byte x = 16 * x16 of a GOB, at index y * 4 + x16.
static const uint8_t k_x1[32] = {0, 2, 16, 18, 1, 3, 17, 19, 4, 6, 20, 22, 5, 7, 21, 23,
                                 8, 10, 24, 26, 9, 11, 25, 27, 12, 14, 28, 30, 13, 15, 29, 31};
static const uint8_t k_s2[32] = {0, 4, 16, 20, 1, 5, 17, 21, 2, 6, 18, 22, 3, 7, 19, 23,
                                 8, 12, 24, 28, 9, 13, 25, 29, 10, 14, 26, 30, 11, 15, 27, 31};

// ---------- the table of descriptions
typedef struct Spec {
    int target, format, flags;
    uint32_t w, h, depth, levels;
    int pkg;   // packaged layout byte 0 + 1 (low nibble block height, high nibble block depth), 0 = automatic
} Spec;

static inline nvn_texdesc desc_of(const Spec *s) {
    nvn_texdesc d;
    memset(&d, 0, sizeof d);
    d.target = s->target, d.format = s->format, d.flags = s->flags, d.samples = 1;
    d.width = s->w, d.height = s->h, d.depth = s->depth, d.levels = s->levels;
    d.pkg_block_h = s->pkg;
    return d;
}

static inline uint32_t chain(uint32_t w, uint32_t h, uint32_t z) {
    uint32_t m = w > h ? w : h;
    if (z > m) m = z;
    uint32_t n = 1;
    while (m > 1) m >>= 1, n++;
    return n;
}

// The formats of the table: NVN format, with its bytes per block (from nvn_fmt_bpb, checked against the name).
static const struct {
    int format;
    uint32_t bpb;
} k_formats[] = {
    {NVN_FORMAT_R8, 1},           {NVN_FORMAT_RG8, 2},          {NVN_FORMAT_R16, 2},          {NVN_FORMAT_RGB565, 2},
    {NVN_FORMAT_RGBA8, 4},        {NVN_FORMAT_RGB10A2, 4},      {NVN_FORMAT_RGBA16F, 8},      {NVN_FORMAT_RGBA32F, 16},
    {NVN_FORMAT_RGBA_DXT1, 8},    {NVN_FORMAT_RGTC1_UNORM, 8},  {NVN_FORMAT_RGBA_DXT5, 16},   {NVN_FORMAT_RGTC2_UNORM, 16},
    {NVN_FORMAT_BPTC_UNORM, 16},  {NVN_FORMAT_RGBA_ASTC_5x5, 16}, {NVN_FORMAT_RGBA_ASTC_8x6, 16}, {NVN_FORMAT_RGBA_ASTC_12x12, 16},
};
#define NFORMATS ((int)(sizeof k_formats / sizeof k_formats[0]))

static const uint32_t k_sizes[][2] = {{1, 1},   {3, 5},    {16, 16},   {64, 8},   {65, 9},  {127, 33},
                                      {256, 256}, {300, 200}, {17, 1},   {1, 70},   {512, 300}, {1024, 64}};
#define NSIZES ((int)(sizeof k_sizes / sizeof k_sizes[0]))

typedef struct Table {
    Spec *s;
    int n, cap;
} Table;

static inline void add(Table *t, Spec s) {
    if (t->n == t->cap) {
        t->cap = t->cap ? t->cap * 2 : 256;
        t->s = realloc(t->s, (size_t)t->cap * sizeof *t->s);
        CHECK(t->s);
    }
    t->s[t->n++] = s;
}

// Every format at every size as a 2D texture with one level and a full chain, automatic block heights; then a seeded
// walk over the other targets, block heights, MINIMAL_LAYOUT and layers.
static inline void build_table(Table *t) {
    for (int f = 0; f < NFORMATS; f++)
        for (int z = 0; z < NSIZES; z++)
            for (int full = 0; full < 2; full++) {
                Spec s = {NVN_TEXTURE_TARGET_2D, k_formats[f].format, 0, k_sizes[z][0], k_sizes[z][1], 1, 1, 0};
                if (full) s.levels = chain(s.w, s.h, 1);
                add(t, s);
            }
    uint64_t rng = 0x9E3779B97F4A7C15ull;
#define NEXT() (rng = rng * 6364136223846793005ull + 1442695040888963407ull, (uint32_t)(rng >> 33))
    static const int targets[] = {NVN_TEXTURE_TARGET_2D, NVN_TEXTURE_TARGET_2D_ARRAY, NVN_TEXTURE_TARGET_3D,
                                  NVN_TEXTURE_TARGET_CUBEMAP, NVN_TEXTURE_TARGET_1D, NVN_TEXTURE_TARGET_1D_ARRAY,
                                  NVN_TEXTURE_TARGET_CUBEMAP_ARRAY, NVN_TEXTURE_TARGET_RECTANGLE};
    for (int i = 0; i < 900; i++) {
        Spec s;
        memset(&s, 0, sizeof s);
        s.target = targets[NEXT() % 8];
        s.format = k_formats[NEXT() % NFORMATS].format;
        s.w = k_sizes[NEXT() % NSIZES][0], s.h = k_sizes[NEXT() % NSIZES][1];
        s.depth = 1;
        if (s.target == NVN_TEXTURE_TARGET_2D_ARRAY) s.depth = 2 + NEXT() % 4;
        if (s.target == NVN_TEXTURE_TARGET_3D) s.depth = 1 + NEXT() % 17;
        if (s.target == NVN_TEXTURE_TARGET_CUBEMAP) s.depth = 6;
        if (s.target == NVN_TEXTURE_TARGET_CUBEMAP_ARRAY) s.depth = 6 * (1 + NEXT() % 2);
        if (s.target == NVN_TEXTURE_TARGET_1D_ARRAY) s.h = 1 + NEXT() % 5;
        if (s.target == NVN_TEXTURE_TARGET_RECTANGLE) s.depth = 1;
        s.levels = NEXT() % 2 ? 1 : chain(s.w, s.target == NVN_TEXTURE_TARGET_1D_ARRAY ? 1 : s.h,
                                           s.target == NVN_TEXTURE_TARGET_3D ? s.depth : 1);
        if (s.target == NVN_TEXTURE_TARGET_RECTANGLE) s.levels = 1;
        s.flags = NEXT() % 3 == 0 ? NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT : 0;
        switch (NEXT() % 4) {
        case 1: s.pkg = 1; break;                                  // one GOB high
        case 2: s.pkg = 3 + (s.target == NVN_TEXTURE_TARGET_3D ? 0x10 : 0); break;   // four high (and two deep)
        case 3: s.pkg = 5 + (s.target == NVN_TEXTURE_TARGET_3D ? 0x20 : 0); break;   // sixteen high (and four deep)
        default: break;
        }
        add(t, s);
    }
#undef NEXT
}

// Storage the region's blocks are written to, walked in storage order, with no formula: for each block of the level
// (x, then y, then z), each GOB of it (one deep layer after the other, each with its GOBs one above the next), the
// 32 pieces of the GOB go to the places the table says. The linear byte (x, y, z) takes `value(...)` and a byte past
// the level's rows, row or slices takes PAD. Returns the bytes walked.
#define PAD 0xEE
static inline uint8_t value(unsigned level, unsigned layer, uint32_t x, uint32_t y, uint32_t z) {
    return (uint8_t)(x * 131u + y * 31u + z * 7u + level * 5u + layer * 3u + (x >> 8) * 17u + (y >> 7) * 11u + 1u);
}

static inline uint64_t walk(const UntileGeom *g, unsigned level, unsigned layer, const uint8_t *piece_of, uint8_t *storage) {
    uint32_t bh = 1u << g->tile_h, bd = 1u << g->tile_d;
    uint32_t blocks_y = (g->rows + 8 * bh - 1) / (8 * bh), blocks_z = (g->slices + bd - 1) / bd;
    uint64_t at = 0;
    for (uint32_t bz = 0; bz < blocks_z; bz++)
        for (uint32_t by = 0; by < blocks_y; by++)
            for (uint32_t bx = 0; bx < g->gobs_x; bx++)
                for (uint32_t gz = 0; gz < bd; gz++)
                    for (uint32_t gy = 0; gy < bh; gy++, at += 512)
                        for (uint32_t row = 0; row < 8; row++)
                            for (uint32_t x16 = 0; x16 < 4; x16++) {
                                uint32_t x = bx * 64 + x16 * 16, y = (by * bh + gy) * 8 + row, z = bz * bd + gz;
                                uint64_t place = at + (uint64_t)piece_of[row * 4 + x16] * 16;
                                if (!storage) continue;
                                for (uint32_t i = 0; i < 16; i++) {
                                    int inside = x + i < g->row_bytes && y < g->rows && z < g->slices;
                                    storage[place + i] = inside ? value(level, layer, x + i, y, z) : PAD;
                                }
                            }
    return at;
}

