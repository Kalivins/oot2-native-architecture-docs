// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Checks of the CPU untiler (runtime/gpu/untile.c, oot2/untile.h) against the layout functions of the NVN2 front end
// (runtime/nvn/nvn_tex.c), without a device. Driven by tests/test_runtime_untile.py: `untile_test <case>`, one case per
// process; a passing case prints "ok <case>".
//
//   order     the two GOB orders: each is a permutation of the 32 pieces of a GOB, they differ by offset bits 5 and 6
//             exchanged, and each equals the literal tables below, which are not computed from the formula: they were
//             read off Lichii's block_linear_unswizzle_2d.comp (src/video_core/host_shaders, GPL-2.0-or-later: its
//             SwizzleTable gives the Tegra X1's order and its native_tiling swap the Switch 2's), run on a ramp
//             (research 65)
//   geometry  a table of formats, sizes, levels, layers and block heights: every region's base and span agree with the
//             layout functions (levels follow each other with no gap and none overlap, a layer ends where the layout
//             says), the storage walked block by block in storage order has as many bytes as the region's span, and the
//             closed-form offset of every byte equals the walk's
//   roundtrip the same table: texels written into storage by that walk (with no formula, the table of pieces and a
//             counter that follows the storage's own order) come out of untile_region as written, in both orders, and the
//             padding of a block never reaches the rows
//   stops     the shapes that have no region say so by name
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/nvn_tex.h"
#include "oot2/untile.h"
#include "untile_table.h"

typedef unsigned long long ull;

static void case_order(void) {
    for (int order = 0; order < 2; order++) {
        const uint8_t *known = order == UNTILE_ORDER_S2 ? k_s2 : k_x1;
        uint8_t seen[32] = {0};
        for (uint32_t y = 0; y < 8; y++)
            for (uint32_t x16 = 0; x16 < 4; x16++) {
                uint32_t o = untile_gob_offset(order, x16 * 16, y);
                CHECK(o % 16 == 0 && o < 512);
                CHECK(o / 16 == known[y * 4 + x16]);
                seen[o / 16]++;
                // the bytes inside a piece keep their order, and the offset repeats every GOB
                CHECK(untile_gob_offset(order, x16 * 16 + 5, y) == o + 5);
                CHECK(untile_gob_offset(order, x16 * 16 + 64, y + 8) == o);
            }
        for (int i = 0; i < 32; i++) CHECK(seen[i] == 1);
    }
    // one order is the other with offset bits 5 and 6 exchanged
    for (uint32_t y = 0; y < 8; y++)
        for (uint32_t x = 0; x < 64; x++) {
            uint32_t a = untile_gob_offset(UNTILE_ORDER_X1, x, y), b = untile_gob_offset(UNTILE_ORDER_S2, x, y);
            uint32_t swapped = (a & ~0x60u) | (a >> 1 & 0x20u) | (a << 1 & 0x40u);
            CHECK(b == swapped);
        }
    CHECK(UNTILE_GOB_ORDER == UNTILE_ORDER_S2);
}

static void case_tables(int roundtrip) {
    Table t = {0};
    build_table(&t);
    uint64_t regions = 0, bytes = 0;
    for (int i = 0; i < t.n; i++) {
        const Spec *s = &t.s[i];
        nvn_texdesc d = desc_of(s);
        nvn_texlayout L;
        nvn_tex_layout(&d, &L);
        int bw, bh;
        CHECK((uint32_t)nvn_fmt_bpb(d.format, &bw, &bh) == L.bpb);
        for (int f = 0; f < NFORMATS; f++)
            if (k_formats[f].format == d.format) CHECK(k_formats[f].bpb == L.bpb);
        // The storage a size of this much could need: skip the table's few large corners of the seeded walk.
        if (L.size > (24ull << 20)) continue;
        uint8_t *storage = L.size ? malloc((size_t)L.size) : NULL;
        CHECK(storage);
        for (unsigned layer = 0; layer < L.layers; layer++) {
            uint64_t expect_base = (uint64_t)layer * L.layer_size;
            for (unsigned level = 0; level < (d.levels ? d.levels : 1); level++) {
                UntileGeom g;
                const char *why = NULL;
                CHECK(untile_geom(&d, level, layer, &g, &why));
                // the region lies where the layout functions say, and the levels follow each other without a gap
                CHECK(g.base == (uint64_t)layer * L.layer_size + nvn_tex_level_offset(&d, level));
                CHECK(g.base == expect_base);
                CHECK(g.span > 0 && g.span % 512 == 0);
                expect_base += g.span;
                CHECK(g.base + g.span <= (uint64_t)(layer + 1) * L.layer_size);
                // the bytes the walk takes, counted with no formula, are the span
                CHECK(walk(&g, level, layer, k_s2, NULL) == g.span);
                if (roundtrip) {
                    uint8_t *pitched = malloc((size_t)(g.rows * g.slices) * (g.row_bytes + 16));
                    CHECK(pitched);
                    for (int order = 0; order < 2; order++) {
                        memset(storage, 0xCD, (size_t)L.size);
                        walk(&g, level, layer, order == UNTILE_ORDER_S2 ? k_s2 : k_x1, storage + g.base);
                        size_t pitch = g.row_bytes + 16;
                        memset(pitched, 0xAB, (size_t)(g.rows * g.slices) * pitch);
                        untile_region(&g, order, storage + g.base, pitched, pitch);
                        for (uint32_t z = 0; z < g.slices; z++)
                            for (uint32_t y = 0; y < g.rows; y++) {
                                const uint8_t *row = pitched + ((size_t)z * g.rows + y) * pitch;
                                for (uint32_t x = 0; x < g.row_bytes; x++) CHECK(row[x] == value(level, layer, x, y, z));
                                for (uint32_t x = g.row_bytes; x < pitch; x++) CHECK(row[x] == 0xAB);   // the pitch's own bytes stay
                            }
                        // the closed-form offset of every byte is where the walk put it
                        for (uint32_t z = 0; z < g.slices; z++)
                            for (uint32_t y = 0; y < g.rows; y++)
                                for (uint32_t x = 0; x < g.row_bytes; x++) {
                                    uint64_t o = untile_offset(&g, order, x, y, z);
                                    CHECK(o < g.span);
                                    CHECK(storage[g.base + o] == value(level, layer, x, y, z));
                                }
                    }
                    free(pitched);
                    regions++, bytes += (uint64_t)g.rows * g.slices * g.row_bytes;
                } else {
                    regions++;
                }
            }
            // the last level ends where the layer's blocks end: what is left is the layer's own padding (a layered texture
            // takes whole blocks)
            CHECK(expect_base <= (uint64_t)(layer + 1) * L.layer_size);
        }
        free(storage);
    }
    fprintf(stderr, "%s: %d descriptions, %llu regions, %llu bytes\n", roundtrip ? "roundtrip" : "geometry", t.n, (ull)regions, (ull)bytes);
    free(t.s);
}

static void case_stops(void) {
    UntileGeom g;
    const char *why = NULL;
    Spec ok = {NVN_TEXTURE_TARGET_2D, NVN_FORMAT_RGBA8, 0, 64, 64, 1, 3, 0};
    nvn_texdesc d = desc_of(&ok);
    CHECK(untile_geom(&d, 2, 0, &g, &why));
    CHECK(!untile_geom(&d, 3, 0, &g, &why) && strstr(why, "level"));
    CHECK(!untile_geom(&d, 0, 1, &g, &why) && strstr(why, "layer"));
    d = desc_of(&ok);
    d.flags = NVN_TEXTURE_FLAGS_LINEAR;
    CHECK(!untile_geom(&d, 0, 0, &g, &why) && strstr(why, "linear"));
    d = desc_of(&ok);
    d.flags = NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET;
    CHECK(!untile_geom(&d, 0, 0, &g, &why) && strstr(why, "linear"));
    d = desc_of(&ok);
    d.flags = NVN_TEXTURE_FLAGS_SPARSE;
    CHECK(!untile_geom(&d, 0, 0, &g, &why) && strstr(why, "sparse"));
    d = desc_of(&ok);
    d.samples = 4;
    CHECK(!untile_geom(&d, 0, 0, &g, &why) && strstr(why, "multisampled"));
    d = desc_of(&ok);
    d.target = NVN_TEXTURE_TARGET_BUFFER;
    CHECK(!untile_geom(&d, 0, 0, &g, &why) && strstr(why, "buffer"));
    d = desc_of(&ok);
    d.format = 0x9D;   // past NVN's enumeration: no source sizes it
    CHECK(!untile_geom(&d, 0, 0, &g, &why) && strstr(why, "no source"));
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: untile_test <order|geometry|roundtrip|stops>\n");
        return 2;
    }
    if (!strcmp(argv[1], "order")) case_order();
    else if (!strcmp(argv[1], "geometry")) case_tables(0);
    else if (!strcmp(argv[1], "roundtrip")) case_tables(1);
    else if (!strcmp(argv[1], "stops")) case_stops();
    else {
        fprintf(stderr, "unknown case %s\n", argv[1]);
        return 2;
    }
    printf("ok %s\n", argv[1]);
    return 0;
}
