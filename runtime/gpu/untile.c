// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Block-linear storage to linear rows (oot2/untile.h). The region walk (a level of a layer from the layout functions of
// the front end, rows of blocks through 16-byte pieces, the segment tables of the pieces' places in a GOB) follows the
// untiler of nxnative dm/gfx/tex/nvn_deswizzle.c at 865cc35 (MIT, licenses/nxnative-MIT.txt), whose second GOB order is
// the Switch 2's; changed here, the order is a parameter of every call, the geometry comes out as one record the GPU
// pass reads too, the tables are derived from the order's own function, and the shapes the pass does not serve stop by
// name instead of returning an error code.
#include <string.h>

#include "oot2/nvn_tex.h"
#include "oot2/untile.h"

uint32_t untile_gob_offset(int order, uint32_t x, uint32_t y) {
    uint32_t x4 = x >> 4 & 1, x5 = x >> 5 & 1, y0 = y & 1, y1 = y >> 1 & 1, y2 = y >> 2 & 1;
    uint32_t pieces = order == UNTILE_ORDER_S2 ? x4 << 6 | y1 << 5 : y1 << 6 | x4 << 5;   // offset bits 5 and 6
    return x5 << 8 | y2 << 7 | pieces | y0 << 4 | (x & 15);
}

static uint32_t mip(uint32_t s, unsigned l) {
    s >>= l;
    return s ? s : 1;
}

int untile_geom(const nvn_texdesc *d, unsigned level, unsigned layer, UntileGeom *g, const char **why) {
    memset(g, 0, sizeof *g);
    int bw, bh;
    unsigned bpb = (unsigned)nvn_fmt_bpb(d->format, &bw, &bh);
    if (!bpb) {
        *why = "a format no source sizes";
        return 0;
    }
    if (d->target == NVN_TEXTURE_TARGET_BUFFER) {
        *why = "a buffer texture";
        return 0;
    }
    if (nvn_tex_is_linear(d)) {
        *why = "a linear texture, whose rows are not tiled";
        return 0;
    }
    if (d->samples > 1) {
        *why = "a multisampled texture";
        return 0;
    }
    if (d->flags & NVN_TEXTURE_FLAGS_SPARSE) {
        *why = "a sparse texture, whose tiled levels are 64 KiB tiles with padded rows";
        return 0;
    }
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    if (level >= (d->levels ? d->levels : 1)) {
        *why = "a level past the texture's";
        return 0;
    }
    if (layer >= L.layers) {
        *why = "a layer past the texture's";
        return 0;
    }
    int one_d = d->target == NVN_TEXTURE_TARGET_1D || d->target == NVN_TEXTURE_TARGET_1D_ARRAY;
    g->width = mip(d->width, level);
    g->height = one_d ? 1 : mip(d->height, level);
    g->depth = d->target == NVN_TEXTURE_TARGET_3D ? mip(d->depth ? d->depth : 1, level) : 1;
    g->slices = g->depth;
    g->bpb = bpb, g->bw = (unsigned)bw, g->bh = (unsigned)bh;
    g->row_bytes = (g->width + g->bw - 1) / g->bw * g->bpb;
    g->rows = (g->height + g->bh - 1) / g->bh;
    nvn_tex_level_tiles(d, level, &g->tile_h, &g->tile_d);
    g->gobs_x = nvn_tex_level_gobs(d, level);
    g->base = (uint64_t)layer * L.layer_size + nvn_tex_level_offset(d, level);
    uint64_t blocks_y = ((uint64_t)g->rows + (8u << g->tile_h) - 1) >> (3 + g->tile_h);
    uint64_t blocks_z = ((uint64_t)g->slices + (1u << g->tile_d) - 1) >> g->tile_d;
    g->span = (uint64_t)g->gobs_x * blocks_y * blocks_z * (512ull << (g->tile_h + g->tile_d));
    return 1;
}

uint64_t untile_offset(const UntileGeom *g, int order, uint32_t x, uint32_t y, uint32_t z) {
    uint64_t block = 512ull << (g->tile_h + g->tile_d);
    uint64_t blocks_y = ((uint64_t)g->rows + (8u << g->tile_h) - 1) >> (3 + g->tile_h);
    uint64_t gob_in_block = ((uint64_t)(z & ((1u << g->tile_d) - 1)) << g->tile_h) + ((y & ((8u << g->tile_h) - 1)) >> 3);
    return (uint64_t)(z >> g->tile_d) * g->gobs_x * blocks_y * block + (uint64_t)(y >> (3 + g->tile_h)) * g->gobs_x * block +
           (uint64_t)(x >> 6) * block + gob_in_block * 512 + untile_gob_offset(order, x, y);
}

void untile_region(const UntileGeom *g, int order, const uint8_t *src, uint8_t *dst, size_t pitch) {
    uint32_t kox[4], koy[8];   // where the piece at x = 16 * (j & 3) and the row y & 7 start in their GOB
    for (uint32_t j = 0; j < 4; j++) kox[j] = untile_gob_offset(order, j * 16, 0);
    for (uint32_t y = 0; y < 8; y++) koy[y] = untile_gob_offset(order, 0, y);
    uint64_t block = 512ull << (g->tile_h + g->tile_d);
    uint64_t blocks_y = ((uint64_t)g->rows + (8u << g->tile_h) - 1) >> (3 + g->tile_h);
    uint32_t nfull = g->row_bytes >> 4, tail = g->row_bytes & 15;
    for (uint32_t z = 0; z < g->slices; z++) {
        uint64_t zb = (uint64_t)(z >> g->tile_d) * g->gobs_x * blocks_y * block +
                      ((uint64_t)(z & ((1u << g->tile_d) - 1)) << g->tile_h) * 512;
        for (uint32_t y = 0; y < g->rows; y++) {
            uint64_t yb = zb + (uint64_t)(y >> (3 + g->tile_h)) * g->gobs_x * block +
                          (uint64_t)((y & ((8u << g->tile_h) - 1)) >> 3) * 512 + koy[y & 7];
            uint8_t *row = dst + ((size_t)z * g->rows + y) * pitch;
            for (uint32_t j = 0; j < nfull; j++) memcpy(row + ((size_t)j << 4), src + yb + (uint64_t)(j >> 2) * block + kox[j & 3], 16);
            if (tail) memcpy(row + ((size_t)nfull << 4), src + yb + (uint64_t)(nfull >> 2) * block + kox[nfull & 3], tail);
        }
    }
}
