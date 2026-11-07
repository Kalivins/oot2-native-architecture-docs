// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Texture storage: size, alignment, level offsets and block dimensions of NVN textures in the console's block-linear
// layout (architecture D19). The game places its textures in its memory pools from these answers, so a wrong answer is
// a wrong layout, not a slower one.
//
// The rules, and where each one comes from:
// - A GOB is 64 bytes by 8 rows (512 bytes). A 2D texture's block height (tile_h, log2 of GOBs) is the power of two
//   picked from its height in blocks times 1.5, truncated to whole GOB rows; a 3D texture keeps 1-GOB blocks and picks
//   its block depth from depth times 1.5. Each level shrinks the block to its own size, levels follow each other with
//   no gap, and a layer of a layered texture is aligned to its block. deko3d source/dk_image.cpp (zlib, notice below),
//   with the truncation NVN's driver applies (DKB Recomp runtime/nvn_tex_probe.c at 9d8fa02: forty answers of the
//   original SDK's CPU queries, which the rounded-up rule missed sixteen times).
// - A packaged layout (SetPackagedTextureLayout) overrides the block height with its byte 0's low nibble and the block
//   depth with its high nibble (nxnative dm/gfx/nvn/nvn_tex.c at 865cc35, found against the original driver).
// - MINIMAL_LAYOUT keeps the automatic blocks only when the layer is no larger than with 1-GOB blocks (nxnative, from a
//   sweep of the original driver; Tears of the Kingdom's 320x320 RGB5A1 textures take 204,800 bytes, not 245,760).
// - LINEAR and LINEAR_RENDER_TARGET store rows of `stride` bytes, by default the row rounded up to 32 and 128 bytes,
//   and count texel rows even for compressed formats; a buffer texture is its elements back to back.
// - A sparse texture is laid out in 64 KiB tiles: its first levels padded to whole tiles in width, the standard tile
//   shape's block height when it has no packaged layout, and its storage whole tiles, 64 KiB aligned (nxnative, from the
//   pre-swizzled data that games ship: Yoshi's Crafted World, Monster Hunter Rise, Metroid Prime Remastered).
// The Switch 2 changes none of these (nvn_tex.h).
//
// Ported from nxnative dm/gfx/nvn/nvn_tex.c at 865cc35 (MIT, Copyright (c) 2026 seedds; licenses/nxnative-MIT.txt),
// which oot-pc-recomp at 8e09aaa and DKB Recomp at 9d8fa02 carry in an older form (runtime/nvn_tex.c, from
// MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_tex.c at b75be8cf), all of them based on deko3d. Altered
// here: the texture-buffer row helper is gone (a back end's), and so is the run-time switch of the minimal layout.
// The layout arithmetic is unchanged.
//
// deko3d's notice, as its licence asks:
//
// Copyright (C) 2018-2020 fincs
//
// This software is provided 'as-is', without any express or implied
// warranty.  In no event will the authors be held liable for any
// damages arising from the use of this software.
//
// Permission is granted to anyone to use this software for any
// purpose, including commercial applications, and to alter it and
// redistribute it freely, subject to the following restrictions:
//
// 1. The origin of this software must not be misrepresented; you
//    must not claim that you wrote the original software. If you use
//    this software in a product, an acknowledgment in the product
//    documentation would be appreciated but is not required.
// 2. Altered source versions must be plainly marked as such, and
//    must not be misrepresented as being the original software.
// 3. This notice may not be removed or altered from any source
//    distribution.
#include "oot2/nvn_tex.h"

typedef struct { uint8_t bpb, bw, bh; } fmt_t;   // bytes per block, block width and height in texels

// Every NVNformat of NVN's enumeration, 0x01 to 0x9C (nvn.h).
static fmt_t fmt_of(int f) {
    static const uint8_t astc[14][2] = {{4, 4},  {5, 4},  {5, 5},  {6, 5},   {6, 6},   {8, 5},   {8, 6},
                                        {8, 8},  {10, 5}, {10, 6}, {10, 8},  {10, 10}, {12, 10}, {12, 12}};
    if (f >= 0x79 && f <= 0x94) {   // ASTC, then ASTC sRGB
        int k = (f - 0x79) % 14;
        return (fmt_t){16, astc[k][0], astc[k][1]};
    }
    if (f >= 0x41 && f <= 0x50) {   // DXT, RGTC, BPTC: 4x4 blocks of 8 or 16 bytes
        int b8 = f <= 0x42 || f == 0x45 || f == 0x46 || f == 0x49 || f == 0x4A;
        return (fmt_t){b8 ? 8 : 16, 4, 4};
    }
    static const struct { uint8_t lo, hi, bpb; } r[] = {
        {0x01, 0x04, 1},  {0x05, 0x09, 2}, {0x0A, 0x0C, 4}, {0x0D, 0x10, 2}, {0x11, 0x15, 4},  {0x16, 0x18, 8},
        {0x19, 0x1C, 3},  {0x1D, 0x21, 6}, {0x22, 0x24, 12}, {0x25, 0x28, 4}, {0x29, 0x2D, 8},  {0x2E, 0x30, 16},
        {0x31, 0x31, 1},  {0x32, 0x32, 2}, {0x33, 0x35, 4}, {0x36, 0x36, 8}, {0x37, 0x38, 4},  {0x39, 0x3C, 2},
        {0x3D, 0x40, 4},  {0x51, 0x52, 1}, {0x53, 0x54, 2}, {0x55, 0x56, 4}, {0x57, 0x58, 2},  {0x59, 0x5A, 4},
        {0x5B, 0x5C, 8},  {0x5D, 0x5E, 3}, {0x5F, 0x60, 6}, {0x61, 0x62, 12}, {0x63, 0x64, 4}, {0x65, 0x66, 8},
        {0x67, 0x68, 16}, {0x69, 0x6C, 4}, {0x6D, 0x70, 4}, {0x71, 0x75, 8}, {0x76, 0x78, 16}, {0x95, 0x98, 2},
        {0x99, 0x9C, 4},
    };
    for (unsigned i = 0; i < sizeof r / sizeof *r; i++)
        if (f >= r[i].lo && f <= r[i].hi) return (fmt_t){r[i].bpb, 1, 1};
    return (fmt_t){0, 1, 1};
}

int nvn_fmt_bpb(int f, int *bw, int *bh) {
    fmt_t t = fmt_of(f);
    if (bw) *bw = t.bw;
    if (bh) *bh = t.bh;
    return t.bpb;
}

static uint32_t mip(uint32_t s, unsigned l) {
    s >>= l;
    return s ? s : 1;
}
static uint32_t blk(uint32_t s, uint32_t b) { return (s + b - 1) / b; }
static unsigned pick_tile(uint32_t gobs) { return gobs >= 16 ? 4 : gobs >= 8 ? 3 : gobs >= 4 ? 2 : gobs >= 2 ? 1 : 0; }
// A level's block: the largest power of two of GOBs (unit rows or slices each) below the level's size.
static unsigned adj_tile(unsigned shift, uint32_t unit, uint32_t dim) {
    if (!shift) return 0;
    uint32_t x = unit << (shift - 1);
    if (x >= dim)
        while (--shift) {
            x >>= 1;
            if (x < dim) break;
        }
    return shift;
}

// The target decides the dimensions per layer (dpl) and whether the texture is layered.
static void shape(const nvn_texdesc *d, unsigned *dpl, int *layered, uint32_t *w, uint32_t *h, uint32_t *z) {
    *w = d->width, *h = d->height, *z = d->depth, *layered = 0;
    switch (d->target) {
    case NVN_TEXTURE_TARGET_1D: *dpl = 1, *h = 1, *z = 1; break;
    case NVN_TEXTURE_TARGET_1D_ARRAY: *dpl = 1, *z = d->height, *h = 1, *layered = 1; break;   // layers in height
    case NVN_TEXTURE_TARGET_3D: *dpl = 3; break;
    case NVN_TEXTURE_TARGET_2D_ARRAY:
    case NVN_TEXTURE_TARGET_2D_MULTISAMPLE_ARRAY: *dpl = 2, *layered = 1; break;
    case NVN_TEXTURE_TARGET_CUBEMAP: *dpl = 2, *z = 6, *layered = 1; break;
    case NVN_TEXTURE_TARGET_CUBEMAP_ARRAY: *dpl = 2, *layered = 1; break;   // depth counts layer-faces, as in OpenGL
    default: *dpl = 2, *z = 1; break;                                       // 2D, multisample, rectangle
    }
    if (*z == 0) *z = 1;
}

static void samples_xy(int s, uint32_t *sx, uint32_t *sy) {
    *sx = *sy = 1;
    if (s == 2) *sx = 2;
    else if (s == 4) *sx = *sy = 2;
    else if (s == 8) *sx = 4, *sy = 2;
}

enum { SPARSE_TILE = 0x10000 };
static int sparse_of(const nvn_texdesc *d) {
    return (d->flags & NVN_TEXTURE_FLAGS_SPARSE) && !nvn_tex_is_linear(d) && d->target != NVN_TEXTURE_TARGET_BUFFER;
}
// The block height of a 2D sparse texture's standard 64 KiB tile (256x256, 256x128, 128x128, 128x64 and 64x64 blocks
// for 1, 2, 4, 8 and 16 bytes per block), and the 3D tile's block height and depth; -1 for other block sizes.
static int sparse_tile_h(unsigned bpb) { return bpb == 1 ? 5 : bpb == 2 || bpb == 4 ? 4 : bpb == 8 || bpb == 16 ? 3 : -1; }
static int sparse_tile_h3(unsigned bpb) { return bpb == 1 || bpb == 2 || bpb == 4 ? 2 : bpb == 8 || bpb == 16 ? 1 : -1; }
static int sparse_tile_d3(unsigned bpb) { return bpb == 1 || bpb == 2 ? 5 : bpb == 4 || bpb == 8 || bpb == 16 ? 4 : -1; }
// The block height that shapes a sparse texture's tiles: the standard one, or its own with a packaged layout.
static unsigned tile_shape_h(const nvn_texdesc *d, unsigned dpl, fmt_t t, unsigned tile_h) {
    return dpl == 2 && !d->pkg_block_h && sparse_tile_h(t.bpb) >= 0 ? (unsigned)sparse_tile_h(t.bpb) : tile_h;
}

// The offset of level `level` from the start of its layer, and the GOB columns it is laid out with (*gobs, or NULL).
static uint64_t level_walk(const nvn_texdesc *d, unsigned level, unsigned dpl, uint32_t w, uint32_t h, uint32_t z,
                           unsigned tile_h, unsigned tile_d, fmt_t t, uint32_t *gobs) {
    uint32_t sx, sy;
    samples_xy(d->samples, &sx, &sy);
    unsigned sh = tile_shape_h(d, dpl, t, tile_h);
    uint32_t tw = 128u >> (sh + tile_d);   // GOB columns of a 64 KiB tile
    if (!tw) tw = 1;
    int tiled = sparse_of(d);   // still in a sparse texture's tiled levels
    uint64_t off = 0;
    for (unsigned i = 0;; i++) {
        uint32_t lw = blk(mip(w * sx, i), t.bw);
        uint32_t lh = dpl >= 2 ? blk(mip(h * sy, i), t.bh) : 1;
        uint32_t ld = dpl >= 3 ? mip(z, i) : 1;
        uint32_t lwb = lw * t.bpb;
        unsigned th = adj_tile(tile_h, 8, lh), td = adj_tile(tile_d, 1, ld);
        uint32_t wg = (lwb + 63) / 64, hg = (lh + 7) / 8;
        tiled = tiled && lwb >= tw * 64 && lh >= 8u << sh && ld >= 1u << tile_d;
        if (tiled) wg = (wg + tw - 1) / tw * tw;
        if (i == level) {
            if (gobs) *gobs = wg;
            return off;
        }
        uint32_t ht = (hg + (1u << th) - 1) >> th, dt = (ld + (1u << td) - 1) >> td;
        off += (uint64_t)wg * ht * dt << (9 + th + td);
    }
}

// The bytes of one layer, every level: whole tiles for a sparse texture, a whole block for a layered one.
static uint64_t layer_bytes(const nvn_texdesc *d, unsigned dpl, int layered, uint32_t w, uint32_t h, uint32_t z,
                            unsigned tile_h, unsigned tile_d, fmt_t t) {
    uint64_t ls = level_walk(d, d->levels ? d->levels : 1, dpl, w, h, z, tile_h, tile_d, t, 0);
    if (sparse_of(d)) ls = (ls + SPARSE_TILE - 1) & ~(uint64_t)(SPARSE_TILE - 1);
    if (layered) {
        uint32_t hh = blk(h, t.bh);
        unsigned th = tile_h, td = tile_d;
        while (th && hh <= (8u << (th - 1))) th--;
        while (td && z <= (1u << (td - 1))) td--;
        unsigned sh = th + td + 9;
        uint64_t g = ls >> sh;
        if (ls != g << sh) ls = (g + 1) << sh;
    }
    return ls;
}

void nvn_tex_layout(const nvn_texdesc *d, nvn_texlayout *o) {
    fmt_t t = fmt_of(d->format);
    unsigned dpl;
    int layered;
    uint32_t w, h, z, sx, sy;
    shape(d, &dpl, &layered, &w, &h, &z);
    samples_xy(d->samples, &sx, &sy);
    o->bpb = t.bpb, o->tileH = o->tileD = 0, o->layers = layered ? z : 1;
    if (!t.bpb) {
        o->size = 0, o->align = 512, o->layer_size = 0;
        return;
    }
    if (d->target == NVN_TEXTURE_TARGET_BUFFER) {
        o->layer_size = o->size = (uint64_t)t.bpb * d->width;
        o->align = t.bpb == 12 ? 4 : t.bpb;
        return;
    }
    if (nvn_tex_is_linear(d)) {   // rows of `stride` bytes, counted in texel rows even for compressed formats
        o->align = (d->flags & NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET) ? 128 : 32;
        uint32_t stride = d->stride ? d->stride : (t.bpb * blk(w, t.bw) + o->align - 1) & ~(o->align - 1);
        o->layer_size = o->size = (uint64_t)stride * h;
        return;
    }
    unsigned tile_h = 0, tile_d = 0;
    if (d->pkg_block_h) {   // packaged byte 0: low nibble block height, high nibble block depth
        unsigned pl = (unsigned)(d->pkg_block_h - 1) & 0xff;
        if (dpl >= 2) tile_h = pl & 0xf;
        if (dpl == 3) tile_d = pl >> 4;
    } else if (dpl == 2 && sparse_of(d) && sparse_tile_h(t.bpb) >= 0 &&
               blk(w * sx, t.bw) * t.bpb >= (128u >> sparse_tile_h(t.bpb)) * 64 &&
               blk(h * sy, t.bh) >= 8u << sparse_tile_h(t.bpb)) {
        tile_h = (unsigned)sparse_tile_h(t.bpb);
    } else if (dpl == 3 && sparse_of(d) && sparse_tile_h3(t.bpb) >= 0 &&
               blk(w * sx, t.bw) * t.bpb >= (128u >> (sparse_tile_h3(t.bpb) + sparse_tile_d3(t.bpb))) * 64 &&
               blk(h * sy, t.bh) >= 8u << sparse_tile_h3(t.bpb) && z >= 1u << sparse_tile_d3(t.bpb)) {
        tile_h = (unsigned)sparse_tile_h3(t.bpb), tile_d = (unsigned)sparse_tile_d3(t.bpb);
    } else if (dpl == 2) {
        uint32_t ah = blk(h * sy, t.bh);
        tile_h = pick_tile(ah * 3 / 16);   // height in blocks times 1.5, in whole GOB rows
    } else if (dpl == 3) {
        tile_d = pick_tile(z + z / 2);
    }
    uint64_t ls = layer_bytes(d, dpl, layered, w, h, z, tile_h, tile_d, t);
    if ((d->flags & NVN_TEXTURE_FLAGS_MINIMAL_LAYOUT) && !d->pkg_block_h && !sparse_of(d) && (tile_h || tile_d)) {
        uint64_t one = layer_bytes(d, dpl, layered, w, h, z, 0, 0, t);
        if (one != ls) tile_h = tile_d = 0, ls = one;
    }
    o->tileH = tile_h, o->tileD = tile_d;
    o->layer_size = ls;
    o->size = layered ? ls * z : ls;
    o->align = sparse_of(d) ? SPARSE_TILE : 512;
}

uint64_t nvn_tex_level_offset(const nvn_texdesc *d, unsigned level) {
    fmt_t t = fmt_of(d->format);
    unsigned dpl;
    int layered;
    uint32_t w, h, z;
    shape(d, &dpl, &layered, &w, &h, &z);
    if (!t.bpb || nvn_tex_is_linear(d) || d->target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    return level_walk(d, level, dpl, w, h, z, L.tileH, L.tileD, t, 0);
}

uint32_t nvn_tex_level_gobs(const nvn_texdesc *d, unsigned level) {
    fmt_t t = fmt_of(d->format);
    unsigned dpl;
    int layered;
    uint32_t w, h, z, g = 0;
    shape(d, &dpl, &layered, &w, &h, &z);
    if (!t.bpb || nvn_tex_is_linear(d) || d->target == NVN_TEXTURE_TARGET_BUFFER) return 0;
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    level_walk(d, level, dpl, w, h, z, L.tileH, L.tileD, t, &g);
    return g;
}

int nvn_tex_sparse_tiles(const nvn_texdesc *d, uint32_t *tw, uint32_t *th, uint32_t *td) {
    fmt_t t = fmt_of(d->format);
    unsigned dpl;
    int layered;
    uint32_t w, h, z, sx, sy;
    shape(d, &dpl, &layered, &w, &h, &z);
    *tw = *th = *td = 0;
    if (!sparse_of(d) || !t.bpb) return 0;
    samples_xy(d->samples, &sx, &sy);
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    unsigned sh = tile_shape_h(d, dpl, t, L.tileH);
    uint32_t gw = 128u >> (sh + L.tileD);   // GOB columns of a tile, as level_walk
    if (!gw) gw = 1;
    *tw = gw * 64 / t.bpb * t.bw / sx, *th = dpl >= 2 ? (8u << sh) * t.bh / sy : 1, *td = dpl >= 3 ? 1u << L.tileD : 1;
    unsigned levels = d->levels ? d->levels : 1, n = 0;
    for (; n < levels; n++) {   // level_walk's test: a level at least one tile wide, high and deep
        uint32_t lwb = blk(mip(w * sx, n), t.bw) * t.bpb, lh = dpl >= 2 ? blk(mip(h * sy, n), t.bh) : 1,
                 ld = dpl >= 3 ? mip(z, n) : 1;
        if (lwb < gw * 64 || lh < 8u << sh || ld < 1u << L.tileD) break;
    }
    return (int)n;
}

void nvn_tex_level_tiles(const nvn_texdesc *d, unsigned level, unsigned *tile_h, unsigned *tile_d) {
    fmt_t t = fmt_of(d->format);
    unsigned dpl;
    int layered;
    uint32_t w, h, z, sx, sy;
    shape(d, &dpl, &layered, &w, &h, &z);
    samples_xy(d->samples, &sx, &sy);
    nvn_texlayout L;
    nvn_tex_layout(d, &L);
    uint32_t lh = dpl >= 2 ? blk(mip(h * sy, level), t.bh) : 1;
    uint32_t ld = dpl >= 3 ? mip(z, level) : 1;
    *tile_h = adj_tile(L.tileH, 8, lh);
    *tile_d = adj_tile(L.tileD, 1, ld);
}
