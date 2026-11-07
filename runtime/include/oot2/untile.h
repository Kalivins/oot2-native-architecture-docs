// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Block-linear storage to linear rows (architecture D19, section 8): the untiling both the compute pass of the Vulkan
// back end (runtime/gpu/shaders/untile.comp) and the checks of it (runtime/gpu/untile.c, no device) follow.
//
// A level of a texture's storage is a grid of blocks of one GOB (64 bytes by 8 rows) wide, 2^tile_h GOBs high and
// 2^tile_d GOBs deep, filled x, then y, then z; the block heights of a level come from the layout functions of the front
// end (oot2/nvn_tex.h: nvn_tex_level_offset, nvn_tex_level_tiles, nvn_tex_level_gobs), never from here. Inside a GOB the
// 16-byte pieces of the 64 x 8 bytes sit in the order of the Tegra X1 on the Switch, and in the order of the Switch 2
// with offset bits 5 and 6 of the byte offset exchanged:
//
//     X1  offset = x5 << 8 | y2 << 7 | y1 << 6 | x4 << 5 | y0 << 4 | x & 15
//     S2  offset = x5 << 8 | y2 << 7 | x4 << 6 | y1 << 5 | y0 << 4 | x & 15      (x, y: byte x and row in the GOB)
//
// The rule is on the byte offset, so it holds whatever the element size: a 16-byte piece holds a whole number of
// texels or of blocks, and moves as one. Measured on the game's own textures (research 65): with the X1's order the
// edges of 2-, 4- and 8-byte textures come out combed, with the Switch 2's they are clean, and Lichii's untiling shader
// agrees with the Switch 2's on 2-, 4-, 8- and 16-byte formats; a 1-byte format follows from the same byte rule (the game
// ships no 1-byte texture).
//
// What this untiles, and what it stops at by name: one level of one layer of a block-linear texture whose format the
// layout functions size, as rows of blocks (a 3D level: every slice, one after another). Sparse textures (tiles of
// 64 KiB, padded rows), multisampled textures, buffer textures and the formats no source sizes have no region here
// (untile_geom returns 0 and says why); linear textures need no untiling (their rows are in the storage as they are).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "oot2/nvn_obj.h"

enum { UNTILE_ORDER_X1 = 0, UNTILE_ORDER_S2 = 1 };
// The order the runtime untiles in. The X1's exists so that the checks can show the two apart: nothing but a check
// asks for it.
#define UNTILE_GOB_ORDER UNTILE_ORDER_S2

// Where byte x of row y lies inside its GOB (0 to 511), in the given order. x and y are taken modulo the GOB.
uint32_t untile_gob_offset(int order, uint32_t x, uint32_t y);

// One level of one layer of a texture's storage, in blocks.
typedef struct UntileGeom {
    uint64_t base;         // from the start of the texture's storage: layer * layer_size + the level's offset
    uint64_t span;         // the bytes the level takes in storage (whole blocks)
    uint32_t width, height, depth;   // the level in texels (depth: the 3D level's slices, 1 for the others)
    uint32_t row_bytes;    // bytes of one row of blocks
    uint32_t rows;         // rows of blocks
    uint32_t slices;       // depth slices (a 3D level's depth; 1 otherwise)
    uint32_t bpb, bw, bh;  // bytes per block, and the block in texels
    uint32_t gobs_x;       // GOB columns of a row as laid out
    uint32_t tile_h, tile_d;   // log2 of the GOBs high and deep of this level's block
} UntileGeom;

// Fill g for `level` of `layer` (0 for a texture that has no layers; a cube map's face, a layered texture's layer).
// 1 on success. 0 when this layer has no region for the texture, with the reason in *why (never NULL on return 0): a
// level or layer past the texture's, or one of the shapes the description above names.
int untile_geom(const nvn_texdesc *d, unsigned level, unsigned layer, UntileGeom *g, const char **why);

// The offset from the start of the level's region of byte x of row y of slice z.
uint64_t untile_offset(const UntileGeom *g, int order, uint32_t x, uint32_t y, uint32_t z);

// Untile the region: src is its g->span bytes, dst receives g->slices * g->rows rows of g->row_bytes bytes, `pitch`
// bytes apart (at least row_bytes; slice k starts at dst + k * rows * pitch).
void untile_region(const UntileGeom *g, int order, const uint8_t *src, uint8_t *dst, size_t pitch);
