// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Texture storage and handles (architecture D10, D19): what the NVN2 front end (runtime/nvn) answers the game about a
// texture's storage, and what a back end needs to read a registered texture and a shader's handle the way the console
// does.
//
// Storage (runtime/nvn/nvn_tex.c): the console's block-linear layout, whose sizes, alignments and level offsets the
// game uses to place textures in its memory pools. The Switch 2 orders the 16-byte pieces inside a GOB otherwise than
// the Tegra X1 (offset bits 5 and 6 exchanged), which only an untiling pass sees: block heights, sizes and level
// offsets are the same (nxnative docs/reference/nvn-objects.md "The Switch 2's order inside a GOB" at 865cc35; Lichii
// src/video_core/memory_manager.h).
//
// Handles: a combined texture handle holds the texture ID in bits 0 to 19 and the sampler ID in bits 20 to 31, the
// packing the translated shaders unpack from their constant banks (xlate/third_party/ryujinx/src/Ryujinx.Graphics.Shader
// Translation/Optimizations/BindlessElimination.cs, GenerateBindlessAccess: handle & 0xfffff, handle >> 20).
#pragma once
#include <stdint.h>

#include "nvn_obj.h"

// ---- storage
// Fills o for description d. o->bpb is 0 for a format no source sizes, and the rest is then meaningless.
void nvn_tex_layout(const nvn_texdesc *d, nvn_texlayout *o);
// The offset of level `level` from the start of a layer (0 for linear and buffer textures).
uint64_t nvn_tex_level_offset(const nvn_texdesc *d, unsigned level);
// The GOB columns level `level` is laid out with (padded to whole 64 KiB tiles in a sparse texture's tiled levels).
uint32_t nvn_tex_level_gobs(const nvn_texdesc *d, unsigned level);
// Block height and block depth (log2 of GOBs) that level `level` actually uses: they shrink with the level's size.
void nvn_tex_level_tiles(const nvn_texdesc *d, unsigned level, unsigned *tile_h, unsigned *tile_d);
// A sparse texture's tiled levels (those before its mip tail) and one 64 KiB tile in texels; 0 when not sparse.
int nvn_tex_sparse_tiles(const nvn_texdesc *d, uint32_t *tw, uint32_t *th, uint32_t *td);
// Bytes per block of an NVNformat (0 for a format no source sizes), and its block width and height in texels.
int nvn_fmt_bpb(int format, int *bw, int *bh);
// LINEAR and LINEAR_RENDER_TARGET store rows one after another; LINEAR_RENDER_TARGET needs no LINEAR beside it.
static inline int nvn_tex_is_linear(const nvn_texdesc *d) {
    return (d->flags & (NVN_TEXTURE_FLAGS_LINEAR | NVN_TEXTURE_FLAGS_LINEAR_RENDER_TARGET)) != 0;
}

// ---- handles
#define NVN_HANDLE_TEXTURE_BITS 20                                // texture IDs below 2^20 (MAX_TEXTURE_POOL_SIZE)
#define NVN_HANDLE_SAMPLER_BITS 12                                // sampler IDs below 2^12 (MAX_SAMPLER_POOL_SIZE)
static inline uint64_t nvn_handle_pack(uint32_t texture_id, uint32_t sampler_id) {
    return (uint64_t)texture_id | (uint64_t)sampler_id << NVN_HANDLE_TEXTURE_BITS;
}
static inline uint32_t nvn_handle_texture_id(uint64_t h) { return (uint32_t)h & ((1u << NVN_HANDLE_TEXTURE_BITS) - 1); }
static inline uint32_t nvn_handle_sampler_id(uint64_t h) { return (uint32_t)h >> NVN_HANDLE_TEXTURE_BITS; }

// ---- textures (runtime/nvn/nvn_texture.c)
#define NVN_TEX_MAGIC MAGIC('T', 'E', 'X', 0)   // tex_t.magic of an initialised texture
// The byte offset of view v (its base level and first layer) inside texture t's storage; 0 for no view.
uint64_t nvn_tex_view_offset(const tex_t *t, const view_t *v);
