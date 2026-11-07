// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// From NVN's values to Vulkan's, and the constant banks a translated module reads (architecture D11, D12; ticket 63):
// what the replay of the recorded commands needs and that has no device state, so that a test checks it without one.
//
// The converters. Each takes the NVN enumeration's value as the record holds it and answers 1 with *out set, or 0 for a
// value that has no Vulkan form, with *out left as it was: the caller names the stop. 0 covers two cases that the callers
// treat alike: a value NVN does not define, and one NVN defines but Vulkan cannot say (the draw primitives LINE_LOOP,
// QUADS, QUAD_STRIP and POLYGON, which need an index rewrite or a changed vertex count; the 32-bit integer vertex
// formats read as float, which Vulkan has no format for). A value is never mapped to a near neighbour.
//
// The banks. A translated module reads the driver's constant banks c[0] and c[2] and the support buffer; the replay
// writes each as one buffer of VKN_BANK_BYTES (VKN_SUPPORT_BYTES for the support buffer) into the frame's scratch memory.
// The fillers write the whole buffer: what they do not name is zero.
//   c[0] graphics    the stage's storage table, rows of {u64 address, u32 size, u32 zero} every 16 bytes: vertex 0x110,
//                    pixel 0x510, task 0x670, mesh 0x770; 2047.0f at 0x610 (the largest point size, to which a shader's
//                    point size is clamped); the draw's base instance at 0x644. The pixel stage's sample position rows
//                    (0x620 and 0x630) stay zero: only a multisampled pass reads them.
//   c[0] compute     NVN2's compute shaders lay out c[0] per shader (the control section counts its parts), but the
//                    translator (xlate/third_party/ryujinx Instructions/InstEmitAmpere.cs, AMapC0) moves every part to the
//                    place the host fills here: the root table words to bank 24 (the push constants, which the replay
//                    writes from OP_PUSH_CONST), uniform slot r to c[0] + 0x210 + 0x10 r, storage slot s to
//                    c[0] + 0x310 + 0x10 s, the texture, separate texture, separate sampler and image handles to c[2]. What
//                    it leaves in c[0] is the 16-byte head: the group counts x, y, z at byte 4. So the bank holds the
//                    group counts at 4, the uniform rows at 0x210 and the storage rows at 0x310, and nothing else.
//   c[2]             one 32-bit handle (texture ID in bits 0 to 19, sampler ID in bits 20 to 31) at the start of each
//                    8-byte row: combined handles 0x20 + 8 r (41 rows), images 0x120 + 8 r (8 rows), separate textures
//                    0x168 + 8 r and separate samplers 0x568 + 8 r (128 rows each). Combined rows 32 to 39 lie over the
//                    eight image rows: a bound image keeps its row.
//   support buffer   Ryujinx's SupportBuffer (std140): the viewport's inverse at 144, its size at 160 and the render scale
//                    1.0 in each of the 73 elements from 192; no alpha test, no BGRA target, no transform feedback.
//
// Provenance: see runtime/gpu/vk_nvn.c.
#pragma once
#include <stdint.h>

#include "volk.h"

// ---- converters
// NVNdrawPrimitive. LINE_LOOP, QUADS, QUAD_STRIP and POLYGON have no Vulkan topology.
int vkn_topology(int nvn_prim, VkPrimitiveTopology *out);
// NVNindexType and the index's size in bytes (bytes may be NULL). UNSIGNED_BYTE is VK_INDEX_TYPE_UINT8, which the device
// takes only with VK_EXT_index_type_uint8 enabled: the replay enables it or rewrites the indices.
int vkn_index_type(int nvn_itype, VkIndexType *out, uint32_t *bytes);
// NVNdepthFunc, NVNstencilFunc, NVNcompareFunc and NVNalphaFunc are one enumeration (NEVER 1 to ALWAYS 8; vk_nvn.c
// asserts it), so one function serves all four.
int vkn_compare(int nvn_func, VkCompareOp *out);
int vkn_stencil_op(int nvn_op, VkStencilOp *out);       // NVNstencilOp
int vkn_blend_factor(int nvn_func, VkBlendFactor *out); // NVNblendFunc
int vkn_blend_op(int nvn_eq, VkBlendOp *out);           // NVNblendEquation
int vkn_cull(int nvn_face, VkCullModeFlags *out);       // NVNface: NONE is no culling
// NVNfrontFace as NVN says it. Vulkan's numbers are the other way round; the replay inverts the answer when it flips Y.
int vkn_front_face(int nvn_front, VkFrontFace *out);
int vkn_polygon_mode(int nvn_mode, VkPolygonMode *out); // NVNpolygonMode: POINT, LINE, FILL
int vkn_logic_op(int nvn_op, VkLogicOp *out);           // NVNlogicOp, the raw value of NVNcolorState (COPY by default)

// The scalar type a vertex attribute shows a shader.
enum { VKN_ATTR_FLOAT = 0, VKN_ATTR_SINT = 1, VKN_ATTR_UINT = 2 };
// A vertex attribute's NVNformat: the Vulkan format (the table of oot2/vk_format.h, VkfRow.vertex, with the NVN format's
// own components: RGB8 is R8G8B8, three bytes), the attribute's size in bytes and its scalar type (bytes and type may be
// NULL). 0 for a format that has no vertex use there, and for an sRGB format, which Vulkan does not read as a vertex.
// A device that does not take a three-component format as a vertex format is the caller's to deal with.
int vkn_vertex_format(int nvn_format, VkFormat *out, uint32_t *bytes, int *type);

// ---- banks
#define VKN_BANK_BYTES 0xA00u      // a driver constant bank
#define VKN_SUPPORT_BYTES 1392u    // the support buffer

typedef struct VknBuf {
    uint64_t addr, size;           // a bound buffer's GPU address and bytes; {0, 0} is unbound
} VknBuf;

// The byte offset of a stage's storage table in c[0], or -1 for a stage kind with none. stage_kind is an NVN_STAGE_*
// (oot2/nvn_program.h): VS 0x110, PS 0x510, CS 0x310, TS 0x670, MS 0x770.
int vkn_c0_ssbo_base(int stage_kind);

// c[0] of a graphics stage (NVN_STAGE_VS, PS, TS or MS; any other kind leaves the bank zero). ssbo may be NULL (nothing
// bound). A size past 32 bits is written as 0xFFFFFFFF, never wrapped.
void vkn_c0_graphics(uint8_t bank[VKN_BANK_BYTES], int stage_kind, const VknBuf ssbo[16], uint32_t base_instance);

// What a compute stage's control section says about its c[0] (NVN2: +0xA9 to +0xB0 the word counts of the eight root
// tables, +0xB1 uniform rows, +0xB2 storage rows, +0xB3 combined handles, +0xB4 separate textures, +0xB5 separate
// samplers, +0xB6 images).
typedef struct VknC0Layout {
    uint8_t roots[8];
    uint8_t ubos, ssbos, textures, sep_textures, sep_samplers, images;
} VknC0Layout;
// Reads the layout from ctl (ctl_size bytes, at least 0xB7). 0 when ctl is too short or a count is out of range: more
// than 16 uniform or storage rows (the tables hold 16), more than 128 combined handles, 128 separate textures or 128
// separate samplers (the c[2] tables), more than 32 images, or a root table of more than 64 words. The first four limits
// are DKB Recomp's (runtime/nvn2_c0.h, nvn2_c0_valid); the separate counts follow the size of c[2]'s tables.
int vkn_c0_layout(const uint8_t *ctl, uint32_t ctl_size, VknC0Layout *out);

// c[0] of a compute stage: the group counts at byte 4, the uniform rows at 0x210 and the storage rows at 0x310 (see the
// banks above). Returns 0, with the bank left zero, when vkn_c0_layout refuses the control section. ubo and ssbo may be
// NULL. roots (the root tables' words) is not read: the translator reads them from bank 24, so the call that writes
// bank 24 is OP_PUSH_CONST's, and the parameter stays only so that the call keeps its shape.
int vkn_c0_compute(uint8_t bank[VKN_BANK_BYTES], const uint8_t *ctl, uint32_t ctl_size, const uint32_t groups[3],
                   const uint32_t roots[8][64], const VknBuf ubo[16], const VknBuf ssbo[16]);

// c[2] from a stage's bound handles: tex[r] the combined texture bound at slot r, img[r] the image at slot r, septex[r]
// the separate texture at slot r, samp[r] the separate sampler at slot r. Each handle's low 32 bits are the row's word.
// Any array may be NULL. A combined handle at slot 32 or more is written when it is not zero and the image slot it lies
// over (slot minus 32) holds no image.
void vkn_c2_fill(uint8_t bank[VKN_BANK_BYTES], const uint64_t tex[41], const uint64_t img[8], const uint64_t septex[128],
                 const uint64_t samp[128]);

// The support buffer for a draw whose viewport is width by height (the inverse is 0 for an axis that is not positive).
void vkn_support_fill(uint8_t b[VKN_SUPPORT_BYTES], float width, float height);
