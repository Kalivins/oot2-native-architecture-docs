// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Tests of the NVN to Vulkan tables and the bank fillers (oot2/vk_nvn.h, runtime/gpu/vk_nvn.c; the Vulkan replay), driven by
// tests/test_runtime_vk_nvn.py: `vk_nvn_test <case>`, one case per process. No device is needed. A passing case prints
// "ok <case>"; a failed check prints its line and exits 1.
//
// The converter cases hold every value the enumeration defines as a row, written with the names of both sides, and probe
// every integer from -8 to 0x200 and the extremes: a row must convert to its Vulkan value, anything else must be refused
// and leave the output as it was. They print the values they accepted, which the driver compares with what nvn.h defines.
//   topology index_type compare stencil_op blend_factor blend_op cull front_face polygon_mode logic_op
//   vertex           one line per NVN format from 0 to 0xA0 (what the table has, what the converter answers, what the front
//                    end sizes the format at), and the exact answers for the formats that are not regular
// The bank cases write every byte they expect by its offset, with the literals the layout is made of, and require every
// other byte to be zero, from a bank that was full of 0xFF before the call.
//   c0_graphics c0_layout c0_compute c2 support
//   layout HEX       the compute layout read from a control section given in hexadecimal, printed (not a check)
#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/nvn.h"
#include "oot2/nvn_program.h"
#include "oot2/nvn_tex.h"
#include "oot2/vk_format.h"
#include "oot2/vk_nvn.h"

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);                                         \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

// ---------- converters

#define PROBE_LO (-8)
#define PROBE_HI 0x200
static const int extremes[] = {INT_MIN, -0x10000, 0x10000, 0x7FFFFFFE, 0x7FFFFFFF, INT_MAX};

// The i-th probed value: -8 to 0x200, then the extremes; 0 when there is none.
static int probe(int i, int *v) {
    if (i <= PROBE_HI - PROBE_LO) return *v = PROBE_LO + i, 1;
    i -= PROBE_HI - PROBE_LO + 1;
    if (i < (int)(sizeof extremes / sizeof *extremes)) return *v = extremes[i], 1;
    return 0;
}

// case_<name>: a converter against its rows. `rows` are {NVN value, Vulkan value}.
#define ENUM_CASE(name, fn, T, ...)                                                                                    \
    typedef struct {                                                                                                   \
        int nvn;                                                                                                       \
        T vk;                                                                                                          \
    } Row_##name;                                                                                                      \
    static const Row_##name rows_##name[] = {__VA_ARGS__};                                                             \
    static void case_##name(void) {                                                                                    \
        int v, accepted = 0;                                                                                           \
        printf("accepts");                                                                                             \
        for (int i = 0; probe(i, &v); i++) {                                                                           \
            const Row_##name *row = NULL;                                                                              \
            for (size_t k = 0; k < sizeof rows_##name / sizeof *rows_##name; k++)                                      \
                if (rows_##name[k].nvn == v) row = &rows_##name[k];                                                    \
            T out;                                                                                                     \
            memset(&out, 0xA5, sizeof out);                                                                            \
            T before = out;                                                                                            \
            int ok = fn(v, &out);                                                                                      \
            if (row) {                                                                                                 \
                CHECK(ok == 1 && out == row->vk);                                                                      \
                accepted++;                                                                                            \
                printf(" 0x%x", (unsigned)v);                                                                          \
            } else {                                                                                                   \
                CHECK(ok == 0 && !memcmp(&out, &before, sizeof out));                                                  \
            }                                                                                                          \
        }                                                                                                              \
        printf("\n");                                                                                                  \
        CHECK(accepted == (int)(sizeof rows_##name / sizeof *rows_##name));                                            \
        printf("ok " #name "\n");                                                                                      \
    }

ENUM_CASE(topology, vkn_topology, VkPrimitiveTopology,
          {NVN_DRAW_PRIMITIVE_POINTS, VK_PRIMITIVE_TOPOLOGY_POINT_LIST},
          {NVN_DRAW_PRIMITIVE_LINES, VK_PRIMITIVE_TOPOLOGY_LINE_LIST},
          {NVN_DRAW_PRIMITIVE_LINE_STRIP, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP},
          {NVN_DRAW_PRIMITIVE_TRIANGLES, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST},
          {NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP},
          {NVN_DRAW_PRIMITIVE_TRIANGLE_FAN, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_FAN},
          {NVN_DRAW_PRIMITIVE_LINES_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_LINE_LIST_WITH_ADJACENCY},
          {NVN_DRAW_PRIMITIVE_LINE_STRIP_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_LINE_STRIP_WITH_ADJACENCY},
          {NVN_DRAW_PRIMITIVE_TRIANGLES_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST_WITH_ADJACENCY},
          {NVN_DRAW_PRIMITIVE_TRIANGLE_STRIP_ADJACENCY, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_STRIP_WITH_ADJACENCY},
          {NVN_DRAW_PRIMITIVE_PATCHES, VK_PRIMITIVE_TOPOLOGY_PATCH_LIST})

// The compare function has one set of rows for four NVN enumerations: each is checked against the same Vulkan value.
ENUM_CASE(compare, vkn_compare, VkCompareOp,
          {NVN_DEPTH_FUNC_NEVER, VK_COMPARE_OP_NEVER},
          {NVN_DEPTH_FUNC_LESS, VK_COMPARE_OP_LESS},
          {NVN_DEPTH_FUNC_EQUAL, VK_COMPARE_OP_EQUAL},
          {NVN_DEPTH_FUNC_LEQUAL, VK_COMPARE_OP_LESS_OR_EQUAL},
          {NVN_DEPTH_FUNC_GREATER, VK_COMPARE_OP_GREATER},
          {NVN_DEPTH_FUNC_NOTEQUAL, VK_COMPARE_OP_NOT_EQUAL},
          {NVN_DEPTH_FUNC_GEQUAL, VK_COMPARE_OP_GREATER_OR_EQUAL},
          {NVN_DEPTH_FUNC_ALWAYS, VK_COMPARE_OP_ALWAYS})

ENUM_CASE(stencil_op, vkn_stencil_op, VkStencilOp,
          {NVN_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP},
          {NVN_STENCIL_OP_ZERO, VK_STENCIL_OP_ZERO},
          {NVN_STENCIL_OP_REPLACE, VK_STENCIL_OP_REPLACE},
          {NVN_STENCIL_OP_INCR, VK_STENCIL_OP_INCREMENT_AND_CLAMP},
          {NVN_STENCIL_OP_DECR, VK_STENCIL_OP_DECREMENT_AND_CLAMP},
          {NVN_STENCIL_OP_INVERT, VK_STENCIL_OP_INVERT},
          {NVN_STENCIL_OP_INCR_WRAP, VK_STENCIL_OP_INCREMENT_AND_WRAP},
          {NVN_STENCIL_OP_DECR_WRAP, VK_STENCIL_OP_DECREMENT_AND_WRAP})

ENUM_CASE(blend_factor, vkn_blend_factor, VkBlendFactor,
          {NVN_BLEND_FUNC_ZERO, VK_BLEND_FACTOR_ZERO},
          {NVN_BLEND_FUNC_ONE, VK_BLEND_FACTOR_ONE},
          {NVN_BLEND_FUNC_SRC_COLOR, VK_BLEND_FACTOR_SRC_COLOR},
          {NVN_BLEND_FUNC_ONE_MINUS_SRC_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC_COLOR},
          {NVN_BLEND_FUNC_SRC_ALPHA, VK_BLEND_FACTOR_SRC_ALPHA},
          {NVN_BLEND_FUNC_ONE_MINUS_SRC_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC_ALPHA},
          {NVN_BLEND_FUNC_DST_ALPHA, VK_BLEND_FACTOR_DST_ALPHA},
          {NVN_BLEND_FUNC_ONE_MINUS_DST_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_DST_ALPHA},
          {NVN_BLEND_FUNC_DST_COLOR, VK_BLEND_FACTOR_DST_COLOR},
          {NVN_BLEND_FUNC_ONE_MINUS_DST_COLOR, VK_BLEND_FACTOR_ONE_MINUS_DST_COLOR},
          {NVN_BLEND_FUNC_SRC_ALPHA_SATURATE, VK_BLEND_FACTOR_SRC_ALPHA_SATURATE},
          {NVN_BLEND_FUNC_SRC1_COLOR, VK_BLEND_FACTOR_SRC1_COLOR},
          {NVN_BLEND_FUNC_ONE_MINUS_SRC1_COLOR, VK_BLEND_FACTOR_ONE_MINUS_SRC1_COLOR},
          {NVN_BLEND_FUNC_SRC1_ALPHA, VK_BLEND_FACTOR_SRC1_ALPHA},
          {NVN_BLEND_FUNC_ONE_MINUS_SRC1_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_SRC1_ALPHA},
          {NVN_BLEND_FUNC_CONSTANT_COLOR, VK_BLEND_FACTOR_CONSTANT_COLOR},
          {NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_COLOR, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_COLOR},
          {NVN_BLEND_FUNC_CONSTANT_ALPHA, VK_BLEND_FACTOR_CONSTANT_ALPHA},
          {NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA, VK_BLEND_FACTOR_ONE_MINUS_CONSTANT_ALPHA})

ENUM_CASE(blend_op, vkn_blend_op, VkBlendOp,
          {NVN_BLEND_EQUATION_ADD, VK_BLEND_OP_ADD},
          {NVN_BLEND_EQUATION_SUB, VK_BLEND_OP_SUBTRACT},
          {NVN_BLEND_EQUATION_REVERSE_SUB, VK_BLEND_OP_REVERSE_SUBTRACT},
          {NVN_BLEND_EQUATION_MIN, VK_BLEND_OP_MIN},
          {NVN_BLEND_EQUATION_MAX, VK_BLEND_OP_MAX})

ENUM_CASE(cull, vkn_cull, VkCullModeFlags,
          {NVN_FACE_NONE, VK_CULL_MODE_NONE},
          {NVN_FACE_FRONT, VK_CULL_MODE_FRONT_BIT},
          {NVN_FACE_BACK, VK_CULL_MODE_BACK_BIT},
          {NVN_FACE_FRONT_AND_BACK, VK_CULL_MODE_FRONT_BIT | VK_CULL_MODE_BACK_BIT})

ENUM_CASE(front_face, vkn_front_face, VkFrontFace,
          {NVN_FRONT_FACE_CW, VK_FRONT_FACE_CLOCKWISE},
          {NVN_FRONT_FACE_CCW, VK_FRONT_FACE_COUNTER_CLOCKWISE})

ENUM_CASE(polygon_mode, vkn_polygon_mode, VkPolygonMode,
          {NVN_POLYGON_MODE_POINT, VK_POLYGON_MODE_POINT},
          {NVN_POLYGON_MODE_LINE, VK_POLYGON_MODE_LINE},
          {NVN_POLYGON_MODE_FILL, VK_POLYGON_MODE_FILL})

ENUM_CASE(logic_op, vkn_logic_op, VkLogicOp,
          {NVN_LOGIC_OP_CLEAR, VK_LOGIC_OP_CLEAR},
          {NVN_LOGIC_OP_AND, VK_LOGIC_OP_AND},
          {NVN_LOGIC_OP_AND_REVERSE, VK_LOGIC_OP_AND_REVERSE},
          {NVN_LOGIC_OP_COPY, VK_LOGIC_OP_COPY},
          {NVN_LOGIC_OP_AND_INVERTED, VK_LOGIC_OP_AND_INVERTED},
          {NVN_LOGIC_OP_NOOP, VK_LOGIC_OP_NO_OP},
          {NVN_LOGIC_OP_XOR, VK_LOGIC_OP_XOR},
          {NVN_LOGIC_OP_OR, VK_LOGIC_OP_OR},
          {NVN_LOGIC_OP_NOR, VK_LOGIC_OP_NOR},
          {NVN_LOGIC_OP_EQUIV, VK_LOGIC_OP_EQUIVALENT},
          {NVN_LOGIC_OP_INVERT, VK_LOGIC_OP_INVERT},
          {NVN_LOGIC_OP_OR_REVERSE, VK_LOGIC_OP_OR_REVERSE},
          {NVN_LOGIC_OP_COPY_INVERTED, VK_LOGIC_OP_COPY_INVERTED},
          {NVN_LOGIC_OP_OR_INVERTED, VK_LOGIC_OP_OR_INVERTED},
          {NVN_LOGIC_OP_NAND, VK_LOGIC_OP_NAND},
          {NVN_LOGIC_OP_SET, VK_LOGIC_OP_SET})

// The index type has a second answer, the size, so it is written out.
static void case_index_type(void) {
    static const struct {
        int nvn;
        VkIndexType vk;
        uint32_t bytes;
    } rows[] = {{NVN_INDEX_TYPE_UNSIGNED_BYTE, VK_INDEX_TYPE_UINT8_EXT, 1},
                {NVN_INDEX_TYPE_UNSIGNED_SHORT, VK_INDEX_TYPE_UINT16, 2},
                {NVN_INDEX_TYPE_UNSIGNED_INT, VK_INDEX_TYPE_UINT32, 4}};
    int v, accepted = 0;
    printf("accepts");
    for (int i = 0; probe(i, &v); i++) {
        const typeof(rows[0]) *row = NULL;
        for (size_t k = 0; k < sizeof rows / sizeof *rows; k++)
            if (rows[k].nvn == v) row = &rows[k];
        VkIndexType out;
        memset(&out, 0xA5, sizeof out);
        VkIndexType before = out;
        uint32_t bytes = 0xA5A5A5A5u;
        int ok = vkn_index_type(v, &out, &bytes);
        if (row) {
            CHECK(ok == 1 && out == row->vk && bytes == row->bytes);
            accepted++;
            printf(" 0x%x", (unsigned)v);
            VkIndexType again;
            CHECK(vkn_index_type(v, &again, NULL) == 1 && again == row->vk);   // the size is optional
        } else {
            CHECK(ok == 0 && !memcmp(&out, &before, sizeof out) && bytes == 0xA5A5A5A5u);
        }
    }
    printf("\n");
    CHECK(accepted == 3);
    printf("ok index_type\n");
}

// ---------- vertex formats

// What nvn.h gives a format that is not a regular R, RG, RGB or RGBA of 8, 16 or 32 bits, which the driver checks from
// the names: the answer of the converter, or none.
static void case_vertex(void) {
    VkFormat f;
    uint32_t bytes;
    int type;
    for (int fmt = -1; fmt <= 0xA0; fmt++) {
        const VkfRow *row = vkf_row(fmt);
        int ok = vkn_vertex_format(fmt, &f, &bytes, &type);
        int nvn_bw, nvn_bh, nvn_bpb = fmt > 0 ? nvn_fmt_bpb(fmt, &nvn_bw, &nvn_bh) : 0;
        printf("vfmt %02x ok=%d table=%d vk=%d bytes=%u type=%d nvn_bpb=%d\n", (unsigned)fmt & 0xFFu, ok,
               row ? (int)row->vertex : 0, ok ? (int)f : 0, ok ? bytes : 0, ok ? type : -1, nvn_bpb);
    }
    // The three-component formats keep three components and the size NVN stores.
    CHECK(vkn_vertex_format(NVN_FORMAT_RGB8, &f, &bytes, &type) && f == VK_FORMAT_R8G8B8_UNORM && bytes == 3 &&
          type == VKN_ATTR_FLOAT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RGB16SN, &f, &bytes, &type) && f == VK_FORMAT_R16G16B16_SNORM && bytes == 6 &&
          type == VKN_ATTR_FLOAT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RGB32UI, &f, &bytes, &type) && f == VK_FORMAT_R32G32B32_UINT && bytes == 12 &&
          type == VKN_ATTR_UINT);
    // Integer formats show a shader an integer, the others a float.
    CHECK(vkn_vertex_format(NVN_FORMAT_R8I, &f, &bytes, &type) && f == VK_FORMAT_R8_SINT && bytes == 1 &&
          type == VKN_ATTR_SINT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RG16UI, &f, &bytes, &type) && f == VK_FORMAT_R16G16_UINT && bytes == 4 &&
          type == VKN_ATTR_UINT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RG32I, &f, &bytes, &type) && f == VK_FORMAT_R32G32_SINT && bytes == 8 &&
          type == VKN_ATTR_SINT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RGBA32F, &f, &bytes, &type) && f == VK_FORMAT_R32G32B32A32_SFLOAT &&
          bytes == 16 && type == VKN_ATTR_FLOAT);
    CHECK(vkn_vertex_format(NVN_FORMAT_R16F, &f, &bytes, &type) && f == VK_FORMAT_R16_SFLOAT && bytes == 2 &&
          type == VKN_ATTR_FLOAT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RGBA8, &f, &bytes, &type) && f == VK_FORMAT_R8G8B8A8_UNORM && bytes == 4 &&
          type == VKN_ATTR_FLOAT);
    // The packed formats and BGRA.
    CHECK(vkn_vertex_format(NVN_FORMAT_RGB10A2, &f, &bytes, &type) && f == VK_FORMAT_A2B10G10R10_UNORM_PACK32 &&
          bytes == 4 && type == VKN_ATTR_FLOAT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RGB10A2UI, &f, &bytes, &type) && f == VK_FORMAT_A2B10G10R10_UINT_PACK32 &&
          bytes == 4 && type == VKN_ATTR_UINT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RGB10A2SN, &f, &bytes, &type) && f == VK_FORMAT_A2B10G10R10_SNORM_PACK32 &&
          bytes == 4 && type == VKN_ATTR_FLOAT);
    CHECK(vkn_vertex_format(NVN_FORMAT_RGB10A2I, &f, &bytes, &type) && f == VK_FORMAT_A2B10G10R10_SINT_PACK32 &&
          bytes == 4 && type == VKN_ATTR_SINT);
    CHECK(vkn_vertex_format(NVN_FORMAT_BGRA8, &f, &bytes, &type) && f == VK_FORMAT_B8G8R8A8_UNORM && bytes == 4 &&
          type == VKN_ATTR_FLOAT);
    // The bytes and the type are optional.
    CHECK(vkn_vertex_format(NVN_FORMAT_RGBA8UI, &f, NULL, NULL) && f == VK_FORMAT_R8G8B8A8_UINT);
    // Refused, and the output left alone: no format, sRGB (the table has a format, Vulkan does not read it as a vertex),
    // depth, block-compressed, packed colour, X formats, and integers read as float.
    static const int refused[] = {NVN_FORMAT_NONE,        NVN_FORMAT_RGBA8_SRGB,   NVN_FORMAT_RGBX8_SRGB,
                                  NVN_FORMAT_BGRA8_SRGB,  NVN_FORMAT_BGRX8_SRGB,   NVN_FORMAT_DEPTH24_STENCIL8,
                                  NVN_FORMAT_DEPTH32F,    NVN_FORMAT_STENCIL8,     NVN_FORMAT_RGBA_DXT5,
                                  NVN_FORMAT_BPTC_UNORM,  NVN_FORMAT_RGBA_ASTC_4x4, NVN_FORMAT_RGBA4,
                                  NVN_FORMAT_RGB565,      NVN_FORMAT_RGBX8,        NVN_FORMAT_RGBX32F,
                                  NVN_FORMAT_RGB9E5F,     NVN_FORMAT_R8_UI2F,      NVN_FORMAT_RG16_I2F,
                                  NVN_FORMAT_RGBA32_UI2F, NVN_FORMAT_RGB10A2_UI2F, NVN_FORMAT_BGRX8,
                                  0x9D,                   0x100,                   -1,
                                  INT_MIN,                INT_MAX};
    for (size_t i = 0; i < sizeof refused / sizeof *refused; i++) {
        VkFormat out = (VkFormat)0x5A5A5A5A;
        uint32_t b = 0xA5A5A5A5u;
        int t = 0x5A5A;
        CHECK(vkn_vertex_format(refused[i], &out, &b, &t) == 0);
        CHECK(out == (VkFormat)0x5A5A5A5A && b == 0xA5A5A5A5u && t == 0x5A5A);
    }
    printf("ok vertex\n");
}

// ---------- banks

static uint32_t rd32(const uint8_t *b, unsigned off) {
    uint32_t v;
    memcpy(&v, b + off, 4);
    return v;
}
static uint64_t rd64(const uint8_t *b, unsigned off) {
    uint64_t v;
    memcpy(&v, b + off, 8);
    return v;
}

// A bank and the bytes a case has accounted for.
typedef struct {
    uint8_t b[VKN_BANK_BYTES];
    uint8_t seen[VKN_BANK_BYTES];
} Bank;

static void fill_ff(Bank *k) {
    memset(k->b, 0xFF, sizeof k->b);
    memset(k->seen, 0, sizeof k->seen);
}

#define EXPECT32(k, off, want)                                                                                         \
    do {                                                                                                               \
        if (rd32((k)->b, (off)) != (uint32_t)(want)) {                                                                 \
            fprintf(stderr, "line %d: word at 0x%x is 0x%x, wanted 0x%x\n", __LINE__, (unsigned)(off),                \
                    (unsigned)rd32((k)->b, (off)), (unsigned)(uint32_t)(want));                                        \
            exit(1);                                                                                                   \
        }                                                                                                              \
        memset((k)->seen + (off), 1, 4);                                                                               \
    } while (0)
#define EXPECT64(k, off, want)                                                                                         \
    do {                                                                                                               \
        if (rd64((k)->b, (off)) != (uint64_t)(want)) {                                                                 \
            fprintf(stderr, "line %d: qword at 0x%x is 0x%llx, wanted 0x%llx\n", __LINE__, (unsigned)(off),           \
                    (unsigned long long)rd64((k)->b, (off)), (unsigned long long)(uint64_t)(want));                    \
            exit(1);                                                                                                   \
        }                                                                                                              \
        memset((k)->seen + (off), 1, 8);                                                                               \
    } while (0)

// Every byte not accounted for is zero (n bytes of the buffer are checked).
static void expect_rest_zero(const uint8_t *b, const uint8_t *seen, unsigned n, int line) {
    for (unsigned i = 0; i < n; i++) {
        if (!seen[i] && b[i]) {
            fprintf(stderr, "line %d: byte 0x%x is 0x%02x, wanted 0\n", line, i, b[i]);
            exit(1);
        }
    }
}
#define REST_ZERO(k) expect_rest_zero((k)->b, (k)->seen, VKN_BANK_BYTES, __LINE__)

static void make_rows(VknBuf rows[16], uint64_t addr_base, uint64_t size_base) {
    for (unsigned r = 0; r < 16; r++) rows[r] = (VknBuf){addr_base + 0x1000 * r, size_base + r};
}

// The graphics stages' tables and the two words every graphics stage has.
static void case_c0_graphics(void) {
    static const struct {
        int stage;
        unsigned base;
    } stages[] = {{NVN_STAGE_VS, 0x110}, {NVN_STAGE_PS, 0x510}, {NVN_STAGE_TS, 0x670}, {NVN_STAGE_MS, 0x770}};
    VknBuf rows[16];
    make_rows(rows, 0x0123456789ABC000ull, 0x100);
    rows[3].size = 0x100000001ull;   // past 32 bits: clamped, never wrapped
    rows[4].size = 0xFFFFFFFFull;
    rows[5].size = 0x80000000ull;
    for (size_t s = 0; s < sizeof stages / sizeof *stages; s++) {
        Bank k;
        fill_ff(&k);
        vkn_c0_graphics(k.b, stages[s].stage, rows, 0xCAFE0007u);
        for (unsigned r = 0; r < 16; r++) {
            unsigned at = stages[s].base + 16 * r;
            EXPECT64(&k, at, rows[r].addr);
            uint32_t want = r == 3 ? 0xFFFFFFFFu : (uint32_t)rows[r].size;
            EXPECT32(&k, at + 8, want);
            EXPECT32(&k, at + 12, 0);
        }
        EXPECT32(&k, 0x610, 0x44FFE000u);   // 2047.0f
        EXPECT32(&k, 0x644, 0xCAFE0007u);
        REST_ZERO(&k);
        // Nothing bound: the same bank without the rows.
        fill_ff(&k);
        vkn_c0_graphics(k.b, stages[s].stage, NULL, 5);
        EXPECT32(&k, 0x610, 0x44FFE000u);
        EXPECT32(&k, 0x644, 5);
        REST_ZERO(&k);
    }
    // Stages that are not graphics leave the bank zero, whatever is bound.
    Bank k;
    for (int stage = -1; stage <= NVN_STAGE_KINDS + 1; stage++) {
        if (stage == NVN_STAGE_VS || stage == NVN_STAGE_PS || stage == NVN_STAGE_TS || stage == NVN_STAGE_MS) continue;
        fill_ff(&k);
        vkn_c0_graphics(k.b, stage, rows, 9);
        REST_ZERO(&k);
    }
    CHECK(vkn_c0_ssbo_base(NVN_STAGE_VS) == 0x110 && vkn_c0_ssbo_base(NVN_STAGE_PS) == 0x510 &&
          vkn_c0_ssbo_base(NVN_STAGE_CS) == 0x310 && vkn_c0_ssbo_base(NVN_STAGE_TS) == 0x670 &&
          vkn_c0_ssbo_base(NVN_STAGE_MS) == 0x770 && vkn_c0_ssbo_base(NVN_STAGE_KINDS) == -1 &&
          vkn_c0_ssbo_base(-1) == -1);
    printf("ok c0_graphics\n");
}

// A compute control section: 0xBC bytes of 0xFF, where only the bytes the layout reads are set.
static void make_ctl(uint8_t ctl[0xBC], const uint8_t roots[8], int ubos, int ssbos, int tex, int septex, int sepsamp,
                     int images) {
    memset(ctl, 0xFF, 0xBC);
    memcpy(ctl + 0xA9, roots, 8);
    ctl[0xB1] = (uint8_t)ubos, ctl[0xB2] = (uint8_t)ssbos, ctl[0xB3] = (uint8_t)tex;
    ctl[0xB4] = (uint8_t)septex, ctl[0xB5] = (uint8_t)sepsamp, ctl[0xB6] = (uint8_t)images;
}

// The layout read from a control section given as hexadecimal digits, printed; the driver compares it with split.py's.
static void case_layout(const char *hex) {
    uint8_t ctl[0x100];
    size_t n = strlen(hex) / 2;
    CHECK(n <= sizeof ctl);
    for (size_t i = 0; i < n; i++) {
        unsigned byte;
        CHECK(sscanf(hex + 2 * i, "%2x", &byte) == 1);
        ctl[i] = (uint8_t)byte;
    }
    VknC0Layout l = {{0}, 0, 0, 0, 0, 0, 0};
    int ok = vkn_c0_layout(ctl, (uint32_t)n, &l);
    printf("layout ok=%d roots=%u,%u,%u,%u,%u,%u,%u,%u ubos=%u ssbos=%u textures=%u sep_textures=%u sep_samplers=%u "
           "images=%u\n",
           ok, l.roots[0], l.roots[1], l.roots[2], l.roots[3], l.roots[4], l.roots[5], l.roots[6], l.roots[7], l.ubos,
           l.ssbos, l.textures, l.sep_textures, l.sep_samplers, l.images);
}

static void case_c0_layout(void) {
    uint8_t ctl[0xBC];
    static const uint8_t roots[8] = {10, 11, 12, 13, 14, 15, 16, 17};
    make_ctl(ctl, roots, 3, 4, 5, 6, 7, 8);   // every count different: a count read from the wrong byte shows
    VknC0Layout l;
    memset(&l, 0, sizeof l);
    CHECK(vkn_c0_layout(ctl, 0xBC, &l));
    for (unsigned i = 0; i < 8; i++) CHECK(l.roots[i] == 10 + i);
    CHECK(l.ubos == 3 && l.ssbos == 4 && l.textures == 5 && l.sep_textures == 6 && l.sep_samplers == 7 && l.images == 8);
    // Only the bytes through +0xB6 are read.
    memset(&l, 0, sizeof l);
    CHECK(vkn_c0_layout(ctl, 0xB7, &l) && l.images == 8);
    VknC0Layout keep = l;
    CHECK(!vkn_c0_layout(ctl, 0xB6, &l) && !memcmp(&l, &keep, sizeof l));
    CHECK(!vkn_c0_layout(ctl, 0, &l) && !vkn_c0_layout(NULL, 0xBC, &l));
    // Each limit: the largest value is taken, one more is refused, with every other count small.
    static const struct {
        const char *what;
        unsigned at, max;
    } limits[] = {{"ubos", 0xB1, 16}, {"ssbos", 0xB2, 16}, {"textures", 0xB3, 128}, {"separate textures", 0xB4, 128},
                  {"separate samplers", 0xB5, 128}, {"images", 0xB6, 32}, {"root 0", 0xA9, 64}, {"root 1", 0xAA, 64},
                  {"root 2", 0xAB, 64}, {"root 3", 0xAC, 64}, {"root 4", 0xAD, 64}, {"root 5", 0xAE, 64},
                  {"root 6", 0xAF, 64}, {"root 7", 0xB0, 64}};
    for (size_t i = 0; i < sizeof limits / sizeof *limits; i++) {
        static const uint8_t none[8] = {0};
        make_ctl(ctl, none, 1, 1, 1, 1, 1, 1);
        ctl[limits[i].at] = (uint8_t)limits[i].max;
        if (!vkn_c0_layout(ctl, 0xBC, &l)) {
            fprintf(stderr, "%s of %u was refused\n", limits[i].what, limits[i].max);
            exit(1);
        }
        ctl[limits[i].at] = (uint8_t)(limits[i].max + 1);
        if (vkn_c0_layout(ctl, 0xBC, &l)) {
            fprintf(stderr, "%s of %u was taken\n", limits[i].what, limits[i].max + 1);
            exit(1);
        }
        ctl[limits[i].at] = 255;
        CHECK(!vkn_c0_layout(ctl, 0xBC, &l));
    }
    printf("ok c0_layout\n");
}

static void case_c0_compute(void) {
    uint8_t ctl[0xBC];
    static const uint8_t table0[8] = {8, 0, 0, 0, 0, 0, 0, 0};
    make_ctl(ctl, table0, 3, 2, 5, 1, 1, 2);
    VknBuf ubo[16], ssbo[16];
    make_rows(ubo, 0x1111000000000000ull, 0x40);
    make_rows(ssbo, 0x2222000000000000ull, 0x80);
    const uint32_t groups[3] = {7, 8, 9};
    static uint32_t roots[8][64];
    for (unsigned t = 0; t < 8; t++)
        for (unsigned w = 0; w < 64; w++) roots[t][w] = 0xB0000000u | t << 8 | w;
    Bank k;
    fill_ff(&k);
    CHECK(vkn_c0_compute(k.b, ctl, 0xBC, groups, roots, ubo, ssbo) == 1);
    EXPECT32(&k, 4, 7);
    EXPECT32(&k, 8, 8);
    EXPECT32(&k, 12, 9);
    for (unsigned r = 0; r < 16; r++) {
        EXPECT64(&k, 0x210 + 16 * r, ubo[r].addr);
        EXPECT32(&k, 0x210 + 16 * r + 8, (uint32_t)ubo[r].size);
        EXPECT32(&k, 0x210 + 16 * r + 12, 0);
        EXPECT64(&k, 0x310 + 16 * r, ssbo[r].addr);
        EXPECT32(&k, 0x310 + 16 * r + 8, (uint32_t)ssbo[r].size);
        EXPECT32(&k, 0x310 + 16 * r + 12, 0);
    }
    REST_ZERO(&k);   // the root words are not in the bank, nor is anything at 0 or from 0x10 to 0x20F
    // Nothing bound and no group counts: zeros, and still a success.
    fill_ff(&k);
    CHECK(vkn_c0_compute(k.b, ctl, 0xBC, NULL, roots, NULL, NULL) == 1);
    REST_ZERO(&k);
    // A size past 32 bits is clamped.
    ubo[1].size = 0x1FFFFFFFFull;
    fill_ff(&k);
    CHECK(vkn_c0_compute(k.b, ctl, 0xBC, groups, roots, ubo, NULL) == 1);
    EXPECT32(&k, 0x210 + 16 + 8, 0xFFFFFFFFu);
    // A control section the layout refuses leaves the bank zero.
    fill_ff(&k);
    CHECK(vkn_c0_compute(k.b, ctl, 0xB6, groups, roots, ubo, ssbo) == 0);
    REST_ZERO(&k);
    make_ctl(ctl, table0, 17, 2, 5, 1, 1, 2);
    fill_ff(&k);
    CHECK(vkn_c0_compute(k.b, ctl, 0xBC, groups, roots, ubo, ssbo) == 0);
    REST_ZERO(&k);
    fill_ff(&k);
    CHECK(vkn_c0_compute(k.b, NULL, 0, groups, roots, ubo, ssbo) == 0);
    REST_ZERO(&k);
    printf("ok c0_compute\n");
}

// c[2]: handles of 32 bits at the start of 8-byte rows.
static void case_c2(void) {
    uint64_t tex[41], img[8], septex[128], samp[128];
    for (unsigned r = 0; r < 41; r++) tex[r] = 0x1000 + r;
    for (unsigned r = 0; r < 128; r++) septex[r] = 0x30000 + r;
    for (unsigned r = 0; r < 128; r++) samp[r] = (uint64_t)(0x400 + r) << 20;
    // The combined slots from 32 share their rows with the image slots. Image slots 2 and 6 are bound, so they keep
    // their rows against the combined slots 34 and 38 (the latter is not bound either); the other image slots are not
    // bound, so combined slots 32, 33, 35, 36 and 39 take their rows; slot 37 is not bound over unbound image 5, so its
    // row is zero; slot 40 lies past the image rows.
    for (unsigned r = 0; r < 8; r++) img[r] = r == 2 || r == 6 ? 0x2000 + r : 0;
    tex[37] = 0, tex[38] = 0;
    tex[0] = 0xDEADBEEF00001234ull;   // the high half is not the row's word
    septex[5] = 0x1111222233334444ull, samp[6] = 0x5555666677778888ull;
    Bank k;
    fill_ff(&k);
    vkn_c2_fill(k.b, tex, img, septex, samp);
    for (unsigned r = 0; r < 32; r++) {
        EXPECT32(&k, 0x20 + 8 * r, r == 0 ? 0x1234u : 0x1000u + r);
        EXPECT32(&k, 0x20 + 8 * r + 4, 0);
    }
    static const uint32_t shared[9] = {0x1000 + 32, 0x1000 + 33, 0x2000 + 2,  0x1000 + 35, 0x1000 + 36,
                                       0,           0x2000 + 6,  0x1000 + 39, 0x1000 + 40};
    for (unsigned r = 0; r < 9; r++) {   // rows 0x120 to 0x160: image slots 0 to 7, then combined slot 40 alone
        EXPECT32(&k, 0x120 + 8 * r, shared[r]);
        EXPECT32(&k, 0x120 + 8 * r + 4, 0);
    }
    for (unsigned r = 0; r < 128; r++) {
        EXPECT32(&k, 0x168 + 8 * r, r == 5 ? 0x33334444u : 0x30000u + r);
        EXPECT32(&k, 0x168 + 8 * r + 4, 0);
        EXPECT32(&k, 0x568 + 8 * r, r == 6 ? 0x77778888u : (uint32_t)(samp[r]));
        EXPECT32(&k, 0x568 + 8 * r + 4, 0);
    }
    REST_ZERO(&k);   // 0 to 0x1F, and from 0x968 on
    // Every image bound: the images win every row they share. The image slot 0 holds its handle.
    for (unsigned r = 0; r < 8; r++) img[r] = 0x2000 + r;
    fill_ff(&k);
    vkn_c2_fill(k.b, tex, img, NULL, NULL);
    for (unsigned r = 0; r < 8; r++) EXPECT32(&k, 0x120 + 8 * r, 0x2000 + r);
    EXPECT32(&k, 0x160, 0x1000 + 40);
    // Arrays that are not given are zero rows.
    fill_ff(&k);
    vkn_c2_fill(k.b, NULL, NULL, NULL, NULL);
    REST_ZERO(&k);
    fill_ff(&k);
    vkn_c2_fill(k.b, NULL, img, NULL, NULL);
    for (unsigned r = 0; r < 8; r++) EXPECT32(&k, 0x120 + 8 * r, 0x2000 + r);
    REST_ZERO(&k);
    printf("ok c2\n");
}

static void case_support(void) {
    uint8_t b[VKN_SUPPORT_BYTES], seen[VKN_SUPPORT_BYTES];
    CHECK(VKN_SUPPORT_BYTES == 1392 && VKN_BANK_BYTES == 0xA00);
    memset(b, 0xFF, sizeof b);
    memset(seen, 0, sizeof seen);
    vkn_support_fill(b, 1024.0f, 512.0f);
    const float one = 1.0f;
    const float want[] = {0.001953125f, 0.00390625f, 1.0f, 0.0f,   // viewport_inverse: 2 / w, 2 / h, 1, 0
                          1024.0f,      512.0f,      1.0f, 0.0f};  // viewport_size: w, h, 1, 0
    for (unsigned i = 0; i < 8; i++) {
        float got;
        memcpy(&got, b + 144 + 4 * i, 4);
        if (got != want[i]) {
            fprintf(stderr, "float %u at 0x%x is %g, wanted %g\n", i, 144 + 4 * i, got, want[i]);
            exit(1);
        }
        memset(seen + 144 + 4 * i, 1, 4);
    }
    for (unsigned e = 0; e < 73; e++) {
        for (unsigned c = 0; c < 4; c++) {
            float got;
            memcpy(&got, b + 192 + 16 * e + 4 * c, 4);
            CHECK(got == one);
            memset(seen + 192 + 16 * e + 4 * c, 1, 4);
        }
    }
    expect_rest_zero(b, seen, VKN_SUPPORT_BYTES, __LINE__);   // alpha test, BGRA flags, scale count, feedback fields
    // An axis that is not positive has no inverse; the size is what it was given.
    const float bad[][2] = {{0.0f, 720.0f}, {-1280.0f, 720.0f}, {1280.0f, 0.0f}};
    for (unsigned i = 0; i < 3; i++) {
        memset(b, 0xFF, sizeof b);
        vkn_support_fill(b, bad[i][0], bad[i][1]);
        float inv[2], size[2];
        memcpy(inv, b + 144, 8), memcpy(size, b + 160, 8);
        CHECK(size[0] == bad[i][0] && size[1] == bad[i][1]);
        CHECK(inv[0] == (bad[i][0] > 0 ? 2.0f / bad[i][0] : 0.0f) && inv[1] == (bad[i][1] > 0 ? 2.0f / bad[i][1] : 0.0f));
        CHECK((bad[i][0] > 0) == (inv[0] != 0) && (bad[i][1] > 0) == (inv[1] != 0));
    }
    printf("ok support\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: vk_nvn_test <case>\n");
        return 2;
    }
    static const struct {
        const char *name;
        void (*run)(void);
    } cases[] = {{"topology", case_topology}, {"index_type", case_index_type}, {"compare", case_compare},
                 {"stencil_op", case_stencil_op}, {"blend_factor", case_blend_factor}, {"blend_op", case_blend_op},
                 {"cull", case_cull}, {"front_face", case_front_face}, {"polygon_mode", case_polygon_mode},
                 {"logic_op", case_logic_op}, {"vertex", case_vertex}, {"c0_graphics", case_c0_graphics},
                 {"c0_layout", case_c0_layout}, {"c0_compute", case_c0_compute}, {"c2", case_c2},
                 {"support", case_support}};
    if (!strcmp(argv[1], "layout") && argc >= 3) {
        case_layout(argv[2]);
        return 0;
    }
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++) {
        if (strcmp(argv[1], cases[i].name)) continue;
        cases[i].run();
        return 0;
    }
    fprintf(stderr, "unknown case %s\n", argv[1]);
    return 2;
}
