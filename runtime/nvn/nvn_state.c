// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// State objects (architecture D10): blend, color, channel mask, multisample, polygon, depth-stencil, vertex attribute
// and vertex stream states, packed in place in the game's object (nvn_obj.h, each within the size the SDK gives it),
// with their setters and getters. In Kokiri Forest the game sets about 2,169 channel masks and a few hundred of each
// other field per frame (a count of its NVN2 calls), so a setter is a store into the packed object and nothing more:
// no lock, no allocation, no call into the back end, which reads the object when a command buffer binds it.
//
// Defaults are NVN's, which are OpenGL's where both have the field: no blending (ONE, ZERO, ADD) and no advanced mode,
// logic op COPY, alpha test ALWAYS, every channel written, multisampling enabled with no sample count, no culling,
// counter-clockwise front faces, filled polygons, depth test off with depth writes on and LESS, stencil test off with
// ALWAYS and KEEP, and zero vertex formats, offsets, strides and divisors. They are the references' values, not read
// from the game's driver; the game sets every field of these objects it uses, except the alpha test, the advanced
// blend fields and the rarer multisample fields.
//
// Programmable sample locations (MultisampleStateSetSampleLocations, GetSampleLocations, GetSampleLocationsGrid) have no
// body: the game never names them, and their layout and grid have no source here. They stop with their name.
//
// Ported from oot-pc-recomp runtime/nvn_api_obj.c and nvn_api_nvn2.c at 8e09aaa (GPL-3.0-or-later; ported there from
// MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c at b75be8cf, MIT, Copyright (c) 2026 jhaizhou; origin
// d2r-mobile; licenses/dkb-xlate-MIT.txt): the setters. The getters, the advanced blend setters and the rest of the
// multisample state are adapted from nxnative dm/gfx/nvn/hle_nvn.c at 865cc35 (MIT, licenses/nxnative-MIT.txt).
// Changed here: the advanced normalized destination and NVN2's coverage truncation are kept and read back (both were
// dropped), the advanced mode is stored as given, a vertex attribute the packed state cannot hold is a named fatal, and
// the boolean and integer arguments are read at their own width.
#include <string.h>

#include "nvn_api.h"

#define ST(T, c) ((T *)arg_p(c, 0))
#define CST(T, c) ((const T *)arg_p(c, 0))
static uint32_t arg_u32(const Cpu *c, int i) { return (uint32_t)arg_x(c, i); }   // an int or enum argument
static void put_bool(void *p, int v) { if (p) *(NVNboolean *)p = v != 0; }
static void put_u32(void *p, uint32_t v) { if (p) memcpy(p, &v, 4); }               // an enum through a pointer

// ---- blend state
NVN_API(nvnBlendStateSetDefaults) {
    *ST(blend_st, c) = (blend_st){0, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO,
                                  NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD, 0, 0, 0, 0};
}
NVN_API(nvnBlendStateSetBlendTarget) { ST(blend_st, c)->target = arg_u32(c, 1); }
// void nvnBlendStateSetBlendFunc(NVNblendState *s, NVNblendFunc srcRGB, dstRGB, srcAlpha, dstAlpha)
NVN_API(nvnBlendStateSetBlendFunc) {
    blend_st *o = ST(blend_st, c);
    o->src_rgb = arg_u32(c, 1), o->dst_rgb = arg_u32(c, 2), o->src_a = arg_u32(c, 3), o->dst_a = arg_u32(c, 4);
}
// void nvnBlendStateSetBlendEquation(NVNblendState *s, NVNblendEquation rgb, NVNblendEquation alpha)
NVN_API(nvnBlendStateSetBlendEquation) {
    blend_st *o = ST(blend_st, c);
    o->eq_rgb = arg_u32(c, 1), o->eq_a = arg_u32(c, 2);
}
NVN_API(nvnBlendStateSetAdvancedMode) { ST(blend_st, c)->adv = arg_u32(c, 1); }
NVN_API(nvnBlendStateSetAdvancedOverlap) { ST(blend_st, c)->adv_ovl = arg_u32(c, 1); }
NVN_API(nvnBlendStateSetAdvancedPremultipliedSrc) { ST(blend_st, c)->adv_npm = !nvn_arg_bool(c, 1); }
NVN_API(nvnBlendStateSetAdvancedNormalizedDst) { ST(blend_st, c)->adv_nnd = !nvn_arg_bool(c, 1); }
NVN_API(nvnBlendStateGetBlendTarget) { ret_x(c, CST(blend_st, c)->target); }
// void nvnBlendStateGetBlendFunc(const NVNblendState *s, NVNblendFunc *srcRGB, *dstRGB, *srcAlpha, *dstAlpha)
NVN_API(nvnBlendStateGetBlendFunc) {
    const blend_st *o = CST(blend_st, c);
    put_u32(arg_p(c, 1), o->src_rgb), put_u32(arg_p(c, 2), o->dst_rgb);
    put_u32(arg_p(c, 3), o->src_a), put_u32(arg_p(c, 4), o->dst_a);
}
// void nvnBlendStateGetBlendEquation(const NVNblendState *s, NVNblendEquation *rgb, NVNblendEquation *alpha)
NVN_API(nvnBlendStateGetBlendEquation) {
    const blend_st *o = CST(blend_st, c);
    put_u32(arg_p(c, 1), o->eq_rgb), put_u32(arg_p(c, 2), o->eq_a);
}
NVN_API(nvnBlendStateGetAdvancedMode) { ret_x(c, CST(blend_st, c)->adv); }
NVN_API(nvnBlendStateGetAdvancedOverlap) { ret_x(c, CST(blend_st, c)->adv_ovl); }
NVN_API(nvnBlendStateGetAdvancedPremultipliedSrc) { ret_x(c, !CST(blend_st, c)->adv_npm); }
NVN_API(nvnBlendStateGetAdvancedNormalizedDst) { ret_x(c, !CST(blend_st, c)->adv_nnd); }

// ---- color state: blend enable per color target (eight), logic op, alpha test
NVN_API(nvnColorStateSetDefaults) { *ST(color_st, c) = (color_st){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}; }
// void nvnColorStateSetBlendEnable(NVNcolorState *s, int index, NVNboolean enable): an index past the eight targets is
// ignored
NVN_API(nvnColorStateSetBlendEnable) {
    color_st *o = ST(color_st, c);
    uint32_t i = arg_u32(c, 1);
    if (i > 7) return;
    o->blend_en = nvn_arg_bool(c, 2) ? (o->blend_en | 1u << i) : (o->blend_en & ~(1u << i));
}
NVN_API(nvnColorStateSetLogicOp) { ST(color_st, c)->logic_op = arg_u32(c, 1); }
NVN_API(nvnColorStateSetAlphaTest) { ST(color_st, c)->alpha_func = arg_u32(c, 1); }
// NVNboolean nvnColorStateGetBlendEnable(const NVNcolorState *s, int index)
NVN_API(nvnColorStateGetBlendEnable) {
    uint32_t i = arg_u32(c, 1);
    ret_x(c, i <= 7 && (CST(color_st, c)->blend_en >> i & 1));
}
NVN_API(nvnColorStateGetLogicOp) { ret_x(c, CST(color_st, c)->logic_op); }
NVN_API(nvnColorStateGetAlphaTest) { ret_x(c, CST(color_st, c)->alpha_func); }

// ---- channel mask state: four bits per color target, r 1, g 2, b 4, a 8, set = written
NVN_API(nvnChannelMaskStateSetDefaults) { ST(chmask_st, c)->mask = 0xffffffffu; }
// void nvnChannelMaskStateSetChannelMask(NVNchannelMaskState *s, int index, NVNboolean r, g, b, a)
NVN_API(nvnChannelMaskStateSetChannelMask) {
    uint32_t i = arg_u32(c, 1);
    if (i > 7) return;
    uint32_t m = (nvn_arg_bool(c, 2) ? 1u : 0) | (nvn_arg_bool(c, 3) ? 2u : 0) | (nvn_arg_bool(c, 4) ? 4u : 0) |
                 (nvn_arg_bool(c, 5) ? 8u : 0);
    chmask_st *o = ST(chmask_st, c);
    o->mask = (o->mask & ~(0xfu << (4 * i))) | m << (4 * i);
}
// void nvnChannelMaskStateGetChannelMask(const NVNchannelMaskState *s, int index, NVNboolean *r, *g, *b, *a)
NVN_API(nvnChannelMaskStateGetChannelMask) {
    uint32_t i = arg_u32(c, 1), m = i <= 7 ? CST(chmask_st, c)->mask >> (4 * i) & 0xf : 0;
    put_bool(arg_p(c, 2), m & 1), put_bool(arg_p(c, 3), m & 2), put_bool(arg_p(c, 4), m & 4);
    put_bool(arg_p(c, 5), m & 8);
}

// ---- multisample state
NVN_API(nvnMultisampleStateSetDefaults) {
    memset(arg_p(c, 0), 0, sizeof(ms_st));
    ST(ms_st, c)->enable = 1;
}
NVN_API(nvnMultisampleStateSetMultisampleEnable) { ST(ms_st, c)->enable = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnMultisampleStateSetSamples) { ST(ms_st, c)->samples = (uint8_t)arg_u32(c, 1); }
NVN_API(nvnMultisampleStateSetAlphaToCoverageEnable) { ST(ms_st, c)->a2c = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnMultisampleStateSetAlphaToCoverageDither) { ST(ms_st, c)->a2c_dither = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnMultisampleStateSetRasterSamples) { ST(ms_st, c)->raster_samples = (uint8_t)arg_u32(c, 1); }
NVN_API(nvnMultisampleStateSetCoverageModulationMode) { ST(ms_st, c)->cov_mod = (uint8_t)arg_u32(c, 1); }
NVN_API(nvnMultisampleStateSetCoverageTruncationEnable) { ST(ms_st, c)->cov_trunc = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnMultisampleStateSetCoverageToColorEnable) { ST(ms_st, c)->cov2color_en = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnMultisampleStateSetCoverageToColorOutput) { ST(ms_st, c)->cov2color_out = (uint8_t)arg_u32(c, 1); }
NVN_API(nvnMultisampleStateSetSampleLocationsEnable) { ST(ms_st, c)->sl_en = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnMultisampleStateSetSampleLocationsGridEnable) { ST(ms_st, c)->sl_grid = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnMultisampleStateGetMultisampleEnable) { ret_x(c, CST(ms_st, c)->enable); }
NVN_API(nvnMultisampleStateGetSamples) { ret_x(c, CST(ms_st, c)->samples); }
NVN_API(nvnMultisampleStateGetAlphaToCoverageEnable) { ret_x(c, CST(ms_st, c)->a2c); }
NVN_API(nvnMultisampleStateGetAlphaToCoverageDither) { ret_x(c, CST(ms_st, c)->a2c_dither); }
NVN_API(nvnMultisampleStateGetRasterSamples) { ret_x(c, CST(ms_st, c)->raster_samples); }
NVN_API(nvnMultisampleStateGetCoverageModulationMode) { ret_x(c, CST(ms_st, c)->cov_mod); }
NVN_API(nvnMultisampleStateGetCoverageTruncationEnable) { ret_x(c, CST(ms_st, c)->cov_trunc); }
NVN_API(nvnMultisampleStateGetCoverageToColorEnable) { ret_x(c, CST(ms_st, c)->cov2color_en); }
NVN_API(nvnMultisampleStateGetCoverageToColorOutput) { ret_x(c, CST(ms_st, c)->cov2color_out); }
NVN_API(nvnMultisampleStateGetSampleLocationsEnable) { ret_x(c, CST(ms_st, c)->sl_en); }
NVN_API(nvnMultisampleStateGetSampleLocationsGridEnable) { ret_x(c, CST(ms_st, c)->sl_grid); }

// ---- polygon state
NVN_API(nvnPolygonStateSetDefaults) {
    *ST(poly_st, c) = (poly_st){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0};
}
NVN_API(nvnPolygonStateSetCullFace) { ST(poly_st, c)->cull = arg_u32(c, 1); }
NVN_API(nvnPolygonStateSetFrontFace) { ST(poly_st, c)->front = arg_u32(c, 1); }
NVN_API(nvnPolygonStateSetPolygonMode) { ST(poly_st, c)->mode = arg_u32(c, 1); }
NVN_API(nvnPolygonStateSetPolygonOffsetEnables) { ST(poly_st, c)->offset_en = arg_u32(c, 1); }
NVN_API(nvnPolygonStateGetCullFace) { ret_x(c, CST(poly_st, c)->cull); }
NVN_API(nvnPolygonStateGetFrontFace) { ret_x(c, CST(poly_st, c)->front); }
NVN_API(nvnPolygonStateGetPolygonMode) { ret_x(c, CST(poly_st, c)->mode); }
NVN_API(nvnPolygonStateGetPolygonOffsetEnables) { ret_x(c, CST(poly_st, c)->offset_en); }

// ---- depth-stencil state: a stencil function and three stencil ops per face
NVN_API(nvnDepthStencilStateSetDefaults) {
    *ST(ds_st, c) = (ds_st){0, 1, NVN_DEPTH_FUNC_LESS, 0, NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS,
                            NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP,
                            NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP};
}
NVN_API(nvnDepthStencilStateSetDepthTestEnable) { ST(ds_st, c)->dt = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnDepthStencilStateSetDepthWriteEnable) { ST(ds_st, c)->dw = nvn_arg_bool(c, 1) != 0; }
NVN_API(nvnDepthStencilStateSetDepthFunc) { ST(ds_st, c)->df = arg_u32(c, 1); }
NVN_API(nvnDepthStencilStateSetStencilTestEnable) { ST(ds_st, c)->st = nvn_arg_bool(c, 1) != 0; }
// void nvnDepthStencilStateSetStencilFunc(NVNdepthStencilState *s, NVNface face, NVNstencilFunc func)
NVN_API(nvnDepthStencilStateSetStencilFunc) {
    ds_st *o = ST(ds_st, c);
    uint32_t face = arg_u32(c, 1), fn = arg_u32(c, 2);
    if (face & NVN_FACE_FRONT) o->sf_f = fn;
    if (face & NVN_FACE_BACK) o->sf_b = fn;
}
// void nvnDepthStencilStateSetStencilOp(NVNdepthStencilState *s, NVNface face, NVNstencilOp fail, zfail, zpass)
NVN_API(nvnDepthStencilStateSetStencilOp) {
    ds_st *o = ST(ds_st, c);
    uint32_t face = arg_u32(c, 1), fail = arg_u32(c, 2), zfail = arg_u32(c, 3), zpass = arg_u32(c, 4);
    if (face & NVN_FACE_FRONT) o->fail_f = fail, o->zfail_f = zfail, o->zpass_f = zpass;
    if (face & NVN_FACE_BACK) o->fail_b = fail, o->zfail_b = zfail, o->zpass_b = zpass;
}
NVN_API(nvnDepthStencilStateGetDepthTestEnable) { ret_x(c, CST(ds_st, c)->dt); }
NVN_API(nvnDepthStencilStateGetDepthWriteEnable) { ret_x(c, CST(ds_st, c)->dw); }
NVN_API(nvnDepthStencilStateGetDepthFunc) { ret_x(c, CST(ds_st, c)->df); }
NVN_API(nvnDepthStencilStateGetStencilTestEnable) { ret_x(c, CST(ds_st, c)->st); }
// NVNstencilFunc nvnDepthStencilStateGetStencilFunc(const NVNdepthStencilState *s, NVNface face): the back face's for
// BACK, the front face's otherwise
NVN_API(nvnDepthStencilStateGetStencilFunc) {
    const ds_st *o = CST(ds_st, c);
    ret_x(c, arg_u32(c, 1) == NVN_FACE_BACK ? o->sf_b : o->sf_f);
}
// void nvnDepthStencilStateGetStencilOp(const NVNdepthStencilState *s, NVNface face, NVNstencilOp *fail, *zfail, *zpass)
NVN_API(nvnDepthStencilStateGetStencilOp) {
    const ds_st *o = CST(ds_st, c);
    int back = arg_u32(c, 1) == NVN_FACE_BACK;
    put_u32(arg_p(c, 2), back ? o->fail_b : o->fail_f);
    put_u32(arg_p(c, 3), back ? o->zfail_b : o->zfail_f);
    put_u32(arg_p(c, 4), back ? o->zpass_b : o->zpass_f);
}

// ---- vertex attribute state: format, offset within its stream's element, stream index
NVN_API(nvnVertexAttribStateSetDefaults) { *ST(va_st, c) = (va_st){0, 0, 0}; }
// void nvnVertexAttribStateSetFormat(NVNvertexAttribState *s, NVNformat format, ptrdiff_t offset): the packed state
// holds formats below 256 (nvn.h's last is 0x9C) and offsets below 2^19
NVN_API(nvnVertexAttribStateSetFormat) {
    uint32_t fmt = arg_u32(c, 1);
    int64_t off = (int64_t)arg_x(c, 2);
    if (fmt > 0xff || off < 0 || off >= 1 << 19)
        rt_unimplemented(c, "nvn", "nvnVertexAttribStateSetFormat: format %#x at offset %lld does not fit the packed "
                         "state", fmt, (long long)off);
    va_st *o = ST(va_st, c);
    o->fmt = fmt, o->offset = (uint32_t)off;
}
NVN_API(nvnVertexAttribStateSetStreamIndex) { ST(va_st, c)->stream = arg_u32(c, 1); }
// void nvnVertexAttribStateGetFormat(const NVNvertexAttribState *s, NVNformat *format, ptrdiff_t *offset)
NVN_API(nvnVertexAttribStateGetFormat) {
    const va_st *o = CST(va_st, c);
    put_u32(arg_p(c, 1), o->fmt);
    int64_t off = o->offset;
    if (arg_p(c, 2)) memcpy(arg_p(c, 2), &off, 8);
}
NVN_API(nvnVertexAttribStateGetStreamIndex) { ret_x(c, CST(va_st, c)->stream); }

// ---- vertex stream state: stride and instance divisor
NVN_API(nvnVertexStreamStateSetDefaults) { *ST(vs_st, c) = (vs_st){0, 0}; }
NVN_API(nvnVertexStreamStateSetStride) { ST(vs_st, c)->stride = (uint32_t)arg_x(c, 1); }
NVN_API(nvnVertexStreamStateSetDivisor) { ST(vs_st, c)->divisor = arg_u32(c, 1); }
NVN_API(nvnVertexStreamStateGetStride) { ret_x(c, CST(vs_st, c)->stride); }
NVN_API(nvnVertexStreamStateGetDivisor) { ret_x(c, CST(vs_st, c)->divisor); }
