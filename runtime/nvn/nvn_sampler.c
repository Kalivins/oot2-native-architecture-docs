// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Samplers (architecture D10, D13): the sampler builder and samplers, both holding the same parameters (samp_par) in
// the game's objects. A sampler owns nothing on the host: it reaches the back end only when the game registers it in a
// sampler pool (nvn_dpool.c), with its whole state, so that the back end can share one host sampler among every
// registration with the same state, under the host's limit on sampler objects (4,000 on the RTX 3060 Ti, D13). The
// game builds about 19,000 samplers over a whole run and registers about 250 of them (a count of its NVN2 calls).
//
// The border colour is kept as the four 32-bit words the game gives, float or integer, as the console's sampler
// descriptor keeps them: the texture's format decides how they are read.
//
// Ported from oot-pc-recomp runtime/nvn_api_obj.c and nvn_ext.c at 8e09aaa (GPL-3.0-or-later), as DKB Recomp carries
// them at 9d8fa02, ported there from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c and totk_nvn_ext.c
// at b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile; licenses/dkb-xlate-MIT.txt). Changed here: the
// integer border colours keep their bits instead of a conversion to float, which lost values past 2^24; a filter,
// wrap, compare or reduction value past NVN's stops by name.
#include <string.h>

#include "nvn_api.h"

#define SAMP_MAGIC MAGIC('S', 'M', 'P', 0)

#define SB(c) (&((samp_builder_t *)arg_p(c, 0))->p)
#define CSB(c) ((const samp_par *)&((const samp_builder_t *)arg_p(c, 0))->p)
#define CS(c) ((const samp_par *)&((const samp_t *)arg_p(c, 0))->p)

// An enumerant the sampler state keeps in a byte. Past `last` it is a value the NVN header does not define, which an
// NVN2 addition could be: a named stop, so that it is seen.
static uint8_t enum_arg(const Cpu *c, int i, uint32_t last, const char *fn) {
    uint32_t v = (uint32_t)arg_x(c, i);
    if (v > last) rt_unimplemented(c, "nvn", "%s: value %u, past the last one NVN defines (%u)", fn, v, last);
    return (uint8_t)v;
}

// void nvnSamplerBuilderSetDefaults(NVNsamplerBuilder *b): keeps the device; OpenGL's defaults
NVN_API(nvnSamplerBuilderSetDefaults) {
    samp_builder_t *b = arg_p(c, 0);
    NVNdevice *dev = b->dev;
    memset(b, 0, sizeof *b);
    b->dev = dev;
    samp_par *p = &b->p;
    p->minf = NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR, p->magf = NVN_MAG_FILTER_LINEAR;
    p->ws = p->wt = p->wr = NVN_WRAP_MODE_REPEAT;
    p->cmp_func = NVN_COMPARE_FUNC_LESS;
    p->lod_min = -1000.f, p->lod_max = 1000.f;
    p->aniso = 1.f;
}
NVN_API(nvnSamplerBuilderSetDevice) { ((samp_builder_t *)arg_p(c, 0))->dev = arg_p(c, 1); }
NVN_API(nvnSamplerBuilderGetDevice) { ret_x(c, (uintptr_t)((const samp_builder_t *)arg_p(c, 0))->dev); }
// void nvnSamplerBuilderSetMinMagFilter(NVNsamplerBuilder *b, NVNminFilter min, NVNmagFilter mag)
NVN_API(nvnSamplerBuilderSetMinMagFilter) {
    samp_par *p = SB(c);
    p->minf = enum_arg(c, 1, NVN_MIN_FILTER_LINEAR_MIPMAP_LINEAR, "nvnSamplerBuilderSetMinMagFilter");
    p->magf = enum_arg(c, 2, NVN_MAG_FILTER_LINEAR, "nvnSamplerBuilderSetMinMagFilter");
}
// void nvnSamplerBuilderSetWrapMode(NVNsamplerBuilder *b, NVNwrapMode s, NVNwrapMode t, NVNwrapMode r)
NVN_API(nvnSamplerBuilderSetWrapMode) {
    samp_par *p = SB(c);
    p->ws = enum_arg(c, 1, NVN_WRAP_MODE_CLAMP_TO_EDGE, "nvnSamplerBuilderSetWrapMode");
    p->wt = enum_arg(c, 2, NVN_WRAP_MODE_CLAMP_TO_EDGE, "nvnSamplerBuilderSetWrapMode");
    p->wr = enum_arg(c, 3, NVN_WRAP_MODE_CLAMP_TO_EDGE, "nvnSamplerBuilderSetWrapMode");
}
// void nvnSamplerBuilderSetLodClamp(NVNsamplerBuilder *b, float min, float max): the floats arrive in v0 and v1
NVN_API(nvnSamplerBuilderSetLodClamp) {
    samp_par *p = SB(c);
    p->lod_min = arg_s(c, 0), p->lod_max = arg_s(c, 1);
}
NVN_API(nvnSamplerBuilderSetLodBias) { SB(c)->lod_bias = arg_s(c, 0); }
NVN_API(nvnSamplerBuilderSetMaxAnisotropy) { SB(c)->aniso = arg_s(c, 0); }
NVN_API(nvnSamplerBuilderSetLodSnap) { SB(c)->lod_snap = arg_s(c, 0); }
// void nvnSamplerBuilderSetCompare(NVNsamplerBuilder *b, NVNcompareMode mode, NVNcompareFunc func)
NVN_API(nvnSamplerBuilderSetCompare) {
    samp_par *p = SB(c);
    p->cmp_mode = enum_arg(c, 1, NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE, "nvnSamplerBuilderSetCompare");
    p->cmp_func = enum_arg(c, 2, NVN_COMPARE_FUNC_ALWAYS, "nvnSamplerBuilderSetCompare");
}
// void nvnSamplerBuilderSetReductionFilter(NVNsamplerBuilder *b, NVNsamplerReduction r)
NVN_API(nvnSamplerBuilderSetReductionFilter) {
    SB(c)->reduction = enum_arg(c, 1, NVN_SAMPLER_REDUCTION_MAX, "nvnSamplerBuilderSetReductionFilter");
}
// void nvnSamplerBuilderSetBorderColor{,i,ui}(NVNsamplerBuilder *b, const float / int / uint32_t *rgba)
static void set_border(Cpu *c) { memcpy(SB(c)->border, arg_p(c, 1), sizeof SB(c)->border); }
NVN_API(nvnSamplerBuilderSetBorderColor) { set_border(c); }
NVN_API(nvnSamplerBuilderSetBorderColori) { set_border(c); }
NVN_API(nvnSamplerBuilderSetBorderColorui) { set_border(c); }

// The getters shared by builder and sampler: (o, NVNminFilter *, NVNmagFilter *), (o, NVNwrapMode *s, *t, *r),
// (o, float *min, float *max), (o, NVNcompareMode *, NVNcompareFunc *), (o, float / int / uint32_t *rgba)
static void get_min_mag(const Cpu *c, const samp_par *p) {
    *(int32_t *)arg_p(c, 1) = p->minf, *(int32_t *)arg_p(c, 2) = p->magf;
}
static void get_wrap(const Cpu *c, const samp_par *p) {
    *(int32_t *)arg_p(c, 1) = p->ws, *(int32_t *)arg_p(c, 2) = p->wt, *(int32_t *)arg_p(c, 3) = p->wr;
}
static void get_lod_clamp(const Cpu *c, const samp_par *p) {
    *(float *)arg_p(c, 1) = p->lod_min, *(float *)arg_p(c, 2) = p->lod_max;
}
static void get_compare(const Cpu *c, const samp_par *p) {
    *(int32_t *)arg_p(c, 1) = p->cmp_mode, *(int32_t *)arg_p(c, 2) = p->cmp_func;
}
static void get_border(const Cpu *c, const samp_par *p) { memcpy(arg_p(c, 1), p->border, sizeof p->border); }

NVN_API(nvnSamplerBuilderGetMinMagFilter) { get_min_mag(c, CSB(c)); }
NVN_API(nvnSamplerBuilderGetWrapMode) { get_wrap(c, CSB(c)); }
NVN_API(nvnSamplerBuilderGetLodClamp) { get_lod_clamp(c, CSB(c)); }
NVN_API(nvnSamplerBuilderGetLodBias) { ret_s(c, CSB(c)->lod_bias); }
NVN_API(nvnSamplerBuilderGetMaxAnisotropy) { ret_s(c, CSB(c)->aniso); }
NVN_API(nvnSamplerBuilderGetLodSnap) { ret_s(c, CSB(c)->lod_snap); }
NVN_API(nvnSamplerBuilderGetCompare) { get_compare(c, CSB(c)); }
NVN_API(nvnSamplerBuilderGetReductionFilter) { ret_x(c, CSB(c)->reduction); }
NVN_API(nvnSamplerBuilderGetBorderColor) { get_border(c, CSB(c)); }
NVN_API(nvnSamplerBuilderGetBorderColori) { get_border(c, CSB(c)); }
NVN_API(nvnSamplerBuilderGetBorderColorui) { get_border(c, CSB(c)); }

// NVNboolean nvnSamplerInitialize(NVNsampler *s, const NVNsamplerBuilder *b)
NVN_API(nvnSamplerInitialize) {
    samp_t *o = arg_p(c, 0);
    o->magic = SAMP_MAGIC;
    o->p = ((const samp_builder_t *)arg_p(c, 1))->p;
    ret_x(c, 1);
}
NVN_API(nvnSamplerFinalize) { ((samp_t *)arg_p(c, 0))->magic = 0; }
NVN_API(nvnSamplerSetDebugLabel) { (void)c; }
NVN_API(nvnSamplerGetMinMagFilter) { get_min_mag(c, CS(c)); }
NVN_API(nvnSamplerGetWrapMode) { get_wrap(c, CS(c)); }
NVN_API(nvnSamplerGetLodClamp) { get_lod_clamp(c, CS(c)); }
NVN_API(nvnSamplerGetLodBias) { ret_s(c, CS(c)->lod_bias); }
NVN_API(nvnSamplerGetMaxAnisotropy) { ret_s(c, CS(c)->aniso); }
NVN_API(nvnSamplerGetCompare) { get_compare(c, CS(c)); }
NVN_API(nvnSamplerGetReductionFilter) { ret_x(c, CS(c)->reduction); }
NVN_API(nvnSamplerGetBorderColor) { get_border(c, CS(c)); }
NVN_API(nvnSamplerGetBorderColori) { get_border(c, CS(c)); }
NVN_API(nvnSamplerGetBorderColorui) { get_border(c, CS(c)); }
// NVNboolean nvnSamplerCompare(const NVNsampler *a, const NVNsampler *b): the same state, bit for bit
NVN_API(nvnSamplerCompare) { ret_x(c, !memcmp(CS(c), &((const samp_t *)arg_p(c, 1))->p, sizeof(samp_par))); }
