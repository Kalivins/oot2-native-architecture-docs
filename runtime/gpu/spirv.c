// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Reading a module's declarations and adding the vertex outputs a pixel module needs (oot2/spirv.h). Both walk the
// instruction stream once and trust only its framing (every instruction's word count), which spv_declarations and
// spv_add_outputs check as they go; the modules are the translator's, which pass spirv-val, but a damaged pack must end
// in a refusal and not in a read past the module.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/spirv.h"

enum {
    OP_EXTENSION = 10, OP_MEMORY_MODEL = 14, OP_ENTRY_POINT = 15, OP_CAPABILITY = 17, OP_TYPE_INT = 21,
    OP_TYPE_FLOAT = 22, OP_TYPE_VECTOR = 23, OP_TYPE_POINTER = 32, OP_TYPE_FORWARD_POINTER = 39, OP_CONSTANT = 43,
    OP_CONSTANT_COMPOSITE = 44, OP_CONSTANT_NULL = 46, OP_FUNCTION = 54, OP_VARIABLE = 59, OP_DECORATE = 71,
    OP_MEMBER_DECORATE = 72, OP_DECORATION_GROUP = 73, OP_GROUP_DECORATE = 74, OP_GROUP_MEMBER_DECORATE = 75,
    OP_DECORATE_ID = 332, OP_DECORATE_STRING = 5632, OP_MEMBER_DECORATE_STRING = 5633,
};
enum { SC_OUTPUT = 3, DEC_LOCATION = 30, DEC_COMPONENT = 31 };

#define REFUSE(...)                                                                                                    \
    do {                                                                                                               \
        if (why && cap) snprintf(why, cap, __VA_ARGS__);                                                               \
        return 0;                                                                                                      \
    } while (0)

int spv_declarations(const uint32_t *w, size_t n, SpvDecl *out, char *why, size_t cap) {
    memset(out, 0, sizeof *out);
    if (n < 5 || w[0] != SPV_MAGIC) REFUSE("not a SPIR-V module (%zu words)", n);
    out->version = w[1];
    for (size_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) REFUSE("malformed instruction at word %zu", i);
        if (op == OP_MEMORY_MODEL || op == OP_ENTRY_POINT || op == OP_FUNCTION) break;   // the declarations are over
        if (op == OP_CAPABILITY && len == 2) {
            uint32_t c = w[i + 1];
            uint32_t k = 0;
            while (k < out->ncaps && out->caps[k] != c) k++;
            if (k == out->ncaps) {
                if (out->ncaps == SPV_MAX_CAPS) REFUSE("more than %d capabilities", SPV_MAX_CAPS);
                out->caps[out->ncaps++] = c;
            }
        } else if (op == OP_EXTENSION && len >= 2) {
            char name[SPV_EXT_LEN];
            size_t bytes = (size_t)(len - 1) * 4, m = bytes < sizeof name - 1 ? bytes : sizeof name - 1;
            memcpy(name, w + i + 1, m);
            name[m] = 0;   // strnlen-style: the string ends at its first NUL
            name[strnlen(name, m)] = 0;
            uint32_t k = 0;
            while (k < out->nexts && strcmp(out->exts[k], name)) k++;
            if (k == out->nexts) {
                if (out->nexts == SPV_MAX_EXTS) REFUSE("more than %d extensions", SPV_MAX_EXTS);
                snprintf(out->exts[out->nexts++], SPV_EXT_LEN, "%s", name);
            }
        }
        i += len;
    }
    return 1;
}

const char *spv_capability_name(uint32_t cap) {
    static const struct { uint32_t v; const char *name; } names[] = {
        {0, "Matrix"}, {1, "Shader"}, {2, "Geometry"}, {3, "Tessellation"}, {4, "Addresses"}, {5, "Linkage"},
        {6, "Kernel"}, {7, "Vector16"}, {8, "Float16Buffer"}, {9, "Float16"}, {10, "Float64"}, {11, "Int64"},
        {12, "Int64Atomics"}, {13, "ImageBasic"}, {14, "ImageReadWrite"}, {15, "ImageMipmap"}, {17, "Pipes"},
        {18, "Groups"}, {19, "DeviceEnqueue"}, {20, "LiteralSampler"}, {21, "AtomicStorage"}, {22, "Int16"},
        {23, "TessellationPointSize"}, {24, "GeometryPointSize"}, {25, "ImageGatherExtended"},
        {27, "StorageImageMultisample"}, {28, "UniformBufferArrayDynamicIndexing"},
        {29, "SampledImageArrayDynamicIndexing"}, {30, "StorageBufferArrayDynamicIndexing"},
        {31, "StorageImageArrayDynamicIndexing"}, {32, "ClipDistance"}, {33, "CullDistance"}, {34, "ImageCubeArray"},
        {35, "SampleRateShading"}, {36, "ImageRect"}, {37, "SampledRect"}, {38, "GenericPointer"}, {39, "Int8"},
        {40, "InputAttachment"}, {41, "SparseResidency"}, {42, "MinLod"}, {43, "Sampled1D"}, {44, "Image1D"},
        {45, "SampledCubeArray"}, {46, "SampledBuffer"}, {47, "ImageBuffer"}, {48, "ImageMSArray"},
        {49, "StorageImageExtendedFormats"}, {50, "ImageQuery"}, {51, "DerivativeControl"},
        {52, "InterpolationFunction"}, {53, "TransformFeedback"}, {54, "GeometryStreams"},
        {55, "StorageImageReadWithoutFormat"}, {56, "StorageImageWriteWithoutFormat"}, {57, "MultiViewport"},
        {61, "GroupNonUniform"}, {62, "GroupNonUniformVote"}, {63, "GroupNonUniformArithmetic"},
        {64, "GroupNonUniformBallot"}, {65, "GroupNonUniformShuffle"}, {66, "GroupNonUniformShuffleRelative"},
        {67, "GroupNonUniformClustered"}, {68, "GroupNonUniformQuad"}, {69, "ShaderLayer"},
        {70, "ShaderViewportIndex"}, {71, "UniformDecoration"}, {4422, "FragmentShadingRateKHR"},
        {4423, "SubgroupBallotKHR"}, {4427, "DrawParameters"}, {4431, "SubgroupVoteKHR"},
        {4433, "StorageBuffer16BitAccess"}, {4434, "UniformAndStorageBuffer16BitAccess"},
        {4435, "StoragePushConstant16"}, {4436, "StorageInputOutput16"}, {4437, "DeviceGroup"}, {4439, "MultiView"},
        {4441, "VariablePointersStorageBuffer"}, {4442, "VariablePointers"}, {4448, "StorageBuffer8BitAccess"},
        {4449, "UniformAndStorageBuffer8BitAccess"}, {4450, "StoragePushConstant8"}, {4464, "DenormPreserve"},
        {4465, "DenormFlushToZero"}, {4466, "SignedZeroInfNanPreserve"}, {4467, "RoundingModeRTE"},
        {4468, "RoundingModeRTZ"}, {5013, "StencilExportEXT"}, {5016, "Int64ImageEXT"}, {5055, "ShaderClockKHR"},
        {5254, "ShaderViewportIndexLayerEXT"}, {5283, "MeshShadingEXT"}, {5284, "FragmentBarycentricKHR"},
        {5301, "ShaderNonUniform"}, {5302, "RuntimeDescriptorArray"}, {5303, "InputAttachmentArrayDynamicIndexing"},
        {5304, "UniformTexelBufferArrayDynamicIndexing"}, {5305, "StorageTexelBufferArrayDynamicIndexing"},
        {5306, "UniformBufferArrayNonUniformIndexing"}, {5307, "SampledImageArrayNonUniformIndexing"},
        {5308, "StorageBufferArrayNonUniformIndexing"}, {5309, "StorageImageArrayNonUniformIndexing"},
        {5310, "InputAttachmentArrayNonUniformIndexing"}, {5311, "UniformTexelBufferArrayNonUniformIndexing"},
        {5312, "StorageTexelBufferArrayNonUniformIndexing"}, {5345, "VulkanMemoryModel"},
        {5346, "VulkanMemoryModelDeviceScope"}, {5347, "PhysicalStorageBufferAddresses"},
        {5363, "FragmentShaderSampleInterlockEXT"}, {5372, "FragmentShaderShadingRateInterlockEXT"},
        {5378, "FragmentShaderPixelInterlockEXT"}, {5379, "DemoteToHelperInvocation"},
        {5612, "AtomicFloat32MinMaxEXT"}, {5613, "AtomicFloat64MinMaxEXT"}, {5616, "AtomicFloat16MinMaxEXT"},
        {6016, "DotProductInputAll"}, {6017, "DotProductInput4x8Bit"}, {6018, "DotProductInput4x8BitPacked"},
        {6019, "DotProduct"}, {6022, "CooperativeMatrixKHR"}, {6025, "BitInstructions"},
        {6026, "GroupNonUniformRotateKHR"}, {6033, "AtomicFloat32AddEXT"}, {6034, "AtomicFloat64AddEXT"},
        {6035, "LongCompositesINTEL"}, {6114, "AtomicFloat16VectorNV"},
    };
    for (size_t i = 0; i < sizeof names / sizeof names[0]; i++)
        if (names[i].v == cap) return names[i].name;
    return NULL;
}

// ---- adding outputs

typedef struct Words {
    uint32_t *w;
    size_t n, cap;
    int bad;
} Words;

static void put(Words *o, uint32_t v) {
    if (o->n == o->cap) {
        size_t c = o->cap ? o->cap * 2 : 256;
        uint32_t *g = realloc(o->w, c * 4);
        if (!g) {
            o->bad = 1;
            return;
        }
        o->w = g, o->cap = c;
    }
    o->w[o->n++] = v;
}

static void inst(Words *o, uint32_t op, const uint32_t *args, uint32_t nargs) {
    put(o, (nargs + 1) << 16 | op);
    for (uint32_t i = 0; i < nargs; i++) put(o, args[i]);
}

typedef struct VecType { uint32_t id, comp, count; } VecType;
typedef struct PtrType { uint32_t id, type; } PtrType;

#define MAX_TYPES 96

static int is_annotation(uint32_t op) {
    return op == OP_DECORATE || op == OP_MEMBER_DECORATE || op == OP_DECORATION_GROUP || op == OP_GROUP_DECORATE ||
           op == OP_GROUP_MEMBER_DECORATE || op == OP_DECORATE_ID || op == OP_DECORATE_STRING ||
           op == OP_MEMBER_DECORATE_STRING;
}

static int is_type_or_constant(uint32_t op) {
    return (op >= 19 && op <= OP_TYPE_FORWARD_POINTER) || (op >= 41 && op <= 52) || op == OP_VARIABLE;
}

uint32_t *spv_add_outputs(const uint32_t *w, size_t n, const SpvOutput *add, uint32_t nadd, size_t *out_words, char *why,
                          size_t cap) {
    if (n < 5 || w[0] != SPV_MAGIC) REFUSE("not a SPIR-V module (%zu words)", n);
    size_t ep_at = 0, ann_end = 0, decl_at = 0, fn_at = 0;
    uint32_t entry_points = 0, scalar[3] = {0, 0, 0}, nvec = 0, nptr = 0;
    VecType vec[MAX_TYPES];
    PtrType ptr[MAX_TYPES];
    for (size_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) REFUSE("malformed instruction at word %zu", i);
        const uint32_t *a = w + i + 1;
        if (op == OP_FUNCTION) {
            fn_at = i;
            break;
        }
        if (op == OP_ENTRY_POINT) {
            if (!entry_points++) ep_at = i;
        } else if (is_annotation(op)) {
            ann_end = i + len;
        } else if (op == OP_TYPE_FLOAT && len >= 3 && a[1] == 32) {
            scalar[SPV_PAD_FLOAT] = a[0];
        } else if (op == OP_TYPE_INT && len >= 4 && a[1] == 32) {
            scalar[a[2] ? SPV_PAD_SINT : SPV_PAD_UINT] = a[0];
        } else if (op == OP_TYPE_VECTOR && len >= 4 && nvec < MAX_TYPES) {
            vec[nvec++] = (VecType){a[0], a[1], a[2]};
        } else if (op == OP_TYPE_POINTER && len >= 4 && a[1] == SC_OUTPUT && nptr < MAX_TYPES) {
            ptr[nptr++] = (PtrType){a[0], a[2]};
        }
        if (!decl_at && is_type_or_constant(op)) decl_at = i;
        i += len;
    }
    if (entry_points != 1) REFUSE("the module has %u entry points; padding handles exactly one", entry_points);
    if (!fn_at) REFUSE("the module has no function");
    if (!ann_end) ann_end = decl_at;
    if (!ann_end || ann_end < ep_at + (w[ep_at] >> 16) || ann_end > fn_at)
        REFUSE("the module's annotations and types are not where the logical layout puts them");

    uint32_t bound = w[3];
    Words types = {0}, decs = {0};
    uint32_t zero[3] = {0, 0, 0}, one[3] = {0, 0, 0};
    uint32_t *vars = calloc(nadd ? nadd : 1, sizeof *vars);
    if (!vars) REFUSE("out of memory");
    for (uint32_t k = 0; k < nadd; k++) {
        uint32_t t = add[k].type, comps = add[k].components, comp0 = add[k].component;
        if (t > SPV_PAD_UINT || comps < 1 || comps > 4 || comp0 + comps > 4) {
            free(vars), free(types.w), free(decs.w);
            REFUSE("output %u has an unpaddable shape (type %u, components %u..%u)", add[k].location, t, comp0,
                   comp0 + comps);
        }
        if (!scalar[t]) {
            scalar[t] = bound++;
            if (t == SPV_PAD_FLOAT) inst(&types, OP_TYPE_FLOAT, (uint32_t[]){scalar[t], 32}, 2);
            else inst(&types, OP_TYPE_INT, (uint32_t[]){scalar[t], 32, t == SPV_PAD_SINT}, 3);
        }
        uint32_t type = scalar[t];
        if (comps > 1) {
            uint32_t v = 0;
            while (v < nvec && !(vec[v].comp == type && vec[v].count == comps)) v++;
            if (v == nvec) {
                if (nvec == MAX_TYPES) {
                    free(vars), free(types.w), free(decs.w);
                    REFUSE("more than %d vector types", MAX_TYPES);
                }
                vec[nvec++] = (VecType){bound++, type, comps};
                inst(&types, OP_TYPE_VECTOR, (uint32_t[]){vec[v].id, type, comps}, 3);
            }
            type = vec[v].id;
        }
        uint32_t p = 0;
        while (p < nptr && ptr[p].type != type) p++;
        if (p == nptr) {
            if (nptr == MAX_TYPES) {
                free(vars), free(types.w), free(decs.w);
                REFUSE("more than %d pointer types", MAX_TYPES);
            }
            ptr[nptr++] = (PtrType){bound++, type};
            inst(&types, OP_TYPE_POINTER, (uint32_t[]){ptr[p].id, SC_OUTPUT, type}, 3);
        }
        // Component c of the variable is location component comp0 + c: 1.0f's bits for the fourth, zero for the rest
        // (attribute memory is untyped, so the integer types read the same bits).
        int has_one = comp0 + comps == 4;
        uint32_t init = bound++;
        if (!has_one) {
            inst(&types, OP_CONSTANT_NULL, (uint32_t[]){type, init}, 2);
        } else if (comps == 1) {
            inst(&types, OP_CONSTANT, (uint32_t[]){type, init, 0x3f800000u}, 3);
        } else {
            if (!zero[t]) {
                zero[t] = bound++, one[t] = bound++;
                inst(&types, OP_CONSTANT, (uint32_t[]){scalar[t], zero[t], 0}, 3);
                inst(&types, OP_CONSTANT, (uint32_t[]){scalar[t], one[t], 0x3f800000u}, 3);
            }
            uint32_t args[6] = {type, init};
            for (uint32_t c = 0; c < comps; c++) args[2 + c] = c + 1 == comps ? one[t] : zero[t];
            inst(&types, OP_CONSTANT_COMPOSITE, args, 2 + comps);
        }
        uint32_t var = bound++;
        inst(&types, OP_VARIABLE, (uint32_t[]){ptr[p].id, var, SC_OUTPUT, init}, 4);
        inst(&decs, OP_DECORATE, (uint32_t[]){var, DEC_LOCATION, add[k].location}, 3);
        if (comp0) inst(&decs, OP_DECORATE, (uint32_t[]){var, DEC_COMPONENT, comp0}, 3);
        vars[k] = var;
    }
    uint32_t ep_len = w[ep_at] >> 16;
    if (types.bad || decs.bad || ep_len + nadd > 0xffff) {
        free(vars), free(types.w), free(decs.w);
        REFUSE("out of memory, or an entry point too long to extend");
    }
    size_t total = n + types.n + decs.n + nadd;
    uint32_t *o = malloc(total * 4), *q = o;
    if (!o) {
        free(vars), free(types.w), free(decs.w);
        REFUSE("out of memory");
    }
    memcpy(q, w, 5 * 4), q[3] = bound, q += 5;
    memcpy(q, w + 5, (ep_at - 5) * 4), q += ep_at - 5;
    *q++ = (ep_len + nadd) << 16 | OP_ENTRY_POINT;
    memcpy(q, w + ep_at + 1, (ep_len - 1) * 4), q += ep_len - 1;
    if (nadd) memcpy(q, vars, (size_t)nadd * 4), q += nadd;
    memcpy(q, w + ep_at + ep_len, (ann_end - ep_at - ep_len) * 4), q += ann_end - ep_at - ep_len;
    if (decs.n) memcpy(q, decs.w, decs.n * 4), q += decs.n;
    memcpy(q, w + ann_end, (fn_at - ann_end) * 4), q += fn_at - ann_end;
    if (types.n) memcpy(q, types.w, types.n * 4), q += types.n;
    memcpy(q, w + fn_at, (n - fn_at) * 4), q += n - fn_at;
    *out_words = (size_t)(q - o);
    free(vars), free(types.w), free(decs.w);
    return o;
}

// ---- zero-initialising workgroup memory

enum { SC_WORKGROUP = 4 };

#define FAIL(...)                                                                                                      \
    do {                                                                                                               \
        if (why && cap) snprintf(why, cap, __VA_ARGS__);                                                               \
        return -1;                                                                                                     \
    } while (0)

int spv_zero_workgroup(const uint32_t *w, size_t n, uint32_t **out, size_t *out_words, char *why, size_t cap) {
    *out = NULL, *out_words = 0;
    if (n < 5 || w[0] != SPV_MAGIC) FAIL("not a SPIR-V module (%zu words)", n);
    uint32_t bound = w[3];
    // The type each pointer type points at, and the null constant made for a type, both by id (every id is below bound).
    uint32_t *pointee = calloc(bound ? bound : 1, sizeof *pointee), *null_of = calloc(bound ? bound : 1, sizeof *null_of);
    if (!pointee || !null_of) {
        free(pointee), free(null_of);
        FAIL("out of memory");
    }
    size_t targets = 0;
    for (size_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (!len || i + len > n) {
            free(pointee), free(null_of);
            FAIL("malformed instruction at word %zu", i);
        }
        if (op == OP_FUNCTION) break;   // the global declarations are over
        if (op == OP_TYPE_POINTER && len >= 4 && w[i + 1] < bound) pointee[w[i + 1]] = w[i + 3];
        if (op == OP_VARIABLE && len == 4 && w[i + 3] == SC_WORKGROUP) targets++;
        i += len;
    }
    if (!targets) {
        free(pointee), free(null_of);
        return 0;
    }
    // Each variable grows by one word (the initializer); a type's null constant is three words, declared right before
    // the first variable that needs it, which is after the type and before the use, as the logical layout asks.
    uint32_t *o = malloc((n + targets * 4) * sizeof *o), *q = o;
    if (!o) {
        free(pointee), free(null_of);
        FAIL("out of memory");
    }
    memcpy(q, w, 5 * 4), q += 5;
    uint32_t next = bound;
    for (size_t i = 5; i < n;) {
        uint32_t op = w[i] & 0xffff, len = w[i] >> 16;
        if (op == OP_FUNCTION) {
            memcpy(q, w + i, (n - i) * 4), q += n - i;
            break;
        }
        if (op == OP_VARIABLE && len == 4 && w[i + 3] == SC_WORKGROUP) {
            uint32_t t = w[i + 1] < bound ? pointee[w[i + 1]] : 0;
            if (!t || t >= bound) {
                uint32_t id = w[i + 2];
                free(o), free(pointee), free(null_of);
                FAIL("workgroup variable %u is not of a pointer type this module declares", id);
            }
            if (!null_of[t]) {
                null_of[t] = next++;
                *q++ = 3u << 16 | OP_CONSTANT_NULL, *q++ = t, *q++ = null_of[t];
            }
            *q++ = 5u << 16 | OP_VARIABLE, *q++ = w[i + 1], *q++ = w[i + 2], *q++ = SC_WORKGROUP, *q++ = null_of[t];
        } else {
            memcpy(q, w + i, (size_t)len * 4), q += len;
        }
        i += len;
    }
    o[3] = next;
    *out = o, *out_words = (size_t)(q - o);
    free(pointee), free(null_of);
    return (int)targets;
}
