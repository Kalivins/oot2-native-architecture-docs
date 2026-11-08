// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The two things the pipeline builder does to a translated SPIR-V module besides handing it to the driver
// (runtime/gpu/spirv.c, architecture D12 and D13): read what it declares (its capabilities and extensions, which the
// device's required list has to give, the pipeline cost study and the pipeline builder), and add the vertex outputs a pixel module
// reads and the vertex module does not write, set to what the console's next stage reads from an unwritten output,
// (0, 0, 0, 1) (DKB Recomp's spirv_add_outputs does the same; this is a new implementation of the rule, see
// THIRD_PARTY_NOTICES.md). Nothing here needs Vulkan.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define SPV_MAGIC 0x07230203u
#define SPV_MAX_CAPS 48
#define SPV_MAX_EXTS 16
#define SPV_EXT_LEN 64

typedef struct SpvDecl {
    uint32_t version;                             // header word 1 (0x00010400 for SPIR-V 1.4)
    uint32_t ncaps;                               // OpCapability values, in order of appearance, each once
    uint32_t caps[SPV_MAX_CAPS];
    uint32_t nexts;                               // OpExtension names, in order of appearance, each once
    char exts[SPV_MAX_EXTS][SPV_EXT_LEN];
} SpvDecl;

// What the module declares before its memory model. 1 with *out filled, or 0 with one line in why (a header that is
// not SPIR-V, an instruction that runs past the end, more declarations than the arrays hold).
int spv_declarations(const uint32_t *words, size_t nwords, SpvDecl *out, char *why, size_t cap);

// The SPIR-V name of capability value cap ("Float64", "PhysicalStorageBufferAddresses", ...); NULL when this build
// does not name it.
const char *spv_capability_name(uint32_t cap);

// One variable to add: Output storage class, `components` consecutive components starting at `component` of
// `location`, of a 32-bit scalar type (0 float, 1 signed int, 2 unsigned int; the numbering of oot2/shader_desc.h).
enum { SPV_PAD_FLOAT, SPV_PAD_SINT, SPV_PAD_UINT };

typedef struct SpvOutput {
    uint32_t location, component, components, type;
} SpvOutput;

// The module with one Output variable per entry of out[0..n) added: declared in the entry point's interface, decorated
// with Location (and Component when it is not 0), initialised so that component c of the variable reads 1.0f's bits
// when location component (component + c) is 3 and 0 otherwise. Existing scalar, vector and Output pointer types are
// reused; a type or constant that is missing is declared. Returns a malloc'd module of *out_words words, or NULL with
// the reason in why (not SPIR-V, no entry point or more than one, no annotations and no types, out of memory).
uint32_t *spv_add_outputs(const uint32_t *words, size_t nwords, const SpvOutput *add, uint32_t nadd, size_t *out_words,
                          char *why, size_t cap);

// The module with every Workgroup variable that has no initializer given the null value of its type. A translated
// compute module can read workgroup words it never wrote (the title movie's last pass does: with those words zero its
// picture is right, with NaN bits every tile is black), and a device leaves workgroup memory as the dispatch before left
// it, so what the module reads depends on what ran before; the console's program is right when the words are finite
// (Inference). The initializer is the null value, the zero-initialization of workgroup memory a Vulkan 1.3 device offers
// (VkPhysicalDeviceVulkan13Features::shaderZeroInitializeWorkgroupMemory, which the device must have enabled), done by the
// driver before the first instruction. Returns how many variables were given one: 0 with *out NULL when the module needs
// nothing (it has no such variable), else *out is a malloc'd module of *out_words words; -1 with the reason in why (not
// SPIR-V, a variable whose type this cannot name, out of memory).
int spv_zero_workgroup(const uint32_t *words, size_t nwords, uint32_t **out, size_t *out_words, char *why, size_t cap);
