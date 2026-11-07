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
