// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the pipeline builder reads of a module's resource description (schema 2, xlate/host/ResultWriter.cs; the text is
// the translator's JSON, stored unchanged in the shader cache pack, oot2/shader_cache.h): the stage, whether the module
// fits the `uni` layout, and its user interface by location, which decides whether a vertex module has to be padded
// for the pixel module it is linked to (research 20, finding G). Nothing here needs Vulkan.
#pragma once
#include <stddef.h>
#include <stdint.h>

// The stage names of the description ("stage": vs, ps, cs, ts, ms), in the order of the runtime's NVN_STAGE_* kinds.
enum { SHD_VS, SHD_PS, SHD_CS, SHD_TS, SHD_MS, SHD_KINDS };
// The scalar types of an interface variable ("float32", "sint32", "uint32").
enum { SHD_FLOAT, SHD_SINT, SHD_UINT };

#define SHD_MAX_IO 64          // user interface entries a description may list per direction
#define SHD_MAX_LOCATIONS 64   // locations an entry or an array of entries may reach

// One user input or output variable (a Location decoration). An array of length n takes n locations in a row.
typedef struct ShdIo {
    int32_t location;
    uint8_t component;     // first component used at the location (0 to 3)
    uint8_t components;    // 1 to 4
    uint8_t type;          // SHD_FLOAT, SHD_SINT or SHD_UINT
    uint8_t bits;          // 32 for every variable the translator declares today
    int32_t array_length;  // 1 for a variable that is not an array
    uint8_t used;
} ShdIo;

typedef struct ShaderDesc {
    uint32_t schema;
    int kind;              // SHD_*, or -1 when the description names no known stage
    int fits;              // layout.fits: the module's resources fit the stage's slots of the layout
    int raw_pointers;      // raw_access.physical_pointers
    uint32_t ninputs, noutputs;
    ShdIo inputs[SHD_MAX_IO], outputs[SHD_MAX_IO];
} ShaderDesc;

const char *shd_kind_name(int kind);   // "vs", "ps", "cs", "ts", "ms"; "?" for another value

// Parse the description text json[0..n) (not NUL-terminated). 1 with *out filled, or 0 with one line in why. A
// description of another schema than the one the runtime reads, or of no stage, or with more interface entries than
// SHD_MAX_IO, is refused.
int shd_parse(const char *json, size_t n, uint32_t schema, ShaderDesc *out, char *why, size_t cap);

// A vertex output a pixel module reads and the vertex module does not write, as one variable of consecutive
// components of one location.
typedef struct ShdPad {
    uint16_t location;
    uint8_t component, components, type;
} ShdPad;

// The padding a vertex module needs to be linked to a pixel module: every component word of a location the pixel
// module declares as input and the vertex module declares no output for, as runs of consecutive components of one type,
// in location and component order. Returns how many (at most max), or -1 with the reason in why when a variable cannot
// be padded (more locations than SHD_MAX_LOCATIONS, a type that is not 32 bits, more runs than max).
int shd_pad_set(const ShaderDesc *vs, const ShaderDesc *ps, ShdPad *out, int max, char *why, size_t cap);
// The canonical text of a pad set ("l:c:n:t;" per run, empty for none), which keys the padded library.
void shd_pad_text(const ShdPad *pad, int n, char *out, size_t cap);
