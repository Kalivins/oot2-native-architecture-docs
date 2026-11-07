// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The module loader (the loader half of architecture D9): the player's plain modules at the bases tools/layout.py
// chose, with the dynamic relocations the console's loader would apply and every import bound.
//
// A plain module is an NSO0 file whose segments are stored uncompressed (header flag bits 0 to 2 and 7 clear), as the
// prepare step writes them. The lifted C was generated from the unrelocated words of text and rodata and holds
// absolute guest addresses, so each module sits at its layout base and relocations may only write its data segment.
//
// Binding order for a symbol a relocation names (runtime.h, host functions): a HOST_OVERRIDE of that name, else the
// module's own definition, else the first module in load order that defines it, else 0 for a weak symbol, else
// rt_hostfn_missing, whose call is the named fatal "unimplemented import: <symbol>". Every definition is offered to
// the overrides too, so a replaced function (the sdk's own longjmp, the C++ throw entry points) is replaced for the
// module that defines it as well.
#pragma once
#include <stdint.h>

#include "memory.h"

#define RT_MAX_LOADED 12   // main, subsdk0 to subsdk9, sdk

typedef struct RtModuleFile {
    const char *name;     // the layout's name: main, subsdk0 .. subsdk9, sdk
    const char *path;     // the plain module
    uint64_t base, size;  // MOD_<NAME>_BASE and MOD_<NAME>_SIZE from layout.h
    const char *sha256;   // SHA-256 of the whole file, lowercase hex, baked at build time; NULL skips the check
} RtModuleFile;

typedef struct RtLoaded {
    const char *name;
    uint64_t base, size;           // the image (text, rodata, data, bss), a whole number of pages
    uint64_t text_size;            // text starts at base
    uint64_t rodata_off, rodata_size, data_off, data_size;
    uint64_t bss_off, bss_size;    // bss_off = data_off + data_size; bss pages past the data are committed on touch
    uint64_t init, fini, init_array, init_array_size, fini_array, fini_array_size;   // guest addresses, 0 if absent
    uint8_t build_id[32];
} RtLoaded;

// Relocations applied and symbols bound, per module.
typedef struct RtLoadStats {
    uint64_t relative, relr, abs64, glob_dat, jump_slot;   // relocations by type (relr: RELATIVE packed in DT_RELR)
    uint32_t imports;                                      // named undefined symbols of the dynamic symbol table
    uint32_t imports_to[RT_MAX_LOADED];                    // ... bound to module i's definition (load order)
    uint32_t imports_to_host;                              // ... bound to a host function
    uint32_t imports_weak;                                 // ... weak and defined nowhere: 0
    uint32_t imports_missing;                              // ... strong and defined nowhere: named fatal on call
    uint32_t replaced;                                     // definitions of this module a host function replaces
} RtLoadStats;

// Map, check, relocate and bind the modules, in load order (layout.h MOD_LIST order), then seal text and rodata.
// After rt_mem_init, before rt_dispatch_init. Fatal on a file that is missing, compressed, damaged (header segment
// hashes, or sha256), of another size than the layout's, or a relocation it cannot apply.
void rt_load_modules(const RtModuleFile *files, int count);
int rt_loaded_count(void);
const RtLoaded *rt_loaded(int i);              // NULL past the count
const RtLoadStats *rt_load_stats(int i);
uint64_t rt_symbol(const char *name);          // the definition in load order, 0 when no module defines it
// Definitions a host function replaces. Relocations naming them hold `host`; direct calls inside the defining module
// reach the same override because the lifter emits rt_replaced (cpu.h) as the body of `guest`.
typedef struct RtReplaced { const char *symbol; uint64_t guest, host; } RtReplaced;
int rt_load_replaced_count(void);
const RtReplaced *rt_load_replaced(int i);
// One rt_log line per module with its counts, and one per replaced definition.
void rt_load_report(void);

// SHA-256 (FIPS 180-4), for the module checks.
void rt_sha256(const void *data, uint64_t size, uint8_t out[32]);
