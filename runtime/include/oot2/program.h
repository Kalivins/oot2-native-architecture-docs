// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the build bakes into each program for the player's dump, and what the program's start-up reads from it.
//
// tools/build_game.py writes program_glue.c beside the lifted objects: rt_program and rt_program_lifted below. The
// runtime library (runtime/program/main.c among it) holds no game data; the glue is the only object that knows the
// player's dump, and it is never part of the repository.
//
// Paths are relative to the game folder (the folder of the executable, or the one --game names) unless absolute.
//
// No entry point: no module's first word is code and rtld is not loaded, so the main thread starts in the boot
// driver (oot2/boot.h), which finds the SDK's nn::init::Start by name.
#pragma once
#include <stdint.h>

#include "loader.h"
#include "runtime.h"

// A file the program checks at each start besides the modules, whose own check is the loader's (RtModuleFile.sha256).
typedef struct RtGameFile {
    const char *what;     // how the player is told about it: "main.npdm", "RomFS index"
    const char *path;
    uint64_t size;
    const char *sha256;   // whole-file SHA-256, lowercase hex; NULL: the size only, for a file too large to hash at
                          // each start
    const char *sample_sha256;   // with sha256 NULL: the SHA-256 of fixed ranges (main.c, as tools/build_game.py
                                 // sample_ranges picks them), lowercase hex; NULL: none. Set for the player's dump when
                                 // the RomFS is read from it in place, so a dump moved, replaced or changed stops the
                                 // start before a wrong byte is served
} RtGameFile;

typedef struct RtProgram {
    const RtModuleFile *modules;   // every module of the layout, in load order
    int module_count;
    const RtGameFile *files;
    int file_count;
    int has_npdm;                  // the dump has a main.npdm; the values below are 0 otherwise
    uint64_t system_resource_size; // main.npdm: TotalSystemResourceSize (GetInfo 16)
    uint32_t main_thread_priority, main_thread_core, main_thread_stack_size;
} RtProgram;

extern const RtProgram rt_program;

// The lifted modules' function tables (a partial build may lift fewer modules than it loads). Returns the count.
int rt_program_lifted(RtModule *out, int max);
