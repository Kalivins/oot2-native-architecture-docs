// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The start-up of the Horizon layer under the SDK (runtime/hos, architecture D9), for the program's main (runtime/program
// /main.c) and the tests. Everything else of that layer is internal to runtime/hos.
#pragma once
#include <stdint.h>

#include "program.h"

// After rt_kernel_init, rt_load_modules and nvn_procs_register, before rt_dispatch_init: registers the service SVCs,
// the boot driver's host functions and the census's stop hook, and logs every override that binds nothing in the
// loaded modules. game_folder holds the program and receives census.tsv; files are the program glue's checked files
// (the RomFS image and index are found among them by their description).
void hos_init(const char *game_folder, const RtGameFile *files, int file_count);
// The main thread's entry for rt_kernel_run: a host function that calls the SDK's nn::init::Start as rtld would.
uint64_t hos_boot_entry(void);
// census.tsv when the run ends without a stop (every guest thread ended); why goes into its stop row.
void hos_census_write(const char *why);
