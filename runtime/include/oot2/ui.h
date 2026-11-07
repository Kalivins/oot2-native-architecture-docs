// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The C face of runtime/ui, the runtime's one C++ directory (architecture section 2: the runtime is C11, and the
// libraries a menu can be built on are C++). C++ stays behind this header: every function here is `extern "C"`, takes
// and returns C types, and the C++ side is built without exceptions and without run-time type information
// (tools/build_game.py, CXX_FLAGS), because the runtime does not unwind C++ exceptions (research 05). No C file
// includes a C++ header, and no C++ object may let an exception leave a function declared here.
//
// Until the menu's library is chosen and fetched, the directory holds only a stand-in: one file that uses the C++
// standard library (a container, a string, an algorithm) so that building and linking it proves the path
// (research 104). The functions below are the stand-in's; the menu replaces them.
#pragma once
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

// How the C++ side was compiled: its language level (the value of __cplusplus, 201703 for C++17) and whether
// exceptions and run-time type information were on (both 0 by the build's flags). A test reads it, so a change of
// the flags that does not reach the compiler shows.
typedef struct UiBuild {
    long dialect;
    int exceptions;
    int rtti;
} UiBuild;

UiBuild ui_standin_build(void);

// Split text at commas, drop the empty entries and the repeats, sort the rest and write them to out joined with
// " | ", terminated within cap bytes (out may be NULL when cap is 0). Returns the length of the whole joined text
// without its terminator: a value of cap or more means the text was cut.
size_t ui_standin_list(const char *text, char *out, size_t cap);

#ifdef __cplusplus
}
#endif
