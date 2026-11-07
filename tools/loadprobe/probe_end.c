// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The end marker of the load probe's modules (ticket 92). The probe has no lifted code, only a stand-in function in
// load_probe.c; tools/loadprobe/load_probe.py links this file right after it, so the function below lies past the
// stand-in, as a module's <mod>_lifted_end lies past its lifted code. load_probe.c checks the order before anything.
#include "oot2/runtime.h"

void probe_lifted_end(Cpu *c);

__attribute__((noinline, used)) void probe_lifted_end(Cpu *c) {
    __asm__ volatile("int3" : : "r"(c) : "memory");   // never called: only its address is used
}
