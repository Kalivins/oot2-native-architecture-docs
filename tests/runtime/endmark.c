// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Linked right after a runtime test's own file, so the function below lies after that file's guest functions: the end
// marker of the test's synthetic module (what <mod>_lifted_end is in the lifter's output).
#include "endmark.h"

__attribute__((noinline, used)) void test_lifted_end(Cpu *c) {
    __asm__ volatile("int3" : : "r"(c) : "memory");
}

// Runtime code that the linker placed right after the module's lifted code: it reads guest memory for the tests of the
// page ranges (tests/runtime/pages_test.c), which must call it host code, not the tail of the last lifted function.
__attribute__((noinline)) uint32_t host_read32_after_end(uint64_t a) { return *(volatile uint32_t *)(uintptr_t)a; }
