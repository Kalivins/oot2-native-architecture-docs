// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Stands for runtime or back-end code in tests/runtime/pages_test.c: accesses to guest memory made by host functions
// that are in no function table. Its own file, linked before the test, so that its code does not lie in the extent of
// a lifted function (the function tables give a function's start, not its end).
#include <stdint.h>

uint32_t host_read32(uint64_t a) { return *(volatile uint32_t *)(uintptr_t)a; }
void host_write32(uint64_t a, uint32_t v) { *(volatile uint32_t *)(uintptr_t)a = v; }
