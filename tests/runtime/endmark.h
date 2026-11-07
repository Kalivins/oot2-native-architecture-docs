// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The end marker of the synthetic modules of the runtime tests (the lifted-code end marker). Every RtModule needs the address of the
// first byte past its lifted code; these tests keep their guest functions in the test's own file and link endmark.c
// right after it, as the lifter's <mod>_table.c is the last object of its module.
#ifndef OOT2_TEST_ENDMARK_H
#define OOT2_TEST_ENDMARK_H

#include "oot2/runtime.h"

void test_lifted_end(Cpu *c);
#define TEST_END ((const void *)test_lifted_end)

#endif
