// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The CPU reference of the ASTC decode checks (tests/runtime/astc_ref.c, taken from LAS Recomp under MIT, see
// THIRD_PARTY_NOTICES.md): one 2D LDR block to RGBA8, for any of the 14 block sizes NVN has formats for.
#pragma once
#include <stdbool.h>
#include <stdint.h>

enum {
    ASTC_REF_OK = 0,
    ASTC_REF_ERROR = 1,   // an illegal encoding (or an HDR void extent): the output is magenta
    ASTC_REF_HDR = 2,     // a partition used an HDR endpoint mode: its endpoints are magenta in the output
};

// One block to bw * bh texels of RGBA8, rows of bw texels. srgb selects the sRGB endpoint expansion. round8 picks how the
// 16-bit result becomes eight bits: 0 keeps the top eight bits (astcenc), 1 takes the nearest of the 255 steps (the GPU
// shader's). Returns one of the codes above.
int astc_ref_block(const uint8_t block[16], uint32_t bw, uint32_t bh, bool srgb, int round8, uint8_t *rgba_out);
