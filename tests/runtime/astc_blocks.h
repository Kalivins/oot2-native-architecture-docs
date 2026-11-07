// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Synthetic ASTC blocks for the checks of the GPU decode (astc_decode_test.c, vk_texture_test.c): a seeded generator of
// 16-byte blocks that tests/runtime/astc_ref.c decodes without error and without an HDR endpoint mode. No byte of any game.
#pragma once
#include <stdint.h>
#include <string.h>

#include "astc_ref.h"

static uint64_t astc_rng_state = 0x9E3779B97F4A7C15ull;

static inline void astc_seed(uint64_t seed) { astc_rng_state = seed ? seed : 0x9E3779B97F4A7C15ull; }

static inline uint64_t astc_rnd(void) {
    astc_rng_state ^= astc_rng_state >> 12, astc_rng_state ^= astc_rng_state << 25, astc_rng_state ^= astc_rng_state >> 27;
    return astc_rng_state * 0x2545F4914F6CDD1Dull;
}

// A void-extent block of a random colour: 0x1FC with the HDR bit clear and the two reserved bits set, every extent coordinate
// all ones ("no extent"), then four UNORM16.
static inline void astc_void_block(uint8_t b[16]) {
    for (int i = 0; i < 16; i++) b[i] = (uint8_t)astc_rnd();
    uint64_t lo = 0x1FCull | (3ull << 10);
    lo |= ((1ull << 52) - 1) << 12;
    for (int i = 0; i < 8; i++) b[i] = (uint8_t)(lo >> (8 * i));
}

// A block as an encoder writes it: every bit the reference's decoding does not depend on is zero. The padding between the
// colour values and the weights, and the bits of a last partial trit or quint block that no value uses, are such bits; the
// specification leaves them to the encoder, which writes zeros.
static inline void astc_canonical(uint8_t b[16], uint32_t bw, uint32_t bh) {
    uint8_t want[12 * 12 * 4], got[12 * 12 * 4];
    astc_ref_block(b, bw, bh, false, 1, want);
    for (int bit = 0; bit < 128; bit++) {
        if (!(b[bit >> 3] >> (bit & 7) & 1)) continue;
        b[bit >> 3] ^= (uint8_t)(1u << (bit & 7));
        astc_ref_block(b, bw, bh, false, 1, got);
        if (memcmp(want, got, (size_t)bw * bh * 4)) b[bit >> 3] ^= (uint8_t)(1u << (bit & 7));
    }
}

// A random block the reference decodes without error and without an HDR endpoint mode, in canonical form. err and hdr
// count the draws left out (either may be NULL). Returns 0 when 4,000,000 draws gave none.
static inline int astc_draw_block(uint32_t bw, uint32_t bh, uint8_t b[16], unsigned *err, unsigned *hdr) {
    uint8_t out[12 * 12 * 4];
    for (unsigned tries = 0; tries < 4000000; tries++) {
        for (int i = 0; i < 16; i++) b[i] = (uint8_t)astc_rnd();
        int st = astc_ref_block(b, bw, bh, false, 1, out);
        if (st == ASTC_REF_ERROR) {
            if (err) (*err)++;
        } else if (st == ASTC_REF_HDR) {
            if (hdr) (*hdr)++;
        } else {
            astc_canonical(b, bw, bh);
            return 1;
        }
    }
    return 0;
}
