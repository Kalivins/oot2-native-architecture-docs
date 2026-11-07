// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Checks of the staging ring and the texture arena (runtime/gpu/texmem.c, oot2/texmem.h) without a device: the memory
// stands in (a chunk is an address range nobody writes, a block a counter), and the timeline is a number the test moves.
// Driven by tests/test_runtime_texmem.py: `texmem_test <case>`, one case per process; a passing case prints "ok <case>";
// a failing one ends in one named fatal, which the driver reads.
//
//   ring        regions of one submission share chunks and never overlap; a chunk comes back, from its start, once its
//               submission is done; at the 384 MiB cap a request waits for the oldest submission instead of growing,
//               the ring's chunks never held more than the cap, and what the waiting submission got did not overlap
//               what the pending one held
//   arena       extents are aligned, never overlap and fill a block exactly; freeing in any order joins them back to
//               whole blocks; a request larger than a block gets a block of its own; the 3.4 GiB cap holds (54 blocks of
//               64 MiB) and the next block that would pass it stops by name; empty blocks go back to the device to make
//               room for a block of another memory type; the memory budget refuses a block as the mirrors' rule does
//   ring_full   (fatal) one submission asks for more than the ring holds with nothing earlier to wait for
//   ring_big    (fatal) a region larger than the ring
//   arena_full  (fatal) the cap
//   arena_budget (fatal) the device's budget
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/texmem.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MiB (1ull << 20)
typedef unsigned long long ull;

// ---------- the stand-ins
static uint64_t g_done, g_next_addr = 0x100000000ull, g_waited;
static uint64_t g_blocks_live, g_budget_used, g_budget_total;
static int g_budget_known;
static uint8_t g_dummy[64];

static int op_chunk_new(uint64_t size, TexChunk *out) {
    *out = (TexChunk){.map = g_dummy, .address = g_next_addr, .size = size};
    g_next_addr += size + MiB;
    return 1;
}
static void op_chunk_free(const TexChunk *c) { (void)c; }
static uint32_t op_type_pick(uint32_t bits) {
    for (uint32_t i = 0; i < 32; i++)
        if (bits >> i & 1) return i;
    return UINT32_MAX;
}
static int op_block_new(uint32_t type, uint64_t size, uint64_t *handle) {
    (void)type;
    static uint64_t next = 1;
    *handle = next++;
    g_blocks_live += size;
    return 1;
}
static uint64_t g_freed_blocks;
static void op_block_free(uint64_t handle) {
    (void)handle;
    g_freed_blocks++;
}
static int op_budget(uint64_t *used, uint64_t *budget) {
    *used = g_budget_used, *budget = g_budget_total;
    return g_budget_known;
}
static uint64_t op_done(void) { return g_done; }
static void op_wait(uint64_t value) {
    g_waited = value;
    if (value > g_done) g_done = value;
}
static const TexMemOps k_ops = {op_chunk_new, op_chunk_free, op_type_pick, op_block_new, op_block_free, op_budget, op_done, op_wait};

static void open_fresh(void) {
    memset(g_dummy, 0, sizeof g_dummy);
    g_done = g_waited = g_blocks_live = g_freed_blocks = 0;
    g_budget_known = 0;
    texmem_open(&k_ops);
}

static int overlap(uint64_t a, uint64_t an, uint64_t b, uint64_t bn) { return a < b + bn && b < a + an; }

static void case_ring(void) {
    open_fresh();
    TexSlot s[8];
    uint64_t sizes[3] = {10 * MiB, 20 * MiB + 5, 3 * MiB};
    // submission 1: small regions share a chunk, in order and aligned, without overlap
    for (int i = 0; i < 3; i++) {
        texring_take(1, sizes[i], TEXRING_ALIGN, &s[i]);
        CHECK(s[i].offset % TEXRING_ALIGN == 0 && s[i].address == s[i].chunk->address + s[i].offset);
        for (int k = 0; k < i; k++) CHECK(!overlap(s[i].address, sizes[i], s[k].address, sizes[k]));
    }
    CHECK(s[0].chunk == s[1].chunk && s[1].chunk == s[2].chunk);
    TexMemStats st;
    texmem_stats(&st);
    CHECK(st.ring_chunks == 1 && st.ring_takes == 3);
    // submission 2 while 1 is pending: a chunk of its own, not 1's bytes
    TexSlot t;
    texring_take(2, 5 * MiB, TEXRING_ALIGN, &t);
    CHECK(t.chunk != s[0].chunk);
    // submission 1 is done: its chunk comes back from the start
    g_done = 1;
    TexSlot u;
    texring_take(3, 7 * MiB, TEXRING_ALIGN, &u);
    CHECK(u.chunk == s[0].chunk && u.offset == 0);
    texmem_close();

    // the cap: two submissions that each want 200 MiB in regions larger than a chunk
    open_fresh();
    TexSlot a[2], b[2];
    for (int i = 0; i < 2; i++) texring_take(1, 100 * MiB, TEXRING_ALIGN, &a[i]);
    texmem_stats(&st);
    CHECK(st.ring_bytes <= TEXRING_CAP);
    texring_take(2, 100 * MiB, TEXRING_ALIGN, &b[0]);     // 300 MiB of chunks: under the cap
    CHECK(g_waited == 0);
    for (int i = 0; i < 2; i++) CHECK(!overlap(b[0].address, 100 * MiB, a[i].address, 100 * MiB));
    texring_take(2, 100 * MiB, TEXRING_ALIGN, &b[1]);     // 400 MiB would pass the cap: submission 1 is waited for
    CHECK(g_waited == 1);
    texmem_stats(&st);
    CHECK(st.ring_waits == 1);
    CHECK(st.ring_peak <= TEXRING_CAP);
    CHECK(b[1].address == a[0].address || b[1].address == a[1].address);   // it took a chunk of submission 1's
    CHECK(!overlap(b[1].address, 100 * MiB, b[0].address, 100 * MiB));
    // many submissions of 150 MiB each: the ring never holds more than the cap
    for (uint64_t v = 3; v < 40; v++) {
        TexSlot x, y;
        texring_take(v, 150 * MiB, TEXRING_ALIGN, &x);
        texring_take(v, 150 * MiB, TEXRING_ALIGN, &y);
        CHECK(!overlap(x.address, 150 * MiB, y.address, 150 * MiB));
    }
    texmem_stats(&st);
    CHECK(st.ring_peak <= TEXRING_CAP && st.ring_waits > 3);
    texmem_close();
}

static void case_ring_full(void) {
    open_fresh();
    TexSlot s;
    texring_take(1, 200 * MiB, TEXRING_ALIGN, &s);
    texring_take(1, 150 * MiB, TEXRING_ALIGN, &s);
    texring_take(1, 100 * MiB, TEXRING_ALIGN, &s);   // 450 MiB in one submission: nothing earlier to wait for
    printf("returned\n");
}

static void case_ring_big(void) {
    open_fresh();
    TexSlot s;
    texring_take(1, TEXRING_CAP + 1, TEXRING_ALIGN, &s);
    printf("returned\n");
}

// An image of `size` bytes at `align` of any memory type.
static TexExtent take(uint64_t size, uint64_t align, uint32_t bits) {
    TexExtent e;
    texarena_alloc(size, align, bits, &e);
    CHECK(e.offset % align == 0);
    return e;
}

static void case_arena(void) {
    open_fresh();
    TexMemStats st;
    // extents in one block: aligned, disjoint, and the block fills exactly
    TexExtent e[64];
    int n = 0;
    uint64_t at = 0;
    for (; n < 64; n++) {
        uint64_t size = (n % 3 + 1) * 256 * 1024 + 4096, align = n % 2 ? 65536 : 4096;
        if (at + size + 65536 > TEXARENA_BLOCK) break;
        e[n] = take(size, align, 1);
        at = e[n].offset + size;
        for (int k = 0; k < n; k++) CHECK(!overlap(e[n].offset, e[n].size, e[k].offset, e[k].size) || e[n].blk != e[k].blk);
    }
    texmem_stats(&st);
    CHECK(st.arena_blocks == 1 && st.arena_bytes == TEXARENA_BLOCK);
    // freeing in a scrambled order (the even ones up, the odd ones down) joins everything back: the whole block is one
    // extent again, which a request of the block's size proves without a second block
    for (int i = 0; i < n; i += 2) texarena_free(&e[i]);
    for (int i = (n - 1) | 1; i >= 1; i -= 2)
        if (i < n) texarena_free(&e[i]);
    texmem_stats(&st);
    CHECK(st.arena_used == 0);
    TexExtent whole = take(TEXARENA_BLOCK, 1, 1);
    texmem_stats(&st);
    CHECK(st.arena_blocks == 1 && st.arena_blocks_made == 1 && whole.offset == 0);
    texarena_free(&whole);
    texmem_close();

    open_fresh();
    TexExtent all[200];
    int m = 0;
    for (; m < 200; m++) all[m] = take(1 * MiB + (uint64_t)m * 4096, 65536, 1);
    texmem_stats(&st);
    CHECK(st.arena_used >= 200 * MiB && st.arena_blocks >= 4);
    for (int i = 0; i < m; i += 2) texarena_free(&all[i]);
    for (int i = 1; i < m; i += 2) texarena_free(&all[i]);
    texmem_stats(&st);
    CHECK(st.arena_used == 0);
    // all free again: a request that was larger than a block's free run before fits one block after the joins
    TexExtent big = take(60 * MiB, 65536, 1);
    CHECK(big.size == 60 * MiB);
    texarena_free(&big);
    // larger than a block: a block of its own
    texmem_stats(&st);
    uint64_t before = st.arena_blocks_made;
    TexExtent huge = take(100 * MiB, 65536, 1);
    texmem_stats(&st);
    CHECK(st.arena_blocks_made == before + 1 && huge.size == 100 * MiB);
    texarena_free(&huge);
    texmem_close();

    // the cap: 60 MiB images, one per block; 54 blocks of 64 MiB are 3,456 MiB, 55 would pass 3.4 GiB
    open_fresh();
    TexExtent held[54];
    for (int i = 0; i < 54; i++) held[i] = take(60 * MiB, 65536, 1);
    texmem_stats(&st);
    CHECK(st.arena_blocks == 54 && st.arena_bytes <= TEXARENA_CAP && st.arena_bytes_peak <= TEXARENA_CAP);
    // one freed: the next image takes its place with no new block
    texarena_free(&held[10]);
    held[10] = take(60 * MiB, 65536, 1);
    texmem_stats(&st);
    CHECK(st.arena_blocks == 54 && st.arena_blocks_made == 54);
    for (int i = 0; i < 54; i++) texarena_free(&held[i]);
    texmem_close();

    // empty blocks of memory type 0 go back to make room for a block of type 1, at the cap
    open_fresh();
    for (int i = 0; i < 54; i++) held[i] = take(60 * MiB, 65536, 1);
    for (int i = 0; i < 54; i++) texarena_free(&held[i]);
    TexExtent other = take(60 * MiB, 65536, 2);   // an image of types {1}: no block of type 0 serves it
    texmem_stats(&st);
    CHECK(other.type == 1 && st.arena_reclaims > 0 && st.arena_bytes <= TEXARENA_CAP);
    texarena_free(&other);
    texmem_close();

    // the budget: a block is made only when used + block + an eighth of the budget fits, and this is the edge
    open_fresh();
    g_budget_known = 1, g_budget_total = 1000 * MiB, g_budget_used = 1000 * MiB - TEXARENA_BLOCK - 125 * MiB;
    TexExtent edge = take(10 * MiB, 4096, 1);
    texarena_free(&edge);
    texmem_close();
}

static void case_arena_full(void) {
    open_fresh();
    TexExtent held[55];
    for (int i = 0; i < 55; i++) held[i] = take(60 * MiB, 65536, 1);
    printf("returned\n");
}

static void case_arena_budget(void) {
    open_fresh();
    g_budget_known = 1, g_budget_total = 1000 * MiB, g_budget_used = 1000 * MiB - TEXARENA_BLOCK - 125 * MiB + 1;
    TexExtent e = take(10 * MiB, 4096, 1);   // one byte past the edge: used + block + budget / 8 passes the budget
    (void)e;
    printf("returned\n");
}

int main(int argc, char **argv) {
    if (argc != 2) {
        fprintf(stderr, "usage: texmem_test <case>\n");
        return 2;
    }
    if (!strcmp(argv[1], "ring")) case_ring();
    else if (!strcmp(argv[1], "arena")) case_arena();
    else if (!strcmp(argv[1], "ring_full")) case_ring_full();
    else if (!strcmp(argv[1], "ring_big")) case_ring_big();
    else if (!strcmp(argv[1], "arena_full")) case_arena_full();
    else if (!strcmp(argv[1], "arena_budget")) case_arena_budget();
    else {
        fprintf(stderr, "unknown case %s\n", argv[1]);
        return 2;
    }
    printf("ok %s\n", argv[1]);
    return 0;
}
