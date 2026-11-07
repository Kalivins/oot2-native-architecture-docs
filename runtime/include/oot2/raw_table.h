/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (C) 2026 The oot2-native contributors */
#ifndef OOT2_RAW_TABLE_H
#define OOT2_RAW_TABLE_H

/*
 * The page table the translator's raw-access helper walks (architecture D15), as the runtime builds it.
 * The translator's copy is xlate/third_party/ryujinx/src/Ryujinx.Graphics.Shader/RawTableAbi.cs, which
 * documents the layout; tests/test_raw_abi.py fails when the two differ.
 *
 * Push constants, bytes 0 to 15: words 0 and 1 the table's device address (low, high), word 2 the first
 * chunk of the span (guest address >> 21), word 3 the number of chunks in the span (at least 1).
 * The table, in 32-bit words from its device address: header, miss log, store bits, top level (one word
 * per chunk of the span), then leaf blocks. The header holds the version, the span base and count
 * (copies of push words 2 and 3), the number of leaf blocks in use and the miss count. A top-level word
 * is 0 for an unmapped chunk, else the leaf block n (1 to RAW_LEAF_BLOCK_CAPACITY). A leaf entry is 2
 * words, the page's delta from guest to device address; the page is mapped when bit 0 of the low word
 * is set (a clear bit 0 is an unmapped page, whatever the other bits hold). A mapped entry is
 * (delta | 1): the delta is a multiple of 4 KiB, so its own bit 0 is clear, and the helper clears bit 0
 * before it adds the delta.
 *
 * The helper trusts the table: the runtime refuses a map that needs more than RAW_LEAF_BLOCK_CAPACITY leaf
 * blocks (a larger top-level word would put the store-bit word past the bit array, into the table's own
 * entries), writes a span count of at least 1, stores every delta as a multiple of 4 KiB with bit 0 set,
 * and allocates every word up to the last leaf block.
 */

#define RAW_VERSION 1

#define RAW_PUSH_TABLE_LOW 0
#define RAW_PUSH_TABLE_HIGH 1
#define RAW_PUSH_SPAN_BASE 2
#define RAW_PUSH_SPAN_COUNT 3
#define RAW_PUSH_BYTES 16

#define RAW_CHUNK_LOG2 21
#define RAW_PAGE_LOG2 12
#define RAW_LEAF_ENTRIES_LOG2 9
#define RAW_ENTRY_WORDS 2
#define RAW_LEAF_BLOCK_WORDS 1024

#define RAW_HEADER_VERSION 0
#define RAW_HEADER_SPAN_BASE 1
#define RAW_HEADER_SPAN_COUNT 2
#define RAW_HEADER_LEAF_BLOCKS 3
#define RAW_HEADER_MISS_COUNT 4
#define RAW_HEADER_WORDS 16

#define RAW_MISS_LOG_WORD 16
#define RAW_MISS_ENTRIES 64
#define RAW_MISS_ENTRY_WORDS 4

#define RAW_MISS_ADDRESS_LOW 0
#define RAW_MISS_ADDRESS_HIGH 1
#define RAW_MISS_KIND 2
#define RAW_MISS_REASON 3

#define RAW_MISS_KIND_LOAD 1
#define RAW_MISS_KIND_STORE 2
#define RAW_MISS_KIND_ATOMIC 3

#define RAW_MISS_REASON_OUTSIDE_SPAN 1
#define RAW_MISS_REASON_NO_CHUNK 2
#define RAW_MISS_REASON_NO_PAGE 3

#define RAW_LEAF_BLOCK_CAPACITY 16384
#define RAW_STORE_BITS_WORD 272
#define RAW_STORE_BITS_WORDS 262144
#define RAW_TOP_WORD 262416

#endif
