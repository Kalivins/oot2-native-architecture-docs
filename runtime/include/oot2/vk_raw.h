// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The raw path of the Vulkan back end (architecture D15): the page table the translator's raw-access helper walks, built
// from the backing map's runs (oot2/backing.h, The raw path) in the layout of oot2/raw_table.h, and what a raw-access
// submission needs around it.
//
// The table. One leaf entry per 4 KiB page a run maps, (device address - guest address) | 1; every other entry 0, so an
// unmapped page reads zero, drops its write and takes a miss log entry. The span runs over every pool in the map (a
// backing is inside a pool, so the map cannot grow out of it). The table is built again, whole, when the map's version
// moved since the copy in use was built; a copy a submission in flight may read is never written, except in the one case
// below, so the table has a few copies (RAW_COPIES_MAX), each in device-local host-visible memory where the device has it
// (the raw-address study: there the walk ran at a direct pointer's speed), else in host-visible memory. Building again
// rather than editing in place is what keeps both of the Hookshot survey's checks true by construction: a backing that
// left the map leaves no entry, and one whose range a newer backing took leaves the newer entries alone. A build costs one
// write of the table (8 bytes per mapped page): see the raw page table study.
//
// A submission (gpu_vk_raw_begin, at its first raw-access stage; gpu_vk_raw_end at its end). The begin keeps the map's
// invariant (backing_raw_begin), takes a current copy and returns the push words 0 to 3 for every raw-access stage of
// the submission. The stages take the copy's address and span as they are recorded, but the device reads the table only
// after the submission is submitted, and the binds between the first stage and the end can move the map (an import over
// pages a bind commits, a mirror cut back for a bind that takes pages from it): so the end writes the map as it is
// then into the same copy, over the same span, when its version moved since the copy was built (refresh), after the
// earlier submissions that read the copy are done. Every raw access of the submission then reaches the map as the
// submission leaves it, whichever stage recorded it. The end records, in the submission's epilogue (after its own work),
// the runtime shader raw_writeback.comp, which copies the table's header and miss log out, zeroes the miss count, and
// for each mirrored page the table maps whose store bit is set clears the bit and copies the page out (up to
// RAW_WRITEBACK_PAGES of them; more is a named fatal at the effect); then queues an effect that, once the device
// reached the value and before it is done,
// logs the misses by address, kind and reason, and writes each copied page into guest memory, stamped as written
// (oot2/pagegen.h). So a raw store into a mirror reaches the CPU when the value is done, as the descriptor path's writes
// do (backing.h, Mirrors), and the next submission's misses are its own. Store bits of pages that are not mirrored are
// left set: the helper reads a bit before it sets it, so a set bit costs it nothing.
//
// One submission is recorded at a time (the submission gate of oot2/gpu.h), so begin and end need no lock of their own;
// the effect takes the out buffers' lock only.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "oot2/backing.h"
#include "oot2/raw_table.h"

#define RAW_COPIES_MAX 4             // copies of the table; a fifth with every copy in a submission not issued is fatal
#define RAW_WRITEBACK_PAGES 32768    // mirrored pages one submission's raw stores may reach (128 MiB of write-back)
// A promotion (backing.h) keeps the mirrors within the write-back, so the stores of a submission cannot reach more pages than it holds.
_Static_assert(BACKING_PROMOTE_MIRROR_MAX / BACKING_HOST_PAGE <= RAW_WRITEBACK_PAGES, "the write-back must hold every promoted mirror");

// ---- the builder (no device: runtime/gpu/raw_table.c)
typedef struct RawLayout {
    uint32_t span_base, span_count;   // push words 2 and 3
    uint32_t blocks;                  // leaf blocks in use
    uint64_t words;                   // the table's size in 32-bit words
    uint64_t pages, mirror_pages;     // page entries the runs write; those of mirrors
} RawLayout;
// The layout of runs' table: 1, or 0 with the reason in why (more leaf blocks than RAW_LEAF_BLOCK_CAPACITY, a delta that
// is not a multiple of 4 KiB, a run that is not whole pages).
int raw_layout(const BackingRun *runs, int n, RawLayout *l, char *why, size_t why_size);
// The same over a span the caller chose, the chunks span_base to span_base + span_count - 1 of 2 MiB (a span_count of 0
// takes the span from the runs): a table rewritten later keeps the push words stages already took, so it keeps the span,
// and a run that reaches outside it is refused with its name.
int raw_layout_span(const BackingRun *runs, int n, uint32_t span_base, uint32_t span_count, RawLayout *l, char *why,
                    size_t why_size);
// A mirrored page as the write-back shader reads it: its leaf slot (its store bit) and the device address of its bytes.
typedef struct RawMirrorPage {
    uint32_t slot, pad;
    uint64_t address;
} RawMirrorPage;
// Write the table of runs, laid out as l: every one of its l->words words (unmapped entries and the logs zero), the
// mirrored pages into pages and the guest addresses their bytes stand for into guest (l->mirror_pages of each; both may
// be NULL when there are none). Runs are written in order: a later run's page replaces an earlier one's.
void raw_write(const BackingRun *runs, int n, const RawLayout *l, uint32_t *words, RawMirrorPage *pages, uint64_t *guest);

// ---- the device (runtime/gpu/vk_raw.c)
// Submission value holds a raw-access program: the push words 0 to 3 (RAW_PUSH_*) for it. The first call in a
// submission does the work; later ones return the same words.
void gpu_vk_raw_begin(uint64_t value, uint32_t push[4]);
// The end of submission value, on its thread before backing_submit_end and gpu_vk_end: nothing when it never began.
void gpu_vk_raw_end(uint64_t value);
// gpu_vk_close, once every submission is done: the copies, the out buffers and the shader go.
void gpu_vk_raw_close(void);

typedef struct RawMiss {
    uint64_t address;
    uint32_t kind, reason;   // RAW_MISS_KIND_*, RAW_MISS_REASON_*
} RawMiss;
typedef struct GpuVkRawStats {
    uint64_t submissions, builds, build_ns, build_ns_max;   // raw-access submissions; table builds and their time
    uint64_t refreshes, refresh_ns, refresh_waits;          // tables rewritten at a submission's end; their time; waits for readers
    uint64_t copies, copy_waits, table_bytes;               // copies made; builds that waited for one; the newest's size
    uint64_t pages, mirror_pages, blocks, span_count;        // the newest build's
    uint64_t writebacks, writeback_bytes, writebacks_skipped;   // raw stores into mirrors written back
    uint64_t misses, miss_submissions;                      // misses counted by the helper; submissions with any
    uint64_t last_miss_value;                               // the newest submission with misses
    uint32_t last_misses;                                   // its entries in last_miss (at most RAW_MISS_ENTRIES)
    RawMiss last_miss[RAW_MISS_ENTRIES];
} GpuVkRawStats;
void gpu_vk_raw_stats(GpuVkRawStats *out);
