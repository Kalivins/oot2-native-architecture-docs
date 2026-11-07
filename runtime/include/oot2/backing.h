// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The backing map of the Vulkan back end (architecture D14): what memory the GPU sees behind every GPU address the game
// can bind, decided per page, and the resolver that turns a bound range into a buffer and an offset for the descriptor
// path (vertex, index, uniform, storage and indirect bindings, copies). The raw path (D15) will read the same decisions.
// Nothing here has a Vulkan type: memory comes from a BackingOps table (runtime/gpu/vk_backing.c on a device, a test's
// stand-ins without one), and its lifetime follows the timeline of oot2/gpu.h.
//
// Pools are named by their GPU range, as gfx.h names them. A page of a pool is, from the first bind that reaches it:
//   extent   VRAM. A pool the CPU cannot see (CPU_NO_ACCESS, physical) gets one extent over the whole pool at its first
//            bind, so no bind inside the pool spans two extents; a pool no bind reaches (render targets, which are
//            images) gets none. Up to BACKING_BLOCK bytes it is a part of a VRAM block; past that, a block of its own.
//   import   host memory imported over committed guest pages: a pool the CPU can see (BACKING_RULE).
//   mirror   VRAM kept current from the guest's pages: the coherence ticket's hook; not made here.
//   none     before that first bind, and for pages of a pool the CPU can see that no import covers.
// A page of a virtual pool is the page of the physical pool its one run maps it to (the front end allows one run per
// virtual pool, research 33 decision 4); a bind over an unmapped page is a named fatal.
//
// The page is 4 KiB in a pool the CPU can see: the page of commit and of host import, and NVN aligns pool storage to
// 4 KiB only, so one 64 KiB page can hold two pools. In the GPU address window (gfx.h) it is 64 KiB.
//
// Imports. A host import covers committed pages only: one over an uncommitted page fails, and after a failed import the
// driver refuses every import of that size or larger until a smaller one succeeds (research 14 F12, research 48 B1, B7).
// Imports are cut to windows of BACKING_IMPORT_WINDOW aligned in the address space, and on Windows the guest's watched
// regions are reserved in pieces of that size (runtime/core/plat.c), so an import walks no further than its window
// (research 48 answer 4: the driver walks the run of committed pages that follows the pointer, to the reservation's end).
// A cell is a pool cut to the windows one bind touches: one window, or two for a bind across a boundary. The bind's own
// pages are committed first, as host I/O commits its target (rt_mem_commit_io), then the cell's import covers the run of
// committed pages around the bind inside the cell (rt_mem_commit_run). A later bind the import does not cover replaces it
// with the larger run. Imports of one page alias the same host memory, so a page in two cells reads the same bytes
// through both.
//
// Textures. A decision is taken at a page's first bind, never at pool_init or texture_register: a pool that only holds
// textures is never bound as a buffer and is never imported (D14, D19). A texture registered over pages a bind already
// imported leaves them imported, and those pages cannot be released by lever L1 while the import holds them
// (backing_page tells the texture ticket which pages are imported).
//
// Retirement. A resolve stamps its pool and the backing it used with the submission's value. A backing that leaves the
// map (replaced by a larger import, over pages the guest decommitted, or its pool finalized) waits in a deferred-release
// queue until the newest value stamped on it is done; the queue holds BACKING_DEFERRED_MAX backings and a full queue
// waits for its oldest. pool_fini returns once every value stamped on its pool is done (gfx.h): the front end hands the
// range to the next pool at once. A pool no submission used returns at once.
//
// The commit epoch (oot2/memory.h). backing_sync, called before a submission, checks every cell again when the epoch
// moved since its last call: an import grows over chunks committed since inside its cell (D15's invariant: every
// committed page a pointer can reach is mapped before a raw-access submission), and an import over pages decommitted
// since is retired. It also frees what retired.
//
// VRAM blocks are made off the frame path: a thread of this module keeps BACKING_SPARE_BLOCKS empty blocks ready, so an
// extent takes its memory from a block made earlier (vkAllocateMemory took 8 to 19 ms per block in DKB Recomp,
// runtime/vk_headroom.c, whose idea this follows; no code is taken from it). A block made on the resolving thread
// because none was ready, and every block of its own, is counted and logged.
//
// Locking: one reader-writer lock. A resolve that finds its backing holds it shared; one that makes a backing holds it
// exclusively, and imports are made under it. Every call may come from any thread.
#pragma once
#include <stdint.h>

#include "oot2/gfx.h"

// ---- the rule (one function, its constants named: spike S5 and research 14 A7 move them)
#define BACKING_IMPORT_WINDOW (64ull << 20)   // research 48: a bounded 64 MiB import costs 0.8 ms to make, 1.6 ms to free
#define BACKING_IMPORT_PROBE 0x1000ull        // research 48 B7: the small import that clears a failed one
#define BACKING_BLOCK (256ull << 20)          // a VRAM block; an extent larger than this has a block of its own
#define BACKING_SPARE_BLOCKS 1                // empty blocks kept ready by the block thread
#define BACKING_DEFERRED_MAX 1024             // backings waiting for retirement; a full queue waits for its oldest
#define BACKING_EXTENT_ALIGN 0x10000ull       // extents are 64 KiB aligned in their block, as window pages are
#define BACKING_HOST_PAGE 0x1000ull           // the page of commit and of host import
enum { BACKING_NONE, BACKING_EXTENT, BACKING_IMPORT, BACKING_MIRROR };

// What a bind asks for: its range and its pool's flags (NVNmemoryPoolFlags).
typedef struct BackingAsk {
    uint64_t addr, size;
    uint32_t pool_flags;
} BackingAsk;
// BACKING_EXTENT for a pool the CPU cannot see; for one it can, BACKING_MIRROR when the mirror hook says so, else
// BACKING_IMPORT. The hook belongs to the coherence ticket (D14's mirrors); until it is set, the answer is an import.
int backing_rule(const BackingAsk *ask);
void backing_set_mirror_hook(int (*want_mirror)(const BackingAsk *ask));

// ---- the memory behind backings
typedef struct BackingMem {
    uint64_t buffer;    // the buffer over the whole allocation (a VkBuffer handle on a device)
    uint64_t memory;    // the allocation (a VkDeviceMemory handle on a device); the ops' own
    uint64_t address;   // the buffer's device address
    uint64_t size;
} BackingMem;
typedef struct BackingOps {
    // A VRAM block of size bytes with one buffer over it: 1, or 0 when the device has no room (a named fatal follows).
    int (*block_new)(uint64_t size, BackingMem *out);
    // Committed host pages [host, host + size), 4 KiB aligned, imported: 1, or 0 when the device refused the import.
    int (*import)(uint64_t host, uint64_t size, BackingMem *out);
    // Free an allocation no submission can use any more.
    void (*free)(const BackingMem *m);
} BackingOps;

// Open the map over ops (the Vulkan back end at gpu_vk_open, after its timeline started; a test with stand-ins), and
// start the block thread. Close frees every backing and block once every submission is done (gpu_sync_finish first),
// and stops the thread; backing_open may follow. Neither may run while another thread uses the map.
void backing_open(const BackingOps *ops);
void backing_close(void);
int backing_is_open(void);

// ---- the back end's pool operations (gfx.h's contract)
int backing_pool_init(const pool_t *p);
void backing_pool_fini(const pool_t *p);
int backing_pool_map_virtual(const pool_t *v, int n, const nvn_mapping_t *requests);

// ---- the resolver
// The backing of [addr, addr + size) (size 0 counts as 1) for the submission with the given value (gpu_sync_begin's;
// 0 for none): its buffer, the offset of addr in it and addr's device address. A range that is not wholly inside one
// pool, or that would need two backings (pages of a virtual pool in no run), is a named fatal that prints its shape.
typedef struct BackingRef {
    uint64_t buffer, offset, address;
    int kind;   // BACKING_EXTENT or BACKING_IMPORT
} BackingRef;
void backing_resolve(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out);
// Before a submission: grow and retire imports after the commit epoch moved, and free what retired.
void backing_sync(void);

// ---- the decision of one page, for tests and for the raw path's page table: the kind, and when it has a backing, the
// device address of addr and, for a page of a virtual pool, the physical GPU address it maps to (else addr).
typedef struct BackingPage {
    int kind;
    uint64_t address, gpu;
} BackingPage;
void backing_page(uint64_t addr, BackingPage *out);

typedef struct BackingStats {
    uint64_t pools, extents, extent_bytes, imports, import_bytes;     // live now
    uint64_t blocks, block_bytes, spare_blocks;                       // VRAM blocks live now, empty ones ready
    uint64_t imports_made, imports_refused, imports_retried, imports_grown, imports_stale;
    uint64_t blocks_made_ahead, blocks_made_inline, blocks_dedicated;  // by the block thread; on a resolving thread
    uint64_t deferred, deferred_peak, deferred_full_waits, freed;      // the deferred-release queue
    uint64_t fini_waits, resolves, resolve_misses, syncs, sync_rescans;
} BackingStats;
void backing_stats(BackingStats *out);
