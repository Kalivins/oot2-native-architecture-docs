// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The backing map (oot2/backing.h, architecture D14): pool records in an index sorted by GPU address, their backings
// (a VRAM extent per pool the CPU cannot see, import cells for a pool it can see, one run for a virtual pool), the VRAM
// blocks extents are cut from and the thread that makes them ahead of use, and the deferred-release queue.
//
// The index is the front end's (runtime/nvn/nvn_pool_index.h), so a nested or overlapping pool is still found. A
// resolve that finds its backing holds the lock shared and stamps the submission's value on its pools and its backing
// with an atomic maximum; one that must make a backing takes the lock exclusively and looks again. A backing leaves the
// map only under the exclusive lock, so a stamp made under the shared lock is seen by whoever removes it. Memory is
// freed outside the lock (freeing a 64 MiB import takes about 1.6 ms, research 48).
//
// A cell holds the imports of the separate runs of committed pages that binds reached inside it, sorted and disjoint.
// A bind outside them commits its own pages, then takes the whole run of committed pages around it: every import of the
// cell inside that run is replaced by one import over the run.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/gpu.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"
#include "plat.h"
#include "../nvn/nvn_pool_index.h"

#define WINDOW BACKING_IMPORT_WINDOW
#define HPAGE BACKING_HOST_PAGE
#define CPU_NO_ACCESS NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS
#define GPU_NO_ACCESS NVN_MEMORY_POOL_FLAGS_GPU_NO_ACCESS
#define PHYSICAL NVN_MEMORY_POOL_FLAGS_PHYSICAL
#define VIRTUAL NVN_MEMORY_POOL_FLAGS_VIRTUAL
#define BLOCK_THREAD_STACK (256u << 10)

_Static_assert(BACKING_IMPORT_WINDOW == PLAT_PIECE, "an import window is one piece of the watched reservations");
_Static_assert(BACKING_BLOCK % BACKING_EXTENT_ALIGN == 0, "blocks hold whole extent pages");

typedef unsigned long long ull;

enum { CLS_CPU, CLS_GPU, CLS_VIRTUAL };   // a pool the CPU sees; one in the window with storage; a virtual one

// A VRAM block: one allocation and its buffer, cut into extents by a first-fit list of free spans (offsets).
typedef struct Span { uint64_t lo, hi; } Span;
typedef struct Block {
    BackingMem mem;
    uint64_t used;
    int dedicated;   // the block of one extent larger than BACKING_BLOCK
    Span *free;
    int nfree, cap;
} Block;

// An extent (a part of a block) or an import. mem is the whole buffer; offset is where gpu_lo lies in it.
typedef struct Backing {
    int kind;
    uint64_t gpu_lo, gpu_hi;
    uint64_t last_use;   // atomic: the newest submission value that resolved into it
    BackingMem mem;
    uint64_t offset;
    Block *block;        // an extent's
} Backing;

// A pool the CPU sees cut to the windows [k1, k2] one bind touched, and the imports binds made inside it.
typedef struct Cell {
    uint64_t k1, k2, lo, hi;
    Backing **imp;   // sorted by address, disjoint
    int nimp, cap;
} Cell;

typedef struct Rec {
    uint64_t lo, hi;
    uint32_t flags;
    int cls;
    uint64_t last_use;      // atomic
    Backing *ext;           // CLS_GPU: the extent, made at the first bind
    Cell *cells;            // CLS_CPU: sorted by (k1, k2)
    int ncells, cap;
    struct Rec *run;        // CLS_VIRTUAL: the physical pool of the one run (NULL: none mapped)
    int64_t run_delta;      // physical GPU address minus virtual GPU address
    int run_lost;           // the run's physical pool was finalized while pages were mapped
    uint64_t *bits, pages, mapped;
} Rec;

static PlatLock g_lock = PLAT_LOCK_INIT;
static int g_open;
static BackingOps g_ops;
static pidx g_index;
static Rec **g_virtual;   // the virtual pools, to cut their runs when a physical pool goes
static int g_nvirtual, g_virtual_cap;
static Block **g_blocks;
static int g_nblocks, g_blocks_cap;
static uint64_t g_seen_epoch;
static BackingStats g_st;     // under g_lock
static uint64_t g_resolves, g_misses;   // atomic
static int (*g_mirror_hook)(const BackingAsk *ask);

// The deferred-release queue (g_lock).
typedef struct Deferred { Backing *b; uint64_t value; } Deferred;
static Deferred *g_def;
static int g_ndef, g_def_cap;

// The block thread.
static PlatMutex g_bt_lock = PLAT_MUTEX_INIT;
static PlatCond g_bt_cond = PLAT_COND_INIT;
static int g_bt_want, g_bt_stop, g_bt_running;
static PlatThread *g_bt;

static uint64_t up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
static uint64_t down(uint64_t v, uint64_t a) { return v & ~(a - 1); }

static void stamp(uint64_t *at, uint64_t value) {
    uint64_t old = __atomic_load_n(at, __ATOMIC_RELAXED);
    while (old < value && !__atomic_compare_exchange_n(at, &old, value, 1, __ATOMIC_RELEASE, __ATOMIC_RELAXED)) {
    }
}

static uint64_t used_by(const Backing *b) { return __atomic_load_n(&b->last_use, __ATOMIC_ACQUIRE); }

static void *grow(void *p, int *cap, int need, size_t size, const char *what) {
    if (need <= *cap) return p;
    int c = *cap ? *cap : 16;
    while (c < need) c *= 2;
    void *q = realloc(p, (size_t)c * size);
    if (!q) rt_fatal("gfx: vulkan: backing map: out of memory for %s", what);
    *cap = c;
    return q;
}

// ---------- the rule
void backing_set_mirror_hook(int (*want_mirror)(const BackingAsk *ask)) { g_mirror_hook = want_mirror; }

int backing_rule(const BackingAsk *ask) {
    if (ask->pool_flags & CPU_NO_ACCESS) return BACKING_EXTENT;
    if (g_mirror_hook && g_mirror_hook(ask)) return BACKING_MIRROR;
    return BACKING_IMPORT;
}

// ---------- VRAM blocks (g_lock exclusive unless said)
static int spare_count(void) {
    int n = 0;
    for (int i = 0; i < g_nblocks; i++) n += !g_blocks[i]->dedicated && !g_blocks[i]->used;
    return n;
}

static Block *block_add(const BackingMem *mem, int dedicated) {
    Block *b = calloc(1, sizeof *b);
    if (!b) rt_fatal("gfx: vulkan: backing map: out of memory for a block");
    b->mem = *mem;
    b->dedicated = dedicated;
    b->free = grow(NULL, &b->cap, 1, sizeof *b->free, "a block's free list");
    b->free[0] = (Span){0, mem->size};
    b->nfree = 1;
    g_blocks = grow(g_blocks, &g_blocks_cap, g_nblocks + 1, sizeof *g_blocks, "the block list");
    g_blocks[g_nblocks++] = b;
    g_st.blocks++, g_st.block_bytes += mem->size;
    return b;
}

static void block_drop(Block *b) {
    for (int i = 0; i < g_nblocks; i++)
        if (g_blocks[i] == b) {
            g_blocks[i] = g_blocks[--g_nblocks];
            break;
        }
    g_st.blocks--, g_st.block_bytes -= b->mem.size;
    free(b->free);
    free(b);
}

// The lowest free span of b that holds size bytes, taken: its offset, or UINT64_MAX.
static uint64_t block_take(Block *b, uint64_t size) {
    for (int i = 0; i < b->nfree; i++) {
        if (b->free[i].hi - b->free[i].lo < size) continue;
        uint64_t at = b->free[i].lo;
        b->free[i].lo += size;
        if (b->free[i].lo == b->free[i].hi) {
            memmove(b->free + i, b->free + i + 1, (size_t)(b->nfree - i - 1) * sizeof *b->free);
            b->nfree--;
        }
        b->used += size;
        return at;
    }
    return UINT64_MAX;
}

static void block_give(Block *b, uint64_t lo, uint64_t size) {
    uint64_t hi = lo + size;
    int at = 0;
    while (at < b->nfree && b->free[at].lo < lo) at++;
    if ((at > 0 && b->free[at - 1].hi > lo) || (at < b->nfree && b->free[at].lo < hi))
        rt_fatal("gfx: vulkan: backing map: block part %#llx+%#llx given back twice", (ull)lo, (ull)size);
    int prev = at > 0 && b->free[at - 1].hi == lo, next = at < b->nfree && b->free[at].lo == hi;
    if (prev && next) {
        b->free[at - 1].hi = b->free[at].hi;
        memmove(b->free + at, b->free + at + 1, (size_t)(b->nfree - at - 1) * sizeof *b->free);
        b->nfree--;
    } else if (prev) {
        b->free[at - 1].hi = hi;
    } else if (next) {
        b->free[at].lo = lo;
    } else {
        b->free = grow(b->free, &b->cap, b->nfree + 1, sizeof *b->free, "a block's free list");
        memmove(b->free + at + 1, b->free + at, (size_t)(b->nfree - at) * sizeof *b->free);
        b->free[at] = (Span){lo, hi};
        b->nfree++;
    }
    b->used -= size;
}

static void bt_kick(void) {
    plat_mutex_lock(&g_bt_lock);
    g_bt_want = 1;
    plat_cond_signal(&g_bt_cond);
    plat_mutex_unlock(&g_bt_lock);
}

// The block thread: keeps BACKING_SPARE_BLOCKS empty blocks, making each without the map's lock.
static void bt_main(void *arg) {
    (void)arg;
    for (;;) {
        plat_mutex_lock(&g_bt_lock);
        while (!g_bt_want && !g_bt_stop) plat_cond_wait(&g_bt_cond, &g_bt_lock);
        int stop = g_bt_stop;
        g_bt_want = 0;
        plat_mutex_unlock(&g_bt_lock);
        if (stop) break;
        for (;;) {
            plat_lock_shared(&g_lock);
            int need = spare_count() < BACKING_SPARE_BLOCKS;
            plat_unlock_shared(&g_lock);
            if (!need) break;
            BackingMem m;
            if (!g_ops.block_new(BACKING_BLOCK, &m)) {
                rt_log("gfx: vulkan: backing map: the device has no room for a spare block of %llu MiB",
                       (ull)(BACKING_BLOCK >> 20));
                break;
            }
            // Retirement may have emptied a block while this one was made: it is kept only if still wanted.
            plat_lock(&g_lock);
            int keep = spare_count() < BACKING_SPARE_BLOCKS;
            if (keep) block_add(&m, 0), g_st.blocks_made_ahead++;
            plat_unlock(&g_lock);
            if (!keep) g_ops.free(&m);
        }
    }
    plat_mutex_lock(&g_bt_lock);
    g_bt_running = 0;
    plat_cond_broadcast(&g_bt_cond);
    plat_mutex_unlock(&g_bt_lock);
}

// A block made on the calling thread, because none was ready or the extent needs one of its own: counted and logged.
static Block *block_inline(uint64_t size, int dedicated, const char *what) {
    BackingMem m;
    uint64_t t0 = plat_now_ns();
    if (!g_ops.block_new(size, &m))
        rt_fatal("gfx: vulkan: backing map: the device has no room for a block of %llu MiB (%s)", (ull)(size >> 20),
                 what);
    Block *b = block_add(&m, dedicated);
    if (dedicated) g_st.blocks_dedicated++;
    else g_st.blocks_made_inline++;
    rt_log("gfx: vulkan: backing map: %s: a block of %llu MiB%s made on the resolving thread in %.1f ms", what,
           (ull)(size >> 20), dedicated ? " of its own" : " (no spare was ready)", (double)(plat_now_ns() - t0) / 1e6);
    return b;
}

// A new extent over the whole of rec's range (g_lock exclusive).
static Backing *extent_new(Rec *r) {
    uint64_t size = up(r->hi - r->lo, BACKING_EXTENT_ALIGN), at = UINT64_MAX;
    char what[96];
    snprintf(what, sizeof what, "the extent of pool [%#llx, %#llx)", (ull)r->lo, (ull)r->hi);
    Block *blk = NULL;
    if (size > BACKING_BLOCK) {
        blk = block_inline(size, 1, what);
        at = block_take(blk, size);
    } else {
        for (int i = 0; i < g_nblocks && at == UINT64_MAX; i++) {
            if (g_blocks[i]->dedicated) continue;
            int was_empty = !g_blocks[i]->used;
            at = block_take(g_blocks[i], size);
            if (at == UINT64_MAX) continue;
            blk = g_blocks[i];
            if (was_empty) bt_kick();   // a spare went into use
        }
        if (at == UINT64_MAX) {
            blk = block_inline(BACKING_BLOCK, 0, what);
            at = block_take(blk, size);
            bt_kick();
        }
    }
    Backing *b = calloc(1, sizeof *b);
    if (!b) rt_fatal("gfx: vulkan: backing map: out of memory for an extent");
    *b = (Backing){.kind = BACKING_EXTENT, .gpu_lo = r->lo, .gpu_hi = r->lo + size, .mem = blk->mem, .offset = at,
                   .block = blk};
    g_st.extents++, g_st.extent_bytes += size;
    return b;
}

// ---------- the deferred-release queue
static void defer(Backing *b) {
    if (!b) return;
    g_def = grow(g_def, &g_def_cap, g_ndef + 1, sizeof *g_def, "the deferred-release queue");
    g_def[g_ndef++] = (Deferred){b, used_by(b)};
    g_st.deferred = (uint64_t)g_ndef;
    if (g_st.deferred > g_st.deferred_peak) g_st.deferred_peak = g_st.deferred;
    if (b->kind == BACKING_EXTENT) g_st.extents--, g_st.extent_bytes -= b->gpu_hi - b->gpu_lo;
    else g_st.imports--, g_st.import_bytes -= b->gpu_hi - b->gpu_lo;
}

// What to free once the lock is dropped.
typedef struct Freeing { BackingMem *m; int n, cap; } Freeing;

static void to_free(Freeing *f, const BackingMem *m) {
    f->m = grow(f->m, &f->cap, f->n + 1, sizeof *f->m, "the free list");
    f->m[f->n++] = *m;
}

// Take every backing whose value is done (every one when all) out of the queue: an extent goes back to its block, and
// an empty block past the spares, a dedicated one and an import are freed after the lock. g_lock exclusive.
static void retire(Freeing *f, int all) {
    uint64_t done = gpu_sync_done();
    int k = 0;
    for (int i = 0; i < g_ndef; i++) {
        Deferred d = g_def[i];
        if (!all && d.value > done) {
            g_def[k++] = d;
            continue;
        }
        Backing *b = d.b;
        if (b->kind == BACKING_EXTENT) {
            Block *blk = b->block;
            block_give(blk, b->offset, b->gpu_hi - b->gpu_lo);
            if (!blk->used && (all || blk->dedicated || spare_count() > BACKING_SPARE_BLOCKS)) {
                to_free(f, &blk->mem);
                block_drop(blk);
            }
        } else {
            to_free(f, &b->mem);
        }
        g_st.freed++;
        free(b);
    }
    g_ndef = k;
    g_st.deferred = (uint64_t)k;
}

static void free_all(Freeing *f) {
    for (int i = 0; i < f->n; i++) g_ops.free(&f->m[i]);
    free(f->m);
    *f = (Freeing){0};
}

static void drain(void) {
    Freeing f = {0};
    plat_lock(&g_lock);
    retire(&f, 0);
    plat_unlock(&g_lock);
    free_all(&f);
}

// Wait until value is done, giving the guest's virtual core up meanwhile when the caller is a guest thread (D7).
static void wait_value(uint64_t value) {
    if (value <= gpu_sync_done()) return;
    int token = rt_kernel_block_begin();
    gpu_sync_wait(value, UINT64_MAX);
    rt_kernel_block_end(token);
}

// Hold the queue to BACKING_DEFERRED_MAX by waiting for its oldest value already issued: a value handed out and not
// yet submitted may be the calling thread's own submission, which must not wait for itself. No lock held.
static void bound(void) {
    for (;;) {
        plat_lock_shared(&g_lock);
        uint64_t issued = gpu_sync_issued(), oldest = UINT64_MAX;
        int over = g_ndef > BACKING_DEFERRED_MAX;
        for (int i = 0; over && i < g_ndef; i++)
            if (g_def[i].value <= issued && g_def[i].value < oldest) oldest = g_def[i].value;
        plat_unlock_shared(&g_lock);
        if (!over || oldest == UINT64_MAX) return;
        plat_lock(&g_lock);
        g_st.deferred_full_waits++;
        plat_unlock(&g_lock);
        wait_value(oldest);
        drain();
    }
}

// ---------- pool records (g_lock exclusive)
static const char *cls_name(const Rec *r) {
    if (r->cls == CLS_CPU) return (r->flags & PHYSICAL) ? "physical CPU-visible" : "CPU-visible";
    if (r->cls == CLS_VIRTUAL) return "virtual";
    return (r->flags & PHYSICAL) ? "physical CPU_NO_ACCESS" : "CPU_NO_ACCESS";
}

// The record of exactly [lo, hi), or NULL.
static Rec *rec_exact(uint64_t lo, uint64_t hi) {
    for (int i = pidx_upper(&g_index, (uintptr_t)lo) - 1; i >= 0 && g_index.e[i].lo == lo; i--)
        if (g_index.e[i].hi == hi) return (Rec *)g_index.e[i].key;
    return NULL;
}

static Rec *rec_find(uint64_t addr, uint64_t size) {
    const pidx_ent *e = pidx_find(&g_index, (uintptr_t)addr, size);
    return e ? (Rec *)e->key : NULL;
}

int backing_pool_init(const pool_t *p) {
    if (!g_open) rt_fatal("gfx: vulkan: backing map: a pool before the map is open");
    if (p->flags & GPU_NO_ACCESS) return 1;   // the GPU cannot reach it, as the front end's index leaves it out
    Rec *r = calloc(1, sizeof *r);
    if (!r) rt_fatal("gfx: vulkan: backing map: out of memory for a pool");
    r->lo = (uintptr_t)p->mem;
    r->hi = r->lo + p->size;
    r->flags = p->flags;
    r->cls = (p->flags & VIRTUAL) ? CLS_VIRTUAL : (p->flags & CPU_NO_ACCESS) ? CLS_GPU : CLS_CPU;
    if (r->cls == CLS_VIRTUAL) {
        r->pages = p->size / GFX_GPU_PAGE;
        r->bits = calloc((size_t)((r->pages + 63) / 64 + 1), sizeof *r->bits);
        if (!r->bits) rt_fatal("gfx: vulkan: backing map: out of memory for a virtual pool");
    }
    pidx_ent e = {.lo = (uintptr_t)r->lo, .hi = (uintptr_t)r->hi, .base = (uintptr_t)r->lo, .gpu = r->lo, .key = r};
    plat_lock(&g_lock);
    if (!pidx_insert(&g_index, &e)) rt_fatal("gfx: vulkan: backing map: out of memory for the pool index");
    if (r->cls == CLS_VIRTUAL) {
        g_virtual = grow(g_virtual, &g_virtual_cap, g_nvirtual + 1, sizeof *g_virtual, "the virtual pool list");
        g_virtual[g_nvirtual++] = r;
    }
    g_st.pools++;
    plat_unlock(&g_lock);
    return 1;
}

// Out of the map: its backings deferred with their own values; the newest value stamped on it or them is returned.
// g_lock exclusive.
static uint64_t rec_end(Rec *r) {
    pidx_remove(&g_index, r, NULL);
    uint64_t value = __atomic_load_n(&r->last_use, __ATOMIC_ACQUIRE);
    if (r->ext && used_by(r->ext) > value) value = used_by(r->ext);
    defer(r->ext);
    for (int i = 0; i < r->ncells; i++) {
        for (int j = 0; j < r->cells[i].nimp; j++) {
            Backing *b = r->cells[i].imp[j];
            if (used_by(b) > value) value = used_by(b);
            defer(b);
        }
        free(r->cells[i].imp);
    }
    for (int i = 0; i < g_nvirtual; i++) {
        if (g_virtual[i] == r) g_virtual[i--] = g_virtual[--g_nvirtual];
        else if (g_virtual[i]->run == r) g_virtual[i]->run = NULL, g_virtual[i]->run_lost = 1;
    }
    g_st.pools--;
    free(r->cells);
    free(r->bits);
    free(r);
    return value;
}

void backing_pool_fini(const pool_t *p) {
    if (p->flags & GPU_NO_ACCESS) return;
    uint64_t lo = (uintptr_t)p->mem, hi = lo + p->size;
    plat_lock(&g_lock);
    Rec *r = rec_exact(lo, hi);
    if (!r) rt_fatal("gfx: vulkan: backing map: pool_fini of [%#llx, %#llx), which pool_init never saw", (ull)lo, (ull)hi);
    uint64_t value = rec_end(r);
    int must_wait = value > gpu_sync_done();
    g_st.fini_waits += (uint64_t)must_wait;
    plat_unlock(&g_lock);
    if (must_wait) wait_value(value);
    drain();
    bound();
}

int backing_pool_map_virtual(const pool_t *v, int n, const nvn_mapping_t *req) {
    uint64_t lo = (uintptr_t)v->mem;
    plat_lock(&g_lock);
    Rec *r = rec_exact(lo, lo + v->size);
    int ok = r && r->cls == CLS_VIRTUAL;
    for (int i = 0; ok && i < n; i++) {
        uint64_t first = (uint64_t)req[i].virtual_offset / GFX_GPU_PAGE, count = req[i].size / GFX_GPU_PAGE;
        const pool_t *pp = (const pool_t *)(const void *)req[i].physical_pool;
        Rec *phys = NULL;
        if (pp) {
            phys = rec_exact((uintptr_t)pp->mem, (uintptr_t)pp->mem + pp->size);
            if (!phys || phys->cls == CLS_VIRTUAL || !(phys->flags & PHYSICAL)) {
                ok = 0;
                break;
            }
        }
        for (uint64_t k = first; k < first + count && k < r->pages; k++) {
            uint64_t m = 1ull << (k % 64);
            int was = (r->bits[k / 64] & m) != 0;
            if (pp) r->bits[k / 64] |= m;
            else r->bits[k / 64] &= ~m;
            if (pp && !was) r->mapped++;
            if (!pp && was) r->mapped--;
        }
        if (pp) {
            r->run = phys;
            r->run_delta = (int64_t)((uintptr_t)pp->mem + (uint64_t)req[i].physical_offset) -
                           (int64_t)(lo + (uint64_t)req[i].virtual_offset);
            r->run_lost = 0;
        } else if (!r->mapped) {
            r->run = NULL, r->run_delta = 0, r->run_lost = 0;
        }
    }
    plat_unlock(&g_lock);
    return ok;
}

// ---------- imports
// [*rs, *re): the run of committed pages inside [lo, hi) that holds the committed pages [plo, phi).
static void committed_run(uint64_t lo, uint64_t hi, uint64_t plo, uint64_t phi, uint64_t *rs, uint64_t *re) {
    uint64_t a = lo, start = lo, end = plo;
    while (a < plo) {
        int c;
        uint64_t n = rt_mem_commit_run(a, plo - a, &c);
        if (n == UINT64_MAX || !n)
            rt_fatal("gfx: vulkan: backing map: %#llx, in a pool the CPU can see, is not guest memory", (ull)a);
        if (!c) start = a + n;
        a += n;
    }
    while (end < hi) {
        int c;
        uint64_t n = rt_mem_commit_run(end, hi - end, &c);
        if (n == UINT64_MAX || !n || !c) break;
        end += n;
    }
    if (end < phi)
        rt_fatal("gfx: vulkan: backing map: [%#llx, %#llx) was committed for a bind and is not now", (ull)plo,
                 (ull)phi);
    *rs = start, *re = end;
}

// Import [lo, hi), committed pages. A refusal is retried once after a 4 KiB import of the committed page probe clears
// it (research 48 B7); a second refusal is a named fatal with the shape.
static Backing *import_new(uint64_t lo, uint64_t hi, uint64_t probe, const char *why) {
    BackingMem m;
    if (!g_ops.import(lo, hi - lo, &m)) {
        g_st.imports_refused++;
        BackingMem pm;
        int cleared = g_ops.import(probe, BACKING_IMPORT_PROBE, &pm);
        if (cleared) g_ops.free(&pm);
        g_st.imports_retried++;
        if (!cleared || !g_ops.import(lo, hi - lo, &m))
            rt_fatal("gfx: vulkan: backing map: the device refused to import committed host memory [%#llx, %#llx) "
                     "(%llu KiB) for %s, %s", (ull)lo, (ull)hi, (ull)((hi - lo) >> 10), why,
                     cleared ? "again after a 4 KiB import cleared the first refusal"
                             : "and a 4 KiB import of its first page was refused too");
    }
    Backing *b = calloc(1, sizeof *b);
    if (!b) rt_fatal("gfx: vulkan: backing map: out of memory for an import");
    *b = (Backing){.kind = BACKING_IMPORT, .gpu_lo = lo, .gpu_hi = hi, .mem = m};
    g_st.imports++, g_st.import_bytes += hi - lo, g_st.imports_made++;
    return b;
}

// The cell of r for windows [k1, k2], made when missing (and make is set).
static Cell *cell_get(Rec *r, uint64_t k1, uint64_t k2, int make) {
    int a = 0, b = r->ncells;
    while (a < b) {
        int m = (a + b) / 2;
        const Cell *c = &r->cells[m];
        if (c->k1 < k1 || (c->k1 == k1 && c->k2 < k2)) a = m + 1;
        else b = m;
    }
    if (a < r->ncells && r->cells[a].k1 == k1 && r->cells[a].k2 == k2) return &r->cells[a];
    if (!make) return NULL;
    r->cells = grow(r->cells, &r->cap, r->ncells + 1, sizeof *r->cells, "a pool's cells");
    memmove(r->cells + a + 1, r->cells + a, (size_t)(r->ncells - a) * sizeof *r->cells);
    uint64_t lo = k1 * WINDOW, hi = (k2 + 1) * WINDOW;
    r->cells[a] = (Cell){k1, k2, down(lo > r->lo ? lo : r->lo, HPAGE), up(hi < r->hi ? hi : r->hi, HPAGE), NULL, 0, 0};
    r->ncells++;
    return &r->cells[a];
}

// The import of c that holds addr, or NULL.
static Backing *cell_import(const Cell *c, uint64_t addr) {
    int a = 0, b = c->nimp;   // the first import that starts above addr
    while (a < b) {
        int m = (a + b) / 2;
        if (c->imp[m]->gpu_lo <= addr) a = m + 1;
        else b = m;
    }
    return a && addr < c->imp[a - 1]->gpu_hi ? c->imp[a - 1] : NULL;
}

// Put the import b over [rs, re) in c in place of every import inside that run, which are deferred. g_lock exclusive.
static void cell_put(Cell *c, uint64_t rs, uint64_t re, Backing *b) {
    int k = 0, at = -1, replaced = 0;
    for (int i = 0; i < c->nimp; i++) {
        Backing *o = c->imp[i];
        if (o->gpu_hi > rs && o->gpu_lo < re) {
            defer(o);
            replaced++;
            continue;
        }
        if (at < 0 && o->gpu_lo >= re) at = k;
        c->imp[k++] = o;
    }
    c->nimp = k;
    if (at < 0) at = k;
    c->imp = grow(c->imp, &c->cap, c->nimp + 1, sizeof *c->imp, "a cell's imports");
    memmove(c->imp + at + 1, c->imp + at, (size_t)(c->nimp - at) * sizeof *c->imp);
    c->imp[at] = b;
    c->nimp++;
    g_st.imports_grown += (uint64_t)replaced;
}

// ---------- the resolver
static void fill(BackingRef *out, const Backing *b, uint64_t addr) {
    out->buffer = b->mem.buffer;
    out->offset = b->offset + (addr - b->gpu_lo);
    out->address = b->mem.address + out->offset;
    out->kind = b->kind;
}

static int virtual_mapped(const Rec *r, uint64_t addr, uint64_t size) {
    for (uint64_t k = (addr - r->lo) / GFX_GPU_PAGE; k <= (addr + size - 1 - r->lo) / GFX_GPU_PAGE; k++)
        if (!(r->bits[k / 64] >> (k % 64) & 1)) return 0;
    return 1;
}

// The backing that holds [addr, addr + size) in r (not virtual), or NULL. g_lock, shared at least.
static Backing *backing_in(Rec *r, uint64_t addr, uint64_t size) {
    if (r->cls == CLS_GPU) return r->ext;
    const Cell *c = cell_get(r, addr / WINDOW, (addr + size - 1) / WINDOW, 0);
    Backing *b = c ? cell_import(c, addr) : NULL;
    return b && addr + size <= b->gpu_hi ? b : NULL;
}

// The backing of [addr, addr + size) when the map holds one, stamped; g_lock, shared at least.
static int lookup(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out) {
    Rec *r = rec_find(addr, size), *v = NULL;
    if (r && r->cls == CLS_VIRTUAL) {
        if (!r->run || !virtual_mapped(r, addr, size)) return 0;
        v = r;
        addr = (uint64_t)((int64_t)addr + r->run_delta);
        r = r->run;
        if (addr < r->lo || addr + size > r->hi) return 0;
    }
    Backing *b = r ? backing_in(r, addr, size) : NULL;
    if (!b) return 0;
    if (v) stamp(&v->last_use, value);
    stamp(&r->last_use, value), stamp(&b->last_use, value);
    fill(out, b, addr);
    return 1;
}

// Make what [addr, addr + size) needs, or stop by name with the shape. g_lock exclusive.
static void make(uint64_t addr, uint64_t size) {
    Rec *r = rec_find(addr, size);
    if (!r) {
        Rec *s = rec_find(addr, 1);
        if (s)
            rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) runs past the end of its %s pool [%#llx, "
                     "%#llx) by %#llx bytes: it would need two backings", (ull)addr, (ull)(addr + size), cls_name(s),
                     (ull)s->lo, (ull)s->hi, (ull)(addr + size - s->hi));
        rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) lies in no pool the GPU can reach", (ull)addr,
                 (ull)(addr + size));
    }
    if (r->cls == CLS_VIRTUAL) {
        if (r->run_lost)
            rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) in virtual pool [%#llx, %#llx), whose "
                     "physical pool was finalized while its pages were mapped", (ull)addr, (ull)(addr + size),
                     (ull)r->lo, (ull)r->hi);
        if (!r->run || !virtual_mapped(r, addr, size))
            rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) in virtual pool [%#llx, %#llx) reaches pages "
                     "no mapping request mapped (%llu of %llu pages mapped): it would need two backings", (ull)addr,
                     (ull)(addr + size), (ull)r->lo, (ull)r->hi, (ull)r->mapped, (ull)r->pages);
        addr = (uint64_t)((int64_t)addr + r->run_delta);
        r = r->run;
    }
    BackingAsk ask = {addr, size, r->flags};
    int kind = backing_rule(&ask);
    if (kind == BACKING_EXTENT) {
        if (!r->ext) r->ext = extent_new(r);
        return;
    }
    if (kind == BACKING_MIRROR)
        rt_fatal("gfx: vulkan: backing map: the mirror hook chose a mirror for [%#llx, %#llx), and mirrors are not "
                 "built yet", (ull)addr, (ull)(addr + size));
    uint64_t plo = down(addr, HPAGE), phi = up(addr + size, HPAGE);
    if (!rt_mem_commit_io(plo, phi - plo, 0))
        rt_fatal("gfx: vulkan: backing map: a bind of [%#llx, %#llx) in pool [%#llx, %#llx) covers memory the guest "
                 "has not mapped", (ull)addr, (ull)(addr + size), (ull)r->lo, (ull)r->hi);
    Cell *c = cell_get(r, addr / WINDOW, (addr + size - 1) / WINDOW, 1);
    uint64_t rs, re;
    committed_run(c->lo, c->hi, plo, phi, &rs, &re);
    char why[96];
    snprintf(why, sizeof why, "a bind of [%#llx, %#llx)", (ull)addr, (ull)(addr + size));
    cell_put(c, rs, re, import_new(rs, re, plo, why));
}

void backing_resolve(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out) {
    if (!size) size = 1;
    if (addr + size < addr) rt_fatal("gfx: vulkan: backing map: a bind of %#llx+%#llx wraps", (ull)addr, (ull)size);
    __atomic_fetch_add(&g_resolves, 1, __ATOMIC_RELAXED);
    plat_lock_shared(&g_lock);
    int hit = lookup(addr, size, value, out);
    plat_unlock_shared(&g_lock);
    if (hit) return;
    __atomic_fetch_add(&g_misses, 1, __ATOMIC_RELAXED);
    plat_lock(&g_lock);
    make(addr, size);
    if (!lookup(addr, size, value, out))
        rt_fatal("gfx: vulkan: backing map: [%#llx, %#llx) has no backing after one was made", (ull)addr,
                 (ull)(addr + size));
    plat_unlock(&g_lock);
    bound();
}

// ---------- the commit epoch
// Every import again: retired when a page under it is no longer committed, else grown to the run of committed pages
// around it inside its cell, which may join its neighbours. g_lock exclusive.
static void rescan_cell(Cell *c) {
    for (int i = 0; i < c->nimp;) {
        Backing *o = c->imp[i];
        uint64_t a = o->gpu_lo;
        int whole = 1;
        while (a < o->gpu_hi && whole) {
            int st;
            uint64_t n = rt_mem_commit_run(a, o->gpu_hi - a, &st);
            whole = n != UINT64_MAX && n && st;
            a += whole ? n : 0;
        }
        if (!whole) {
            defer(o);
            memmove(c->imp + i, c->imp + i + 1, (size_t)(c->nimp - i - 1) * sizeof *c->imp);
            c->nimp--;
            g_st.imports_stale++;
            continue;
        }
        uint64_t rs, re;
        committed_run(c->lo, c->hi, o->gpu_lo, o->gpu_hi, &rs, &re);
        if (rs == o->gpu_lo && re == o->gpu_hi) {
            i++;
            continue;
        }
        char why[96];
        snprintf(why, sizeof why, "chunks committed beside the import [%#llx, %#llx)", (ull)o->gpu_lo, (ull)o->gpu_hi);
        Backing *b = import_new(rs, re, o->gpu_lo, why);
        cell_put(c, rs, re, b);
        i = 0;   // the list changed; imports already grown are whole runs now and pass at once
    }
}

void backing_sync(void) {
    if (!g_open) return;
    uint64_t e = rt_mem_ready() ? rt_mem_commit_epoch() : 0;
    Freeing f = {0};
    plat_lock(&g_lock);
    g_st.syncs++;
    if (e != g_seen_epoch) {
        g_seen_epoch = e;   // read before the scan: a commit during it moves the epoch again
        g_st.sync_rescans++;
        for (int i = 0; i < g_index.n; i++) {
            Rec *r = (Rec *)g_index.e[i].key;
            for (int j = 0; r->cls == CLS_CPU && j < r->ncells; j++) rescan_cell(&r->cells[j]);
        }
    }
    retire(&f, 0);
    plat_unlock(&g_lock);
    free_all(&f);
    bound();
}

// ---------- decisions and counters
void backing_page(uint64_t addr, BackingPage *out) {
    *out = (BackingPage){BACKING_NONE, 0, addr};
    plat_lock_shared(&g_lock);
    Rec *r = rec_find(addr, 1);
    if (r && r->cls == CLS_VIRTUAL) {
        if (r->run && virtual_mapped(r, addr, 1)) out->gpu = (uint64_t)((int64_t)addr + r->run_delta);
        r = r->run && virtual_mapped(r, addr, 1) ? r->run : NULL;
    }
    const Backing *b = r && r->cls == CLS_GPU ? r->ext : NULL;
    for (int i = 0; r && r->cls == CLS_CPU && i < r->ncells && !b; i++) b = cell_import(&r->cells[i], out->gpu);
    if (b) {
        out->kind = b->kind;
        out->address = b->mem.address + b->offset + (out->gpu - b->gpu_lo);
    }
    plat_unlock_shared(&g_lock);
}

void backing_stats(BackingStats *out) {
    plat_lock_shared(&g_lock);
    *out = g_st;
    out->spare_blocks = (uint64_t)spare_count();
    plat_unlock_shared(&g_lock);
    out->resolves = __atomic_load_n(&g_resolves, __ATOMIC_RELAXED);
    out->resolve_misses = __atomic_load_n(&g_misses, __ATOMIC_RELAXED);
}

// ---------- open and close
void backing_open(const BackingOps *ops) {
    if (g_open) rt_fatal("gfx: vulkan: backing map: opened twice");
    if (!ops || !ops->block_new || !ops->import || !ops->free)
        rt_fatal("gfx: vulkan: backing map: an operation is missing from the table");
    g_ops = *ops;
    memset(&g_st, 0, sizeof g_st);
    g_resolves = g_misses = 0;
    g_seen_epoch = rt_mem_ready() ? rt_mem_commit_epoch() : 0;
    g_open = 1;
    g_bt_stop = 0, g_bt_want = 1, g_bt_running = 1;
    g_bt = plat_thread_start(bt_main, NULL, BLOCK_THREAD_STACK);
    if (!g_bt) rt_fatal("gfx: vulkan: backing map: the block thread did not start");
}

int backing_is_open(void) { return g_open; }

void backing_close(void) {
    if (!g_open) return;
    plat_mutex_lock(&g_bt_lock);
    g_bt_stop = 1;
    plat_cond_signal(&g_bt_cond);
    while (g_bt_running) plat_cond_wait(&g_bt_cond, &g_bt_lock);
    plat_mutex_unlock(&g_bt_lock);
    plat_thread_free(g_bt);
    g_bt = NULL;
    Freeing f = {0};
    plat_lock(&g_lock);
    while (g_index.n) rec_end((Rec *)g_index.e[g_index.n - 1].key);
    retire(&f, 1);
    while (g_nblocks) {   // the spares
        to_free(&f, &g_blocks[0]->mem);
        block_drop(g_blocks[0]);
    }
    g_open = 0;
    plat_unlock(&g_lock);
    free_all(&f);
}
