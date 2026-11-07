// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the backing map (runtime/gpu/backing.c, oot2/backing.h) that need no GPU, driven by
// tests/test_runtime_backing.py: `backing_test <case>`, one case per process. A passing case prints "ok <case>"; a
// failing case must end in one named fatal, which the driver checks.
//
// The guest memory is the runtime's own (rt_mem_init and SetHeapSize, first-touch commit), so the commit query and the
// commit epoch are the real ones. The device is a set of stand-ins: a block is host memory from calloc, an import is the
// host range itself (so bytes written through an import's reference land in guest memory, as on a device), and the
// import refuses, as the driver does (research 48 B1, B7), any range with a page that is not committed, judged
// without the runtime: by VirtualQuery on Windows, elsewhere by whether the kernel can read the page (a write of one
// byte from it into a pipe fails with EFAULT when it cannot). The timeline is oot2/gpu.h's over a counter the test
// raises, as tests/runtime/gpu_test.c drives it.
//
//   decisions   the decision of each kind of page before and after binds (CPU-visible, texture-only, CPU_NO_ACCESS,
//               virtual over a CPU_NO_ACCESS physical pool, virtual over a CPU-visible physical pool), and the rule
//   uncommitted an import is laid over committed pages only: separate runs get separate imports, a bind across the gap
//               commits it and joins them; no import ever reaches a page that is not committed
//   epoch       a chunk committed after its pool was mapped is imported at the next sync; one in a pool never bound is
//               not; an import over pages the guest decommitted is retired
//   deferred    a replaced import is freed only once the value that used it is done
//   finiwait    pool_fini of a pool a submission in flight used returns only after that submission is done; one no
//               submission used returns at once
//   cross       a bind across a 64 MiB window boundary inside one pool gets one import that holds all of it
//   refused     an import the device refuses once is retried after a 4 KiB import, as research 48 B7 requires
//   blocks      VRAM blocks are made by the block thread ahead of use; an extent larger than a block has its own
//   flat        4,660 pools made, bound and finalized, three times: nothing of the map or the stand-ins grows
//   bench       the resolver's cost per lookup over 4,500 pools, each already backed
//   raw_invariant  D15's invariant at a raw-access submission's begin: every committed page of every pool mapped and
//               no other; a chunk committed later and a pool made later mapped at the next one; the runs the table
//               is built from, a virtual pool's pages through its physical pool's backings
//   raw_finiwait a raw-access submission holds back the release of every backing and pool_fini, without a resolve
//   span_pools, span_virtual, import_fails, mirror   stop by name (the driver checks the line)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _WIN32
#include <windows.h>
#else
#include <errno.h>
#include <unistd.h>
#endif

#include "oot2/backing.h"
#include "oot2/gpu.h"
#include "oot2/loader.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define KiB (1ull << 10)
#define MiB (1ull << 20)
#define CPU_NO_ACCESS NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS
#define PHYSICAL NVN_MEMORY_POOL_FLAGS_PHYSICAL
#define VIRTUAL NVN_MEMORY_POOL_FLAGS_VIRTUAL
#define WIN GFX_GPU_WINDOW_BASE
typedef unsigned long long ull;

// ---------- the timeline over a test's counter
static PlatMutex g_src_lock = PLAT_MUTEX_INIT;
static PlatCond g_src_cond = PLAT_COND_INIT;
static uint64_t g_reached;

static uint64_t counter_wait(void *ctx, uint64_t value, uint64_t timeout_ns) {
    (void)ctx;
    plat_mutex_lock(&g_src_lock);
    uint64_t start = plat_now_ns();
    while (g_reached < value) {
        uint64_t spent = plat_now_ns() - start;
        if (spent >= timeout_ns) break;
        plat_cond_wait_ns(&g_src_cond, &g_src_lock, timeout_ns - spent);
    }
    uint64_t r = g_reached;
    plat_mutex_unlock(&g_src_lock);
    return r;
}

static void counter_raise(uint64_t to) {
    plat_mutex_lock(&g_src_lock);
    if (to > g_reached) g_reached = to;
    plat_cond_broadcast(&g_src_cond);
    plat_mutex_unlock(&g_src_lock);
}

// A submission that resolves nothing itself: the value it signals, issued, not done until the counter reaches it.
static uint64_t submission(void) {
    uint64_t v = gpu_sync_begin();
    gpu_sync_end();
    return v;
}

// ---------- the stand-in device
// 1 when every page of [a, a + n) is committed. VirtualQuery answers a whole run of pages in one state, and walks it
// to tell its length, as a driver's import does (research 48 answer 4), so it is asked once per run, not per page.
static int range_committed(uint64_t a, uint64_t n) {
#ifdef _WIN32
    while (n) {
        MEMORY_BASIC_INFORMATION m;
        if (VirtualQuery((void *)(uintptr_t)a, &m, sizeof m) != sizeof m || m.State != MEM_COMMIT) return 0;
        uint64_t run = (uint64_t)(uintptr_t)m.BaseAddress + m.RegionSize - a;
        if (run >= n) return 1;
        a += run, n -= run;
    }
    return 1;
#else
    static int fd[2] = {-1, -1};
    if (fd[0] < 0) CHECK(pipe(fd) == 0);
    for (uint64_t p = a; p < a + n; p += BACKING_HOST_PAGE) {
        char c;
        if (write(fd[1], (const void *)(uintptr_t)p, 1) != 1) return 0;
        CHECK(read(fd[0], &c, 1) == 1);
    }
    return 1;
#endif
}

enum { TAG_BLOCK = 1, TAG_IMPORT = 2 };
static PlatMutex g_dev_lock = PLAT_MUTEX_INIT;
static struct {
    uint64_t blocks, imports, frees_block, frees_import, live_blocks, live_imports;
    uint64_t violations, refusals, refuse_next, refused_size;   // refused_size: B7's memory of a failed import
    uint64_t last_lo, last_hi;
    int block_threads_main;   // blocks made on the test's main thread
} D;
static unsigned long g_main_thread;

static unsigned long thread_id(void) {
#ifdef _WIN32
    return GetCurrentThreadId();
#else
    return (unsigned long)pthread_self();
#endif
}

static int dev_block_new(uint64_t size, BackingMem *out) {
    void *p = calloc(1, (size_t)size);
    if (!p) return 0;
    plat_mutex_lock(&g_dev_lock);
    D.blocks++, D.live_blocks++;
    D.block_threads_main += thread_id() == g_main_thread;
    plat_mutex_unlock(&g_dev_lock);
    *out = (BackingMem){.buffer = (uintptr_t)p, .memory = TAG_BLOCK, .address = (uintptr_t)p, .size = size};
    return 1;
}

static int dev_import(uint64_t host, uint64_t size, BackingMem *out) {
    plat_mutex_lock(&g_dev_lock);
    int ok = !(host % BACKING_HOST_PAGE) && size && !(size % BACKING_HOST_PAGE);
    if (D.refuse_next) D.refuse_next--, ok = 0;
    if (D.refused_size && size >= D.refused_size) ok = 0;   // B7: until a smaller import succeeds
    if (ok && !range_committed(host, size)) {
        D.violations++;
        ok = 0;
    }
    if (!ok) {
        D.refusals++;
        D.refused_size = size;
    } else {
        if (D.refused_size && size < D.refused_size) D.refused_size = 0;
        D.imports++, D.live_imports++;
        D.last_lo = host, D.last_hi = host + size;
    }
    plat_mutex_unlock(&g_dev_lock);
    if (ok) *out = (BackingMem){.buffer = host, .memory = TAG_IMPORT, .address = host, .size = size};
    return ok;
}

static void dev_free(const BackingMem *m) {
    plat_mutex_lock(&g_dev_lock);
    if (m->memory == TAG_BLOCK) {
        D.frees_block++, D.live_blocks--;
        free((void *)(uintptr_t)m->buffer);
    } else {
        D.frees_import++, D.live_imports--;
    }
    plat_mutex_unlock(&g_dev_lock);
}

static const BackingOps k_ops = {.block_new = dev_block_new, .import = dev_import, .free = dev_free};

// ---------- guest memory and pools
static Cpu c;
static uint64_t g_heap;

static uint64_t svc(uint32_t imm, uint64_t x0, uint64_t x1, uint64_t x2) {
    c.x[0] = x0, c.x[1] = x1, c.x[2] = x2;
    rt_svc(&c, imm, 0);
    return c.x[0];
}

static void heap_size(uint64_t size) { CHECK(svc(0x01, 0, size, 0) == 0); }

// Commit the chunks of [a, a + n) by touching them, as the game's stores would.
static void touch(uint64_t a, uint64_t n) {
    for (uint64_t p = a; p < a + n; p += BACKING_HOST_PAGE) *(volatile uint8_t *)(uintptr_t)p = 1;
}

static void setup(uint64_t heap) {
    g_main_thread = thread_id();
    RtMemConfig cfg = {.pool_size = 2048 * MiB, .heap_region_size = 1024 * MiB, .alias_region_size = 64 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    heap_size(heap);
    g_heap = c.x[1];
    GpuTimelineSource src = {.wait = counter_wait};
    gpu_sync_start(&src, 0);
    backing_open(&k_ops);
}

static pool_t pool(uint64_t mem, uint64_t size, uint32_t flags) {
    pool_t p = {.magic = MAGIC('P', 'O', 'O', 'L'), .flags = flags, .mem = (uint8_t *)(uintptr_t)mem, .size = size,
                .game_mem = (flags & VIRTUAL) ? NULL : (uint8_t *)(uintptr_t)mem};
    CHECK(backing_pool_init(&p));
    return p;
}

static BackingRef resolve(uint64_t addr, uint64_t size, uint64_t value) {
    BackingRef r;
    memset(&r, 0xA5, sizeof r);
    backing_resolve(addr, size, value, &r);
    return r;
}

static int page_kind(uint64_t addr) {
    BackingPage p;
    backing_page(addr, &p);
    return p.kind;
}

static BackingStats stats(void) {
    BackingStats s;
    backing_stats(&s);
    return s;
}

// The host bytes a reference names: the stand-in's buffer is host memory, a block's or the imported range itself.
static uint8_t *bytes(const BackingRef *r) { return (uint8_t *)(uintptr_t)(r->buffer + r->offset); }

static void map(pool_t *v, const pool_t *phys, int64_t poff, int64_t voff, uint64_t size) {
    nvn_mapping_t m = {.physical_pool = (const NVNmemoryPool *)(const void *)phys, .physical_offset = poff,
                       .virtual_offset = voff, .size = size};
    CHECK(backing_pool_map_virtual(v, 1, &m));
}

static void done(const char *name) {
    backing_close();
    printf("ok %s\n", name);
}

// ---------- cases
static int want_mirror(const BackingAsk *a) { return a->size == 12345; }

static void case_decisions(void) {
    setup(256 * MiB);
    pool_t pc = pool(g_heap + 1 * MiB, 8 * MiB, 0);                        // a CPU-visible pool
    pool_t tex = pool(g_heap + 9 * MiB, 4 * MiB, 0);                       // one only textures use: never bound
    pool_t w = pool(WIN, 1 * MiB, CPU_NO_ACCESS);                          // a CPU_NO_ACCESS pool
    pool_t pw = pool(WIN + 1 * MiB, 2 * MiB, CPU_NO_ACCESS | PHYSICAL);    // a physical one in the window
    pool_t v = pool(WIN + 16 * MiB, 2 * MiB, CPU_NO_ACCESS | VIRTUAL);     // a virtual pool over it
    pool_t pcv = pool(g_heap + 16 * MiB, 2 * MiB, PHYSICAL);               // a CPU-visible physical pool (the title's)
    pool_t v2 = pool(WIN + 32 * MiB, 2 * MiB, CPU_NO_ACCESS | VIRTUAL);    // a virtual pool over that
    touch(g_heap + 1 * MiB, 8 * MiB);   // the CPU-visible pool's pages and the texture pool's, one run of commits
    touch(g_heap + 9 * MiB, 4 * MiB);
    touch(g_heap + 16 * MiB, 2 * MiB);
    map(&v, &pw, 64 * KiB, 0, 1 * MiB);
    map(&v2, &pcv, 128 * KiB, 0, 1 * MiB);
    uint64_t probes[] = {g_heap + 1 * MiB, g_heap + 9 * MiB, WIN + 0x1000, WIN + 1 * MiB, WIN + 16 * MiB, WIN + 32 * MiB};
    for (int i = 0; i < 6; i++) CHECK(page_kind(probes[i]) == BACKING_NONE);

    // CPU-visible: an import over the run of committed pages around the bind, inside the pool, though the run of
    // commits goes on into the texture pool beside it
    BackingRef r = resolve(g_heap + 1 * MiB + 0x100, 0x200, 0);
    CHECK(r.kind == BACKING_IMPORT && bytes(&r) == (uint8_t *)(uintptr_t)(g_heap + 1 * MiB + 0x100));
    CHECK(r.address == r.buffer + r.offset);
    CHECK(D.last_lo == g_heap + 1 * MiB && D.last_hi == g_heap + 9 * MiB);
    CHECK(page_kind(g_heap + 9 * MiB - 1) == BACKING_IMPORT);
    CHECK(page_kind(g_heap + 9 * MiB) == BACKING_NONE);   // committed, in the same window, never bound

    // CPU_NO_ACCESS: one extent over the whole pool at the first bind
    r = resolve(WIN + 0x1000, 0x100, 0);
    CHECK(r.kind == BACKING_EXTENT && r.offset % BACKING_EXTENT_ALIGN == 0x1000);
    CHECK(page_kind(WIN) == BACKING_EXTENT && page_kind(WIN + 1 * MiB - 1) == BACKING_EXTENT);
    BackingRef r2 = resolve(WIN + 0x80000, 0x80000, 0);
    CHECK(r2.buffer == r.buffer && r2.offset == r.offset + 0x7F000 && stats().extents == 1);

    // Virtual over a physical pool in the window: the physical pool's extent, at the run's offset
    BackingRef rv = resolve(WIN + 16 * MiB + 0x2000, 0x100, 0);
    BackingRef rp = resolve(WIN + 1 * MiB + 64 * KiB + 0x2000, 0x100, 0);
    CHECK(rv.kind == BACKING_EXTENT && rv.buffer == rp.buffer && rv.offset == rp.offset && rv.address == rp.address);
    memcpy(bytes(&rv), "through the virtual pool", 25);
    CHECK(!memcmp(bytes(&rp), "through the virtual pool", 25));
    BackingPage pg;
    backing_page(WIN + 16 * MiB + 0x2000, &pg);
    CHECK(pg.kind == BACKING_EXTENT && pg.gpu == WIN + 1 * MiB + 64 * KiB + 0x2000 && pg.address == rp.address);
    CHECK(page_kind(WIN + 17 * MiB) == BACKING_NONE);   // past the mapped megabyte

    // Virtual over a CPU-visible physical pool: an import of the guest pages the run names
    BackingRef rc = resolve(WIN + 32 * MiB + 0x3000, 0x40, 0);
    CHECK(rc.kind == BACKING_IMPORT && bytes(&rc) == (uint8_t *)(uintptr_t)(g_heap + 16 * MiB + 128 * KiB + 0x3000));
    *(volatile uint32_t *)(uintptr_t)(g_heap + 16 * MiB + 128 * KiB + 0x3000) = 0xC0FFEE;
    CHECK(*(uint32_t *)bytes(&rc) == 0xC0FFEE);

    // The rule
    BackingAsk ask = {g_heap, 0x100, 0};
    CHECK(backing_rule(&ask) == BACKING_IMPORT);
    ask.pool_flags = CPU_NO_ACCESS;
    CHECK(backing_rule(&ask) == BACKING_EXTENT);
    backing_set_mirror_hook(want_mirror);
    ask = (BackingAsk){g_heap, 12345, 0};
    CHECK(backing_rule(&ask) == BACKING_MIRROR);
    ask.size = 1;
    CHECK(backing_rule(&ask) == BACKING_IMPORT);
    backing_set_mirror_hook(NULL);

    pool_t *all[] = {&v2, &pcv, &v, &pw, &w, &tex, &pc};
    for (int i = 0; i < 7; i++) backing_pool_fini(all[i]);
    BackingStats s = stats();
    CHECK(s.pools == 0 && s.extents == 0 && s.imports == 0 && s.deferred == 0 && D.violations == 0);
    CHECK(D.live_imports == 0);
    done("decisions");
}

static void case_uncommitted(void) {
    setup(64 * MiB);
    uint64_t p = g_heap + 4 * MiB, a = p, cc = p + 256 * KiB;
    pool_t pp = pool(p, 8 * MiB, 0);
    touch(a, 64 * KiB);
    touch(cc, 64 * KiB);
    resolve(a + 0x10, 0x100, 0);
    CHECK(D.last_lo == a && D.last_hi == a + 64 * KiB);
    resolve(cc + 0x10, 0x100, 0);
    CHECK(D.last_lo == cc && D.last_hi == cc + 64 * KiB && stats().imports == 2);
    CHECK(page_kind(a + 64 * KiB) == BACKING_NONE);
    // A bind across the gap commits it (its own pages, as host I/O does) and joins both runs into one import.
    BackingRef r = resolve(a + 0x8000, 0x40000, 0);
    CHECK(r.kind == BACKING_IMPORT && D.last_lo == a && D.last_hi == cc + 64 * KiB);
    BackingStats s = stats();
    CHECK(s.imports == 1 && s.imports_grown == 2 && page_kind(a + 128 * KiB) == BACKING_IMPORT);
    CHECK(D.violations == 0 && D.refusals == 0);
    backing_pool_fini(&pp);
    done("uncommitted");
}

static void case_epoch(void) {
    setup(64 * MiB);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 4 * MiB, 0), other = pool(p + 4 * MiB, 4 * MiB, 0);
    touch(p, 64 * KiB);
    resolve(p, 0x100, 0);
    uint64_t e0 = rt_mem_commit_epoch();
    touch(p + 64 * KiB, 64 * KiB);   // committed after the pool was mapped
    touch(p + 4 * MiB, 64 * KiB);    // in the other pool, in the same window, never bound
    CHECK(rt_mem_commit_epoch() > e0);
    CHECK(page_kind(p + 64 * KiB) == BACKING_NONE);
    backing_sync();
    CHECK(page_kind(p + 64 * KiB) == BACKING_IMPORT && D.last_lo == p && D.last_hi == p + 128 * KiB);
    CHECK(page_kind(p + 4 * MiB) == BACKING_NONE);
    BackingStats s = stats();
    CHECK(s.imports == 1 && s.imports_grown == 1 && s.sync_rescans == 1 && D.frees_import == 1);
    backing_sync();   // nothing committed since: no rescan
    CHECK(stats().sync_rescans == 1);
    heap_size(2 * MiB);   // the guest gives the pool's pages back
    backing_sync();
    s = stats();
    CHECK(s.imports == 0 && s.imports_stale == 1 && page_kind(p) == BACKING_NONE && D.frees_import == 2);
    CHECK(D.violations == 0);
    backing_pool_fini(&pp);
    backing_pool_fini(&other);
    done("epoch");
}

static void case_deferred(void) {
    setup(64 * MiB);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 4 * MiB, 0);
    touch(p, 64 * KiB);
    uint64_t v = gpu_sync_begin();
    resolve(p, 0x100, v);   // used by a submission
    gpu_sync_end();
    touch(p + 64 * KiB, 64 * KiB);
    backing_sync();   // the import grows; the old one waits for v
    BackingStats s = stats();
    CHECK(s.imports_grown == 1 && s.deferred == 1 && D.frees_import == 0 && D.live_imports == 2);
    backing_sync();
    CHECK(D.frees_import == 0);
    counter_raise(v);
    CHECK(gpu_sync_wait(v, UINT64_MAX));
    backing_sync();
    s = stats();
    CHECK(s.deferred == 0 && D.frees_import == 1 && D.live_imports == 1);
    backing_pool_fini(&pp);
    CHECK(D.live_imports == 0);
    done("deferred");
}

static pool_t g_fini_pool;
static int g_fini_returned;
static PlatMutex g_join_lock = PLAT_MUTEX_INIT;
static PlatCond g_join_cond = PLAT_COND_INIT;

static void finisher(void *arg) {
    (void)arg;
    backing_pool_fini(&g_fini_pool);
    plat_mutex_lock(&g_join_lock);
    g_fini_returned = 1;
    plat_cond_broadcast(&g_join_cond);
    plat_mutex_unlock(&g_join_lock);
}

static void case_finiwait(void) {
    setup(64 * MiB);
    pool_t unused = pool(WIN + 8 * MiB, 1 * MiB, CPU_NO_ACCESS);
    resolve(WIN + 8 * MiB, 0x100, 0);
    g_fini_pool = pool(WIN, 1 * MiB, CPU_NO_ACCESS);
    uint64_t v = gpu_sync_begin();
    resolve(WIN + 0x100, 0x100, v);
    gpu_sync_end();
    submission();   // a later submission that does not use the pool
    CHECK(plat_thread_start(finisher, NULL, 256 * KiB));
    plat_sleep_ns(200 * 1000 * 1000);
    plat_mutex_lock(&g_join_lock);
    int early = g_fini_returned;
    plat_mutex_unlock(&g_join_lock);
    CHECK(!early);   // the submission that used the pool is in flight
    counter_raise(v);
    plat_mutex_lock(&g_join_lock);
    while (!g_fini_returned) plat_cond_wait(&g_join_cond, &g_join_lock);
    plat_mutex_unlock(&g_join_lock);
    CHECK(gpu_sync_done() >= v);
    BackingStats s = stats();
    CHECK(s.fini_waits == 1 && s.extents == 1 && s.deferred == 0);
    uint64_t t0 = plat_now_ns();
    backing_pool_fini(&unused);   // no submission used it: at once, though one is still in flight
    CHECK(plat_now_ns() - t0 < 100 * 1000 * 1000 && stats().fini_waits == 1 && gpu_sync_done() < gpu_sync_issued());
    counter_raise(gpu_sync_issued());
    done("finiwait");
}

static void case_cross(void) {
    setup(256 * MiB);
    uint64_t boundary = (g_heap + 64 * MiB + BACKING_IMPORT_WINDOW - 1) & ~(BACKING_IMPORT_WINDOW - 1);
    uint64_t p = boundary - 4 * MiB;
    pool_t pp = pool(p, 8 * MiB, 0);
    touch(p, 8 * MiB);
    BackingRef r = resolve(boundary - 0x800, 0x1000, 0);   // across the boundary
    CHECK(r.kind == BACKING_IMPORT && D.last_lo == p && D.last_hi == p + 8 * MiB);
    CHECK(bytes(&r) == (uint8_t *)(uintptr_t)(boundary - 0x800));
    // Inside the window below, and inside the one above: a cell each, whose imports alias the first one's pages.
    BackingRef q = resolve(boundary - 2 * MiB, 0x100, 0);
    CHECK(D.last_lo == p && D.last_hi == boundary && bytes(&q) == (uint8_t *)(uintptr_t)(boundary - 2 * MiB));
    BackingRef s = resolve(boundary + 0x100, 0x100, 0);
    CHECK(D.last_lo == boundary && D.last_hi == p + 8 * MiB && bytes(&s) == (uint8_t *)(uintptr_t)(boundary + 0x100));
    CHECK(stats().imports == 3 && D.imports == 3 && D.violations == 0);
    backing_pool_fini(&pp);
    done("cross");
}

static void case_refused(void) {
    setup(64 * MiB);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 4 * MiB, 0);
    touch(p, 1 * MiB);
    D.refuse_next = 1;
    BackingRef r = resolve(p + 0x100, 0x100, 0);
    BackingStats s = stats();
    CHECK(r.kind == BACKING_IMPORT && s.imports_refused == 1 && s.imports_retried == 1 && s.imports == 1);
    CHECK(D.refusals == 1 && D.imports == 2 && D.frees_import == 1);   // the 4 KiB import, then the import itself
    backing_pool_fini(&pp);
    done("refused");
}

static void wait_spares(uint64_t made) {
    uint64_t t0 = plat_now_ns();
    while (stats().spare_blocks < BACKING_SPARE_BLOCKS || stats().blocks_made_ahead < made) {
        CHECK(plat_now_ns() - t0 < 5ull * 1000 * 1000 * 1000);
        plat_sleep_ns(1000 * 1000);
    }
}

static void case_blocks(void) {
    setup(64 * MiB);
    wait_spares(1);
    pool_t a = pool(WIN, 16 * MiB, CPU_NO_ACCESS);
    resolve(WIN, 0x100, 0);   // the spare goes into use
    wait_spares(2);           // and the thread makes another
    pool_t big = pool(WIN + 1024 * MiB, BACKING_BLOCK + 64 * KiB, CPU_NO_ACCESS);
    resolve(WIN + 1024 * MiB, 0x100, 0);
    BackingStats s = stats();
    CHECK(s.blocks_made_inline == 0 && s.blocks_dedicated == 1 && s.blocks_made_ahead == 2 && s.blocks == 3);
    CHECK(D.block_threads_main == 1);   // the block of its own only
    backing_pool_fini(&big);
    CHECK(D.frees_block == 1 && stats().blocks == 2);
    backing_pool_fini(&a);
    s = stats();
    CHECK(s.blocks == 1 && s.spare_blocks == 1 && D.frees_block == 2);   // one spare is kept
    done("blocks");
}

// Until the block thread is idle with exactly the spares: a block it was making while extents retired is freed by it.
static void settle(void) {
    uint64_t t0 = plat_now_ns();
    for (;;) {
        plat_mutex_lock(&g_dev_lock);
        uint64_t live = D.live_blocks;
        plat_mutex_unlock(&g_dev_lock);
        if (live == BACKING_SPARE_BLOCKS && stats().blocks == BACKING_SPARE_BLOCKS) return;
        CHECK(plat_now_ns() - t0 < 5ull * 1000 * 1000 * 1000);
        plat_sleep_ns(1000 * 1000);
    }
}

static void case_flat(void) {
    enum { POOLS = 4660, HALF = POOLS / 2 };
    setup(512 * MiB);
    static pool_t p[POOLS];
    touch(g_heap, HALF * 64 * KiB);
    uint64_t base_blocks = 0, base_imports = 0, base_private = 0;
    for (int round = 0; round < 3; round++) {
        for (int i = 0; i < HALF; i++) p[i] = pool(g_heap + (uint64_t)i * 64 * KiB, 64 * KiB, 0);
        for (int i = 0; i < HALF; i++) p[HALF + i] = pool(WIN + (uint64_t)i * 64 * KiB, 64 * KiB, CPU_NO_ACCESS);
        for (int i = 0; i < POOLS; i++) resolve((uintptr_t)p[i].mem + 0x40, 0x100, 0);
        BackingStats s = stats();
        CHECK(s.pools == POOLS && s.imports == HALF && s.extents == HALF);
        for (int i = 0; i < POOLS; i++) backing_pool_fini(&p[i]);
        settle();
        s = stats();
        PlatProcMem m;
        plat_process_memory(&m);
        printf("round %d: pools %llu, imports %llu, extents %llu, deferred %llu, blocks %llu, stand-in blocks %llu, "
               "stand-in imports %llu, private %llu MiB\n", round, (ull)s.pools, (ull)s.imports, (ull)s.extents,
               (ull)s.deferred, (ull)s.blocks, (ull)D.live_blocks, (ull)D.live_imports, (ull)(m.private_bytes >> 20));
        CHECK(s.pools == 0 && s.imports == 0 && s.extents == 0 && s.deferred == 0 && D.live_imports == 0);
        CHECK(s.blocks == BACKING_SPARE_BLOCKS && D.live_blocks == BACKING_SPARE_BLOCKS);
        if (round == 0) base_blocks = D.blocks, base_imports = D.imports, base_private = m.private_bytes;
        else CHECK(m.private_bytes < base_private + 16 * MiB);
    }
    CHECK(D.imports == base_imports * 3 && D.blocks >= base_blocks);
    done("flat");
}

static uint64_t g_rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    g_rng ^= g_rng << 13, g_rng ^= g_rng >> 7, g_rng ^= g_rng << 17;
    return g_rng;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static void case_bench(void) {
    enum { CPU = 4000, GPU = 500, POOLS = CPU + GPU, BATCHES = 20, PER = 100000 };
    setup(512 * MiB);
    static pool_t p[POOLS];
    static uint64_t at[PER];
    touch(g_heap, CPU * 64 * KiB);
    for (int i = 0; i < CPU; i++) p[i] = pool(g_heap + (uint64_t)i * 64 * KiB, 64 * KiB, 0);
    for (int i = 0; i < GPU; i++) p[CPU + i] = pool(WIN + (uint64_t)i * MiB, MiB, CPU_NO_ACCESS);
    for (int i = 0; i < POOLS; i++) resolve((uintptr_t)p[i].mem, 0x100, 0);
    uint64_t misses = stats().resolve_misses, sink = 0;
    // Lookups spread over every pool (each pool's records cold in the cache), then over a set of 32 pools, as one draw's
    // vertex, index and uniform bindings return to the same few pools.
    for (int hot = 0; hot < 2; hot++) {
        uint64_t ns[BATCHES], set = hot ? 32 : POOLS;
        for (int b = 0; b < BATCHES; b++) {
            for (int i = 0; i < PER; i++) {
                const pool_t *q = &p[(rnd() % set) * (POOLS / set)];
                at[i] = (uintptr_t)q->mem + (rnd() % (q->size - 0x100)) / 16 * 16;
            }
            uint64_t t0 = plat_now_ns();
            for (int i = 0; i < PER; i++) {
                BackingRef r;
                backing_resolve(at[i], 0x100, 1, &r);
                sink += r.offset;
            }
            ns[b] = plat_now_ns() - t0;
        }
        qsort(ns, BATCHES, sizeof *ns, cmp_u64);
        printf("bench: %d pools (%d CPU-visible, %d CPU_NO_ACCESS), lookups over %llu of them, %d a batch: p50 %.1f "
               "ns, best %.1f ns, worst %.1f ns per lookup\n", POOLS, CPU, GPU, (ull)set, PER,
               (double)ns[BATCHES / 2] / PER, (double)ns[0] / PER, (double)ns[BATCHES - 1] / PER);
    }
    CHECK(stats().resolve_misses == misses && sink);
    for (int i = 0; i < POOLS; i++) backing_pool_fini(&p[i]);
    done("bench");
}

// ---------- cases that stop by name
static void case_span_pools(void) {
    setup(64 * MiB);
    pool(WIN, 1 * MiB, CPU_NO_ACCESS);
    pool(WIN + 1 * MiB, 1 * MiB, CPU_NO_ACCESS);
    resolve(WIN + 1 * MiB - 0x1000, 0x2000, 0);   // half in each: two backings
    printf("returned\n");
}

static void case_span_virtual(void) {
    setup(64 * MiB);
    pool_t pw = pool(WIN, 2 * MiB, CPU_NO_ACCESS | PHYSICAL);
    pool_t v = pool(WIN + 16 * MiB, 2 * MiB, CPU_NO_ACCESS | VIRTUAL);
    map(&v, &pw, 0, 0, 1 * MiB);
    resolve(WIN + 17 * MiB - 0x1000, 0x2000, 0);   // its last mapped page and the first unmapped one
    printf("returned\n");
}

static void case_import_fails(void) {
    setup(64 * MiB);
    pool(g_heap + 4 * MiB, 4 * MiB, 0);
    touch(g_heap + 4 * MiB, 64 * KiB);
    D.refuse_next = 3;
    resolve(g_heap + 4 * MiB, 0x100, 0);
    printf("returned\n");
}

static void case_mirror(void) {
    setup(64 * MiB);
    pool(g_heap + 4 * MiB, 4 * MiB, 0);
    backing_set_mirror_hook(want_mirror);
    resolve(g_heap + 4 * MiB, 12345, 0);
    printf("returned\n");
}

// ---------- the raw path (backing.h, The raw path)
// The run of runs[0..n) that holds a (the last one: the table's builder writes them in order), or NULL.
static const BackingRun *run_at(const BackingRun *runs, int n, uint64_t a) {
    const BackingRun *hit = NULL;
    for (int i = 0; i < n; i++)
        if (runs[i].lo <= a && a < runs[i].hi) hit = &runs[i];
    return hit;
}

// The device address a raw access of a reaches through the runs, or 0 when no run maps it.
static uint64_t run_address(const BackingRun *runs, int n, uint64_t a) {
    const BackingRun *r = run_at(runs, n, a);
    return r ? r->address + (a - r->lo) : 0;
}

// D15's invariant: every committed page of every pool is mapped at a raw-access submission's begin, and no page that is
// not committed; a chunk committed later, and a pool made later, are mapped at the next one; the runs say what each
// page maps to, a virtual pool's pages through the physical pool's backings.
static void case_raw_invariant(void) {
    setup(256 * MiB);
    uint64_t boundary = (g_heap + 64 * MiB + BACKING_IMPORT_WINDOW - 1) & ~(BACKING_IMPORT_WINDOW - 1);
    pool_t pc = pool(g_heap + 1 * MiB, 8 * MiB, 0);
    pool_t w = pool(WIN, 1 * MiB, CPU_NO_ACCESS);   // never bound
    pool_t pw = pool(WIN + 1 * MiB, 2 * MiB, CPU_NO_ACCESS | PHYSICAL);
    pool_t v = pool(WIN + 16 * MiB, 2 * MiB, CPU_NO_ACCESS | VIRTUAL);
    pool_t pcv = pool(g_heap + 16 * MiB, 2 * MiB, PHYSICAL);
    pool_t v2 = pool(WIN + 32 * MiB, 2 * MiB, CPU_NO_ACCESS | VIRTUAL);
    pool_t x = pool(boundary - 1 * MiB, 2 * MiB, 0);   // across an import window
    touch(g_heap + 1 * MiB, 1 * MiB);   // two runs of commits in pc, with a gap between them
    touch(g_heap + 4 * MiB, 1 * MiB);
    touch(g_heap + 16 * MiB, 2 * MiB);
    touch(boundary - 1 * MiB, 2 * MiB);
    map(&v, &pw, 64 * KiB, 0, 1 * MiB);
    map(&v2, &pcv, 128 * KiB, 0, 1 * MiB);
    resolve(g_heap + 1 * MiB + 0x100, 0x100, 0);   // one run already imported by a bind
    uint64_t version0 = backing_version();

    backing_raw_begin(submission());
    BackingStats s = stats();
    CHECK(s.raw_begins == 1 && s.raw_passes == 1 && s.raw_fill_extents == 2 && D.violations == 0);
    CHECK(s.raw_fill_imports == 4);   // pc's second run, pcv, x's two windows; the bound run was imported already
    for (uint64_t a = g_heap + 1 * MiB; a < g_heap + 9 * MiB; a += BACKING_HOST_PAGE) {
        int committed = a < g_heap + 2 * MiB || (a >= g_heap + 4 * MiB && a < g_heap + 5 * MiB);
        CHECK(committed ? page_kind(a) == BACKING_IMPORT : page_kind(a) == BACKING_NONE);
    }
    CHECK(page_kind(WIN) == BACKING_EXTENT && page_kind(WIN + 1 * MiB) == BACKING_EXTENT);
    CHECK(page_kind(boundary - 1 * MiB) == BACKING_IMPORT && page_kind(boundary + 1 * MiB - 1) == BACKING_IMPORT);
    CHECK(backing_version() > version0);

    BackingRun *runs;
    uint64_t version;
    int n = backing_raw_runs(&runs, &version);
    CHECK(version == backing_version());
    uint64_t probes[] = {g_heap + 1 * MiB + 0x1000, g_heap + 4 * MiB + 0x3000, WIN + 0x5000, WIN + 1 * MiB + 0x7000,
                         g_heap + 16 * MiB + 0x9000, boundary - 0x1000, boundary + 0x1000};
    for (int i = 0; i < 7; i++) {
        BackingPage pg;
        backing_page(probes[i], &pg);
        const BackingRun *r = run_at(runs, n, probes[i]);
        CHECK(pg.kind != BACKING_NONE && r && run_address(runs, n, probes[i]) == pg.address);
        CHECK(r->guest + (probes[i] - r->lo) == probes[i]);
    }
    CHECK(!run_at(runs, n, g_heap + 3 * MiB) && !run_at(runs, n, WIN + 17 * MiB));   // the gap; past v's mapping
    // A virtual pool's page maps where its physical page does, and its bytes stand for the physical page's.
    BackingPage pg;
    backing_page(WIN + 1 * MiB + 64 * KiB + 0x2000, &pg);
    CHECK(run_address(runs, n, WIN + 16 * MiB + 0x2000) == pg.address);
    backing_page(g_heap + 16 * MiB + 128 * KiB + 0x3000, &pg);
    CHECK(run_address(runs, n, WIN + 32 * MiB + 0x3000) == pg.address);
    const BackingRun *vr = run_at(runs, n, WIN + 32 * MiB + 0x3000);
    CHECK(vr->guest + (WIN + 32 * MiB + 0x3000 - vr->lo) == g_heap + 16 * MiB + 128 * KiB + 0x3000);
    free(runs);

    // Nothing moved: the next begin runs no pass, and the version stays.
    uint64_t version1 = backing_version();
    backing_raw_begin(submission());
    CHECK(stats().raw_passes == 1 && backing_version() == version1);
    // A chunk committed after its pool was mapped, apart from every run, and a pool made since: mapped at the next one.
    touch(g_heap + 7 * MiB, 64 * KiB);
    pool_t late = pool(g_heap + 24 * MiB, 1 * MiB, 0);
    touch(g_heap + 24 * MiB, 1 * MiB);
    CHECK(page_kind(g_heap + 7 * MiB) == BACKING_NONE && page_kind(g_heap + 24 * MiB) == BACKING_NONE);
    backing_raw_begin(submission());
    s = stats();
    CHECK(s.raw_passes == 2 && s.raw_fill_imports == 6 && backing_version() > version1);
    CHECK(page_kind(g_heap + 7 * MiB) == BACKING_IMPORT && page_kind(g_heap + 7 * MiB + 64 * KiB) == BACKING_NONE);
    CHECK(page_kind(g_heap + 24 * MiB + 1 * MiB - 1) == BACKING_IMPORT && D.violations == 0);

    counter_raise(gpu_sync_issued());
    pool_t *all[] = {&late, &x, &v2, &pcv, &v, &pw, &w, &pc};
    for (int i = 0; i < 8; i++) backing_pool_fini(all[i]);
    s = stats();
    CHECK(s.pools == 0 && s.extents == 0 && s.imports == 0 && s.deferred == 0 && D.live_imports == 0);
    done("raw_invariant");
}

// A raw-access submission reaches every backing without a resolve: a backing that leaves the map waits for it, and so
// does pool_fini of a pool it never resolved.
static void case_raw_finiwait(void) {
    setup(64 * MiB);
    uint64_t p = g_heap + 4 * MiB;
    pool_t pp = pool(p, 4 * MiB, 0);
    touch(p, 64 * KiB);
    g_fini_pool = pool(WIN, 1 * MiB, CPU_NO_ACCESS);
    uint64_t v = gpu_sync_begin();
    backing_raw_begin(v);   // maps both pools, resolves neither
    gpu_sync_end();
    touch(p + 64 * KiB, 64 * KiB);
    backing_sync();   // the import grows: the old one waits for v
    BackingStats s = stats();
    CHECK(s.imports_grown == 1 && s.deferred == 1 && D.frees_import == 0);
    CHECK(plat_thread_start(finisher, NULL, 256 * KiB));
    plat_sleep_ns(200 * 1000 * 1000);
    plat_mutex_lock(&g_join_lock);
    int early = g_fini_returned;
    plat_mutex_unlock(&g_join_lock);
    CHECK(!early);   // the raw-access submission is in flight
    counter_raise(v);
    plat_mutex_lock(&g_join_lock);
    while (!g_fini_returned) plat_cond_wait(&g_join_cond, &g_join_lock);
    plat_mutex_unlock(&g_join_lock);
    backing_sync();
    s = stats();
    CHECK(s.fini_waits == 1 && s.deferred == 0 && D.frees_import == 1);
    backing_pool_fini(&pp);
    done("raw_finiwait");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: backing_test <case>\n");
        return 2;
    }
    static const struct { const char *name; void (*fn)(void); } cases[] = {
        {"decisions", case_decisions}, {"uncommitted", case_uncommitted}, {"epoch", case_epoch},
        {"deferred", case_deferred},   {"finiwait", case_finiwait},       {"cross", case_cross},
        {"refused", case_refused},     {"blocks", case_blocks},           {"flat", case_flat},
        {"bench", case_bench},         {"span_pools", case_span_pools},   {"span_virtual", case_span_virtual},
        {"import_fails", case_import_fails}, {"mirror", case_mirror},
        {"raw_invariant", case_raw_invariant}, {"raw_finiwait", case_raw_finiwait},
    };
    for (size_t i = 0; i < sizeof cases / sizeof *cases; i++)
        if (!strcmp(argv[1], cases[i].name)) {
            cases[i].fn();
            return 0;
        }
    fprintf(stderr, "backing_test: no case %s\n", argv[1]);
    return 2;
}
