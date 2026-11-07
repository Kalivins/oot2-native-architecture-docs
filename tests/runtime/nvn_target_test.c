// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the fixed frame rate (the fixed frame rate gate; architecture D22; the frame-rate study; runtime/nvn/nvn_window.c,
// runtime/core/vsync.c, oot2/nvn_pace.h), driven by tests/test_runtime_nvn_target.py: `nvn_target_test <case> [args]`.
// Each case runs in its own process and prints "ok <case>".
//
// The settings store's frame_rate_target N puts the acquire gate on a second schedule of N slots a second on the clock's
// origin: slot m is due when the system counter reaches ceil(m * 31,250,000 / N), and the game's own interval is ignored.
// The cases show that with a synthetic guest thread written by hand as a host function in the lifter's shape (as
// nvn_sync_test.c does): it calls NVN entry points through their synthetic addresses and records the clock at each acquire
// and each SyncWait and the record each present left. No game code, no dump.
//
// The clock is driven (rt_vsync_drive): a test thread moves the time, each step only once the game has done what the
// time allows, so an hour runs in seconds. The expected counts are written here by hand (ref_count), never computed with
// the front end's rule. N comes from the environment (OOT2_FRAME_RATE_TARGET, the store's variable; the python harness
// sets it per case), or from the store's calls in the cases that load it.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/gfx.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/nvn_pace.h"
#include "oot2/nvn_tex.h"
#include "oot2/settings.h"
#include "oot2/vsync.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

typedef unsigned long long ull;
#define MiB (1ull << 20)
#define HEAP_BYTES (64 * MiB)
#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))   // the guest address of guest function i
#define PC (MB + 0x4000)             // the guest pc the guest threads' calls report
#define LOW32(v) ((v) & 0xffffffffu)
#define ALREADY_SIGNALED 0u          // NVNsyncWaitResult
#define CONDITION_SATISFIED 1u
#define TIMEOUT_EXPIRED 2u

// The schedules, written here without vsync.c: 31,250,000 counts a second of 32 ns; slot m of a schedule of hz a second
// is due at ceil(m * 31.25e6 / hz), and a count is in slot floor(count * hz / 31.25e6).
#define COUNTS 31250000ull
#define ORIGIN 123456789012ull
static uint64_t ref_count(uint64_t m, uint64_t hz) { return (m * COUNTS + hz - 1) / hz; }
static uint64_t ref_slot(uint64_t count, uint64_t hz) { return (uint64_t)((unsigned __int128)count * hz / COUNTS); }
static uint64_t ref_ns(uint64_t m, uint64_t hz) { return ORIGIN + ref_count(m, hz) * 32ull; }
static uint64_t count_ns(uint64_t count) { return ORIGIN + count * 32ull; }

static Cpu c;   // the host thread's Cpu: entry points called before the run
static uint64_t heap, heap_used, loader, g_area;
static uint64_t g_dev, g_queue, g_win;
static uint64_t f_acquire, f_present, f_setint, f_syncinit, f_syncwait;

// ---------- the driven time
static uint64_t g_now;   // atomic
static int g_clock_thread;   // the case binds an event, so the clock thread runs and waits through drv_wait
static uint64_t drv_now(void *ctx) {
    (void)ctx;
    return __atomic_load_n(&g_now, __ATOMIC_ACQUIRE);
}
static uint64_t g_drv_waits;   // waits the clock thread has entered (atomic): the driver starts once it is parked
static void drv_wait(void *ctx, uint64_t deadline) {
    (void)ctx;
    if (!g_clock_thread) {
        fprintf(stderr, "FAIL: the clock thread waited\n");
        exit(1);
    }
    __atomic_add_fetch(&g_drv_waits, 1, __ATOMIC_RELEASE);
    uint64_t start = plat_now_ns();
    while (__atomic_load_n(&g_now, __ATOMIC_ACQUIRE) < deadline) {
        if (plat_now_ns() - start < 1000000ull) plat_yield();
        else plat_sleep_ns(200000ull);
    }
}
static void set_now(uint64_t ns) {
    CHECK(ns >= __atomic_load_n(&g_now, __ATOMIC_RELAXED));
    __atomic_store_n(&g_now, ns, __ATOMIC_RELEASE);
}
static void drive_clock(uint64_t first_count) {
    g_now = count_ns(first_count);
    RtVsyncDriver d = {.origin_ns = ORIGIN, .now = drv_now, .wait_until = drv_wait};
    rt_vsync_drive(&d);
}

// ---------- the guest side
enum { IDX = 0x0, SYNC = 0x200, STACKS = 0x100000, AREA = 0x120000 };
#define W(off) (g_area + (off))
static inline uint64_t sv(Cpu *t, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    t->x[0] = a0, t->x[1] = a1, t->x[2] = a2, t->x[3] = a3;
    rt_svc(t, imm, PC);
    return t->x[0];
}
// A call through a PLT slot, as the lifter writes one.
static inline uint64_t gcall(Cpu *t, uint64_t fn, uint64_t a0, uint64_t a1, uint64_t a2) {
    t->x[0] = a0, t->x[1] = a1, t->x[2] = a2, t->x[3] = 0, t->x[30] = PC + 4, t->pc = fn;
    rt_dispatch(t);
    return t->x[0];
}
static inline uint64_t ld_acq(const uint64_t *p) { return __atomic_load_n(p, __ATOMIC_ACQUIRE); }
static inline void st_rel(uint64_t *p, uint64_t v) { __atomic_store_n(p, v, __ATOMIC_RELEASE); }

enum { ORDER_WAIT_PRESENT, ORDER_PRESENT_WAIT };   // where the game presents: after its wait on the acquire sync, or before

#define MAXF 360004
static uint64_t g_nframes;
static int g_handshake;                    // the game waits for GO after its acquire and wait (the driver's "work" time)
static int g_order;
static uint64_t A_ACQ, A_SYNC, A_GO, A_PRES;   // acquires returned, waits returned, presents allowed, presents done (atomic)
static uint64_t g_ret_cnt[MAXF];           // the system counter each acquire returned at, as the game read it
static uint64_t g_sync_cnt[MAXF];          // the counter each SyncWait on the acquire's sync returned at
static uint32_t g_sync_res[MAXF];          // its result
static NvnPresentRecord g_rec[MAXF];       // the record each present left, read right after it
static uint64_t g_cpu_start, g_cpu_end;    // the game thread's CPU time around its frames

static void game_wait(Cpu *t, uint64_t k) {
    g_sync_res[k] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), ~0ull, 0));
    g_sync_cnt[k] = rt_vsync_counter();
    st_rel(&A_SYNC, k + 1);
}

static void game_body(Cpu *t) {
    CHECK((gcall(t, f_syncinit, W(SYNC), g_dev, 0) & 0xff) == 1);   // one sync, reused by every acquire
    g_cpu_start = plat_thread_cpu_ns(NULL);
    for (uint64_t k = 0; k < g_nframes; k++) {
        CHECK(LOW32(gcall(t, f_acquire, g_win, W(SYNC), W(IDX))) == 0);
        g_ret_cnt[k] = rt_vsync_counter();
        st_rel(&A_ACQ, k + 1);
        if (g_order == ORDER_WAIT_PRESENT) game_wait(t, k);
        if (g_handshake) {
            while (ld_acq(&A_GO) < k + 1) sv(t, 0x1E, 0, 0, 0, 0);   // GetSystemTick: the exit is a safe point
        }
        uint32_t idx = *(volatile uint32_t *)(uintptr_t)W(IDX);
        t->x[0] = g_queue, t->x[1] = g_win, t->x[2] = idx, t->x[3] = 0, t->x[30] = PC + 4, t->pc = f_present;
        rt_dispatch(t);
        CHECK(nvn_present_record(nvn_present_count(), &g_rec[k]));
        st_rel(&A_PRES, k + 1);
        if (g_order == ORDER_PRESENT_WAIT) game_wait(t, k);
    }
    g_cpu_end = plat_thread_cpu_ns(NULL);
}

static void g_main_game(Cpu *t) {
    game_body(t);
    sv(t, 0x0A, 0, 0, 0, 0);
}
static void guest_end(Cpu *t) { (void)t; }
static const GuestFn g_fns[] = {g_main_game, guest_end};
#define NFN ((int)(sizeof g_fns / sizeof *g_fns))
static uint64_t g_addrs[NFN];

// ---------- set-up
static const uint64_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static const GfxBackend *g_backend;   // a case's own back end, set before objects()

static void setup(void) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512 * MiB, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256 * MiB, .system_resource_size = MiB};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x1, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                        .program_id = 0x0100000000001000ull, .is_application = 1, .no_async_preempt = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++)
        k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    if (g_backend) gfx_set_backend(g_backend);
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    nvn_procs_register();
    for (int i = 0; i < NFN; i++) g_addrs[i] = G(i);
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, g_addrs, g_fns, (uint64_t)NFN, TEST_END};
    rt_dispatch_init(mods, 1);
    c.x[0] = 0, c.x[1] = HEAP_BYTES;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    c.excl_addr = ~0ull;
}

static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP_BYTES);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}

static uint64_t vcall(uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    rt_call(&c, fn);
    return c.x[0];
}
static uint64_t proc(const char *name) {
    static uint64_t gpa;
    if (!gpa) {
        size_t len = strlen("nvnDeviceGetProcAddress") + 1;
        uint64_t s = galloc(len, 16);
        memcpy((void *)(uintptr_t)s, "nvnDeviceGetProcAddress", len);
        c.x[0] = s, c.x[1] = 0;
        rt_call(&c, loader);
        gpa = c.x[0];
    }
    CHECK(gpa != 0);
    size_t len = strlen(name) + 1;
    uint64_t s = galloc(len, 16);
    memcpy((void *)(uintptr_t)s, name, len);
    c.x[0] = 0, c.x[1] = s;
    rt_call(&c, gpa);
    CHECK(c.x[0] != 0);
    return c.x[0];
}
#define NVN(name, ...) vcall(proc(#name), (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / 8)

static uint64_t texture(uint64_t serial) {
    uint64_t t = galloc(0x100, 16);
    tex_t *x = (tex_t *)(uintptr_t)t;
    x->magic = NVN_TEX_MAGIC, x->serial = serial, x->d.width = 1280, x->d.height = 720, x->d.depth = 1;
    return t;
}

static uint64_t window(int n, int interval) {
    uint64_t arr = galloc(8 * 4, 8), b = galloc(0x40, 16), w = galloc(0x180, 16);
    for (int i = 0; i < n; i++) *(uint64_t *)(uintptr_t)(arr + 8ull * (uint64_t)i) = texture((uint64_t)i + 1);
    NVN(nvnWindowBuilderSetDevice, b, g_dev);
    NVN(nvnWindowBuilderSetDefaults, b);
    NVN(nvnWindowBuilderSetNativeWindow, b, 0xAB00);
    NVN(nvnWindowBuilderSetTextures, b, (uint64_t)n, arr);
    NVN(nvnWindowBuilderSetPresentInterval, b, (uint64_t)(uint32_t)interval);
    CHECK((NVN(nvnWindowInitialize, w, b) & 0xff) == 1);
    return w;
}

#define UNDRIVEN (~0ull)
#define PLACEMENT_ENV (-1)   // leave the placement as the process starts: the default, or what OOT2_HOLD_PLACEMENT names
// Memory, kernel, entry points, a device, a queue and a window of n textures at the given interval; the clock is driven
// and stands at first_count (UNDRIVEN: the host's own clock). hold: the experiment hold, set explicitly so the environment
// cannot change a case; placement: the acquire placement, set the same way (PLACEMENT_ENV: as the process starts).
static void objects(int n, int interval, uint64_t first_count, int hold, int placement) {
    setup();
    if (first_count != UNDRIVEN) drive_clock(first_count);
    if (hold >= 0) nvn_present_hold(hold);
    if (placement >= 0) nvn_acquire_wait(placement);
    uint64_t db = galloc(0x40, 16);
    g_dev = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, db);
    CHECK((NVN(nvnDeviceInitialize, g_dev, db) & 0xff) == 1);
    uint64_t qb = galloc(0x40, 16);
    g_queue = galloc(0x2000, 16);
    NVN(nvnQueueBuilderSetDevice, qb, g_dev);
    NVN(nvnQueueBuilderSetDefaults, qb);
    NVN(nvnQueueBuilderSetFlags, qb, 0);
    NVN(nvnQueueBuilderSetCommandMemorySize, qb, 0x10000);
    NVN(nvnQueueBuilderSetComputeMemorySize, qb, 0x40000);
    NVN(nvnQueueBuilderSetControlMemorySize, qb, 0x4000);
    uint64_t size = NVN(nvnQueueBuilderGetQueueMemorySize, qb);
    NVN(nvnQueueBuilderSetQueueMemory, qb, galloc(size, 0x1000), size);
    CHECK((NVN(nvnQueueInitialize, g_queue, qb) & 0xff) == 1);
    g_win = window(n, interval);
    g_area = galloc(AREA, 0x10000);
    f_acquire = proc("nvnWindowAcquireTexture");
    f_present = proc("nvnQueuePresentTexture");
    f_setint = proc("nvnWindowSetPresentInterval");
    f_syncinit = proc("nvnSyncInitialize");
    f_syncwait = proc("nvnSyncWait");
}

// The target the process starts with: the store's variable, which the harness sets (0 when it does not).
static uint64_t env_rate(void) {
    const char *e = getenv("OOT2_FRAME_RATE_TARGET");
    return e && *e ? strtoull(e, NULL, 10) : 0;
}

// ---------- the drivers, host threads
static void fail(const char *what, uint64_t k) {
    fprintf(stderr, "FAIL step %llu: %s\n", (ull)k, what);
    exit(1);
}

// Wait until the word reaches want; 10 s of wall time is a failure of the case. Yields for a millisecond, then sleeps,
// so the thread it waits for gets a processor on a small or loaded machine.
static void await(const uint64_t *p, uint64_t want, uint64_t k, const char *what) {
    uint64_t start = plat_now_ns(), end = start + 10000000000ull;
    while (ld_acq(p) < want) {
        uint64_t now = plat_now_ns();
        if (now > end) fail(what, k);
        if (now - start < 1000000ull) plat_yield();
        else plat_sleep_ns(1000000ull);
    }
}

// A call that must come back with the clock stopped: if it waits for a count, the guard ends the case after 5 s.
static uint64_t g_guard_gen;
static const char *g_guard_msg;
static void guard_main(void *arg) {
    uint64_t gen = (uint64_t)(uintptr_t)arg;
    plat_sleep_ns(5000000000ull);
    if (ld_acq(&g_guard_gen) == gen) {
        fprintf(stderr, "FAIL: %s\n", g_guard_msg);
        exit(1);
    }
}
static void guard_begin(const char *msg) {
    g_guard_msg = msg;
    uint64_t gen = __atomic_add_fetch(&g_guard_gen, 1, __ATOMIC_ACQ_REL);
    CHECK(plat_thread_start(guard_main, (void *)(uintptr_t)gen, 1ull << 20));
}
static void guard_end(void) { __atomic_add_fetch(&g_guard_gen, 1, __ATOMIC_ACQ_REL); }

static uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }

// The hour: slot t+1 of the schedule (g_hz slots a second, the game's frame k released at slot k * g_p) is reached only once
// the game has done everything slot t allows, and not more. Where the game presents after its wait, the waits that ended
// by slot t are frames 0 to t / p and the presents the same; where it presents before, the present of the next frame is
// made too (its sync is what the game then waits on). A wait that came back early, or a present ahead of that, shows as a
// count ahead of its slot. Each slot is entered 0 to 15 steps of a seventeenth of a period late. And at the first slots and
// then every 997th, a nanosecond before the due count (one count short of it): the wait on the sync must still be
// waiting, which a period rounded to a whole nanosecond or tick, or summed, or a gate one count early, would not be.
static uint64_t g_hz = 60, g_p = 1, g_total_slots;
static uint64_t jitter_ns(uint64_t t) { return (t * 7919 % 16) * (1000000000ull / g_hz / 17); }
static void driver_hour(void *arg) {
    (void)arg;
    for (uint64_t t = 0;; t++) {
        uint64_t want_sync = min_u64(t / g_p + 1, g_nframes);
        uint64_t want_pres = min_u64(t / g_p + 1 + (g_order == ORDER_PRESENT_WAIT), g_nframes);
        // the acquire that follows the last present (or the last wait) has returned before the clock moves: it never waits
        uint64_t want_acq = min_u64(g_order == ORDER_WAIT_PRESENT ? want_pres + 1 : want_pres, g_nframes);
        await(&A_SYNC, want_sync, t, "the game's wait on the acquire sync has not ended (the sync is signalled too late)");
        await(&A_PRES, want_pres, t, "the game has not presented the frames this slot allows");
        await(&A_ACQ, want_acq, t, "an acquire did not come back at once");
        if (ld_acq(&A_SYNC) > want_sync) fail("a wait on the acquire sync ended ahead of its slot (signalled too early)", t);
        if (ld_acq(&A_PRES) > want_pres) fail("the game presented a frame ahead of its slot", t);
        if (ld_acq(&A_ACQ) > want_acq) fail("an acquire came back ahead of its slot", t);
        if (t == g_total_slots) break;
        if ((t + 1) % g_p == 0 && (t < 8 || (t + 1) % 997 == 0)) {
            set_now(ref_ns(t + 1, g_hz) - 1);   // one count short of the due count: 31 ns into the count before it
            plat_sleep_ns(3000000ull);
            if (ld_acq(&A_SYNC) != want_sync) fail("a wait on the acquire sync ended one count before its due count", t);
        }
        set_now(ref_ns(t + 1, g_hz) + jitter_ns(t));
    }
}

static uint64_t fnv(uint64_t h, uint64_t v) {
    for (int i = 0; i < 8; i++) h = (h ^ ((v >> (8 * i)) & 0xff)) * 0x100000001b3ull;
    return h;
}

// hour <hz> <periods> <interval> <order> <seconds>: the schedule is hz slots a second (the target's, or 60 with none), the
// game's frame k is released at slot k * periods (1 under a target, the game's interval without one), the game asks for
// <interval>. Every acquire comes back at once; every wait ends in its slot and the record's due count is the model's.
static void case_hour(uint64_t hz, uint64_t periods, int interval, int order, uint64_t seconds) {
    uint64_t rate = env_rate();
    g_hz = hz, g_p = periods, g_order = order, g_total_slots = seconds * hz, g_nframes = g_total_slots / periods + 1;
    CHECK(g_nframes <= MAXF && g_total_slots % periods == 0);
    objects(3, interval, 0, 0, NVN_ACQUIRE_WAIT_SYNC);
    CHECK(plat_thread_start(driver_hour, NULL, 1ull << 20));
    uint64_t t0 = plat_now_ns(), h = 0xcbf29ce484222325ull;
    rt_kernel_run(G(0), 0);
    uint64_t n = g_nframes;
    for (uint64_t k = 0; k < n; k++) {
        uint64_t slot = k * periods, due = ref_count(slot, hz);
        uint64_t ret = k ? (k - 1) * periods : 0;
        if (ref_slot(g_ret_cnt[k], hz) != ret || ref_slot(g_sync_cnt[k], hz) != slot || g_rec[k].due_count != due ||
            g_rec[k].due != ref_slot(due, 60) || g_rec[k].requested != interval || g_rec[k].held != 0 ||
            g_rec[k].target != (int32_t)rate || g_rec[k].placement != NVN_ACQUIRE_WAIT_SYNC || g_rec[k].shown != 0 ||
            g_rec[k].seq != k + 1 || g_rec[k].texture != (int)(k % 3) ||
            (g_sync_res[k] != CONDITION_SATISFIED && g_sync_res[k] != ALREADY_SIGNALED) ||
            (k == 0 && g_sync_res[k] != ALREADY_SIGNALED)) {
            fprintf(stderr, "FAIL frame %llu: acquire returned in slot %llu (want %llu), wait ended in slot %llu (want %llu) "
                            "with result %u, due count %llu (want %llu) tick %llu, requested %d, held %d, target %d, "
                            "placement %d\n", (ull)k, (ull)ref_slot(g_ret_cnt[k], hz), (ull)ret,
                    (ull)ref_slot(g_sync_cnt[k], hz), (ull)slot, g_sync_res[k], (ull)g_rec[k].due_count, (ull)due,
                    (ull)g_rec[k].due, g_rec[k].requested, g_rec[k].held, g_rec[k].target, g_rec[k].placement);
            exit(1);
        }
        h = fnv(fnv(fnv(fnv(h, g_rec[k].due_count), g_rec[k].due), g_rec[k].seq), (uint64_t)g_rec[k].texture);
    }
    CHECK(nvn_present_count() == n && gfx_null_presents() == n);
    printf("hour: %llu presents, %llu slots a second, the game asking for interval %d%s; every acquire came back at once, "
           "every wait ended in the slot %llu slots apart from slot 0 to slot %llu, every due count the model's, in %.1f s "
           "of wall time\n", (ull)n, (ull)hz, interval, order == ORDER_WAIT_PRESENT ? "" : " and presenting before its wait",
           (ull)periods, (ull)(g_nframes - 1) * periods, (double)(plat_now_ns() - t0) / 1e9);
    printf("digest %016llx\n", (ull)h);
    printf("ok hour\n");
}

// events <hz> <seconds>: the same, with the clock thread running (an event is bound), over `seconds` of driven time. The
// driver moves the time through the 60 Hz ticks and through the slots in time order, and the clock thread signals one
// event per tick whatever the target is: every tick of the time is one signal, none skipped, none extra.
static void await_ticks(uint64_t n) {
    uint64_t start = plat_now_ns();
    while (rt_vsync_stats().ticks < n) {
        uint64_t now = plat_now_ns();
        if (now - start > 10000000000ull) fail("the clock thread did not signal its tick", n);
        if (now - start < 1000000ull) plat_yield();
        else plat_sleep_ns(100000ull);
    }
}
static void driver_events(void *arg) {
    (void)arg;
    uint64_t tick = 1, ticks = g_total_slots * 60 / g_hz;   // the 60 Hz ticks in the run
    for (uint64_t t = 0;; t++) {
        uint64_t want_sync = min_u64(t / g_p + 1, g_nframes), want_pres = min_u64(t / g_p + 1, g_nframes);
        uint64_t want_acq = min_u64(want_pres + 1, g_nframes);
        await(&A_SYNC, want_sync, t, "the game's wait on the acquire sync has not ended");
        await(&A_PRES, want_pres, t, "the game has not presented the frames this slot allows");
        await(&A_ACQ, want_acq, t, "an acquire did not come back at once");
        if (ld_acq(&A_SYNC) > want_sync || ld_acq(&A_PRES) > want_pres) fail("the game ran ahead of its slot", t);
        if (t == g_total_slots) break;
        uint64_t at = ref_ns(t + 1, g_hz) + jitter_ns(t);
        while (tick <= ticks && ref_ns(tick, 60) <= at) {   // the ticks that come first, each signalled before the next
            set_now(ref_ns(tick, 60));
            await_ticks(tick);
            tick++;
        }
        set_now(at);
    }
    while (tick <= ticks) {   // the ticks after the last slot's step
        set_now(ref_ns(tick, 60));
        await_ticks(tick);
        tick++;
    }
}
static void case_events(uint64_t hz, uint64_t seconds) {
    uint64_t rate = env_rate();
    g_hz = hz, g_p = 1, g_order = ORDER_WAIT_PRESENT, g_total_slots = seconds * hz, g_nframes = g_total_slots + 1;
    CHECK(g_nframes <= MAXF && rate == hz);
    g_clock_thread = 1;
    objects(3, 1, 0, 0, NVN_ACQUIRE_WAIT_SYNC);
    uint32_t r;
    rt_vsync_bind(rt_kernel_event_create(&r));   // starts the clock thread
    // The clock thread reads the time once to find its first tick: the driver may move the time only after it is parked on
    // that tick's deadline, or a loaded machine lets it start late and skip tick 1.
    await(&g_drv_waits, 1, 0, "the clock thread did not start");
    CHECK(plat_thread_start(driver_events, NULL, 1ull << 20));
    rt_kernel_run(G(0), 0);
    uint64_t ticks = seconds * 60;
    // The last tick and the last slot are the same instant (a whole number of seconds), so the game can end while the clock
    // thread is still signalling that tick: wait for it, then read the counts (the driver moves the time no further).
    await_ticks(ticks);
    RtVsyncStats s = rt_vsync_stats();
    if (s.ticks != ticks || s.skipped != 0 || s.last != ticks || s.events != 1) {
        fprintf(stderr, "FAIL: the clock signalled %llu ticks (want %llu), skipped %llu, last %llu, events %d\n",
                (ull)s.ticks, (ull)ticks, (ull)s.skipped, (ull)s.last, s.events);
        exit(1);
    }
    for (uint64_t k = 0; k < g_nframes; k++)
        CHECK(g_rec[k].due_count == ref_count(k, hz) && g_rec[k].target == (int32_t)rate &&
              ref_slot(g_sync_cnt[k], hz) == k);
    printf("events: %llu presents at %llu a second and %llu vsync events over %llu s, none skipped\n",
           (ull)g_nframes, (ull)hz, (ull)s.ticks, (ull)seconds);
    printf("ok events\n");
}

// ---------- scripts: exact counts, the clock stopped between steps
static uint64_t g_gate[MAXF];   // by hand, per frame k: the count its acquire's sync is signalled at (default placement)
                                // or its acquire returns at (old placement); frame 0 has none
static int g_script_placement;
static void driver_script(void *arg) {
    (void)arg;
    for (uint64_t k = 0; k < g_nframes; k++) {
        if (k > 0) {
            if (g_script_placement == NVN_ACQUIRE_WAIT_SYNC) await(&A_ACQ, k + 1, k, "the acquire did not come back at once");
            uint64_t ns = count_ns(g_gate[k]);
            set_now(ns - 1);               // one count short: still held, or still not signalled
            plat_sleep_ns(5000000ull);     // only a probe: the recorded counts are the authoritative check
            if (g_script_placement == NVN_ACQUIRE_WAIT_SYNC) {
                if (ld_acq(&A_SYNC) != k) fail("the wait on the acquire sync ended before its due count", k);
            } else if (ld_acq(&A_ACQ) != k) {
                fail("the acquire came back before its due count", k);
            }
            set_now(ns);
        }
        await(&A_SYNC, k + 1, k, "the wait on the acquire sync did not end at its due count");
        st_rel(&A_GO, k + 1);
        await(&A_PRES, k + 1, k, "the present did not come back");
    }
}

// script <placement> <hold> <periods> <frames>: the target from the environment, the game asking for interval 1 (the
// target replaces it) and the E2 hold <hold>: the gate of frame k is slot k * periods of the target's schedule, where
// periods is what the case expects by hand (1; or the hold's ticks rounded up to whole slots, when that is more).
static void case_script(int placement, int hold, uint64_t periods, uint64_t frames) {
    uint64_t rate = env_rate();
    CHECK(rate != 0 && frames <= MAXF);
    g_nframes = frames, g_handshake = 1, g_order = ORDER_WAIT_PRESENT, g_script_placement = placement;
    for (uint64_t k = 1; k < frames; k++) g_gate[k] = ref_count(k * periods, rate);
    objects(3, 1, 0, hold, placement);
    CHECK(nvn_acquire_wait_mode() == placement && nvn_present_hold_ticks() == hold);
    CHECK(plat_thread_start(driver_script, NULL, 1ull << 20));
    rt_kernel_run(G(0), 0);
    for (uint64_t k = 0; k < frames; k++) {
        uint64_t due = k ? g_gate[k] : 0;
        uint64_t acq = placement == NVN_ACQUIRE_WAIT_SYNC ? (k ? g_gate[k - 1] : 0) : due;
        printf("frame %llu acquired at count %llu synced at count %llu due count %llu requested %d held %d target %d\n",
               (ull)k, (ull)g_ret_cnt[k], (ull)g_sync_cnt[k], (ull)g_rec[k].due_count, g_rec[k].requested, g_rec[k].held,
               g_rec[k].target);
        CHECK(g_ret_cnt[k] == acq);          // the acquire returned at once (default) or at its gate (old placement)
        CHECK(g_sync_cnt[k] == due);         // the wait on its sync ended at the gate, to the count
        CHECK(g_rec[k].due_count == due && g_rec[k].due == ref_slot(due, 60));
        CHECK(g_rec[k].requested == 1 && g_rec[k].held == hold && g_rec[k].target == (int32_t)rate);
        CHECK(g_rec[k].placement == placement && g_rec[k].seq == k + 1);
    }
    printf("ok script\n");
}

// ---------- the calls below run on the host thread with the clock stopped. A call that waits for a count never comes back;
// the guard ends the case after 5 s.
static uint64_t host_acquire_sync(uint64_t w, uint64_t sync) {
    uint64_t idx = galloc(4, 4), a[3] = {w, sync, idx};
    CHECK(LOW32(vcall(f_acquire, a, 3)) == 0);
    return *(uint32_t *)(uintptr_t)idx;
}
static void host_present(uint64_t w, uint64_t i) {
    uint64_t a[3] = {g_queue, w, i};
    vcall(f_present, a, 3);
}
static uint64_t host_sync(void) {
    uint64_t s = galloc(0x40, 16), a[2] = {s, g_dev};
    CHECK((vcall(f_syncinit, a, 2) & 0xff) == 1);
    return s;
}
static uint32_t host_wait(uint64_t s, uint64_t timeout) {
    uint64_t a[2] = {s, timeout};
    return (uint32_t)LOW32(vcall(f_syncwait, a, 2));
}
// the sync is not signalled one count before `count` and is at it
static void expect_gate(uint64_t s, uint64_t count) {
    CHECK(host_wait(s, 0) == TIMEOUT_EXPIRED);
    set_now(count_ns(count) - 1);
    CHECK(host_wait(s, 0) == TIMEOUT_EXPIRED);
    set_now(count_ns(count));
    CHECK(host_wait(s, 0) == ALREADY_SIGNALED);
}

static void store_load(void) {
    char why[400];
    CHECK(rt_settings_load(NULL, 0, NULL, why, sizeof why));
    nvn_settings_bind();
}
static void set_rate(const char *text) {
    char why[400];
    CHECK(rt_settings_set(RT_SET_frame_rate_target, text, why, sizeof why));
}

// change: the target changes while the game runs, 40, 100, 0, 50, and each acquire after the change uses the new one; the
// game asks for interval 2 throughout, which the target ignores and the run without one honours. Counts by hand (31.25e6
// counts a second):
//   40: slot 1 is count 781,250.  A change to 100 at count 781,250: the present is due where it is, the next gate is slot 3 of
//   the 100 Hz schedule, the one after the slot (2) that count 781,250 falls in: 937,500. Off at 937,500: interval 2, the
//   60 Hz tick the count falls in is 1, so tick 3: 1,562,500. 50 at 1,562,500: it is in slot 2 of 50 a second, so slot 3:
//   1,875,000.
static void case_change(void) {
    store_load();
    objects(3, 2, 0, 0, NVN_ACQUIRE_WAIT_SYNC);
    guard_begin("an acquire or a present waited for a count with the clock stopped (the default placement)");
    uint64_t s0 = host_sync(), s1 = host_sync(), s2 = host_sync(), s3 = host_sync(), s4 = host_sync();
    set_rate("40");
    CHECK(host_acquire_sync(g_win, s0) == 0);
    CHECK(host_wait(s0, 0) == ALREADY_SIGNALED);
    host_present(g_win, 0);                            // due count 0, target 40
    CHECK(host_acquire_sync(g_win, s1) == 1);          // returns at once; the gate is slot 1 of 40: 781,250
    guard_end();
    expect_gate(s1, 781250);
    set_rate("100");                                   // applies at the next acquire; s1 keeps its count
    CHECK(host_wait(s1, 0) == ALREADY_SIGNALED);
    host_present(g_win, 1);                            // at 781,250: due there, target 100
    guard_begin("an acquire waited for a count with the clock stopped");
    CHECK(host_acquire_sync(g_win, s2) == 2);          // slot 3 of 100: 937,500
    guard_end();
    expect_gate(s2, 937500);
    host_present(g_win, 2);                            // at 937,500, target 100
    set_rate("0");
    guard_begin("an acquire waited for a count with the clock stopped");
    CHECK(host_acquire_sync(g_win, s3) == 0);          // the game's interval 2 again: tick 3, count 1,562,500
    guard_end();
    expect_gate(s3, 1562500);
    host_present(g_win, 0);                            // at 1,562,500, no target
    set_rate("50");
    guard_begin("an acquire waited for a count with the clock stopped");
    CHECK(host_acquire_sync(g_win, s4) == 1);          // slot 3 of 50: 1,875,000
    guard_end();
    expect_gate(s4, 1875000);
    host_present(g_win, 1);
    static const uint64_t due[5] = {0, 781250, 937500, 1562500, 1875000};
    static const int target[5] = {40, 100, 100, 0, 50};
    NvnPresentRecord r;
    CHECK(nvn_present_count() == 5);
    for (uint64_t k = 0; k < 5; k++) {
        CHECK(nvn_present_record(k + 1, &r));
        printf("present %llu: due count %llu tick %llu requested %d target %d\n", (ull)k + 1, (ull)r.due_count, (ull)r.due,
               r.requested, r.target);
        CHECK(r.due_count == due[k] && r.due == ref_slot(due[k], 60) && r.requested == 2 && r.target == target[k]);
    }
    printf("ok change\n");
}

// late: the game presents 3.4 slots late (target 40). The present is due at the slot it arrives in, and the next acquire's
// sync is signalled one slot after that: no burst of syncs that are already due to catch up with the lost slots.
static void case_late(void) {
    objects(3, 1, 0, 0, NVN_ACQUIRE_WAIT_SYNC);
    uint64_t rate = env_rate(), s = host_sync(), s2 = host_sync(), s3 = host_sync();
    CHECK(rate == 40);
    guard_begin("an acquire waited for a count with the clock stopped");
    CHECK(host_acquire_sync(g_win, s) == 0);
    host_present(g_win, 0);                            // due 0
    CHECK(host_acquire_sync(g_win, s) == 1);           // gate: slot 1, 781,250
    set_now(count_ns(4 * 781250 + 2 * 781250 / 5));    // slot 4 and 0.4 of the next: three slots and 0.4 past the gate
    CHECK(host_wait(s, 0) == ALREADY_SIGNALED);
    host_present(g_win, 1);                            // the slot in progress is 4: due 3,125,000
    CHECK(host_acquire_sync(g_win, s2) == 2);          // the gate is slot 5: 3,906,250
    CHECK(host_acquire_sync(g_win, s3) == 0);          // an acquire with no present between has the same gate
    guard_end();
    CHECK(host_wait(s2, 0) == TIMEOUT_EXPIRED && host_wait(s3, 0) == TIMEOUT_EXPIRED);   // not already due: no burst
    set_now(count_ns(3906250) - 1);
    CHECK(host_wait(s2, 0) == TIMEOUT_EXPIRED && host_wait(s3, 0) == TIMEOUT_EXPIRED);
    set_now(count_ns(3906250));
    CHECK(host_wait(s2, 0) == ALREADY_SIGNALED && host_wait(s3, 0) == ALREADY_SIGNALED);
    NvnPresentRecord r;
    CHECK(nvn_present_record(2, &r) && r.due_count == 3125000 && r.target == 40);
    printf("late: a present three slots and 0.4 late was due at its own slot, 3,125,000, and the next sync at slot 5, "
           "3,906,250\n");
    printf("ok late\n");
}

// null_sync: a game that passes no sync is not held under the default placement, with a target too
static void case_null_sync(void) {
    objects(3, 1, 0, 0, NVN_ACQUIRE_WAIT_SYNC);
    guard_begin("an acquire with no sync waited for a count with the clock stopped");
    for (int i = 0; i < 6; i++) {
        (void)host_acquire_sync(g_win, 0);
        host_present(g_win, (uint64_t)i % 3);
    }
    guard_end();
    NvnPresentRecord r;
    CHECK(nvn_present_record(6, &r) && r.due_count == 0 && r.target == (int32_t)env_rate() && r.target != 0);
    printf("ok null_sync\n");
}

// seam: the back end's present_pace gets the due count as well as the tick: a present made before its wait is told the
// due count of its gate, which is a slot still to come (target 40), and the record's due count is that
static uint64_t g_pace_count[8], g_pace_tick[8];
static int g_npace;
static void seam_pace(uint64_t seq, uint64_t tick, uint64_t count) {
    CHECK(g_npace < 8 && seq == (uint64_t)g_npace + 1);
    g_pace_tick[g_npace] = tick, g_pace_count[g_npace++] = count;
}
static GfxBackend g_seam;
static void case_seam(void) {
    g_seam = gfx_null_backend;
    g_seam.name = "pace-seam";
    g_seam.present_pace = seam_pace;
    g_backend = &g_seam;
    objects(3, 1, 0, 0, NVN_ACQUIRE_WAIT_SYNC);
    uint64_t s = host_sync(), s2 = host_sync();
    CHECK(env_rate() == 40);
    guard_begin("an acquire or a present waited for a count with the clock stopped (the default placement)");
    CHECK(host_acquire_sync(g_win, s) == 0);
    host_present(g_win, 0);                    // the first frame: due 0, in the slot in progress
    CHECK(host_acquire_sync(g_win, s) == 1);   // the gate is slot 1 of 40, 781,250, and the clock stands at 0
    host_present(g_win, 1);                    // before its wait: due 781,250, which is 1.5 ticks: the tick it falls in is 1
    CHECK(host_acquire_sync(g_win, s2) == 2);  // the gate is slot 2: 1,562,500
    host_present(g_win, 2);                    // before its wait: due 1,562,500, tick 3
    guard_end();
    static const uint64_t count[3] = {0, 781250, 1562500}, tick[3] = {0, 1, 3};
    CHECK(g_npace == 3);
    NvnPresentRecord r;
    for (int k = 0; k < 3; k++) {
        CHECK(g_pace_count[k] == count[k] && g_pace_tick[k] == tick[k]);
        CHECK(nvn_present_record((uint64_t)k + 1, &r) && r.due_count == count[k] && r.due == tick[k] && r.target == 40);
    }
    printf("seam: the back end was told the due counts 0, 781,250 and 1,562,500 and the ticks 0, 1 and 3 they fall in\n");
    printf("ok seam\n");
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

// realtime <seconds>: the host's clock, the target from the environment, the game waiting on the sync with no timeout: no
// wait ends before its due count, the frames take about the right time, and the thread that waits burns no processor:
// the game thread's CPU time over the run stays under a third of the wall time (a spin would be all of it).
static void case_realtime(uint64_t seconds) {
    uint64_t rate = env_rate();
    CHECK(rate != 0);
    g_hz = rate, g_p = 1, g_order = ORDER_WAIT_PRESENT, g_nframes = seconds * rate + 1;
    CHECK(g_nframes <= MAXF);
    objects(3, 1, UNDRIVEN, 0, NVN_ACQUIRE_WAIT_SYNC);
    uint64_t t0 = plat_now_ns();
    rt_kernel_run(G(0), 0);
    uint64_t span = plat_now_ns() - t0;
    uint64_t late_max = 0, late_sum = 0, n = 0;
    static uint64_t late[MAXF];
    for (uint64_t k = 1; k < g_nframes; k++) {
        CHECK(g_sync_res[k] == CONDITION_SATISFIED || g_sync_res[k] == ALREADY_SIGNALED);
        uint64_t gate = ref_count(ref_slot(g_rec[k - 1].due_count, rate) + 1, rate);
        CHECK(g_sync_cnt[k] >= gate);   // never early
        late[n] = (g_sync_cnt[k] - gate) * 32ull;
        late_sum += late[n], late_max = late[n] > late_max ? late[n] : late_max, n++;
    }
    uint64_t cpu = g_cpu_end - g_cpu_start;
    CHECK(span >= (seconds * 1000000000ull) * 9 / 10);
    CHECK(cpu * 3 < span);
    qsort(late, n, sizeof *late, cmp_u64);
    printf("realtime: %llu presents at %llu a second took %.0f ms of the host's clock; no wait ended early; the game thread "
           "used %.1f ms of CPU (%.1f percent), lateness of the wake after its due count: mean %.0f us, p50 %.0f us, "
           "p99 %.0f us, worst %.0f us\n", (ull)g_nframes, (ull)rate, (double)span / 1e6, (double)cpu / 1e6,
           100.0 * (double)cpu / (double)span, (double)late_sum / (double)n / 1e3, (double)late[n / 2] / 1e3,
           (double)late[n * 99 / 100] / 1e3, (double)late_max / 1e3);
    printf("ok realtime\n");
}

// schedule: the primitives against the model by hand: the due count of slot m is ceil(m * 31.25e6 / hz), a count is in slot
// floor(count * hz / 31.25e6), the schedule at 60 is the clock's ticks, at 30 and 20 every second and third tick, and an hour
// of any rate is exactly 3600 * 31.25e6 counts with no drift.
static void case_schedule(void) {
    static const uint32_t hzs[] = {20, 30, 40, 45, 50, 60, 100, 120};
    for (unsigned i = 0; i < sizeof hzs / sizeof *hzs; i++) {
        uint64_t hz = hzs[i];
        CHECK(rt_sched_due_counter(0, hzs[i]) == 0);
        CHECK(rt_sched_due_counter(hz * 3600, hzs[i]) == 3600 * COUNTS);   // an hour, exactly
        for (uint64_t m = 1; m <= 400000; m += (m < 1000 ? 1 : 997)) {
            uint64_t d = rt_sched_due_counter(m, hzs[i]);
            CHECK(d == ref_count(m, hz));
            CHECK(rt_sched_slot_of(d, hzs[i]) == m && rt_sched_slot_of(d - 1, hzs[i]) == m - 1);
        }
        CHECK(rt_sched_due_counter(1ull << 36, hzs[i]) == ref_count(1ull << 36, hz));
    }
    for (uint64_t n = 0; n <= 5000; n++) {   // the clock's own tick is the schedule at 60; at 30 and 20 slot m is tick 2m and 3m
        CHECK(rt_vsync_due_counter(n) == rt_sched_due_counter(n, 60));
        CHECK(rt_sched_due_counter(n, 30) == rt_vsync_due_counter(2 * n));
        CHECK(rt_sched_due_counter(n, 20) == rt_vsync_due_counter(3 * n));
    }
    CHECK(rt_sched_due_counter(1, 40) == 781250 && rt_sched_due_counter(3, 100) == 937500 &&
          rt_sched_due_counter(3, 60) == 1562500 && rt_sched_due_counter(2, 60) == 1041667);
    printf("ok schedule\n");
}

// A case that has not ended after HANG_NS exits 3 and dumps the kernel.
#define HANG_NS 170000000000ull
static void watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(HANG_NS);
    rt_kernel_dump();
    _Exit(3);
}

static uint64_t arg_u(char **argv, int i) { return strtoull(argv[i], NULL, 10); }

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: nvn_target_test <case> [args]\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    const char *cs = argv[1];
    if (!strcmp(cs, "hour") && argc == 7) case_hour(arg_u(argv, 2), arg_u(argv, 3), (int)arg_u(argv, 4), (int)arg_u(argv, 5), arg_u(argv, 6));
    else if (!strcmp(cs, "events") && argc == 4) case_events(arg_u(argv, 2), arg_u(argv, 3));
    else if (!strcmp(cs, "script") && argc == 6) case_script((int)arg_u(argv, 2), (int)arg_u(argv, 3), arg_u(argv, 4), arg_u(argv, 5));
    else if (!strcmp(cs, "change")) case_change();
    else if (!strcmp(cs, "late")) case_late();
    else if (!strcmp(cs, "null_sync")) case_null_sync();
    else if (!strcmp(cs, "seam")) case_seam();
    else if (!strcmp(cs, "realtime") && argc == 3) case_realtime(arg_u(argv, 2));
    else if (!strcmp(cs, "schedule")) case_schedule();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
