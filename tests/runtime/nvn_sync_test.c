// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the acquire sync (ticket 95; architecture D22; runtime/nvn/nvn_window.c, runtime/nvn/nvn_queue.c,
// oot2/nvn_pace.h), driven by tests/test_runtime_nvn_sync.py: `nvn_sync_test <case>`. Each case runs in its own process
// and prints "ok <case>".
//
// Under the default placement AcquireTexture returns at once and the sync it returns is signalled at the gate tick: the
// previous present's due tick plus the interval read at the acquire. These cases show that with a synthetic guest thread
// written by hand as a host function in the lifter's shape (as tests/runtime/nvn_pace_test.c does for the old placement):
// it calls NVN entry points through their synthetic addresses (nvnSyncInitialize, nvnWindowAcquireTexture, nvnSyncWait,
// nvnQueuePresentTexture, nvnWindowSetPresentInterval) with the guest's arguments, and records the tick each acquire came
// back at, the tick each SyncWait came back at and the record each present left. No game code, no dump.
//
// The clock is driven (rt_vsync_drive) and nothing binds a vsync event, so the clock thread never starts: the sync waits
// on rt_vsync_tick() through rt_vsync_wait_tick_for, which follows the driver's time, and a test thread moves it. The hour
// cases move it tick by tick, each step only once the game has done what the tick allows, so an hour runs in seconds. The
// expected ticks are written here by hand, never computed with the front end's rule.
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
#define SYNC_ALL 0u                  // NVN_SYNC_CONDITION_ALL_GPU_COMMANDS_COMPLETE

// The schedule, written here without vsync.c: 31,250,000 counts a second of 32 ns, tick n due at ceil(n * 31.25e6 / 60).
#define ORIGIN 123456789012ull
#define HOUR_TICKS 216000ull
static uint64_t ref_due_ns(uint64_t n) { return ORIGIN + ((n * 31250000ull + 59) / 60) * 32ull; }

static Cpu c;   // the host thread's Cpu: entry points called before the run
static uint64_t heap, heap_used, loader, g_area;
static uint64_t g_dev, g_queue, g_win;
static uint64_t f_acquire, f_present, f_setint, f_syncinit, f_syncwait, f_fencesync;

// ---------- the driven time
static uint64_t g_now;   // atomic
static uint64_t drv_now(void *ctx) {
    (void)ctx;
    return __atomic_load_n(&g_now, __ATOMIC_ACQUIRE);
}
static void drv_wait(void *ctx, uint64_t deadline) {   // the clock thread never starts in these cases
    (void)ctx, (void)deadline;
    fprintf(stderr, "FAIL: the clock thread waited\n");
    exit(1);
}
static void set_now(uint64_t ns) {
    CHECK(ns >= __atomic_load_n(&g_now, __ATOMIC_RELAXED));
    __atomic_store_n(&g_now, ns, __ATOMIC_RELEASE);
}
static void drive_clock(uint64_t first_tick) {
    g_now = ref_due_ns(first_tick);
    RtVsyncDriver d = {.origin_ns = ORIGIN, .now = drv_now, .wait_until = drv_wait};
    rt_vsync_drive(&d);
}

// ---------- the guest side
enum { IDX = 0x0, WORK = 0x40, DONE = 0x48, HANDLES = 0x100, SYNC = 0x200, STACKS = 0x100000, AREA = 0x120000 };
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

#define MAXF 216002
typedef struct { int set_acq, set_mid, set_pres; } Step;   // SetPresentInterval before the acquire / between the acquire and the wait / before the present
static const Step *g_steps;
static uint64_t g_nframes;
static int g_handshake;                    // the game waits for GO after its acquire and wait (the driver's "work" time)
static int g_order;
static uint64_t g_timeout = ~0ull;         // the timeout the game passes to SyncWait
static uint64_t A_ACQ, A_SYNC, A_GO, A_PRES;   // acquires returned, waits returned, presents allowed, presents done (atomic)
static uint64_t g_ret_tick[MAXF];          // the tick each acquire returned at, as the game read it
static uint64_t g_sync_tick[MAXF];         // the tick each SyncWait on the acquire's sync returned at
static uint32_t g_sync_res[MAXF];          // its result
static NvnPresentRecord g_rec[MAXF];       // the record each present left, read right after it

static void game_wait(Cpu *t, uint64_t k) {
    g_sync_res[k] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), g_timeout, 0));
    g_sync_tick[k] = rt_vsync_tick();
    st_rel(&A_SYNC, k + 1);
}

static void game_body(Cpu *t) {
    CHECK((gcall(t, f_syncinit, W(SYNC), g_dev, 0) & 0xff) == 1);   // one sync, reused by every acquire
    for (uint64_t k = 0; k < g_nframes; k++) {
        if (g_steps && g_steps[k].set_acq) gcall(t, f_setint, g_win, (uint64_t)g_steps[k].set_acq, 0);
        CHECK(LOW32(gcall(t, f_acquire, g_win, W(SYNC), W(IDX))) == 0);
        g_ret_tick[k] = rt_vsync_tick();
        st_rel(&A_ACQ, k + 1);
        if (g_steps && g_steps[k].set_mid) gcall(t, f_setint, g_win, (uint64_t)g_steps[k].set_mid, 0);
        if (g_order == ORDER_WAIT_PRESENT) game_wait(t, k);
        if (g_handshake) {
            while (ld_acq(&A_GO) < k + 1) sv(t, 0x1E, 0, 0, 0, 0);   // GetSystemTick: the exit is a safe point
        }
        if (g_steps && g_steps[k].set_pres) gcall(t, f_setint, g_win, (uint64_t)g_steps[k].set_pres, 0);
        uint32_t idx = *(volatile uint32_t *)(uintptr_t)W(IDX);
        t->x[0] = g_queue, t->x[1] = g_win, t->x[2] = idx, t->x[3] = 0, t->x[30] = PC + 4, t->pc = f_present;
        rt_dispatch(t);
        CHECK(nvn_present_record(nvn_present_count(), &g_rec[k]));
        st_rel(&A_PRES, k + 1);
        if (g_order == ORDER_PRESENT_WAIT) game_wait(t, k);
    }
}

// 0: the game alone
static void g_main_game(Cpu *t) {
    game_body(t);
    sv(t, 0x0A, 0, 0, 0, 0);
}

// 1: a worker of a worse priority on the same core, counting while the game thread gives the core up
static void g_worker(Cpu *t) {
    while (!ld_acq((const uint64_t *)(uintptr_t)W(DONE))) {
        st_rel((uint64_t *)(uintptr_t)W(WORK), ld_acq((const uint64_t *)(uintptr_t)W(WORK)) + 1);
        sv(t, 0x1E, 0, 0, 0, 0);
    }
    sv(t, 0x0A, 0, 0, 0, 0);
}

// 2: the game, with the worker started first
static void g_main_block(Cpu *t) {
    t->x[1] = G(1), t->x[2] = 0, t->x[3] = W(STACKS) + 0x10000, t->x[4] = 50, t->x[5] = 0;
    rt_svc(t, 0x08, PC);   // CreateThread
    CHECK(t->x[0] == 0);
    uint32_t h = (uint32_t)t->x[1];
    CHECK(sv(t, 0x09, h, 0, 0, 0) == 0);   // StartThread
    game_body(t);
    st_rel((uint64_t *)(uintptr_t)W(DONE), 1);
    *(volatile uint32_t *)(uintptr_t)W(HANDLES) = h;
    CHECK(sv(t, 0x18, 0, W(HANDLES), 1, ~0ull) == 0);   // WaitSynchronization: the worker ended
    CHECK(sv(t, 0x16, h, 0, 0, 0) == 0);                // CloseHandle
    sv(t, 0x0A, 0, 0, 0, 0);
}

// 3: the timeouts. The window is at interval 4 and the clock stands at a tick: the second acquire's sync is signalled 4
// ticks after the first present's due tick. The game asks with timeout 0, then 5 ms, then 100 ms; the driver moves the time.
static uint64_t A_R2;   // the 5 ms wait came back
static uint32_t g_r[5];
static uint64_t g_r_tick[5];
static void g_main_timeout(Cpu *t) {
    CHECK((gcall(t, f_syncinit, W(SYNC), g_dev, 0) & 0xff) == 1);
    CHECK(LOW32(gcall(t, f_acquire, g_win, W(SYNC), W(IDX))) == 0);   // the first: free
    t->x[0] = g_queue, t->x[1] = g_win, t->x[2] = 0, t->x[3] = 0, t->x[30] = PC + 4, t->pc = f_present;
    rt_dispatch(t);
    CHECK(LOW32(gcall(t, f_acquire, g_win, W(SYNC), W(IDX))) == 0);   // signalled 4 ticks later
    st_rel(&A_ACQ, 1);
    g_r[0] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), 0, 0)), g_r_tick[0] = rt_vsync_tick();
    st_rel(&A_SYNC, 1);
    g_r[1] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), 5000000ull, 0)), g_r_tick[1] = rt_vsync_tick();
    st_rel(&A_R2, 1);
    g_r[2] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), 100000000ull, 0)), g_r_tick[2] = rt_vsync_tick();
    g_r[3] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), 0, 0)), g_r_tick[3] = rt_vsync_tick();
    st_rel(&A_SYNC, 2);
    sv(t, 0x0A, 0, 0, 0, 0);
}

// 4: the timeout on the host's clock (interval 4): 5 ms ends a wait for a tick still 3 or more ticks off, an unlimited wait
// then reaches it
static uint32_t g_rt[3];
static uint64_t g_rt_ms[3];
static void g_main_realtime_timeout(Cpu *t) {
    CHECK((gcall(t, f_syncinit, W(SYNC), g_dev, 0) & 0xff) == 1);
    CHECK(LOW32(gcall(t, f_acquire, g_win, W(SYNC), W(IDX))) == 0);
    t->x[0] = g_queue, t->x[1] = g_win, t->x[2] = 0, t->x[3] = 0, t->x[30] = PC + 4, t->pc = f_present;
    rt_dispatch(t);
    CHECK(LOW32(gcall(t, f_acquire, g_win, W(SYNC), W(IDX))) == 0);
    uint64_t t0 = plat_now_ns();
    g_rt[0] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), 5000000ull, 0));
    g_rt_ms[0] = (plat_now_ns() - t0) / 1000000ull;
    t0 = plat_now_ns();
    g_rt[1] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), 10000000000ull, 0));   // 10 s: the tick is 4 ticks off at most
    g_rt_ms[1] = (plat_now_ns() - t0) / 1000000ull;
    g_rt[2] = (uint32_t)LOW32(gcall(t, f_syncwait, W(SYNC), 0, 0));
    sv(t, 0x0A, 0, 0, 0, 0);
}

// 5 and 6: a wait on a sync no acquire has fenced yet, and the thread that fences it. The game is parked on the sync (no
// timeout) while the waker sleeps; the waker's acquire must wake it.
static uint64_t g_wake_sync, g_wake_tick;
static uint32_t g_wake_res;
static void g_main_wake(Cpu *t) {
    t->x[1] = G(6), t->x[2] = 0, t->x[3] = W(STACKS) + 0x10000, t->x[4] = 50, t->x[5] = 0;
    rt_svc(t, 0x08, PC);   // CreateThread
    CHECK(t->x[0] == 0);
    uint32_t h = (uint32_t)t->x[1];
    CHECK(sv(t, 0x09, h, 0, 0, 0) == 0);   // StartThread
    g_wake_res = (uint32_t)LOW32(gcall(t, f_syncwait, g_wake_sync, ~0ull, 0));
    g_wake_tick = rt_vsync_tick();
    *(volatile uint32_t *)(uintptr_t)W(HANDLES) = h;
    CHECK(sv(t, 0x18, 0, W(HANDLES), 1, ~0ull) == 0);   // WaitSynchronization: the waker ended
    CHECK(sv(t, 0x16, h, 0, 0, 0) == 0);                // CloseHandle
    sv(t, 0x0A, 0, 0, 0, 0);
}
static void g_waker(Cpu *t) {
    sv(t, 0x0B, 100000000ull, 0, 0, 0);   // SleepThread 100 ms: the game is parked on the unfenced sync by then
    CHECK(LOW32(gcall(t, f_acquire, g_win, g_wake_sync, W(IDX))) == 0);   // the first acquire: no tick
    sv(t, 0x0A, 0, 0, 0, 0);
}

static void guest_end(Cpu *t) { (void)t; }
static const GuestFn g_fns[] = {g_main_game, g_worker, g_main_block, g_main_timeout, g_main_realtime_timeout,
                                g_main_wake, g_waker, guest_end};
#define NFN ((int)(sizeof g_fns / sizeof *g_fns))
static uint64_t g_addrs[NFN];

// ---------- set-up
static const uint64_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static void setup(const GfxBackend *b) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512 * MiB, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256 * MiB, .system_resource_size = MiB};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x1, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                        .program_id = 0x0100000000001000ull, .is_application = 1, .no_async_preempt = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++)
        k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    if (b) gfx_set_backend(b);
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
// and stands at first_tick (UNDRIVEN: the host's own clock). hold: the experiment hold, set explicitly so the environment
// cannot change a case; -1 leaves it as the process starts. placement: the acquire placement, set the same way.
static void objects(const GfxBackend *b, int n, int interval, uint64_t first_tick, int hold, int placement) {
    setup(b);
    if (first_tick != UNDRIVEN) drive_clock(first_tick);
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
    f_fencesync = proc("nvnQueueFenceSync");
}

// ---------- the drivers, host threads
static void fail(const char *what, uint64_t k) {
    fprintf(stderr, "FAIL frame %llu: %s\n", (ull)k, what);
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

// A call that must come back with the clock stopped: if it waits for a tick, the guard ends the case after 5 s.
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

static uint64_t g_hour_interval, g_total_ticks = HOUR_TICKS;
static uint64_t min_u64(uint64_t a, uint64_t b) { return a < b ? a : b; }
// The hour: tick t+1 is reached only once the game has done everything the tick allows, and not more. Where the game
// presents after its wait, the waits that ended by tick t are frames 0 to t/I and the presents the same; where it presents
// before, the present of the next frame is made too (its sync is what the game then waits on). A wait that came back early,
// or a present ahead of that, shows as a count ahead of its tick. Each tick is entered 0 to 15 ms into it.
static void driver_hour(void *arg) {
    (void)arg;
    uint64_t I = g_hour_interval;
    for (uint64_t t = 0;; t++) {
        uint64_t want_sync = min_u64(t / I + 1, g_nframes);
        uint64_t want_pres = min_u64(t / I + 1 + (g_order == ORDER_PRESENT_WAIT), g_nframes);
        // the acquire that follows the last present (or the last wait) has returned before the clock moves: it never waits
        uint64_t want_acq = min_u64(g_order == ORDER_WAIT_PRESENT ? want_pres + 1 : want_pres, g_nframes);
        await(&A_SYNC, want_sync, t, "the game's wait on the acquire sync has not ended (the sync is signalled too late)");
        await(&A_PRES, want_pres, t, "the game has not presented the frames this tick allows");
        await(&A_ACQ, want_acq, t, "an acquire did not come back at once");
        if (ld_acq(&A_SYNC) > want_sync) fail("a wait on the acquire sync ended ahead of its tick (signalled too early)", t);
        if (ld_acq(&A_PRES) > want_pres) fail("the game presented a frame ahead of its tick", t);
        if (ld_acq(&A_ACQ) > want_acq) fail("an acquire came back ahead of its tick", t);
        if (t == g_total_ticks) break;
        set_now(ref_due_ns(t + 1) + (t * 7919 % 16) * 1000000ull);
    }
}

typedef struct { uint64_t gate, work_ticks; int work_milli, due, requested, held; } Exp;   // the model's answer, by hand
static const Exp *g_exp;
static int g_script_placement;   // the placement the script expects
// The script (the game waits on its sync, then "works", then presents). Default placement: the acquire must come back at
// once, with the clock stopped before the gate tick, and the sync must stay unsignalled until the clock is at the gate tick,
// then end the wait at it. Old placement: the acquire is the held call, the wait on the sync ends at once. Then the game
// "works" for work_ticks and work_milli thousandths of a tick past the gate tick before it presents.
static void driver_script(void *arg) {
    (void)arg;
    for (uint64_t k = 0; k < g_nframes; k++) {
        const Exp *e = &g_exp[k];
        if (e->gate > rt_vsync_tick()) {
            if (g_script_placement == NVN_ACQUIRE_WAIT_SYNC)
                await(&A_ACQ, k + 1, k, "the acquire did not come back at once");
            set_now(ref_due_ns(e->gate) - 1);   // a nanosecond before the gate tick: still held, or still not signalled
            plat_sleep_ns(5000000ull);   // only a probe: the recorded ticks are the authoritative check
            if (g_script_placement == NVN_ACQUIRE_WAIT_SYNC) {
                if (ld_acq(&A_SYNC) != k) fail("the wait on the acquire sync ended before its gate tick", k);
            } else if (ld_acq(&A_ACQ) != k) {
                fail("the acquire came back before its gate tick", k);
            }
            set_now(ref_due_ns(e->gate));
        }
        await(&A_SYNC, k + 1, k, "the wait on the acquire sync did not end at its gate tick");
        uint64_t w = e->gate + e->work_ticks;
        uint64_t arrive = ref_due_ns(w) + (ref_due_ns(w + 1) - ref_due_ns(w)) * (uint64_t)e->work_milli / 1000;
        if (arrive > __atomic_load_n(&g_now, __ATOMIC_RELAXED)) set_now(arrive);
        st_rel(&A_GO, k + 1);
        await(&A_PRES, k + 1, k, "the present did not come back");
    }
}

// The result of the wait on the acquire sync: nothing, at once, for the first acquire and the old placement; otherwise the
// wait that had to block (CONDITION_SATISFIED) or, when the driver had already moved the clock to the tick before the game
// got to its wait, ALREADY_SIGNALED: the tick, not the result, is what the cases check.
static int sync_res_ok(uint32_t res, uint64_t k, int placement) {
    if (k == 0 || placement == NVN_ACQUIRE_WAIT_ACQUIRE) return res == ALREADY_SIGNALED;
    return res == CONDITION_SATISFIED || res == ALREADY_SIGNALED;
}

static void check_script(const Exp *exp, uint64_t n) {
    for (uint64_t k = 0; k < n; k++) {
        // the acquire's own tick: the tick of the last present (default), or the gate tick (the call waited)
        uint64_t acq = g_script_placement == NVN_ACQUIRE_WAIT_SYNC ? (k ? (uint64_t)exp[k - 1].due : 0) : exp[k].gate;
        printf("frame %llu acquired at tick %llu synced at tick %llu due %llu requested %d held %d placement %d\n",
               (ull)k, (ull)g_ret_tick[k], (ull)g_sync_tick[k], (ull)g_rec[k].due, g_rec[k].requested, g_rec[k].held,
               g_rec[k].placement);
        CHECK(g_ret_tick[k] == acq);
        CHECK(g_sync_tick[k] == exp[k].gate);
        CHECK(sync_res_ok(g_sync_res[k], k, g_script_placement));
        CHECK(g_rec[k].due == (uint64_t)exp[k].due);
        CHECK(g_rec[k].requested == exp[k].requested);
        CHECK(g_rec[k].held == exp[k].held);
        CHECK(g_rec[k].placement == g_script_placement);
        CHECK(g_rec[k].seq == k + 1 && g_rec[k].shown == 0);
    }
}

static void run_script(const Exp *exp, const Step *steps, uint64_t n, int window_interval, int hold, int placement,
                       int expect_placement) {
    g_exp = exp, g_steps = steps, g_nframes = n, g_handshake = 1, g_order = ORDER_WAIT_PRESENT;
    g_script_placement = expect_placement;
    objects(NULL, 3, window_interval, 0, hold, placement);
    CHECK(nvn_acquire_wait_mode() == expect_placement);
    CHECK(plat_thread_start(driver_script, NULL, 1ull << 20));
    rt_kernel_run(G(0), 0);
    check_script(exp, n);
}

// ---------- cases
// hour: 216,000 driven ticks. Frame k's acquire comes back at once, at the tick of frame k-1's present; the wait on its sync
// ends at tick k*I, the frame's present is due then (a tick still to come when the game presents before its wait), and the
// record keeps what the game asked for. The wait of frame 0 is for nothing.
static void case_hour(uint64_t interval, int order) {
    g_hour_interval = interval, g_order = order;
    g_nframes = g_total_ticks / interval + 1;
    objects(NULL, 3, (int)interval, 0, -1, PLACEMENT_ENV);
    CHECK(nvn_present_hold_ticks() == 0 && nvn_acquire_wait_mode() == NVN_ACQUIRE_WAIT_SYNC);   // the defaults
    CHECK(plat_thread_start(driver_hour, NULL, 1ull << 20));
    uint64_t t0 = plat_now_ns();
    rt_kernel_run(G(0), 0);
    uint64_t n = g_nframes;
    for (uint64_t k = 0; k < n; k++) {
        uint64_t ret = k ? (k - 1) * interval : 0;
        if (g_ret_tick[k] != ret || g_sync_tick[k] != k * interval || g_rec[k].due != k * interval ||
            !sync_res_ok(g_sync_res[k], k, NVN_ACQUIRE_WAIT_SYNC) || g_rec[k].requested != (int)interval ||
            g_rec[k].held != 0 || g_rec[k].placement != NVN_ACQUIRE_WAIT_SYNC || g_rec[k].shown != 0 ||
            g_rec[k].seq != k + 1 || g_rec[k].texture != (int)(k % 3)) {
            fprintf(stderr, "FAIL frame %llu: acquire came back at tick %llu (want %llu), wait ended at tick %llu (want "
                            "%llu) with result %u, due %llu, requested %d, held %d, placement %d\n", (ull)k,
                    (ull)g_ret_tick[k], (ull)ret, (ull)g_sync_tick[k], (ull)(k * interval), g_sync_res[k],
                    (ull)g_rec[k].due, g_rec[k].requested, g_rec[k].held, g_rec[k].placement);
            exit(1);
        }
    }
    CHECK(nvn_present_count() == n && gfx_null_presents() == n);
    printf("hour: %llu presents at interval %llu, the game %s its wait on the acquire sync; every acquire came back at once, "
           "every wait ended %llu ticks apart from tick 0 to tick %llu, in %.1f s of wall time\n", (ull)n,
           (ull)interval, order == ORDER_WAIT_PRESENT ? "presenting after" : "presenting before", (ull)interval,
           (ull)((n - 1) * interval), (double)(plat_now_ns() - t0) / 1e9);
    printf("ok hour%llu%s\n", (ull)interval, order == ORDER_WAIT_PRESENT ? "" : "_pwait");
}

// The game thread gives its core up while it waits on the acquire sync: a worker of a worse priority on the same core
// counts meanwhile, and the game takes the core back when the tick comes. The acquire itself came back at once.
static void driver_block(void *arg) {
    (void)arg;
    await(&A_PRES, 1, 0, "the first present did not come back");
    await(&A_ACQ, 2, 1, "the second acquire did not come back at once");
    await((const uint64_t *)(uintptr_t)W(WORK), 1000, 0,
          "the worker never ran while the game thread waited on the acquire sync: it kept its virtual core");
    if (ld_acq(&A_SYNC) != 1) fail("the wait on the second acquire's sync ended with the clock at tick 0", 1);
    printf("block: the worker counted %llu while the game thread waited on the acquire sync\n",
           (ull)ld_acq((const uint64_t *)(uintptr_t)W(WORK)));
    set_now(ref_due_ns(1));
    await(&A_SYNC, 2, 1, "the game thread did not take its core back at the tick");
    await(&A_PRES, 2, 1, "the second present did not come back");
}
static void case_block(void) {
    g_nframes = 2, g_order = ORDER_WAIT_PRESENT;
    objects(NULL, 2, 1, 0, 0, PLACEMENT_ENV);
    CHECK(plat_thread_start(driver_block, NULL, 1ull << 20));
    rt_kernel_run(G(2), 0);
    CHECK(g_ret_tick[0] == 0 && g_ret_tick[1] == 0 && g_sync_tick[1] == 1 && g_rec[1].due == 1);
    printf("block: acquire 2 came back at tick %llu, its sync ended the wait at tick %llu\n", (ull)g_ret_tick[1],
           (ull)g_sync_tick[1]);
    printf("ok block\n");
}

// The calls below run on the host thread with the clock stopped. A call that waits for a tick never comes back; the
// guard ends the case after 5 s.
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

// The acquire returns with the clock stopped, however long the gate is; the sync is not signalled until the clock is at the
// gate tick (a wait with timeout 0 says TIMEOUT_EXPIRED a nanosecond before it and ALREADY_SIGNALED at it); a present of
// the same acquire, before the wait, is due at the gate tick; the records keep the interval the game asked for.
static void case_acquire_nowait(void) {
    objects(NULL, 3, 2, 5, 0, PLACEMENT_ENV);
    uint64_t s = host_sync();
    guard_begin("AcquireTexture, or a present, waited for a tick with the clock stopped (the default placement)");
    CHECK(host_acquire_sync(g_win, s) == 0);
    CHECK(host_wait(s, 0) == ALREADY_SIGNALED);   // the first acquire: signalled at once
    host_present(g_win, 0);                        // due 5
    CHECK(host_acquire_sync(g_win, s) == 1);       // returns at once; gate tick 5 + 2
    CHECK(rt_vsync_tick() == 5);
    CHECK(host_wait(s, 0) == TIMEOUT_EXPIRED);
    set_now(ref_due_ns(7) - 1);
    CHECK(host_wait(s, 0) == TIMEOUT_EXPIRED);
    host_present(g_win, 1);                        // before its wait: due at the gate tick 7, not at the tick in progress 6
    uint64_t s2 = host_sync();
    CHECK(host_acquire_sync(g_win, s2) == 2);      // with no wait in between: gate tick 7 + 2 = 9, not the clock's 6 + 2
    guard_end();
    set_now(ref_due_ns(7));
    CHECK(host_wait(s, 0) == ALREADY_SIGNALED);
    CHECK(host_wait(s2, 0) == TIMEOUT_EXPIRED);
    set_now(ref_due_ns(9) - 1);
    CHECK(host_wait(s2, 0) == TIMEOUT_EXPIRED);
    set_now(ref_due_ns(9));
    CHECK(host_wait(s2, 0) == ALREADY_SIGNALED);
    NvnPresentRecord r;
    CHECK(nvn_present_count() == 2 && nvn_present_record(1, &r) && r.due == 5 && r.requested == 2 && r.held == 0);
    CHECK(nvn_present_record(2, &r) && r.due == 7 && r.requested == 2 && r.texture == 1 &&
          r.placement == NVN_ACQUIRE_WAIT_SYNC);
    printf("acquire: the second acquire came back at once at tick 5; its sync said TIMEOUT_EXPIRED until tick 7 and "
           "ALREADY_SIGNALED at it; the present made before the wait is due at tick 7; the acquire after it is "
           "signalled at tick 9\n");
    printf("ok acquire_nowait\n");
}

// The first acquire of a window, and the acquires before any present, signal their sync at once, whatever the interval and
// the hold; so does the first of another window. The clock stands at tick 7, stopped; the interval is 3 and the hold 10.
static void case_first(void) {
    objects(NULL, 3, 3, 7, 10, PLACEMENT_ENV);
    uint64_t w2 = window(2, 3), s = host_sync(), s2 = host_sync();
    guard_begin("an acquire with no previous present held something with the clock stopped");
    CHECK(host_acquire_sync(g_win, s) == 0 && host_wait(s, 0) == ALREADY_SIGNALED);
    CHECK(host_acquire_sync(g_win, s) == 1 && host_wait(s, 0) == ALREADY_SIGNALED);   // two in a row, no present between
    host_present(g_win, 0);                                                           // this window has presented
    CHECK(host_acquire_sync(w2, s2) == 0 && host_wait(s2, 0) == ALREADY_SIGNALED);    // the other has not
    CHECK(host_acquire_sync(g_win, s) == 2 && host_wait(s, 0) == TIMEOUT_EXPIRED);    // this one has: 7 + max(3, 10)
    guard_end();
    NvnPresentRecord r;
    CHECK(nvn_present_count() == 1 && nvn_present_record(1, &r) && r.due == 7 && r.requested == 3 && r.held == 10);
    printf("first: acquires with no previous present signalled their sync at once at tick 7 (interval 3, hold 10); the "
           "acquire after the present did not\n");
    printf("ok first\n");
}

// A game that passes no sync cannot wait on anything: nothing is held, and its presents take the tick in progress.
static void case_null_sync(void) {
    objects(NULL, 3, 3, 5, 0, PLACEMENT_ENV);
    guard_begin("an acquire without a sync held something with the clock stopped");
    CHECK(host_acquire_sync(g_win, 0) == 0);
    host_present(g_win, 0);
    CHECK(host_acquire_sync(g_win, 0) == 1);   // would be the gate tick 8
    host_present(g_win, 1);
    guard_end();
    NvnPresentRecord r;
    CHECK(nvn_present_record(1, &r) && r.due == 5 && nvn_present_record(2, &r) && r.due == 5 && r.requested == 3);
    printf("null_sync: acquires with no sync came back at once at tick 5 and their presents are due at tick 5\n");
    printf("ok null_sync\n");
}

// A queue's fence on the sync replaces what the acquire gave it: the tick goes, only the fence value stays.
static void case_fence_clears_tick(void) {
    objects(NULL, 3, 2, 5, 0, PLACEMENT_ENV);
    uint64_t s = host_sync();
    guard_begin("an acquire waited for a tick with the clock stopped");
    CHECK(host_acquire_sync(g_win, s) == 0);
    host_present(g_win, 0);
    CHECK(host_acquire_sync(g_win, s) == 1 && host_wait(s, 0) == TIMEOUT_EXPIRED);   // gate tick 7
    uint64_t a[4] = {g_queue, s, SYNC_ALL, 0};
    vcall(f_fencesync, a, 4);
    CHECK(host_wait(s, 0) == ALREADY_SIGNALED);                                       // a queue fence has no tick
    guard_end();
    printf("fence: QueueFenceSync on a sync an acquire gave a tick to cleared the tick\n");
    printf("ok fence_clears_tick\n");
}

// A back end whose completed fence value the case sets: the sync is signalled when the tick has been reached AND the fence
// value of the texture's last present is done.
static uint64_t g_done_fence;   // atomic
static uint64_t ctl_done(void) { return ld_acq(&g_done_fence); }
static int ctl_wait(uint64_t value, uint64_t timeout_ns) {   // never waits: the value is done or it is not
    (void)timeout_ns;
    return ld_acq(&g_done_fence) >= value;
}
static GfxBackend g_ctl;
static void case_fence_and_tick(void) {
    g_ctl = gfx_null_backend;
    g_ctl.name = "ctl";
    g_ctl.fence_done = ctl_done, g_ctl.fence_wait = ctl_wait;
    objects(&g_ctl, 2, 2, 5, 0, PLACEMENT_ENV);
    uint64_t s = host_sync();
    ((queue_t *)(uintptr_t)g_queue)->last = 3;   // the queue has reached fence value 3: what each present records
    guard_begin("an acquire waited for a tick with the clock stopped");
    CHECK(host_acquire_sync(g_win, s) == 0);
    host_present(g_win, 0);                        // due 5, texture 0 fenced at 3
    CHECK(host_acquire_sync(g_win, s) == 1);
    host_present(g_win, 1);                        // due 7 (the gate tick of its acquire), texture 1 fenced at 3
    CHECK(host_acquire_sync(g_win, s) == 0);       // texture 0 again: gate tick 7 + 2 = 9, fence value 3
    guard_end();
    g_done_fence = 3;
    CHECK(host_wait(s, 0) == TIMEOUT_EXPIRED);     // the fence is done, the tick (5) is not
    g_done_fence = 0;
    set_now(ref_due_ns(9));
    CHECK(host_wait(s, 0) == TIMEOUT_EXPIRED);     // the tick is reached, the fence is not
    CHECK(host_wait(s, 3000000ull) == TIMEOUT_EXPIRED);   // a wait that has to block: the tick is there at once, the fence not
    g_done_fence = 3;
    CHECK(host_wait(s, 0) == ALREADY_SIGNALED);    // both
    printf("fence_and_tick: the sync was signalled only with the tick reached and the fence value done\n");
    printf("ok fence_and_tick\n");
}

// The setting, flipped between two acquires: a sync an earlier acquire returned keeps its tick, and the record of each
// present says which placement was in effect.
static void case_flip(void) {
    objects(NULL, 3, 2, 5, 0, NVN_ACQUIRE_WAIT_SYNC);
    uint64_t s = host_sync();
    guard_begin("an acquire waited for a tick with the clock stopped");
    CHECK(host_acquire_sync(g_win, s) == 0);
    host_present(g_win, 0);                        // sync placement
    CHECK(host_acquire_sync(g_win, s) == 1);       // gate tick 7
    nvn_acquire_wait(NVN_ACQUIRE_WAIT_ACQUIRE);
    CHECK(nvn_acquire_wait_mode() == NVN_ACQUIRE_WAIT_ACQUIRE);
    CHECK(host_wait(s, 0) == TIMEOUT_EXPIRED);     // the acquire's sync keeps its tick
    host_present(g_win, 1);                        // old placement now: due at the tick in progress (5), not the gate tick 7
    guard_end();
    set_now(ref_due_ns(7));
    CHECK(host_wait(s, 0) == ALREADY_SIGNALED);
    NvnPresentRecord r;
    CHECK(nvn_present_record(1, &r) && r.placement == NVN_ACQUIRE_WAIT_SYNC && r.due == 5);
    CHECK(nvn_present_record(2, &r) && r.placement == NVN_ACQUIRE_WAIT_ACQUIRE && r.due == 5);
    printf("flip: the placement flipped between two acquires; the sync of the first kept its tick; each record names its "
           "placement; a present of the old placement takes the tick in progress, not a gate tick left by the sync\n");
    printf("ok flip\n");
}

// The timeouts, on the driven clock: a wait with timeout 0 never waits; 5 ms of the clock's own time ends a wait for a tick
// 4 ticks off; 100 ms of it carries the wait to the tick, which the driver moves the clock to.
static void driver_timeout(void *arg) {
    (void)arg;
    uint64_t ready = 4;   // the first present is due at tick 0 and the interval is 4
    await(&A_SYNC, 1, 0, "the timeout 0 wait did not come back");
    plat_sleep_ns(30000000ull);   // the 5 ms wait is under way
    uint64_t guard = ref_due_ns(ready) - 2000000ull;   // the first wait's clock never reaches the tick
    uint64_t start = plat_now_ns();
    while (ld_acq(&A_R2) < 1) {
        if (plat_now_ns() - start > 10000000000ull) fail("the 5 ms wait did not end (its timeout is not on the clock's time)", 1);
        uint64_t n = __atomic_load_n(&g_now, __ATOMIC_RELAXED) + 1000000ull;
        if (n < guard) set_now(n);
        plat_sleep_ns(1000000ull);
    }
    while (ld_acq(&A_SYNC) < 2) {   // the 100 ms wait: on to the tick
        uint64_t n = __atomic_load_n(&g_now, __ATOMIC_RELAXED) + 1000000ull;
        set_now(n < ref_due_ns(ready) + 1000000ull ? n : ref_due_ns(ready) + 1000000ull);
        plat_sleep_ns(1000000ull);
    }
}
static void case_timeout(void) {
    objects(NULL, 3, 4, 0, 0, PLACEMENT_ENV);
    CHECK(plat_thread_start(driver_timeout, NULL, 1ull << 20));
    rt_kernel_run(G(3), 0);
    CHECK(g_r[0] == TIMEOUT_EXPIRED && g_r_tick[0] == 0);
    CHECK(g_r[1] == TIMEOUT_EXPIRED && g_r_tick[1] < 4);
    CHECK(g_r[2] == CONDITION_SATISFIED && g_r_tick[2] >= 4);
    CHECK(g_r[3] == ALREADY_SIGNALED);
    printf("timeout: timeout 0 said TIMEOUT_EXPIRED at tick %llu; 5 ms ended at tick %llu with TIMEOUT_EXPIRED; 100 ms "
           "carried the wait to tick %llu (CONDITION_SATISFIED), then ALREADY_SIGNALED\n", (ull)g_r_tick[0],
           (ull)g_r_tick[1], (ull)g_r_tick[2]);
    printf("ok timeout\n");
}

// ---- the script cases. Ticks written by hand; frame k's gate is the tick its sync's wait must end at.
// late: interval 1. The game's work after the acquire takes 0.3 tick, 0, 1.5 ticks, exactly 1 tick, 0.2 tick, 10.2 ticks, 0,
// 0. A late present takes the tick it arrives in; the next sync is signalled at that tick plus 1: no catch-up, no burst, and
// the clock's own tick numbers throughout.
static void case_late(void) {
    static const Exp exp[] = {
        {0, 0, 300, 0, 1, 0},    // first acquire: nothing to wait for; present inside tick 0
        {1, 0, 0, 1, 1, 0},      // gate 1
        {2, 1, 500, 3, 1, 0},    // work 1.5 ticks: arrives in tick 3; due 3
        {4, 1, 0, 5, 1, 0},      // gate 3 + 1 = 4; work exactly 1 tick: arrives at the boundary of tick 5; due 5
        {6, 0, 200, 6, 1, 0},    // gate 6
        {7, 10, 200, 17, 1, 0},  // gate 7; work 10.2 ticks: arrives in tick 17
        {18, 0, 0, 18, 1, 0},    // gate 18: not 8, 9, ... (no catch-up burst)
        {19, 0, 0, 19, 1, 0},
    };
    run_script(exp, NULL, 8, 1, 0, PLACEMENT_ENV, NVN_ACQUIRE_WAIT_SYNC);
    printf("ok late\n");
}

// change: the interval changes 1 -> 2 -> 1 -> 2 -> 1; each change applies at the next acquire, and the record keeps the
// interval at the present. Frame 4 sets 3 between its wait and its present: its record says 3, and the acquire after it
// (which sets 2 first) is signalled 2 ticks after, not 3.
static void case_change(void) {
    static const Step steps[] = {{0, 0, 0}, {2, 0, 0}, {0, 0, 0}, {1, 0, 0}, {0, 0, 3}, {2, 0, 0}, {1, 0, 0}};
    static const Exp exp[] = {
        {0, 0, 0, 0, 1, 0}, {2, 0, 0, 2, 2, 0}, {4, 0, 0, 4, 2, 0}, {5, 0, 0, 5, 1, 0}, {6, 0, 0, 6, 3, 0},
        {8, 0, 0, 8, 2, 0}, {9, 0, 0, 9, 1, 0},
    };
    run_script(exp, steps, 7, 1, 0, PLACEMENT_ENV, NVN_ACQUIRE_WAIT_SYNC);
    printf("ok change\n");
}

// change_mid: the interval changes between the acquire and the wait: the sync keeps the tick the acquire gave it (read at
// the acquire, as ticket 88 reads it), and the change applies at the next acquire.
static void case_change_mid(void) {
    static const Step steps[] = {{0, 0, 0}, {0, 3, 0}, {0, 0, 0}, {0, 1, 0}, {0, 0, 0}};
    static const Exp exp[] = {
        {0, 0, 0, 0, 1, 0},   // no previous present
        {1, 0, 0, 1, 3, 0},   // interval 1 at the acquire: 0 + 1; the game sets 3 before its wait: the record says 3
        {4, 0, 0, 4, 3, 0},   // 3 at the acquire: 1 + 3
        {7, 0, 0, 7, 1, 0},   // 3 at the acquire: 4 + 3; the game sets 1 before its wait
        {8, 0, 0, 8, 1, 0},   // 1 at the acquire: 7 + 1
    };
    run_script(exp, steps, 5, 1, 0, PLACEMENT_ENV, NVN_ACQUIRE_WAIT_SYNC);
    printf("ok change_mid\n");
}

// hold: the host holds every present for 2 ticks while the game asks for 1: syncs signalled 2 ticks apart, the records say 1.
static void case_hold(int placement) {
    static const Exp exp[] = {
        {0, 0, 0, 0, 1, 2}, {2, 0, 0, 2, 1, 2}, {4, 0, 0, 4, 1, 2}, {6, 0, 0, 6, 1, 2}, {8, 0, 0, 8, 1, 2},
        {10, 0, 0, 10, 1, 2},
    };
    run_script(exp, NULL, 6, 1, 2, placement, placement);
    printf("hold: syncs signalled 2 ticks apart under placement %d; the records say interval 1 and hold 2\n", placement);
    printf("ok %s\n", placement == NVN_ACQUIRE_WAIT_SYNC ? "hold_sync" : "hold_call");
}

// hold_max: the hold raises the interval and never lowers it: hold 2, the game asks for 3 from frame 1 on.
static void case_hold_max(int placement) {
    static const Step steps[] = {{0, 0, 0}, {3, 0, 0}, {0, 0, 0}, {0, 0, 0}};
    static const Exp exp[] = {{0, 0, 0, 0, 1, 2}, {3, 0, 0, 3, 3, 2}, {6, 0, 0, 6, 3, 2}, {9, 0, 0, 9, 3, 2}};
    run_script(exp, steps, 4, 1, 2, placement, placement);
    printf("ok %s\n", placement == NVN_ACQUIRE_WAIT_SYNC ? "hold_max_sync" : "hold_max_call");
}

// hold_zero: with the hold on, a game that asks for interval 0 is held too, in both placements.
static void case_hold_zero(int placement) {
    static const Exp exp[] = {{0, 0, 0, 0, 0, 2}, {2, 0, 0, 2, 0, 2}, {4, 0, 0, 4, 0, 2}};
    run_script(exp, NULL, 3, 0, 2, placement, placement);
    printf("ok %s\n", placement == NVN_ACQUIRE_WAIT_SYNC ? "hold_zero_sync" : "hold_zero_call");
}

// setting: nothing sets the placement but the environment (OOT2_HOLD_PLACEMENT, set by the harness) and the call. The records
// of both placements name theirs, and the ticks tell them apart: the acquire of the old placement is the held call.
static void case_setting(int call, int expect, const char *name) {
    static const Exp exp[] = {{0, 0, 0, 0, 2, 0}, {2, 0, 0, 2, 2, 0}, {4, 0, 0, 4, 2, 0}, {6, 0, 0, 6, 2, 0}};
    run_script(exp, NULL, 4, 2, 0, call, expect);
    printf("ok %s\n", name);
}

// setting_bad: a value that is neither sync nor acquire is refused by name, with exit code 2, by the store's first read
// (research 103's rule); the harness checks the exit code and the line.
static void case_setting_bad(void) {
    objects(NULL, 3, 1, 0, 0, PLACEMENT_ENV);
    guard_begin("an unknown OOT2_HOLD_PLACEMENT was taken for a placement");
    (void)host_acquire_sync(g_win, 0);
    fprintf(stderr, "FAIL: the unknown value was accepted\n");
    exit(1);
}

// pace_seam: a present before the wait on its sync tells the back end the due tick it took, which is a tick still to come
// (present_pace's argument, never the tick in progress), and it is the tick the record keeps. The seam only notes what it
// was told.
static uint64_t g_pace_seq[8], g_pace_due[8], g_pace_tick[8];
static int g_npace;
static void seam_pace(uint64_t seq, uint64_t due) {
    CHECK(g_npace < 8);
    g_pace_seq[g_npace] = seq, g_pace_due[g_npace] = due, g_pace_tick[g_npace++] = rt_vsync_tick();
}
static GfxBackend g_seam;
static void case_pace_seam(void) {
    g_seam = gfx_null_backend;
    g_seam.name = "pace-seam";
    g_seam.present_pace = seam_pace;
    objects(&g_seam, 3, 2, 5, 0, PLACEMENT_ENV);
    uint64_t s = host_sync(), s2 = host_sync();
    guard_begin("AcquireTexture, or a present, waited for a tick with the clock stopped (the default placement)");
    CHECK(host_acquire_sync(g_win, s) == 0);
    host_present(g_win, 0);                    // the first frame: due 5, the tick in progress
    CHECK(host_acquire_sync(g_win, s) == 1);   // gate tick 7, the clock at 5
    host_present(g_win, 1);                    // before its wait: due 7
    CHECK(host_acquire_sync(g_win, s2) == 2);  // gate tick 9
    guard_end();
    set_now(ref_due_ns(7));
    CHECK(host_wait(s, 0) == ALREADY_SIGNALED);
    set_now(ref_due_ns(8));
    host_present(g_win, 2);                    // the clock at 8, the gate tick 9: due 9
    CHECK(g_npace == 3);
    NvnPresentRecord r;
    static const uint64_t due[3] = {5, 7, 9}, at[3] = {5, 5, 8};
    for (int k = 0; k < 3; k++) {
        CHECK(g_pace_seq[k] == (uint64_t)k + 1 && g_pace_due[k] == due[k] && g_pace_tick[k] == at[k]);
        CHECK(nvn_present_record((uint64_t)k + 1, &r) && r.due == g_pace_due[k]);
    }
    printf("pace_seam: presents 2 and 3 were made at ticks 5 and 8 and told the back end the due ticks 7 and 9, the tick "
           "of the acquire's sync; each is the due tick of its record\n");
    printf("ok pace_seam\n");
}

// wake: a SyncWait parked on a sync no acquire has fenced yet is woken by the acquire that fences it. No timeout: a
// missing wake-up leaves the game parked, and the guard ends the case after 5 s.
static void case_wake(void) {
    objects(NULL, 3, 1, 0, 0, PLACEMENT_ENV);
    g_wake_sync = host_sync();
    guard_begin("the SyncWait on a sync that no acquire had fenced was not woken by the acquire that fenced it");
    rt_kernel_run(G(5), 0);
    guard_end();
    CHECK(g_wake_res == CONDITION_SATISFIED && g_wake_tick == 0);
    printf("wake: a game thread parked on a sync not yet fenced was woken by the acquire that fenced it, at tick %llu\n",
           (ull)g_wake_tick);
    printf("ok wake\n");
}

// setting_store: the placement from the settings store (the command line layer, as oot2 --set hold_placement=acquire gives
// it), with the NVN2 front end's hook bound, then changed while running through the file layer's call: a live key, applied
// from the next acquire. The command line holds the key, so the change by rt_settings_set is not effective; a second
// store case without it is setting_live.
static void store_load(const char *arg) {
    char why[400];
    RtSettingArg a = {arg, "--set"};
    CHECK(rt_settings_load(&a, arg ? 1 : 0, NULL, why, sizeof why));
    nvn_settings_bind();
}
static void case_setting_store(void) {
    store_load("hold_placement=acquire");
    CHECK(rt_setting(RT_SET_hold_placement) == 1 && nvn_acquire_wait_mode() == NVN_ACQUIRE_WAIT_ACQUIRE);
    static const Exp exp[] = {{0, 0, 0, 0, 2, 0}, {2, 0, 0, 2, 2, 0}, {4, 0, 0, 4, 2, 0}};
    g_exp = exp, g_steps = NULL, g_nframes = 3, g_handshake = 1, g_order = ORDER_WAIT_PRESENT;
    g_script_placement = NVN_ACQUIRE_WAIT_ACQUIRE;
    objects(NULL, 3, 2, 0, 0, PLACEMENT_ENV);
    CHECK(plat_thread_start(driver_script, NULL, 1ull << 20));
    rt_kernel_run(G(0), 0);
    check_script(exp, 3);
    printf("ok setting_store\n");
}

// setting_live: no layer holds the key, so the file layer's value is effective and a change applies at the next acquire,
// in both directions, with the records naming each present's placement and the hook saying so in the log.
static void case_setting_live(void) {
    store_load(NULL);
    objects(NULL, 3, 2, 5, 0, PLACEMENT_ENV);
    char why[400];
    CHECK(nvn_acquire_wait_mode() == NVN_ACQUIRE_WAIT_SYNC);
    uint64_t s = host_sync();
    guard_begin("an acquire waited for a tick with the clock stopped");
    CHECK(host_acquire_sync(g_win, s) == 0);
    host_present(g_win, 0);                          // sync placement
    CHECK(rt_settings_set(RT_SET_hold_placement, "acquire", why, sizeof why));
    CHECK(nvn_acquire_wait_mode() == NVN_ACQUIRE_WAIT_ACQUIRE);
    guard_end();
    set_now(ref_due_ns(7));
    CHECK(host_acquire_sync(g_win, s) == 1);         // the old placement: waits for tick 7, which the clock is at
    host_present(g_win, 1);
    CHECK(rt_settings_set(RT_SET_hold_placement, "sync", why, sizeof why));
    CHECK(nvn_acquire_wait_mode() == NVN_ACQUIRE_WAIT_SYNC);
    host_present(g_win, 2);
    NvnPresentRecord r;
    CHECK(nvn_present_record(1, &r) && r.placement == NVN_ACQUIRE_WAIT_SYNC);
    CHECK(nvn_present_record(2, &r) && r.placement == NVN_ACQUIRE_WAIT_ACQUIRE);
    CHECK(nvn_present_record(3, &r) && r.placement == NVN_ACQUIRE_WAIT_SYNC);
    // a call to nvn_acquire_wait overrides the store from then on
    nvn_acquire_wait(NVN_ACQUIRE_WAIT_ACQUIRE);
    CHECK(rt_settings_set(RT_SET_hold_placement, "sync", why, sizeof why));
    CHECK(nvn_acquire_wait_mode() == NVN_ACQUIRE_WAIT_ACQUIRE);
    printf("setting_live: the key hold_placement changed the placement at the next acquire, both ways; nvn_acquire_wait "
           "overrides it\n");
    printf("ok setting_live\n");
}

// realtime: the host's clock, interval 2, 12 frames, the game waiting on the sync with no timeout: no wait ends before its
// gate tick (the previous present's due tick plus 2), and the frames take about 22 ticks.
static void case_realtime(void) {
    g_nframes = 12, g_order = ORDER_WAIT_PRESENT;
    objects(NULL, 3, 2, UNDRIVEN, 0, PLACEMENT_ENV);
    uint64_t t0 = plat_now_ns();
    rt_kernel_run(G(0), 0);
    uint64_t span = plat_now_ns() - t0;
    for (uint64_t k = 1; k < g_nframes; k++) {
        CHECK(g_sync_tick[k] >= g_rec[k - 1].due + 2);   // never early
        CHECK(g_rec[k].due >= g_rec[k - 1].due + 2);
        CHECK(g_sync_res[k] == CONDITION_SATISFIED || g_sync_res[k] == ALREADY_SIGNALED);
    }
    CHECK(span >= 21 * 16666666ull);   // 11 gaps of 2 ticks, the first starting inside a tick
    printf("realtime: 12 presents at interval 2 took %.0f ms of the host's clock (22 ticks are 367 ms); no wait on the "
           "acquire sync ended early\n", (double)span / 1e6);
    printf("ok realtime\n");
}

// realtime_timeout: on the host's clock a finite timeout ends a wait for a tick that is still far off (interval 4: 4 ticks
// from the present's due tick, at least 3 whole ticks from the call), and an unlimited wait then reaches the tick.
static void case_realtime_timeout(void) {
    objects(NULL, 3, 4, UNDRIVEN, 0, PLACEMENT_ENV);
    rt_kernel_run(G(4), 0);
    CHECK(g_rt[0] == TIMEOUT_EXPIRED && g_rt[1] == CONDITION_SATISFIED && g_rt[2] == ALREADY_SIGNALED);
    CHECK(g_rt_ms[0] >= 5);   // never early
    // A long timeout does not hold the wait past its tick (about 70 ms at most away). The bound is half the timeout, so a
    // loaded machine does not turn it red, and a wait that sat out its whole timeout does.
    CHECK(g_rt_ms[1] < 5000);
    printf("realtime: a 5 ms timeout ended a wait for a tick 4 ticks away after %llu ms of the host's clock, then a 10 s "
           "wait reached the tick after %llu ms\n", (ull)g_rt_ms[0], (ull)g_rt_ms[1]);
    printf("ok realtime_timeout\n");
}

// trace: the pacing trace (OOT2_PACE_TRACE, set by the harness) with two probes written to OOT2_PROBE_FILE by this case: a
// 64-bit counter the game bumps once a frame, found through a pointer chain of two words, and an absolute double. Three
// frames at interval 2 in the default placement. The probe file is written before the first acquire (the trace reads it
// when it starts), with the guest addresses of the case's own objects: no game, no module offsets.
static void case_trace(int with_trace) {
    const char *pf = getenv("OOT2_PROBE_FILE");
    CHECK(pf && *pf);
    objects(NULL, 3, 2, 0, 0, PLACEMENT_ENV);
    uint64_t chain = W(0x300), obj = W(0x340), dbl = W(0x380);
    *(uint64_t *)(uintptr_t)chain = obj - 0x10;          // *+0x10 of the chain's word is the object's counter
    *(uint64_t *)(uintptr_t)(obj + 0x8) = 4242;          // =+0x8 of the object
    *(double *)(uintptr_t)dbl = 2.5;
    FILE *f = fopen(pf, "w");
    CHECK(f != NULL);
    fprintf(f, "# a probe file of the case\n");
    fprintf(f, "frames @0x%llx *+0x0 =+0x18 u64\n", (ull)chain);   // chain -> obj - 0x10, +0x18 = obj + 8
    fprintf(f, "ratio @0x%llx =+0x0 f64\n", (ull)dbl);
    fprintf(f, "nowhere @0x10 =+0x0 u32\n");                     // unmapped: prints ?
    if (!with_trace) fprintf(f, "this line does not parse\n");     // never read in a run without the trace
    fclose(f);
    static const Exp exp[] = {{0, 0, 0, 0, 2, 0}, {2, 0, 0, 2, 2, 0}, {4, 0, 0, 4, 2, 0}};
    g_script_placement = NVN_ACQUIRE_WAIT_SYNC;
    g_exp = exp, g_steps = NULL, g_nframes = 3, g_handshake = 1, g_order = ORDER_WAIT_PRESENT;
    CHECK(plat_thread_start(driver_script, NULL, 1ull << 20));
    rt_kernel_run(G(0), 0);
    check_script(exp, 3);
    printf("ok %s\n", with_trace ? "trace" : "trace_off");
}

// A case that has not ended after HANG_NS exits 3 and dumps the kernel.
#define HANG_NS 90000000000ull
static void watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(HANG_NS);
    rt_kernel_dump();
    _Exit(3);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: nvn_sync_test <case>\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    const char *cs = argv[1];
    if (!strcmp(cs, "hour1")) case_hour(1, ORDER_WAIT_PRESENT);
    else if (!strcmp(cs, "hour2")) case_hour(2, ORDER_WAIT_PRESENT);
    else if (!strcmp(cs, "hour1_pwait")) case_hour(1, ORDER_PRESENT_WAIT);
    else if (!strcmp(cs, "hour2_pwait")) case_hour(2, ORDER_PRESENT_WAIT);
    else if (!strcmp(cs, "block")) case_block();
    else if (!strcmp(cs, "acquire_nowait")) case_acquire_nowait();
    else if (!strcmp(cs, "first")) case_first();
    else if (!strcmp(cs, "null_sync")) case_null_sync();
    else if (!strcmp(cs, "fence_clears_tick")) case_fence_clears_tick();
    else if (!strcmp(cs, "fence_and_tick")) case_fence_and_tick();
    else if (!strcmp(cs, "flip")) case_flip();
    else if (!strcmp(cs, "timeout")) case_timeout();
    else if (!strcmp(cs, "late")) case_late();
    else if (!strcmp(cs, "change")) case_change();
    else if (!strcmp(cs, "change_mid")) case_change_mid();
    else if (!strcmp(cs, "hold_sync")) case_hold(NVN_ACQUIRE_WAIT_SYNC);
    else if (!strcmp(cs, "hold_call")) case_hold(NVN_ACQUIRE_WAIT_ACQUIRE);
    else if (!strcmp(cs, "hold_max_sync")) case_hold_max(NVN_ACQUIRE_WAIT_SYNC);
    else if (!strcmp(cs, "hold_max_call")) case_hold_max(NVN_ACQUIRE_WAIT_ACQUIRE);
    else if (!strcmp(cs, "hold_zero_sync")) case_hold_zero(NVN_ACQUIRE_WAIT_SYNC);
    else if (!strcmp(cs, "hold_zero_call")) case_hold_zero(NVN_ACQUIRE_WAIT_ACQUIRE);
    else if (!strcmp(cs, "setting_default")) case_setting(PLACEMENT_ENV, NVN_ACQUIRE_WAIT_SYNC, "setting_default");
    else if (!strcmp(cs, "setting_env")) case_setting(PLACEMENT_ENV, NVN_ACQUIRE_WAIT_ACQUIRE, "setting_env");
    else if (!strcmp(cs, "setting_call_sync")) case_setting(NVN_ACQUIRE_WAIT_SYNC, NVN_ACQUIRE_WAIT_SYNC, "setting_call_sync");
    else if (!strcmp(cs, "setting_call_call")) case_setting(NVN_ACQUIRE_WAIT_ACQUIRE, NVN_ACQUIRE_WAIT_ACQUIRE, "setting_call_call");
    else if (!strcmp(cs, "setting_bad")) case_setting_bad();
    else if (!strcmp(cs, "pace_seam")) case_pace_seam();
    else if (!strcmp(cs, "wake")) case_wake();
    else if (!strcmp(cs, "setting_store")) case_setting_store();
    else if (!strcmp(cs, "setting_live")) case_setting_live();
    else if (!strcmp(cs, "trace")) case_trace(1);
    else if (!strcmp(cs, "trace_off")) case_trace(0);
    else if (!strcmp(cs, "realtime")) case_realtime();
    else if (!strcmp(cs, "realtime_timeout")) case_realtime_timeout();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
