// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the acquire gate (architecture D22; runtime/nvn/nvn_window.c, oot2/nvn_pace.h), driven by
// tests/test_runtime_nvn_pace.py: `nvn_pace_test <case>`. Each case runs in its own process and prints "ok <case>".
//
// The "game" is a guest thread written by hand as a host function in the lifter's shape: it calls the NVN entry points
// through their synthetic addresses with the guest's arguments (acquire, present, SetPresentInterval), as generated code
// does, and records the tick each acquire returned at and the record each present left. No game code, no dump.
//
// The clock is driven (rt_vsync_drive, the vsync clock) and nothing binds a vsync event, so the clock thread never starts: the
// gate reads rt_vsync_tick() and waits through rt_vsync_wait_tick, which follows the driver's time. A driver thread (a
// plain host thread) moves the time. The hour cases move it tick by tick, each step only once the game has done what
// the tick allows, so an hour runs in seconds with no wall-clock wait. The script cases check the model of oot2/nvn_pace.h
// against ticks written out in this file: a tick before the gate's, the acquire must still be held; at it, released.
// The expected ticks are written here by hand, never computed with the front end's rule.
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

// The schedule, written here without vsync.c: 31,250,000 counts a second of 32 ns, tick n due at ceil(n * 31.25e6 / 60).
#define ORIGIN 123456789012ull
#define HOUR_TICKS 216000ull
static uint64_t ref_due_ns(uint64_t n) { return ORIGIN + ((n * 31250000ull + 59) / 60) * 32ull; }

static Cpu c;   // the host thread's Cpu: entry points called before the run
static uint64_t heap, heap_used, loader, g_area;
static uint64_t g_dev, g_queue, g_win;
static uint64_t f_acquire, f_present, f_setint;

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
enum { IDX = 0x0, WORK = 0x40, DONE = 0x48, HANDLES = 0x100, STACKS = 0x100000, AREA = 0x120000 };
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

#define MAXF 216002
typedef struct { int set_acq, set_pres; } Step;   // SetPresentInterval before the acquire / before the present; 0: none
static const Step *g_steps;
static uint64_t g_nframes;
static int g_handshake;                    // the game waits for GO after each acquire (the driver's "work" time)
static uint64_t A_ACQ, A_GO, A_PRES;       // acquires returned, presents allowed, presents done (atomic)
static uint64_t g_acq_tick[MAXF];          // the tick each acquire returned at, as the game read it
static NvnPresentRecord g_rec[MAXF];       // the record each present left, read right after it

static void game_body(Cpu *t) {
    for (uint64_t k = 0; k < g_nframes; k++) {
        if (g_steps && g_steps[k].set_acq) gcall(t, f_setint, g_win, (uint64_t)g_steps[k].set_acq, 0);
        CHECK(LOW32(gcall(t, f_acquire, g_win, 0, W(IDX))) == 0);
        g_acq_tick[k] = rt_vsync_tick();
        st_rel(&A_ACQ, k + 1);
        if (g_handshake)
            while (ld_acq(&A_GO) < k + 1) sv(t, 0x1E, 0, 0, 0, 0);   // GetSystemTick: the exit is a safe point
        if (g_steps && g_steps[k].set_pres) gcall(t, f_setint, g_win, (uint64_t)g_steps[k].set_pres, 0);
        uint32_t idx = *(volatile uint32_t *)(uintptr_t)W(IDX);
        t->x[0] = g_queue, t->x[1] = g_win, t->x[2] = idx, t->x[3] = 0, t->x[30] = PC + 4, t->pc = f_present;
        rt_dispatch(t);
        CHECK(nvn_present_record(nvn_present_count(), &g_rec[k]));
        st_rel(&A_PRES, k + 1);
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

static void guest_end(Cpu *t) { (void)t; }
static const GuestFn g_fns[] = {g_main_game, g_worker, g_main_block, guest_end};
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
// Memory, kernel, entry points, a device, a queue and a window of n textures at the given interval; the clock is driven
// and stands at first_tick (UNDRIVEN: the host's own clock). hold: the experiment hold, set explicitly so the environment
// cannot change a case; -1 leaves it as the process starts (off, or what OOT2_HOLD_TICKS names).
static void objects(const GfxBackend *b, int n, int interval, uint64_t first_tick, int hold) {
    setup(b);
    if (first_tick != UNDRIVEN) drive_clock(first_tick);
    if (hold >= 0) nvn_present_hold(hold);
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
// The hour: tick t+1 is reached only once the game has presented every frame whose acquire tick is at or before t, and
// before that the game must not have presented more (an acquire that returned early would show as a frame ahead of its
// tick). Each tick is entered 0 to 15 ms into it.
static void driver_hour(void *arg) {
    (void)arg;
    uint64_t I = g_hour_interval;
    for (uint64_t t = 0;; t++) {
        uint64_t want = t / I + 1;
        await(&A_PRES, want, t, "the game has not presented the frames this tick allows (an acquire is held too long)");
        if (ld_acq(&A_PRES) > want) fail("the game presented a frame ahead of its tick (an acquire returned early)", t);
        if (t == g_total_ticks) break;
        set_now(ref_due_ns(t + 1) + (t * 7919 % 16) * 1000000ull);
    }
}

typedef struct { uint64_t gate, work_ticks; int work_milli, due, requested, held; } Exp;   // the model's answer, by hand
static const Exp *g_exp;
// The script: for each frame, the acquire must stay held until the clock is at its gate tick, and come back at it; then the
// game "works" for work_ticks and work_milli thousandths of a tick past the gate tick before it presents.
static void driver_script(void *arg) {
    (void)arg;
    for (uint64_t k = 0; k < g_nframes; k++) {
        const Exp *e = &g_exp[k];
        if (e->gate > rt_vsync_tick()) {
            set_now(ref_due_ns(e->gate) - 1);   // a nanosecond before the gate tick: still held
            plat_sleep_ns(5000000ull);   // only a probe: the recorded acquire tick is the authoritative check
            if (ld_acq(&A_ACQ) != k) fail("the acquire came back before its gate tick", k);
            set_now(ref_due_ns(e->gate));
        }
        await(&A_ACQ, k + 1, k, "the acquire did not come back at its gate tick");
        uint64_t w = e->gate + e->work_ticks;
        uint64_t arrive = ref_due_ns(w) + (ref_due_ns(w + 1) - ref_due_ns(w)) * (uint64_t)e->work_milli / 1000;
        if (arrive > __atomic_load_n(&g_now, __ATOMIC_RELAXED)) set_now(arrive);
        st_rel(&A_GO, k + 1);
        await(&A_PRES, k + 1, k, "the present did not come back");
    }
}

static void check_script(const Exp *exp, uint64_t n) {
    for (uint64_t k = 0; k < n; k++) {
        printf("frame %llu acquired at tick %llu due %llu requested %d held %d\n", (ull)k, (ull)g_acq_tick[k],
               (ull)g_rec[k].due, g_rec[k].requested, g_rec[k].held);
        CHECK(g_acq_tick[k] == exp[k].gate);
        CHECK(g_rec[k].due == (uint64_t)exp[k].due);
        CHECK(g_rec[k].requested == exp[k].requested);
        CHECK(g_rec[k].held == exp[k].held);
        CHECK(g_rec[k].seq == k + 1 && g_rec[k].shown == 0);
    }
}

static void run_script(const Exp *exp, const Step *steps, uint64_t n, int window_interval, int hold) {
    g_exp = exp, g_steps = steps, g_nframes = n, g_handshake = 1;
    objects(NULL, 3, window_interval, 0, hold);
    CHECK(plat_thread_start(driver_script, NULL, 1ull << 20));
    rt_kernel_run(G(0), 0);
    check_script(exp, n);
}

// ---------- cases
static void case_hour(uint64_t interval) {
    g_hour_interval = interval;
    g_nframes = g_total_ticks / interval + 1;
    objects(NULL, 3, (int)interval, 0, -1);
    CHECK(nvn_present_hold_ticks() == 0);   // the default: no hold
    CHECK(plat_thread_start(driver_hour, NULL, 1ull << 20));
    uint64_t t0 = plat_now_ns();
    rt_kernel_run(G(0), 0);
    uint64_t n = g_nframes;
    for (uint64_t k = 0; k < n; k++) {
        if (g_acq_tick[k] != k * interval || g_rec[k].due != k * interval || g_rec[k].requested != (int)interval ||
            g_rec[k].held != 0 || g_rec[k].shown != 0 || g_rec[k].seq != k + 1 || g_rec[k].texture != (int)(k % 3)) {
            fprintf(stderr, "FAIL frame %llu: acquired at tick %llu (want %llu), due %llu, requested %d, held %d, "
                            "shown %llu\n", (ull)k, (ull)g_acq_tick[k], (ull)(k * interval), (ull)g_rec[k].due,
                    g_rec[k].requested, g_rec[k].held, (ull)g_rec[k].shown);
            exit(1);
        }
    }
    CHECK(nvn_present_count() == n && gfx_null_presents() == n);
    printf("hour: %llu presents at interval %llu, every %llu ticks from tick 0 to tick %llu, in %.1f s of wall time\n",
           (ull)n, (ull)interval, (ull)interval, (ull)((n - 1) * interval), (double)(plat_now_ns() - t0) / 1e9);
    printf("ok hour%llu\n", (ull)interval);
}

// The game thread gives its core up while it waits in AcquireTexture: a worker of a worse priority on the same core
// counts meanwhile, and the game takes the core back when the tick comes.
static void driver_block(void *arg) {
    (void)arg;
    await(&A_PRES, 1, 0, "the first present did not come back");
    await((const uint64_t *)(uintptr_t)W(WORK), 1000, 0,
          "the worker never ran while the game thread was held in AcquireTexture: it kept its virtual core");
    if (ld_acq(&A_ACQ) != 1) fail("the second acquire came back with the clock at tick 0", 1);
    printf("block: the worker counted %llu while the game thread waited in AcquireTexture\n",
           (ull)ld_acq((const uint64_t *)(uintptr_t)W(WORK)));
    set_now(ref_due_ns(1));
    await(&A_ACQ, 2, 1, "the game thread did not take its core back at the tick");
    await(&A_PRES, 2, 1, "the second present did not come back");
}
static void case_block(void) {
    g_nframes = 2;
    objects(NULL, 2, 1, 0, 0);
    CHECK(plat_thread_start(driver_block, NULL, 1ull << 20));
    rt_kernel_run(G(2), 0);
    CHECK(g_acq_tick[0] == 0 && g_acq_tick[1] == 1 && g_rec[1].due == 1);
    printf("block: acquire 2 came back at tick %llu\n", (ull)g_acq_tick[1]);
    printf("ok block\n");
}

// The calls below run on the host thread with the clock stopped. A call that waits for a tick never comes back; the
// guard ends the case after 5 s.
static uint64_t host_acquire(uint64_t w) {
    uint64_t idx = galloc(4, 4), a[3] = {w, 0, idx};
    CHECK(LOW32(vcall(f_acquire, a, 3)) == 0);
    return *(uint32_t *)(uintptr_t)idx;
}
static void host_present(uint64_t w, uint64_t i) {
    uint64_t a[3] = {g_queue, w, i};
    vcall(f_present, a, 3);
}

static uint64_t g_pace_seq[1024], g_pace_due[1024];
static int g_npace;
static void seam_pace(uint64_t seq, uint64_t due, uint64_t count) {
    CHECK(g_npace < 1024 && count == ((due * 31250000ull + 59) / 60));   // the due count of that tick, by hand
    g_pace_seq[g_npace] = seq, g_pace_due[g_npace++] = due;
}
static void seam_present(const tex_t *t, const int crop[4]) {
    CHECK(g_npace >= 1 && g_pace_seq[g_npace - 1] == nvn_present_count());   // pace comes first, for this very present
    gfx_null_backend.present(t, crop);
}
static uint64_t seam_shown(uint64_t seq) {   // an even present is shown one tick after its due tick; the rest are unknown
    CHECK(seq >= 1 && seq <= (uint64_t)g_npace);
    return seq % 2 == 0 ? g_pace_due[seq - 1] + 1 : 0;
}
static GfxBackend g_seam;
static void seam_setup(void) {
    g_seam = gfx_null_backend;
    g_seam.name = "pace";
    g_seam.present_pace = seam_pace, g_seam.present = seam_present, g_seam.present_shown = seam_shown;
}

// QueuePresentTexture returns with the clock stopped: the window is at interval 2, and a present never waits for the due
// tick plus the interval that the next acquire does. Three presents in a row, each due at the tick in progress.
static void case_present_nowait(void) {
    seam_setup();
    objects(&g_seam, 3, 2, 5, 0);
    guard_begin("QueuePresentTexture (or the first AcquireTexture) waited for a tick with the clock stopped");
    uint64_t first = host_acquire(g_win);
    for (uint64_t i = 0; i < 3; i++) host_present(g_win, i);
    guard_end();
    CHECK(first == 0 && nvn_present_count() == 3 && g_npace == 3);
    for (uint64_t s = 1; s <= 3; s++) {
        NvnPresentRecord r;
        CHECK(nvn_present_record(s, &r));
        CHECK(r.seq == s && r.due == 5 && r.requested == 2 && r.texture == (int)(s - 1) && r.held == 0);
        CHECK(g_pace_seq[s - 1] == s && g_pace_due[s - 1] == 5);
    }
    printf("present: 3 presents came back with the clock stopped at tick 5, each due at tick 5, requested interval 2\n");
    printf("ok present_nowait\n");
}

// The first acquire of a window waits for nothing, whatever the interval and the hold; so do the acquires before any
// present, and the first of another window. The clock stands at tick 7, stopped; the interval is 3 and the hold 10.
static void case_first(void) {
    objects(NULL, 3, 3, 7, 10);
    uint64_t w2 = window(2, 3);
    guard_begin("an acquire with no previous present waited for a tick with the clock stopped");
    CHECK(host_acquire(g_win) == 0 && host_acquire(g_win) == 1);   // two in a row, no present between them
    host_present(g_win, 0);                                         // this window has presented; the other has not
    CHECK(host_acquire(w2) == 0 && host_acquire(w2) == 1);
    guard_end();
    NvnPresentRecord r;
    CHECK(nvn_present_count() == 1 && nvn_present_record(1, &r) && r.due == 7 && r.requested == 3 && r.held == 10);
    printf("first: acquires with no previous present came back at once at tick 7 (interval 3, hold 10)\n");
    printf("ok first\n");
}

// Interval 0 and below wait for nothing and are stored as written: 600 presents in the ring of 512.
static void case_ring(void) {
    objects(NULL, 3, 0, 3, 0);
    guard_begin("an acquire at interval 0 or below waited for a tick with the clock stopped");
    for (uint64_t k = 0; k < 600; k++) {
        if (k == 300) {
            uint64_t a[2] = {g_win, (uint64_t)(uint32_t)-3};
            vcall(f_setint, a, 2);
        }
        host_present(g_win, host_acquire(g_win));
    }
    guard_end();
    CHECK(nvn_present_count() == 600);
    NvnPresentRecord r;
    CHECK(!nvn_present_record(0, &r) && !nvn_present_record(601, &r));
    CHECK(nvn_present_record(600, &r) && r.seq == 600 && r.requested == -3 && r.due == 3 && r.texture == 599 % 3);
    CHECK(nvn_present_record(89, &r) && r.seq == 89 && r.requested == 0 && r.texture == 88 % 3);   // the oldest kept
    CHECK(!nvn_present_record(88, &r) && !nvn_present_record(1, &r));
    CHECK(nvn_present_record(301, &r) && r.requested == -3 && nvn_present_record(300, &r) && r.requested == 0);
    printf("ring: 600 presents; the newest %d are kept (89 to 600), intervals 0 and -3 as written, no acquire waited\n",
           NVN_PRESENT_RING);
    printf("ok ring\n");
}

// The back end gets the due tick of every present, before the present, and its shown ticks come back in the records.
static void case_seam(void) {
    seam_setup();
    g_hour_interval = 1, g_total_ticks = 40, g_nframes = 41;
    objects(&g_seam, 3, 1, 0, 0);
    CHECK(plat_thread_start(driver_hour, NULL, 1ull << 20));
    rt_kernel_run(G(0), 0);
    CHECK(g_npace == 41 && nvn_present_count() == 41);
    for (uint64_t s = 1; s <= 41; s++) {
        NvnPresentRecord r;
        CHECK(nvn_present_record(s, &r));
        CHECK(g_pace_seq[s - 1] == s && g_pace_due[s - 1] == s - 1 && r.due == s - 1);
        CHECK(r.shown == (s % 2 == 0 ? s : 0));   // due + 1 for an even present; unknown (0) for an odd one
    }
    printf("seam: 41 presents; the back end was told each present's number and due tick before the present; shown ticks "
           "came back for the even ones, 0 for the odd ones\n");
    printf("ok seam\n");
}

// ---- the script cases. Ticks written by hand; frame k's gate is the tick its acquire must come back at.
// late: interval 1. The game's work after the acquire takes 0.3 tick, 0, 1.5 ticks, exactly 1 tick, 0.2 tick, 10.2
// ticks, 0, 0. A late present takes the tick it arrives in; the next acquire waits for that tick plus 1: no catch-up, no
// burst, and the clock's own tick numbers throughout.
static void case_late(void) {
    static const Exp exp[] = {
        {0, 0, 300, 0, 1, 0},    // first acquire: no wait; present inside tick 0
        {1, 0, 0, 1, 1, 0},      // gate 1
        {2, 1, 500, 3, 1, 0},    // work 1.5 ticks: arrives in tick 3; due 3
        {4, 1, 0, 5, 1, 0},      // gate 3 + 1 = 4; work exactly 1 tick: arrives at the boundary of tick 5; due 5
        {6, 0, 200, 6, 1, 0},    // gate 6
        {7, 10, 200, 17, 1, 0},  // gate 7; work 10.2 ticks: arrives in tick 17
        {18, 0, 0, 18, 1, 0},    // gate 18: not 8, 9, ... (no catch-up burst)
        {19, 0, 0, 19, 1, 0},
    };
    run_script(exp, NULL, 8, 1, 0);
    printf("ok late\n");
}

// change: the interval changes 1 -> 2 -> 1 -> 2 -> 1; each change applies at the next acquire, and the record keeps the
// interval at the present. Frame 4 sets 3 between its acquire and its present: its record says 3, and the acquire after it
// (which sets 2 first) waits for 2.
static void case_change(void) {
    static const Step steps[] = {{0, 0}, {2, 0}, {0, 0}, {1, 0}, {0, 3}, {2, 0}, {1, 0}};
    static const Exp exp[] = {
        {0, 0, 0, 0, 1, 0},   // interval 1, no previous present
        {2, 0, 0, 2, 2, 0},   // set 2: waits for 0 + 2
        {4, 0, 0, 4, 2, 0},   // 2 + 2
        {5, 0, 0, 5, 1, 0},   // set 1: 4 + 1
        {6, 0, 0, 6, 3, 0},   // 5 + 1; sets 3 before the present
        {8, 0, 0, 8, 2, 0},   // sets 2 before the acquire: 6 + 2, not 6 + 3
        {9, 0, 0, 9, 1, 0},   // set 1: 8 + 1
    };
    run_script(exp, steps, 7, 1, 0);
    printf("ok change\n");
}

// hold: the host holds every present for 2 ticks while the game asks for 1: presents 2 ticks apart, the records say 1.
static void case_hold(void) {
    static const Exp exp[] = {
        {0, 0, 0, 0, 1, 2}, {2, 0, 0, 2, 1, 2}, {4, 0, 0, 4, 1, 2}, {6, 0, 0, 6, 1, 2}, {8, 0, 0, 8, 1, 2},
        {10, 0, 0, 10, 1, 2},
    };
    run_script(exp, NULL, 6, 1, 2);
    printf("hold: presents 2 ticks apart; the records say interval 1 and hold 2\n");
    printf("ok hold\n");
}

// hold_max: the hold raises the interval and never lowers it: hold 2, the game asks for 3 from frame 1 on.
static void case_hold_max(void) {
    static const Step steps[] = {{0, 0}, {3, 0}, {0, 0}, {0, 0}};
    static const Exp exp[] = {{0, 0, 0, 0, 1, 2}, {3, 0, 0, 3, 3, 2}, {6, 0, 0, 6, 3, 2}, {9, 0, 0, 9, 3, 2}};
    run_script(exp, steps, 4, 1, 2);
    printf("ok hold_max\n");
}

// hold_default / hold_env / hold_call: nothing sets the hold. With the environment clear it is off; with OOT2_HOLD_TICKS=3
// it holds 3; and a call to nvn_present_hold made before the first window is used wins over the environment.
static void case_hold_env(int expect, int call) {
    static const Exp off[] = {{0, 0, 0, 0, 1, 0}, {1, 0, 0, 1, 1, 0}, {2, 0, 0, 2, 1, 0}, {3, 0, 0, 3, 1, 0}};
    static const Exp on[] = {{0, 0, 0, 0, 1, 3}, {3, 0, 0, 3, 1, 3}, {6, 0, 0, 6, 1, 3}, {9, 0, 0, 9, 1, 3}};
    run_script(expect ? on : off, NULL, 4, 1, call ? 0 : -1);
    CHECK(nvn_present_hold_ticks() == expect);
    nvn_present_hold(0);
    CHECK(nvn_present_hold_ticks() == 0);
    printf("ok %s\n", call ? "hold_call" : expect ? "hold_env" : "hold_default");
}

// realtime: the host's clock, interval 2, 12 frames: no acquire comes back before its due tick plus 2, and the frames take
// about 22 ticks. Nothing here bounds how late a frame may be: the machine may be loaded.
static void case_realtime(void) {
    g_nframes = 12;
    objects(NULL, 3, 2, UNDRIVEN, 0);
    uint64_t t0 = plat_now_ns();
    rt_kernel_run(G(0), 0);
    uint64_t span = plat_now_ns() - t0;
    for (uint64_t k = 1; k < g_nframes; k++) {
        CHECK(g_acq_tick[k] >= g_rec[k - 1].due + 2);   // never early
        CHECK(g_rec[k].due >= g_rec[k - 1].due + 2);
    }
    CHECK(span >= 21 * 16666666ull);   // 11 gaps of 2 ticks, the first starting inside a tick
    printf("realtime: 12 presents at interval 2 took %.0f ms of the host's clock (22 ticks are 367 ms); none came back "
           "early\n", (double)span / 1e6);
    printf("ok realtime\n");
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
        fprintf(stderr, "usage: nvn_pace_test <case>\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    const char *cs = argv[1];
    if (!strcmp(cs, "hour1")) case_hour(1);
    else if (!strcmp(cs, "hour2")) case_hour(2);
    else if (!strcmp(cs, "block")) case_block();
    else if (!strcmp(cs, "present_nowait")) case_present_nowait();
    else if (!strcmp(cs, "first")) case_first();
    else if (!strcmp(cs, "ring")) case_ring();
    else if (!strcmp(cs, "seam")) case_seam();
    else if (!strcmp(cs, "late")) case_late();
    else if (!strcmp(cs, "change")) case_change();
    else if (!strcmp(cs, "hold")) case_hold();
    else if (!strcmp(cs, "hold_max")) case_hold_max();
    else if (!strcmp(cs, "hold_default")) case_hold_env(0, 0);
    else if (!strcmp(cs, "hold_env")) case_hold_env(3, 0);
    else if (!strcmp(cs, "hold_call")) case_hold_env(0, 1);
    else if (!strcmp(cs, "realtime")) case_realtime();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
