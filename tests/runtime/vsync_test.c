// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the virtual 60 Hz display clock (runtime/core/vsync.c): its schedule, the events it signals and the
// guest threads that wait on them. Synthetic guest programs written by hand in the lifter's shape run on guest threads
// and virtual cores, as in kernel_test.c; no game code is involved.
//
//   vsync_test CASE    runs one case; prints "ok CASE" and exits 0, or exits 1 with a FAIL line. The case that ends in
//                      a fatal error exits with the runtime's code (tests/test_runtime_vsync.py).
//
// The driven cases replace the clock's time source and wait (rt_vsync_drive). The driver runs on the clock's thread: it
// checks every deadline the clock asks for against the schedule computed here, independently of vsync.c, and moves its
// time to the deadline (plus a lateness the case chooses) only once the guest has found the previous tick's events
// signalled, and to tick 1's only once the case has bound every event, so an hour of ticks runs in lockstep in a few
// seconds. The realtime case runs the host's clock for a few seconds and measures what guest threads waiting on the
// events see.
//
// The guest functions come first and together, then a last one (guest_end) that no thread runs, as in kernel_test.c:
// asynchronous preemption takes the module's range, up to its end marker (endmark.c), as lifted code.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/settings.h"
#include "oot2/vsync.h"
#include "plat.h"

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))   // the guest address of guest function i
#define PC 0x7000000040ull           // the guest pc the test's SVCs report
#define R_INVALID_STATE 0xFA01u
#define EVENTS 4

// The schedule, written here without vsync.c: the system counter counts 31,250,000 per second, 32 ns a count, and
// tick n is due at count ceil(n * 31,250,000 / 60).
#define COUNTER_PER_S 31250000ull
#define NS_PER_COUNT 32ull
#define HOUR_TICKS 216000ull                 // 3600 s at 60 Hz
#define ORIGIN 123456789012ull               // the driven time of tick 0: any value, the schedule counts from it
#define LATE_TICKS 2                         // the late case's one wake-up passes this many ticks...
#define LATE_EXTRA_NS 8000000ull             // ...and comes this far into the next
#define REALTIME_S 5
#define REALTIME_CAP 512                     // wakes recorded per event: 300 are expected in 5 s

static uint64_t ref_due_counter(uint64_t n) { return (n * COUNTER_PER_S + 59) / 60; }   // n below 5.9e11
static uint64_t ref_due_ns(uint64_t n) { return ORIGIN + ref_due_counter(n) * NS_PER_COUNT; }

// Guest heap words, from the heap base the host sets before the run.
static uint64_t HEAP;
enum { ACK = 0x0, RAN = 0x8, DONE = 0x10, STOP = 0x18, BATCHES = 0x20, STRICT = 0x28, RESULTS = 0x100, SEEN = 0x200,
       HANDLES = 0x400, WAKES = 0x1000, STACKS = 0x100000 };
#define W(off) (HEAP + (off))
static uint32_t g_w[EVENTS], g_r[EVENTS];   // made by the host before the run (rt_kernel_event_create)

// ---- helpers the guest functions use: inline, no locks
static inline uint64_t sv(Cpu *c, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3;
    rt_svc(c, imm, PC);
    return c->x[0];
}
static inline uint32_t create(Cpu *c, uint64_t entry, uint64_t arg, int prio, int core) {
    c->x[1] = entry, c->x[2] = arg, c->x[3] = W(STACKS) + 0x10000 * (arg + 1), c->x[4] = (uint32_t)prio,
    c->x[5] = (uint32_t)core;
    rt_svc(c, 0x08, PC);
    CHECK(c->x[0] == 0);
    return (uint32_t)c->x[1];
}
static inline void start(Cpu *c, uint32_t h) { CHECK(sv(c, 0x09, h, 0, 0, 0) == 0); }
static inline void join(Cpu *c, uint32_t h) {
    ST32(W(HANDLES), h);
    CHECK(sv(c, 0x18, 0, W(HANDLES), 1, ~0ull) == 0);
    CHECK(sv(c, 0x16, h, 0, 0, 0) == 0);
}
static inline void exit_thread(Cpu *c) { sv(c, 0x0A, 0, 0, 0, 0); }
static inline uint64_t ld_acq(uint64_t off) { return __atomic_load_n((uint64_t *)(uintptr_t)W(off), __ATOMIC_ACQUIRE); }
static inline void st_rel(uint64_t off, uint64_t v) {
    __atomic_store_n((uint64_t *)(uintptr_t)W(off), v, __ATOMIC_RELEASE);
}

// ---- guest functions, in address order (the module table below lists them)
static void guest_end(Cpu *c);

// 0: the lockstep checker of the driven cases. For each batch of signals the clock sends, every bound event is found
// signalled: polled with ResetSignal until it is, and a second ResetSignal finds it clear, since the driver holds the
// clock until the batch is acknowledged. The batch's tick is recorded (the first 64) and, when no tick is skipped
// (STRICT), checked against the batch number; then the batch is acknowledged.
static void g_main_lockstep(Cpu *c) {
    uint64_t batches = LD64(W(BATCHES)), strict = LD64(W(STRICT));
    for (uint64_t b = 1; b <= batches; b++) {
        for (int i = 0; i < EVENTS; i++) {
            uint64_t r;
            while ((r = sv(c, 0x17, g_r[i], 0, 0, 0)) == R_INVALID_STATE) plat_pause();
            CHECK(r == 0);
        }
        for (int i = 0; i < EVENTS; i++) CHECK(sv(c, 0x17, g_r[i], 0, 0, 0) == R_INVALID_STATE);
        uint64_t t = rt_vsync_tick();
        if (b <= 64) ST64(W(SEEN) + 8 * (b - 1), t);
        if (strict) CHECK(t == b);
        st_rel(ACK, b);
    }
    exit_thread(c);
}

// 1: a worker of a worse priority that runs while the main thread waits, with a safe point in its loop
static void g_worker(Cpu *c) {
    st_rel(RAN, 1);
    while (!ld_acq(DONE)) sv(c, 0x1E, 0, 0, 0, 0);   // GetSystemTick: the SVC's exit is a safe point
    exit_thread(c);
}

// 2: one core. The main thread waits on a vsync event, so it gives the core up and the worker runs; the tick wakes it,
// and it takes the core back at the worker's next safe point (it has the better priority).
static void g_main_wake(Cpu *c) {
    uint32_t w = create(c, G(1), 0, 50, 0);
    start(c, w);
    CHECK(rt_vsync_tick() == 0);
    ST32(W(HANDLES) + 0x10, g_r[0]);
    CHECK(sv(c, 0x18, 0, W(HANDLES) + 0x10, 1, ~0ull) == 0 && c->x[1] == 0);
    ST64(W(RESULTS), rt_vsync_tick());
    ST64(W(RESULTS) + 8, ld_acq(RAN));
    CHECK(sv(c, 0x17, g_r[0], 0, 0, 0) == 0);
    st_rel(DONE, 1);
    join(c, w);
    exit_thread(c);
}

// 3: a waiter per event (its argument is the event's index): waits, clears, records the system counter, until STOP
static void g_waiter(Cpu *c) {
    uint64_t i = c->x[0], n = 0, h = W(HANDLES) + 0x20 + 0x10 * i;
    ST32(h, g_r[i]);
    sv(c, 0x17, g_r[i], 0, 0, 0);   // a tick before the thread existed is not one it waited for
    while (!ld_acq(STOP)) {
        CHECK(sv(c, 0x18, 0, h, 1, ~0ull) == 0);
        uint64_t t = rt_ticks();
        CHECK(sv(c, 0x17, g_r[i], 0, 0, 0) == 0);
        if (n < REALTIME_CAP) ST64(W(WAKES) + 8 * (i * REALTIME_CAP + n), t);
        n++;
    }
    ST64(W(RESULTS) + 8 * i, n);
    exit_thread(c);
}

// 4: the realtime case: a waiter per event, each on a core of its own, for REALTIME_S seconds
static void g_main_realtime(Cpu *c) {
    uint32_t h[EVENTS];
    for (int i = 0; i < EVENTS; i++) h[i] = create(c, G(3), (uint64_t)i, 40, i);
    for (int i = 0; i < EVENTS; i++) start(c, h[i]);
    sv(c, 0x0B, REALTIME_S * 1000000000ull, 0, 0, 0);   // SleepThread
    st_rel(STOP, 1);
    for (int i = 0; i < EVENTS; i++) join(c, h[i]);
    exit_thread(c);
}

static void guest_end(Cpu *c) { (void)c; }

static const GuestFn g_fns[] = {g_main_lockstep, g_worker, g_main_wake, g_waiter, g_main_realtime, guest_end};
#define NFN ((int)(sizeof g_fns / sizeof *g_fns))
static uint64_t g_addrs[NFN];

// ---- the host side
static const uint64_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static void boot(uint64_t cores) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = cores, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                        .program_id = 0x0100000000001000ull, .is_application = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++)
        k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    for (int i = 0; i < EVENTS; i++) g_w[i] = rt_kernel_event_create(&g_r[i]);
    for (int i = 0; i < NFN; i++) g_addrs[i] = G(i);
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, g_addrs, g_fns, (uint64_t)NFN, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 64ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
}

// ---- the driver, on the clock's thread
static struct {
    uint64_t now;          // the driven time (atomic: guest threads read it through rt_vsync_tick)
    uint64_t calls;        // wait_until calls so far
    uint64_t expect;       // the tick the next deadline must belong to
    uint64_t batches;      // batches the case runs; the clock's next wait lasts until the process ends
    uint64_t late_call;    // the call whose wake-up passes LATE_TICKS ticks (0: none)
    int jitter;            // wake up 0 to 15 ms after each deadline, varying
    int wait_ran;          // the first call waits until the worker runs (the wake case)
    uint64_t hour_due;     // the deadline the clock asked for tick HOUR_TICKS
    uint64_t bound;        // 1 once the case has bound every event it binds (atomic): tick 1 waits for it
} D;

static uint64_t drv_now(void *ctx) {
    (void)ctx;
    return __atomic_load_n(&D.now, __ATOMIC_ACQUIRE);
}

// Wait until the word at p reaches want; 10 s of wall time is a failure of the case. The driver runs on the clock's
// thread, which is above normal priority on Windows: past a millisecond of yielding it sleeps a millisecond between
// looks, so the thread it waits for gets a processor even when every processor has such a thread spinning (several
// copies of this program at once on a small runner).
static void await_word(const uint64_t *p, uint64_t want, uint64_t tick, const char *what) {
    uint64_t start = plat_now_ns(), end = start + 10000000000ull;
    while (__atomic_load_n(p, __ATOMIC_ACQUIRE) < want) {
        uint64_t now = plat_now_ns();
        if (now > end) {
            fprintf(stderr, "FAIL tick %llu: %s\n", (unsigned long long)tick, what);
            exit(1);
        }
        if (now - start < 1000000ull) plat_yield();
        else plat_sleep_ns(1000000ull);
    }
}

static void await_guest(uint64_t off, uint64_t want, uint64_t tick, const char *what) {
    await_word((const uint64_t *)(uintptr_t)W(off), want, tick, what);
}

static void all_bound(void) { __atomic_store_n(&D.bound, 1, __ATOMIC_RELEASE); }

// Every event bound, the first twice: an event bound twice is still one event, signalled once per tick. The clock
// starts at the first bind, and its first driven wait could return at once and send tick 1's signals to the events
// bound so far; one bound after that would wait for tick 2, which the driver holds until the guest has found tick 1's
// four, so the case would stop there. The driver therefore holds tick 1 until the case has bound every event
// (all_bound). To keep that hold tested, the case gives the clock its chance between the first bind and the others: it
// waits until the clock has signalled a tick, or 100 ms. Without the hold, the clock signals tick 1 to one event in
// that time, and the case fails at tick 2 every run instead of once in a while.
static void bind_all(void) {
    rt_vsync_bind(g_w[0]);
    for (uint64_t end = plat_now_ns() + 100000000ull; !rt_vsync_stats().ticks && plat_now_ns() < end;) plat_yield();
    for (int i = 1; i < EVENTS; i++) rt_vsync_bind(g_w[i]);
    rt_vsync_bind(g_w[0]);
    CHECK(rt_vsync_stats().events == EVENTS);
    all_bound();
}

static void drv_wait(void *ctx, uint64_t deadline) {
    (void)ctx;
    uint64_t k = ++D.calls, t = D.expect, want = ref_due_ns(t);
    if (deadline != want) {
        fprintf(stderr, "FAIL tick %llu: the clock waits for origin + %llu ns, the schedule says origin + %llu ns "
                        "(%lld ns off)\n", (unsigned long long)t, (unsigned long long)(deadline - ORIGIN),
                (unsigned long long)(want - ORIGIN), (long long)(deadline - want));
        exit(1);
    }
    if (t == HOUR_TICKS) D.hour_due = deadline;
    if (k > D.batches)
        for (;;) plat_sleep_ns(1000000000ull);   // the case has every batch it runs: the clock waits here for good
    if (k == 1) await_word(&D.bound, 1, t, "the case did not bind its events");
    if (k > 1) await_guest(ACK, k - 1, t, "the guest did not find every bound event signalled at the previous tick");
    if (D.wait_ran) await_guest(RAN, 1, t, "the thread waiting on the vsync event kept its core: the worker never ran");
    uint64_t now = deadline;
    if (k == D.late_call) {
        now = ref_due_ns(t + LATE_TICKS) + LATE_EXTRA_NS;
        D.expect = t + LATE_TICKS + 1;
    } else {
        // 0 to 15 ms: under a tick, since two deadlines are 16,666,656 ns apart at least
        if (D.jitter) now += k * 7919 % 16 * 1000000ull;
        D.expect = t + 1;
    }
    __atomic_store_n(&D.now, now, __ATOMIC_RELEASE);
}

static void drive(uint64_t batches, uint64_t late_call, int jitter, int wait_ran) {
    D.now = ORIGIN, D.expect = 1, D.batches = batches, D.late_call = late_call, D.jitter = jitter;
    D.wait_ran = wait_ran;
    RtVsyncDriver d = {.origin_ns = ORIGIN, .now = drv_now, .wait_until = drv_wait};
    rt_vsync_drive(&d);
}

// ---- cases
// The schedule: the due times against the reference above, the current tick at and around each, an hour exactly.
static void case_clock(void) {
    for (int i = 0; i < 1000; i++) {   // undriven: the current tick follows the system counter
        uint64_t c0 = rt_ticks(), n = rt_vsync_tick(), c1 = rt_ticks();
        CHECK(rt_vsync_due_counter(n) <= c1 && rt_vsync_due_counter(n + 1) > c0);
    }
    static const uint64_t ns[] = {1, 2, 3, 4, 59, 60, 61, 3599, 216000, 216001, 1ull << 32, 1ull << 36};
    CHECK(rt_vsync_due_counter(0) == 0);
    CHECK(rt_vsync_due_counter(1) == 520834 && rt_vsync_due_counter(2) == 1041667);
    CHECK(rt_vsync_due_counter(3) == 1562500);
    CHECK(rt_vsync_due_counter(HOUR_TICKS) == 3600 * COUNTER_PER_S);
    for (unsigned i = 0; i < sizeof ns / sizeof *ns; i++) {
        uint64_t n = ns[i];
        CHECK(rt_vsync_due_counter(n) == ref_due_counter(n));
        CHECK(rt_vsync_due_counter(n + 3) - rt_vsync_due_counter(n) == 1562500);   // three ticks: 1/20 s exactly
    }
    drive(0, 0, 0, 0);
    for (unsigned i = 0; i < sizeof ns / sizeof *ns; i++) {
        uint64_t n = ns[i], due = ref_due_ns(n);
        CHECK(rt_vsync_due_ns(n) == due);
        __atomic_store_n(&D.now, due, __ATOMIC_RELEASE);
        CHECK(rt_vsync_tick() == n);
        __atomic_store_n(&D.now, due - 1, __ATOMIC_RELEASE);
        CHECK(rt_vsync_tick() == n - 1);
        __atomic_store_n(&D.now, ref_due_ns(n + 1) - 1, __ATOMIC_RELEASE);
        CHECK(rt_vsync_tick() == n);
    }
    __atomic_store_n(&D.now, ORIGIN - 1, __ATOMIC_RELEASE);
    CHECK(rt_vsync_tick() == 0);
    RtVsyncStats s = rt_vsync_stats();
    CHECK(s.ticks == 0 && s.events == 0);
}

// An hour of driven ticks, each woken 0 to 15 ms late: every deadline on the schedule, every event once per tick.
static void case_hour(void) {
    boot(0x1);
    drive(HOUR_TICKS, 0, 1, 0);
    ST64(W(BATCHES), HOUR_TICKS);
    ST64(W(STRICT), 1);
    uint64_t t0 = plat_now_ns();
    bind_all();
    rt_kernel_run(G(0), 0);
    RtVsyncStats s = rt_vsync_stats();
    printf("hour: %llu ticks driven in %.1f s of wall time; tick %llu due at origin + %llu.%09llu s; %d events, each "
           "signalled once at every tick; %llu skipped; late at most %.3f ms (the driver's wake-ups)\n",
           (unsigned long long)s.ticks, (double)(plat_now_ns() - t0) / 1e9, (unsigned long long)HOUR_TICKS,
           (unsigned long long)((D.hour_due - ORIGIN) / 1000000000ull),
           (unsigned long long)((D.hour_due - ORIGIN) % 1000000000ull), s.events, (unsigned long long)s.skipped,
           (double)s.late_max_ns / 1e6);
    CHECK(D.hour_due - ORIGIN == 3600ull * 1000000000ull);
    CHECK(s.ticks == HOUR_TICKS && s.last == HOUR_TICKS && s.skipped == 0 && s.events == EVENTS);
    CHECK(s.late_max_ns == 15000000ull);
}

// One wake-up 2.5 ticks late: one batch for the tick in progress, two ticks counted as skipped, no burst to catch up,
// and the next deadline back on the grid (the driver checks it).
static void case_late(void) {
    boot(0x1);
    drive(20, 10, 0, 0);
    ST64(W(BATCHES), 20);
    ST64(W(STRICT), 0);
    bind_all();
    rt_kernel_run(G(0), 0);
    RtVsyncStats s = rt_vsync_stats();
    for (uint64_t b = 1; b <= 20; b++) CHECK(LD64(W(SEEN) + 8 * (b - 1)) == (b < 10 ? b : b + LATE_TICKS));
    printf("late: one wake-up %d ticks and %.1f ms late gave one batch for tick %d; %llu skipped, last tick %llu\n",
           LATE_TICKS, (double)LATE_EXTRA_NS / 1e6, 10 + LATE_TICKS, (unsigned long long)s.skipped,
           (unsigned long long)s.last);
    CHECK(s.ticks == 20 && s.skipped == LATE_TICKS && s.last == 20 + LATE_TICKS && s.late_max_ns == LATE_EXTRA_NS);
}

// A guest thread waiting on a vsync event gives its core up and is woken by the tick.
static void case_wake(void) {
    boot(0x1);
    drive(1, 0, 0, 1);
    rt_vsync_bind(g_w[0]);
    all_bound();
    rt_kernel_run(G(2), 0);
    uint64_t tick = LD64(W(RESULTS)), ran = LD64(W(RESULTS) + 8);
    printf("wake: the waiting thread woke at tick %llu; the worker ran while it waited: %llu\n",
           (unsigned long long)tick, (unsigned long long)ran);
    CHECK(tick == 1 && ran == 1);
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

// The host's clock for REALTIME_S seconds: per event, how many wakes, how many ticks seen, and the lateness of each
// wake after the due time of the tick it fell in, as the guest thread saw it (the signal, the kernel's wake-up and the
// thread taking its core). The bound is loose for continuous integration runners, whose timing margins are not
// measured; OOT2_VSYNC_P99_US sets a tighter one for a run on a quiet desktop.
//
// The case holds the system timer the way the program does (rt_timer_hold_start, ticket 116; timer.c), so the default,
// which holds 0.5 ms, is what runs, in CI too, and it prints the resolution it saw at the start and at the end of the
// run. The desktop's bound of 0.6 ms (research 55, Q3) is a variable of the run and never a default:
//   OOT2_VSYNC_P99_US=600 py -3.14 -I -m unittest discover -s tests -p test_runtime_vsync.py -v
// CI keeps the 4 ms above. OOT2_TIMER_RESOLUTION=0 runs the case on the system's own resolution, for an A/B run.
static void case_realtime(void) {
    const char *env = getenv("OOT2_VSYNC_P99_US");
    uint64_t bound_ns = (env && *env ? strtoull(env, NULL, 10) : 4000) * 1000ull;
    rt_timer_hold_start();
    uint64_t timer_start = plat_timer_resolution_ns();
    boot(0xF);
    bind_all();
    rt_kernel_run(G(4), 0);
    static uint64_t late[REALTIME_CAP];
    int bad = 0;
    for (int i = 0; i < EVENTS; i++) {
        uint64_t n = LD64(W(RESULTS) + 8 * (uint64_t)i), m = n < REALTIME_CAP ? n : REALTIME_CAP;
        uint64_t distinct = 0, prev = 0;
        uint64_t first = 0, last = 0, t0 = 0, t1 = 0;
        for (uint64_t j = 0; j < m; j++) {
            uint64_t t = LD64(W(WAKES) + 8 * ((uint64_t)i * REALTIME_CAP + j)), k = t * 60 / COUNTER_PER_S;
            late[j] = (t - ref_due_counter(k)) * NS_PER_COUNT;
            if (!j || k != prev) distinct++;
            if (!j) {
                first = k;
                t0 = t;
            }
            last = k;
            t1 = t;
            prev = k;
        }
        qsort(late, m, sizeof *late, cmp_u64);
        uint64_t elapsed = m ? last - first + 1 : 0, twice = m - distinct, missed = elapsed - distinct;
        uint64_t p50 = m ? late[m / 2] : 0, p99 = m ? late[(m * 99 + 99) / 100 - 1] : 0, max = m ? late[m - 1] : 0;
        double rate = t1 > t0 ? (double)(m - 1) * COUNTER_PER_S / (double)(t1 - t0) : 0;
        printf("realtime: event %d: %llu wakes over %llu ticks (%.3f a second), %llu ticks missed, %llu woken twice; "
               "lateness p50 %llu us, p99 %llu us, max %llu us (bound on p99: %llu us)\n", i, (unsigned long long)n,
               (unsigned long long)elapsed, rate, (unsigned long long)missed, (unsigned long long)twice,
               (unsigned long long)(p50 / 1000), (unsigned long long)(p99 / 1000), (unsigned long long)(max / 1000),
               (unsigned long long)(bound_ns / 1000));
        // 60 a second: about REALTIME_S * 60 wakes, a tick seen twice or missed at most once in a hundred
        if (n > REALTIME_CAP || m < REALTIME_S * 60 * 9 / 10 || twice * 100 > m || missed * 100 > elapsed) bad = 1;
        if (p99 > bound_ns) bad = 1;
    }
    RtVsyncStats s = rt_vsync_stats();
    printf("realtime: the clock signalled %llu ticks, skipped %llu, its signals at most %.3f ms after a due time\n",
           (unsigned long long)s.ticks, (unsigned long long)s.skipped, (double)s.late_max_ns / 1e6);
    printf("realtime: system timer resolution %llu ns at the start and %llu ns at the end (finest %llu ns; "
           "timer_resolution = %d, so the request %s)\n", (unsigned long long)timer_start,
           (unsigned long long)plat_timer_resolution_ns(), (unsigned long long)plat_timer_finest_ns(),
           (int)rt_setting(RT_SET_timer_resolution), rt_setting(RT_SET_timer_resolution) ? "was made" : "was not made");
    CHECK(!bad);
}

// A case that has not ended after HANG_NS exits 3: a hang reports where it stopped instead of running into the
// driver's timeout.
#define HANG_NS 120000000000ull
static void watchdog(void *arg) {
    (void)arg;
    plat_sleep_ns(HANG_NS);
    uint64_t h = __atomic_load_n(&HEAP, __ATOMIC_RELAXED);
    RtVsyncStats s = rt_vsync_stats();
    rt_log("HUNG: clock calls %llu, ticks %llu, ack %llu", (unsigned long long)D.calls, (unsigned long long)s.ticks,
           (unsigned long long)(h ? LD64(h + ACK) : 0));
    rt_kernel_dump();
    _Exit(3);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: vsync_test <case>\n");
        return 2;
    }
    CHECK(plat_thread_start(watchdog, NULL, 1ull << 20));
    const char *cs = argv[1];
    if (!strcmp(cs, "clock")) case_clock();
    else if (!strcmp(cs, "hour")) case_hour();
    else if (!strcmp(cs, "late")) case_late();
    else if (!strcmp(cs, "wake")) case_wake();
    else if (!strcmp(cs, "realtime")) case_realtime();
    else if (!strcmp(cs, "drive_after_start")) {   // fatal: the clock runs on the host's time already
        boot(0x1);
        rt_vsync_bind(g_w[0]);
        drive(1, 0, 0, 0);
    } else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
