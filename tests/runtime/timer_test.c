// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the system timer resolution (ticket 116): rt_timer_hold_start (runtime/core/timer.c) and the platform
// layer's plat_timer_hold and plat_timer_release (runtime/core/plat.c), driven by tests/test_runtime_timer.py:
// `timer_test <case>`, each case in its own process.
//
// The machine-wide timer resolution is shared with every process, so a case never assumes what the system default is:
// the cases that run on every platform install a recorder through plat_timer_ops, which answers the request and prints
// one unbuffered line per call ("timer seam: set 5000 on" or "off", the units of 100 ns), so what the process asked and
// gave back is read from its output even when _Exit follows. The one case that asks the real system (held) reads the
// resolution back while it holds the request and asserts only that it is at or under what the system allows; "the
// resolution went back to the default" is never asserted, since another process may hold a fine one at any time.
//
//   start          one request, made before the vsync clock's thread exists, then the clock runs on the held timer
//   normal         main returns: one release
//   exit_process   a guest thread issues ExitProcess (SVC 0x07): plat_exit releases
//   fatal          rt_fatal: exit 70, one release
//   fatal_in_hook  rt_fatal whose stop hook raises another fatal: exit 70, still one release
//   unimplemented  rt_unimplemented: exit 71
//   guest_stop     rt_brk, a guest stop: exit 72
//   census_end     rt_census_end: the census mode's end, through plat_exit
//   ctrl           (Windows) a console control event, raised by the process itself, ends it: one release, and the
//                  system's exit code 0xC000013A is unchanged
//   off            the setting is off through OOT2_TIMER_RESOLUTION=0 (set by the driver): no request, no release
//   off_set        the setting is off through --set, with the store loaded: no request, no release
//   held           no recorder: the real system's resolution read back while held (Windows)
//
// The guest function of exit_process is a C function in the lifter's shape, with endmark.c linked right after this file,
// as in vsync_test.c.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

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
#define G(i) (MB + 0x100ull * (i))
#define PC 0x7000000040ull
#define DEFAULT_NS 15625000ull   // what the recorder reports when nothing is held

// ---- the recorder
static int g_on, g_off;           // calls so far (touched by the main thread, and once by a control event's thread)
static int g_refuse;             // 1: set answers that the system refused
static uint64_t g_events_at_on = ~0ull, g_ticks_at_on = ~0ull;   // the vsync clock's state when the request came
static int g_in_force;

static int seam_set(uint64_t want_ns, int on, uint64_t *now_ns) {
    if (on) {
        g_on++;
        RtVsyncStats s = rt_vsync_stats();
        g_events_at_on = s.events;
        g_ticks_at_on = s.ticks;
    } else {
        g_off++;
    }
    printf("timer seam: set %llu %s\n", (unsigned long long)(want_ns / 100), on ? "on" : "off");
    g_in_force = on;
    *now_ns = on ? want_ns : DEFAULT_NS;
    return g_refuse ? -1 : 0;
}

static uint64_t seam_now(void) { return g_in_force ? 500000ull : DEFAULT_NS; }

static void install_seam(void) {
    static const PlatTimerOps ops = {seam_set, seam_now};
    plat_timer_ops(&ops);
}

// ---- a guest program and a kernel, for the cases that need guest threads
static const uint64_t APP_SVCS[] = {0x01, 0x07, 0x0A};

static void g_main_exit(Cpu *c) {   // 0: ExitProcess
    c->x[0] = c->x[1] = c->x[2] = c->x[3] = 0;
    rt_svc(c, 0x07, PC);
    CHECK(0);   // never returns
}

static void guest_end(Cpu *c) { (void)c; }

static const GuestFn g_fns[] = {g_main_exit, guest_end};
#define NFN ((int)(sizeof g_fns / sizeof *g_fns))
static uint64_t g_addrs[NFN];

static void boot(void) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x1, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512,
                        .program_id = 0x0100000000001000ull, .is_application = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++)
        k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    for (int i = 0; i < NFN; i++) g_addrs[i] = G(i);
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, g_addrs, g_fns, (uint64_t)NFN, TEST_END};
    rt_dispatch_init(mods, 1);
}

// ---- cases
// The request comes before the clock's thread exists (no event is bound, so no tick was sent), and the clock then runs.
static void case_start(void) {
    install_seam();
    rt_timer_hold_start();
    CHECK(g_on == 1 && g_off == 0);
    CHECK(g_events_at_on == 0 && g_ticks_at_on == 0);
    CHECK(plat_timer_resolution_ns() == 500000ull);
    boot();
    uint32_t r;
    uint32_t w = rt_kernel_event_create(&r);
    rt_vsync_bind(w);   // starts the clock's thread
    uint64_t end = plat_now_ns() + 5000000000ull;
    while (!rt_vsync_stats().ticks && plat_now_ns() < end) plat_sleep_ns(1000000ull);
    CHECK(rt_vsync_stats().ticks >= 1);
    CHECK(g_on == 1 && g_off == 0);
    printf("start: one request of 0.5 ms with %llu events bound and %llu ticks sent; the clock then ran\n",
           (unsigned long long)g_events_at_on, (unsigned long long)g_ticks_at_on);
}

static void case_normal(void) {
    install_seam();
    rt_timer_hold_start();
    CHECK(g_on == 1 && g_off == 0);
    rt_timer_hold_start();   // a second start changes nothing
    CHECK(g_on == 1 && g_off == 0);
    printf("normal: returning from main\n");
}

static void case_exit_process(void) {
    install_seam();
    rt_timer_hold_start();
    CHECK(g_on == 1 && g_off == 0);
    boot();
    rt_kernel_run(G(0), 0);   // the main thread ends the process
    CHECK(0);
}

static void case_fatal(void) {
    install_seam();
    rt_timer_hold_start();
    CHECK(g_on == 1 && g_off == 0);
    rt_fatal("timer test: a fatal error");
}

static void hook_that_fails(int code, const char *line) {
    (void)code, (void)line;
    rt_fatal("timer test: a second fatal error, inside the stop hook");
}

static void case_fatal_in_hook(void) {
    install_seam();
    rt_timer_hold_start();
    rt_set_stop_hook(hook_that_fails);
    rt_fatal("timer test: a fatal error whose hook fails");
}

static void case_unimplemented(void) {
    install_seam();
    rt_timer_hold_start();
    Cpu c;
    memset(&c, 0, sizeof c);
    rt_unimplemented(&c, "test", "timer test: an unimplemented thing");
}

static void case_guest_stop(void) {
    install_seam();
    rt_timer_hold_start();
    Cpu c;
    memset(&c, 0, sizeof c);
    rt_brk(&c, 0x7, 0x7000000100ull);
}

static void case_census_end(void) {
    install_seam();
    rt_timer_hold_start();
    CHECK(g_on == 1 && g_off == 0);
    rt_census_end(0, "timer test: the census mode ended the run");
}

static void case_ctrl(void) {
#ifdef _WIN32
    // The driver started this process in a group of its own, so a Ctrl+Break to the group is a Ctrl+Break to us and to
    // nobody else. A process with no console (a runner's service session, or DETACHED_PROCESS) has nothing to raise it
    // on: it makes one first, since AllocConsole drops the control handlers installed before it.
    DWORD list[2];
    if (!GetConsoleProcessList(list, 2)) {
        fprintf(stderr, "ctrl: no console to raise the event on; allocating one\n");
        CHECK(AllocConsole());
    }
    install_seam();
    rt_timer_hold_start();
    CHECK(g_on == 1 && g_off == 0);
    CHECK(GenerateConsoleCtrlEvent(CTRL_BREAK_EVENT, GetCurrentProcessId()));
    Sleep(15000);   // the system's handler ends the process; the control event's thread has released the request
    fprintf(stderr, "FAIL: the control event did not end the process\n");
    exit(1);
#else
    fprintf(stderr, "ctrl: Windows only\n");
    exit(2);
#endif
}

static void case_off(void) {   // OOT2_TIMER_RESOLUTION=0 in the environment, the store never loaded
    install_seam();
    rt_timer_hold_start();
    CHECK(g_on == 0 && g_off == 0);
    CHECK(rt_setting(RT_SET_timer_resolution) == 0);
    printf("off: no request\n");
}

static void case_off_set(void) {   // --set timer_resolution=0 through the store
    install_seam();
    static const RtSettingArg args[] = {{"timer_resolution=0", "--set"}};
    char why[600];
    CHECK(rt_settings_load(args, 1, NULL, why, sizeof why));
    rt_timer_hold_start();
    CHECK(g_on == 0 && g_off == 0);
    printf("off_set: no request\n");
}

static void case_on_by_default(void) {   // no variable, no layer: the default is on
    install_seam();
    CHECK(rt_setting(RT_SET_timer_resolution) == 1);
    rt_timer_hold_start();
    CHECK(g_on == 1);
    printf("on_by_default: requested\n");
}

static void case_refused(void) {   // the system refuses the request: the run goes on, nothing is held, nothing released
    install_seam();
    g_refuse = 1;
    rt_timer_hold_start();
    CHECK(g_on == 1 && g_off == 0);
    plat_timer_release();
    CHECK(g_off == 0);
    printf("refused: the run goes on\n");
}

#ifdef _WIN32
// This process's own request, asked of the system directly: NtSetTimerResolution(off) answers 0 when this process holds
// a request and STATUS_TIMER_RESOLUTION_NOT_SET when it does not, whatever other processes (a browser, another game
// run) hold. The read-back alone cannot tell: it passes when somebody else holds 0.5 ms.
#define STATUS_TIMER_RESOLUTION_NOT_SET ((LONG)0xC0000245)
typedef LONG(NTAPI *NtSetTimerResolutionFn)(ULONG want, BOOLEAN on, PULONG now);
static LONG direct_off(void) {
    static NtSetTimerResolutionFn fn;
    if (!fn) fn = (NtSetTimerResolutionFn)(void *)GetProcAddress(GetModuleHandleW(L"ntdll.dll"), "NtSetTimerResolution");
    CHECK(fn != NULL);
    ULONG cur = 0;
    return fn(5000, FALSE, &cur);
}
#endif

// The real system: held, read back. At or under 0.5 ms, or the finest the system allows when that is coarser (a virtual
// machine's timer may be); never an assertion about what it was or will be when released. Two process-local checks
// besides, which hold whatever else runs on the machine: the request is this process's (taking it back directly
// answers 0, then NOT_SET), and plat_timer_release gives it back (a direct take-back right after answers NOT_SET).
static void case_held(void) {
    rt_timer_hold_start();
    uint64_t now = plat_timer_resolution_ns(), finest = plat_timer_finest_ns();
    printf("held: resolution %llu ns while held, the finest the system allows %llu ns\n", (unsigned long long)now,
           (unsigned long long)finest);
#ifdef _WIN32
    uint64_t bound = finest > 500000ull ? finest : 500000ull;
    CHECK(now > 0 && finest > 0 && now <= bound);
    CHECK(direct_off() == 0);   // this process held a request
    CHECK(direct_off() == STATUS_TIMER_RESOLUTION_NOT_SET);
    plat_timer_release();       // the flag says held: the call it makes finds nothing, and is silent
    rt_timer_hold_start();      // a second, whole request
    plat_timer_release();
    CHECK(direct_off() == STATUS_TIMER_RESOLUTION_NOT_SET);   // the release really gave the request back
    printf("direct: this process's request taken back directly once, the release found it gone\n");
#else
    CHECK(now == 0 && finest == 0);
    plat_timer_release();
    plat_timer_release();   // a second release is silent
#endif
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc != 2) {
        fprintf(stderr, "usage: timer_test <case>\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "start")) case_start();
    else if (!strcmp(cs, "normal")) case_normal();
    else if (!strcmp(cs, "exit_process")) case_exit_process();
    else if (!strcmp(cs, "fatal")) case_fatal();
    else if (!strcmp(cs, "fatal_in_hook")) case_fatal_in_hook();
    else if (!strcmp(cs, "unimplemented")) case_unimplemented();
    else if (!strcmp(cs, "guest_stop")) case_guest_stop();
    else if (!strcmp(cs, "census_end")) case_census_end();
    else if (!strcmp(cs, "ctrl")) case_ctrl();
    else if (!strcmp(cs, "off")) case_off();
    else if (!strcmp(cs, "off_set")) case_off_set();
    else if (!strcmp(cs, "on_by_default")) case_on_by_default();
    else if (!strcmp(cs, "refused")) case_refused();
    else if (!strcmp(cs, "held")) case_held();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    printf("ok %s\n", cs);
    return 0;
}
