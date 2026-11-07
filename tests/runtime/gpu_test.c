// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the Vulkan back end's module (runtime/gpu, oot2/gpu.h, oot2/gpu_vk.h) that need no GPU: the
// back-end choice, the runtime's embedded shaders and Volk; the timeline and its completion thread over a test's
// counter; the settings and the choice of a device from synthetic candidates; the device-loss report's file; the check
// of a recording the back end cannot replay yet. Driven by tests/test_runtime_gpu.py: `gpu_test <case> [folder]`. Each
// case runs in its own process. A passing case prints "ok <case>"; a failing case must end in one named fatal, which the
// driver checks. Nothing here needs a Vulkan device: continuous integration has none (the device's own tests are
// tests/runtime/vk_test.c).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/loader.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

// Every embedded shader: "shader <name> <bytes> <SHA-256 of its SPIR-V>", for the driver to compare with glslang.
static void case_shaders(void) {
    CHECK(gpu_shader_count > 0);
    for (int i = 0; i < gpu_shader_count; i++) {
        const GpuShader *s = &gpu_shaders[i];
        CHECK(s->name && s->code);
        CHECK(s->size >= 20 && s->size % 4 == 0);
        CHECK(s->code[0] == 0x07230203u);   // the SPIR-V magic number, in the host's order
        uint8_t d[32];
        rt_sha256(s->code, s->size, d);
        printf("shader %s %llu ", s->name, (unsigned long long)s->size);
        for (int k = 0; k < 32; k++) printf("%02x", d[k]);
        printf("\n");
    }
    printf("ok shaders %d\n", gpu_shader_count);
}

static void case_lookup(void) {
    const GpuShader *fill = gpu_shader("fill");
    CHECK(fill && !strcmp(fill->name, "fill"));
    CHECK(fill >= gpu_shaders && fill < gpu_shaders + gpu_shader_count);
    CHECK(gpu_shader("no_such_shader") == NULL);
    CHECK(gpu_shader("") == NULL);
    printf("ok lookup\n");
}

static void case_select_null(void) {
    CHECK(gfx_backend_known("null") && gfx_backend_known("vulkan"));
    CHECK(!gfx_backend_known("metal") && !gfx_backend_known("") && !gfx_backend_known("Null"));
    CHECK(!strcmp(GFX_DEFAULT_BACKEND, "null"));
    gfx_select(GFX_DEFAULT_BACKEND, "the default");
    CHECK(gfx_backend() == &gfx_null_backend);
    printf("ok select_null\n");
}

// The Vulkan back end opens its device, or stops by name (no loader, no device, no device with every item, a bad
// setting). On a machine with a device that has every item the case returns with the table installed.
static void case_select_vulkan(void) {
    gfx_select("vulkan", "--gfx");
    CHECK(gfx_backend() == &gfx_vulkan_backend);
    printf("gfx_select returned for vulkan\n");
}

static void case_select_unknown(void) {
    gfx_select("metal", "OOT2_GFX");
    printf("gfx_select returned for metal\n");
}

// Volk is compiled into the runtime and its entry points link; before volkInitialize it knows no instance version.
static void case_volk(void) {
    CHECK(volkGetInstanceVersion() == 0);
    printf("volk: Vulkan headers %d\n", VK_HEADER_VERSION);
    printf("ok volk\n");
}

// ---------- settings
static void case_settings(void) {
    GpuVkSettings s;
    char why[256];
    CHECK(gpu_vk_settings(NULL, NULL, NULL, &s, why, sizeof why));
    CHECK(!s.device[0] && s.validation == GPU_VALIDATION_OFF && s.fault_report == 1);
    CHECK(gpu_vk_settings("", "", "", &s, why, sizeof why));
    CHECK(!s.device[0] && s.validation == GPU_VALIDATION_OFF && s.fault_report == 1);
    CHECK(gpu_vk_settings("rtx 3060", "sync", "off", &s, why, sizeof why));
    CHECK(!strcmp(s.device, "rtx 3060") && s.validation == GPU_VALIDATION_SYNC && s.fault_report == 0);
    CHECK(gpu_vk_settings("1", "on", "on", &s, why, sizeof why));
    CHECK(!strcmp(s.device, "1") && s.validation == GPU_VALIDATION_ON && s.fault_report == 1);
    CHECK(gpu_vk_settings(NULL, "gpu", NULL, &s, why, sizeof why) && s.validation == GPU_VALIDATION_GPU);
    const char *bad[][3] = {{NULL, "On", NULL}, {NULL, "1", NULL}, {NULL, NULL, "1"}, {NULL, NULL, "yes"}};
    for (int i = 0; i < 4; i++) {
        CHECK(!gpu_vk_settings(bad[i][0], bad[i][1], bad[i][2], &s, why, sizeof why));
        printf("refused: %s\n", why);
    }
    char longname[400];
    memset(longname, 'x', sizeof longname - 1);
    longname[sizeof longname - 1] = 0;
    CHECK(!gpu_vk_settings(longname, NULL, NULL, &s, why, sizeof why));
    printf("refused: %s\n", why);
    printf("ok settings\n");
}

// ---------- the choice of a device, from synthetic candidates
static GpuVkCandidate candidate(const char *name, VkPhysicalDeviceType type, int lack1, int lack2) {
    GpuVkCandidate c = {.name = name, .type = type};
    memset(c.has, 1, sizeof c.has);
    if (lack1 >= 0) c.has[lack1] = 0;
    if (lack2 >= 0) c.has[lack2] = 0;
    return c;
}

static void choose_line(const char *label, const GpuVkCandidate *c, int n, const char *setting) {
    char why[2048];
    int pick = gpu_vk_choose(c, n, setting, why, sizeof why);
    printf("choose %s: %d%s%s\n", label, pick, pick < 0 ? " " : "", pick < 0 ? why : "");
}

static void case_choose(void) {
    const VkPhysicalDeviceType D = VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU, I = VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU,
                               C = VK_PHYSICAL_DEVICE_TYPE_CPU;
    const int mesh = OOT2_VK_ITEM_mesh_shaders, atomic = OOT2_VK_ITEM_float32_atomic_add;
    GpuVkCandidate a[] = {candidate("Integrated Thing", I, -1, -1), candidate("NVIDIA GeForce RTX 3060 Ti", D, -1, -1)};
    choose_line("discrete-first", a, 2, "");
    GpuVkCandidate b[] = {candidate("NVIDIA GeForce GTX 1060 6GB", D, mesh, atomic), candidate("Integrated Thing", I, -1, -1)};
    choose_line("lacking-discrete", b, 2, NULL);
    GpuVkCandidate c[] = {candidate("llvmpipe", C, -1, -1)};
    choose_line("cpu-only", c, 1, "");
    GpuVkCandidate d[] = {candidate("NVIDIA GeForce GTX 1060 6GB", D, mesh, -1), candidate("llvmpipe", C, mesh, atomic)};
    choose_line("none-usable", d, 2, "");
    choose_line("index", a, 2, "0");
    choose_line("index-past", a, 2, "5");
    choose_line("name", a, 2, "rtx 3060");
    choose_line("name-lacking", b, 2, "gtx");
    choose_line("name-none", a, 2, "radeon");
    GpuVkCandidate e[] = {candidate("NVIDIA GeForce RTX 3060 Ti", D, mesh, -1),
                          candidate("NVIDIA GeForce RTX 3060 Ti", D, -1, -1)};
    choose_line("name-second-usable", e, 2, "3060");
    choose_line("index-lacking", e, 2, "0");
    GpuVkCandidate f[] = {candidate("Queueless", D, -1, -1)};
    f[0].unusable = "no queue family with graphics and compute";
    choose_line("unusable", f, 1, "");
    choose_line("empty", a, 0, "");
    printf("ok choose\n");
}

// ---------- the timeline over a test's counter
static PlatMutex g_src_lock = PLAT_MUTEX_INIT;
static PlatCond g_src_cond = PLAT_COND_INIT;
static uint64_t g_reached;   // the "device's" counter, raised by the test

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

static uint64_t counter_now(void) {
    plat_mutex_lock(&g_src_lock);
    uint64_t r = g_reached;
    plat_mutex_unlock(&g_src_lock);
    return r;
}

static void start_counter(void) {
    GpuTimelineSource src = {.wait = counter_wait};
    gpu_sync_start(&src, 0);
}

// Threads the tests start, joined by counting the ones that ended.
static PlatMutex g_join_lock = PLAT_MUTEX_INIT;
static PlatCond g_join_cond = PLAT_COND_INIT;
static int g_ended;
static void ended(void) {
    plat_mutex_lock(&g_join_lock);
    g_ended++;
    plat_cond_broadcast(&g_join_cond);
    plat_mutex_unlock(&g_join_lock);
}
static void join(int n) {
    plat_mutex_lock(&g_join_lock);
    while (g_ended < n) plat_cond_wait(&g_join_cond, &g_join_lock);
    plat_mutex_unlock(&g_join_lock);
}

static uint64_t next_random(uint64_t *s) {
    *s ^= *s << 13, *s ^= *s >> 7, *s ^= *s << 17;
    return *s;
}

#define ORDER_SUBMITTERS 2
#define ORDER_PER_THREAD 5000
#define ORDER_VALUES (ORDER_SUBMITTERS * ORDER_PER_THREAD)
static int g_want[ORDER_VALUES + 1];     // effects each value queued (set before its gpu_sync_end)
static int g_ran[ORDER_VALUES + 1];      // effects of each value that ran
static uint64_t g_last_effect;           // value << 8 | index of the last effect that ran (the completion thread only)
static uint64_t g_effects, g_waits_checked;
static int g_submitters_left = ORDER_SUBMITTERS;

// An effect: the device reached its value; every earlier value is done and its own is not; effects run in value order
// and, within a value, in the order they were queued.
static void order_effect(void *arg, uint64_t value) {
    uint64_t index = (uint64_t)(uintptr_t)arg;
    CHECK(value >= 1 && value <= ORDER_VALUES);
    CHECK(counter_now() >= value);
    CHECK(gpu_sync_done() == value - 1);
    uint64_t key = value << 8 | index;
    CHECK(key > g_last_effect);
    CHECK(index == (uint64_t)__atomic_load_n(&g_ran[value], __ATOMIC_RELAXED));
    g_last_effect = key;
    __atomic_add_fetch(&g_ran[value], 1, __ATOMIC_RELEASE);
    g_effects++;
}

static void order_submitter(void *arg) {
    uint64_t seed = 0x9E3779B97F4A7C15ull * (1 + (uintptr_t)arg), last = 0;
    for (int i = 0; i < ORDER_PER_THREAD; i++) {
        uint64_t v = gpu_sync_begin();
        CHECK(v > last && v <= ORDER_VALUES);
        last = v;
        int k = (int)(next_random(&seed) % 4);   // 0 to 3 effects
        for (int j = 0; j < k; j++) gpu_sync_effect(order_effect, (void *)(uintptr_t)j);
        g_want[v] = k;
        gpu_sync_end();
        if (next_random(&seed) % 64 == 0) plat_yield();
    }
    __atomic_sub_fetch(&g_submitters_left, 1, __ATOMIC_RELEASE);
    ended();
}

// The "GPU": raises the counter toward what was issued, in uneven steps.
static void order_device(void *arg) {
    uint64_t seed = 12345;
    (void)arg;
    for (;;) {
        uint64_t issued = gpu_sync_issued(), now = counter_now();
        if (now < issued) counter_raise(now + 1 + next_random(&seed) % (issued - now));
        else if (!__atomic_load_n(&g_submitters_left, __ATOMIC_ACQUIRE) && now == ORDER_VALUES) break;
        else plat_yield();
    }
    ended();
}

// A guest that waits: once a wait for v returns 1, every effect of v has run.
static void order_waiter(void *arg) {
    uint64_t seed = 777;
    (void)arg;
    while (__atomic_load_n(&g_submitters_left, __ATOMIC_ACQUIRE) || gpu_sync_done() < ORDER_VALUES) {
        uint64_t handed = gpu_sync_handed();
        if (!handed) {
            plat_yield();
            continue;
        }
        uint64_t v = 1 + next_random(&seed) % handed;
        if (!gpu_sync_wait(v, 2000000000ull)) continue;
        CHECK(gpu_sync_done() >= v);
        CHECK(__atomic_load_n(&g_ran[v], __ATOMIC_ACQUIRE) == g_want[v]);
        g_waits_checked++;
    }
    ended();
}

static void case_sync_order(void) {
    start_counter();
    for (int i = 0; i < ORDER_SUBMITTERS; i++) CHECK(plat_thread_start(order_submitter, (void *)(uintptr_t)i, 1 << 20));
    CHECK(plat_thread_start(order_device, NULL, 1 << 20));
    CHECK(plat_thread_start(order_waiter, NULL, 1 << 20));
    join(ORDER_SUBMITTERS + 2);
    gpu_sync_finish();
    CHECK(gpu_sync_handed() == ORDER_VALUES && gpu_sync_issued() == ORDER_VALUES && gpu_sync_done() == ORDER_VALUES);
    uint64_t want = 0;
    for (int v = 1; v <= ORDER_VALUES; v++) {
        CHECK(g_ran[v] == g_want[v]);
        want += (uint64_t)g_want[v];
    }
    CHECK(g_effects == want);
    gpu_sync_stop();
    printf("ok sync_order values %d effects %llu waits %llu\n", ORDER_VALUES, (unsigned long long)g_effects,
           (unsigned long long)g_waits_checked);
}

static int g_slow_ran;
static void slow_effect(void *arg, uint64_t value) {
    (void)arg, (void)value;
    plat_sleep_ns(50000000ull);
    __atomic_store_n(&g_slow_ran, 1, __ATOMIC_RELEASE);
}

// A wait that times out, one that succeeds later, a value never handed out, and finish.
static void case_sync_wait(void) {
    start_counter();
    uint64_t v = gpu_sync_begin();
    gpu_sync_end();
    CHECK(v == 1 && gpu_sync_done() == 0);
    uint64_t t0 = plat_now_ns();
    CHECK(gpu_sync_wait(v, 30000000ull) == 0);
    uint64_t waited = plat_now_ns() - t0;
    CHECK(waited >= 30000000ull);
    t0 = plat_now_ns();
    CHECK(gpu_sync_wait(v + 5, 1000000000ull) == 0);   // never handed out: at once
    uint64_t unhanded = plat_now_ns() - t0;
    CHECK(unhanded < 100000000ull);
    counter_raise(v);
    CHECK(gpu_sync_wait(v, 5000000000ull) == 1);
    CHECK(gpu_sync_done() == v);
    uint64_t w = gpu_sync_begin();
    gpu_sync_effect(slow_effect, NULL);
    gpu_sync_end();
    counter_raise(w);
    gpu_sync_finish();
    CHECK(__atomic_load_n(&g_slow_ran, __ATOMIC_ACQUIRE) && gpu_sync_done() == w);
    gpu_sync_stop();
    printf("ok sync_wait timed out after %.1f ms, unhanded answered in %.3f ms\n", (double)waited / 1e6,
           (double)unhanded / 1e6);
}

static void case_sync_stop_in_flight(void) {
    start_counter();
    gpu_sync_begin();
    gpu_sync_end();
    gpu_sync_stop();
    printf("gpu_sync_stop returned\n");
}

// ---------- the device-loss report's file
static void case_report_write(const char *dir) {
    uint8_t bin[300];
    for (int i = 0; i < (int)sizeof bin; i++) bin[i] = (uint8_t)(i * 7 + 3);
    char a[4300], b[4300], c[4300];
    CHECK(gpu_write_report(dir, "first report\n", bin, sizeof bin, a, sizeof a));
    CHECK(gpu_write_report(dir, "second report\n", NULL, 0, b, sizeof b));
    CHECK(strcmp(a, b));
    printf("report %s\nreport %s\n", a, b);
    char file[4300];   // a folder that cannot be made: its name is taken by a file
    snprintf(file, sizeof file, "%s/blocker", dir);
    FILE *f = fopen(file, "wb");
    CHECK(f && fclose(f) == 0);
    CHECK(!gpu_write_report(file, "third report\n", NULL, 0, c, sizeof c));
    printf("refused: %s\n", c);
    printf("ok report_write\n");
}

// ---------- the check of a recording
static uint64_t recording(uint8_t *buf, const uint32_t *ops, const uint32_t *sizes, int n) {
    size_t at = 0;
    for (int i = 0; i < n; i++) {
        op_hdr h = {ops[i], sizes[i]};
        memset(buf + at, 0, sizes[i]);
        memcpy(buf + at, &h, sizeof h);
        at += sizes[i];
    }
    return (uint64_t)(uintptr_t)buf;
}

static _Alignas(8) uint8_t g_rec[256];

static void case_recording_empty(void) {
    const uint32_t ops[] = {OP_BEGIN, OP_END}, sizes[] = {8, 8};
    gpu_check_recording(7, 0, recording(g_rec, ops, sizes, 2));
    printf("ok recording_empty\n");
}

static void case_recording_op(void) {
    const uint32_t ops[] = {OP_BEGIN, OP_DRAW, OP_END}, sizes[] = {8, 32, 8};
    gpu_check_recording(7, 1, recording(g_rec, ops, sizes, 3));
    printf("gpu_check_recording returned\n");
}

static void case_recording_broken(void) {
    const uint32_t ops[] = {OP_DRAW, OP_END}, sizes[] = {16, 8};
    gpu_check_recording(9, 0, recording(g_rec, ops, sizes, 2));
    printf("gpu_check_recording returned\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: gpu_test <case> [folder]\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "shaders")) case_shaders();
    else if (!strcmp(cs, "lookup")) case_lookup();
    else if (!strcmp(cs, "select_null")) case_select_null();
    else if (!strcmp(cs, "select_vulkan")) case_select_vulkan();
    else if (!strcmp(cs, "select_unknown")) case_select_unknown();
    else if (!strcmp(cs, "volk")) case_volk();
    else if (!strcmp(cs, "settings")) case_settings();
    else if (!strcmp(cs, "choose")) case_choose();
    else if (!strcmp(cs, "sync_order")) case_sync_order();
    else if (!strcmp(cs, "sync_wait")) case_sync_wait();
    else if (!strcmp(cs, "sync_stop_in_flight")) case_sync_stop_in_flight();
    else if (!strcmp(cs, "report_write") && argc >= 3) case_report_write(argv[2]);
    else if (!strcmp(cs, "recording_empty")) case_recording_empty();
    else if (!strcmp(cs, "recording_op")) case_recording_op();
    else if (!strcmp(cs, "recording_broken")) case_recording_broken();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
