// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkwatchcost: spike S5 (architecture D14, D19 and D27). What the write watch and host-memory import cost with the
// guest heap committed at the game's scale, measured on the runtime's own address space (runtime/core/memory.c) with
// synthetic guest memory and no game code. One process per measurement: tools/watchcost/watch_cost.py runs the matrix
// and checks the machine before each run.
//
//   vkwatchcost watch COMMIT_MIB SCOPE DIRTY ITERS [SEED]
//       SetHeapSize(COMMIT_MIB) with every page touched; then ITERS times: write DIRTY distinct pages (one 8-byte store
//       each, timed), then one rt_mem_written query with reset over SCOPE (timed). Scopes: heap (the heap as
//       SetHeapSize mapped it), region (the whole heap region as reserved), alias (the alias region as reserved, nothing
//       mapped in it), pools (4,500 ranges of 64 KiB to 64 MiB that tile the heap, one query each).
//   vkwatchcost import COMMIT_MIB ITERS [WINDOW_MIB [PLACE]]
//       the heap as above; ITERS times: import a window of it (64 MiB unless stated) through VK_EXT_external_memory_host
//       into a buffer with device-address usage, bind it, read its address, destroy the buffer and free the memory,
//       each step timed. PLACE: cycle (the windows in turn, the default), first (the heap's first window, with the
//       rest of the committed heap after it), last (its last window), bounded (the first window, with the page after
//       it made read-only during the import, so that the run of pages of one kind ends at the window).
//   vkwatchcost hold COMMIT_MIB IMPORT_MIB touch|notouch
//       the heap committed (every page touched, or not); memory before and after importing IMPORT_MIB in 64 MiB windows
//       held at once, after the process trims its own working set, and after the imports are freed and it trims again.
//   vkwatchcost behave [COUNTER]
//       verdicts: B1 an import over a window with one uncommitted 64 KiB chunk fails; B2 an import over committed pages
//       succeeds and the GPU and the CPU see the same bytes; B3 CPU writes into imported pages show in the write watch.
//       Findings without a verdict: B4 whether GPU writes into imported pages show in it; B5 what a query with reset
//       does with written pages past the end of its buffer; B6 whether the system answers a query that spans the heap
//       and alias reservations; B7 what a 64 MiB import of committed pages does right after B1's failed import, and
//       after a smaller import has succeeded. COUNTER breaks one setup so that its verdict must turn FAIL: fill-hole
//       (B1), hole (B2), no-write (B3).
//
// Output: one line per fact in "key=value" fields; "samples" lines hold every timed value in microseconds. Available
// RAM is read before and after the work and at every iteration: below 1 GiB the result says so, below 0.5 GiB the probe
// stops with exit 5 and the run is invalid (architecture section 1). Third-party implicit Vulkan layers are the
// caller's to disable (VK_LOADER_LAYERS_DISABLE=~implicit~). Windows only.
//
// Exit codes: 0 measured (behave: every verdict PASS), 1 a verdict FAIL or a check failed, 2 usage, or Vulkan lacks
// something, 3 a Vulkan call failed, 5 available RAM fell below 0.5 GiB, 70 runtime fatal.

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

#define VK_NO_PROTOTYPES
#include "volk.h"

#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/memory.h"
#include "oot2/runtime.h"

#define KiB (1ull << 10)
#define MiB (1ull << 20)
#define GiB (1ull << 30)
#define PAGE 0x1000ull
#define WINDOW (64 * MiB)
#define CHUNK (64 * KiB)
#define POOLS 4500
#define MAX_REPORT (1ull << 21)   // page addresses a query may return: the 8 GiB heap region's pages
#define INVALID_BELOW (512 * MiB)
#define REPORT_BELOW (1 * GiB)

__attribute__((noreturn, format(printf, 2, 3))) static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "vkwatchcost: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stdout);
    exit(code);
}

static double now_us(void) {
    static double scale;
    LARGE_INTEGER t;
    if (!scale) {
        LARGE_INTEGER f;
        QueryPerformanceFrequency(&f);
        scale = 1e6 / (double)f.QuadPart;
    }
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart * scale;
}

static double mib(uint64_t v) { return (double)v / MiB; }

// ---- the machine: available RAM, with the guard of architecture section 1
static uint64_t min_avail = UINT64_MAX;

static uint64_t avail_bytes(void) {
    MEMORYSTATUSEX m = {.dwLength = sizeof m};
    return GlobalMemoryStatusEx(&m) ? m.ullAvailPhys : 0;
}

static uint64_t avail_check(const char *when) {
    uint64_t a = avail_bytes();
    if (a < min_avail) min_avail = a;
    if (a < INVALID_BELOW) {
        printf("invalid when=%s avail_mib=%.0f: available RAM fell below 0.5 GiB\n", when, mib(a));
        die(5, "available RAM %.0f MiB below 0.5 GiB at %s: the run is invalid", mib(a), when);
    }
    return a;
}

// ---- the guest heap
static Cpu cpu;

static uint64_t svc(uint32_t imm, uint64_t x0, uint64_t x1) {
    cpu.x[0] = x0, cpu.x[1] = x1;
    rt_svc(&cpu, imm, 0);
    return cpu.x[0];
}

static uint64_t info(uint32_t id) {
    uint64_t v = 0;
    if (!rt_mem_info(id, &v)) die(1, "GetInfo %u is not a memory id", id);
    return v;
}

// The runtime's default address space (heap region 8 GiB, alias 64 GiB, pool 9 GiB) with the heap set to bytes and
// nothing committed yet. Returns the heap's address.
static uint64_t heap_map(uint64_t bytes) {
    RtMemConfig cfg = rt_mem_default_config();
    rt_mem_init(&cfg);
    uint32_t r = (uint32_t)svc(0x01, 0, bytes);
    if (r) die(1, "SetHeapSize(%#llx) returned %#x", (unsigned long long)bytes, r);
    uint64_t heap = cpu.x[1];
    if (heap != info(4)) die(1, "SetHeapSize put the heap at %#llx, GetInfo says %#llx", (unsigned long long)heap,
                             (unsigned long long)info(4));
    return heap;
}

// heap_map, then, when touch is set, one store into every page, which commits it through the first-touch handler as
// the game's own first stores do (64 KiB chunks) and makes it resident; otherwise the heap is committed by
// rt_mem_commit_io and left untouched (committed, not resident).
static uint64_t heap_setup(uint64_t bytes, int touch) {
    const uint64_t heap = heap_map(bytes);
    double t0 = now_us();
    if (touch) {
        for (uint64_t off = 0; off < bytes; off += PAGE) {
            *(volatile uint64_t *)(uintptr_t)(heap + off) = off;
            if (!(off & (GiB - 1))) avail_check("touch");
        }
    } else if (!rt_mem_commit_io(heap, bytes, 1)) {
        die(1, "committing the heap failed");
    }
    printf("setup commit_mib=%llu touch=%d ms=%.1f heap=%#llx heap_region_mib=%.0f alias=%#llx alias_region_mib=%.0f\n",
           (unsigned long long)(bytes / MiB), touch, (now_us() - t0) / 1000, (unsigned long long)heap, mib(info(5)),
           (unsigned long long)info(2), mib(info(3)));
    return heap;
}

static void print_samples(const char *kind, const double *v, int n) {
    printf("samples kind=%s n=%d values=", kind, n);
    for (int i = 0; i < n; i++) printf(i ? ",%.2f" : "%.2f", v[i]);
    printf("\n");
}

// ---- 4,500 pool-sized ranges that tile the heap: 64 KiB to 64 MiB in 64 KiB steps, sizes drawn log-uniform and
// scaled to the heap (so most are small and a few are large), deterministic for a seed.
static uint64_t rng_state;
static uint64_t rng(void) {   // splitmix64
    uint64_t z = (rng_state += 0x9e3779b97f4a7c15ull);
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ull;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebull;
    return z ^ (z >> 31);
}

static int make_pools(uint64_t bytes, uint64_t *off, uint64_t *size) {
    const uint64_t chunks = bytes / CHUNK, cap = WINDOW / CHUNK;
    int n = chunks < POOLS ? (int)chunks : POOLS;
    double *w = malloc(sizeof *w * (size_t)n), sum = 0;
    uint64_t *c = malloc(sizeof *c * (size_t)n), left = chunks - (uint64_t)n;
    if (!w || !c) die(1, "out of memory");
    for (int i = 0; i < n; i++) {
        double u = (double)(rng() >> 11) / 9007199254740992.0;
        w[i] = __builtin_exp(u * __builtin_log((double)cap));
        sum += w[i];
    }
    for (int i = 0; i < n; i++) {
        uint64_t extra = (uint64_t)((double)(chunks - (uint64_t)n) * w[i] / sum);
        if (extra > cap - 1) extra = cap - 1;
        if (extra > left) extra = left;
        c[i] = 1 + extra;
        left -= extra;
    }
    for (int i = 0; left; i = (i + 1) % n)
        if (c[i] < cap) c[i]++, left--;
    uint64_t at = 0, big = 0, small = 0;
    for (int i = 0; i < n; i++) {
        off[i] = at, size[i] = c[i] * CHUNK, at += size[i];
        big += c[i] == cap, small += c[i] == 1;
    }
    if (at != chunks * CHUNK) die(1, "the pools cover %#llx of %#llx bytes", (unsigned long long)at,
                                   (unsigned long long)(chunks * CHUNK));
    printf("pools n=%d covered_mib=%.0f of_64kib=%llu of_64mib=%llu\n", n, mib(at), (unsigned long long)small,
           (unsigned long long)big);
    free(w);
    free(c);
    return n;
}

static int cmp_u64(const void *a, const void *b) {
    uint64_t x = *(const uint64_t *)a, y = *(const uint64_t *)b;
    return x < y ? -1 : x > y;
}

static int mode_watch(int argc, char **argv) {
    if (argc < 6) return 2;
    const uint64_t bytes = strtoull(argv[2], NULL, 10) * MiB;
    const char *scope = argv[3];
    const uint64_t dirty = strtoull(argv[4], NULL, 10);
    const int iters = atoi(argv[5]);
    rng_state = argc > 6 ? strtoull(argv[6], NULL, 10) : 1;
    const int sc = !strcmp(scope, "heap") ? 0 : !strcmp(scope, "region") ? 1 : !strcmp(scope, "alias") ? 2
                 : !strcmp(scope, "pools") ? 3 : -1;
    if (sc < 0 || !bytes || bytes % (2 * MiB) || iters < 1 || dirty > bytes / PAGE) return 2;
    avail_check("start");
    printf("cell mode=watch commit_mib=%llu scope=%s dirty=%llu iters=%d avail_mib=%.0f\n",
           (unsigned long long)(bytes / MiB), scope, (unsigned long long)dirty, iters, mib(avail_bytes()));
    const uint64_t heap = heap_setup(bytes, 1);
    uint64_t *pages = malloc(MAX_REPORT * sizeof *pages);
    double *q = malloc(sizeof *q * (size_t)iters), *wr = malloc(sizeof *wr * (size_t)iters);
    if (!pages || !q || !wr) die(1, "out of memory");
    static uint64_t poff[POOLS], psize[POOLS];
    int npools = sc == 3 ? make_pools(bytes, poff, psize) : 0;
    const uint64_t qa = sc == 0 ? heap : sc == 1 ? info(4) : info(2), qn = sc == 0 ? bytes : sc == 1 ? info(5) : info(3);
    // The touches above wrote every page: one reset over the heap region first.
    uint64_t first = rt_mem_written(info(4), info(5), pages, MAX_REPORT, 1);
    if (first == UINT64_MAX) die(2, "no write watch on this platform");
    printf("reset first_pages=%llu\n", (unsigned long long)first);
    // The pages written: (start + i * stride) mod npages for i below dirty, distinct while the prime stride does not
    // divide npages.
    const uint64_t npages = bytes / PAGE, stride = 1000003;
    if (npages % stride == 0) die(1, "the page count shares a factor with the stride");
    const uint64_t expect = sc == 2 ? 0 : dirty;
    int bad = 0;
    for (int it = 0; it < iters; it++) {
        const uint64_t start = rng() % npages;
        double t0 = now_us();
        for (uint64_t i = 0; i < dirty; i++)
            *(volatile uint64_t *)(uintptr_t)(heap + ((start + i * stride) % npages) * PAGE + 64) = (uint64_t)it;
        double t1 = now_us();
        uint64_t got = 0;
        if (sc == 3) {
            uint64_t *at = pages;
            for (int p = 0; p < npools; p++) {
                uint64_t n = rt_mem_written(heap + poff[p], psize[p], at, psize[p] / PAGE, 1);
                if (n == UINT64_MAX) die(1, "query over pool %d failed", p);
                got += n, at += n;
            }
        } else {
            got = rt_mem_written(qa, qn, pages, MAX_REPORT, 1);
            if (got == UINT64_MAX) die(1, "query over %s failed", scope);
        }
        double t2 = now_us();
        wr[it] = t1 - t0, q[it] = t2 - t1;
        if (got != expect) {
            if (bad++ < 3) printf("mismatch iter=%d got=%llu expect=%llu\n", it, (unsigned long long)got,
                                  (unsigned long long)expect);
        } else if (it == 0 && got) {   // the reported pages are distinct heap pages
            qsort(pages, got, sizeof *pages, cmp_u64);
            for (uint64_t i = 0; i + 1 < got; i++)
                if (pages[i] >= pages[i + 1] || pages[i] < heap || pages[i] >= heap + bytes) bad++;
        }
        avail_check("iteration");
    }
    print_samples("query", q, iters);
    print_samples("write", wr, iters);
    printf("result mode=watch commit_mib=%llu scope=%s dirty=%llu iters=%d count_ok=%d min_avail_mib=%.0f "
           "below_1gib=%d\n", (unsigned long long)(bytes / MiB), scope, (unsigned long long)dirty, iters, !bad,
           mib(min_avail), min_avail < REPORT_BELOW);
    return bad ? 1 : 0;
}

// ---- Vulkan: one device with VK_EXT_external_memory_host and buffer device addresses
static VkInstance inst;
static VkPhysicalDevice gpu;
static VkDevice dev;
static VkQueue queue;
static uint32_t family = UINT32_MAX;
static VkDeviceSize import_align;

#define VK_CHECK(call)                                                                              \
    do {                                                                                            \
        VkResult r_ = (call);                                                                       \
        if (r_ != VK_SUCCESS) die(3, "%s failed: %d (%s:%d)", #call, (int)r_, __FILE__, __LINE__); \
    } while (0)

static void vulkan_init(void) {
    if (volkInitialize() != VK_SUCCESS) die(2, "no Vulkan loader");
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkwatchcost",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app};
    if (vkCreateInstance(&ici, NULL, &inst) != VK_SUCCESS) die(2, "no Vulkan instance");
    volkLoadInstance(inst);
    uint32_t n = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &n, NULL));
    if (!n) die(2, "no Vulkan device");
    VkPhysicalDevice *all = calloc(n, sizeof *all);
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &n, all));
    VkPhysicalDeviceProperties props = {0};
    for (int pass = 0; pass < 2 && !gpu; pass++)
        for (uint32_t i = 0; i < n && !gpu; i++) {
            vkGetPhysicalDeviceProperties(all[i], &props);
            if (pass || props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) gpu = all[i];
        }
    free(all);
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &ne, NULL);
    VkExtensionProperties *ext = calloc(ne ? ne : 1, sizeof *ext);
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &ne, ext);
    int has = 0;
    for (uint32_t i = 0; i < ne; i++) has |= !strcmp(ext[i].extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    free(ext);
    if (!has) die(2, "%s has no %s", props.deviceName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp = {
        .sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &hp};
    vkGetPhysicalDeviceProperties2(gpu, &p2);
    import_align = hp.minImportedHostPointerAlignment;
    VkPhysicalDeviceVulkan12Features f12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    VkPhysicalDeviceFeatures2 f2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &f12};
    vkGetPhysicalDeviceFeatures2(gpu, &f2);
    if (!f12.bufferDeviceAddress) die(2, "%s has no bufferDeviceAddress", props.deviceName);
    VkPhysicalDeviceVulkan12Features want12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES,
                                               .bufferDeviceAddress = VK_TRUE};
    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nq, NULL);
    VkQueueFamilyProperties *qf = calloc(nq ? nq : 1, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nq, qf);
    for (uint32_t i = 0; i < nq && family == UINT32_MAX; i++)
        if (qf[i].queueFlags & (VK_QUEUE_GRAPHICS_BIT | VK_QUEUE_COMPUTE_BIT)) family = i;
    free(qf);
    if (family == UINT32_MAX) die(2, "no graphics or compute queue");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
                                   .queueCount = 1, .pQueuePriorities = &prio};
    const char *dext[1] = {VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &want12, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = dext};
    VK_CHECK(vkCreateDevice(gpu, &dci, NULL, &dev));
    volkLoadDevice(dev);
    vkGetDeviceQueue(dev, family, 0, &queue);
    printf("device name=\"%s\" driver=%u.%u api=%u.%u.%u min_import_align=%llu\n", props.deviceName,
           props.driverVersion >> 22, (props.driverVersion >> 14) & 0xff, VK_API_VERSION_MAJOR(props.apiVersion),
           VK_API_VERSION_MINOR(props.apiVersion), VK_API_VERSION_PATCH(props.apiVersion),
           (unsigned long long)import_align);
}

// One import of [addr, addr+size): the host pointer's memory types, a buffer that can take external host memory, the
// import itself, the bind and the device address. us[0] the host pointer's properties, us[1] the buffer and its
// requirements, us[2] the import (vkAllocateMemory), us[3] bind and address, us[4] destroy and free (import_free).
typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    VkDeviceAddress addr;
    const char *failed;   // the step that failed, or NULL
    VkResult res;
    double us[5];
} Imp;

static int import_window(uint64_t addr, uint64_t size, Imp *o) {
    memset(o, 0, sizeof *o);
    double t0 = now_us();
    VkMemoryHostPointerPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    o->res = vkGetMemoryHostPointerPropertiesEXT(dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                                 (void *)(uintptr_t)addr, &hp);
    if (o->res != VK_SUCCESS) return o->failed = "host pointer properties", 0;
    double tp = now_us();
    VkExternalMemoryBufferCreateInfo ebi = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                            .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bci = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &ebi, .size = size,
                              .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                       VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
                              .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    o->res = vkCreateBuffer(dev, &bci, NULL, &o->buf);
    if (o->res != VK_SUCCESS) return o->failed = "create buffer", 0;
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, o->buf, &req);
    uint32_t bits = hp.memoryTypeBits & req.memoryTypeBits, type = 0;
    if (!bits) {
        vkDestroyBuffer(dev, o->buf, NULL);
        o->buf = VK_NULL_HANDLE, o->res = VK_ERROR_FEATURE_NOT_PRESENT;
        return o->failed = "no memory type for both", 0;
    }
    while (!(bits >> type & 1)) type++;
    double t1 = now_us();
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            .pHostPointer = (void *)(uintptr_t)addr};
    VkMemoryAllocateFlagsInfo fl = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = &imp,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo mai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fl, .allocationSize = size,
                                .memoryTypeIndex = type};
    o->res = vkAllocateMemory(dev, &mai, NULL, &o->mem);
    double t2 = now_us();
    if (o->res != VK_SUCCESS) {
        vkDestroyBuffer(dev, o->buf, NULL);
        o->buf = VK_NULL_HANDLE, o->mem = VK_NULL_HANDLE;
        o->us[0] = tp - t0, o->us[1] = t1 - tp, o->us[2] = t2 - t1;
        return o->failed = "import (vkAllocateMemory)", 0;
    }
    o->res = vkBindBufferMemory(dev, o->buf, o->mem, 0);
    if (o->res != VK_SUCCESS) return o->failed = "bind", 0;
    VkBufferDeviceAddressInfo dai = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = o->buf};
    o->addr = vkGetBufferDeviceAddress(dev, &dai);
    double t3 = now_us();
    o->us[0] = tp - t0, o->us[1] = t1 - tp, o->us[2] = t2 - t1, o->us[3] = t3 - t2;
    return o->addr != 0;
}

static void import_free(Imp *o) {
    double t0 = now_us();
    if (o->buf) vkDestroyBuffer(dev, o->buf, NULL);
    if (o->mem) vkFreeMemory(dev, o->mem, NULL);
    o->buf = VK_NULL_HANDLE, o->mem = VK_NULL_HANDLE;
    o->us[4] = now_us() - t0;
}

static int mode_import(int argc, char **argv) {
    if (argc < 4) return 2;
    const uint64_t bytes = strtoull(argv[2], NULL, 10) * MiB, win = (argc > 4 ? strtoull(argv[4], NULL, 10) : 64) * MiB;
    const int iters = atoi(argv[3]);
    const char *place = argc > 5 ? argv[5] : "cycle";
    const int pl = !strcmp(place, "cycle") ? 0 : !strcmp(place, "first") ? 1 : !strcmp(place, "last") ? 2
                 : !strcmp(place, "bounded") ? 3 : -1;
    if (!win || win % PAGE || win > bytes || bytes % (2 * MiB) || iters < 1 || pl < 0 || (pl == 3 && win == bytes))
        return 2;
    avail_check("start");
    printf("cell mode=import commit_mib=%llu window_mib=%llu place=%s iters=%d avail_mib=%.0f\n",
           (unsigned long long)(bytes / MiB), (unsigned long long)(win / MiB), place, iters, mib(avail_bytes()));
    const uint64_t heap = heap_setup(bytes, 1);
    vulkan_init();
    const uint64_t windows = bytes / win;
    double *v[6];
    for (int k = 0; k < 6; k++)
        if (!(v[k] = malloc(sizeof(double) * (size_t)iters))) die(1, "out of memory");
    for (int it = 0; it < iters; it++) {
        Imp o;
        uint64_t at = heap + (pl == 0 ? (uint64_t)it % windows : pl == 2 ? windows - 1 : 0) * win;
        DWORD old;
        if (pl == 3 && !VirtualProtect((void *)(uintptr_t)(at + win), PAGE, PAGE_READONLY, &old))
            die(1, "VirtualProtect of the page after the window failed");
        int ok = import_window(at, win, &o);
        if (pl == 3 && !VirtualProtect((void *)(uintptr_t)(at + win), PAGE, PAGE_READWRITE, &old))
            die(1, "VirtualProtect back failed");
        if (!ok)
            die(1, "importing %#llx failed at %s: %d", (unsigned long long)at, o.failed ? o.failed : "address",
                (int)o.res);
        import_free(&o);
        v[5][it] = 0;
        for (int k = 0; k < 5; k++) v[k][it] = o.us[k], v[5][it] += o.us[k];
        avail_check("iteration");
    }
    print_samples("import_host_properties", v[0], iters);
    print_samples("import_buffer", v[1], iters);
    print_samples("import_allocate", v[2], iters);
    print_samples("import_bind_address", v[3], iters);
    print_samples("import_destroy_free", v[4], iters);
    print_samples("import_total", v[5], iters);
    printf("result mode=import commit_mib=%llu window_mib=%llu place=%s iters=%d windows=%llu first_pass=%llu min_avail_mib=%.0f below_1gib=%d\n",
           (unsigned long long)(bytes / MiB), (unsigned long long)(win / MiB), place, iters, (unsigned long long)windows,
           (unsigned long long)(windows < (uint64_t)iters ? windows : (uint64_t)iters), mib(min_avail),
           min_avail < REPORT_BELOW);
    vkDestroyDevice(dev, NULL);
    return 0;
}

// Resident bytes of [a, a+n) page by page (QueryWorkingSetEx), whatever the runtime's commit map says.
static uint64_t resident(uint64_t a, uint64_t n) {
    enum { BATCH = 4096 };
    static PSAPI_WORKING_SET_EX_INFORMATION q[BATCH];
    uint64_t pages = n / PAGE, done = 0, res = 0;
    while (done < pages) {
        DWORD k = pages - done < BATCH ? (DWORD)(pages - done) : BATCH;
        for (DWORD i = 0; i < k; i++) q[i].VirtualAddress = (void *)(uintptr_t)(a + (done + i) * PAGE);
        if (!K32QueryWorkingSetEx(GetCurrentProcess(), q, k * (DWORD)sizeof q[0])) return UINT64_MAX;
        for (DWORD i = 0; i < k; i++) res += q[i].VirtualAttributes.Valid;
        done += k;
    }
    return res * PAGE;
}

static void hold_report(const char *stage, uint64_t heap, uint64_t bytes, uint64_t imported) {
    PROCESS_MEMORY_COUNTERS_EX c = {.cb = sizeof c};
    K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *)&c, sizeof c);
    uint64_t a = avail_check(stage);
    printf("memory stage=%s avail_mib=%.0f private_mib=%.1f ws_mib=%.1f imported_resident_mib=%.1f "
           "other_resident_mib=%.1f\n", stage, mib(a), mib(c.PrivateUsage), mib(c.WorkingSetSize),
           mib(imported ? resident(heap, imported) : 0), mib(resident(heap + imported, bytes - imported)));
}

static void trim(void) {
    SetProcessWorkingSetSize(GetCurrentProcess(), (SIZE_T)-1, (SIZE_T)-1);
    Sleep(2000);
}

static int mode_hold(int argc, char **argv) {
    if (argc < 5) return 2;
    const uint64_t bytes = strtoull(argv[2], NULL, 10) * MiB, imp = strtoull(argv[3], NULL, 10) * MiB;
    const int touch = !strcmp(argv[4], "touch");
    if (!imp || imp % WINDOW || imp > bytes || bytes % (2 * MiB) || (!touch && strcmp(argv[4], "notouch"))) return 2;
    avail_check("start");
    printf("cell mode=hold commit_mib=%llu import_mib=%llu touch=%d avail_mib=%.0f\n", (unsigned long long)(bytes / MiB),
           (unsigned long long)(imp / MiB), touch, mib(avail_bytes()));
    const uint64_t heap = heap_setup(bytes, touch);
    hold_report("heap", heap, bytes, imp);
    vulkan_init();
    hold_report("device", heap, bytes, imp);
    const uint64_t n = imp / WINDOW;
    Imp *o = calloc(n, sizeof *o);
    double t0 = now_us();
    for (uint64_t i = 0; i < n; i++)
        if (!import_window(heap + i * WINDOW, WINDOW, &o[i])) die(1, "import %llu failed at %s: %d",
                                                                  (unsigned long long)i, o[i].failed, (int)o[i].res);
    printf("imported windows=%llu ms=%.1f\n", (unsigned long long)n, (now_us() - t0) / 1000);
    hold_report("imported", heap, bytes, imp);
    trim();
    hold_report("imported_trimmed", heap, bytes, imp);
    t0 = now_us();
    for (uint64_t i = 0; i < n; i++) import_free(&o[i]);
    printf("freed windows=%llu ms=%.1f\n", (unsigned long long)n, (now_us() - t0) / 1000);
    hold_report("freed", heap, bytes, imp);
    trim();
    hold_report("freed_trimmed", heap, bytes, imp);
    printf("result mode=hold commit_mib=%llu import_mib=%llu touch=%d min_avail_mib=%.0f below_1gib=%d\n",
           (unsigned long long)(bytes / MiB), (unsigned long long)(imp / MiB), touch, mib(min_avail),
           min_avail < REPORT_BELOW);
    vkDestroyDevice(dev, NULL);
    return 0;
}

// ---- behaviours
static int verdict(const char *id, int pass, const char *what) {
    printf("verdict %s %s: %s\n", id, pass ? "PASS" : "FAIL", what);
    return pass;
}

static const char *page_state(uint64_t a) {
    MEMORY_BASIC_INFORMATION m;
    if (!VirtualQuery((void *)(uintptr_t)a, &m, sizeof m)) return "unknown";
    return m.State == MEM_COMMIT ? "committed" : m.State == MEM_RESERVE ? "reserved" : "free";
}

static VkCommandPool cpool;
static VkCommandBuffer cbuf;
static VkFence fence;

static void gpu_begin(void) {
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkResetCommandBuffer(cbuf, 0));
    VK_CHECK(vkBeginCommandBuffer(cbuf, &bi));
}

static void gpu_end_and_wait(void) {
    VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER, .srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                          .dstAccessMask = VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cbuf, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
    VK_CHECK(vkEndCommandBuffer(cbuf));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cbuf};
    VK_CHECK(vkResetFences(dev, 1, &fence));
    VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, UINT64_MAX));
}

static int mode_behave(int argc, char **argv) {
    const char *counter = argc > 2 ? argv[2] : "none";
    const int fill_hole = !strcmp(counter, "fill-hole"), hole = !strcmp(counter, "hole"),
              no_write = !strcmp(counter, "no-write");
    if (strcmp(counter, "none") && !fill_hole && !hole && !no_write) return 2;
    printf("cell mode=behave counter=%s avail_mib=%.0f\n", counter, mib(avail_check("start")));
    const uint64_t heap = heap_map(256 * MiB);   // mapped, nothing committed: each case commits what it states
    vulkan_init();
    VkCommandPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO,
                                   .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT, .queueFamilyIndex = family};
    VK_CHECK(vkCreateCommandPool(dev, &pci, NULL, &cpool));
    VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cpool,
                                       .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &cbuf));
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(dev, &fci, NULL, &fence));
    static uint64_t pages[WINDOW / PAGE];
    const uint64_t maxp = WINDOW / PAGE;
    int all = 1;
    Imp o;

    // B2: a window committed whole (counter: with a 64 KiB chunk left out); the CPU writes 1 MiB, the GPU copies it to
    // +8 MiB and fills 64 KiB at +16 MiB; the CPU must read both after the fence.
    const uint64_t w2 = heap + WINDOW;
    if (hole) {
        if (!rt_mem_commit_io(w2, 16 * MiB, 1) || !rt_mem_commit_io(w2 + 16 * MiB + CHUNK, WINDOW - 16 * MiB - CHUNK, 1))
            die(1, "B2: committing the window failed");
    } else if (!rt_mem_commit_io(w2, WINDOW, 1)) {
        die(1, "B2: committing the window failed");
    }
    for (uint32_t i = 0; i < MiB / 4; i++) ((volatile uint32_t *)(uintptr_t)w2)[i] = i * 2654435761u;
    int ok = import_window(w2, WINDOW, &o);
    printf("B2 import_ok=%d failed_at=\"%s\" result=%d device_address=%#llx\n", ok, o.failed ? o.failed : "",
           (int)o.res, (unsigned long long)o.addr);
    int same = 0;
    if (ok) {
        // B4's baseline: what the watch reports, query after query with reset every 10 ms, while nobody writes. All but
        // the first MiB of the window was committed and never touched before the import.
        if (rt_mem_written(w2, WINDOW, pages, maxp, 1) == UINT64_MAX) die(1, "no write watch over the window");
        uint64_t idle = 0, quiet = 0, polls = 0;
        double t0 = now_us(), last = t0;
        while (quiet < 20 && polls < 1000) {
            Sleep(10);
            uint64_t n = rt_mem_written(w2, WINDOW, pages, maxp, 1);
            polls++, idle += n, quiet = n ? 0 : quiet + 1;
            if (n) last = now_us();
        }
        printf("B4 untouched_pages=%llu written_with_nobody_writing=%llu last_seen_ms=%.0f polls=%llu\n",
               (unsigned long long)(WINDOW - MiB) / PAGE, (unsigned long long)idle, (last - t0) / 1000,
               (unsigned long long)polls);
        gpu_begin();
        VkBufferCopy cp = {.srcOffset = 0, .dstOffset = 8 * MiB, .size = MiB};
        vkCmdCopyBuffer(cbuf, o.buf, o.buf, 1, &cp);
        vkCmdFillBuffer(cbuf, o.buf, 16 * MiB, 64 * KiB, 0xA5A5A5A5u);
        gpu_end_and_wait();
        int copied = !memcmp((const void *)(uintptr_t)w2, (const void *)(uintptr_t)(w2 + 8 * MiB), MiB), filled = 1;
        for (uint32_t i = 0; i < 64 * KiB / 4; i++) filled &= ((volatile uint32_t *)(uintptr_t)(w2 + 16 * MiB))[i] == 0xA5A5A5A5u;
        same = copied && filled;
        // B4: the GPU wrote 256 + 16 pages of the window; the write watch was reset just before.
        uint64_t seen = rt_mem_written(w2, WINDOW, pages, maxp, 0);
        printf("B2 cpu_reads_gpu_copy=%d cpu_reads_gpu_fill=%d\n", copied, filled);
        printf("B4 gpu_written_pages=%llu seen_by_write_watch=%llu\n", (unsigned long long)(MiB / PAGE + 64 * KiB / PAGE),
               (unsigned long long)seen);
    }
    all &= verdict("B2", ok && same, "an import over committed pages succeeds, and the GPU and the CPU see the same bytes");

    // B3: CPU writes into the imported window, 37 pages, after a reset of its watch.
    int exact = 0;
    if (ok) {
        rt_mem_written(w2, WINDOW, pages, maxp, 1);
        if (!no_write)
            for (uint64_t k = 0; k < 37; k++) *(volatile uint64_t *)(uintptr_t)(w2 + (100 + 7 * k) * PAGE + 8) = k + 1;
        uint64_t n = rt_mem_written(w2, WINDOW, pages, maxp, 0);
        exact = n == 37;
        for (uint64_t k = 0; exact && k < 37; k++) exact = pages[k] == w2 + (100 + 7 * k) * PAGE;
        printf("B3 cpu_written_pages=%d seen_by_write_watch=%llu exact=%d\n", no_write ? 0 : 37, (unsigned long long)n,
               exact);
    }
    import_free(&o);
    all &= verdict("B3", ok && exact, "CPU writes into imported pages show in the write watch, page by page");

    // B1: a 64 MiB window committed but for one 64 KiB chunk in its middle.
    // It runs after B2 and B3: a failed import changes what later imports do (B7).
    const uint64_t w1 = heap, gap = w1 + 32 * MiB;
    if (!rt_mem_commit_io(w1, 32 * MiB, 1) || !rt_mem_commit_io(gap + CHUNK, WINDOW - 32 * MiB - CHUNK, 1))
        die(1, "B1: committing the window failed");
    if (fill_hole && !rt_mem_commit_io(gap, CHUNK, 1)) die(1, "B1: committing the chunk failed");
    printf("B1 chunk=%s rest=%s\n", page_state(gap), page_state(gap + CHUNK));
    ok = import_window(w1, WINDOW, &o);
    printf("B1 import_ok=%d failed_at=\"%s\" result=%d\n", ok, o.failed ? o.failed : "", (int)o.res);
    import_free(&o);
    all &= verdict("B1", !ok, "an import over a 64 MiB window with one uncommitted 64 KiB chunk fails");

    // B7: after B1's failure, the B2 window again (committed whole, 64 MiB), then 4 KiB of it, then 64 MiB again.
    if (!hole) {
        Imp a, b, c;
        int ra = import_window(w2, WINDOW, &a);
        import_free(&a);
        int rb = import_window(w2, PAGE, &b);
        import_free(&b);
        int rc = import_window(w2, WINDOW, &c);
        import_free(&c);
        printf("B7 after_failure_64mib=%d (%d) then_4kib=%d (%d) then_64mib=%d (%d)\n", ra, (int)a.res, rb,
               (int)b.res, rc, (int)c.res);
    }

    // B5: 100 pages written, then a query with reset whose buffer holds 10: what stays marked?
    const uint64_t w5 = heap + 2 * WINDOW;
    if (!rt_mem_commit_io(w5, 4 * MiB, 1)) die(1, "B5: committing failed");
    rt_mem_written(w5, 4 * MiB, pages, maxp, 1);
    for (uint64_t k = 0; k < 100; k++) *(volatile uint64_t *)(uintptr_t)(w5 + 10 * k * PAGE) = k;
    uint64_t first = rt_mem_written(w5, 4 * MiB, pages, 10, 1), left = rt_mem_written(w5, 4 * MiB, pages, maxp, 0);
    printf("B5 written=100 buffer=10 returned=%llu still_marked=%llu\n", (unsigned long long)first,
           (unsigned long long)left);

    // B6: the system's own answer for a range that runs from the heap reservation into the alias reservation.
    ULONG_PTR cnt = 16;
    ULONG gran = 0;
    UINT r = GetWriteWatch(0, (void *)(uintptr_t)(info(4) + info(5) - PAGE), 2 * PAGE, (PVOID *)pages, &cnt, &gran);
    printf("B6 system_answer=%s runtime_answer=%s\n", r ? "refused" : "answered",
           rt_mem_written(info(4) + info(5) - PAGE, 2 * PAGE, pages, maxp, 0) == UINT64_MAX ? "UINT64_MAX" : "pages");

    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, cpool, NULL);
    vkDestroyDevice(dev, NULL);
    printf("result mode=behave counter=%s all_pass=%d min_avail_mib=%.0f\n", counter, all, mib(min_avail));
    return all ? 0 : 1;
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) goto usage;
    int r = !strcmp(argv[1], "watch") ? mode_watch(argc, argv)
          : !strcmp(argv[1], "import") ? mode_import(argc, argv)
          : !strcmp(argv[1], "hold") ? mode_hold(argc, argv)
          : !strcmp(argv[1], "behave") ? mode_behave(argc, argv) : 2;
    if (r != 2) return r;
usage:
    fprintf(stderr, "usage: vkwatchcost watch COMMIT_MIB heap|region|alias|pools DIRTY ITERS [SEED]\n"
                    "       vkwatchcost import COMMIT_MIB ITERS [WINDOW_MIB [cycle|first|last|bounded]]\n"
                    "       vkwatchcost hold COMMIT_MIB IMPORT_MIB touch|notouch\n"
                    "       vkwatchcost behave [none|fill-hole|hole|no-write]\n");
    return 2;
}
