// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkcomputecost: the measurement of spike S6 (architecture D21 and section 6.1),
// how long the driver takes to build compute pipelines from translated modules.
//
//   vkcomputecost JOB
//
// The job file is vkpipecost's format (tools/pipecost/vkpc_main.c) cut down to
// compute, one directive per line, paths last on their line:
//   validation off|on            cache internal|disabled     threads N
//   device SUBSTRING             out PATH                    param NAME VALUE
//   extension NAME               (a device extension the modules need; exit 2 if absent)
//   module PATH                  (a compute module, SPIR-V)
//   set                          bind BINDING TYPE COUNT     (appended to the last set)
//   global                       (a set 0 shared by every pipeline: its `bind` lines follow, as after `set`)
//   push BYTES                   (a compute push constant range of that size, for every pipeline)
//   compute MODULE SET           warmup MODULE SET           (indices into the lists above)
// TYPE is ubo, ssbo, tex (combined image sampler), texbuf, septex (sampled
// image), sampler, img (storage image) or imgbuf, as in vkpipecost. A set of at
// most maxPushDescriptors descriptors is a push-descriptor set (D13); the
// pipeline layout is that one set and no push constants, which is what the
// first translated compute modules declare. A job with a `global` set is the
// layout `uni`: the pipeline layout is the global set at set 0, the
// module's own set at set 1 and the `push` range, as the translator numbers it.
//
// Params: `pin32 1` requires subgroup size 32 on the stage, as oot-pc-recomp
// does for modules with subgroup operations; `pipeline_cache PATH` is the
// application's VkPipelineCache, read when the file exists and written at the
// end otherwise; `heartbeat_s N` prints a line on stderr every N seconds with
// the builds in flight (default 30, 0 off); `entry NAME` (default main).
//
// `warmup` pipelines are built first, one at a time on the main thread, so that
// the driver's and the compiler's start-up is not charged to the first measured
// item; their rows say `warmup`. `compute` pipelines are handed out in file
// order to `threads` workers. Each worker reads the module, creates it, builds
// the pipeline, then destroys both, so host memory holds one module per worker.
// A pipeline the driver refuses is a row with its VkResult, not an exit; a lost
// device ends the run.
//
// Rows (TSV): phase, what, index, ms, worker, extra. `ms` is the wall time of
// vkCreateComputePipelines; extra holds `result=` (VkResult), `cpu_ms=` (CPU
// time of the building thread), `module_ms=` (vkCreateShaderModule) and
// `bytes=`. A summary goes to stdout as one JSON line.
//
// Exit codes: 0 measured, 1 the job file is wrong or validation reported
// errors, 2 Vulkan lacks something the job needs, 3 a Vulkan call failed or
// the device was lost.
//
// Time, memory, threads, output and the device's feature copy are vkpipecost's
// (spike S3), copied rather than shared so that S3's tool stays as it was when
// it measured. Windows only. No game data: modules are inputs, written by the
// caller (tools/computecost/compute_cost.py).

#define VK_NO_PROTOTYPES
#include "volk.h"

#include <stdarg.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define MAX_PATH_LEN 1024
#define MAX_BINDS 96
#define MAX_WORKERS 64
#define MAX_EXTS 16

typedef struct {
    int validation, internal_cache, threads;
    char device_filter[256];
    char out[MAX_PATH_LEN];
} Options;

static Options opt = {.validation = 0, .internal_cache = 1, .threads = 1};
static VkInstance inst;
static VkPhysicalDevice gpu;
static VkDevice dev;
static VkPhysicalDeviceProperties gpu_props;
static uint32_t max_push_descriptors, subgroup_size, pin32_stages;
static unsigned val_errors, val_warnings;
static VkDebugUtilsMessengerEXT messenger;

__attribute__((noreturn, format(printf, 2, 3))) static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "vkcomputecost: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stdout);
    exit(code);
}

#define VK_CHECK(call)                                                                   \
    do {                                                                                 \
        VkResult r_ = (call);                                                            \
        if (r_ != VK_SUCCESS) die(3, "%s failed: %d (%s:%d)", #call, (int)r_, __FILE__, __LINE__); \
    } while (0)

// ---- time, memory, threads (vkpipecost's) ----------------------------------------

static double qpc_per_ms, cycles_per_ms;

static double now_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / qpc_per_ms;
}

static double thread_cpu_ms(void) {
    ULONG64 c = 0;
    QueryThreadCycleTime(GetCurrentThread(), &c);
    return (double)c / cycles_per_ms;
}

static void clocks_init(void) {
    LARGE_INTEGER f;
    QueryPerformanceFrequency(&f);
    qpc_per_ms = (double)f.QuadPart / 1000.0;
    // Thread cycles per millisecond, from a 50 ms spin against the wall clock.
    ULONG64 c0 = 0, c1 = 0;
    QueryThreadCycleTime(GetCurrentThread(), &c0);
    double t0 = now_ms();
    volatile unsigned x = 0;
    while (now_ms() - t0 < 50.0) x++;
    QueryThreadCycleTime(GetCurrentThread(), &c1);
    cycles_per_ms = (double)(c1 - c0) / (now_ms() - t0);
}

typedef struct {
    DWORD cb;
    DWORD PageFaultCount;
    SIZE_T PeakWorkingSetSize, WorkingSetSize, QuotaPeakPagedPoolUsage, QuotaPagedPoolUsage,
        QuotaPeakNonPagedPoolUsage, QuotaNonPagedPoolUsage, PagefileUsage, PeakPagefileUsage, PrivateUsage;
} MemCounters;
BOOL WINAPI K32GetProcessMemoryInfo(HANDLE, MemCounters *, DWORD);

static void memory_mb(double *ws, double *priv, double *peak) {
    MemCounters m = {.cb = sizeof m};
    K32GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof m);
    if (ws) *ws = (double)m.WorkingSetSize / 1048576.0;
    if (priv) *priv = (double)m.PrivateUsage / 1048576.0;
    if (peak) *peak = (double)m.PeakWorkingSetSize / 1048576.0;
}

static double available_mb(void) {
    MEMORYSTATUSEX s = {.dwLength = sizeof s};
    GlobalMemoryStatusEx(&s);
    return (double)s.ullAvailPhys / 1048576.0;
}

typedef void (*WorkFn)(void *ctx, uint32_t item, uint32_t worker);

typedef struct {
    WorkFn fn;
    void *ctx;
    uint32_t n;
    volatile LONG next;
} Pool;

typedef struct {
    Pool *pool;
    uint32_t worker;
} PoolArg;

static DWORD WINAPI pool_main(LPVOID p) {
    PoolArg *a = p;
    for (;;) {
        LONG i = InterlockedIncrement(&a->pool->next) - 1;
        if ((uint32_t)i >= a->pool->n) break;
        a->pool->fn(a->pool->ctx, (uint32_t)i, a->worker);
    }
    return 0;
}

// Runs fn for items 0..n-1 on `threads` workers pulling from a shared counter;
// returns the wall time in milliseconds.
static double run_parallel(uint32_t n, int threads, WorkFn fn, void *ctx) {
    if (threads < 1) threads = 1;
    if (threads > MAX_WORKERS) threads = MAX_WORKERS;
    Pool pool = {fn, ctx, n, 0};
    PoolArg args[MAX_WORKERS];
    HANDLE h[MAX_WORKERS];
    double t0 = now_ms();
    if (threads == 1) {
        args[0] = (PoolArg){&pool, 0};
        pool_main(&args[0]);
        return now_ms() - t0;
    }
    for (int i = 0; i < threads; i++) {
        args[i] = (PoolArg){&pool, (uint32_t)i};
        h[i] = CreateThread(NULL, 4 << 20, pool_main, &args[i], 0, NULL);
        if (!h[i]) die(3, "CreateThread failed");
    }
    WaitForMultipleObjects((DWORD)threads, h, TRUE, INFINITE);
    for (int i = 0; i < threads; i++) CloseHandle(h[i]);
    return now_ms() - t0;
}

// ---- output (vkpipecost's) ----------------------------------------------------------

static FILE *out_file;
static CRITICAL_SECTION out_lock;

static void out_open(void) {
    InitializeCriticalSection(&out_lock);
    if (!opt.out[0]) return;
    out_file = fopen(opt.out, "wb");
    if (!out_file) die(1, "cannot write %s", opt.out);
    fprintf(out_file, "phase\twhat\tindex\tms\tworker\textra\n");
    fflush(out_file);
}

// Each row is flushed: a run killed at its timeout keeps the rows it finished.
static void out_row(const char *phase, const char *what, long index, double ms, int worker, const char *extra) {
    if (!out_file) return;
    EnterCriticalSection(&out_lock);
    fprintf(out_file, "%s\t%s\t%ld\t%.4f\t%d\t%s\n", phase, what, index, ms, worker, extra ? extra : "");
    fflush(out_file);
    LeaveCriticalSection(&out_lock);
}

static void out_close(void) {
    if (out_file) fclose(out_file);
    out_file = NULL;
}

static char summary[16384];
static size_t summary_len;

static void summary_add(const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int n = vsnprintf(summary + summary_len, sizeof summary - summary_len, fmt, ap);
    va_end(ap);
    if (n > 0 && summary_len + (size_t)n < sizeof summary) summary_len += (size_t)n;
}

static void summary_num(const char *key, double value) { summary_add(", \"%s\": %.6g", key, value); }

static void summary_end(void) {
    double ws, priv, peak;
    memory_mb(&ws, &priv, &peak);
    uint32_t dv = gpu_props.driverVersion;
    char driver[32];
    snprintf(driver, sizeof driver, "%u.%02u", (dv >> 22) & 0x3FF, (dv >> 14) & 0xFF);
    summary_add(", \"device\": \"%s\", \"driver\": \"%s\", \"internal_cache\": %d, \"threads\": %d, "
                "\"validation\": %d, \"validation_errors\": %u, \"validation_warnings\": %u, "
                "\"subgroup_size\": %u, \"max_push_descriptors\": %u, "
                "\"working_set_mb\": %.1f, \"private_mb\": %.1f, \"peak_working_set_mb\": %.1f}",
                gpu_props.deviceName, driver, opt.internal_cache, opt.threads, opt.validation, val_errors,
                val_warnings, subgroup_size, max_push_descriptors, ws, priv, peak);
    printf("%s\n", summary);
    fflush(stdout);
}

// ---- the job file ---------------------------------------------------------------

typedef struct {
    uint32_t binding;
    VkDescriptorType type;
    uint32_t count;
} Bind;

typedef struct {
    Bind b[MAX_BINDS];
    uint32_t n;
} SetDesc;

typedef struct {
    char path[MAX_PATH_LEN];
} Module;

typedef struct {
    uint32_t module, set, warmup;
} Item;

typedef struct {
    Module *modules;
    uint32_t nmodules;
    SetDesc *sets;
    uint32_t nsets;
    SetDesc global;              // set 0 of the layout `uni`, shared by every pipeline
    int has_global;
    uint32_t push_bytes;         // the push constant range of every pipeline, 0 for none
    Item *items;
    uint32_t nitems;
    char exts[MAX_EXTS][VK_MAX_EXTENSION_NAME_SIZE];
    uint32_t nexts;
    char words[64][64];  // `param NAME VALUE` lines
    char values[64][MAX_PATH_LEN];
    uint32_t nparams;
} Job;

static char *rest_after(char *s, int fields) {
    for (int i = 0; i < fields; i++) {
        while (*s == ' ' || *s == '\t') s++;
        while (*s && *s != ' ' && *s != '\t') s++;
    }
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    _fseeki64(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    void *p = n > 0 ? malloc((size_t)n) : NULL;
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) {
        free(p);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (size_t)n;
    return p;
}

static VkDescriptorType parse_type(const char *s) {
    static const struct { const char *name; VkDescriptorType type; } T[] = {
        {"ubo", VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}, {"ssbo", VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
        {"tex", VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER}, {"texbuf", VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER},
        {"septex", VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE}, {"sampler", VK_DESCRIPTOR_TYPE_SAMPLER},
        {"img", VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}, {"imgbuf", VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER},
    };
    for (size_t i = 0; i < sizeof T / sizeof T[0]; i++)
        if (!strcmp(s, T[i].name)) return T[i].type;
    die(1, "unknown descriptor type %s", s);
}

#define GROW(arr, n, cap)                                                   \
    do {                                                                    \
        if ((n) == (cap)) {                                                 \
            (cap) = (cap) ? 2 * (cap) : 64;                                 \
            (arr) = realloc((arr), (cap) * sizeof *(arr));                  \
            if (!(arr)) die(3, "out of memory");                            \
        }                                                                   \
    } while (0)

static void job_read(const char *path, Job *job) {
    memset(job, 0, sizeof *job);
    FILE *f = fopen(path, "rb");
    if (!f) die(1, "cannot open the job file %s", path);
    char line[4096];
    unsigned lineno = 0;
    uint32_t cap_m = 0, cap_s = 0, cap_i = 0;
    SetDesc *current = NULL;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
        if (!n || line[0] == '#') continue;
        char w[64] = {0}, a[64] = {0}, b[64] = {0}, c[64] = {0};
        int k = sscanf(line, "%63s %63s %63s %63s", w, a, b, c);
        if (!strcmp(w, "validation") && k == 2) opt.validation = !strcmp(a, "on");
        else if (!strcmp(w, "cache") && k == 2) opt.internal_cache = strcmp(a, "disabled") != 0;
        else if (!strcmp(w, "threads") && k == 2) opt.threads = atoi(a);
        else if (!strcmp(w, "device") && k >= 2) snprintf(opt.device_filter, sizeof opt.device_filter, "%s", rest_after(line, 1));
        else if (!strcmp(w, "out") && k >= 2) snprintf(opt.out, sizeof opt.out, "%s", rest_after(line, 1));
        else if (!strcmp(w, "extension") && k == 2) {
            if (job->nexts == MAX_EXTS) die(1, "line %u: too many extensions", lineno);
            snprintf(job->exts[job->nexts++], VK_MAX_EXTENSION_NAME_SIZE, "%s", a);
        } else if (!strcmp(w, "module") && k >= 2) {
            GROW(job->modules, job->nmodules, cap_m);
            snprintf(job->modules[job->nmodules++].path, MAX_PATH_LEN, "%s", rest_after(line, 1));
        } else if (!strcmp(w, "global") && k == 1) {
            job->has_global = 1;
            current = &job->global;
            memset(current, 0, sizeof *current);
        } else if (!strcmp(w, "push") && k == 2) {
            job->push_bytes = (uint32_t)atoi(a);
        } else if (!strcmp(w, "set") && k == 1) {
            GROW(job->sets, job->nsets, cap_s);
            current = &job->sets[job->nsets++];
            memset(current, 0, sizeof *current);
        } else if (!strcmp(w, "bind") && k == 4) {
            if (!current) die(1, "line %u: bind before set", lineno);
            if (current->n == MAX_BINDS) die(1, "line %u: too many bindings", lineno);
            current->b[current->n++] = (Bind){(uint32_t)atoi(a), parse_type(b), (uint32_t)atoi(c)};
        } else if ((!strcmp(w, "compute") || !strcmp(w, "warmup")) && k == 3) {
            GROW(job->items, job->nitems, cap_i);
            job->items[job->nitems++] = (Item){(uint32_t)atoi(a), (uint32_t)atoi(b), w[0] == 'w'};
        } else if (!strcmp(w, "param") && k >= 3) {
            if (job->nparams == 64) die(1, "too many params");
            snprintf(job->words[job->nparams], 64, "%s", a);
            snprintf(job->values[job->nparams], MAX_PATH_LEN, "%s", rest_after(line, 2));
            job->nparams++;
        } else die(1, "line %u not understood: %s", lineno, line);
    }
    fclose(f);
    for (uint32_t i = 0; i < job->nitems; i++)
        if (job->items[i].module >= job->nmodules || job->items[i].set >= job->nsets)
            die(1, "pipeline %u refers past the lists", i);
}

static const char *job_param(const Job *job, const char *name, const char *fallback) {
    for (uint32_t i = 0; i < job->nparams; i++)
        if (!strcmp(job->words[i], name)) return job->values[i];
    return fallback;
}

static long job_param_int(const Job *job, const char *name, long fallback) {
    const char *v = job_param(job, name, NULL);
    return v ? strtol(v, NULL, 0) : fallback;
}

// ---- the device ------------------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT types,
                                                 const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)user;
    if (types == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) return VK_FALSE;  // the loader's own notices
    const char *kind = "info";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { val_errors++; kind = "error"; }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { val_warnings++; kind = "warning"; }
    if (val_errors + val_warnings <= 50)
        fprintf(stderr, "vkcomputecost: validation %s: %s\n", kind, data->pMessage ? data->pMessage : "(no text)");
    return VK_FALSE;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceShaderAtomicFloatFeaturesEXT af;
    VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT af2;
    VkPhysicalDevicePipelineBinaryFeaturesKHR bin;
    VkPhysicalDeviceMaintenance5FeaturesKHR m5;
} Features;

// The chain holds the extension blocks only when their extensions are enabled.
static void features_chain(Features *f, int atomic_float, int atomic_float2, int binaries) {
    memset(f, 0, sizeof *f);
    f->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f->af.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    f->af2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
    f->bin.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR;
    f->m5.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR;
    void **next = &f->f2.pNext;
    *next = &f->v11; next = &f->v11.pNext;
    *next = &f->v12; next = &f->v12.pNext;
    *next = &f->v13; next = &f->v13.pNext;
    if (atomic_float) { *next = &f->af; next = &f->af.pNext; }
    if (atomic_float2) { *next = &f->af2; next = &f->af2.pNext; }
    if (binaries) { *next = &f->bin; next = &f->bin.pNext; *next = &f->m5; }
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static void device_init(const Job *job, int pin32) {
    clocks_init();
    if (volkInitialize() != VK_SUCCESS) die(2, "no Vulkan loader");
    const char *layers[1] = {"VK_LAYER_KHRONOS_validation"};
    const char *iexts[1] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkDebugUtilsMessengerCreateInfoEXT mci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        .pfnUserCallback = on_message};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkcomputecost",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pNext = opt.validation ? &mci : NULL,
                                .pApplicationInfo = &app, .enabledLayerCount = opt.validation ? 1 : 0,
                                .ppEnabledLayerNames = layers, .enabledExtensionCount = opt.validation ? 1 : 0,
                                .ppEnabledExtensionNames = iexts};
    VK_CHECK(vkCreateInstance(&ici, NULL, &inst));
    volkLoadInstance(inst);
    if (opt.validation) VK_CHECK(vkCreateDebugUtilsMessengerEXT(inst, &mci, NULL, &messenger));

    uint32_t ngpu = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ngpu, NULL));
    if (!ngpu) die(2, "no Vulkan device");
    VkPhysicalDevice *gpus = calloc(ngpu, sizeof *gpus);
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ngpu, gpus));
    for (int pass = 0; pass < 2 && !gpu; pass++)
        for (uint32_t i = 0; i < ngpu && !gpu; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(gpus[i], &p);
            if (pass == 0 && p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            if (opt.device_filter[0] && !strstr(p.deviceName, opt.device_filter)) continue;
            gpu = gpus[i];
            gpu_props = p;
        }
    free(gpus);
    if (!gpu) die(2, "no device matches");

    VkPhysicalDevicePushDescriptorPropertiesKHR pd = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PUSH_DESCRIPTOR_PROPERTIES_KHR};
    VkPhysicalDeviceVulkan13Properties p13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES, .pNext = &pd};
    VkPhysicalDeviceVulkan11Properties p11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, .pNext = &p13};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p11};
    vkGetPhysicalDeviceProperties2(gpu, &p2);
    max_push_descriptors = pd.maxPushDescriptors;
    subgroup_size = p11.subgroupSize;
    pin32_stages = p13.requiredSubgroupSizeStages;

    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, NULL);
    VkExtensionProperties *dexts = calloc(next ? next : 1, sizeof *dexts);
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, dexts);
    const char *dext[MAX_EXTS + 4];
    uint32_t ndext = 0;
    if (!has_ext(dexts, next, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) die(2, "the device lacks %s", VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    dext[ndext++] = VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME;
    int atomic_float = 0, atomic_float2 = 0;
    for (uint32_t i = 0; i < job->nexts; i++) {
        if (!has_ext(dexts, next, job->exts[i])) die(2, "the device lacks %s", job->exts[i]);
        dext[ndext++] = job->exts[i];
        atomic_float |= !strcmp(job->exts[i], VK_EXT_SHADER_ATOMIC_FLOAT_EXTENSION_NAME);
        atomic_float2 |= !strcmp(job->exts[i], VK_EXT_SHADER_ATOMIC_FLOAT_2_EXTENSION_NAME);
    }
    int binaries = has_ext(dexts, next, VK_KHR_PIPELINE_BINARY_EXTENSION_NAME) &&
                   has_ext(dexts, next, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    if (binaries) {
        dext[ndext++] = VK_KHR_MAINTENANCE_5_EXTENSION_NAME;
        dext[ndext++] = VK_KHR_PIPELINE_BINARY_EXTENSION_NAME;
    }
    free(dexts);
    if (!opt.internal_cache && !binaries) die(2, "cache disabled needs VK_KHR_pipeline_binary");

    // Every feature the device offers, except robustness and protected memory,
    // which change what the compiler emits or what memory is (vkpipecost's rule).
    Features have, want;
    features_chain(&have, atomic_float, atomic_float2, binaries);
    vkGetPhysicalDeviceFeatures2(gpu, &have.f2);
    features_chain(&want, atomic_float, atomic_float2, binaries);
    want.f2.features = have.f2.features;
    want.f2.features.robustBufferAccess = VK_FALSE;
    memcpy((char *)&want.v11 + offsetof(VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess),
           (char *)&have.v11 + offsetof(VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess),
           sizeof have.v11 - offsetof(VkPhysicalDeviceVulkan11Features, storageBuffer16BitAccess));
    want.v11.protectedMemory = VK_FALSE;
    memcpy((char *)&want.v12 + offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge),
           (char *)&have.v12 + offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge),
           sizeof have.v12 - offsetof(VkPhysicalDeviceVulkan12Features, samplerMirrorClampToEdge));
    memcpy((char *)&want.v13 + offsetof(VkPhysicalDeviceVulkan13Features, robustImageAccess),
           (char *)&have.v13 + offsetof(VkPhysicalDeviceVulkan13Features, robustImageAccess),
           sizeof have.v13 - offsetof(VkPhysicalDeviceVulkan13Features, robustImageAccess));
    want.v13.robustImageAccess = VK_FALSE;
    memcpy((char *)&want.af + offsetof(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT, shaderBufferFloat32Atomics),
           (char *)&have.af + offsetof(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT, shaderBufferFloat32Atomics),
           sizeof have.af - offsetof(VkPhysicalDeviceShaderAtomicFloatFeaturesEXT, shaderBufferFloat32Atomics));
    memcpy((char *)&want.af2 + offsetof(VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT, shaderBufferFloat16Atomics),
           (char *)&have.af2 + offsetof(VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT, shaderBufferFloat16Atomics),
           sizeof have.af2 - offsetof(VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT, shaderBufferFloat16Atomics));
    want.bin.pipelineBinaries = have.bin.pipelineBinaries;
    want.m5.maintenance5 = have.m5.maintenance5;
    if (!want.f2.features.shaderInt64 || !want.v12.bufferDeviceAddress)
        die(2, "the device lacks shaderInt64 or bufferDeviceAddress");
    if (pin32 && (!want.v13.subgroupSizeControl || !(pin32_stages & VK_SHADER_STAGE_COMPUTE_BIT)))
        die(2, "pin32 needs subgroupSizeControl for the compute stage");

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, NULL);
    VkQueueFamilyProperties *qf = calloc(nqf, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, qf);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < nqf && family == UINT32_MAX; i++)
        if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) family = i;
    free(qf);
    if (family == UINT32_MAX) die(2, "no compute queue");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
                                   .queueCount = 1, .pQueuePriorities = &prio};
    VkDevicePipelineBinaryInternalCacheControlKHR cc = {
        .sType = VK_STRUCTURE_TYPE_DEVICE_PIPELINE_BINARY_INTERNAL_CACHE_CONTROL_KHR, .pNext = &want.f2,
        .disableInternalCache = VK_TRUE};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO,
                              .pNext = opt.internal_cache ? (void *)&want.f2 : (void *)&cc,
                              .queueCreateInfoCount = 1, .pQueueCreateInfos = &qci, .enabledExtensionCount = ndext,
                              .ppEnabledExtensionNames = dext};
    VK_CHECK(vkCreateDevice(gpu, &dci, NULL, &dev));
    volkLoadDevice(dev);
}

static void device_finish(void) {
    vkDeviceWaitIdle(dev);
    vkDestroyDevice(dev, NULL);
    if (messenger) vkDestroyDebugUtilsMessengerEXT(inst, messenger, NULL);
    vkDestroyInstance(inst, NULL);
}

// ---- the build -------------------------------------------------------------------

typedef struct {
    const Job *job;
    const Item *items;           // the measured pipelines, in hand-out order
    VkDescriptorSetLayout *set_layouts;
    VkDescriptorSetLayout global_layout;   // set 0 of the layout `uni`, when the job has one
    VkPipelineLayout *layouts;   // per set description
    VkPipelineCache cache;
    int pin32;
    const char *entry;
    const char *phase;
    volatile LONG done, failed;
    volatile LONG in_flight[MAX_WORKERS];  // item + 1 while a worker builds it, 0 otherwise
    double started[MAX_WORKERS];
    volatile LONG stop;
} BuildCtx;

static VkDescriptorSetLayout set_layout_make(const SetDesc *d, int allow_push) {
    VkDescriptorSetLayoutBinding lb[MAX_BINDS];
    uint32_t total = 0;
    for (uint32_t i = 0; i < d->n; i++) {
        lb[i] = (VkDescriptorSetLayoutBinding){d->b[i].binding, d->b[i].type, d->b[i].count, VK_SHADER_STAGE_COMPUTE_BIT, NULL};
        total += d->b[i].count;
    }
    VkDescriptorSetLayoutCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags = allow_push && total <= max_push_descriptors ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0,
        .bindingCount = d->n, .pBindings = lb};
    VkDescriptorSetLayout out;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &ci, NULL, &out));
    return out;
}

static void layouts_create(BuildCtx *b) {
    const Job *job = b->job;
    b->set_layouts = calloc(job->nsets ? job->nsets : 1, sizeof *b->set_layouts);
    b->layouts = calloc(job->nsets ? job->nsets : 1, sizeof *b->layouts);
    if (job->has_global) b->global_layout = set_layout_make(&job->global, 0);
    VkPushConstantRange push = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = job->push_bytes};
    for (uint32_t s = 0; s < job->nsets; s++) {
        b->set_layouts[s] = set_layout_make(&job->sets[s], 1);
        VkDescriptorSetLayout pair[2] = {b->global_layout, b->set_layouts[s]};
        VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                         .setLayoutCount = job->has_global ? 2 : 1,
                                         .pSetLayouts = job->has_global ? pair : &b->set_layouts[s],
                                         .pushConstantRangeCount = job->push_bytes ? 1 : 0, .pPushConstantRanges = &push};
        VK_CHECK(vkCreatePipelineLayout(dev, &li, NULL, &b->layouts[s]));
    }
}

// Builds one pipeline and writes its row; returns 0 on success, 1 when the
// module cannot be read or the driver refused it.
static int build_one(BuildCtx *b, const Item *it, long index, uint32_t worker, const char *what) {
    const Module *m = &b->job->modules[it->module];
    size_t size = 0;
    uint32_t *code = read_file(m->path, &size);
    char extra[256];
    if (!code || size % 4 || code[0] != 0x07230203u) {
        free(code);
        snprintf(extra, sizeof extra, "result=unreadable bytes=%zu", size);
        out_row(b->phase, what, index, 0, (int)worker, extra);
        return 1;
    }
    VkShaderModuleCreateInfo mci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    VkShaderModule sm = VK_NULL_HANDLE;
    double t0 = now_ms();
    VkResult r = vkCreateShaderModule(dev, &mci, NULL, &sm);
    double module_ms = now_ms() - t0;
    double ms = 0, cpu = 0;
    VkPipeline p = VK_NULL_HANDLE;
    if (r == VK_SUCCESS) {
        VkPipelineShaderStageRequiredSubgroupSizeCreateInfo rs = {
            .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = 32};
        VkComputePipelineCreateInfo ci = {
            .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
            .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .pNext = b->pin32 ? &rs : NULL,
                      .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = sm, .pName = b->entry},
            .layout = b->layouts[it->set]};
        double c0 = thread_cpu_ms();
        t0 = now_ms();
        r = vkCreateComputePipelines(dev, b->cache, 1, &ci, NULL, &p);
        ms = now_ms() - t0;
        cpu = thread_cpu_ms() - c0;
    }
    if (r == VK_ERROR_DEVICE_LOST) die(3, "device lost building pipeline %ld (%s)", index, m->path);
    snprintf(extra, sizeof extra, "result=%d cpu_ms=%.3f module_ms=%.3f bytes=%zu", (int)r, cpu, module_ms, size);
    out_row(b->phase, what, index, ms, (int)worker, extra);
    if (p) vkDestroyPipeline(dev, p, NULL);
    if (sm) vkDestroyShaderModule(dev, sm, NULL);
    free(code);
    return r != VK_SUCCESS;
}

static void w_build(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    b->started[w] = now_ms();
    InterlockedExchange(&b->in_flight[w], (LONG)i + 1);
    if (build_one(b, &b->items[i], (long)i, w, "pipeline")) InterlockedIncrement(&b->failed);
    InterlockedExchange(&b->in_flight[w], 0);
    InterlockedIncrement(&b->done);
}

typedef struct {
    BuildCtx *b;
    uint32_t n;
    double period_ms, t0;
    HANDLE wake;
} Heartbeat;

static DWORD WINAPI heartbeat_main(LPVOID p) {
    Heartbeat *h = p;
    while (WaitForSingleObject(h->wake, (DWORD)h->period_ms) == WAIT_TIMEOUT) {
        double ws;
        memory_mb(&ws, NULL, NULL);
        double now = now_ms();
        char line[2048];
        int at = snprintf(line, sizeof line, "vkcomputecost: %.0f s, %ld of %u done, working set %.0f MB, available %.0f MB, building",
                          (now - h->t0) / 1000.0, (long)h->b->done, h->n, ws, available_mb());
        for (int w = 0; w < MAX_WORKERS && at < (int)sizeof line - 32; w++) {
            LONG item = h->b->in_flight[w];
            if (item) at += snprintf(line + at, sizeof line - (size_t)at, " %ld (%.0f s)", (long)item - 1,
                                     (now - h->b->started[w]) / 1000.0);
        }
        fprintf(stderr, "%s\n", line);
        fflush(stderr);
    }
    return 0;
}

static int run(Job *job) {
    int pin32 = job_param_int(job, "pin32", 0) != 0;
    device_init(job, pin32);
    out_open();
    summary_len = 0;
    summary_add("{\"mode\": \"compute\"");
    BuildCtx b = {.job = job, .pin32 = pin32, .entry = job_param(job, "entry", "main"), .phase = "compute"};
    layouts_create(&b);

    const char *pc_path = job_param(job, "pipeline_cache", NULL);
    int pc_write = 0;
    if (pc_path) {
        size_t size = 0;
        void *data = read_file(pc_path, &size);
        pc_write = !data;
        double t0 = now_ms();
        VkPipelineCacheCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO,
                                        .initialDataSize = data ? size : 0, .pInitialData = data};
        VK_CHECK(vkCreatePipelineCache(dev, &ci, NULL, &b.cache));
        summary_num("pipeline_cache_in_bytes", data ? (double)size : 0);
        summary_num("pipeline_cache_create_ms", now_ms() - t0);
        free(data);
    }

    // Warm-up first, alone; then the measured pipelines in file order.
    Item *measured = calloc(job->nitems ? job->nitems : 1, sizeof *measured);
    uint32_t nm = 0, nw = 0, wfail = 0;
    for (uint32_t i = 0; i < job->nitems; i++) {
        if (job->items[i].warmup) wfail += (uint32_t)build_one(&b, &job->items[i], (long)nw++, 0, "warmup");
        else measured[nm++] = job->items[i];
    }
    b.items = measured;
    summary_num("warmups", nw);
    summary_num("warmup_failed", wfail);

    double ws0;
    memory_mb(&ws0, NULL, NULL);
    double avail0 = available_mb();
    Heartbeat hb = {.b = &b, .n = nm, .period_ms = 1000.0 * (double)job_param_int(job, "heartbeat_s", 30), .t0 = now_ms(),
                    .wake = CreateEventA(NULL, TRUE, FALSE, NULL)};
    HANDLE ht = hb.period_ms > 0 ? CreateThread(NULL, 1 << 20, heartbeat_main, &hb, 0, NULL) : NULL;
    double wall = run_parallel(nm, opt.threads, w_build, &b);
    SetEvent(hb.wake);
    if (ht) {
        WaitForSingleObject(ht, INFINITE);
        CloseHandle(ht);
    }
    CloseHandle(hb.wake);
    summary_num("pipelines", nm);
    summary_num("failed", b.failed);
    summary_num("wall_ms", wall);
    summary_num("available_mb_start", avail0);
    summary_num("available_mb_end", available_mb());
    summary_num("working_set_start_mb", ws0);
    summary_num("pin32", pin32);

    if (b.cache) {
        if (pc_write) {
            size_t size = 0;
            VK_CHECK(vkGetPipelineCacheData(dev, b.cache, &size, NULL));
            void *data = malloc(size ? size : 1);
            VK_CHECK(vkGetPipelineCacheData(dev, b.cache, &size, data));
            FILE *f = fopen(pc_path, "wb");
            if (!f || fwrite(data, 1, size, f) != size) die(1, "cannot write %s", pc_path);
            fclose(f);
            free(data);
            summary_num("pipeline_cache_out_bytes", (double)size);
        }
        vkDestroyPipelineCache(dev, b.cache, NULL);
    }
    for (uint32_t s = 0; s < job->nsets; s++) {
        vkDestroyPipelineLayout(dev, b.layouts[s], NULL);
        vkDestroyDescriptorSetLayout(dev, b.set_layouts[s], NULL);
    }
    free(measured);
    summary_end();
    out_close();
    device_finish();
    return val_errors ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc != 2) die(1, "usage: vkcomputecost JOB");
    static Job job;
    job_read(argv[1], &job);
    return run(&job);
}
