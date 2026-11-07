// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkmeshcost: the measurement of spike S4 (architecture D16, D21 and section 6.1), how long the driver takes
// to build the game's translated task and mesh pipelines cold.
//
//   vkmeshcost JOB
//
// The job file is vkcomputecost's format (tools/computecost/vkcomputecost.c) with the graphics pipeline libraries of
// vkpipecost (tools/pipecost/vkpc_build.c) and the layout `uni` of vkmesh (tools/firstdraw/vkmesh.c), one directive per
// line, paths last on their line:
//   validation off|on|sync       cache internal|disabled      threads N
//   device SUBSTRING             out PATH                     param NAME VALUE
//   extension NAME               (a device extension the modules need; exit 2 if absent)
//   module t|m|f PATH            (a task, mesh or fragment module, SPIR-V)
//   pre TASK MESH                (a pre-rasterization library: TASK is a module index, or - for a mesh stage alone)
//   frag MODULE                  (a fragment shader library)
//   link PRE FRAG TARGETS        (fast-link pre item PRE, frag item FRAG and the fragment output library of TARGETS
//                                 colour attachments, 1 to 8; PRE and FRAG index the `pre` and `frag` lines)
//   warmup TASK MESH             (a pre-rasterization library built first, alone, and thrown away)
//
// Params: `phases pre,frag,link` (the order the phases run, default pre; `link` needs `pre` and `frag` in the same run
// because it links the libraries they built); `pin32 1|0` requires subgroup size 32 with full subgroups on the task and
// mesh stages (default 1, as research 13 recommendation 5 asks); `format NAME` the colour format of the output libraries
// (rgba8, rgba16f, r11g11b10f; default rgba16f); `pipeline_cache PATH` the application's VkPipelineCache, read when the
// file exists and written at the end otherwise; `heartbeat_s N` a line on stderr every N seconds with the builds in
// flight (default 30, 0 off); `counter nopush` leaves the push constants out of the layout, which validation must catch.
//
// Every pipeline is built the way D12 says: viewport, scissor, culling, depth, stencil and blend state dynamic, the
// libraries created with the library and retain-link-time-optimisation flags as spike S3 created them. The layout is
// `uni` (xlate/host/UniLayout.cs): set 0 the global update-after-bind arrays, set 1 the push descriptors of the mesh and
// fragment stages, set 2 the task stage's descriptors, 128 bytes of push constants, one layout for every pipeline.
//
// Rows (TSV): phase, what, index, ms, worker, extra. `ms` is the wall time of vkCreateGraphicsPipelines; extra holds
// `result=` (VkResult), `cpu_ms=` (CPU time of the building thread), `module_ms=` (vkCreateShaderModule, all modules of
// the item) and `bytes=` (SPIR-V read). A pipeline the driver refuses is a row with its VkResult, not an exit; a lost
// device ends the run. A summary goes to stdout as one JSON line.
//
// Exit codes: 0 measured, 1 the job file is wrong or validation reported errors, 2 Vulkan lacks something the job needs,
// 3 a Vulkan call failed or the device was lost.
//
// Time, memory, threads, output and the device's feature copy are vkcomputecost's, copied rather than shared so that
// the earlier spikes' tools stay as they were when they measured. Windows only. No game data: the modules are inputs,
// written by the caller (tools/meshcost/mesh_cost.py).

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
#define MAX_WORKERS 64
#define MAX_EXTS 16
#define PUSH_BYTES 128
#define GLOBAL_IMAGES 33025
#define GLOBAL_SAMPLERS 4096
#define MESH_STAGES (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT)
#define PHASE_PRE 1
#define PHASE_FRAG 2
#define PHASE_LINK 4

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
static uint32_t subgroup_size, pin32_stages;
static unsigned val_errors, val_warnings;
static VkDebugUtilsMessengerEXT messenger;

__attribute__((noreturn, format(printf, 2, 3))) static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "vkmeshcost: ");
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

// ---- time, memory, threads (vkcomputecost's) ---------------------------------------

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

// Runs fn for items 0..n-1 on `threads` workers pulling from a shared counter; returns the wall time in milliseconds.
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

// ---- output (vkcomputecost's) ---------------------------------------------------------

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

static void summary_num(const char *key, double value) { summary_add(", \"%s\": %.12g", key, value); }

static void summary_end(void) {
    double ws, priv, peak;
    memory_mb(&ws, &priv, &peak);
    uint32_t dv = gpu_props.driverVersion;
    char driver[32];
    snprintf(driver, sizeof driver, "%u.%02u", (dv >> 22) & 0x3FF, (dv >> 14) & 0xFF);
    summary_add(", \"device\": \"%s\", \"driver\": \"%s\", \"internal_cache\": %d, \"threads\": %d, "
                "\"validation\": %d, \"validation_errors\": %u, \"validation_warnings\": %u, "
                "\"subgroup_size\": %u, \"working_set_mb\": %.1f, \"private_mb\": %.1f, \"peak_working_set_mb\": %.1f}",
                gpu_props.deviceName, driver, opt.internal_cache, opt.threads, opt.validation, val_errors,
                val_warnings, subgroup_size, ws, priv, peak);
    printf("%s\n", summary);
    fflush(stdout);
}

// ---- the job file -----------------------------------------------------------------------

typedef struct {
    char path[MAX_PATH_LEN];
} Module;

typedef struct {
    int task, mesh;   // module indices; task is -1 for a mesh stage alone
} PreItem;

typedef struct {
    uint32_t pre, frag, targets;
} LinkItem;

typedef struct {
    Module *modules;
    uint32_t nmodules;
    PreItem *pre, *warm;
    uint32_t npre, nwarm;
    int *frag;        // module indices
    uint32_t nfrag;
    LinkItem *link;
    uint32_t nlink;
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

#define GROW(arr, n, cap)                                                   \
    do {                                                                    \
        if ((n) == (cap)) {                                                 \
            (cap) = (cap) ? 2 * (cap) : 64;                                 \
            (arr) = realloc((arr), (cap) * sizeof *(arr));                  \
            if (!(arr)) die(3, "out of memory");                            \
        }                                                                   \
    } while (0)

static int module_ref(const char *s) { return !strcmp(s, "-") ? -1 : atoi(s); }

static void job_read(const char *path, Job *job) {
    memset(job, 0, sizeof *job);
    FILE *f = fopen(path, "rb");
    if (!f) die(1, "cannot open the job file %s", path);
    char line[4096];
    unsigned lineno = 0;
    uint32_t cap_m = 0, cap_p = 0, cap_w = 0, cap_f = 0, cap_l = 0;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
        if (!n || line[0] == '#') continue;
        char w[64] = {0}, a[64] = {0}, b[64] = {0}, c[64] = {0};
        int k = sscanf(line, "%63s %63s %63s %63s", w, a, b, c);
        if (!strcmp(w, "validation") && k == 2) opt.validation = !strcmp(a, "on") ? 1 : !strcmp(a, "sync") ? 2 : 0;
        else if (!strcmp(w, "cache") && k == 2) opt.internal_cache = strcmp(a, "disabled") != 0;
        else if (!strcmp(w, "threads") && k == 2) opt.threads = atoi(a);
        else if (!strcmp(w, "device") && k >= 2) snprintf(opt.device_filter, sizeof opt.device_filter, "%s", rest_after(line, 1));
        else if (!strcmp(w, "out") && k >= 2) snprintf(opt.out, sizeof opt.out, "%s", rest_after(line, 1));
        else if (!strcmp(w, "extension") && k == 2) {
            if (job->nexts == MAX_EXTS) die(1, "line %u: too many extensions", lineno);
            snprintf(job->exts[job->nexts++], VK_MAX_EXTENSION_NAME_SIZE, "%s", a);
        } else if (!strcmp(w, "module") && k >= 3) {
            if (strcmp(a, "t") && strcmp(a, "m") && strcmp(a, "f")) die(1, "line %u: a module is t, m or f", lineno);
            GROW(job->modules, job->nmodules, cap_m);
            snprintf(job->modules[job->nmodules++].path, MAX_PATH_LEN, "%s", rest_after(line, 2));
        } else if (!strcmp(w, "pre") && k == 3) {
            GROW(job->pre, job->npre, cap_p);
            job->pre[job->npre++] = (PreItem){module_ref(a), module_ref(b)};
        } else if (!strcmp(w, "warmup") && k == 3) {
            GROW(job->warm, job->nwarm, cap_w);
            job->warm[job->nwarm++] = (PreItem){module_ref(a), module_ref(b)};
        } else if (!strcmp(w, "frag") && k == 2) {
            GROW(job->frag, job->nfrag, cap_f);
            job->frag[job->nfrag++] = atoi(a);
        } else if (!strcmp(w, "link") && k == 4) {
            GROW(job->link, job->nlink, cap_l);
            job->link[job->nlink++] = (LinkItem){(uint32_t)atoi(a), (uint32_t)atoi(b), (uint32_t)atoi(c)};
        } else if (!strcmp(w, "param") && k >= 3) {
            if (job->nparams == 64) die(1, "too many params");
            snprintf(job->words[job->nparams], 64, "%s", a);
            snprintf(job->values[job->nparams], MAX_PATH_LEN, "%s", rest_after(line, 2));
            job->nparams++;
        } else die(1, "line %u not understood: %s", lineno, line);
    }
    fclose(f);
    for (uint32_t i = 0; i < job->npre + job->nwarm; i++) {
        const PreItem *it = i < job->npre ? &job->pre[i] : &job->warm[i - job->npre];
        if (it->mesh < 0 || (uint32_t)it->mesh >= job->nmodules || (it->task >= 0 && (uint32_t)it->task >= job->nmodules))
            die(1, "a pre-rasterization library refers past the module list");
    }
    for (uint32_t i = 0; i < job->nfrag; i++)
        if (job->frag[i] < 0 || (uint32_t)job->frag[i] >= job->nmodules) die(1, "a fragment library refers past the module list");
    for (uint32_t i = 0; i < job->nlink; i++) {
        const LinkItem *l = &job->link[i];
        if (l->pre >= job->npre || l->frag >= job->nfrag || l->targets < 1 || l->targets > 8)
            die(1, "link %u refers past the pre and frag lists, or to a target count outside 1 to 8", i);
    }
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

static VkFormat parse_format(const char *s) {
    if (!strcmp(s, "rgba8")) return VK_FORMAT_R8G8B8A8_UNORM;
    if (!strcmp(s, "rgba16f")) return VK_FORMAT_R16G16B16A16_SFLOAT;
    if (!strcmp(s, "r11g11b10f")) return VK_FORMAT_B10G11R11_UFLOAT_PACK32;
    die(1, "unknown colour format %s", s);
}

static int parse_phases(const char *s) {
    int phases = 0;
    char buf[128];
    snprintf(buf, sizeof buf, "%s", s);
    for (char *tok = strtok(buf, ","); tok; tok = strtok(NULL, ",")) {
        if (!strcmp(tok, "pre")) phases |= PHASE_PRE;
        else if (!strcmp(tok, "frag")) phases |= PHASE_FRAG;
        else if (!strcmp(tok, "link")) phases |= PHASE_LINK;
        else die(1, "unknown phase %s", tok);
    }
    return phases;
}

// ---- the device ---------------------------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT types,
                                                 const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)user;
    if (types == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) return VK_FALSE;  // the loader's own notices
    const char *kind = "info";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { val_errors++; kind = "error"; }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { val_warnings++; kind = "warning"; }
    if (val_errors + val_warnings <= 50)
        fprintf(stderr, "vkmeshcost: validation %s: %s\n", kind, data->pMessage ? data->pMessage : "(no text)");
    return VK_FALSE;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3;
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
    f->mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    f->gpl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
    f->eds3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    f->af.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_FEATURES_EXT;
    f->af2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_ATOMIC_FLOAT_2_FEATURES_EXT;
    f->bin.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR;
    f->m5.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR;
    void **next = &f->f2.pNext;
    *next = &f->v11; next = &f->v11.pNext;
    *next = &f->v12; next = &f->v12.pNext;
    *next = &f->v13; next = &f->v13.pNext;
    *next = &f->mesh; next = &f->mesh.pNext;
    *next = &f->gpl; next = &f->gpl.pNext;
    *next = &f->eds3; next = &f->eds3.pNext;
    if (atomic_float) { *next = &f->af; next = &f->af.pNext; }
    if (atomic_float2) { *next = &f->af2; next = &f->af2.pNext; }
    if (binaries) { *next = &f->bin; next = &f->bin.pNext; *next = &f->m5; }
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

#define COPY_FROM(member_first, type, field)                                                      \
    memcpy((char *)&want.field + offsetof(type, member_first), (char *)&have.field + offsetof(type, member_first), \
           sizeof have.field - offsetof(type, member_first))

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
    VkValidationFeatureEnableEXT sync_on = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    VkValidationFeaturesEXT vf = {.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, .pNext = &mci,
                                  .enabledValidationFeatureCount = opt.validation == 2 ? 1u : 0u,
                                  .pEnabledValidationFeatures = &sync_on};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkmeshcost",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pNext = opt.validation ? (void *)&vf : NULL,
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

    VkPhysicalDeviceVulkan13Properties p13 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_PROPERTIES};
    VkPhysicalDeviceVulkan11Properties p11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES, .pNext = &p13};
    VkPhysicalDeviceVulkan12Properties p12 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_PROPERTIES, .pNext = &p11};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p12};
    vkGetPhysicalDeviceProperties2(gpu, &p2);
    subgroup_size = p11.subgroupSize;
    pin32_stages = p13.requiredSubgroupSizeStages;
    if (p12.maxPerStageDescriptorUpdateAfterBindSamplers < GLOBAL_SAMPLERS ||
        p12.maxDescriptorSetUpdateAfterBindSamplers < GLOBAL_SAMPLERS ||
        p12.maxPerStageDescriptorUpdateAfterBindSampledImages < GLOBAL_IMAGES ||
        p12.maxDescriptorSetUpdateAfterBindSampledImages < GLOBAL_IMAGES)
        die(2, "the global arrays are longer than the device's update-after-bind limits");

    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, NULL);
    VkExtensionProperties *dexts = calloc(next ? next : 1, sizeof *dexts);
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, dexts);
    const char *dext[MAX_EXTS + 8];
    uint32_t ndext = 0;
    const char *need[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME,
                          VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME, VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME,
                          VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME};
    for (size_t i = 0; i < sizeof need / sizeof need[0]; i++) {
        if (!has_ext(dexts, next, need[i])) die(2, "the device lacks %s", need[i]);
        dext[ndext++] = need[i];
    }
    int atomic_float = 0, atomic_float2 = 0;
    for (uint32_t i = 0; i < job->nexts; i++) {
        if (!has_ext(dexts, next, job->exts[i])) die(2, "the device lacks %s", job->exts[i]);
        if (!strcmp(job->exts[i], VK_EXT_MESH_SHADER_EXTENSION_NAME)) continue;  // already listed
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

    // Every feature the device offers, except robustness and protected memory, which change what the compiler emits or
    // what memory is (vkpipecost's rule).
    Features have, want;
    features_chain(&have, atomic_float, atomic_float2, binaries);
    vkGetPhysicalDeviceFeatures2(gpu, &have.f2);
    features_chain(&want, atomic_float, atomic_float2, binaries);
    want.f2.features = have.f2.features;
    want.f2.features.robustBufferAccess = VK_FALSE;
    COPY_FROM(storageBuffer16BitAccess, VkPhysicalDeviceVulkan11Features, v11);
    want.v11.protectedMemory = VK_FALSE;
    COPY_FROM(samplerMirrorClampToEdge, VkPhysicalDeviceVulkan12Features, v12);
    COPY_FROM(robustImageAccess, VkPhysicalDeviceVulkan13Features, v13);
    want.v13.robustImageAccess = VK_FALSE;
    want.mesh.meshShader = have.mesh.meshShader;
    want.mesh.taskShader = have.mesh.taskShader;
    want.gpl.graphicsPipelineLibrary = have.gpl.graphicsPipelineLibrary;
    COPY_FROM(extendedDynamicState3TessellationDomainOrigin, VkPhysicalDeviceExtendedDynamicState3FeaturesEXT, eds3);
    COPY_FROM(shaderBufferFloat32Atomics, VkPhysicalDeviceShaderAtomicFloatFeaturesEXT, af);
    COPY_FROM(shaderBufferFloat16Atomics, VkPhysicalDeviceShaderAtomicFloat2FeaturesEXT, af2);
    want.bin.pipelineBinaries = have.bin.pipelineBinaries;
    want.m5.maintenance5 = have.m5.maintenance5;
    if (!want.f2.features.shaderInt64 || !want.v12.bufferDeviceAddress)
        die(2, "the device lacks shaderInt64 or bufferDeviceAddress");
    if (!want.mesh.meshShader || !want.mesh.taskShader || !want.gpl.graphicsPipelineLibrary)
        die(2, "the device lacks meshShader, taskShader or graphicsPipelineLibrary");
    if (!want.v12.descriptorBindingSampledImageUpdateAfterBind || !want.v12.descriptorBindingPartiallyBound)
        die(2, "the device lacks update-after-bind sampled images or partially bound arrays");
    const VkShaderStageFlags both = VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT;
    if (pin32 && (!want.v13.subgroupSizeControl || !want.v13.computeFullSubgroups || (pin32_stages & both) != both))
        die(2, "pin32 needs subgroupSizeControl, computeFullSubgroups and a required subgroup size on the task and mesh stages");

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, NULL);
    VkQueueFamilyProperties *qf = calloc(nqf, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, qf);
    uint32_t family = UINT32_MAX;
    for (uint32_t i = 0; i < nqf && family == UINT32_MAX; i++)
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) family = i;
    free(qf);
    if (family == UINT32_MAX) die(2, "no graphics queue");
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

// ---- the layout `uni` ------------------------------------------------------------------------

typedef struct {
    VkDescriptorSetLayout set[3];
    VkPipelineLayout layout;
} Uni;

static void bind(VkDescriptorSetLayoutBinding *lb, uint32_t *n, uint32_t first, uint32_t count, VkDescriptorType type,
                 VkShaderStageFlags stages) {
    for (uint32_t i = 0; i < count; i++) lb[(*n)++] = (VkDescriptorSetLayoutBinding){first + i, type, 1, stages, NULL};
}

static void uni_create(Uni *u, int push_constants) {
    // Set 0: the global update-after-bind arrays, every image type a view of one binding.
    VkDescriptorSetLayoutBinding glb[2] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, GLOBAL_IMAGES, MESH_STAGES, NULL},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, GLOBAL_SAMPLERS, MESH_STAGES, NULL},
    };
    VkDescriptorBindingFlags gflags[2] = {
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo gbf = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                       .bindingCount = 2, .pBindingFlags = gflags};
    VkDescriptorSetLayoutCreateInfo g = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &gbf,
                                         .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                         .bindingCount = 2, .pBindings = glb};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &g, NULL, &u->set[0]));

    // Set 1: push descriptors, the support buffer at 0, then the mesh stage's slots and the pixel stage's (D13).
    VkDescriptorSetLayoutBinding pb[32];
    uint32_t np = 0;
    bind(pb, &np, 0, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, MESH_STAGES);
    bind(pb, &np, 1, 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_MESH_BIT_EXT);
    bind(pb, &np, 9, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_MESH_BIT_EXT);
    bind(pb, &np, 14, 11, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
    bind(pb, &np, 25, 5, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_FRAGMENT_BIT);
    bind(pb, &np, 30, 2, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, VK_SHADER_STAGE_FRAGMENT_BIT);
    VkDescriptorSetLayoutCreateInfo p = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                         .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                         .bindingCount = np, .pBindings = pb};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &p, NULL, &u->set[1]));

    // Set 2: the task stage's own descriptors, written by the host for each draw.
    VkDescriptorSetLayoutBinding tb[16];
    uint32_t nt = 0;
    bind(tb, &nt, 0, 8, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, VK_SHADER_STAGE_TASK_BIT_EXT);
    bind(tb, &nt, 8, 8, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, VK_SHADER_STAGE_TASK_BIT_EXT);
    VkDescriptorSetLayoutCreateInfo t = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                         .bindingCount = nt, .pBindings = tb};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &t, NULL, &u->set[2]));

    VkPushConstantRange pcr = {MESH_STAGES, 0, PUSH_BYTES};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 3,
                                     .pSetLayouts = u->set, .pushConstantRangeCount = push_constants ? 1u : 0u,
                                     .pPushConstantRanges = &pcr};
    VK_CHECK(vkCreatePipelineLayout(dev, &li, NULL, &u->layout));
}

static void uni_destroy(Uni *u) {
    vkDestroyPipelineLayout(dev, u->layout, NULL);
    for (int i = 0; i < 3; i++) vkDestroyDescriptorSetLayout(dev, u->set[i], NULL);
}

// ---- the libraries (vkpipecost's D12 dynamic state and library flags) ----------------------------

static const VkDynamicState DYN_PRE_RASTER[] = {
    VK_DYNAMIC_STATE_VIEWPORT_WITH_COUNT, VK_DYNAMIC_STATE_SCISSOR_WITH_COUNT, VK_DYNAMIC_STATE_LINE_WIDTH,
    VK_DYNAMIC_STATE_DEPTH_BIAS, VK_DYNAMIC_STATE_CULL_MODE, VK_DYNAMIC_STATE_FRONT_FACE,
    VK_DYNAMIC_STATE_DEPTH_BIAS_ENABLE, VK_DYNAMIC_STATE_RASTERIZER_DISCARD_ENABLE, VK_DYNAMIC_STATE_POLYGON_MODE_EXT,
    VK_DYNAMIC_STATE_DEPTH_CLAMP_ENABLE_EXT,
};
static const VkDynamicState DYN_FRAGMENT[] = {
    VK_DYNAMIC_STATE_DEPTH_TEST_ENABLE, VK_DYNAMIC_STATE_DEPTH_WRITE_ENABLE, VK_DYNAMIC_STATE_DEPTH_COMPARE_OP,
    VK_DYNAMIC_STATE_DEPTH_BOUNDS_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_TEST_ENABLE, VK_DYNAMIC_STATE_STENCIL_OP,
    VK_DYNAMIC_STATE_STENCIL_COMPARE_MASK, VK_DYNAMIC_STATE_STENCIL_WRITE_MASK, VK_DYNAMIC_STATE_STENCIL_REFERENCE,
    VK_DYNAMIC_STATE_DEPTH_BOUNDS,
};
static const VkDynamicState DYN_OUTPUT[] = {
    VK_DYNAMIC_STATE_BLEND_CONSTANTS, VK_DYNAMIC_STATE_LOGIC_OP_ENABLE_EXT, VK_DYNAMIC_STATE_COLOR_BLEND_ENABLE_EXT,
    VK_DYNAMIC_STATE_COLOR_BLEND_EQUATION_EXT, VK_DYNAMIC_STATE_COLOR_WRITE_MASK_EXT,
};

typedef struct {
    VkPipelineShaderStageCreateInfo stages[3];
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo pin;
    VkPipelineViewportStateCreateInfo vps;
    VkPipelineRasterizationStateCreateInfo rs;
    VkPipelineMultisampleStateCreateInfo ms;
    VkPipelineDepthStencilStateCreateInfo ds;
    VkPipelineColorBlendAttachmentState cba[8];
    VkPipelineColorBlendStateCreateInfo cbs;
    VkFormat colour[8];
    VkPipelineRenderingCreateInfo rendering;
    VkGraphicsPipelineLibraryCreateInfoEXT lib;
    VkDynamicState dyn[16];
    VkPipelineDynamicStateCreateInfo dynamic;
    VkGraphicsPipelineCreateInfo gp;
} LibInfo;

static VkFormat colour_format = VK_FORMAT_R16G16B16A16_SFLOAT;

// Fills a create info for one library part. task, mesh and frag are VK_NULL_HANDLE where the part has no such stage.
static void lib_info(LibInfo *p, VkGraphicsPipelineLibraryFlagsEXT part, VkShaderModule task, VkShaderModule mesh,
                     VkShaderModule frag, VkPipelineLayout layout, uint32_t targets, int pin32) {
    memset(p, 0, sizeof *p);
    p->vps.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO;
    p->rs = (VkPipelineRasterizationStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                     .polygonMode = VK_POLYGON_MODE_FILL, .lineWidth = 1.0f};
    p->ms = (VkPipelineMultisampleStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                   .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    p->ds.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO;
    for (uint32_t i = 0; i < 8; i++) {
        p->cba[i].colorWriteMask = 0xF;
        p->colour[i] = colour_format;
    }
    p->cbs = (VkPipelineColorBlendStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                                   .attachmentCount = targets, .pAttachments = p->cba};
    p->rendering = (VkPipelineRenderingCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                                   .colorAttachmentCount = targets, .pColorAttachmentFormats = p->colour,
                                                   .depthAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT,
                                                   .stencilAttachmentFormat = VK_FORMAT_D32_SFLOAT_S8_UINT};
    uint32_t nd = 0;
    const VkDynamicState *list = NULL;
    size_t count = 0;
    if (part == VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT) { list = DYN_PRE_RASTER; count = sizeof DYN_PRE_RASTER / sizeof *list; }
    else if (part == VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) { list = DYN_FRAGMENT; count = sizeof DYN_FRAGMENT / sizeof *list; }
    else { list = DYN_OUTPUT; count = sizeof DYN_OUTPUT / sizeof *list; }
    for (size_t i = 0; i < count; i++) p->dyn[nd++] = list[i];
    p->dynamic = (VkPipelineDynamicStateCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO,
                                                    .dynamicStateCount = nd, .pDynamicStates = p->dyn};
    p->pin = (VkPipelineShaderStageRequiredSubgroupSizeCreateInfo){
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = 32};
    VkPipelineShaderStageCreateFlags flags = pin32 ? VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0;
    void *pin_next = pin32 ? (void *)&p->pin : NULL;
    uint32_t ns = 0;
    if (part == VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT) {
        if (task)
            p->stages[ns++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = pin_next, .flags = flags, .stage = VK_SHADER_STAGE_TASK_BIT_EXT, .module = task, .pName = "main"};
        p->stages[ns++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = pin_next, .flags = flags, .stage = VK_SHADER_STAGE_MESH_BIT_EXT, .module = mesh, .pName = "main"};
    } else if (part == VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) {
        p->stages[ns++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = frag, .pName = "main"};
    }
    p->lib = (VkGraphicsPipelineLibraryCreateInfoEXT){.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
                                                      .pNext = &p->rendering, .flags = part};
    int pre = part == VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;
    int fs = part == VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;
    int out = part == VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT;
    p->gp = (VkGraphicsPipelineCreateInfo){
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &p->lib,
        .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR | VK_PIPELINE_CREATE_RETAIN_LINK_TIME_OPTIMIZATION_INFO_BIT_EXT,
        .stageCount = ns, .pStages = ns ? p->stages : NULL, .pViewportState = pre ? &p->vps : NULL,
        .pRasterizationState = pre ? &p->rs : NULL, .pMultisampleState = (fs || out) ? &p->ms : NULL,
        .pDepthStencilState = fs ? &p->ds : NULL, .pColorBlendState = out ? &p->cbs : NULL,
        .pDynamicState = &p->dynamic, .layout = (pre || fs) ? layout : VK_NULL_HANDLE};
}

// ---- the build -----------------------------------------------------------------------------------

typedef struct {
    const Job *job;
    VkPipelineLayout layout;
    VkPipelineCache cache;
    int pin32, keep;
    const char *phase;
    VkPipeline *pre, *frag;     // the libraries a link phase needs, kept
    VkPipeline out[9];          // the fragment output library of 1 to 8 targets
    volatile LONG done, failed;
    volatile LONG in_flight[MAX_WORKERS];  // item + 1 while a worker builds it, 0 otherwise
    double started[MAX_WORKERS];
} BuildCtx;

// Reads a SPIR-V file and creates its module; returns 0, or 1 when the file is unreadable or not SPIR-V.
static int module_make(const char *path, VkShaderModule *sm, size_t *bytes, double *ms, VkResult *result) {
    size_t size = 0;
    uint32_t *code = read_file(path, &size);
    if (!code || size % 4 || code[0] != 0x07230203u) {
        free(code);
        *bytes = size;
        return 1;
    }
    VkShaderModuleCreateInfo mci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = size, .pCode = code};
    double t0 = now_ms();
    *result = vkCreateShaderModule(dev, &mci, NULL, sm);
    *ms += now_ms() - t0;
    *bytes += size;
    free(code);
    return 0;
}

// Builds one library and writes its row. kept receives the pipeline when non-NULL, else it is destroyed.
static int build_lib(BuildCtx *b, VkGraphicsPipelineLibraryFlagsEXT part, int task, int mesh, int frag, long index,
                     uint32_t worker, const char *phase, const char *what, VkPipeline *kept) {
    const Job *job = b->job;
    VkShaderModule sm[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};  // task, mesh, fragment
    int ref[3] = {task, mesh, frag};
    size_t bytes = 0;
    double module_ms = 0, ms = 0, cpu = 0;
    VkResult r = VK_SUCCESS;
    char extra[256];
    for (int i = 0; i < 3 && r == VK_SUCCESS; i++) {
        if (ref[i] < 0) continue;
        if (module_make(job->modules[ref[i]].path, &sm[i], &bytes, &module_ms, &r)) {
            snprintf(extra, sizeof extra, "result=unreadable bytes=%zu", bytes);
            out_row(phase, what, index, 0, (int)worker, extra);
            for (int j = 0; j < 3; j++) if (sm[j]) vkDestroyShaderModule(dev, sm[j], NULL);
            return 1;
        }
    }
    VkPipeline p = VK_NULL_HANDLE;
    if (r == VK_SUCCESS) {
        LibInfo li;
        lib_info(&li, part, sm[0], sm[1], sm[2], b->layout, 0, b->pin32);
        double c0 = thread_cpu_ms();
        double t0 = now_ms();
        r = vkCreateGraphicsPipelines(dev, b->cache, 1, &li.gp, NULL, &p);
        ms = now_ms() - t0;
        cpu = thread_cpu_ms() - c0;
    }
    if (r == VK_ERROR_DEVICE_LOST) die(3, "device lost building %s %ld", what, index);
    snprintf(extra, sizeof extra, "result=%d cpu_ms=%.3f module_ms=%.3f bytes=%zu", (int)r, cpu, module_ms, bytes);
    out_row(phase, what, index, ms, (int)worker, extra);
    for (int j = 0; j < 3; j++) if (sm[j]) vkDestroyShaderModule(dev, sm[j], NULL);
    if (kept) *kept = p;
    else if (p) vkDestroyPipeline(dev, p, NULL);
    return r != VK_SUCCESS;
}

static void w_pre(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    const PreItem *it = &b->job->pre[i];
    b->started[w] = now_ms();
    InterlockedExchange(&b->in_flight[w], (LONG)i + 1);
    if (build_lib(b, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, it->task, it->mesh, -1, (long)i, w,
                  b->phase, "pre_raster_lib", b->keep ? &b->pre[i] : NULL))
        InterlockedIncrement(&b->failed);
    InterlockedExchange(&b->in_flight[w], 0);
    InterlockedIncrement(&b->done);
}

static void w_frag(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    b->started[w] = now_ms();
    InterlockedExchange(&b->in_flight[w], (LONG)i + 1);
    if (build_lib(b, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, -1, -1, b->job->frag[i], (long)i, w, b->phase,
                  "fragment_lib", b->keep ? &b->frag[i] : NULL))
        InterlockedIncrement(&b->failed);
    InterlockedExchange(&b->in_flight[w], 0);
    InterlockedIncrement(&b->done);
}

static void w_link(void *c, uint32_t i, uint32_t w) {
    BuildCtx *b = c;
    const LinkItem *l = &b->job->link[i];
    char extra[64];
    if (!b->pre[l->pre] || !b->frag[l->frag]) {
        InterlockedIncrement(&b->failed);
        out_row(b->phase, "fast_link", (long)i, 0, (int)w, "result=missing_library");
        InterlockedIncrement(&b->done);
        return;
    }
    VkPipeline libs[3] = {b->pre[l->pre], b->frag[l->frag], b->out[l->targets]};
    VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = 3,
                                         .pLibraries = libs};
    VkGraphicsPipelineCreateInfo gp = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li,
                                       .layout = b->layout};
    VkPipeline p = VK_NULL_HANDLE;
    double c0 = thread_cpu_ms();
    double t0 = now_ms();
    VkResult r = vkCreateGraphicsPipelines(dev, b->cache, 1, &gp, NULL, &p);
    double ms = now_ms() - t0;
    snprintf(extra, sizeof extra, "result=%d cpu_ms=%.3f", (int)r, thread_cpu_ms() - c0);
    if (r == VK_ERROR_DEVICE_LOST) die(3, "device lost linking %u", i);
    out_row(b->phase, "fast_link", (long)i, ms, (int)w, extra);
    if (r != VK_SUCCESS) InterlockedIncrement(&b->failed);
    if (p) vkDestroyPipeline(dev, p, NULL);
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
        int at = snprintf(line, sizeof line, "vkmeshcost: %.0f s, %ld of %u done, working set %.0f MB, available %.0f MB, building",
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

// One phase: n items on the job's threads, with a heartbeat; returns the wall time in milliseconds.
static double phase_run(BuildCtx *b, const Job *job, const char *name, uint32_t n, WorkFn fn) {
    b->phase = name;
    b->done = 0;
    Heartbeat hb = {.b = b, .n = n, .period_ms = 1000.0 * (double)job_param_int(job, "heartbeat_s", 30), .t0 = now_ms(),
                    .wake = CreateEventA(NULL, TRUE, FALSE, NULL)};
    HANDLE ht = hb.period_ms > 0 ? CreateThread(NULL, 1 << 20, heartbeat_main, &hb, 0, NULL) : NULL;
    double wall = run_parallel(n, opt.threads, fn, b);
    SetEvent(hb.wake);
    if (ht) {
        WaitForSingleObject(ht, INFINITE);
        CloseHandle(ht);
    }
    CloseHandle(hb.wake);
    return wall;
}

static int run(Job *job) {
    int pin32 = job_param_int(job, "pin32", 1) != 0;
    int phases = parse_phases(job_param(job, "phases", "pre"));
    if (!phases) die(1, "no phase to run");
    if ((phases & PHASE_LINK) && (phases & (PHASE_PRE | PHASE_FRAG)) != (PHASE_PRE | PHASE_FRAG))
        die(1, "the link phase needs the pre and frag phases in the same run");
    colour_format = parse_format(job_param(job, "format", "rgba16f"));
    device_init(job, pin32);
    out_open();
    summary_len = 0;
    summary_add("{\"mode\": \"mesh\"");
    Uni uni;
    uni_create(&uni, strcmp(job_param(job, "counter", ""), "nopush") != 0);
    BuildCtx b = {.job = job, .layout = uni.layout, .pin32 = pin32, .keep = (phases & PHASE_LINK) != 0};
    if (b.keep) {
        b.pre = calloc(job->npre ? job->npre : 1, sizeof *b.pre);
        b.frag = calloc(job->nfrag ? job->nfrag : 1, sizeof *b.frag);
        if (!b.pre || !b.frag) die(3, "out of memory");
    }

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

    // The fragment output libraries, 1 to 8 targets, only when something links; then the warm-up, alone.
    if (phases & PHASE_LINK) {
        for (uint32_t t = 1; t <= 8; t++) {
            LibInfo li;
            lib_info(&li, VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, VK_NULL_HANDLE, VK_NULL_HANDLE,
                     VK_NULL_HANDLE, VK_NULL_HANDLE, t, 0);
            double t0 = now_ms();
            VK_CHECK(vkCreateGraphicsPipelines(dev, b.cache, 1, &li.gp, NULL, &b.out[t]));
            out_row("setup", "output_lib", (long)t, now_ms() - t0, 0, NULL);
        }
    }
    uint32_t wfail = 0;
    for (uint32_t i = 0; i < job->nwarm; i++)
        wfail += (uint32_t)build_lib(&b, VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, job->warm[i].task,
                                     job->warm[i].mesh, -1, (long)i, 0, "warmup", "pre_raster_lib", NULL);
    summary_num("warmups", job->nwarm);
    summary_num("warmup_failed", wfail);

    double ws0;
    memory_mb(&ws0, NULL, NULL);
    double avail0 = available_mb();
    if (phases & PHASE_PRE) {
        summary_num("pre_wall_ms", phase_run(&b, job, "pre", job->npre, w_pre));
        summary_num("pre_items", job->npre);
        summary_num("pre_failed", b.failed);
    }
    LONG failed_pre = b.failed;
    if (phases & PHASE_FRAG) {
        summary_num("frag_wall_ms", phase_run(&b, job, "frag", job->nfrag, w_frag));
        summary_num("frag_items", job->nfrag);
        summary_num("frag_failed", b.failed - failed_pre);
    }
    LONG failed_frag = b.failed;
    if (phases & PHASE_LINK) {
        summary_num("link_wall_ms", phase_run(&b, job, "link", job->nlink, w_link));
        summary_num("link_items", job->nlink);
        summary_num("link_failed", b.failed - failed_frag);
    }
    summary_num("failed", b.failed);
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
    if (b.keep) {
        for (uint32_t i = 0; i < job->npre; i++) if (b.pre[i]) vkDestroyPipeline(dev, b.pre[i], NULL);
        for (uint32_t i = 0; i < job->nfrag; i++) if (b.frag[i]) vkDestroyPipeline(dev, b.frag[i], NULL);
        for (uint32_t t = 1; t <= 8; t++) if (b.out[t]) vkDestroyPipeline(dev, b.out[t], NULL);
        free(b.pre);
        free(b.frag);
    }
    uni_destroy(&uni);
    summary_end();
    out_close();
    device_finish();
    return val_errors ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc != 2) die(1, "usage: vkmeshcost JOB");
    static Job job;
    job_read(argv[1], &job);
    return run(&job);
}
