// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkpipecost: the measurements of spike S3 (architecture section 12).
//
//   vkpipecost build  JOB libs|mono|objects    build times of the job's pairs
//   vkpipecost cache  JOB build|load|binaries  pipeline cache and pipeline binaries
//   vkpipecost draws  JOB                      per-draw cost of the descriptor layouts
//   vkpipecost replay JOB                      replay with and without a recorder thread
//
// The job file is text, one directive per line, paths last on their line:
//   validation off|on            cache internal|disabled     threads N
//   device SUBSTRING             out PATH                    push BYTES
//   module v|f PATH              set                         global SETNUMBER
//   bind BINDING TYPE COUNT STAGES                (appended to the last set or global)
//   rebind N                     (set 0 bindings of fragment modules move by N, binding 0 excepted)
//   format rgba8|bgra8|rgba16f|r11g11b10f|rgb10a2|r32f  depth d32s8|d24s8|d32|none
//   pair VS FS SET TARGETS FORMAT                  (indices into the lists above)
//   param NAME VALUE                               (read by a mode)
// TYPE is ubo, ssbo, tex (combined image sampler), texbuf, septex (sampled
// image), sampler, img (storage image) or imgbuf; STAGES is v, f or vf.
// `cache disabled` creates the device with the driver's internal pipeline cache
// switched off (VK_KHR_pipeline_binary's cache control), so every build is cold.
//
// Per-item times go to the `out` file as TSV rows (phase, what, index,
// milliseconds, worker, extra); a summary goes to stdout as one JSON line.
// Exit codes: 0 measured, 1 the job file is wrong, 2 Vulkan lacks something the
// job needs, 3 a Vulkan call failed.
//
// No game data: modules are inputs, written by the caller (tools/pipecost/pipe_cost.py).

#include "vkpc.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

Options opt = {.validation = 0, .internal_cache = 1, .threads = 1};
VkInstance inst;
VkPhysicalDevice gpu;
VkDevice dev;
VkQueue queue;
uint32_t queue_family;
VkPhysicalDeviceProperties gpu_props;
VkPhysicalDeviceMemoryProperties mem_props;
unsigned val_errors, val_warnings;
int have_binaries;

static VkDebugUtilsMessengerEXT messenger;

void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "vkpipecost: ");
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    fflush(stdout);
    exit(code);
}

// ---- time, memory, threads -------------------------------------------------------

static double qpc_per_ms, cycles_per_ms;

double now_ms(void) {
    LARGE_INTEGER t;
    QueryPerformanceCounter(&t);
    return (double)t.QuadPart / qpc_per_ms;
}

double thread_cpu_ms(void) {
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

void memory_mb(double *ws, double *priv, double *peak) {
    MemCounters m = {.cb = sizeof m};
    K32GetProcessMemoryInfo(GetCurrentProcess(), &m, sizeof m);
    if (ws) *ws = (double)m.WorkingSetSize / 1048576.0;
    if (priv) *priv = (double)m.PrivateUsage / 1048576.0;
    if (peak) *peak = (double)m.PeakWorkingSetSize / 1048576.0;
}

double available_mb(void) {
    MEMORYSTATUSEX s = {.dwLength = sizeof s};
    GlobalMemoryStatusEx(&s);
    return (double)s.ullAvailPhys / 1048576.0;
}

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

double run_parallel(uint32_t n, int threads, WorkFn fn, void *ctx) {
    if (threads < 1) threads = 1;
    Pool pool = {fn, ctx, n, 0};
    PoolArg args[64];
    HANDLE h[64];
    if (threads > 64) threads = 64;
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

struct Thread {
    HANDLE h;
    void (*fn)(void *);
    void *arg;
};

static DWORD WINAPI thread_main(LPVOID p) {
    Thread *t = p;
    t->fn(t->arg);
    return 0;
}

Thread *thread_start(void (*fn)(void *), void *arg) {
    Thread *t = calloc(1, sizeof *t);
    t->fn = fn;
    t->arg = arg;
    t->h = CreateThread(NULL, 4 << 20, thread_main, t, 0, NULL);
    if (!t->h) die(3, "CreateThread failed");
    return t;
}

void thread_join(Thread *t) {
    WaitForSingleObject(t->h, INFINITE);
    CloseHandle(t->h);
    free(t);
}

// ---- output --------------------------------------------------------------------

static FILE *out_file;
static CRITICAL_SECTION out_lock;

void out_open(void) {
    InitializeCriticalSection(&out_lock);
    if (!opt.out[0]) return;
    out_file = fopen(opt.out, "wb");
    if (!out_file) die(1, "cannot write %s", opt.out);
    fprintf(out_file, "phase\twhat\tindex\tms\tworker\textra\n");
}

void out_row(const char *phase, const char *what, long index, double ms, int worker, const char *extra) {
    if (!out_file) return;
    EnterCriticalSection(&out_lock);
    fprintf(out_file, "%s\t%s\t%ld\t%.4f\t%d\t%s\n", phase, what, index, ms, worker, extra ? extra : "");
    LeaveCriticalSection(&out_lock);
}

void out_close(void) {
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

void summary_begin(const char *mode) {
    summary_len = 0;
    summary_add("{\"mode\": \"%s\"", mode);
}

void summary_num(const char *key, double value) { summary_add(", \"%s\": %.6g", key, value); }

void summary_str(const char *key, const char *value) { summary_add(", \"%s\": \"%s\"", key, value); }

void summary_end(void) {
    double ws, priv, peak;
    memory_mb(&ws, &priv, &peak);
    uint32_t dv = gpu_props.driverVersion;
    char driver[32];
    snprintf(driver, sizeof driver, "%u.%02u", (dv >> 22) & 0x3FF, (dv >> 14) & 0xFF);
    summary_add(", \"device\": \"%s\", \"driver\": \"%s\", \"internal_cache\": %d, \"threads\": %d, "
                "\"validation\": %d, \"validation_errors\": %u, \"validation_warnings\": %u, "
                "\"working_set_mb\": %.1f, \"private_mb\": %.1f, \"peak_working_set_mb\": %.1f}",
                gpu_props.deviceName, driver, opt.internal_cache, opt.threads, opt.validation, val_errors,
                val_warnings, ws, priv, peak);
    printf("%s\n", summary);
    fflush(stdout);
}

// ---- the job file ---------------------------------------------------------------

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
    if (!f) die(1, "cannot open %s", path);
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) die(1, "%s is empty", path);
    void *p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) die(1, "cannot read %s", path);
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
    return 0;
}

static VkShaderStageFlags parse_stages(const char *s) {
    VkShaderStageFlags f = 0;
    for (; *s; s++) {
        if (*s == 'v') f |= VK_SHADER_STAGE_VERTEX_BIT;
        else if (*s == 'f') f |= VK_SHADER_STAGE_FRAGMENT_BIT;
        else die(1, "stages are v, f or vf");
    }
    return f;
}

static VkFormat parse_format(const char *s) {
    static const struct { const char *name; VkFormat f; } F[] = {
        {"rgba8", VK_FORMAT_R8G8B8A8_UNORM}, {"bgra8", VK_FORMAT_B8G8R8A8_UNORM},
        {"rgba16f", VK_FORMAT_R16G16B16A16_SFLOAT}, {"r11g11b10f", VK_FORMAT_B10G11R11_UFLOAT_PACK32},
        {"rgb10a2", VK_FORMAT_A2B10G10R10_UNORM_PACK32}, {"r32f", VK_FORMAT_R32_SFLOAT},
        {"d32s8", VK_FORMAT_D32_SFLOAT_S8_UINT}, {"d24s8", VK_FORMAT_D24_UNORM_S8_UINT}, {"d32", VK_FORMAT_D32_SFLOAT},
        {"none", VK_FORMAT_UNDEFINED},
    };
    for (size_t i = 0; i < sizeof F / sizeof F[0]; i++)
        if (!strcmp(s, F[i].name)) return F[i].f;
    die(1, "unknown format %s", s);
    return 0;
}

#define GROW(arr, n, cap)                                                   \
    do {                                                                    \
        if ((n) == (cap)) {                                                 \
            (cap) = (cap) ? 2 * (cap) : 64;                                 \
            (arr) = realloc((arr), (cap) * sizeof *(arr));                  \
            if (!(arr)) die(3, "out of memory");                            \
        }                                                                   \
    } while (0)

void job_read(const char *path, Job *job) {
    memset(job, 0, sizeof *job);
    job->depth_format = VK_FORMAT_D32_SFLOAT_S8_UINT;
    FILE *f = fopen(path, "rb");
    if (!f) die(1, "cannot open the job file %s", path);
    char line[4096];
    unsigned lineno = 0;
    uint32_t cap_m = 0, cap_s = 0, cap_p = 0;
    SetDesc *current = NULL;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
        if (!n || line[0] == '#') continue;
        char w[64] = {0}, a[64] = {0}, b[64] = {0}, c[64] = {0}, d[64] = {0}, e[64] = {0};
        int k = sscanf(line, "%63s %63s %63s %63s %63s %63s", w, a, b, c, d, e);
        if (!strcmp(w, "validation") && k == 2) opt.validation = !strcmp(a, "on");
        else if (!strcmp(w, "cache") && k == 2) opt.internal_cache = strcmp(a, "disabled") != 0;
        else if (!strcmp(w, "threads") && k == 2) opt.threads = atoi(a);
        else if (!strcmp(w, "device") && k >= 2) snprintf(opt.device_filter, sizeof opt.device_filter, "%s", rest_after(line, 1));
        else if (!strcmp(w, "out") && k >= 2) snprintf(opt.out, sizeof opt.out, "%s", rest_after(line, 1));
        else if (!strcmp(w, "push") && k == 2) job->push_bytes = (uint32_t)atoi(a);
        else if (!strcmp(w, "rebind") && k == 2) job->rebind = (uint32_t)atoi(a);
        else if (!strcmp(w, "module") && k >= 3) {
            GROW(job->modules, job->nmodules, cap_m);
            Module *m = &job->modules[job->nmodules++];
            memset(m, 0, sizeof *m);
            m->stage = a[0] == 'v' ? VK_SHADER_STAGE_VERTEX_BIT : VK_SHADER_STAGE_FRAGMENT_BIT;
            snprintf(m->path, sizeof m->path, "%s", rest_after(line, 2));
        } else if (!strcmp(w, "set") && k == 1) {
            GROW(job->sets, job->nsets, cap_s);
            current = &job->sets[job->nsets++];
            memset(current, 0, sizeof *current);
        } else if (!strcmp(w, "global") && k == 2) {
            job->has_global = 1;
            job->global_set = (uint32_t)atoi(a);
            current = &job->global;
        } else if (!strcmp(w, "bind") && k == 5) {
            if (!current) die(1, "line %u: bind before set or global", lineno);
            if (current->n == MAX_BINDS) die(1, "line %u: too many bindings", lineno);
            current->b[current->n++] = (Bind){(uint32_t)atoi(a), parse_type(b), (uint32_t)atoi(c), parse_stages(d)};
        } else if (!strcmp(w, "format") && k == 2) {
            if (job->nformats == 8) die(1, "too many formats");
            job->formats[job->nformats++] = parse_format(a);
        } else if (!strcmp(w, "depth") && k == 2) job->depth_format = parse_format(a);
        else if (!strcmp(w, "pair") && k == 6) {
            GROW(job->pairs, job->npairs, cap_p);
            job->pairs[job->npairs++] = (Pair){(uint32_t)atoi(a), (uint32_t)atoi(b), (uint32_t)atoi(c),
                                               (uint32_t)atoi(d), (uint32_t)atoi(e)};
        } else if (!strcmp(w, "param") && k >= 3) {
            if (job->nparams == 64) die(1, "too many params");
            snprintf(job->words[job->nparams], 64, "%s", a);
            snprintf(job->values[job->nparams], MAX_PATH_LEN, "%s", rest_after(line, 2));
            job->nparams++;
        } else die(1, "line %u not understood: %s", lineno, line);
    }
    fclose(f);
    for (uint32_t i = 0; i < job->npairs; i++) {
        Pair *p = &job->pairs[i];
        if (p->vs >= job->nmodules || p->fs >= job->nmodules || p->set >= job->nsets || p->format >= job->nformats ||
            p->targets > 8)
            die(1, "pair %u refers past the lists", i);
        if (job->modules[p->vs].stage != VK_SHADER_STAGE_VERTEX_BIT ||
            job->modules[p->fs].stage != VK_SHADER_STAGE_FRAGMENT_BIT)
            die(1, "pair %u: stages are not vertex then fragment", i);
    }
}

uint32_t spirv_rebind(uint32_t *words, size_t nwords, uint32_t add) {
    enum { OP_DECORATE = 71, DEC_BINDING = 33, DEC_SET = 34 };
    if (nwords < 5 || words[0] != 0x07230203u) return 0;
    uint32_t bound = words[3];
    uint8_t *set0 = calloc(bound ? bound : 1, 1);
    for (int pass = 0; pass < 2; pass++)
        for (size_t i = 5; i < nwords;) {
            uint32_t n = words[i] >> 16, op = words[i] & 0xFFFF;
            if (!n || i + n > nwords) break;
            if (op == OP_DECORATE && n >= 4 && words[i + 1] < bound) {
                if (pass == 0 && words[i + 2] == DEC_SET && words[i + 3] == 0) set0[words[i + 1]] = 1;
                if (pass == 1 && words[i + 2] == DEC_BINDING && set0[words[i + 1]] && words[i + 3] != 0) {
                    words[i + 3] += add;
                    set0[words[i + 1]] = 2;
                }
            }
            i += n;
        }
    uint32_t moved = 0;
    for (uint32_t id = 0; id < bound; id++) moved += set0[id] == 2;
    free(set0);
    return moved;
}

void module_load(const Job *job, Module *m) {
    if (m->code) return;
    m->code = read_file(m->path, &m->size);
    if (m->size % 4 || m->code[0] != 0x07230203u) die(1, "%s is not SPIR-V", m->path);
    if (job->rebind && m->stage == VK_SHADER_STAGE_FRAGMENT_BIT) spirv_rebind(m->code, m->size / 4, job->rebind);
}

void module_free(Module *m) {
    free(m->code);
    m->code = NULL;
}

const char *job_param(const Job *job, const char *name, const char *fallback) {
    for (uint32_t i = 0; i < job->nparams; i++)
        if (!strcmp(job->words[i], name)) return job->values[i];
    return fallback;
}

long job_param_int(const Job *job, const char *name, long fallback) {
    const char *v = job_param(job, name, NULL);
    return v ? strtol(v, NULL, 0) : fallback;
}

VkShaderModule module_create(const Module *m) {
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = m->size,
                                   .pCode = m->code};
    VkShaderModule s;
    VK_CHECK(vkCreateShaderModule(dev, &ci, NULL, &s));
    return s;
}

VkDescriptorSetLayout set_layout_create(const SetDesc *d, VkDescriptorSetLayoutCreateFlags flags, int uab) {
    VkDescriptorSetLayoutBinding lb[MAX_BINDS];
    VkDescriptorBindingFlags bf[MAX_BINDS];
    for (uint32_t i = 0; i < d->n; i++) {
        lb[i] = (VkDescriptorSetLayoutBinding){d->b[i].binding, d->b[i].type, d->b[i].count, d->b[i].stages, NULL};
        bf[i] = uab ? VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT : 0;
    }
    VkDescriptorSetLayoutBindingFlagsCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                       .bindingCount = d->n, .pBindingFlags = bf};
    VkDescriptorSetLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .pNext = uab ? &fi : NULL,
                                          .flags = flags | (uab ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT : 0),
                                          .bindingCount = d->n, .pBindings = lb};
    VkDescriptorSetLayout l;
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &ci, NULL, &l));
    return l;
}

uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & want) == want) return i;
    die(2, "no memory type fits");
    return 0;
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
        fprintf(stderr, "vkpipecost: validation %s: %s\n", kind, data->pMessage ? data->pMessage : "(no text)");
    return VK_FALSE;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
    VkPhysicalDeviceShaderObjectFeaturesEXT so;
    VkPhysicalDeviceExtendedDynamicState3FeaturesEXT eds3;
    VkPhysicalDeviceVertexInputDynamicStateFeaturesEXT vi;
    VkPhysicalDevicePipelineBinaryFeaturesKHR bin;
    VkPhysicalDeviceMaintenance5FeaturesKHR m5;
} Features;

static void features_chain(Features *f, int binaries) {
    memset(f, 0, sizeof *f);
    f->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f->gpl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
    f->so.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SHADER_OBJECT_FEATURES_EXT;
    f->eds3.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTENDED_DYNAMIC_STATE_3_FEATURES_EXT;
    f->vi.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VERTEX_INPUT_DYNAMIC_STATE_FEATURES_EXT;
    f->bin.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PIPELINE_BINARY_FEATURES_KHR;
    f->m5.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MAINTENANCE_5_FEATURES_KHR;
    f->f2.pNext = &f->v11;
    f->v11.pNext = &f->v12;
    f->v12.pNext = &f->v13;
    f->v13.pNext = &f->gpl;
    f->gpl.pNext = &f->so;
    f->so.pNext = &f->eds3;
    f->eds3.pNext = &f->vi;
    f->vi.pNext = binaries ? (void *)&f->bin : NULL;
    f->bin.pNext = &f->m5;
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

void device_init(void) {
    clocks_init();
    if (volkInitialize() != VK_SUCCESS) die(2, "no Vulkan loader");
    const char *layers[1] = {"VK_LAYER_KHRONOS_validation"};
    const char *iexts[1] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkDebugUtilsMessengerCreateInfoEXT mci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        .pfnUserCallback = on_message};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkpipecost",
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
    vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);

    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, NULL);
    VkExtensionProperties *dexts = calloc(next ? next : 1, sizeof *dexts);
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, dexts);
    const char *need[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME,
                          VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME, VK_EXT_SHADER_OBJECT_EXTENSION_NAME,
                          VK_EXT_EXTENDED_DYNAMIC_STATE_3_EXTENSION_NAME,
                          VK_EXT_VERTEX_INPUT_DYNAMIC_STATE_EXTENSION_NAME};
    const char *dext[16];
    uint32_t ndext = 0;
    for (size_t i = 0; i < sizeof need / sizeof need[0]; i++) {
        if (!has_ext(dexts, next, need[i])) die(2, "the device lacks %s", need[i]);
        dext[ndext++] = need[i];
    }
    have_binaries = has_ext(dexts, next, VK_KHR_PIPELINE_BINARY_EXTENSION_NAME) &&
                    has_ext(dexts, next, VK_KHR_MAINTENANCE_5_EXTENSION_NAME);
    if (have_binaries) {
        dext[ndext++] = VK_KHR_MAINTENANCE_5_EXTENSION_NAME;
        dext[ndext++] = VK_KHR_PIPELINE_BINARY_EXTENSION_NAME;
    }
    free(dexts);
    if (!opt.internal_cache && !have_binaries) die(2, "cache disabled needs VK_KHR_pipeline_binary");

    // Every feature the device offers, except robustness and protected memory,
    // which change what the compiler emits or what memory is.
    Features have, want;
    features_chain(&have, have_binaries);
    vkGetPhysicalDeviceFeatures2(gpu, &have.f2);
    want = have;
    features_chain(&want, have_binaries);
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
    want.gpl.graphicsPipelineLibrary = have.gpl.graphicsPipelineLibrary;
    want.so.shaderObject = have.so.shaderObject;
    memcpy((char *)&want.eds3 + offsetof(VkPhysicalDeviceExtendedDynamicState3FeaturesEXT,
                                         extendedDynamicState3TessellationDomainOrigin),
           (char *)&have.eds3 + offsetof(VkPhysicalDeviceExtendedDynamicState3FeaturesEXT,
                                         extendedDynamicState3TessellationDomainOrigin),
           sizeof have.eds3 - offsetof(VkPhysicalDeviceExtendedDynamicState3FeaturesEXT,
                                       extendedDynamicState3TessellationDomainOrigin));
    want.vi.vertexInputDynamicState = have.vi.vertexInputDynamicState;
    want.bin.pipelineBinaries = have.bin.pipelineBinaries;
    want.m5.maintenance5 = have.m5.maintenance5;
    if (!want.gpl.graphicsPipelineLibrary || !want.so.shaderObject || !want.vi.vertexInputDynamicState ||
        !want.v13.dynamicRendering || !want.v12.descriptorBindingSampledImageUpdateAfterBind)
        die(2, "the device lacks a feature S3 needs");

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, NULL);
    VkQueueFamilyProperties *qf = calloc(nqf, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, qf);
    queue_family = UINT32_MAX;
    for (uint32_t i = 0; i < nqf && queue_family == UINT32_MAX; i++)
        if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) queue_family = i;
    free(qf);
    if (queue_family == UINT32_MAX) die(2, "no graphics queue");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = queue_family,
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
    vkGetDeviceQueue(dev, queue_family, 0, &queue);
}

void device_finish(void) {
    vkDeviceWaitIdle(dev);
    vkDestroyDevice(dev, NULL);
    if (messenger) vkDestroyDebugUtilsMessengerEXT(inst, messenger, NULL);
    vkDestroyInstance(inst, NULL);
}

int main(int argc, char **argv) {
    if (argc < 3) die(1, "usage: vkpipecost build|cache|draws|replay JOB [PHASE]");
    static Job job;
    job_read(argv[2], &job);
    // The cache mode loads modules as it goes (20,000 pipelines do not fit in memory as SPIR-V).
    if (strcmp(argv[1], "cache"))
        for (uint32_t i = 0; i < job.nmodules; i++) module_load(&job, &job.modules[i]);
    device_init();
    out_open();
    int rc = 1;
    if (!strcmp(argv[1], "build") && argc == 4) rc = mode_build(&job, argv[3]);
    else if (!strcmp(argv[1], "cache") && argc == 4) rc = mode_cache(&job, argv[3]);
    else if (!strcmp(argv[1], "draws")) rc = mode_draws(&job);
    else if (!strcmp(argv[1], "replay")) rc = mode_replay(&job);
    else die(1, "unknown mode or missing phase");
    out_close();
    device_finish();
    return rc;
}
