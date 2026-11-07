// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vktensorcost: the replay of spike S8 (architecture D20b and section 6.2), the
// GPU time per frame of the compute chains that hold the game's tensor programs.
//
//   vktensorcost info
//   vktensorcost run JOB
//
// The job file is written by tools/tensorcost/tensor_cost.py, one directive per
// line, paths last on their line, '#' starts a comment. Every number is already
// resolved there (formats, enums, bindings, bytes): this program creates what the
// file names and replays it.
//
//   validation off|on          device SUBSTRING            out PATH
//   frames N  warmup N  batch N          (measured frames, unmeasured frames, frames per submission)
//   mode normal|empty|half               (counter-runs: an empty shader in place of every program,
//                                         or the largest group count of every dispatch halved)
//   heap IMAGES SAMPLERS                 (set 0: arrays of sampled images and of samplers)
//   module PATH                          (SPIR-V, numbered from 0 in file order)
//   buffer ID SIZE FILL [ARG]            FILL = zero | f16 SEED | u32 SEED | word VALUE | file PATH
//                                        (f16: halves in about [-0.5, 0.5]; u32: random words; word: every word VALUE)
//   poke ID OFFSET VALUE                 (a 32-bit word written after the fill)
//   image ID FORMAT W H LAYERS USAGE SEED   (FORMAT is the VkFormat number, USAGE = sampled|storage|both,
//                                            filled with bytes from SEED)
//   imagefill ID WORD                    (every 32-bit word of that image is WORD, in place of the SEED bytes)
//   sampler ID MAG MIN WRAPU WRAPV WRAPW MAXLOD    (VkFilter and VkSamplerAddressMode numbers)
//   heapimage INDEX IMAGE       heapsampler INDEX SAMPLER
//   phase NAME                  (the dispatches that follow, until `end`, are one frame of that phase)
//   dispatch NAME MODULE GX GY GZ
//   push HEX                    (up to 128 bytes of push constants, hex digits, for the last dispatch)
//   bind BINDING ubo|ssbo|simg ID [OFFSET RANGE]   (set 1, push descriptor; RANGE 0 = to the end)
//   end
//   readback ID PATH            (after the last phase, write that buffer to a file)
//
// Layout (architecture D13, ticket 60): set 0 holds the global arrays (binding 0
// sampled images, binding 1 samplers; partially bound, update after bind), set 1 is
// a push-descriptor set of 32 bindings, 128 bytes of push constants. The modules
// the translator emits for compute declare exactly this.
//
// Rows (TSV): phase, frame, what, ms; `what` is `frame` (first timestamp to last) or
// the dispatch's name (its timestamp to the previous one, so launch gaps are inside).
// A summary goes to stdout as one JSON line per phase with p50 and p99 per item.
//
// Exit codes: 0 measured, 1 the job file is wrong or validation reported errors,
// 2 Vulkan lacks something the job needs, 3 a Vulkan call failed. Windows only.
// No game data: modules, constants and sizes are inputs, written by the caller.

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
#define MAX_ITEMS 512
#define MAX_BINDS 32
#define MAX_PHASES 16
#define MAX_DISPATCH 64
#define MAX_MODULES 64
#define PUSH_BYTES 128
#define SET1_BINDINGS 32

static VkInstance inst;
static VkPhysicalDevice gpu;
static VkDevice dev;
static VkQueue queue;
static uint32_t family = UINT32_MAX;
static VkPhysicalDeviceProperties gpu_props;
static VkPhysicalDeviceMemoryProperties mem_props;
static uint32_t subgroup_size;
static unsigned val_errors, val_warnings;
static VkDebugUtilsMessengerEXT messenger;
static int has_push_descriptor;

__attribute__((noreturn, format(printf, 2, 3))) static void die(int code, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "vktensorcost: ");
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

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    _fseeki64(f, 0, SEEK_END);
    long long n = _ftelli64(f);
    _fseeki64(f, 0, SEEK_SET);
    void *p = malloc(n > 0 ? (size_t)n : 1);
    if (!p || (n > 0 && fread(p, 1, (size_t)n, f) != (size_t)n)) {
        free(p);
        fclose(f);
        return NULL;
    }
    fclose(f);
    *size = (size_t)(n > 0 ? n : 0);
    return p;
}

static uint64_t rng_state;
static uint64_t rng_next(void) {
    uint64_t z = (rng_state += 0x9E3779B97F4A7C15ull);
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

// ---- the job ----------------------------------------------------------------------

enum { FILL_ZERO, FILL_F16, FILL_U32, FILL_FILE, FILL_WORD };

typedef struct {
    char id[48];
    VkDeviceSize size;
    int fill;
    uint64_t seed;
    char path[MAX_PATH_LEN];
    uint32_t npoke, poke_off[8], poke_val[8];
    VkBuffer buf;
    VkDeviceMemory mem;
} Buffer;

typedef struct {
    char id[48];
    VkFormat format;
    uint32_t w, h, layers;
    int sampled, storage;
    uint64_t seed;
    int has_word;
    uint32_t word;
    VkImage img;
    VkImageView view;
    VkDeviceMemory mem;
} Image;

typedef struct {
    char id[48];
    uint32_t mag, min, wrap[3];
    float max_lod;
    VkSampler s;
} Sampler;

typedef struct {
    uint32_t binding;
    VkDescriptorType type;
    int resource;     // index into buffers or images
    VkDeviceSize offset, range;
} Bind;

typedef struct {
    char name[64];
    int module;
    uint32_t groups[3];
    uint8_t push[PUSH_BYTES];
    uint32_t npush, nbinds;
    Bind binds[MAX_BINDS];
    VkPipeline pipe;
} Dispatch;

typedef struct {
    char name[64];
    uint32_t nd;
    Dispatch d[MAX_DISPATCH];
} Phase;

typedef struct {
    int validation, mode;     // mode 0 normal, 1 empty, 2 half
    char device[256], out[MAX_PATH_LEN];
    uint32_t frames, warmup, batch, heap_images, heap_samplers;
    uint32_t nmod; char mod[MAX_MODULES][MAX_PATH_LEN];
    uint32_t nbuf; Buffer *buf;
    uint32_t nimg; Image *img;
    uint32_t nsamp; Sampler *samp;
    uint32_t nheap_i, heap_i_idx[MAX_ITEMS], heap_i_res[MAX_ITEMS];
    uint32_t nheap_s, heap_s_idx[MAX_ITEMS], heap_s_res[MAX_ITEMS];
    uint32_t nphase; Phase *phase;
    char readback_id[48], readback_path[MAX_PATH_LEN];
} Job;

static Job job;

static int find_buffer(const char *id) {
    for (uint32_t i = 0; i < job.nbuf; i++) if (!strcmp(job.buf[i].id, id)) return (int)i;
    return -1;
}
static int find_image(const char *id) {
    for (uint32_t i = 0; i < job.nimg; i++) if (!strcmp(job.img[i].id, id)) return (int)i;
    return -1;
}
static int find_sampler(const char *id) {
    for (uint32_t i = 0; i < job.nsamp; i++) if (!strcmp(job.samp[i].id, id)) return (int)i;
    return -1;
}

static void bad(int line, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    fprintf(stderr, "vktensorcost: job line %d: ", line);
    vfprintf(stderr, fmt, ap);
    fprintf(stderr, "\n");
    va_end(ap);
    exit(1);
}

static int split_line(char *s, char **tok, int max) {
    int n = 0;
    while (*s && n < max) {
        while (*s == ' ' || *s == '\t') s++;
        if (!*s) break;
        tok[n++] = s;
        while (*s && *s != ' ' && *s != '\t') s++;
        if (*s) *s++ = 0;
    }
    return n;
}

static uint32_t u32_of(const char *s, int line) {
    char *e;
    unsigned long long v = strtoull(s, &e, 0);
    if (*e || e == s) bad(line, "not a number: %s", s);
    return (uint32_t)v;
}

static void job_read(const char *path) {
    size_t n;
    char *text = read_file(path, &n);
    if (!text) die(1, "cannot read job file %s", path);
    text = realloc(text, n + 1);
    text[n] = 0;
    job.buf = calloc(MAX_ITEMS, sizeof(Buffer));
    job.img = calloc(MAX_ITEMS, sizeof(Image));
    job.samp = calloc(MAX_ITEMS, sizeof(Sampler));
    job.phase = calloc(MAX_PHASES, sizeof(Phase));
    job.frames = 300; job.warmup = 30; job.batch = 16; job.heap_images = 64; job.heap_samplers = 16;
    Phase *ph = NULL;
    Dispatch *cur = NULL;
    int line = 0;
    for (char *p = text; *p;) {
        char *eol = strchr(p, '\n');
        if (eol) *eol = 0;
        char *next = eol ? eol + 1 : p + strlen(p);
        line++;
        size_t len = strlen(p);
        if (len && p[len - 1] == '\r') p[len - 1] = 0;
        char *hash = strchr(p, '#');
        if (hash) *hash = 0;
        char *t[16];
        int nt = split_line(p, t, 16);
        p = next;
        if (!nt) continue;
        const char *d = t[0];
        if (!strcmp(d, "validation") && nt == 2) job.validation = !strcmp(t[1], "on");
        else if (!strcmp(d, "device") && nt == 2) snprintf(job.device, sizeof job.device, "%s", t[1]);
        else if (!strcmp(d, "out") && nt == 2) snprintf(job.out, sizeof job.out, "%s", t[1]);
        else if (!strcmp(d, "frames") && nt == 2) job.frames = u32_of(t[1], line);
        else if (!strcmp(d, "warmup") && nt == 2) job.warmup = u32_of(t[1], line);
        else if (!strcmp(d, "batch") && nt == 2) job.batch = u32_of(t[1], line);
        else if (!strcmp(d, "mode") && nt == 2) {
            job.mode = !strcmp(t[1], "empty") ? 1 : !strcmp(t[1], "half") ? 2 : 0;
            if (job.mode == 0 && strcmp(t[1], "normal")) bad(line, "mode %s", t[1]);
        } else if (!strcmp(d, "heap") && nt == 3) {
            job.heap_images = u32_of(t[1], line);
            job.heap_samplers = u32_of(t[2], line);
        } else if (!strcmp(d, "module") && nt == 2) {
            if (job.nmod >= MAX_MODULES) bad(line, "too many modules");
            snprintf(job.mod[job.nmod++], MAX_PATH_LEN, "%s", t[1]);
        } else if (!strcmp(d, "buffer") && nt >= 4) {
            if (job.nbuf >= MAX_ITEMS) bad(line, "too many buffers");
            Buffer *b = &job.buf[job.nbuf++];
            snprintf(b->id, sizeof b->id, "%s", t[1]);
            b->size = strtoull(t[2], NULL, 0);
            if (!strcmp(t[3], "zero")) b->fill = FILL_ZERO;
            else if (!strcmp(t[3], "f16") && nt == 5) { b->fill = FILL_F16; b->seed = strtoull(t[4], NULL, 0); }
            else if (!strcmp(t[3], "u32") && nt == 5) { b->fill = FILL_U32; b->seed = strtoull(t[4], NULL, 0); }
            else if (!strcmp(t[3], "word") && nt == 5) { b->fill = FILL_WORD; b->seed = strtoull(t[4], NULL, 0); }
            else if (!strcmp(t[3], "file") && nt == 5) { b->fill = FILL_FILE; snprintf(b->path, sizeof b->path, "%s", t[4]); }
            else bad(line, "buffer fill %s", t[3]);
            if (!b->size) bad(line, "empty buffer");
        } else if (!strcmp(d, "poke") && nt == 4) {
            int i = find_buffer(t[1]);
            if (i < 0 || job.buf[i].npoke >= 8) bad(line, "poke: no buffer %s or too many", t[1]);
            job.buf[i].poke_off[job.buf[i].npoke] = u32_of(t[2], line);
            job.buf[i].poke_val[job.buf[i].npoke++] = u32_of(t[3], line);
        } else if (!strcmp(d, "image") && nt == 8) {
            if (job.nimg >= MAX_ITEMS) bad(line, "too many images");
            Image *im = &job.img[job.nimg++];
            snprintf(im->id, sizeof im->id, "%s", t[1]);
            im->format = (VkFormat)u32_of(t[2], line);
            im->w = u32_of(t[3], line); im->h = u32_of(t[4], line); im->layers = u32_of(t[5], line);
            im->sampled = !strcmp(t[6], "sampled") || !strcmp(t[6], "both");
            im->storage = !strcmp(t[6], "storage") || !strcmp(t[6], "both");
            if (!im->sampled && !im->storage) bad(line, "image usage %s", t[6]);
            im->seed = strtoull(t[7], NULL, 0);
            if (!im->w || !im->h || !im->layers) bad(line, "empty image");
        } else if (!strcmp(d, "imagefill") && nt == 3) {
            int i = find_image(t[1]);
            if (i < 0) bad(line, "imagefill: no image %s", t[1]);
            job.img[i].has_word = 1;
            job.img[i].word = u32_of(t[2], line);
        } else if (!strcmp(d, "sampler") && nt == 8) {
            if (job.nsamp >= MAX_ITEMS) bad(line, "too many samplers");
            Sampler *s = &job.samp[job.nsamp++];
            snprintf(s->id, sizeof s->id, "%s", t[1]);
            s->mag = u32_of(t[2], line); s->min = u32_of(t[3], line);
            for (int k = 0; k < 3; k++) s->wrap[k] = u32_of(t[4 + k], line);
            s->max_lod = (float)atof(t[7]);
        } else if (!strcmp(d, "heapimage") && nt == 3) {
            int i = find_image(t[2]);
            if (i < 0 || job.nheap_i >= MAX_ITEMS) bad(line, "heapimage: no image %s", t[2]);
            job.heap_i_idx[job.nheap_i] = u32_of(t[1], line);
            job.heap_i_res[job.nheap_i++] = (uint32_t)i;
        } else if (!strcmp(d, "heapsampler") && nt == 3) {
            int i = find_sampler(t[2]);
            if (i < 0 || job.nheap_s >= MAX_ITEMS) bad(line, "heapsampler: no sampler %s", t[2]);
            job.heap_s_idx[job.nheap_s] = u32_of(t[1], line);
            job.heap_s_res[job.nheap_s++] = (uint32_t)i;
        } else if (!strcmp(d, "phase") && nt == 2) {
            if (ph || job.nphase >= MAX_PHASES) bad(line, "nested or too many phases");
            ph = &job.phase[job.nphase++];
            snprintf(ph->name, sizeof ph->name, "%s", t[1]);
        } else if (!strcmp(d, "dispatch") && nt == 6) {
            if (!ph || ph->nd >= MAX_DISPATCH) bad(line, "dispatch outside a phase or too many");
            cur = &ph->d[ph->nd++];
            snprintf(cur->name, sizeof cur->name, "%s", t[1]);
            cur->module = (int)u32_of(t[2], line);
            if ((uint32_t)cur->module >= job.nmod) bad(line, "no module %d", cur->module);
            for (int k = 0; k < 3; k++) cur->groups[k] = u32_of(t[3 + k], line);
        } else if (!strcmp(d, "push") && nt == 2) {
            if (!cur) bad(line, "push outside a dispatch");
            size_t hl = strlen(t[1]);
            if (hl % 2 || hl / 2 > PUSH_BYTES) bad(line, "push of %zu hex digits", hl);
            for (size_t k = 0; k < hl / 2; k++) {
                char b[3] = {t[1][2 * k], t[1][2 * k + 1], 0};
                cur->push[k] = (uint8_t)strtoul(b, NULL, 16);
            }
            cur->npush = (uint32_t)(hl / 2);
        } else if (!strcmp(d, "bind") && (nt == 4 || nt == 6)) {
            if (!cur || cur->nbinds >= MAX_BINDS) bad(line, "bind outside a dispatch or too many");
            Bind *b = &cur->binds[cur->nbinds++];
            b->binding = u32_of(t[1], line);
            if (b->binding >= SET1_BINDINGS) bad(line, "binding %u", b->binding);
            if (!strcmp(t[2], "simg")) {
                b->type = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
                b->resource = find_image(t[3]);
            } else {
                b->type = !strcmp(t[2], "ubo") ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                b->resource = find_buffer(t[3]);
            }
            if (strcmp(t[2], "ubo") && strcmp(t[2], "ssbo") && strcmp(t[2], "simg")) bad(line, "bind type %s", t[2]);
            if (b->resource < 0) bad(line, "bind: no resource %s", t[3]);
            if (nt == 6) { b->offset = strtoull(t[4], NULL, 0); b->range = strtoull(t[5], NULL, 0); }
        } else if (!strcmp(d, "end") && nt == 1) {
            if (!ph) bad(line, "end outside a phase");
            ph = NULL; cur = NULL;
        } else if (!strcmp(d, "readback") && nt == 3) {
            snprintf(job.readback_id, sizeof job.readback_id, "%s", t[1]);
            snprintf(job.readback_path, sizeof job.readback_path, "%s", t[2]);
        } else bad(line, "unknown or malformed directive %s", d);
    }
    free(text);
    if (ph) die(1, "job file: phase %s has no `end`", ph->name);
    if (!job.nphase) die(1, "job file has no phase");
    if (!job.nmod) die(1, "job file has no module");
    if (job.frames == 0 || job.batch == 0 || job.batch > 64) die(1, "frames and batch must be positive (batch at most 64)");
}

// ---- Vulkan setup (vkhmmacost's, trimmed) ---------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT types,
                                                 const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)user;
    if (types == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) return VK_FALSE;
    const char *kind = "info";
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { val_errors++; kind = "error"; }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { val_warnings++; kind = "warning"; }
    if (val_errors + val_warnings <= 50)
        fprintf(stderr, "vktensorcost: validation %s: %s\n", kind, data->pMessage ? data->pMessage : "(no text)");
    return VK_FALSE;
}

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
} Features;

static void features_chain(Features *f) {
    memset(f, 0, sizeof *f);
    f->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f->f2.pNext = &f->v11;
    f->v11.pNext = &f->v12;
    f->v12.pNext = &f->v13;
}

static int has_ext(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static void instance_init(int validation) {
    if (volkInitialize() != VK_SUCCESS) die(2, "no Vulkan loader");
    const char *layers[1] = {"VK_LAYER_KHRONOS_validation"};
    const char *iexts[1] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkDebugUtilsMessengerCreateInfoEXT mci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT,
        .pfnUserCallback = on_message};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vktensorcost",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pNext = validation ? &mci : NULL,
                                .pApplicationInfo = &app, .enabledLayerCount = validation ? 1 : 0,
                                .ppEnabledLayerNames = layers, .enabledExtensionCount = validation ? 1 : 0,
                                .ppEnabledExtensionNames = iexts};
    VK_CHECK(vkCreateInstance(&ici, NULL, &inst));
    volkLoadInstance(inst);
    if (validation) VK_CHECK(vkCreateDebugUtilsMessengerEXT(inst, &mci, NULL, &messenger));

    uint32_t ngpu = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ngpu, NULL));
    if (!ngpu) die(2, "no Vulkan device");
    VkPhysicalDevice *gpus = calloc(ngpu, sizeof *gpus);
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ngpu, gpus));
    for (int pass = 0; pass < 2 && !gpu; pass++)
        for (uint32_t i = 0; i < ngpu && !gpu; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(gpus[i], &p);
            if (job.device[0] && !strstr(p.deviceName, job.device)) continue;
            if (pass == 0 && p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            gpu = gpus[i];
            gpu_props = p;
        }
    free(gpus);
    if (!gpu) die(2, "no device matches '%s'", job.device);
    VkPhysicalDeviceVulkan11Properties p11 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &p11};
    vkGetPhysicalDeviceProperties2(gpu, &p2);
    subgroup_size = p11.subgroupSize;
    vkGetPhysicalDeviceMemoryProperties(gpu, &mem_props);

    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, NULL);
    VkExtensionProperties *dexts = calloc(next ? next : 1, sizeof *dexts);
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, dexts);
    has_push_descriptor = has_ext(dexts, next, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME);
    free(dexts);
    if (!has_push_descriptor) die(2, "VK_KHR_push_descriptor is missing");
}

static void device_init(void) {
    Features have, want;
    features_chain(&have);
    vkGetPhysicalDeviceFeatures2(gpu, &have.f2);
    features_chain(&want);
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

    uint32_t nqf = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, NULL);
    VkQueueFamilyProperties *qf = calloc(nqf, sizeof *qf);
    vkGetPhysicalDeviceQueueFamilyProperties(gpu, &nqf, qf);
    for (uint32_t i = 0; i < nqf && family == UINT32_MAX; i++)
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].timestampValidBits) family = i;
    free(qf);
    if (family == UINT32_MAX) die(2, "no compute queue with timestamps");
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qci = {.sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .queueFamilyIndex = family,
                                   .queueCount = 1, .pQueuePriorities = &prio};
    const char *dext[1] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &want.f2, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = dext};
    VK_CHECK(vkCreateDevice(gpu, &dci, NULL, &dev));
    volkLoadDevice(dev);
    vkGetDeviceQueue(dev, family, 0, &queue);
}

static int cmd_info(void) {
    instance_init(0);
    printf("{\"device\":\"%s\",\"driver\":%u,\"subgroup_size\":%u,\"api\":\"%u.%u.%u\",\"timestamp_period_ns\":%.4f}\n",
           gpu_props.deviceName, gpu_props.driverVersion, subgroup_size,
           VK_API_VERSION_MAJOR(gpu_props.apiVersion), VK_API_VERSION_MINOR(gpu_props.apiVersion),
           VK_API_VERSION_PATCH(gpu_props.apiVersion), (double)gpu_props.limits.timestampPeriod);
    return 0;
}

// ---- memory, resources ---------------------------------------------------------------------

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mem_props.memoryTypes[i].propertyFlags & want) == want) return i;
    die(2, "no memory type for flags 0x%x", (unsigned)want);
}

static void buffer_make(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags flags, VkBuffer *buf,
                        VkDeviceMemory *mem) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VK_CHECK(vkCreateBuffer(dev, &bi, NULL, buf));
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(dev, *buf, &mr);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                               .memoryTypeIndex = memory_type(mr.memoryTypeBits, flags)};
    VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, mem));
    VK_CHECK(vkBindBufferMemory(dev, *buf, *mem, 0));
}

static VkCommandPool cmd_pool;
static VkBuffer stage_buf;
static VkDeviceMemory stage_mem;
static void *stage_ptr;
static VkDeviceSize stage_size;

static VkCommandBuffer one_shot_begin(void) {
    VkCommandBufferAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                      .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev, &alloc, &cb));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

static void one_shot_end(VkCommandBuffer cb) {
    VK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    vkFreeCommandBuffers(dev, cmd_pool, 1, &cb);
}

static void stage_reserve(VkDeviceSize need) {
    if (need <= stage_size) return;
    if (stage_buf) {
        vkUnmapMemory(dev, stage_mem);
        vkDestroyBuffer(dev, stage_buf, NULL);
        vkFreeMemory(dev, stage_mem, NULL);
    }
    stage_size = need;
    buffer_make(stage_size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, &stage_buf, &stage_mem);
    VK_CHECK(vkMapMemory(dev, stage_mem, 0, VK_WHOLE_SIZE, 0, &stage_ptr));
}

static uint16_t f16_small(uint64_t r) {
    // a half in about [-0.5, 0.5]: sign, exponent 0x0D..0x0E (0.25 to 1 before the mantissa), random mantissa
    uint16_t sign = (uint16_t)((r >> 40) & 1) << 15;
    uint16_t exp = (uint16_t)(0x0C + ((r >> 41) & 1));
    return (uint16_t)(sign | (exp << 10) | (uint16_t)(r & 0x3FF));
}

static void buffer_fill(Buffer *b, uint8_t *dst) {
    memset(dst, 0, (size_t)b->size);
    rng_state = b->seed * 2654435761ull + 12345;
    switch (b->fill) {
    case FILL_F16:
        for (VkDeviceSize i = 0; i + 2 <= b->size; i += 2) *(uint16_t *)(dst + i) = f16_small(rng_next());
        break;
    case FILL_U32:
        for (VkDeviceSize i = 0; i + 4 <= b->size; i += 4) *(uint32_t *)(dst + i) = (uint32_t)rng_next();
        break;
    case FILL_WORD:
        for (VkDeviceSize i = 0; i + 4 <= b->size; i += 4) *(uint32_t *)(dst + i) = (uint32_t)b->seed;
        break;
    case FILL_FILE: {
        size_t n;
        void *data = read_file(b->path, &n);
        if (!data) die(1, "cannot read %s", b->path);
        memcpy(dst, data, n < b->size ? n : (size_t)b->size);
        free(data);
        break;
    }
    default: break;
    }
    for (uint32_t i = 0; i < b->npoke; i++)
        if (b->poke_off[i] + 4 <= b->size) memcpy(dst + b->poke_off[i], &b->poke_val[i], 4);
}

static void buffers_create(void) {
    for (uint32_t i = 0; i < job.nbuf; i++) {
        Buffer *b = &job.buf[i];
        buffer_make(b->size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, &b->buf, &b->mem);
        stage_reserve(b->size);
        buffer_fill(b, (uint8_t *)stage_ptr);
        VkCommandBuffer cb = one_shot_begin();
        VkBufferCopy c = {.size = b->size};
        vkCmdCopyBuffer(cb, stage_buf, b->buf, 1, &c);
        one_shot_end(cb);
    }
}

static uint32_t format_texel_bytes(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UNORM: case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SNORM: case VK_FORMAT_R8_SINT: return 1;
    case VK_FORMAT_R8G8_UNORM: case VK_FORMAT_R16_UNORM: case VK_FORMAT_R16_SFLOAT: case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R8G8_UINT: return 2;
    case VK_FORMAT_R16G16B16A16_SFLOAT: case VK_FORMAT_R32G32_SFLOAT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R16G16B16A16_UNORM: case VK_FORMAT_R16G16B16A16_UINT: return 8;
    case VK_FORMAT_R32G32B32A32_SFLOAT: case VK_FORMAT_R32G32B32A32_UINT: return 16;
    default: return 4;     // the 32-bit packed and 4 x 8-bit formats
    }
}

static void images_create(void) {
    for (uint32_t i = 0; i < job.nimg; i++) {
        Image *im = &job.img[i];
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                                .format = im->format, .extent = {im->w, im->h, 1}, .mipLevels = 1, .arrayLayers = im->layers,
                                .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                         (im->sampled ? VK_IMAGE_USAGE_SAMPLED_BIT : 0) |
                                         (im->storage ? VK_IMAGE_USAGE_STORAGE_BIT : 0),
                                .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        VkResult r = vkCreateImage(dev, &ii, NULL, &im->img);
        if (r != VK_SUCCESS) die(2, "image %s (format %d, %ux%u): vkCreateImage returned %d", im->id, (int)im->format, im->w, im->h, (int)r);
        VkMemoryRequirements mr;
        vkGetImageMemoryRequirements(dev, im->img, &mr);
        VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                   .memoryTypeIndex = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, &im->mem));
        VK_CHECK(vkBindImageMemory(dev, im->img, im->mem, 0));
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im->img,
                                    .viewType = im->layers > 1 ? VK_IMAGE_VIEW_TYPE_2D_ARRAY : VK_IMAGE_VIEW_TYPE_2D,
                                    .format = im->format,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, im->layers}};
        VK_CHECK(vkCreateImageView(dev, &vi, NULL, &im->view));
        VkDeviceSize bytes = (VkDeviceSize)im->w * im->h * im->layers * format_texel_bytes(im->format);
        stage_reserve(bytes);
        rng_state = im->seed * 2654435761ull + 777;
        for (VkDeviceSize k = 0; k + 8 <= bytes; k += 8)
            *(uint64_t *)((uint8_t *)stage_ptr + k) = im->has_word ? ((uint64_t)im->word << 32 | im->word) : rng_next();
        VkCommandBuffer cb = one_shot_begin();
        VkImageMemoryBarrier to_dst = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER, .dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT,
                                       .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                       .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                       .image = im->img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, im->layers}};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 0, NULL, 0, NULL, 1, &to_dst);
        VkBufferImageCopy c = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, im->layers}, .imageExtent = {im->w, im->h, 1}};
        vkCmdCopyBufferToImage(cb, stage_buf, im->img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &c);
        VkImageMemoryBarrier to_gen = to_dst;
        to_gen.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
        to_gen.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        to_gen.oldLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL;
        to_gen.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, NULL, 0, NULL, 1, &to_gen);
        one_shot_end(cb);
    }
    for (uint32_t i = 0; i < job.nsamp; i++) {
        Sampler *s = &job.samp[i];
        VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = (VkFilter)s->mag,
                                  .minFilter = (VkFilter)s->min, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                  .addressModeU = (VkSamplerAddressMode)s->wrap[0], .addressModeV = (VkSamplerAddressMode)s->wrap[1],
                                  .addressModeW = (VkSamplerAddressMode)s->wrap[2], .maxLod = s->max_lod,
                                  .borderColor = VK_BORDER_COLOR_FLOAT_TRANSPARENT_BLACK};
        VK_CHECK(vkCreateSampler(dev, &si, NULL, &s->s));
    }
}

// ---- layouts, pipelines ----------------------------------------------------------------------

static VkDescriptorSetLayout set0_layout, set1_layout;
static VkPipelineLayout pipe_layout;
static VkDescriptorPool pool0;
static VkDescriptorSet set0;

static void layouts_create(void) {
    VkDescriptorBindingFlags bf[2] = {VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
                                      VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT};
    VkDescriptorSetLayoutBindingFlagsCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                      .bindingCount = 2, .pBindingFlags = bf};
    VkDescriptorSetLayoutBinding b0[2] = {
        {.binding = 0, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .descriptorCount = job.heap_images, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT},
        {.binding = 1, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .descriptorCount = job.heap_samplers, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT}};
    VkDescriptorSetLayoutCreateInfo l0 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &fi,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                          .bindingCount = 2, .pBindings = b0};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &l0, NULL, &set0_layout));

    VkDescriptorSetLayoutBinding b1[SET1_BINDINGS];
    for (uint32_t i = 0; i < SET1_BINDINGS; i++) {
        b1[i] = (VkDescriptorSetLayoutBinding){.binding = i, .descriptorCount = 1, .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT};
        b1[i].descriptorType = i <= 8 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : i <= 24 ? VK_DESCRIPTOR_TYPE_STORAGE_BUFFER
                                                                                    : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    }
    VkDescriptorSetLayoutCreateInfo l1 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                          .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                          .bindingCount = SET1_BINDINGS, .pBindings = b1};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &l1, NULL, &set1_layout));

    VkDescriptorSetLayout sl[2] = {set0_layout, set1_layout};
    VkPushConstantRange pc = {VK_SHADER_STAGE_COMPUTE_BIT, 0, PUSH_BYTES};
    VkPipelineLayoutCreateInfo pl = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 2,
                                     .pSetLayouts = sl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pc};
    VK_CHECK(vkCreatePipelineLayout(dev, &pl, NULL, &pipe_layout));

    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, job.heap_images}, {VK_DESCRIPTOR_TYPE_SAMPLER, job.heap_samplers}};
    VkDescriptorPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                     .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1,
                                     .poolSizeCount = 2, .pPoolSizes = ps};
    VK_CHECK(vkCreateDescriptorPool(dev, &pi, NULL, &pool0));
    VkDescriptorSetAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = pool0,
                                      .descriptorSetCount = 1, .pSetLayouts = &set0_layout};
    VK_CHECK(vkAllocateDescriptorSets(dev, &alloc, &set0));

    for (uint32_t i = 0; i < job.nheap_i; i++) {
        if (job.heap_i_idx[i] >= job.heap_images) die(1, "heapimage index %u past the heap", job.heap_i_idx[i]);
        VkDescriptorImageInfo ii = {.imageView = job.img[job.heap_i_res[i]].view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set0, .dstBinding = 0,
                                  .dstArrayElement = job.heap_i_idx[i], .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &ii};
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
    }
    for (uint32_t i = 0; i < job.nheap_s; i++) {
        if (job.heap_s_idx[i] >= job.heap_samplers) die(1, "heapsampler index %u past the heap", job.heap_s_idx[i]);
        VkDescriptorImageInfo ii = {.sampler = job.samp[job.heap_s_res[i]].s};
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = set0, .dstBinding = 1,
                                  .dstArrayElement = job.heap_s_idx[i], .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &ii};
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
    }
}

// void main() {} with local size 32, for the empty counter-run
static const uint32_t empty_spv[] = {
    0x07230203, 0x00010300, 0, 5, 0,
    0x00020011, 1, 0x0003000E, 0, 1, 0x0005000F, 5, 1, 0x6E69616D, 0, 0x00060010, 1, 17, 32, 1, 1,
    0x00020013, 2, 0x00030021, 3, 2, 0x00050036, 2, 1, 0, 3, 0x000200F8, 4, 0x000100FD, 0x00010038};

static VkShaderModule module_make(const uint32_t *words, size_t bytes) {
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = bytes, .pCode = words};
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(dev, &mi, NULL, &m));
    return m;
}

static VkPipeline pipeline_make(VkShaderModule m) {
    VkComputePipelineCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO, .layout = pipe_layout,
                                      .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                                                .stage = VK_SHADER_STAGE_COMPUTE_BIT, .module = m, .pName = "main"}};
    VkPipeline p;
    VkResult r = vkCreateComputePipelines(dev, VK_NULL_HANDLE, 1, &ci, NULL, &p);
    if (r != VK_SUCCESS) die(3, "vkCreateComputePipelines returned %d", (int)r);
    return p;
}

static void pipelines_create(void) {
    VkShaderModule mods[MAX_MODULES];
    VkPipeline pipes[MAX_MODULES] = {0};
    VkShaderModule empty = job.mode == 1 ? module_make(empty_spv, sizeof empty_spv) : VK_NULL_HANDLE;
    VkPipeline empty_pipe = job.mode == 1 ? pipeline_make(empty) : VK_NULL_HANDLE;
    for (uint32_t i = 0; i < job.nmod; i++) {
        size_t n;
        void *data = read_file(job.mod[i], &n);
        if (!data || n % 4) die(1, "cannot read module %s", job.mod[i]);
        mods[i] = module_make(data, n);
        free(data);
        if (job.mode != 1) pipes[i] = pipeline_make(mods[i]);
    }
    for (uint32_t p = 0; p < job.nphase; p++)
        for (uint32_t d = 0; d < job.phase[p].nd; d++)
            job.phase[p].d[d].pipe = job.mode == 1 ? empty_pipe : pipes[job.phase[p].d[d].module];
}

// ---- the replay ----------------------------------------------------------------------------------

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double percentile(double *v, uint32_t n, double q) {
    if (!n) return 0;
    qsort(v, n, sizeof *v, cmp_double);
    double pos = q * (n - 1);
    uint32_t lo = (uint32_t)pos;
    uint32_t hi = lo + 1 < n ? lo + 1 : lo;
    return v[lo] + (v[hi] - v[lo]) * (pos - lo);
}

static uint32_t effective_groups(const Dispatch *d, uint32_t k) {
    uint32_t g = d->groups[k];
    if (job.mode != 2) return g;
    uint32_t big = 0;
    for (uint32_t i = 1; i < 3; i++) if (d->groups[i] > d->groups[big]) big = i;
    return k == big ? (g + 1) / 2 : g;
}

static void record_frame(VkCommandBuffer cb, Phase *ph, VkQueryPool qp, uint32_t q0, VkPipeline *last) {
    vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, qp, q0);
    for (uint32_t i = 0; i < ph->nd; i++) {
        Dispatch *d = &ph->d[i];
        if (d->pipe != *last) { vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, d->pipe); *last = d->pipe; }
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 0, 1, &set0, 0, NULL);
        VkWriteDescriptorSet w[MAX_BINDS];
        VkDescriptorBufferInfo bi[MAX_BINDS];
        VkDescriptorImageInfo ii[MAX_BINDS];
        for (uint32_t k = 0; k < d->nbinds; k++) {
            Bind *b = &d->binds[k];
            w[k] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = b->binding,
                                          .descriptorCount = 1, .descriptorType = b->type};
            if (b->type == VK_DESCRIPTOR_TYPE_STORAGE_IMAGE) {
                ii[k] = (VkDescriptorImageInfo){.imageView = job.img[b->resource].view, .imageLayout = VK_IMAGE_LAYOUT_GENERAL};
                w[k].pImageInfo = &ii[k];
            } else {
                VkDeviceSize size = job.buf[b->resource].size;
                bi[k] = (VkDescriptorBufferInfo){.buffer = job.buf[b->resource].buf, .offset = b->offset,
                                                 .range = b->range ? b->range : size - b->offset};
                w[k].pBufferInfo = &bi[k];
            }
        }
        if (d->nbinds) vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe_layout, 1, d->nbinds, w);
        uint8_t push[PUSH_BYTES];
        memcpy(push, d->push, PUSH_BYTES);
        vkCmdPushConstants(cb, pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, PUSH_BYTES, push);
        vkCmdDispatch(cb, effective_groups(d, 0), effective_groups(d, 1), effective_groups(d, 2));
        VkMemoryBarrier mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER,
                              .srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                              .dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT};
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, NULL, 0, NULL);
        vkCmdWriteTimestamp(cb, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, qp, q0 + 1 + i);
    }
}

static void run_phase(Phase *ph, FILE *rows) {
    uint32_t per = ph->nd + 1, batch = job.batch;
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = per * batch};
    VkQueryPool qp;
    VK_CHECK(vkCreateQueryPool(dev, &qi, NULL, &qp));
    double period_ms = (double)gpu_props.limits.timestampPeriod * 1e-6;
    uint32_t total = job.warmup + job.frames;
    uint32_t nmeas = job.frames;
    double *frame_ms = calloc(nmeas, sizeof(double));
    double *disp_ms = calloc((size_t)nmeas * ph->nd, sizeof(double));
    uint64_t *ts = calloc(per * batch, sizeof(uint64_t));
    VkFence fence;
    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(dev, &fci, NULL, &fence));
    uint32_t done = 0;
    while (done < total) {
        uint32_t n = total - done < batch ? total - done : batch;
        VkCommandBufferAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = cmd_pool,
                                          .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VkCommandBuffer cb;
        VK_CHECK(vkAllocateCommandBuffers(dev, &alloc, &cb));
        VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                       .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
        VK_CHECK(vkBeginCommandBuffer(cb, &bi));
        vkCmdResetQueryPool(cb, qp, 0, per * batch);
        VkPipeline last = VK_NULL_HANDLE;
        for (uint32_t f = 0; f < n; f++) record_frame(cb, ph, qp, f * per, &last);
        VK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
        VK_CHECK(vkResetFences(dev, 1, &fence));
        VK_CHECK(vkQueueSubmit(queue, 1, &si, fence));
        VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 60ull * 1000000000ull));
        VK_CHECK(vkGetQueryPoolResults(dev, qp, 0, per * n, per * n * sizeof(uint64_t), ts, sizeof(uint64_t),
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        for (uint32_t f = 0; f < n; f++) {
            uint32_t g = done + f;
            if (g < job.warmup) continue;
            uint32_t m = g - job.warmup;
            const uint64_t *t = ts + f * per;
            frame_ms[m] = (double)(t[per - 1] - t[0]) * period_ms;
            for (uint32_t i = 0; i < ph->nd; i++) disp_ms[(size_t)m * ph->nd + i] = (double)(t[i + 1] - t[i]) * period_ms;
        }
        vkFreeCommandBuffers(dev, cmd_pool, 1, &cb);
        done += n;
    }
    for (uint32_t m = 0; m < nmeas; m++) {
        if (rows) fprintf(rows, "%s\t%u\tframe\t%.6f\n", ph->name, m, frame_ms[m]);
        for (uint32_t i = 0; i < ph->nd && rows; i++)
            fprintf(rows, "%s\t%u\t%s\t%.6f\n", ph->name, m, ph->d[i].name, disp_ms[(size_t)m * ph->nd + i]);
    }
    double mean = 0;
    for (uint32_t m = 0; m < nmeas; m++) mean += frame_ms[m];
    mean /= nmeas;
    double fmax = 0;
    for (uint32_t m = 0; m < nmeas; m++) if (frame_ms[m] > fmax) fmax = frame_ms[m];
    printf("{\"phase\":\"%s\",\"frames\":%u,\"mode\":\"%s\",\"mean_ms\":%.6f,\"max_ms\":%.6f,\"dispatches\":[", ph->name, nmeas,
           job.mode == 1 ? "empty" : job.mode == 2 ? "half" : "normal", mean, fmax);
    double *col = calloc(nmeas, sizeof(double));
    for (uint32_t i = 0; i < ph->nd; i++) {
        for (uint32_t m = 0; m < nmeas; m++) col[m] = disp_ms[(size_t)m * ph->nd + i];
        double p50 = percentile(col, nmeas, 0.5);
        double p99 = percentile(col, nmeas, 0.99);
        printf("%s{\"name\":\"%s\",\"p50_ms\":%.6f,\"p99_ms\":%.6f}", i ? "," : "", ph->d[i].name, p50, p99);
    }
    double fp50 = percentile(frame_ms, nmeas, 0.5), fp99 = percentile(frame_ms, nmeas, 0.99);
    printf("],\"frame_p50_ms\":%.6f,\"frame_p99_ms\":%.6f}\n", fp50, fp99);
    free(col); free(frame_ms); free(disp_ms); free(ts);
    vkDestroyFence(dev, fence, NULL);
    vkDestroyQueryPool(dev, qp, NULL);
}

static void readback(void) {
    if (!job.readback_id[0]) return;
    int i = find_buffer(job.readback_id);
    if (i < 0) die(1, "readback: no buffer %s", job.readback_id);
    Buffer *b = &job.buf[i];
    stage_reserve(b->size);
    VkCommandBuffer cb = one_shot_begin();
    VkBufferCopy c = {.size = b->size};
    vkCmdCopyBuffer(cb, b->buf, stage_buf, 1, &c);
    one_shot_end(cb);
    FILE *f = fopen(job.readback_path, "wb");
    if (!f || fwrite(stage_ptr, 1, (size_t)b->size, f) != b->size) die(1, "cannot write %s", job.readback_path);
    fclose(f);
}

static int cmd_run(const char *path) {
    job_read(path);
    instance_init(job.validation);
    device_init();
    VkCommandPoolCreateInfo pi = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family,
                                  .flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT};
    VK_CHECK(vkCreateCommandPool(dev, &pi, NULL, &cmd_pool));
    buffers_create();
    images_create();
    layouts_create();
    pipelines_create();
    FILE *rows = job.out[0] ? fopen(job.out, "wb") : NULL;
    if (job.out[0] && !rows) die(1, "cannot write %s", job.out);
    if (rows) fprintf(rows, "phase\tframe\twhat\tms\n");
    for (uint32_t p = 0; p < job.nphase; p++) run_phase(&job.phase[p], rows);
    if (rows) fclose(rows);
    readback();
    VK_CHECK(vkDeviceWaitIdle(dev));
    fprintf(stderr, "vktensorcost: validation errors %u, warnings %u\n", val_errors, val_warnings);
    return val_errors ? 1 : 0;
}

int main(int argc, char **argv) {
    if (argc == 2 && !strcmp(argv[1], "info")) return cmd_info();
    if (argc == 3 && !strcmp(argv[1], "run")) return cmd_run(argv[2]);
    fprintf(stderr, "usage: vktensorcost info | run JOB\n");
    return 1;
}
