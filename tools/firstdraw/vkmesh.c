// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkmesh: one task, mesh and fragment draw through Vulkan, for ticket 72 (architecture D13 and D16).
//
//   vkmesh <draw.txt>
//
// The sibling of vkdraw for the mesh path: it builds the pipeline from graphics pipeline libraries (the pre-rasterization
// library holds the task and mesh stages, created with required subgroup size 32 and full subgroups; a fragment shader
// library and a fragment output library; fast-linked) in the descriptor layout `uni` (xlate/host/UniLayout.cs): set 0 the
// global update-after-bind arrays, set 1 the push descriptors of the mesh and fragment stages, set 2 the task stage's own
// descriptors, 128 bytes of push constants. It draws one task and mesh draw into one R32G32B32A32_SFLOAT target and writes
// the target back as raw little-endian floats, row by row. Validation messages are counted through a debug messenger; the
// counts and the device go to stdout as one JSON line.
//
// The first task (runtime/include/oot2/mesh_abi.h) goes in push constant words 6 and 7 as the device address of one 32-bit
// word, for both draw forms. A direct draw points it at a word of a small host-visible buffer (word 0 is zero for ever, word
// 1 holds the first task); an indirect draw first copies its records (16 bytes each: three group counts, then the first
// task) from a staging buffer to a device buffer, in the same command buffer, with a barrier to the indirect read and to
// the stages' shader reads, and points the address at the word of the record it draws.
//
// The draw file is text, one directive per line, paths last on their line:
//   ts PATH (optional)              ms PATH                  fs PATH               out PATH
//   size W H                        clear R G B A
//   uniform BINDING STAGES PATH     storage BINDING STAGES PATH      (set 1, pushed; STAGES from t, m, f)
//   taskuniform BINDING PATH        taskstorage BINDING PATH         (set 2, the task stage's)
//   global IMAGES SAMPLERS          (lengths of the set 0 arrays; default 33025 4096)
//   push OFFSET VALUE               (a 32-bit word of the push constants; words 6 to 8 are the draw's)
//   feature NAME                    subgroup OPERATION
//   validation off|on|gpu|sync      device SUBSTRING
//   draw direct COUNT FIRST         draw indirect INDEX PATH         (the draw: PATH holds the records)
//   flag NAME                       nobarrier (leave out the barrier after the records' copy), ignorefirst (a direct
//                                   draw points at the zero word whatever its first task), nopin (no required subgroup
//                                   size, no full subgroups), vertexlibrary (also link an empty vertex input library)
// STAGES t, m or f; NAME of a feature is a Vulkan feature member (shaderInt64, bufferDeviceAddress, ...); OPERATION a
// subgroup operation (basic, vote, arithmetic, ballot, shuffle, relative, clustered, quad).
//
// Exit codes: 0 drawn (validation counts are in the output), 1 the draw file is wrong, 2 Vulkan is missing or lacks
// something the draw needs, 3 a Vulkan call failed.
//
// No game data: the modules and blobs are inputs, written by the caller.

#define VK_NO_PROTOTYPES
#include "volk.h"

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_PATH_LEN 1024
#define MAX_BINDINGS 32
#define MAX_FEATURES 64
#define PUSH_BYTES 128
#define FIRST_TASK_WORD 6      // runtime/include/oot2/mesh_abi.h
#define RECORD_BYTES 16        // MESH_INDIRECT_RECORD_BYTES
#define COMMAND_BYTES 12       // MESH_INDIRECT_COMMAND_BYTES
#define FIRST_OFFSET 12        // MESH_INDIRECT_FIRST_OFFSET
#define MESH_STAGES (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT | VK_SHADER_STAGE_FRAGMENT_BIT)

typedef enum { BIND_UNIFORM, BIND_STORAGE } BindKind;

typedef struct {
    BindKind kind;
    uint32_t binding;
    VkShaderStageFlags stages;
    char path[MAX_PATH_LEN];
} Binding;

typedef struct {
    char ts[MAX_PATH_LEN], ms[MAX_PATH_LEN], fs[MAX_PATH_LEN], out[MAX_PATH_LEN];
    uint32_t width, height;
    float clear[4];
    Binding pushed[MAX_BINDINGS];
    uint32_t npushed;
    Binding task[MAX_BINDINGS];
    uint32_t ntask;
    uint32_t global_images, global_samplers;
    uint32_t push[PUSH_BYTES / 4];
    char features[MAX_FEATURES][64];
    uint32_t nfeatures;
    VkSubgroupFeatureFlags subgroup_ops;
    int validation; // 0 off, 1 core, 2 core + GPU-assisted, 3 core + synchronization
    char device_filter[256];
    int indirect;           // 0 no draw yet, 1 direct, 2 indirect
    uint32_t count, first, index;
    char records[MAX_PATH_LEN];
    int nobarrier, ignorefirst, nopin, vertexlibrary;
} Config;

static Config cfg = {.validation = 1, .global_images = 33025, .global_samplers = 4096};
static unsigned val_errors, val_warnings, loader_messages;

static void die(int code, const char *what) {
    fprintf(stderr, "vkmesh: %s\n", what);
    exit(code);
}

#define VK_CHECK(call)                                                          \
    do {                                                                        \
        VkResult r_ = (call);                                                   \
        if (r_ != VK_SUCCESS) {                                                 \
            fprintf(stderr, "vkmesh: %s failed: %d\n", #call, (int)r_);         \
            exit(3);                                                            \
        }                                                                       \
    } while (0)

// ---- the draw file ----------------------------------------------------------

static char *rest_after(char *s, int fields) {
    // The text after the first `fields` space-separated fields, for a path.
    for (int i = 0; i < fields; i++) {
        while (*s == ' ' || *s == '\t') s++;
        while (*s && *s != ' ' && *s != '\t') s++;
    }
    while (*s == ' ' || *s == '\t') s++;
    return s;
}

static void copy_path(char *dst, const char *src, const char *line) {
    size_t n = strlen(src);
    if (n == 0 || n >= MAX_PATH_LEN) {
        fprintf(stderr, "vkmesh: no path, or a path too long, in: %s\n", line);
        exit(1);
    }
    memcpy(dst, src, n + 1);
}

static VkShaderStageFlags parse_stages(const char *s, const char *line) {
    VkShaderStageFlags f = 0;
    for (; *s; s++) {
        if (*s == 't') f |= VK_SHADER_STAGE_TASK_BIT_EXT;
        else if (*s == 'm') f |= VK_SHADER_STAGE_MESH_BIT_EXT;
        else if (*s == 'f') f |= VK_SHADER_STAGE_FRAGMENT_BIT;
        else { fprintf(stderr, "vkmesh: stages are t, m and f: %s\n", line); exit(1); }
    }
    if (!f) { fprintf(stderr, "vkmesh: no stage in: %s\n", line); exit(1); }
    return f;
}

static void add_binding(Binding *list, uint32_t *n, BindKind kind, uint32_t binding, VkShaderStageFlags stages,
                        const char *path, const char *line) {
    if (*n == MAX_BINDINGS) die(1, "too many bindings");
    for (uint32_t j = 0; j < *n; j++)
        if (list[j].binding == binding) die(1, "a binding appears twice");
    Binding *bd = &list[(*n)++];
    bd->kind = kind;
    bd->binding = binding;
    bd->stages = stages;
    copy_path(bd->path, path, line);
}

static void read_config(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) die(1, "cannot open the draw file");
    char line[2048];
    unsigned lineno = 0;
    while (fgets(line, sizeof line, f)) {
        lineno++;
        size_t n = strlen(line);
        while (n && (line[n - 1] == '\n' || line[n - 1] == '\r' || line[n - 1] == ' ')) line[--n] = 0;
        if (!n || line[0] == '#') continue;
        char word[64] = {0}, a[64] = {0}, b[64] = {0}, c[64] = {0}, d[64] = {0}, e[64] = {0};
        int k = sscanf(line, "%63s %63s %63s %63s %63s %63s", word, a, b, c, d, e);
        if (!strcmp(word, "ts")) copy_path(cfg.ts, rest_after(line, 1), line);
        else if (!strcmp(word, "ms")) copy_path(cfg.ms, rest_after(line, 1), line);
        else if (!strcmp(word, "fs")) copy_path(cfg.fs, rest_after(line, 1), line);
        else if (!strcmp(word, "out")) copy_path(cfg.out, rest_after(line, 1), line);
        else if (!strcmp(word, "size") && k == 3) { cfg.width = (uint32_t)atoi(a); cfg.height = (uint32_t)atoi(b); }
        else if (!strcmp(word, "clear") && k == 5) {
            cfg.clear[0] = (float)atof(a); cfg.clear[1] = (float)atof(b);
            cfg.clear[2] = (float)atof(c); cfg.clear[3] = (float)atof(d);
        } else if ((!strcmp(word, "uniform") || !strcmp(word, "storage")) && k >= 4) {
            add_binding(cfg.pushed, &cfg.npushed, word[0] == 'u' ? BIND_UNIFORM : BIND_STORAGE, (uint32_t)atoi(a),
                        parse_stages(b, line), rest_after(line, 3), line);
        } else if ((!strcmp(word, "taskuniform") || !strcmp(word, "taskstorage")) && k >= 3) {
            add_binding(cfg.task, &cfg.ntask, !strcmp(word, "taskuniform") ? BIND_UNIFORM : BIND_STORAGE, (uint32_t)atoi(a),
                        VK_SHADER_STAGE_TASK_BIT_EXT, rest_after(line, 2), line);
        } else if (!strcmp(word, "global") && k == 3) {
            cfg.global_images = (uint32_t)strtoul(a, NULL, 0);
            cfg.global_samplers = (uint32_t)strtoul(b, NULL, 0);
            if (!cfg.global_images || !cfg.global_samplers) die(1, "the global arrays need a length");
        } else if (!strcmp(word, "push") && k == 3) {
            unsigned long off = strtoul(a, NULL, 0);
            if (off % 4 || off >= PUSH_BYTES) die(1, "a push constant word is misaligned or past 128 bytes");
            cfg.push[off / 4] = (uint32_t)strtoul(b, NULL, 0);
        } else if (!strcmp(word, "feature") && k == 2) {
            if (cfg.nfeatures == MAX_FEATURES) die(1, "too many features");
            snprintf(cfg.features[cfg.nfeatures++], 64, "%s", a);
        } else if (!strcmp(word, "subgroup") && k == 2) {
            static const struct { const char *name; VkSubgroupFeatureFlags bit; } ops[] = {
                {"basic", VK_SUBGROUP_FEATURE_BASIC_BIT}, {"vote", VK_SUBGROUP_FEATURE_VOTE_BIT},
                {"arithmetic", VK_SUBGROUP_FEATURE_ARITHMETIC_BIT}, {"ballot", VK_SUBGROUP_FEATURE_BALLOT_BIT},
                {"shuffle", VK_SUBGROUP_FEATURE_SHUFFLE_BIT}, {"relative", VK_SUBGROUP_FEATURE_SHUFFLE_RELATIVE_BIT},
                {"clustered", VK_SUBGROUP_FEATURE_CLUSTERED_BIT}, {"quad", VK_SUBGROUP_FEATURE_QUAD_BIT},
            };
            int found = 0;
            for (size_t i = 0; i < sizeof ops / sizeof ops[0]; i++)
                if (!strcmp(a, ops[i].name)) { cfg.subgroup_ops |= ops[i].bit; found = 1; }
            if (!found) die(1, "unknown subgroup operation");
        } else if (!strcmp(word, "validation") && k == 2) {
            if (!strcmp(a, "off")) cfg.validation = 0;
            else if (!strcmp(a, "on")) cfg.validation = 1;
            else if (!strcmp(a, "gpu")) cfg.validation = 2;
            else if (!strcmp(a, "sync")) cfg.validation = 3;
            else die(1, "validation is off, on, gpu or sync");
        } else if (!strcmp(word, "device") && k >= 2) {
            snprintf(cfg.device_filter, sizeof cfg.device_filter, "%s", rest_after(line, 1));
        } else if (!strcmp(word, "draw") && k == 4 && !strcmp(a, "direct")) {
            cfg.indirect = 1;
            cfg.count = (uint32_t)strtoul(b, NULL, 0);
            cfg.first = (uint32_t)strtoul(c, NULL, 0);
        } else if (!strcmp(word, "draw") && k >= 4 && !strcmp(a, "indirect")) {
            cfg.indirect = 2;
            cfg.index = (uint32_t)strtoul(b, NULL, 0);
            copy_path(cfg.records, rest_after(line, 3), line);
        } else if (!strcmp(word, "flag") && k == 2) {
            if (!strcmp(a, "nobarrier")) cfg.nobarrier = 1;
            else if (!strcmp(a, "ignorefirst")) cfg.ignorefirst = 1;
            else if (!strcmp(a, "nopin")) cfg.nopin = 1;
            else if (!strcmp(a, "vertexlibrary")) cfg.vertexlibrary = 1;
            else die(1, "flag is nobarrier, ignorefirst, nopin or vertexlibrary");
        } else {
            fprintf(stderr, "vkmesh: line %u not understood: %s\n", lineno, line);
            exit(1);
        }
    }
    fclose(f);
    if (!cfg.ms[0] || !cfg.fs[0] || !cfg.out[0]) die(1, "ms, fs and out are required");
    if (!cfg.width || !cfg.height || cfg.width > 16384 || cfg.height > 16384) die(1, "size is missing or out of range");
    if (!cfg.indirect) die(1, "a draw is required");
}

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "vkmesh: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fprintf(stderr, "vkmesh: %s is empty\n", path); exit(1); }
    void *p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "vkmesh: cannot read %s\n", path); exit(1); }
    fclose(f);
    *size = (size_t)n;
    return p;
}

// ---- features ---------------------------------------------------------------

typedef struct {
    VkPhysicalDeviceFeatures2 f2;
    VkPhysicalDeviceVulkan11Features v11;
    VkPhysicalDeviceVulkan12Features v12;
    VkPhysicalDeviceVulkan13Features v13;
    VkPhysicalDeviceMeshShaderFeaturesEXT mesh;
    VkPhysicalDeviceGraphicsPipelineLibraryFeaturesEXT gpl;
} Features;

static void features_chain(Features *f) {
    memset(f, 0, sizeof *f);
    f->f2.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2;
    f->v11.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_1_FEATURES;
    f->v12.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES;
    f->v13.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES;
    f->mesh.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT;
    f->gpl.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GRAPHICS_PIPELINE_LIBRARY_FEATURES_EXT;
    f->f2.pNext = &f->v11;
    f->v11.pNext = &f->v12;
    f->v12.pNext = &f->v13;
    f->v13.pNext = &f->mesh;
    f->mesh.pNext = &f->gpl;
}

// Where a named feature lives: which structure, and the offset of its VkBool32.
typedef struct { const char *name; int which; size_t offset; } FeatureRef;
#define F10(m) {#m, 0, offsetof(VkPhysicalDeviceFeatures, m)}
#define F11(m) {#m, 1, offsetof(VkPhysicalDeviceVulkan11Features, m)}
#define F12(m) {#m, 2, offsetof(VkPhysicalDeviceVulkan12Features, m)}
#define F13(m) {#m, 3, offsetof(VkPhysicalDeviceVulkan13Features, m)}
static const FeatureRef FEATURES[] = {
    F10(shaderFloat64), F10(shaderInt64), F10(shaderInt16), F10(shaderClipDistance), F10(shaderCullDistance),
    F10(shaderImageGatherExtended), F10(shaderStorageImageReadWithoutFormat),
    F10(shaderStorageImageWriteWithoutFormat), F10(shaderStorageImageMultisample), F10(imageCubeArray),
    F10(fragmentStoresAndAtomics), F10(vertexPipelineStoresAndAtomics), F10(sampleRateShading),
    F10(shaderStorageImageExtendedFormats), F10(shaderResourceMinLod),
    F11(shaderDrawParameters), F11(storageBuffer16BitAccess), F11(uniformAndStorageBuffer16BitAccess),
    F12(shaderFloat16), F12(shaderInt8), F12(bufferDeviceAddress), F12(scalarBlockLayout),
    F12(storageBuffer8BitAccess), F12(uniformAndStorageBuffer8BitAccess), F12(shaderBufferInt64Atomics),
    F12(shaderSharedInt64Atomics), F12(shaderOutputLayer), F12(shaderOutputViewportIndex),
    F12(runtimeDescriptorArray), F12(shaderSampledImageArrayNonUniformIndexing),
    F12(descriptorBindingPartiallyBound), F12(vulkanMemoryModel), F12(subgroupBroadcastDynamicId),
    F12(descriptorBindingSampledImageUpdateAfterBind), F10(shaderSampledImageArrayDynamicIndexing),
    F13(shaderDemoteToHelperInvocation), F13(shaderIntegerDotProduct), F13(subgroupSizeControl),
    F13(computeFullSubgroups), F13(maintenance4),
};

static VkBool32 *feature_slot(Features *f, const FeatureRef *r) {
    char *base = r->which == 0 ? (char *)&f->f2.features
               : r->which == 1 ? (char *)&f->v11
               : r->which == 2 ? (char *)&f->v12
                               : (char *)&f->v13;
    return (VkBool32 *)(base + r->offset);
}

// ---- validation messages ------------------------------------------------------

static VKAPI_ATTR VkBool32 VKAPI_CALL on_message(VkDebugUtilsMessageSeverityFlagBitsEXT severity,
                                                 VkDebugUtilsMessageTypeFlagsEXT types,
                                                 const VkDebugUtilsMessengerCallbackDataEXT *data, void *user) {
    (void)user;
    const char *kind = "info";
    if (types == VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT) {
        // The loader's own messages (third-party layers registered on the machine), not a verdict on the draw: counted apart.
        loader_messages++;
        fprintf(stderr, "vkmesh: loader: %s\n", data->pMessage ? data->pMessage : "(no text)");
        return VK_FALSE;
    }
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { val_errors++; kind = "error"; }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { val_warnings++; kind = "warning"; }
    fprintf(stderr, "vkmesh: validation %s: %s\n", kind, data->pMessage ? data->pMessage : "(no text)");
    return VK_FALSE;
}

// ---- small helpers ----------------------------------------------------------

static VkPhysicalDevice gpu;
static VkDevice dev;
static VkPhysicalDeviceMemoryProperties memprops;

static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < memprops.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (memprops.memoryTypes[i].propertyFlags & want) == want) return i;
    die(2, "no memory type fits");
    return 0;
}

typedef struct {
    VkBuffer buf;
    VkDeviceMemory mem;
    void *map;
    VkDeviceSize size;
    VkDeviceAddress addr;
} Buffer;

// A buffer of `size` bytes: host visible and mapped (host), or device local; with a device address when `addressable`.
static Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, int host, int addressable, const void *data, size_t data_size) {
    Buffer b = {.size = size};
    if (addressable) usage |= VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
    VK_CHECK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b.buf, &req);
    VkMemoryAllocateFlagsInfo flags = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                       .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = addressable ? &flags : NULL,
                                  .allocationSize = req.size,
                                  .memoryTypeIndex = memory_type(req.memoryTypeBits, host
                                      ? (VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)
                                      : VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    if (host) {
        VK_CHECK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
        memset(b.map, 0, (size_t)size);
        if (data) memcpy(b.map, data, data_size < size ? data_size : (size_t)size);
    }
    if (addressable) {
        VkBufferDeviceAddressInfo address_info = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = b.buf};
        b.addr = vkGetBufferDeviceAddress(dev, &address_info);
    }
    return b;
}

static void free_buffer(Buffer *b) {
    if (!b->buf) return;
    vkDestroyBuffer(dev, b->buf, NULL);
    vkFreeMemory(dev, b->mem, NULL);
}

typedef struct {
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
} Image;

static Image make_target(uint32_t w, uint32_t h) {
    Image im;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R32G32B32A32_SFLOAT, .extent = {w, h, 1}, .mipLevels = 1,
                            .arrayLayers = 1, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VK_CHECK(vkCreateImage(dev, &ii, NULL, &im.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, im.img, &req);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                  .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, &im.mem));
    VK_CHECK(vkBindImageMemory(dev, im.img, im.mem, 0));
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im.img,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VK_CHECK(vkCreateImageView(dev, &vi, NULL, &im.view));
    return im;
}

static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
                          VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                          VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = src_stage,
                               .srcAccessMask = src_access, .dstStageMask = dst_stage, .dstAccessMask = dst_access,
                               .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VK_REMAINING_ARRAY_LAYERS}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1,
                           .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
}

static void buffer_barrier(VkCommandBuffer cb, VkBuffer buf, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                           VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkBufferMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2, .srcStageMask = src_stage,
                                .srcAccessMask = src_access, .dstStageMask = dst_stage, .dstAccessMask = dst_access,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .buffer = buf, .size = VK_WHOLE_SIZE};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1,
                           .pBufferMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
}

static VkShaderModule load_module(const char *path) {
    size_t n;
    uint32_t *code = read_file(path, &n);
    if (n % 4 || code[0] != 0x07230203u) { fprintf(stderr, "vkmesh: %s is not SPIR-V\n", path); exit(1); }
    VkShaderModuleCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = n, .pCode = code};
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(dev, &ci, NULL, &m));
    free(code);
    return m;
}

static int has_extension(const VkExtensionProperties *exts, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(exts[i].extensionName, name)) return 1;
    return 0;
}

// The descriptors of one set, from the draw file: a buffer for each binding, and the layout binding and write that name it.
typedef struct {
    Buffer bufs[MAX_BINDINGS];
    VkDescriptorSetLayoutBinding lb[MAX_BINDINGS];
    VkWriteDescriptorSet writes[MAX_BINDINGS];
    VkDescriptorBufferInfo info[MAX_BINDINGS];
    uint32_t n;
} SetData;

static void set_data(SetData *s, const Binding *list, uint32_t n) {
    s->n = n;
    for (uint32_t i = 0; i < n; i++) {
        const Binding *bd = &list[i];
        size_t size;
        void *data = read_file(bd->path, &size);
        VkDescriptorType type = bd->kind == BIND_UNIFORM ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        s->bufs[i] = make_buffer(size, bd->kind == BIND_UNIFORM ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                                 : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1, 0, data, size);
        s->info[i] = (VkDescriptorBufferInfo){s->bufs[i].buf, 0, size};
        free(data);
        s->lb[i] = (VkDescriptorSetLayoutBinding){bd->binding, type, 1, bd->stages, NULL};
        s->writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = bd->binding,
                                              .descriptorCount = 1, .descriptorType = type, .pBufferInfo = &s->info[i]};
    }
}

// ---- pipeline libraries -------------------------------------------------------------

static VkPipelineLayout layout;
static VkFormat colour_format = VK_FORMAT_R32G32B32A32_SFLOAT;

// Create one library of the graphics pipeline: pre-rasterization shaders (the task and mesh stages), the fragment shader, the
// fragment output interface, or the (empty) vertex input interface.
static VkPipeline make_library(VkGraphicsPipelineLibraryFlagsEXT part, VkShaderModule ts, VkShaderModule ms, VkShaderModule fs) {
    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo pin = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO, .requiredSubgroupSize = 32};
    VkPipelineShaderStageCreateFlags stage_flags = cfg.nopin ? 0 : VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT;
    void *stage_next = cfg.nopin ? NULL : (void *)&pin;
    VkPipelineShaderStageCreateInfo stages[3];
    uint32_t ns = 0;
    if (part == VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT) {
        if (ts)
            stages[ns++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
                .pNext = stage_next, .flags = stage_flags, .stage = VK_SHADER_STAGE_TASK_BIT_EXT, .module = ts, .pName = "main"};
        stages[ns++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext = stage_next, .flags = stage_flags, .stage = VK_SHADER_STAGE_MESH_BIT_EXT, .module = ms, .pName = "main"};
    } else if (part == VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT) {
        stages[ns++] = (VkPipelineShaderStageCreateInfo){.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fs, .pName = "main"};
    }

    VkViewport vp = {0, 0, (float)cfg.width, (float)cfg.height, 0, 1};
    VkRect2D sc = {{0, 0}, {cfg.width, cfg.height}};
    VkPipelineVertexInputStateCreateInfo vis = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    VkPipelineInputAssemblyStateCreateInfo ias = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkPipelineViewportStateCreateInfo vps = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                             .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                                                 .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms_state = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                                     .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineDepthStencilStateCreateInfo ds = {.sType = VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cbs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                               .attachmentCount = 1, .pAttachments = &cba};
    VkPipelineRenderingCreateInfo rendering = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                               .colorAttachmentCount = 1, .pColorAttachmentFormats = &colour_format};
    VkGraphicsPipelineLibraryCreateInfoEXT lib = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_LIBRARY_CREATE_INFO_EXT,
                                                  .pNext = &rendering, .flags = part};
    int pre = part == VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT;
    int frag = part == VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT;
    int out = part == VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT;
    int vin = part == VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT;
    VkGraphicsPipelineCreateInfo gp = {
        .sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &lib, .flags = VK_PIPELINE_CREATE_LIBRARY_BIT_KHR,
        .stageCount = ns, .pStages = ns ? stages : NULL, .pVertexInputState = vin ? &vis : NULL,
        .pInputAssemblyState = vin ? &ias : NULL, .pViewportState = pre ? &vps : NULL,
        .pRasterizationState = pre ? &rs : NULL, .pMultisampleState = (frag || out) ? &ms_state : NULL,
        .pDepthStencilState = frag ? &ds : NULL, .pColorBlendState = out ? &cbs : NULL,
        .layout = (pre || frag) ? layout : VK_NULL_HANDLE};
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gp, NULL, &p));
    return p;
}

// ---- main ---------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc != 2) die(1, "usage: vkmesh <draw.txt>");
    read_config(argv[1]);
    if (volkInitialize() != VK_SUCCESS) die(2, "no Vulkan loader");

    // Instance, with the validation layer and a messenger unless told otherwise.
    const char *layers[1] = {"VK_LAYER_KHRONOS_validation"};
    uint32_t nlayers = 0;
    if (cfg.validation) {
        uint32_t n = 0;
        vkEnumerateInstanceLayerProperties(&n, NULL);
        VkLayerProperties *lp = calloc(n ? n : 1, sizeof *lp);
        vkEnumerateInstanceLayerProperties(&n, lp);
        for (uint32_t i = 0; i < n; i++)
            if (!strcmp(lp[i].layerName, layers[0])) nlayers = 1;
        free(lp);
        if (!nlayers) die(2, "validation was asked for and VK_LAYER_KHRONOS_validation is not installed");
    }
    const char *iexts[1] = {VK_EXT_DEBUG_UTILS_EXTENSION_NAME};
    VkDebugUtilsMessengerCreateInfoEXT mci = {
        .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT,
        .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT,
        .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT |
                       VK_DEBUG_UTILS_MESSAGE_TYPE_PERFORMANCE_BIT_EXT,
        .pfnUserCallback = on_message};
    VkValidationFeatureEnableEXT vfe[2];
    uint32_t nvfe = 0;
    if (cfg.validation == 2) {
        vfe[nvfe++] = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_EXT;
        vfe[nvfe++] = VK_VALIDATION_FEATURE_ENABLE_GPU_ASSISTED_RESERVE_BINDING_SLOT_EXT;
    } else if (cfg.validation == 3) {
        vfe[nvfe++] = VK_VALIDATION_FEATURE_ENABLE_SYNCHRONIZATION_VALIDATION_EXT;
    }
    VkValidationFeaturesEXT vf = {.sType = VK_STRUCTURE_TYPE_VALIDATION_FEATURES_EXT, .pNext = &mci,
                                  .enabledValidationFeatureCount = nvfe, .pEnabledValidationFeatures = vfe};
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkmesh",
                             .apiVersion = VK_API_VERSION_1_3};
    VkInstanceCreateInfo ici = {.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pNext = cfg.validation ? (void *)&vf : NULL,
                                .pApplicationInfo = &app, .enabledLayerCount = nlayers, .ppEnabledLayerNames = layers,
                                .enabledExtensionCount = cfg.validation ? 1 : 0, .ppEnabledExtensionNames = iexts};
    VkInstance inst;
    VK_CHECK(vkCreateInstance(&ici, NULL, &inst));
    volkLoadInstance(inst);
    VkDebugUtilsMessengerEXT messenger = VK_NULL_HANDLE;
    if (cfg.validation) VK_CHECK(vkCreateDebugUtilsMessengerEXT(inst, &mci, NULL, &messenger));

    // Device: the first one whose name contains the filter, discrete first.
    uint32_t ngpu = 0;
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ngpu, NULL));
    if (!ngpu) die(2, "no Vulkan device");
    VkPhysicalDevice *gpus = calloc(ngpu, sizeof *gpus);
    VK_CHECK(vkEnumeratePhysicalDevices(inst, &ngpu, gpus));
    VkPhysicalDeviceProperties props = {0};
    for (int pass = 0; pass < 2 && !gpu; pass++)
        for (uint32_t i = 0; i < ngpu && !gpu; i++) {
            VkPhysicalDeviceProperties p;
            vkGetPhysicalDeviceProperties(gpus[i], &p);
            if (pass == 0 && p.deviceType != VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) continue;
            if (cfg.device_filter[0] && !strstr(p.deviceName, cfg.device_filter)) continue;
            gpu = gpus[i];
            props = p;
        }
    free(gpus);
    if (!gpu) die(2, "no device matches");
    if (props.apiVersion < VK_API_VERSION_1_3) die(2, "the device does not offer Vulkan 1.3");
    vkGetPhysicalDeviceMemoryProperties(gpu, &memprops);

    uint32_t next = 0;
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, NULL);
    VkExtensionProperties *dexts = calloc(next ? next : 1, sizeof *dexts);
    vkEnumerateDeviceExtensionProperties(gpu, NULL, &next, dexts);
    const char *need[] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME, VK_EXT_MESH_SHADER_EXTENSION_NAME,
                          VK_EXT_GRAPHICS_PIPELINE_LIBRARY_EXTENSION_NAME, VK_KHR_PIPELINE_LIBRARY_EXTENSION_NAME};
    for (size_t i = 0; i < sizeof need / sizeof need[0]; i++)
        if (!has_extension(dexts, next, need[i])) { fprintf(stderr, "vkmesh: the device lacks %s\n", need[i]); exit(2); }
    free(dexts);

    // The subgroup: size 32 required in the task and mesh stages, and the operations the modules use in all three stages.
    VkPhysicalDeviceSubgroupProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceSubgroupSizeControlProperties sgc = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_SIZE_CONTROL_PROPERTIES,
                                                         .pNext = &sgp};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgc};
    vkGetPhysicalDeviceProperties2(gpu, &p2);
    if (sgc.minSubgroupSize > 32 || sgc.maxSubgroupSize < 32) die(2, "the device cannot run a subgroup of 32");
    if ((sgc.requiredSubgroupSizeStages & (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT)) !=
        (VK_SHADER_STAGE_TASK_BIT_EXT | VK_SHADER_STAGE_MESH_BIT_EXT))
        die(2, "the device cannot require a subgroup size in the task and mesh stages");
    if ((sgp.supportedOperations & cfg.subgroup_ops) != cfg.subgroup_ops) die(2, "a subgroup operation is missing");
    if (cfg.subgroup_ops && (sgp.supportedStages & MESH_STAGES) != MESH_STAGES)
        die(2, "subgroup operations are not supported in the task, mesh and fragment stages");

    // Features: exactly the ones asked for, plus what the harness itself needs.
    Features have, want;
    features_chain(&have);
    features_chain(&want);
    vkGetPhysicalDeviceFeatures2(gpu, &have.f2);
    want.v13.dynamicRendering = VK_TRUE;
    want.v13.synchronization2 = VK_TRUE;
    // The global set of the layout: update-after-bind arrays, partially bound, indexed by a slot's handle.
    if (!have.v12.descriptorBindingSampledImageUpdateAfterBind || !have.v12.descriptorBindingPartiallyBound ||
        !have.f2.features.shaderSampledImageArrayDynamicIndexing)
        die(2, "the device lacks update-after-bind sampled images, partially bound arrays or their dynamic indexing");
    want.v12.descriptorBindingSampledImageUpdateAfterBind = VK_TRUE;
    want.v12.descriptorBindingPartiallyBound = VK_TRUE;
    want.f2.features.shaderSampledImageArrayDynamicIndexing = VK_TRUE;
    // Device addresses (the first task word's), the pinned subgroup size and the stages themselves
    if (!have.v12.bufferDeviceAddress || !have.v13.subgroupSizeControl || !have.v13.computeFullSubgroups)
        die(2, "the device lacks bufferDeviceAddress, subgroupSizeControl or computeFullSubgroups");
    want.v12.bufferDeviceAddress = VK_TRUE;
    want.v13.subgroupSizeControl = VK_TRUE;
    want.v13.computeFullSubgroups = VK_TRUE;
    if (!have.mesh.meshShader || !have.mesh.taskShader || !have.gpl.graphicsPipelineLibrary)
        die(2, "the device lacks meshShader, taskShader or graphicsPipelineLibrary");
    want.mesh.meshShader = VK_TRUE;
    want.mesh.taskShader = VK_TRUE;
    want.gpl.graphicsPipelineLibrary = VK_TRUE;
    for (uint32_t i = 0; i < cfg.nfeatures; i++) {
        const FeatureRef *ref = NULL;
        for (size_t j = 0; j < sizeof FEATURES / sizeof FEATURES[0]; j++)
            if (!strcmp(FEATURES[j].name, cfg.features[i])) ref = &FEATURES[j];
        if (!ref) { fprintf(stderr, "vkmesh: unknown feature %s\n", cfg.features[i]); exit(1); }
        if (!*feature_slot(&have, ref)) { fprintf(stderr, "vkmesh: the device lacks %s\n", cfg.features[i]); exit(2); }
        *feature_slot(&want, ref) = VK_TRUE;
    }

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
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &want.f2, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = sizeof need / sizeof need[0],
                              .ppEnabledExtensionNames = need};
    VK_CHECK(vkCreateDevice(gpu, &dci, NULL, &dev));
    volkLoadDevice(dev);
    VkQueue queue;
    vkGetDeviceQueue(dev, family, 0, &queue);

    // Resources: the pushed set's buffers, the task set's, the first-task words, the records.
    VkPhysicalDeviceDescriptorIndexingProperties dip = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
    VkPhysicalDeviceProperties2 p2b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &dip};
    vkGetPhysicalDeviceProperties2(gpu, &p2b);
    if (dip.maxPerStageDescriptorUpdateAfterBindSampledImages < cfg.global_images ||
        dip.maxDescriptorSetUpdateAfterBindSampledImages < cfg.global_images ||
        dip.maxPerStageDescriptorUpdateAfterBindSamplers < cfg.global_samplers ||
        dip.maxDescriptorSetUpdateAfterBindSamplers < cfg.global_samplers)
        die(2, "the global arrays are longer than the device's update-after-bind limits");

    static SetData pushed, task;
    set_data(&pushed, cfg.pushed, cfg.npushed);
    set_data(&task, cfg.task, cfg.ntask);

    // The first task: word 0 is zero for ever (a direct draw with first 0 points here, and writes nothing), word 1 the first task of
    // a direct draw that has one.
    uint32_t first_words[4] = {0, cfg.first, 0, 0};
    Buffer firstbuf = make_buffer(sizeof first_words, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 1, 1, first_words, sizeof first_words);
    Buffer records = {0}, records_stage = {0};
    uint32_t record_count = 0;
    if (cfg.indirect == 2) {
        size_t n;
        void *data = read_file(cfg.records, &n);
        if (n % RECORD_BYTES) die(1, "the records file is not a whole number of 16-byte records");
        record_count = (uint32_t)(n / RECORD_BYTES);
        if (cfg.index >= record_count) die(1, "the record index is past the records file");
        records_stage = make_buffer(n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, 1, 0, data, n);
        records = make_buffer(n, VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT, 0, 1, NULL, 0);
        free(data);
    }

    // The layout `uni`: set 0 the global arrays, set 1 pushed descriptors, set 2 the task stage's, 128 bytes of push constants.
    VkDescriptorSetLayout dsl[3];
    VkDescriptorSetLayoutCreateInfo dlci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                            .bindingCount = pushed.n, .pBindings = pushed.lb};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl[1]));
    VkDescriptorSetLayoutCreateInfo tlci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .bindingCount = task.n, .pBindings = task.lb};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &tlci, NULL, &dsl[2]));

    VkShaderStageFlags gst = MESH_STAGES;
    VkDescriptorSetLayoutBinding glb[2] = {
        {0, VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, cfg.global_images, gst, NULL},
        {1, VK_DESCRIPTOR_TYPE_SAMPLER, cfg.global_samplers, gst, NULL},
    };
    VkDescriptorBindingFlags gflags[2] = {
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
        VK_DESCRIPTOR_BINDING_UPDATE_AFTER_BIND_BIT | VK_DESCRIPTOR_BINDING_PARTIALLY_BOUND_BIT,
    };
    VkDescriptorSetLayoutBindingFlagsCreateInfo gbf = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_BINDING_FLAGS_CREATE_INFO,
                                                       .bindingCount = 2, .pBindingFlags = gflags};
    VkDescriptorSetLayoutCreateInfo gdlci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO, .pNext = &gbf,
                                             .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_UPDATE_AFTER_BIND_POOL_BIT,
                                             .bindingCount = 2, .pBindings = glb};
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &gdlci, NULL, &dsl[0]));
    VkDescriptorPoolSize gps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, cfg.global_images},
                                   {VK_DESCRIPTOR_TYPE_SAMPLER, cfg.global_samplers}};
    VkDescriptorPoolCreateInfo gpci0 = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                        .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1,
                                        .poolSizeCount = 2, .pPoolSizes = gps};
    VkDescriptorPool gpool;
    VK_CHECK(vkCreateDescriptorPool(dev, &gpci0, NULL, &gpool));
    VkDescriptorSetAllocateInfo gai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = gpool,
                                       .descriptorSetCount = 1, .pSetLayouts = &dsl[0]};
    VkDescriptorSet gset;
    VK_CHECK(vkAllocateDescriptorSets(dev, &gai, &gset));

    // Set 2: an ordinary set, written before the draw (the runtime writes one for each task draw).
    VkDescriptorPool tpool = VK_NULL_HANDLE;
    VkDescriptorSet tset = VK_NULL_HANDLE;
    if (task.n) {
        uint32_t nu = 0, ns = 0;
        for (uint32_t i = 0; i < task.n; i++) {
            if (task.lb[i].descriptorType == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) nu++;
            else ns++;
        }
        VkDescriptorPoolSize tps[2];
        uint32_t ntps = 0;
        if (nu) tps[ntps++] = (VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, nu};
        if (ns) tps[ntps++] = (VkDescriptorPoolSize){VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, ns};
        VkDescriptorPoolCreateInfo tpci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = 1,
                                           .poolSizeCount = ntps, .pPoolSizes = tps};
        VK_CHECK(vkCreateDescriptorPool(dev, &tpci, NULL, &tpool));
        VkDescriptorSetAllocateInfo tai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = tpool,
                                           .descriptorSetCount = 1, .pSetLayouts = &dsl[2]};
        VK_CHECK(vkAllocateDescriptorSets(dev, &tai, &tset));
        for (uint32_t i = 0; i < task.n; i++) task.writes[i].dstSet = tset;
        vkUpdateDescriptorSets(dev, task.n, task.writes, 0, NULL);
    }

    VkPushConstantRange pcr = {MESH_STAGES, 0, PUSH_BYTES};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 3,
                                       .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VK_CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));

    Image target = make_target(cfg.width, cfg.height);
    Buffer readback = make_buffer((VkDeviceSize)cfg.width * cfg.height * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, 1, 0, NULL, 0);

    // Pipeline: three libraries (the vertex input interface is not part of a pipeline with a mesh stage) fast-linked.
    VkShaderModule tsm = cfg.ts[0] ? load_module(cfg.ts) : VK_NULL_HANDLE, msm = load_module(cfg.ms), fsm = load_module(cfg.fs);
    VkPipeline libs[4];
    uint32_t nlibs = 0;
    if (cfg.vertexlibrary) libs[nlibs++] = make_library(VK_GRAPHICS_PIPELINE_LIBRARY_VERTEX_INPUT_INTERFACE_BIT_EXT, tsm, msm, fsm);
    libs[nlibs++] = make_library(VK_GRAPHICS_PIPELINE_LIBRARY_PRE_RASTERIZATION_SHADERS_BIT_EXT, tsm, msm, fsm);
    libs[nlibs++] = make_library(VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_SHADER_BIT_EXT, tsm, msm, fsm);
    libs[nlibs++] = make_library(VK_GRAPHICS_PIPELINE_LIBRARY_FRAGMENT_OUTPUT_INTERFACE_BIT_EXT, tsm, msm, fsm);
    VkPipelineLibraryCreateInfoKHR li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LIBRARY_CREATE_INFO_KHR, .libraryCount = nlibs,
                                         .pLibraries = libs};
    VkGraphicsPipelineCreateInfo link = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &li, .layout = layout};
    VkPipeline pipe;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &link, NULL, &pipe));

    // The first task's address, in push constant words 6 and 7, and word 8 kept: zero.
    VkDeviceAddress first_addr;
    if (cfg.indirect == 2) first_addr = records.addr + (VkDeviceAddress)cfg.index * RECORD_BYTES + FIRST_OFFSET;
    else first_addr = firstbuf.addr + ((cfg.first && !cfg.ignorefirst) ? 4 : 0);
    cfg.push[FIRST_TASK_WORD] = (uint32_t)(first_addr & 0xFFFFFFFFu);
    cfg.push[FIRST_TASK_WORD + 1] = (uint32_t)(first_addr >> 32);
    cfg.push[FIRST_TASK_WORD + 2] = 0;

    // Commands: the records' copy and its barrier, the draw, the readback.
    VkCommandPoolCreateInfo cpci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = family};
    VkCommandPool pool;
    VK_CHECK(vkCreateCommandPool(dev, &cpci, NULL, &pool));
    VkCommandBufferAllocateInfo cbai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = pool,
                                        .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkCommandBufferBeginInfo cbbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                     .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(cb, &cbbi));
    if (cfg.indirect == 2) {
        VkBufferCopy copy = {0, 0, records.size};
        vkCmdCopyBuffer(cb, records_stage.buf, records.buf, 1, &copy);
        // The conversion pass's role: its write must reach the indirect command read and the stages' reads of the first task's word
        if (!cfg.nobarrier)
            buffer_barrier(cb, records.buf, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_TASK_SHADER_BIT_EXT |
                               VK_PIPELINE_STAGE_2_MESH_SHADER_BIT_EXT,
                           VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_2_SHADER_READ_BIT);
    }
    image_barrier(cb, target.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                  VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                  VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkRect2D sc = {{0, 0}, {cfg.width, cfg.height}};
    VkRenderingAttachmentInfo att = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = target.view,
                                     .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    memcpy(att.clearValue.color.float32, cfg.clear, sizeof cfg.clear);
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = sc, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &att};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &gset, 0, NULL);
    if (pushed.n) vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, pushed.n, pushed.writes);
    if (task.n) vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 2, 1, &tset, 0, NULL);
    vkCmdPushConstants(cb, layout, MESH_STAGES, 0, PUSH_BYTES, cfg.push);
    if (cfg.indirect == 1) vkCmdDrawMeshTasksEXT(cb, cfg.count, 1, 1);
    else vkCmdDrawMeshTasksIndirectEXT(cb, records.buf, (VkDeviceSize)cfg.index * RECORD_BYTES, 1, COMMAND_BYTES);
    vkCmdEndRendering(cb);
    image_barrier(cb, target.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                  VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                  VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy back = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .imageExtent = {cfg.width, cfg.height, 1}};
    vkCmdCopyImageToBuffer(cb, target.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &back);
    buffer_barrier(cb, readback.buf, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                   VK_PIPELINE_STAGE_2_HOST_BIT, VK_ACCESS_2_HOST_READ_BIT);
    VK_CHECK(vkEndCommandBuffer(cb));

    VkFenceCreateInfo fci = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VkFence fence;
    VK_CHECK(vkCreateFence(dev, &fci, NULL, &fence));
    VkSubmitInfo submit = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cb};
    VK_CHECK(vkQueueSubmit(queue, 1, &submit, fence));
    VK_CHECK(vkWaitForFences(dev, 1, &fence, VK_TRUE, 10ull * 1000 * 1000 * 1000));

    FILE *out = fopen(cfg.out, "wb");
    if (!out) die(1, "cannot write the output file");
    size_t total = (size_t)cfg.width * cfg.height * 16;
    if (fwrite(readback.map, 1, total, out) != total) die(1, "short write of the output file");
    fclose(out);

    // Teardown, so that the validation layer can report leaks and misuse at exit.
    vkDestroyFence(dev, fence, NULL);
    vkDestroyCommandPool(dev, pool, NULL);
    vkDestroyPipeline(dev, pipe, NULL);
    for (uint32_t i = 0; i < nlibs; i++) vkDestroyPipeline(dev, libs[i], NULL);
    if (tsm) vkDestroyShaderModule(dev, tsm, NULL);
    vkDestroyShaderModule(dev, msm, NULL);
    vkDestroyShaderModule(dev, fsm, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    if (tpool) vkDestroyDescriptorPool(dev, tpool, NULL);
    vkDestroyDescriptorPool(dev, gpool, NULL);
    for (int i = 0; i < 3; i++) vkDestroyDescriptorSetLayout(dev, dsl[i], NULL);
    free_buffer(&readback);
    free_buffer(&records);
    free_buffer(&records_stage);
    free_buffer(&firstbuf);
    for (uint32_t i = 0; i < pushed.n; i++) free_buffer(&pushed.bufs[i]);
    for (uint32_t i = 0; i < task.n; i++) free_buffer(&task.bufs[i]);
    vkDestroyImageView(dev, target.view, NULL);
    vkDestroyImage(dev, target.img, NULL);
    vkFreeMemory(dev, target.mem, NULL);
    vkDestroyDevice(dev, NULL);
    if (messenger) vkDestroyDebugUtilsMessengerEXT(inst, messenger, NULL);
    vkDestroyInstance(inst, NULL);

    uint32_t dv = props.driverVersion;
    char driver[32];
    if (props.vendorID == 0x10DE)
        snprintf(driver, sizeof driver, "%u.%02u", (dv >> 22) & 0x3FF, (dv >> 14) & 0xFF);
    else
        snprintf(driver, sizeof driver, "%u", dv);
    printf("{\"device\": \"%s\", \"driver\": \"%s\", \"api\": \"%u.%u.%u\", \"validation\": \"%s\", "
           "\"validation_errors\": %u, \"validation_warnings\": %u, \"loader_messages\": %u, \"records\": %u}\n",
           props.deviceName, driver, VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
           VK_API_VERSION_PATCH(props.apiVersion),
           cfg.validation == 0 ? "off" : cfg.validation == 1 ? "on" : cfg.validation == 2 ? "gpu" : "sync",
           val_errors, val_warnings, loader_messages, record_count);
    return 0;
}
