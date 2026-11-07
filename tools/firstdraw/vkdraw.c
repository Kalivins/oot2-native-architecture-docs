// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// vkdraw: one draw through Vulkan, for spike S2 (architecture section 12).
//
//   vkdraw <draw.txt>
//
// Draws a non-indexed triangle list with a vertex and a fragment SPIR-V module
// through the descriptor layout `uni` (architecture D13, xlate/host/UniLayout.cs):
// set 0 the global update-after-bind arrays (binding 0 sampled images, binding 1
// samplers, partially bound), set 1 the push descriptors of the draw's buffers,
// and 128 bytes of push constants (the texture slots); into one
// R32G32B32A32_SFLOAT colour target, and writes the target back as raw
// little-endian floats, row by row. Validation messages are counted through a
// debug messenger, the loader's general messages apart from the validation
// layer's; the counts and the device go to stdout as one JSON line.
//
// The draw file is text, one directive per line, paths last on their line:
//   vs PATH                         fs PATH                  out PATH
//   size W H                        clear R G B A
//   vertices COUNT STRIDE PATH      attribute LOCATION FORMAT OFFSET
//   uniform BINDING STAGES PATH     storage BINDING STAGES PATH      (set 1)
//   global IMAGES SAMPLERS          (lengths of the set 0 arrays; default 33025 4096)
//   image ENTRY TYPE W H L PATH     (RGBA32F texels, at set 0 binding 0 [ENTRY]; TYPE 2d,
//                                    2darray, 3d or cube; L layers, depth or 6, slices in order)
//   sampler ENTRY FILTER ADDRESS    (at set 0 binding 1 [ENTRY])
//   push OFFSET VALUE               (a 32-bit word of the push constants)
//   feature NAME                    subgroup OPERATION
//   validation off|on|gpu|sync      device SUBSTRING
// STRIDE 0 (PATH -) draws without a vertex buffer, for a vertex program that has
// no inputs. FORMAT is r32f, rg32f, rgb32f, rgba32f or the same with i or ui for 32-bit
// integers (r32i, rgba32ui, ...); STAGES is v, f or vf; FILTER is nearest or
// linear; ADDRESS is clamp or repeat; NAME is a Vulkan feature member
// (shaderFloat64, shaderDrawParameters, ...); OPERATION is a subgroup operation
// (basic, vote, arithmetic, ballot, shuffle, relative, clustered, quad).
//
// Exit codes: 0 drawn (validation counts are in the output), 1 the draw file is
// wrong, 2 Vulkan is missing or lacks something the draw needs, 3 a Vulkan
// call failed.
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
#define MAX_ATTRS 32
#define MAX_BINDINGS 64
#define MAX_FEATURES 64
#define MAX_ENTRIES 64
#define PUSH_BYTES 128

typedef enum { BIND_UNIFORM, BIND_STORAGE } BindKind;

typedef struct {
    BindKind kind;
    uint32_t binding;
    VkShaderStageFlags stages;
    char path[MAX_PATH_LEN];
} Binding;

typedef struct {
    uint32_t entry, width, height, layers;
    VkImageViewType view;
    char path[MAX_PATH_LEN];
} GlobalImage;

typedef struct {
    uint32_t entry;
    VkFilter filter;
    VkSamplerAddressMode address;
} GlobalSampler;

typedef struct {
    uint32_t location;
    VkFormat format;
    uint32_t offset;
} Attr;

typedef struct {
    char vs[MAX_PATH_LEN], fs[MAX_PATH_LEN], out[MAX_PATH_LEN], vbuf[MAX_PATH_LEN];
    uint32_t width, height, stride, vertex_count;
    float clear[4];
    Attr attrs[MAX_ATTRS];
    uint32_t nattrs;
    Binding binds[MAX_BINDINGS];
    uint32_t nbinds;
    uint32_t global_images, global_samplers;
    GlobalImage images[MAX_ENTRIES];
    uint32_t nimages;
    GlobalSampler samplers[MAX_ENTRIES];
    uint32_t nsamplers;
    uint32_t push[PUSH_BYTES / 4];
    char features[MAX_FEATURES][64];
    uint32_t nfeatures;
    VkSubgroupFeatureFlags subgroup_ops;
    int validation; // 0 off, 1 core, 2 core + GPU-assisted, 3 core + synchronization
    char device_filter[256];
} Config;

static Config cfg = {.validation = 1, .global_images = 33025, .global_samplers = 4096};
static unsigned val_errors, val_warnings, loader_messages;

static void die(int code, const char *what) {
    fprintf(stderr, "vkdraw: %s\n", what);
    exit(code);
}

#define VK_CHECK(call)                                                          \
    do {                                                                        \
        VkResult r_ = (call);                                                   \
        if (r_ != VK_SUCCESS) {                                                 \
            fprintf(stderr, "vkdraw: %s failed: %d\n", #call, (int)r_);         \
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
        fprintf(stderr, "vkdraw: no path, or a path too long, in: %s\n", line);
        exit(1);
    }
    memcpy(dst, src, n + 1);
}

static VkShaderStageFlags parse_stages(const char *s, const char *line) {
    VkShaderStageFlags f = 0;
    for (; *s; s++) {
        if (*s == 'v') f |= VK_SHADER_STAGE_VERTEX_BIT;
        else if (*s == 'f') f |= VK_SHADER_STAGE_FRAGMENT_BIT;
        else { fprintf(stderr, "vkdraw: stages are v, f or vf: %s\n", line); exit(1); }
    }
    if (!f) { fprintf(stderr, "vkdraw: no stage in: %s\n", line); exit(1); }
    return f;
}

static VkFormat parse_format(const char *s, const char *line) {
    if (!strcmp(s, "r32f")) return VK_FORMAT_R32_SFLOAT;
    if (!strcmp(s, "rg32f")) return VK_FORMAT_R32G32_SFLOAT;
    if (!strcmp(s, "rgb32f")) return VK_FORMAT_R32G32B32_SFLOAT;
    if (!strcmp(s, "rgba32f")) return VK_FORMAT_R32G32B32A32_SFLOAT;
    if (!strcmp(s, "r32i")) return VK_FORMAT_R32_SINT;
    if (!strcmp(s, "rg32i")) return VK_FORMAT_R32G32_SINT;
    if (!strcmp(s, "rgb32i")) return VK_FORMAT_R32G32B32_SINT;
    if (!strcmp(s, "rgba32i")) return VK_FORMAT_R32G32B32A32_SINT;
    if (!strcmp(s, "r32ui")) return VK_FORMAT_R32_UINT;
    if (!strcmp(s, "rg32ui")) return VK_FORMAT_R32G32_UINT;
    if (!strcmp(s, "rgb32ui")) return VK_FORMAT_R32G32B32_UINT;
    if (!strcmp(s, "rgba32ui")) return VK_FORMAT_R32G32B32A32_UINT;
    fprintf(stderr, "vkdraw: unknown attribute format in: %s\n", line);
    exit(1);
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
        char word[64] = {0}, a[64] = {0}, b[64] = {0}, c[64] = {0}, d[64] = {0}, e[64] = {0}, g[64] = {0};
        int k = sscanf(line, "%63s %63s %63s %63s %63s %63s %63s", word, a, b, c, d, e, g);
        if (!strcmp(word, "vs")) copy_path(cfg.vs, rest_after(line, 1), line);
        else if (!strcmp(word, "fs")) copy_path(cfg.fs, rest_after(line, 1), line);
        else if (!strcmp(word, "out")) copy_path(cfg.out, rest_after(line, 1), line);
        else if (!strcmp(word, "size") && k == 3) { cfg.width = (uint32_t)atoi(a); cfg.height = (uint32_t)atoi(b); }
        else if (!strcmp(word, "clear") && k == 5) {
            cfg.clear[0] = (float)atof(a); cfg.clear[1] = (float)atof(b);
            cfg.clear[2] = (float)atof(c); cfg.clear[3] = (float)atof(d);
        } else if (!strcmp(word, "vertices") && k >= 4) {
            cfg.vertex_count = (uint32_t)atoi(a);
            cfg.stride = (uint32_t)atoi(b);
            copy_path(cfg.vbuf, rest_after(line, 3), line);
        } else if (!strcmp(word, "attribute") && k == 4) {
            if (cfg.nattrs == MAX_ATTRS) die(1, "too many attributes");
            Attr *at = &cfg.attrs[cfg.nattrs++];
            at->location = (uint32_t)atoi(a);
            at->format = parse_format(b, line);
            at->offset = (uint32_t)atoi(c);
        } else if ((!strcmp(word, "uniform") || !strcmp(word, "storage")) && k >= 4) {
            if (cfg.nbinds == MAX_BINDINGS) die(1, "too many bindings");
            Binding *bd = &cfg.binds[cfg.nbinds++];
            bd->kind = word[0] == 'u' ? BIND_UNIFORM : BIND_STORAGE;
            bd->binding = (uint32_t)atoi(a);
            bd->stages = parse_stages(b, line);
            copy_path(bd->path, rest_after(line, 3), line);
        } else if (!strcmp(word, "global") && k == 3) {
            cfg.global_images = (uint32_t)strtoul(a, NULL, 0);
            cfg.global_samplers = (uint32_t)strtoul(b, NULL, 0);
            if (!cfg.global_images || !cfg.global_samplers) die(1, "the global arrays need a length");
        } else if (!strcmp(word, "image") && k == 7) {
            if (cfg.nimages == MAX_ENTRIES) die(1, "too many images");
            GlobalImage *gi = &cfg.images[cfg.nimages++];
            gi->entry = (uint32_t)strtoul(a, NULL, 0);
            if (!strcmp(b, "2d")) gi->view = VK_IMAGE_VIEW_TYPE_2D;
            else if (!strcmp(b, "2darray")) gi->view = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
            else if (!strcmp(b, "3d")) gi->view = VK_IMAGE_VIEW_TYPE_3D;
            else if (!strcmp(b, "cube")) gi->view = VK_IMAGE_VIEW_TYPE_CUBE;
            else die(1, "image type is 2d, 2darray, 3d or cube");
            gi->width = (uint32_t)atoi(c);
            gi->height = (uint32_t)atoi(d);
            gi->layers = (uint32_t)atoi(e);
            if (!gi->width || !gi->height || !gi->layers || (gi->view == VK_IMAGE_VIEW_TYPE_CUBE && gi->layers != 6) ||
                (gi->view == VK_IMAGE_VIEW_TYPE_2D && gi->layers != 1))
                die(1, "an image's size does not fit its type");
            copy_path(gi->path, rest_after(line, 6), line);
        } else if (!strcmp(word, "sampler") && k == 4) {
            if (cfg.nsamplers == MAX_ENTRIES) die(1, "too many samplers");
            GlobalSampler *gs = &cfg.samplers[cfg.nsamplers++];
            gs->entry = (uint32_t)strtoul(a, NULL, 0);
            if (!strcmp(b, "nearest")) gs->filter = VK_FILTER_NEAREST;
            else if (!strcmp(b, "linear")) gs->filter = VK_FILTER_LINEAR;
            else die(1, "sampler filter is nearest or linear");
            if (!strcmp(c, "clamp")) gs->address = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
            else if (!strcmp(c, "repeat")) gs->address = VK_SAMPLER_ADDRESS_MODE_REPEAT;
            else die(1, "sampler address mode is clamp or repeat");
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
        } else {
            fprintf(stderr, "vkdraw: line %u not understood: %s\n", lineno, line);
            exit(1);
        }
    }
    fclose(f);
    if (!cfg.vs[0] || !cfg.fs[0] || !cfg.out[0] || !cfg.vbuf[0]) die(1, "vs, fs, out and vertices are required");
    if (!cfg.width || !cfg.height || cfg.width > 16384 || cfg.height > 16384) die(1, "size is missing or out of range");
    if (!cfg.vertex_count) die(1, "vertices needs a count");
    if (!cfg.stride && cfg.nattrs) die(1, "attributes need a vertex stride");
}

static void *read_file(const char *path, size_t *size) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "vkdraw: cannot open %s\n", path); exit(1); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0) { fprintf(stderr, "vkdraw: %s is empty\n", path); exit(1); }
    void *p = malloc((size_t)n);
    if (!p || fread(p, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "vkdraw: cannot read %s\n", path); exit(1); }
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
        // The loader's own messages (third-party layers registered on the machine),
        // not a verdict on the draw: counted apart.
        loader_messages++;
        fprintf(stderr, "vkdraw: loader: %s\n", data->pMessage ? data->pMessage : "(no text)");
        return VK_FALSE;
    }
    if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT) { val_errors++; kind = "error"; }
    else if (severity & VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT) { val_warnings++; kind = "warning"; }
    fprintf(stderr, "vkdraw: validation %s: %s\n", kind, data->pMessage ? data->pMessage : "(no text)");
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
} Buffer;

static Buffer make_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const void *data, size_t data_size) {
    Buffer b = {.size = size};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = usage};
    VK_CHECK(vkCreateBuffer(dev, &bi, NULL, &b.buf));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, b.buf, &req);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                  .memoryTypeIndex = memory_type(req.memoryTypeBits,
                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, &b.mem));
    VK_CHECK(vkBindBufferMemory(dev, b.buf, b.mem, 0));
    VK_CHECK(vkMapMemory(dev, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map));
    memset(b.map, 0, (size_t)size);
    if (data) memcpy(b.map, data, data_size < size ? data_size : (size_t)size);
    return b;
}

typedef struct {
    VkImage img;
    VkDeviceMemory mem;
    VkImageView view;
} Image;

// An RGBA32F image of one level: 2D with `layers` array layers (6 for a cube), or 3D `layers` deep.
static Image make_image_typed(VkImageViewType view, uint32_t w, uint32_t h, uint32_t layers, VkImageUsageFlags usage) {
    Image im;
    int is3d = view == VK_IMAGE_VIEW_TYPE_3D;
    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                            .flags = view == VK_IMAGE_VIEW_TYPE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
                            .imageType = is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R32G32B32A32_SFLOAT, .extent = {w, h, is3d ? layers : 1}, .mipLevels = 1,
                            .arrayLayers = is3d ? 1 : layers, .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = usage, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VK_CHECK(vkCreateImage(dev, &ii, NULL, &im.img));
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, im.img, &req);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                  .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, &im.mem));
    VK_CHECK(vkBindImageMemory(dev, im.img, im.mem, 0));
    VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = im.img,
                                .viewType = view, .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, is3d ? 1 : layers}};
    VK_CHECK(vkCreateImageView(dev, &vi, NULL, &im.view));
    return im;
}

static Image make_image(uint32_t w, uint32_t h, VkImageUsageFlags usage) {
    return make_image_typed(VK_IMAGE_VIEW_TYPE_2D, w, h, 1, usage);
}

static void barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to,
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

static VkShaderModule load_module(const char *path) {
    size_t n;
    uint32_t *code = read_file(path, &n);
    if (n % 4 || code[0] != 0x07230203u) { fprintf(stderr, "vkdraw: %s is not SPIR-V\n", path); exit(1); }
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

// ---- main ---------------------------------------------------------------------

int main(int argc, char **argv) {
    if (argc != 2) die(1, "usage: vkdraw <draw.txt>");
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
    VkApplicationInfo app = {.sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .pApplicationName = "vkdraw",
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
    if (!has_extension(dexts, next, VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME)) die(2, "no VK_KHR_push_descriptor");
    free(dexts);

    // Subgroup operations the modules use, in the stages that use them.
    VkPhysicalDeviceSubgroupProperties sgp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SUBGROUP_PROPERTIES};
    VkPhysicalDeviceProperties2 p2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &sgp};
    vkGetPhysicalDeviceProperties2(gpu, &p2);
    if ((sgp.supportedOperations & cfg.subgroup_ops) != cfg.subgroup_ops) die(2, "a subgroup operation is missing");
    if (cfg.subgroup_ops && (sgp.supportedStages & (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT)) !=
                                (VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT))
        die(2, "subgroup operations are not supported in the vertex and fragment stages");

    // Features: exactly the ones asked for, plus dynamic rendering and synchronization2.
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
    for (uint32_t i = 0; i < cfg.nfeatures; i++) {
        const FeatureRef *ref = NULL;
        for (size_t j = 0; j < sizeof FEATURES / sizeof FEATURES[0]; j++)
            if (!strcmp(FEATURES[j].name, cfg.features[i])) ref = &FEATURES[j];
        if (!ref) { fprintf(stderr, "vkdraw: unknown feature %s\n", cfg.features[i]); exit(1); }
        if (!*feature_slot(&have, ref)) { fprintf(stderr, "vkdraw: the device lacks %s\n", cfg.features[i]); exit(2); }
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
    const char *dext[1] = {VK_KHR_PUSH_DESCRIPTOR_EXTENSION_NAME};
    VkDeviceCreateInfo dci = {.sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pNext = &want.f2, .queueCreateInfoCount = 1,
                              .pQueueCreateInfos = &qci, .enabledExtensionCount = 1, .ppEnabledExtensionNames = dext};
    VK_CHECK(vkCreateDevice(gpu, &dci, NULL, &dev));
    volkLoadDevice(dev);
    VkQueue queue;
    vkGetDeviceQueue(dev, family, 0, &queue);

    // Resources.
    size_t vsize;
    Buffer vbuf = {0};
    if (cfg.stride) {
        void *vdata = read_file(cfg.vbuf, &vsize);
        if (vsize < (size_t)cfg.vertex_count * cfg.stride) die(1, "the vertex file is shorter than count times stride");
        vbuf = make_buffer(vsize, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, vdata, vsize);
        free(vdata);
    }

    VkPhysicalDeviceDescriptorIndexingProperties dip = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_DESCRIPTOR_INDEXING_PROPERTIES};
    VkPhysicalDeviceProperties2 p2b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2, .pNext = &dip};
    vkGetPhysicalDeviceProperties2(gpu, &p2b);
    if (dip.maxPerStageDescriptorUpdateAfterBindSampledImages < cfg.global_images ||
        dip.maxDescriptorSetUpdateAfterBindSampledImages < cfg.global_images ||
        dip.maxPerStageDescriptorUpdateAfterBindSamplers < cfg.global_samplers ||
        dip.maxDescriptorSetUpdateAfterBindSamplers < cfg.global_samplers)
        die(2, "the global arrays are longer than the device's update-after-bind limits");

    // Set 1: the draw's buffers, pushed.
    Buffer bufs[MAX_BINDINGS] = {0};
    VkDescriptorSetLayoutBinding lb[MAX_BINDINGS];
    VkWriteDescriptorSet writes[MAX_BINDINGS];
    VkDescriptorBufferInfo binfo[MAX_BINDINGS];
    for (uint32_t i = 0; i < cfg.nbinds; i++) {
        Binding *bd = &cfg.binds[i];
        for (uint32_t j = 0; j < i; j++)
            if (cfg.binds[j].binding == bd->binding) die(1, "a binding appears twice");
        size_t n;
        void *data = read_file(bd->path, &n);
        VkDescriptorType type = bd->kind == BIND_UNIFORM ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bufs[i] = make_buffer(n, bd->kind == BIND_UNIFORM ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT
                                                          : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, data, n);
        binfo[i] = (VkDescriptorBufferInfo){bufs[i].buf, 0, n};
        free(data);
        lb[i] = (VkDescriptorSetLayoutBinding){bd->binding, type, 1, bd->stages, NULL};
        writes[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = bd->binding,
                                           .descriptorCount = 1, .descriptorType = type, .pBufferInfo = &binfo[i]};
    }
    VkDescriptorSetLayoutCreateInfo dlci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
                                            .flags = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR,
                                            .bindingCount = cfg.nbinds, .pBindings = lb};
    VkDescriptorSetLayout dsl[2];
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &dlci, NULL, &dsl[1]));

    // Set 0: the global arrays, update-after-bind and partially bound, written once before the draw.
    VkShaderStageFlags gst = VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT;
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

    Image texs[MAX_ENTRIES] = {0};
    Buffer staging[MAX_ENTRIES] = {0};
    VkSampler samplers[MAX_ENTRIES] = {0};
    VkDescriptorImageInfo giinfo[MAX_ENTRIES], gsinfo[MAX_ENTRIES];
    VkWriteDescriptorSet gw[2 * MAX_ENTRIES];
    uint32_t ngw = 0;
    for (uint32_t i = 0; i < cfg.nimages; i++) {
        GlobalImage *gi = &cfg.images[i];
        if (gi->entry >= cfg.global_images) die(1, "an image entry is past the global array");
        size_t n;
        void *data = read_file(gi->path, &n);
        if (n != (size_t)gi->width * gi->height * gi->layers * 16) die(1, "an image file is not W x H x L x 16 bytes");
        staging[i] = make_buffer(n, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, data, n);
        free(data);
        texs[i] = make_image_typed(gi->view, gi->width, gi->height, gi->layers,
                                   VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT);
        giinfo[i] = (VkDescriptorImageInfo){VK_NULL_HANDLE, texs[i].view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        gw[ngw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = gset, .dstBinding = 0,
                                           .dstArrayElement = gi->entry, .descriptorCount = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &giinfo[i]};
    }
    for (uint32_t i = 0; i < cfg.nsamplers; i++) {
        GlobalSampler *gs = &cfg.samplers[i];
        if (gs->entry >= cfg.global_samplers) die(1, "a sampler entry is past the global array");
        VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = gs->filter,
                                  .minFilter = gs->filter, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                  .addressModeU = gs->address, .addressModeV = gs->address,
                                  .addressModeW = gs->address, .maxLod = 0.0f};
        VK_CHECK(vkCreateSampler(dev, &si, NULL, &samplers[i]));
        gsinfo[i] = (VkDescriptorImageInfo){samplers[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        gw[ngw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = gset, .dstBinding = 1,
                                           .dstArrayElement = gs->entry, .descriptorCount = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &gsinfo[i]};
    }
    if (ngw) vkUpdateDescriptorSets(dev, ngw, gw, 0, NULL);

    VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, PUSH_BYTES};
    VkPipelineLayoutCreateInfo plci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = 2,
                                       .pSetLayouts = dsl, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VkPipelineLayout layout;
    VK_CHECK(vkCreatePipelineLayout(dev, &plci, NULL, &layout));

    Image target = make_image(cfg.width, cfg.height, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    Buffer readback = make_buffer((VkDeviceSize)cfg.width * cfg.height * 16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, NULL, 0);

    // Pipeline.
    VkShaderModule vsm = load_module(cfg.vs), fsm = load_module(cfg.fs);
    VkPipelineShaderStageCreateInfo stages[2] = {
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_VERTEX_BIT, .module = vsm, .pName = "main"},
        {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_FRAGMENT_BIT, .module = fsm, .pName = "main"},
    };
    VkVertexInputBindingDescription vib = {0, cfg.stride, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription via[MAX_ATTRS];
    for (uint32_t i = 0; i < cfg.nattrs; i++)
        via[i] = (VkVertexInputAttributeDescription){cfg.attrs[i].location, 0, cfg.attrs[i].format, cfg.attrs[i].offset};
    VkPipelineVertexInputStateCreateInfo vis = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO,
                                                .vertexBindingDescriptionCount = cfg.stride ? 1 : 0,
                                                .pVertexBindingDescriptions = &vib,
                                                .vertexAttributeDescriptionCount = cfg.nattrs, .pVertexAttributeDescriptions = via};
    VkPipelineInputAssemblyStateCreateInfo ias = {.sType = VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO,
                                                  .topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST};
    VkViewport vp = {0, 0, (float)cfg.width, (float)cfg.height, 0, 1};
    VkRect2D sc = {{0, 0}, {cfg.width, cfg.height}};
    VkPipelineViewportStateCreateInfo vps = {.sType = VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO,
                                             .viewportCount = 1, .pViewports = &vp, .scissorCount = 1, .pScissors = &sc};
    VkPipelineRasterizationStateCreateInfo rs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO,
                                                 .polygonMode = VK_POLYGON_MODE_FILL, .cullMode = VK_CULL_MODE_NONE,
                                                 .frontFace = VK_FRONT_FACE_COUNTER_CLOCKWISE, .lineWidth = 1.0f};
    VkPipelineMultisampleStateCreateInfo ms = {.sType = VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO,
                                               .rasterizationSamples = VK_SAMPLE_COUNT_1_BIT};
    VkPipelineColorBlendAttachmentState cba = {.colorWriteMask = 0xF};
    VkPipelineColorBlendStateCreateInfo cbs = {.sType = VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO,
                                               .attachmentCount = 1, .pAttachments = &cba};
    VkFormat cfmt = VK_FORMAT_R32G32B32A32_SFLOAT;
    VkPipelineRenderingCreateInfo prci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO,
                                          .colorAttachmentCount = 1, .pColorAttachmentFormats = &cfmt};
    VkGraphicsPipelineCreateInfo gpci = {.sType = VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO, .pNext = &prci,
                                         .stageCount = 2, .pStages = stages, .pVertexInputState = &vis,
                                         .pInputAssemblyState = &ias, .pViewportState = &vps, .pRasterizationState = &rs,
                                         .pMultisampleState = &ms, .pColorBlendState = &cbs, .layout = layout};
    VkPipeline pipe;
    VK_CHECK(vkCreateGraphicsPipelines(dev, VK_NULL_HANDLE, 1, &gpci, NULL, &pipe));

    // Commands: texture uploads, the draw, the readback.
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
    for (uint32_t i = 0; i < cfg.nimages; i++) {
        barrier(cb, texs[i].img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT);
        int is3d = cfg.images[i].view == VK_IMAGE_VIEW_TYPE_3D;
        VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, is3d ? 1 : cfg.images[i].layers},
                                  .imageExtent = {cfg.images[i].width, cfg.images[i].height, is3d ? cfg.images[i].layers : 1}};
        vkCmdCopyBufferToImage(cb, staging[i].buf, texs[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        barrier(cb, texs[i].img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL,
                VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    barrier(cb, target.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
            VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
            VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo att = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = target.view,
                                     .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    memcpy(att.clearValue.color.float32, cfg.clear, sizeof cfg.clear);
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = sc, .layerCount = 1,
                          .colorAttachmentCount = 1, .pColorAttachments = &att};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 0, 1, &gset, 0, NULL);
    if (cfg.nbinds) vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, layout, 1, cfg.nbinds, writes);
    vkCmdPushConstants(cb, layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, PUSH_BYTES, cfg.push);
    VkDeviceSize zero = 0;
    if (cfg.stride) vkCmdBindVertexBuffers(cb, 0, 1, &vbuf.buf, &zero);
    vkCmdDraw(cb, cfg.vertex_count, 1, 0, 0);
    vkCmdEndRendering(cb);
    barrier(cb, target.img, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
            VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
            VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy back = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                              .imageExtent = {cfg.width, cfg.height, 1}};
    vkCmdCopyImageToBuffer(cb, target.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &back);
    VkBufferMemoryBarrier2 hb = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                 .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .buffer = readback.buf, .size = VK_WHOLE_SIZE};
    VkDependencyInfo hdi = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1,
                            .pBufferMemoryBarriers = &hb};
    vkCmdPipelineBarrier2(cb, &hdi);
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
    vkDestroyShaderModule(dev, vsm, NULL);
    vkDestroyShaderModule(dev, fsm, NULL);
    vkDestroyPipelineLayout(dev, layout, NULL);
    vkDestroyDescriptorPool(dev, gpool, NULL);
    vkDestroyDescriptorSetLayout(dev, dsl[0], NULL);
    vkDestroyDescriptorSetLayout(dev, dsl[1], NULL);
    Buffer *all[] = {&vbuf, &readback};
    for (size_t i = 0; i < 2; i++) {
        if (!all[i]->buf) continue;
        vkDestroyBuffer(dev, all[i]->buf, NULL);
        vkFreeMemory(dev, all[i]->mem, NULL);
    }
    for (uint32_t i = 0; i < cfg.nbinds; i++) {
        vkDestroyBuffer(dev, bufs[i].buf, NULL);
        vkFreeMemory(dev, bufs[i].mem, NULL);
    }
    for (uint32_t i = 0; i < cfg.nimages; i++) {
        vkDestroyImageView(dev, texs[i].view, NULL);
        vkDestroyImage(dev, texs[i].img, NULL);
        vkFreeMemory(dev, texs[i].mem, NULL);
        vkDestroyBuffer(dev, staging[i].buf, NULL);
        vkFreeMemory(dev, staging[i].mem, NULL);
    }
    for (uint32_t i = 0; i < cfg.nsamplers; i++) vkDestroySampler(dev, samplers[i], NULL);
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
           "\"validation_errors\": %u, \"validation_warnings\": %u, \"loader_messages\": %u}\n",
           props.deviceName, driver, VK_API_VERSION_MAJOR(props.apiVersion), VK_API_VERSION_MINOR(props.apiVersion),
           VK_API_VERSION_PATCH(props.apiVersion),
           cfg.validation == 0 ? "off" : cfg.validation == 1 ? "on" : cfg.validation == 2 ? "gpu" : "sync",
           val_errors, val_warnings, loader_messages);
    return 0;
}
