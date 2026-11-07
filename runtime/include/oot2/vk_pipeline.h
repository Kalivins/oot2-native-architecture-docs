// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The pipeline builder of the Vulkan back end (architecture D12, D13 and the build policy of section 6.1; ticket 60,
// research 20, 23 and 59): the one pipeline layout of the `uni` scheme, pipeline libraries per module, fast links per
// (pair, attachment formats), compute pipelines per module, and the per-player pipeline cache. The Vulkan back end
// calls it at run time and prepare's pipeline step calls it through a program linked with the runtime library, so a
// cache hit at launch sees the create info prepare built (the code that makes every create info is here, once).
//
// What it builds from: the shader cache pack (oot2/shader_cache.h), by the job key of the stage (research 35, decision
// 1: "<key16>.<variant>", the key an nvn_stage carries); modules and descriptions are read from the mapped pack and
// never copied. A key the pack lacks, a job that did not translate, a description of another schema or stage, a module
// that does not fit its stage's slots, one that declares a capability the device's required list does not give
// (oot2/vk_caps.h): each is a named fatal that prints the stage's shape, because no prepared module could ever answer
// it (research 35: the bytes the game passes at run time have not been compared with the archives).
//
// Every get is blocking: it creates what is missing before it returns, on the calling thread, and any number of threads
// may ask at once (two asking for the same pipeline build it once). No pipeline is created or released while a
// submission is in flight (section 6.1; research 16, D4): a get that would create one with the queue busy is a named
// fatal, the miss path of a later ticket drains first. A pipeline that exists already is returned with no such check.
//
// Layout of the one pipeline layout (research 20, finding A; research 23 and 59; xlate/host/UniLayout.cs is the source
// the tables here are tested against):
//   set 0  the global arrays, update-after-bind and partially bound: binding 0 the 33,025 sampled images, binding 1 the
//          4,096 samplers; the same set layout for graphics and compute (the bindless ticket fills it)
//   set 1  push descriptors, 32 at most: binding 0 the support buffer; graphics: vertex and mesh uniform 1 to 8 and
//          storage 9 to 13, pixel uniform 14 to 24, storage 25 to 29, storage images 30 and 31; compute: uniform 1 to 8,
//          storage 9 to 24, storage images 25 to 31
//   set 2  the task stage's buffers, written per draw (graphics layout only): uniform 0 to 7, storage 8 to 15
//   push constants, 128 bytes for every stage (graphics layout: vertex, fragment, task and mesh)
// The graphics layout holds the three sets for every graphics pipeline, vertex or mesh, so that a fragment library is
// built once per pixel module whatever it is linked to (libraries link only under identical layouts).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/nvn_program.h"
#include "oot2/shader_cache.h"
#include "oot2/shader_desc.h"
#include "oot2/vk_required.h"

// ---- the layout's tables (no device needed)
#define VKP_GLOBAL_IMAGES 33025u
#define VKP_GLOBAL_SAMPLERS 4096u
#define VKP_PUSH_BYTES 128u
#define VKP_SET1_MAX 32u
#define VKP_SET2_MAX 16u
#define VKP_MAX_DYNAMIC 48u
#define VKP_MAX_COLOR 8u

enum { VKP_GRAPHICS, VKP_COMPUTE };

// Set 0's two bindings and their binding flags (update-after-bind, partially bound); returns 2.
uint32_t vkp_set0_bindings(VkDescriptorSetLayoutBinding out[2], VkDescriptorBindingFlags flags[2]);
// Set 1's bindings for the graphics or compute layout, in binding order; returns how many (32 for both).
uint32_t vkp_set1_bindings(int layout, VkDescriptorSetLayoutBinding out[VKP_SET1_MAX]);
// Set 2's bindings (the task stage's buffers); returns 16.
uint32_t vkp_set2_bindings(VkDescriptorSetLayoutBinding out[VKP_SET2_MAX]);
// The push constant range of the graphics or compute layout.
VkPushConstantRange vkp_push_range(int layout);
// The states a library of `part` (VkGraphicsPipelineLibraryFlagBitsEXT, one bit) makes dynamic on a device with these
// facts (D12: vertex input, topology, viewport, culling, depth, stencil, blend, logic op wherever the device allows);
// returns how many.
uint32_t vkp_dynamic_states(VkGraphicsPipelineLibraryFlagsEXT part, const Oot2VkFacts *facts,
                            VkDynamicState out[VKP_MAX_DYNAMIC]);

// ---- the store: modules and descriptions by stage key (no device needed)
typedef struct VkpStage {
    const char *key;            // the job key, "<key16>.<variant>"
    const uint8_t *answers;     // the answers digest (32 bytes), NULL for the default answers
    const nvn_stage *shape;     // what to print when the stage cannot be served; may be NULL
} VkpStage;

typedef struct VkpModule {
    const uint32_t *code;       // SPIR-V words, inside the mapped pack
    uint32_t bytes;
    int from;                   // SHC_FROM_BASE or SHC_FROM_VARIANTS
    ShaderDesc desc;
} VkpModule;

// Open the packs (variants may be NULL). 1, or 0 with the reason in why.
int vkp_store_open(const char *base, const char *variants, char *why, size_t cap);
void vkp_store_close(void);
// The module of stage s, whose description must name stage kind `expect` (SHD_*) and say it fits its layout. A named
// fatal otherwise (no row, a failure record, a description that cannot be read or is of another stage).
void vkp_module(const VkpStage *s, int expect, VkpModule *out);

// ---- the device: layouts and the pipeline cache
typedef struct VkpLayouts {
    VkDescriptorSetLayout global;           // set 0, shared by both layouts
    VkDescriptorSetLayout graphics_push;    // set 1, graphics
    VkDescriptorSetLayout compute_push;     // set 1, compute
    VkDescriptorSetLayout task;             // set 2, graphics
    VkPipelineLayout graphics, compute;
} VkpLayouts;

// The layouts alone (vk_layout.c), on the back end's device: the tests make them without a store. vkp_open makes them
// itself.
void vkp_layouts_create(VkpLayouts *out);
void vkp_layouts_destroy(VkpLayouts *l);

typedef enum VkpCacheState {
    VKP_CACHE_NONE,            // no file: a first start, an empty cache
    VKP_CACHE_LOADED,          // the file's header names this device and driver: its data seeds the cache
    VKP_CACHE_OTHER_DRIVER,    // the header names another device or driver: dropped, the cache starts empty
    VKP_CACHE_UNREADABLE,      // the file is not a pipeline cache of header version one (or is cut short): dropped
} VkpCacheState;

// Judge cache file data against a device's properties: VKP_CACHE_LOADED when the header is version one and names
// this vendor, device and pipelineCacheUUID, else why it is refused. Nothing is created.
VkpCacheState vkp_cache_judge(const void *data, size_t size, const VkPhysicalDeviceProperties *props);

// After gpu_vk_open and vkp_store_open: the layouts, the vertex input library and the pipeline cache, seeded from the
// file at cache_path when it is valid for this device and driver (NULL: no file). Once.
void vkp_open(const char *cache_path);
// Destroy everything made (the queue must be idle). The cache is not saved: vkp_cache_save does that.
void vkp_close(void);
const VkpLayouts *vkp_layouts(void);
VkpCacheState vkp_cache_state(void);
// Write the cache to the path vkp_open was given, through a temporary file and a rename, so that an interrupted write
// leaves the old file whole. 1, or 0 with the reason in why.
int vkp_cache_save(char *why, size_t cap);

// ---- pipelines
typedef struct VkpFormats {     // the attachments a graphics pipeline draws into
    uint32_t ncolor;            // 0 to VKP_MAX_COLOR
    VkFormat color[VKP_MAX_COLOR];
    VkFormat depth, stencil;    // VK_FORMAT_UNDEFINED for none
} VkpFormats;

// The graphics pipeline of a vertex and a pixel stage drawing into these formats: the fast link of the vertex stage's
// pre-rasterization library (padded with the outputs the pixel stage reads and it does not write, keyed by the
// module and the pad set), the pixel stage's fragment library, the output library of the formats and the shared vertex
// input library. Created once, then kept until a vkp_release of either stage or vkp_close.
VkPipeline vkp_graphics(const VkpStage *vs, const VkpStage *ps, const VkpFormats *formats);
// The compute pipeline of a compute stage.
VkPipeline vkp_compute(const VkpStage *cs);
// Tests only: while non-zero, a vertex module is linked as translated, with no padding (the counter-run of the padding
// rule: the validation layer then reports VUID-RuntimeSpirv-OpEntryPoint-08743 for a pixel input nothing writes).
extern int vkp_unpad;
// Destroy every library and pipeline that was made from the stage with this key (for program_release, when no program
// holds it). The caller guarantees that no submission in flight uses them.
void vkp_release(const char *key);

typedef struct VkpStats {
    // Libraries made (the shared vertex input library, the pre-rasterization and fragment libraries, the output
    // libraries) and how many the cache served; stage_* count only the libraries that hold a shader. The driver caches
    // what it compiled: a library with no shader and a fast link are never reported as hits.
    uint64_t libraries, library_hits;
    uint64_t stage_libraries, stage_library_hits;
    uint64_t links, link_hits;            // fast links
    uint64_t computes, compute_hits;
    uint64_t padded;                      // vertex libraries made from a padded module
    uint64_t resident;                    // objects held now
    uint64_t build_ns[3];                 // wall time in the driver: libraries, links, computes
} VkpStats;
void vkp_stats(VkpStats *out);
