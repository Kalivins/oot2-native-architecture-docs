// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Per-draw costs (spike S3, D12 and D13), with synthetic modules whose
// descriptor counts follow the corpus (tools/pipecost/pipe_cost.py writes them):
//
//   draws   records frames of draws through each variant and times the
//           recording: descriptor layout (kind) and bind method (pipeline or
//           shader objects), GPU time from timestamps;
//   replay  the same draws behind a replay that resolves guest ops (pipeline
//           lookup, backing-map search, uniform snapshot, texture handles),
//           with the Vulkan calls on the replay thread or on a recorder thread.
//
// Variants are `param variantN KIND BIND FIRST COUNT`: pairs FIRST..FIRST+COUNT-1
// of the job, KIND one of
//   dkb      one set per pipeline holding both stages (DKB Recomp): pushed when it
//            has at most 32 descriptors, else allocated from a per-frame pool;
//   d13      D13 as written: the global update-after-bind set plus the pair's
//            set pushed, with 16 bytes of push constants; a pair past 32 takes
//            an allocated set instead (counted as fallback draws);
//   uni      one layout for every pipeline: the global set, a push set of
//            buffers in fixed per-stage slots (`param uni_set N`), texture
//            indices in push constants; a pair's set lists the buffers it uses
//            and its textures as bindings from 1000 (index slot = binding - 1000);
//   unitpl   uni, pushed through a descriptor update template per pair;
// and BIND one of mono, fast (libraries, fast link), opt (libraries, optimised
// link), objects (unlinked shader objects). The global set is the job's
// `global`, always set 0 here; the per-draw set is set 1 (set 0 for dkb).

#include "vkpc.h"

#include <stdlib.h>
#include <string.h>

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#define TEX_FIRST 1000
#define TEXTURES 64
#define SAMPLERS 16
#define SLOTS 2
#define RING_BYTES (64u << 20)
#define TARGET 1024
#define MAX_VARIANTS 16

enum { K_DKB, K_D13, K_UNI, K_UNITPL };
enum { B_MONO, B_FAST, B_OPT, B_OBJECTS };
static const char *KIND_NAMES[] = {"dkb", "d13", "uni", "unitpl"};
static const char *BIND_NAMES[] = {"mono", "fast", "opt", "objects"};

typedef struct {
    int kind, bind;
    uint32_t first, count;
    VkPipelineLayout *layout;           // per pair
    VkDescriptorSetLayout *set;         // per pair: its per-draw set (dkb, d13)
    uint32_t *pushed;                   // per pair: that set is pushed
    VkDescriptorSetLayout uni_set;
    VkPipelineLayout uni_layout;
    VkDescriptorUpdateTemplate *tpl;    // unitpl, per pair
    VkPipeline *pipe;
    VkShaderEXT *so;                    // 2 per pair
    VkPipeline libs[4096];
    uint32_t nlibs;
} Variant;

typedef struct {
    uint32_t binding;
    VkDescriptorType type;
    VkDeviceSize offset, range;
} PBuf;

// One draw, resolved: what the recording needs, nothing else. The first three
// words are serialised as they are by the replay's packet ring.
typedef struct {
    uint32_t pair;  // index within the variant; UINT32_MAX and UINT32_MAX - 1 mark frames
    uint32_t nbuf, ntex;
    PBuf b[40];
    uint32_t tex_binding[32];
    uint32_t tex_index[32];
} Packet;

static struct {
    Job *job;
    VkDescriptorSetLayout global_layout;
    VkDescriptorPool global_pool;
    VkDescriptorSet global_set;
    VkBuffer ring;
    VkDeviceMemory ring_mem;
    uint8_t *ring_map;
    VkImage tex[TEXTURES];
    VkImageView view[TEXTURES];
    VkDeviceMemory tex_mem;
    VkSampler sampler[SAMPLERS];
    VkImage target;
    VkImageView target_view;
    VkDeviceMemory target_mem;
    VkCommandPool cpool[SLOTS];
    VkCommandBuffer cb[SLOTS];
    VkFence fence[SLOTS];
    VkDescriptorPool dpool[SLOTS];
    VkQueryPool queries;
    VkShaderModule *mods;
    uint32_t draws, frames, warmup;
    Variant var[MAX_VARIANTS];
    uint32_t nvar;
} X;

static long fallback_draws;

// ---- setup ---------------------------------------------------------------------------

static VkDeviceMemory alloc_bind_images(VkImage *imgs, uint32_t n) {
    VkMemoryRequirements req;
    vkGetImageMemoryRequirements(dev, imgs[0], &req);
    VkDeviceSize stride = (req.size + req.alignment - 1) & ~(req.alignment - 1);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = stride * n,
                               .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory mem;
    VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, &mem));
    for (uint32_t i = 0; i < n; i++) VK_CHECK(vkBindImageMemory(dev, imgs[i], mem, stride * i));
    return mem;
}

static void image_barrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                               .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                               .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,
                               .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = img,
                               .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
}

static void init_images(VkCommandBuffer cb) {
    VkClearColorValue grey = {.float32 = {0.5f, 0.25f, 0.75f, 1.0f}};
    VkImageSubresourceRange r = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    for (int i = 0; i < TEXTURES; i++) {
        image_barrier(cb, X.tex[i], VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL);
        vkCmdClearColorImage(cb, X.tex[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &grey, 1, &r);
        image_barrier(cb, X.tex[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
    }
    image_barrier(cb, X.target, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
}

static void ctx_init(Job *job) {
    X.job = job;
    X.draws = (uint32_t)job_param_int(job, "draws", 3000);
    X.frames = (uint32_t)job_param_int(job, "frames", 300);
    X.warmup = (uint32_t)job_param_int(job, "warmup", 30);
    if (X.frames <= X.warmup + SLOTS || X.frames > 4096) die(1, "frames must be above warmup and at most 4096");
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = RING_BYTES,
                             .usage = VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT};
    VK_CHECK(vkCreateBuffer(dev, &bi, NULL, &X.ring));
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(dev, X.ring, &req);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    VK_CHECK(vkAllocateMemory(dev, &alloc, NULL, &X.ring_mem));
    VK_CHECK(vkBindBufferMemory(dev, X.ring, X.ring_mem, 0));
    VK_CHECK(vkMapMemory(dev, X.ring_mem, 0, VK_WHOLE_SIZE, 0, (void **)&X.ring_map));
    for (uint32_t i = 0; i < RING_BYTES / 4; i++) ((float *)X.ring_map)[i] = (float)(i % 97) * 0.001f;

    VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {16, 16, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT};
    for (int i = 0; i < TEXTURES; i++) VK_CHECK(vkCreateImage(dev, &ii, NULL, &X.tex[i]));
    X.tex_mem = alloc_bind_images(X.tex, TEXTURES);
    for (int i = 0; i < TEXTURES; i++) {
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = X.tex[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        VK_CHECK(vkCreateImageView(dev, &vi, NULL, &X.view[i]));
    }
    for (int i = 0; i < SAMPLERS; i++) {
        VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO,
                                  .magFilter = i & 1 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
                                  .minFilter = i & 2 ? VK_FILTER_LINEAR : VK_FILTER_NEAREST,
                                  .addressModeU = VK_SAMPLER_ADDRESS_MODE_REPEAT, .addressModeV = VK_SAMPLER_ADDRESS_MODE_REPEAT,
                                  .addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT, .maxLod = (float)(i >> 2)};
        VK_CHECK(vkCreateSampler(dev, &si, NULL, &X.sampler[i]));
    }
    ii.extent = (VkExtent3D){TARGET, TARGET, 1};
    ii.usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT;
    VK_CHECK(vkCreateImage(dev, &ii, NULL, &X.target));
    X.target_mem = alloc_bind_images(&X.target, 1);
    VkImageViewCreateInfo tv = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = X.target,
                                .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = VK_FORMAT_R8G8B8A8_UNORM,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VK_CHECK(vkCreateImageView(dev, &tv, NULL, &X.target_view));

    for (int s = 0; s < SLOTS; s++) {
        VkCommandPoolCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = queue_family};
        VK_CHECK(vkCreateCommandPool(dev, &ci, NULL, &X.cpool[s]));
        VkCommandBufferAllocateInfo cai = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = X.cpool[s],
                                           .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY, .commandBufferCount = 1};
        VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &X.cb[s]));
        VkFenceCreateInfo fi = {.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = VK_FENCE_CREATE_SIGNALED_BIT};
        VK_CHECK(vkCreateFence(dev, &fi, NULL, &X.fence[s]));
        VkDescriptorPoolSize ps[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, X.draws * 32},
                                      {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, X.draws * 12},
                                      {VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, X.draws * 32}};
        VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO, .maxSets = X.draws,
                                          .poolSizeCount = 3, .pPoolSizes = ps};
        VK_CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &X.dpool[s]));
    }
    VkQueryPoolCreateInfo qi = {.sType = VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO, .queryType = VK_QUERY_TYPE_TIMESTAMP,
                                .queryCount = 2 * SLOTS};
    VK_CHECK(vkCreateQueryPool(dev, &qi, NULL, &X.queries));
    vkResetQueryPool(dev, X.queries, 0, 2 * SLOTS);

    VkCommandBufferBeginInfo cbi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                    .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(X.cb[0], &cbi));
    init_images(X.cb[0]);
    VK_CHECK(vkEndCommandBuffer(X.cb[0]));
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &X.cb[0]};
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    VK_CHECK(vkResetCommandPool(dev, X.cpool[0], 0));

    // The global set: the job's arrays (33,025 sampled images, 4,096 samplers),
    // update-after-bind and partially bound; the first entries are written.
    if (!job->has_global || job->global_set != 0) die(1, "draws needs `global 0`");
    X.global_layout = set_layout_create(&job->global, 0, 1);
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, 0}, {VK_DESCRIPTOR_TYPE_SAMPLER, 0}};
    for (uint32_t i = 0; i < job->global.n; i++)
        ps[job->global.b[i].type == VK_DESCRIPTOR_TYPE_SAMPLER].descriptorCount += job->global.b[i].count;
    VkDescriptorPoolCreateInfo dpi = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1,
                                      .poolSizeCount = 2, .pPoolSizes = ps};
    VK_CHECK(vkCreateDescriptorPool(dev, &dpi, NULL, &X.global_pool));
    VkDescriptorSetAllocateInfo sai = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                       .descriptorPool = X.global_pool, .descriptorSetCount = 1, .pSetLayouts = &X.global_layout};
    VK_CHECK(vkAllocateDescriptorSets(dev, &sai, &X.global_set));
    VkDescriptorImageInfo iv[TEXTURES], sv[SAMPLERS];
    for (int i = 0; i < TEXTURES; i++) iv[i] = (VkDescriptorImageInfo){VK_NULL_HANDLE, X.view[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    for (int i = 0; i < SAMPLERS; i++) sv[i] = (VkDescriptorImageInfo){X.sampler[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
    double t0 = now_ms();
    for (int i = 0; i < TEXTURES; i++) {
        // A texture registered with the game is one image write into the global set: timed one by one.
        VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = X.global_set, .dstBinding = 0,
                                  .dstArrayElement = (uint32_t)i, .descriptorCount = 1,
                                  .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &iv[i]};
        vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);
    }
    summary_num("global_image_write_us", (now_ms() - t0) * 1000.0 / TEXTURES);
    VkWriteDescriptorSet w = {.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = X.global_set, .dstBinding = 1,
                              .descriptorCount = SAMPLERS, .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = sv};
    vkUpdateDescriptorSets(dev, 1, &w, 0, NULL);

    X.mods = calloc(job->nmodules, sizeof *X.mods);
    for (uint32_t i = 0; i < job->nmodules; i++) X.mods[i] = module_create(&job->modules[i]);
}

static void ctx_destroy(void) {
    VK_CHECK(vkDeviceWaitIdle(dev));
    for (uint32_t i = 0; i < X.job->nmodules; i++) vkDestroyShaderModule(dev, X.mods[i], NULL);
    free(X.mods);
    vkDestroyDescriptorPool(dev, X.global_pool, NULL);
    vkDestroyDescriptorSetLayout(dev, X.global_layout, NULL);
    vkDestroyQueryPool(dev, X.queries, NULL);
    for (int s = 0; s < SLOTS; s++) {
        vkDestroyDescriptorPool(dev, X.dpool[s], NULL);
        vkDestroyFence(dev, X.fence[s], NULL);
        vkDestroyCommandPool(dev, X.cpool[s], NULL);
    }
    vkDestroyImageView(dev, X.target_view, NULL);
    vkDestroyImage(dev, X.target, NULL);
    vkFreeMemory(dev, X.target_mem, NULL);
    for (int i = 0; i < SAMPLERS; i++) vkDestroySampler(dev, X.sampler[i], NULL);
    for (int i = 0; i < TEXTURES; i++) {
        vkDestroyImageView(dev, X.view[i], NULL);
        vkDestroyImage(dev, X.tex[i], NULL);
    }
    vkFreeMemory(dev, X.tex_mem, NULL);
    vkDestroyBuffer(dev, X.ring, NULL);
    vkFreeMemory(dev, X.ring_mem, NULL);
}

// ---- variants -------------------------------------------------------------------------

static uint32_t set_count(const SetDesc *d) {
    uint32_t n = 0;
    for (uint32_t i = 0; i < d->n; i++) n += d->b[i].count;
    return n;
}

static VkPipelineLayout layout_of(VkDescriptorSetLayout *sets, uint32_t n) {
    VkPushConstantRange pcr = {VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, X.job->push_bytes};
    VkPipelineLayoutCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, .setLayoutCount = n,
                                     .pSetLayouts = sets, .pushConstantRangeCount = 1, .pPushConstantRanges = &pcr};
    VkPipelineLayout l;
    VK_CHECK(vkCreatePipelineLayout(dev, &ci, NULL, &l));
    return l;
}

static void variant_init(Variant *v) {
    Job *job = X.job;
    v->layout = calloc(v->count, sizeof *v->layout);
    v->set = calloc(v->count, sizeof *v->set);
    v->pushed = calloc(v->count, sizeof *v->pushed);
    v->pipe = calloc(v->count, sizeof *v->pipe);
    v->so = calloc(2 * v->count, sizeof *v->so);
    v->tpl = calloc(v->count, sizeof *v->tpl);
    if (v->kind == K_UNI || v->kind == K_UNITPL) {
        long u = job_param_int(job, "uni_set", -1);
        if (u < 0 || (uint32_t)u >= job->nsets) die(1, "uni needs param uni_set");
        if (set_count(&job->sets[u]) > 32) die(1, "the universal set holds more than 32 descriptors");
        v->uni_set = set_layout_create(&job->sets[u], VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR, 0);
        VkDescriptorSetLayout sets[2] = {X.global_layout, v->uni_set};
        v->uni_layout = layout_of(sets, 2);
    }
    double t0 = now_ms();
    for (uint32_t j = 0; j < v->count; j++) {
        Pair *p = &job->pairs[v->first + j];
        const SetDesc *d = &job->sets[p->set];
        VkDescriptorSetLayout sets[2];
        uint32_t ns;
        if (v->kind == K_DKB || v->kind == K_D13) {
            v->pushed[j] = set_count(d) <= 32;
            v->set[j] = set_layout_create(d, v->pushed[j] ? VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT_KHR : 0, 0);
            if (v->kind == K_DKB) {
                sets[0] = v->set[j];
                ns = 1;
            } else {
                sets[0] = X.global_layout;
                sets[1] = v->set[j];
                ns = 2;
            }
            v->layout[j] = layout_of(sets, ns);
        } else {
            v->pushed[j] = 1;
            v->layout[j] = v->uni_layout;
            sets[0] = X.global_layout;
            sets[1] = v->uni_set;
            ns = 2;
            if (v->kind == K_UNITPL) {
                VkDescriptorUpdateTemplateEntry e[MAX_BINDS];
                uint32_t ne = 0;
                for (uint32_t k = 0; k < d->n; k++)
                    if (d->b[k].binding < TEX_FIRST) {
                        e[ne] = (VkDescriptorUpdateTemplateEntry){d->b[k].binding, 0, 1, d->b[k].type,
                                                                  ne * sizeof(VkDescriptorBufferInfo), sizeof(VkDescriptorBufferInfo)};
                        ne++;
                    }
                VkDescriptorUpdateTemplateCreateInfo tci = {
                    .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_UPDATE_TEMPLATE_CREATE_INFO, .descriptorUpdateEntryCount = ne,
                    .pDescriptorUpdateEntries = e, .templateType = VK_DESCRIPTOR_UPDATE_TEMPLATE_TYPE_PUSH_DESCRIPTORS_KHR,
                    .descriptorSetLayout = v->uni_set, .pipelineBindPoint = VK_PIPELINE_BIND_POINT_GRAPHICS,
                    .pipelineLayout = v->uni_layout, .set = 1};
                VK_CHECK(vkCreateDescriptorUpdateTemplate(dev, &tci, NULL, &v->tpl[j]));
            }
        }
        if (v->bind == B_OBJECTS)
            shader_objects_for(job, &job->modules[p->vs], &job->modules[p->fs], sets, ns, &v->so[2 * j]);
        else {
            if (v->nlibs + 4 > sizeof v->libs / sizeof v->libs[0]) die(1, "too many libraries");
            v->pipe[j] = pipeline_for(job, v->bind == B_MONO ? 0 : v->bind == B_FAST ? 1 : 2, X.mods[p->vs], X.mods[p->fs],
                                      v->layout[j], p->targets, job->formats[p->format], v->libs, &v->nlibs);
        }
    }
    fprintf(stderr, "vkpipecost: variant %s/%s: %u pipelines in %.0f ms\n", KIND_NAMES[v->kind], BIND_NAMES[v->bind],
            v->count, now_ms() - t0);
}

static void variant_destroy(Variant *v) {
    VK_CHECK(vkDeviceWaitIdle(dev));
    for (uint32_t j = 0; j < v->count; j++) {
        if (v->pipe[j]) vkDestroyPipeline(dev, v->pipe[j], NULL);
        if (v->so[2 * j]) {
            vkDestroyShaderEXT(dev, v->so[2 * j], NULL);
            vkDestroyShaderEXT(dev, v->so[2 * j + 1], NULL);
        }
        if (v->tpl[j]) vkDestroyDescriptorUpdateTemplate(dev, v->tpl[j], NULL);
        if (v->kind == K_DKB || v->kind == K_D13) {
            vkDestroyPipelineLayout(dev, v->layout[j], NULL);
            vkDestroyDescriptorSetLayout(dev, v->set[j], NULL);
        }
    }
    for (uint32_t i = 0; i < v->nlibs; i++) vkDestroyPipeline(dev, v->libs[i], NULL);
    if (v->uni_layout) vkDestroyPipelineLayout(dev, v->uni_layout, NULL);
    if (v->uni_set) vkDestroyDescriptorSetLayout(dev, v->uni_set, NULL);
    free(v->layout);
    free(v->set);
    free(v->pushed);
    free(v->pipe);
    free(v->so);
    free(v->tpl);
}

static void variants_read(void) {
    for (int i = 0; i < MAX_VARIANTS; i++) {
        char key[32], kind[32], bind[32];
        snprintf(key, sizeof key, "variant%d", i);
        const char *v = job_param(X.job, key, NULL);
        if (!v) continue;
        Variant *var = &X.var[X.nvar++];
        memset(var, 0, sizeof *var);
        if (sscanf(v, "%31s %31s %u %u", kind, bind, &var->first, &var->count) != 4) die(1, "%s: KIND BIND FIRST COUNT", key);
        var->kind = -1;
        var->bind = -1;
        for (int k = 0; k < 4; k++) {
            if (!strcmp(kind, KIND_NAMES[k])) var->kind = k;
            if (!strcmp(bind, BIND_NAMES[k])) var->bind = k;
        }
        if (var->kind < 0 || var->bind < 0 || var->first + var->count > X.job->npairs || !var->count)
            die(1, "%s is wrong: %s", key, v);
    }
    if (!X.nvar) die(1, "no variant");
}

// ---- recording -------------------------------------------------------------------------

typedef struct {
    uint32_t last_pair;
    uint32_t slot;
} RecState;

static void begin_frame(Variant *v, uint32_t slot, RecState *rs) {
    VkCommandBuffer cb = X.cb[slot];
    VK_CHECK(vkResetCommandPool(dev, X.cpool[slot], 0));
    VK_CHECK(vkResetDescriptorPool(dev, X.dpool[slot], 0));
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    vkCmdResetQueryPool(cb, X.queries, 2 * slot, 2);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, X.queries, 2 * slot);
    VkRenderingAttachmentInfo att = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = X.target_view,
                                     .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {TARGET, TARGET}},
                          .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &att};
    vkCmdBeginRendering(cb, &ri);
    set_all_dynamic_state(cb, TARGET, TARGET, 1, v->bind == B_OBJECTS);
    vkCmdSetVertexInputEXT(cb, 0, NULL, 0, NULL);
    if (v->bind == B_OBJECTS) {
        // Stages the device supports and the draws do not use are bound empty, once.
        VkShaderStageFlagBits unused[3] = {VK_SHADER_STAGE_TESSELLATION_CONTROL_BIT,
                                           VK_SHADER_STAGE_TESSELLATION_EVALUATION_BIT, VK_SHADER_STAGE_GEOMETRY_BIT};
        VkShaderEXT none[3] = {VK_NULL_HANDLE, VK_NULL_HANDLE, VK_NULL_HANDLE};
        vkCmdBindShadersEXT(cb, 3, unused, none);
    }
    if (v->kind != K_DKB)
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, v->kind == K_D13 ? v->layout[0] : v->uni_layout, 0, 1,
                                &X.global_set, 0, NULL);
    rs->last_pair = UINT32_MAX;
    rs->slot = slot;
}

static void end_frame(uint32_t slot) {
    VkCommandBuffer cb = X.cb[slot];
    vkCmdEndRendering(cb);
    vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, X.queries, 2 * slot + 1);
    VK_CHECK(vkEndCommandBuffer(cb));
}

static void submit(uint32_t slot) {
    VkSubmitInfo si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &X.cb[slot]};
    VK_CHECK(vkQueueSubmit(queue, 1, &si, X.fence[slot]));
}

static void record_draw(Variant *v, RecState *rs, const Packet *p) {
    VkCommandBuffer cb = X.cb[rs->slot];
    uint32_t j = p->pair;
    if (j != rs->last_pair) {
        if (v->bind == B_OBJECTS) {
            VkShaderStageFlagBits st[2] = {VK_SHADER_STAGE_VERTEX_BIT, VK_SHADER_STAGE_FRAGMENT_BIT};
            vkCmdBindShadersEXT(cb, 2, st, &v->so[2 * j]);
        } else vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, v->pipe[j]);
        rs->last_pair = j;
    }
    VkDescriptorBufferInfo bi[40];
    VkDescriptorImageInfo ii[32];
    VkWriteDescriptorSet w[72];
    uint32_t nw = 0;
    for (uint32_t k = 0; k < p->nbuf; k++) {
        bi[k] = (VkDescriptorBufferInfo){X.ring, p->b[k].offset, p->b[k].range};
        w[nw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = p->b[k].binding,
                                         .descriptorCount = 1, .descriptorType = p->b[k].type, .pBufferInfo = &bi[k]};
    }
    // Page table address (words 0, 1) and span base and count (words 2, 3), raw_table.h. Placeholder: no table is allocated, so a module that
    // really executes a raw access reads outside any table; allocate one before drawing raw-access modules
    uint32_t pc[32] = {0x10000000u, 0x1, 0, 1};
    if (v->kind == K_DKB || v->kind == K_D13) {
        for (uint32_t k = 0; k < p->ntex; k++) {
            ii[k] = (VkDescriptorImageInfo){X.sampler[p->tex_index[k] % SAMPLERS], X.view[p->tex_index[k] % TEXTURES],
                                            VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
            w[nw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = p->tex_binding[k],
                                             .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER,
                                             .pImageInfo = &ii[k]};
        }
        uint32_t set = v->kind == K_DKB ? 0 : 1;
        if (v->pushed[j]) vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, v->layout[j], set, nw, w);
        else {
            VkDescriptorSet ds;
            VkDescriptorSetAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
                                              .descriptorPool = X.dpool[rs->slot], .descriptorSetCount = 1,
                                              .pSetLayouts = &v->set[j]};
            VK_CHECK(vkAllocateDescriptorSets(dev, &alloc, &ds));
            for (uint32_t k = 0; k < nw; k++) w[k].dstSet = ds;
            vkUpdateDescriptorSets(dev, nw, w, 0, NULL);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, v->layout[j], set, 1, &ds, 0, NULL);
            fallback_draws++;
        }
        if (v->kind == K_D13)
            vkCmdPushConstants(cb, v->layout[j], VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16, pc);
    } else {
        uint32_t top = 0;
        for (uint32_t k = 0; k < p->ntex; k++) {
            uint32_t s = p->tex_binding[k] - TEX_FIRST;
            pc[4 + s] = p->tex_index[k];
            if (s + 1 > top) top = s + 1;
        }
        if (v->kind == K_UNI) vkCmdPushDescriptorSetKHR(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, v->uni_layout, 1, nw, w);
        else vkCmdPushDescriptorSetWithTemplateKHR(cb, v->tpl[j], v->uni_layout, 1, bi);
        vkCmdPushConstants(cb, v->uni_layout, VK_SHADER_STAGE_VERTEX_BIT | VK_SHADER_STAGE_FRAGMENT_BIT, 0, 16 + 4 * top, pc);
    }
    vkCmdDraw(cb, 3, 1, 0, 0);
}

// The order of pairs in a frame: runs of `run_length` draws of one pair, pairs shuffled.
static uint32_t *draw_sequence(uint32_t count) {
    uint32_t run = (uint32_t)job_param_int(X.job, "run_length", 1);
    uint32_t *perm = malloc(count * sizeof *perm), *seq = malloc(X.draws * sizeof *seq);
    uint64_t r = 0x9E3779B97F4A7C15ull ^ (uint64_t)job_param_int(X.job, "seed", 3);
    for (uint32_t i = 0; i < count; i++) perm[i] = i;
    for (uint32_t i = count - 1; i > 0; i--) {
        r ^= r << 13, r ^= r >> 7, r ^= r << 17;
        uint32_t k = (uint32_t)(r % (i + 1));
        uint32_t t = perm[i];
        perm[i] = perm[k];
        perm[k] = t;
    }
    for (uint32_t i = 0; i < X.draws; i++) seq[i] = perm[(i / (run ? run : 1)) % count];
    free(perm);
    return seq;
}

// A draw's packet as the draws mode builds it: buffers at fresh ring offsets,
// textures from the fixed set of images.
static void make_packet(Variant *v, uint32_t i, uint32_t j, uint32_t slot, Packet *p) {
    const SetDesc *d = &X.job->sets[X.job->pairs[v->first + j].set];
    p->pair = j;
    p->nbuf = p->ntex = 0;
    VkDeviceSize half = RING_BYTES / SLOTS;
    for (uint32_t k = 0; k < d->n; k++) {
        const Bind *b = &d->b[k];
        if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b->binding >= TEX_FIRST) {
            if (p->ntex == 32) die(1, "a pair has more than 32 textures");
            p->tex_binding[p->ntex] = b->binding;
            p->tex_index[p->ntex++] = (i * 7 + k) % TEXTURES;
        } else {
            if (p->nbuf == 40) die(1, "a pair has more than 40 buffers");
            VkDeviceSize off = slot * half + ((VkDeviceSize)(i * 40 + k) * 256) % (half - 256);
            p->b[p->nbuf++] = (PBuf){b->binding, b->type, off, 256};
        }
    }
}

static int cmp_double(const void *a, const void *b) {
    double x = *(const double *)a, y = *(const double *)b;
    return x < y ? -1 : x > y;
}

static double pct(const double *src, uint32_t n, double q) {
    double *v = malloc(n * sizeof *v);
    memcpy(v, src, n * sizeof *v);
    qsort(v, n, sizeof *v, cmp_double);
    uint32_t k = (uint32_t)(q / 100.0 * n + 0.999999);
    if (k < 1) k = 1;
    if (k > n) k = n;
    double r = v[k - 1];
    free(v);
    return r;
}

static double gpu_ms(uint32_t slot) {
    uint64_t ts[2];
    if (vkGetQueryPoolResults(dev, X.queries, 2 * slot, 2, sizeof ts, ts, sizeof ts[0], VK_QUERY_RESULT_64_BIT) != VK_SUCCESS)
        return -1;
    return (double)(ts[1] - ts[0]) * gpu_props.limits.timestampPeriod / 1e6;
}

static void run_variant_draws(Variant *v, int index) {
    uint32_t *seq = draw_sequence(v->count);
    Packet *pk[SLOTS];
    for (int s = 0; s < SLOTS; s++) {
        pk[s] = malloc(X.draws * sizeof(Packet));
        for (uint32_t i = 0; i < X.draws; i++) make_packet(v, i, seq[i], (uint32_t)s, &pk[s][i]);
    }
    double *rec = calloc(X.frames, sizeof *rec), *sub = calloc(X.frames, sizeof *sub), *gpu = calloc(X.frames, sizeof *gpu);
    uint32_t ngpu = 0;
    long fb_frame = 0;
    char what[64];
    snprintf(what, sizeof what, "%s/%s", KIND_NAMES[v->kind], BIND_NAMES[v->bind]);
    for (uint32_t f = 0; f < X.frames; f++) {
        uint32_t s = f % SLOTS;
        VK_CHECK(vkWaitForFences(dev, 1, &X.fence[s], VK_TRUE, UINT64_MAX));
        if (f >= SLOTS + X.warmup) gpu[ngpu++] = gpu_ms(s);
        VK_CHECK(vkResetFences(dev, 1, &X.fence[s]));
        RecState rs;
        long fb0 = fallback_draws;
        double t0 = now_ms();
        begin_frame(v, s, &rs);
        for (uint32_t i = 0; i < X.draws; i++) record_draw(v, &rs, &pk[s][i]);
        end_frame(s);
        double t1 = now_ms();
        submit(s);
        double t2 = now_ms();
        rec[f] = t1 - t0;
        sub[f] = t2 - t1;
        fb_frame = fallback_draws - fb0;
        out_row("draws", what, (long)f, rec[f], index, NULL);
    }
    VK_CHECK(vkWaitForFences(dev, SLOTS, X.fence, VK_TRUE, UINT64_MAX));
    uint32_t n = X.frames - X.warmup;
    char key[96];
    const char *name = KIND_NAMES[v->kind], *bind = BIND_NAMES[v->bind];
#define KEY(what_) (snprintf(key, sizeof key, "%s_%s_%s", name, bind, what_), key)
    double p50 = pct(rec + X.warmup, n, 50);
    summary_num(KEY("record_ms_p50"), p50);
    summary_num(KEY("record_ms_p99"), pct(rec + X.warmup, n, 99));
    summary_num(KEY("submit_ms_p50"), pct(sub + X.warmup, n, 50));
    summary_num(KEY("record_ns_per_draw_p50"), p50 * 1e6 / X.draws);
    if (ngpu) summary_num(KEY("gpu_ms_p50"), pct(gpu, ngpu, 50));
    summary_num(KEY("fallback_draws_per_frame"), (double)fb_frame);
#undef KEY
    fprintf(stderr, "vkpipecost: %s: record p50 %.3f ms (%.0f ns/draw), submit %.3f ms, gpu %.3f ms, fallback %ld/frame\n",
            what, p50, p50 * 1e6 / X.draws, pct(sub + X.warmup, n, 50), ngpu ? pct(gpu, ngpu, 50) : -1.0, fb_frame);
    free(rec);
    free(sub);
    free(gpu);
    for (int s = 0; s < SLOTS; s++) free(pk[s]);
    free(seq);
}

int mode_draws(Job *job) {
    summary_begin("draws");
    ctx_init(job);
    variants_read();
    summary_num("draws", X.draws);
    summary_num("frames", X.frames);
    summary_num("run_length", (double)job_param_int(job, "run_length", 1));
    for (uint32_t i = 0; i < X.nvar; i++) {
        variant_init(&X.var[i]);
        run_variant_draws(&X.var[i], (int)i);
        variant_destroy(&X.var[i]);
    }
    ctx_destroy();
    summary_end();
    return val_errors ? 1 : 0;
}

// ---- replay -------------------------------------------------------------------------------
//
// Guest ops are resolved into packets on the replay thread: the pipeline through
// a hash table keyed by program, each buffer through a binary search of a sorted
// backing map (D14), uniform buffers copied from guest memory into the frame's
// ring slice (the snapshot of D17a), texture handles through a table. Then
// either the replay thread records the packet itself (recorder off), or it
// appends the packet to a 16 MiB ring that a recorder thread drains (on). In
// both cases at most SLOTS frames are between the start of a replay and GPU
// completion, as in section 6.1.

#define POOLS 4096
#define PACKET_RING (16u << 20)
#define HANDLES 33025

typedef struct {
    uint64_t base;
    uint32_t size, offset;
} Range;

typedef struct {
    uint32_t program;
    uint64_t addr[40];
    uint32_t handle[32];
} Op;

static struct {
    uint8_t *guest;
    size_t guest_size;
    Range ranges[POOLS];
    uint32_t table_keys[8192], table_vals[8192];
    uint32_t *handles;           // texture handle to image index
    Op *ops;
    uint32_t snapshot;           // bytes copied per uniform buffer
    uint8_t *ring;               // packet ring
    uint64_t head, tail;         // producer and consumer positions, atomics
    uint64_t submitted;          // frames the recorder has submitted, atomic
    int stop;
    double blocked_ms;           // replay thread, waiting for ring space
    double rec_busy[4096];
    Variant *v;
} R;

static uint32_t lookup_pipeline(uint32_t program) {
    uint32_t h = (program * 2654435761u) >> 19;
    for (;;) {
        if (R.table_keys[h] == program + 1) return R.table_vals[h];
        if (!R.table_keys[h]) die(3, "program %u has no pipeline", program);
        h = (h + 1) & 8191;
    }
}

static const Range *lookup_range(uint64_t addr) {
    uint32_t lo = 0, hi = POOLS;
    while (hi - lo > 1) {
        uint32_t mid = (lo + hi) / 2;
        if (R.ranges[mid].base <= addr) lo = mid;
        else hi = mid;
    }
    return &R.ranges[lo];
}

static void resolve(const Op *op, Packet *p, uint32_t slot, VkDeviceSize *ring_pos) {
    Variant *v = R.v;
    p->pair = lookup_pipeline(op->program);
    const SetDesc *d = &X.job->sets[X.job->pairs[v->first + p->pair].set];
    p->nbuf = p->ntex = 0;
    VkDeviceSize half = RING_BYTES / SLOTS;
    for (uint32_t k = 0; k < d->n; k++) {
        const Bind *b = &d->b[k];
        if (b->type == VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER || b->binding >= TEX_FIRST) {
            p->tex_binding[p->ntex] = b->binding;
            p->tex_index[p->ntex] = R.handles[op->handle[p->ntex % 32] % HANDLES];
            p->ntex++;
            continue;
        }
        uint64_t addr = op->addr[p->nbuf];
        const Range *r = lookup_range(addr);
        uint64_t off = r->offset + (addr - r->base);
        if (b->type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER) {
            // Snapshot: the bytes the draw reads, copied now.
            if (*ring_pos + R.snapshot > half) *ring_pos = 0;
            VkDeviceSize at = slot * half + *ring_pos;
            memcpy(X.ring_map + at, R.guest + off % (R.guest_size - R.snapshot), R.snapshot);
            *ring_pos = (*ring_pos + R.snapshot + 255) & ~(VkDeviceSize)255;
            p->b[p->nbuf++] = (PBuf){b->binding, b->type, at, R.snapshot};
        } else {
            p->b[p->nbuf++] = (PBuf){b->binding, b->type, (off & ~(uint64_t)255) % (RING_BYTES - 4096), 256};
        }
    }
}

// Packet ring, one producer and one consumer: a size word, the packet's first
// three words, its buffers and texture words; a zero size word means "wrap to 0".
static size_t packet_bytes(const Packet *p) {
    return (16 + p->nbuf * sizeof(PBuf) + 8 * p->ntex + 15) & ~(size_t)15;
}

static void ring_wait_space(size_t n) {
    if (R.head + n - __atomic_load_n(&R.tail, __ATOMIC_ACQUIRE) <= PACKET_RING) return;
    double t0 = now_ms();
    while (R.head + n - __atomic_load_n(&R.tail, __ATOMIC_ACQUIRE) > PACKET_RING) YieldProcessor();
    R.blocked_ms += now_ms() - t0;
}

static void ring_push(const Packet *p) {
    size_t n = packet_bytes(p);
    uint64_t at = R.head % PACKET_RING;
    if (at + n > PACKET_RING) {
        ring_wait_space(PACKET_RING - at);
        *(uint32_t *)(R.ring + at) = 0;
        __atomic_store_n(&R.head, R.head + (PACKET_RING - at), __ATOMIC_RELEASE);
        at = 0;
    }
    ring_wait_space(n);
    uint8_t *q = R.ring + at;
    *(uint32_t *)q = (uint32_t)n;
    memcpy(q + 4, &p->pair, 12);
    memcpy(q + 16, p->b, p->nbuf * sizeof(PBuf));
    memcpy(q + 16 + p->nbuf * sizeof(PBuf), p->tex_binding, p->ntex * 4);
    memcpy(q + 16 + p->nbuf * sizeof(PBuf) + p->ntex * 4, p->tex_index, p->ntex * 4);
    __atomic_store_n(&R.head, R.head + n, __ATOMIC_RELEASE);
}

static int ring_pop(Packet *p) {
    for (;;) {
        uint64_t head = __atomic_load_n(&R.head, __ATOMIC_ACQUIRE);
        if (head == R.tail) {
            if (__atomic_load_n(&R.stop, __ATOMIC_ACQUIRE)) return 0;
            YieldProcessor();
            continue;
        }
        uint64_t at = R.tail % PACKET_RING;
        uint32_t n = *(uint32_t *)(R.ring + at);
        if (!n) {
            __atomic_store_n(&R.tail, R.tail + (PACKET_RING - at), __ATOMIC_RELEASE);
            continue;
        }
        uint8_t *q = R.ring + at;
        memcpy(&p->pair, q + 4, 12);
        memcpy(p->b, q + 16, p->nbuf * sizeof(PBuf));
        memcpy(p->tex_binding, q + 16 + p->nbuf * sizeof(PBuf), p->ntex * 4);
        memcpy(p->tex_index, q + 16 + p->nbuf * sizeof(PBuf) + p->ntex * 4, p->ntex * 4);
        __atomic_store_n(&R.tail, R.tail + n, __ATOMIC_RELEASE);
        return 1;
    }
}

static void recorder_main(void *arg) {
    (void)arg;
    Variant *v = R.v;
    Packet p;
    RecState rs = {0};
    uint32_t frame = 0;
    double busy0 = 0;
    while (ring_pop(&p)) {
        if (p.pair == UINT32_MAX) {
            uint32_t s = frame % SLOTS;
            VK_CHECK(vkWaitForFences(dev, 1, &X.fence[s], VK_TRUE, UINT64_MAX));
            VK_CHECK(vkResetFences(dev, 1, &X.fence[s]));
            busy0 = now_ms();
            begin_frame(v, s, &rs);
        } else if (p.pair == UINT32_MAX - 1) {
            end_frame(rs.slot);
            submit(rs.slot);
            if (frame < 4096) R.rec_busy[frame] = now_ms() - busy0;
            frame++;
            __atomic_store_n(&R.submitted, frame, __ATOMIC_RELEASE);
        } else record_draw(v, &rs, &p);
    }
}

static void replay_setup(Variant *v) {
    R.v = v;
    R.snapshot = (uint32_t)job_param_int(X.job, "snapshot_bytes", 256);
    R.guest_size = 64u << 20;
    R.guest = malloc(R.guest_size);
    for (size_t i = 0; i < R.guest_size; i += 4) *(uint32_t *)(R.guest + i) = (uint32_t)(i * 2654435761u);
    uint64_t base = 0x80000000ull;
    for (uint32_t i = 0; i < POOLS; i++) {
        uint32_t size = 0x10000u << (i % 5);
        R.ranges[i] = (Range){base, size, (uint32_t)((i * 0x10000u) % (R.guest_size - 0x10000))};
        base += size + 0x10000;
    }
    memset(R.table_keys, 0, sizeof R.table_keys);
    for (uint32_t j = 0; j < v->count; j++) {
        uint32_t program = 0x1000 + j * 37, h = (program * 2654435761u) >> 19;
        while (R.table_keys[h]) h = (h + 1) & 8191;
        R.table_keys[h] = program + 1;
        R.table_vals[h] = j;
    }
    R.handles = malloc(HANDLES * sizeof *R.handles);
    for (uint32_t i = 0; i < HANDLES; i++) R.handles[i] = i % TEXTURES;
    uint32_t *seq = draw_sequence(v->count);
    R.ops = malloc(X.draws * sizeof *R.ops);
    uint64_t r = 77;
    for (uint32_t i = 0; i < X.draws; i++) {
        Op *op = &R.ops[i];
        op->program = 0x1000 + seq[i] * 37;
        for (uint32_t k = 0; k < 40; k++) {
            r ^= r << 13, r ^= r >> 7, r ^= r << 17;
            const Range *g = &R.ranges[r % POOLS];
            op->addr[k] = g->base + (r >> 20) % (g->size - 4096);
        }
        for (uint32_t k = 0; k < 32; k++) op->handle[k] = (uint32_t)((r >> (k % 40)) % HANDLES);
    }
    free(seq);
    R.ring = malloc(PACKET_RING);
}

static void replay_run(int recorder, int index) {
    Variant *v = R.v;
    R.head = R.tail = R.submitted = 0;
    R.stop = 0;
    R.blocked_ms = 0;
    VK_CHECK(vkWaitForFences(dev, SLOTS, X.fence, VK_TRUE, UINT64_MAX));
    Thread *t = recorder ? thread_start(recorder_main, NULL) : NULL;
    double *busy = calloc(X.frames, sizeof *busy), *cpu = calloc(X.frames, sizeof *cpu);
    double t_warm = now_ms();
    for (uint32_t f = 0; f < X.frames; f++) {
        uint32_t s = f % SLOTS;
        if (f == X.warmup) t_warm = now_ms();
        double wait = 0;
        if (recorder) {
            // Frame f starts once frame f - SLOTS has been submitted: the recorder
            // waits for its fence before it reuses the slot.
            double w0 = now_ms();
            while (f >= SLOTS && __atomic_load_n(&R.submitted, __ATOMIC_ACQUIRE) + SLOTS <= f) YieldProcessor();
            wait += now_ms() - w0;
        }
        double blocked0 = R.blocked_ms;
        double t0 = now_ms(), c0 = thread_cpu_ms();
        RecState rs;
        if (recorder) {
            Packet m = {.pair = UINT32_MAX};
            ring_push(&m);
        } else {
            double w0 = now_ms();
            VK_CHECK(vkWaitForFences(dev, 1, &X.fence[s], VK_TRUE, UINT64_MAX));
            wait += now_ms() - w0;
            VK_CHECK(vkResetFences(dev, 1, &X.fence[s]));
            begin_frame(v, s, &rs);
        }
        VkDeviceSize ring_pos = 0;
        Packet p;
        for (uint32_t i = 0; i < X.draws; i++) {
            resolve(&R.ops[i], &p, s, &ring_pos);
            if (recorder) ring_push(&p);
            else record_draw(v, &rs, &p);
        }
        if (recorder) {
            Packet m = {.pair = UINT32_MAX - 1};
            ring_push(&m);
        } else {
            end_frame(s);
            submit(s);
        }
        double fence_wait = recorder ? 0 : wait;
        busy[f] = now_ms() - t0 - (R.blocked_ms - blocked0) - fence_wait;
        cpu[f] = thread_cpu_ms() - c0;
    }
    double end = now_ms();
    __atomic_store_n(&R.stop, 1, __ATOMIC_RELEASE);
    if (t) thread_join(t);
    VK_CHECK(vkWaitForFences(dev, SLOTS, X.fence, VK_TRUE, UINT64_MAX));
    uint32_t n = X.frames - X.warmup;
    char key[96];
    const char *name = KIND_NAMES[v->kind], *bind = BIND_NAMES[v->bind];
#define KEY(what_) (snprintf(key, sizeof key, "%s_%s_rec%s_%s", name, bind, recorder ? "on" : "off", what_), key)
    summary_num(KEY("replay_busy_ms_p50"), pct(busy + X.warmup, n, 50));
    summary_num(KEY("replay_busy_ms_p99"), pct(busy + X.warmup, n, 99));
    summary_num(KEY("replay_cpu_ms_p50"), pct(cpu + X.warmup, n, 50));
    summary_num(KEY("frame_ms_mean"), (end - t_warm) / n);
    if (recorder) {
        summary_num(KEY("recorder_busy_ms_p50"), pct(R.rec_busy + X.warmup, n, 50));
        summary_num(KEY("ring_blocked_ms_total"), R.blocked_ms);
    }
#undef KEY
    fprintf(stderr, "vkpipecost: replay %s/%s recorder %s: replay busy p50 %.3f ms, cpu %.3f ms, frame %.3f ms\n", name, bind,
            recorder ? "on" : "off", pct(busy + X.warmup, n, 50), pct(cpu + X.warmup, n, 50), (end - t_warm) / n);
    for (uint32_t f = 0; f < X.frames; f++) out_row(recorder ? "replay_on" : "replay_off", name, (long)f, busy[f], index, NULL);
    free(busy);
    free(cpu);
}

int mode_replay(Job *job) {
    summary_begin("replay");
    ctx_init(job);
    variants_read();
    summary_num("draws", X.draws);
    summary_num("frames", X.frames);
    int repeats = (int)job_param_int(job, "repeats", 1);
    for (uint32_t i = 0; i < X.nvar; i++) {
        variant_init(&X.var[i]);
        replay_setup(&X.var[i]);
        // Off and on alternate, so that drift on the machine hits both.
        for (int r = 0; r < repeats; r++) {
            replay_run(0, (int)i);
            replay_run(1, (int)i);
        }
        free(R.guest);
        free(R.handles);
        free(R.ops);
        free(R.ring);
        variant_destroy(&X.var[i]);
    }
    ctx_destroy();
    summary_end();
    return val_errors ? 1 : 0;
}
