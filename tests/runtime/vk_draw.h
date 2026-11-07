// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// One draw or dispatch on the back end's device through the layouts of oot2/vk_pipeline.h, for the pipeline builder's
// device tests (tests/runtime/vk_pipeline_test.c). The draw is the one tools/firstdraw/vkdraw.c makes (a non-indexed
// triangle list into one colour target, the target read back as raw texels), except that the layouts come from the
// builder, every dynamic state a library makes dynamic is set, and the work goes through the back end's internal submit
// (gpu_vk_begin to gpu_vk_end). The caller writes the global array entries itself, as the bindless ticket will later.
#pragma once
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "volk.h"

#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/vk_pipeline.h"
#include "plat.h"

#define DRAW_CHECK(cond)                                                                                               \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "check failed at %s:%d: %s\n", __FILE__, __LINE__, #cond);                                \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

typedef struct DrawBuf {
    VkBuffer buf;
    VkDeviceMemory mem;
    void *map;
    VkDeviceSize size;
} DrawBuf;

static inline uint32_t draw_memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits & (1u << i)) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    DRAW_CHECK(!"no memory type fits");
    return 0;
}

static inline DrawBuf draw_buffer(VkDeviceSize size, VkBufferUsageFlags usage, const void *data, size_t data_size) {
    DrawBuf b = {.size = size};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size ? size : 16, .usage = usage};
    DRAW_CHECK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &b.buf) == VK_SUCCESS);
    VkMemoryRequirements req;
    vkGetBufferMemoryRequirements(gpu_vk.device, b.buf, &req);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                               .memoryTypeIndex = draw_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                                   VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)};
    DRAW_CHECK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &b.mem) == VK_SUCCESS);
    DRAW_CHECK(vkBindBufferMemory(gpu_vk.device, b.buf, b.mem, 0) == VK_SUCCESS);
    DRAW_CHECK(vkMapMemory(gpu_vk.device, b.mem, 0, VK_WHOLE_SIZE, 0, &b.map) == VK_SUCCESS);
    memset(b.map, 0, (size_t)(size ? size : 16));
    if (data) memcpy(b.map, data, data_size < size ? data_size : (size_t)size);
    return b;
}

static inline void draw_buffer_free(DrawBuf *b) {
    if (!b->buf) return;
    vkDestroyBuffer(gpu_vk.device, b->buf, NULL);
    vkFreeMemory(gpu_vk.device, b->mem, NULL);
    memset(b, 0, sizeof *b);
}

typedef struct DrawBind {
    uint32_t binding;
    VkDescriptorType type;     // uniform or storage buffer
    const void *data;
    size_t size;
} DrawBind;

typedef struct DrawTex {
    uint32_t entry;            // the index in the global image array
    VkImageViewType view;      // 2D, 2D array, 3D or cube
    uint32_t w, h, layers;     // layers: array layers, depth, or 6
    const void *texels;        // RGBA32F, w * h * layers texels in slice order
} DrawTex;

typedef struct DrawSamp {
    uint32_t entry;
    VkFilter filter;
    VkSamplerAddressMode address;
} DrawSamp;

typedef struct DrawAttr {
    uint32_t location;
    VkFormat format;
    uint32_t offset;
} DrawAttr;

typedef struct DrawSpec {
    VkPipeline pipeline;
    VkFormat format;           // VK_FORMAT_R32G32B32A32_SFLOAT, R16G16B16A16_SFLOAT or R8G8B8A8_UNORM
    uint32_t width, height;
    float clear[4];
    uint32_t vertex_count;
    const void *vertices;
    uint32_t vertices_size, stride;
    DrawAttr attrs[32];
    uint32_t nattrs;
    DrawBind binds[48];
    uint32_t nbinds;
    DrawTex tex[64];
    uint32_t ntex;
    DrawSamp samp[64];
    uint32_t nsamp;
    uint32_t push[VKP_PUSH_BYTES / 4];
    void *out;                 // width * height texels of format, written
} DrawSpec;

static inline uint32_t draw_texel_bytes(VkFormat f) {
    return f == VK_FORMAT_R32G32B32A32_SFLOAT ? 16 : f == VK_FORMAT_R16G16B16A16_SFLOAT ? 8 : 4;
}

static inline void draw_image_barrier(VkCommandBuffer cb, VkImage img, uint32_t layers, VkImageLayout from, VkImageLayout to,
                                      VkPipelineStageFlags2 ss, VkAccessFlags2 sa, VkPipelineStageFlags2 ds,
                                      VkAccessFlags2 da) {
    VkImageMemoryBarrier2 b = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = ss, .srcAccessMask = sa,
                               .dstStageMask = ds, .dstAccessMask = da, .oldLayout = from, .newLayout = to,
                               .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                               .image = img, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &b};
    vkCmdPipelineBarrier2(cb, &di);
}

// Every state the libraries of vkp_dynamic_states make dynamic, set to the defaults a draw of these tests wants.
static inline void draw_set_state(VkCommandBuffer cb, uint32_t w, uint32_t h, uint32_t targets) {
    VkViewport vp = {0, 0, (float)w, (float)h, 0, 1};
    VkRect2D sc = {{0, 0}, {w, h}};
    vkCmdSetViewportWithCount(cb, 1, &vp);
    vkCmdSetScissorWithCount(cb, 1, &sc);
    vkCmdSetLineWidth(cb, 1.0f);
    vkCmdSetDepthBias(cb, 0, 0, 0);
    vkCmdSetCullMode(cb, VK_CULL_MODE_NONE);
    vkCmdSetFrontFace(cb, VK_FRONT_FACE_COUNTER_CLOCKWISE);
    vkCmdSetDepthBiasEnable(cb, VK_FALSE);
    vkCmdSetRasterizerDiscardEnable(cb, VK_FALSE);
    vkCmdSetPolygonModeEXT(cb, VK_POLYGON_MODE_FILL);
    vkCmdSetDepthClampEnableEXT(cb, VK_FALSE);
    vkCmdSetDepthClipNegativeOneToOneEXT(cb, VK_FALSE);
    vkCmdSetPrimitiveTopology(cb, VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST);
    vkCmdSetPrimitiveRestartEnable(cb, VK_FALSE);
    vkCmdSetDepthTestEnable(cb, VK_FALSE);
    vkCmdSetDepthWriteEnable(cb, VK_FALSE);
    vkCmdSetDepthCompareOp(cb, VK_COMPARE_OP_ALWAYS);
    vkCmdSetDepthBoundsTestEnable(cb, VK_FALSE);
    vkCmdSetStencilTestEnable(cb, VK_FALSE);
    vkCmdSetStencilOp(cb, VK_STENCIL_FACE_FRONT_AND_BACK, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP, VK_STENCIL_OP_KEEP,
                      VK_COMPARE_OP_ALWAYS);
    vkCmdSetStencilCompareMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilWriteMask(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0xFF);
    vkCmdSetStencilReference(cb, VK_STENCIL_FACE_FRONT_AND_BACK, 0);
    vkCmdSetDepthBounds(cb, 0.0f, 1.0f);
    float bc[4] = {0, 0, 0, 0};
    vkCmdSetBlendConstants(cb, bc);
    vkCmdSetLogicOpEnableEXT(cb, VK_FALSE);
    VkBool32 en[VKP_MAX_COLOR] = {0};
    VkColorBlendEquationEXT eq[VKP_MAX_COLOR];
    VkColorComponentFlags wm[VKP_MAX_COLOR];
    for (uint32_t i = 0; i < VKP_MAX_COLOR; i++) {
        eq[i] = (VkColorBlendEquationEXT){VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD,
                                          VK_BLEND_FACTOR_ONE, VK_BLEND_FACTOR_ZERO, VK_BLEND_OP_ADD};
        wm[i] = 0xF;
    }
    vkCmdSetColorBlendEnableEXT(cb, 0, targets, en);
    vkCmdSetColorBlendEquationEXT(cb, 0, targets, eq);
    vkCmdSetColorWriteMaskEXT(cb, 0, targets, wm);
}

// The global set 0 of the builder's layout, allocated from a pool of its own, with the entries the spec names written.
typedef struct DrawGlobal {
    VkDescriptorPool pool;
    VkDescriptorSet set;
} DrawGlobal;

static inline DrawGlobal draw_global_set(void) {
    DrawGlobal g;
    VkDescriptorPoolSize ps[2] = {{VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, VKP_GLOBAL_IMAGES},
                                  {VK_DESCRIPTOR_TYPE_SAMPLER, VKP_GLOBAL_SAMPLERS}};
    VkDescriptorPoolCreateInfo pci = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
                                      .flags = VK_DESCRIPTOR_POOL_CREATE_UPDATE_AFTER_BIND_BIT, .maxSets = 1,
                                      .poolSizeCount = 2, .pPoolSizes = ps};
    DRAW_CHECK(vkCreateDescriptorPool(gpu_vk.device, &pci, NULL, &g.pool) == VK_SUCCESS);
    VkDescriptorSetLayout l = vkp_layouts()->global;
    VkDescriptorSetAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO, .descriptorPool = g.pool,
                                      .descriptorSetCount = 1, .pSetLayouts = &l};
    DRAW_CHECK(vkAllocateDescriptorSets(gpu_vk.device, &alloc, &g.set) == VK_SUCCESS);
    return g;
}

static inline void draw_push_buffers(VkCommandBuffer cb, VkPipelineBindPoint point, const DrawBind *binds, uint32_t n,
                                     const DrawBuf *bufs) {
    VkWriteDescriptorSet w[48];
    VkDescriptorBufferInfo bi[48];
    for (uint32_t i = 0; i < n; i++) {
        bi[i] = (VkDescriptorBufferInfo){bufs[i].buf, 0, binds[i].size};
        w[i] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstBinding = binds[i].binding,
                                      .descriptorCount = 1, .descriptorType = binds[i].type, .pBufferInfo = &bi[i]};
    }
    const VkpLayouts *lay = vkp_layouts();
    if (n)
        vkCmdPushDescriptorSetKHR(cb, point, point == VK_PIPELINE_BIND_POINT_COMPUTE ? lay->compute : lay->graphics, 1, n, w);
}

static inline void draw_run(const DrawSpec *s) {
    VkDevice dev = gpu_vk.device;
    const VkpLayouts *lay = vkp_layouts();
    DrawBuf bufs[48] = {{0}};
    for (uint32_t i = 0; i < s->nbinds; i++)
        bufs[i] = draw_buffer(s->binds[i].size, s->binds[i].type == VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER
                                                    ? VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT : VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                              s->binds[i].data, s->binds[i].size);
    DrawBuf vbuf = {0};
    if (s->stride) vbuf = draw_buffer(s->vertices_size, VK_BUFFER_USAGE_VERTEX_BUFFER_BIT, s->vertices, s->vertices_size);
    DrawGlobal g = draw_global_set();
    DrawBuf staging[64] = {{0}};
    VkImage timg[64] = {0};
    VkDeviceMemory tmem[64] = {0};
    VkImageView tview[64] = {0};
    VkSampler samplers[64] = {0};
    VkDescriptorImageInfo iinfo[64], sinfo[64];
    VkWriteDescriptorSet gw[128];
    uint32_t ngw = 0;
    for (uint32_t i = 0; i < s->ntex; i++) {
        const DrawTex *t = &s->tex[i];
        int is3d = t->view == VK_IMAGE_VIEW_TYPE_3D;
        size_t bytes = (size_t)t->w * t->h * t->layers * 16;
        staging[i] = draw_buffer(bytes, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, t->texels, bytes);
        VkImageCreateInfo ii = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
                                .flags = t->view == VK_IMAGE_VIEW_TYPE_CUBE ? VK_IMAGE_CREATE_CUBE_COMPATIBLE_BIT : 0,
                                .imageType = is3d ? VK_IMAGE_TYPE_3D : VK_IMAGE_TYPE_2D,
                                .format = VK_FORMAT_R32G32B32A32_SFLOAT, .extent = {t->w, t->h, is3d ? t->layers : 1},
                                .mipLevels = 1, .arrayLayers = is3d ? 1 : t->layers, .samples = VK_SAMPLE_COUNT_1_BIT,
                                .tiling = VK_IMAGE_TILING_OPTIMAL,
                                .usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
        DRAW_CHECK(vkCreateImage(dev, &ii, NULL, &timg[i]) == VK_SUCCESS);
        VkMemoryRequirements req;
        vkGetImageMemoryRequirements(dev, timg[i], &req);
        VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
                                   .memoryTypeIndex = draw_memory_type(req.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
        DRAW_CHECK(vkAllocateMemory(dev, &alloc, NULL, &tmem[i]) == VK_SUCCESS);
        DRAW_CHECK(vkBindImageMemory(dev, timg[i], tmem[i], 0) == VK_SUCCESS);
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = timg[i], .viewType = t->view,
                                    .format = VK_FORMAT_R32G32B32A32_SFLOAT,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, is3d ? 1 : t->layers}};
        DRAW_CHECK(vkCreateImageView(dev, &vi, NULL, &tview[i]) == VK_SUCCESS);
        iinfo[i] = (VkDescriptorImageInfo){VK_NULL_HANDLE, tview[i], VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
        gw[ngw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g.set, .dstBinding = 0,
                                           .dstArrayElement = t->entry, .descriptorCount = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE, .pImageInfo = &iinfo[i]};
    }
    for (uint32_t i = 0; i < s->nsamp; i++) {
        VkSamplerCreateInfo si = {.sType = VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO, .magFilter = s->samp[i].filter,
                                  .minFilter = s->samp[i].filter, .mipmapMode = VK_SAMPLER_MIPMAP_MODE_NEAREST,
                                  .addressModeU = s->samp[i].address, .addressModeV = s->samp[i].address,
                                  .addressModeW = s->samp[i].address, .maxLod = 0.0f};
        DRAW_CHECK(vkCreateSampler(dev, &si, NULL, &samplers[i]) == VK_SUCCESS);
        sinfo[i] = (VkDescriptorImageInfo){samplers[i], VK_NULL_HANDLE, VK_IMAGE_LAYOUT_UNDEFINED};
        gw[ngw++] = (VkWriteDescriptorSet){.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET, .dstSet = g.set, .dstBinding = 1,
                                           .dstArrayElement = s->samp[i].entry, .descriptorCount = 1,
                                           .descriptorType = VK_DESCRIPTOR_TYPE_SAMPLER, .pImageInfo = &sinfo[i]};
    }
    if (ngw) vkUpdateDescriptorSets(dev, ngw, gw, 0, NULL);

    // The colour target and its readback.
    VkImageCreateInfo ti = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D, .format = s->format,
                            .extent = {s->width, s->height, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT,
                            .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    VkImage target;
    DRAW_CHECK(vkCreateImage(dev, &ti, NULL, &target) == VK_SUCCESS);
    VkMemoryRequirements treq;
    vkGetImageMemoryRequirements(dev, target, &treq);
    VkMemoryAllocateInfo tai = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = treq.size,
                                .memoryTypeIndex = draw_memory_type(treq.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)};
    VkDeviceMemory target_mem;
    DRAW_CHECK(vkAllocateMemory(dev, &tai, NULL, &target_mem) == VK_SUCCESS);
    DRAW_CHECK(vkBindImageMemory(dev, target, target_mem, 0) == VK_SUCCESS);
    VkImageViewCreateInfo tvi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = target,
                                 .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = s->format,
                                 .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkImageView target_view;
    DRAW_CHECK(vkCreateImageView(dev, &tvi, NULL, &target_view) == VK_SUCCESS);
    size_t total = (size_t)s->width * s->height * draw_texel_bytes(s->format);
    DrawBuf readback = draw_buffer(total, VK_BUFFER_USAGE_TRANSFER_DST_BIT, NULL, 0);

    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    for (uint32_t i = 0; i < s->ntex; i++) {
        const DrawTex *t = &s->tex[i];
        int is3d = t->view == VK_IMAGE_VIEW_TYPE_3D;
        draw_image_barrier(cb, timg[i], VK_REMAINING_ARRAY_LAYERS, VK_IMAGE_LAYOUT_UNDEFINED,
                           VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT);
        VkBufferImageCopy copy = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, is3d ? 1 : t->layers},
                                  .imageExtent = {t->w, t->h, is3d ? t->layers : 1}};
        vkCmdCopyBufferToImage(cb, staging[i].buf, timg[i], VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &copy);
        draw_image_barrier(cb, timg[i], VK_REMAINING_ARRAY_LAYERS, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                           VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_PIPELINE_STAGE_2_COPY_BIT,
                           VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT,
                           VK_ACCESS_2_SHADER_SAMPLED_READ_BIT);
    }
    draw_image_barrier(cb, target, 1, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                       VK_PIPELINE_STAGE_2_NONE, 0, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT);
    VkRenderingAttachmentInfo att = {.sType = VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO, .imageView = target_view,
                                     .imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL,
                                     .loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR, .storeOp = VK_ATTACHMENT_STORE_OP_STORE};
    memcpy(att.clearValue.color.float32, s->clear, sizeof s->clear);
    VkRenderingInfo ri = {.sType = VK_STRUCTURE_TYPE_RENDERING_INFO, .renderArea = {{0, 0}, {s->width, s->height}},
                          .layerCount = 1, .colorAttachmentCount = 1, .pColorAttachments = &att};
    vkCmdBeginRendering(cb, &ri);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s->pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, lay->graphics, 0, 1, &g.set, 0, NULL);
    draw_set_state(cb, s->width, s->height, 1);
    VkVertexInputBindingDescription2EXT vb = {.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_BINDING_DESCRIPTION_2_EXT, .binding = 0,
                                              .stride = s->stride, .inputRate = VK_VERTEX_INPUT_RATE_VERTEX, .divisor = 1};
    VkVertexInputAttributeDescription2EXT va[32];
    for (uint32_t i = 0; i < s->nattrs; i++)
        va[i] = (VkVertexInputAttributeDescription2EXT){.sType = VK_STRUCTURE_TYPE_VERTEX_INPUT_ATTRIBUTE_DESCRIPTION_2_EXT,
                                                        .location = s->attrs[i].location, .binding = 0,
                                                        .format = s->attrs[i].format, .offset = s->attrs[i].offset};
    vkCmdSetVertexInputEXT(cb, s->stride ? 1 : 0, &vb, s->nattrs, va);
    draw_push_buffers(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, s->binds, s->nbinds, bufs);
    vkCmdPushConstants(cb, lay->graphics, vkp_push_range(VKP_GRAPHICS).stageFlags, 0, VKP_PUSH_BYTES, s->push);
    VkDeviceSize zero = 0;
    if (s->stride) vkCmdBindVertexBuffers(cb, 0, 1, &vbuf.buf, &zero);
    vkCmdDraw(cb, s->vertex_count, 1, 0, 0);
    vkCmdEndRendering(cb);
    draw_image_barrier(cb, target, 1, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL,
                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COPY_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    VkBufferImageCopy back = {.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1}, .imageExtent = {s->width, s->height, 1}};
    vkCmdCopyImageToBuffer(cb, target, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, readback.buf, 1, &back);
    VkBufferMemoryBarrier2 hb = {.sType = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
                                 .srcStageMask = VK_PIPELINE_STAGE_2_COPY_BIT, .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                                 .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT,
                                 .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                 .buffer = readback.buf, .size = VK_WHOLE_SIZE};
    VkDependencyInfo hdi = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .bufferMemoryBarrierCount = 1, .pBufferMemoryBarriers = &hb};
    vkCmdPipelineBarrier2(cb, &hdi);
    DRAW_CHECK(gpu_vk_end(NULL, 0) == v);
    DRAW_CHECK(gpu_sync_wait(v, 20ull * 1000 * 1000 * 1000));
    memcpy(s->out, readback.map, total);

    draw_buffer_free(&readback);
    vkDestroyImageView(dev, target_view, NULL);
    vkDestroyImage(dev, target, NULL);
    vkFreeMemory(dev, target_mem, NULL);
    for (uint32_t i = 0; i < s->ntex; i++) {
        vkDestroyImageView(dev, tview[i], NULL);
        vkDestroyImage(dev, timg[i], NULL);
        vkFreeMemory(dev, tmem[i], NULL);
        draw_buffer_free(&staging[i]);
    }
    for (uint32_t i = 0; i < s->nsamp; i++) vkDestroySampler(dev, samplers[i], NULL);
    vkDestroyDescriptorPool(dev, g.pool, NULL);
    draw_buffer_free(&vbuf);
    for (uint32_t i = 0; i < s->nbinds; i++) draw_buffer_free(&bufs[i]);
}

// One compute dispatch: the buffers bound at their bindings of set 1 (the caller keeps them; DrawBuf data is read back
// from the mapped memory after the wait), push constants, groups in x.
static inline void dispatch_run(VkPipeline pipeline, const DrawBind *binds, uint32_t nbinds, DrawBuf *bufs,
                                const uint32_t push[VKP_PUSH_BYTES / 4], uint32_t groups_x) {
    const VkpLayouts *lay = vkp_layouts();
    DrawGlobal g = draw_global_set();
    uint64_t v = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cb = gpu_vk_cmd();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeline);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, lay->compute, 0, 1, &g.set, 0, NULL);
    draw_push_buffers(cb, VK_PIPELINE_BIND_POINT_COMPUTE, binds, nbinds, bufs);
    vkCmdPushConstants(cb, lay->compute, vkp_push_range(VKP_COMPUTE).stageFlags, 0, VKP_PUSH_BYTES, push);
    vkCmdDispatch(cb, groups_x, 1, 1);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    DRAW_CHECK(gpu_vk_end(NULL, 0) == v);
    DRAW_CHECK(gpu_sync_wait(v, 20ull * 1000 * 1000 * 1000));
    vkDestroyDescriptorPool(gpu_vk.device, g.pool, NULL);
}
