// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The recorder's packets (oot2/vk_rec.h): each command of a frame as a packet of the ring on the replaying thread, and
// the same packet made into the Vulkan call on the recorder thread. A command with scalar arguments is one entry of
// VKREC_FIXED or VKREC_DRAWS: the entry gives the packet's fields, how the arguments fill them and how they become the
// call, and this file expands it three ways (the packet type, the encoder, the executor's case). A command with arrays or
// nested structures is written by hand below: the arrays follow the fixed part inside the packet and the pointers of the
// fixed part point at them, so the recorder passes the packet's own memory to the driver, and the caller's storage is
// free the moment the encoder returns. The ring never wraps a packet and never moves, so those pointers stay valid.
//
// A structure with a pNext chain has no packet: a named fatal. Nothing the replay builds has one, and a chain copied
// without knowing what it holds would be a wrong call, not a missing one.
//
// Design as in DKB Recomp's runtime/vk_rec.c at 9d8fa02 (GPL-3.0-or-later; there after LAS Recomp runtime/nvn/vulkan/
// record.c, MIT, Copyright (c) 2026 Peiikai): one packet per command, the call's arguments copied after a fixed part,
// pointers into the packet. Changed here: the commands are those of the uni layout (push descriptors and push constants
// with no per-draw descriptor sets, the dynamic state of oot2/vk_pipeline.h, mesh draws), a draw packet carries the
// number of its breadcrumb, there is no direct fallback for a command without a packet, and the executor goes through a
// table of calls so that the encoding is tested without a device.
#include <string.h>

#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_rec.h"

#define A8(x) (((x) + 7) & ~(size_t)7)

// Copy bytes after the fixed part at *at; returns where they went, NULL when there is nothing to copy.
static void *tail_copy(uint8_t **at, const void *src, size_t bytes) {
    if (!src || !bytes) return NULL;
    void *d = *at;
    memcpy(d, src, bytes);
    *at += A8(bytes);
    return d;
}

static void no_pnext(const void *pnext, const char *what) {
    if (pnext) rt_fatal("gfx: vulkan: recorder: %s with a pNext chain has no packet", what);
}

// ---------- commands with scalar arguments
#define PACK_F(name, params, packargs, callargs, ...)                                                                  \
    typedef struct {                                                                                                   \
        __VA_ARGS__                                                                                                    \
    } A_##name;                                                                                                        \
    void vkrec_##name params {                                                                                         \
        A_##name *a = vkrec_put(VKR_##name, sizeof *a);                                                                \
        *a = (A_##name){VKREC_UNPAREN packargs};                                                                       \
        vkrec_publish();                                                                                               \
    }
VKREC_FIXED(PACK_F)
#undef PACK_F

// A draw's packet carries the number of its breadcrumb first.
#define PACK_D(name, kind, params, packargs, callargs, ...)                                                            \
    typedef struct {                                                                                                   \
        uint32_t crumb;                                                                                                \
        __VA_ARGS__                                                                                                    \
    } A_##name;                                                                                                        \
    void vkrec_##name params {                                                                                         \
        uint32_t crumb_ = vkrec_crumb_next(kind);                                                                      \
        A_##name *a = vkrec_put(VKR_##name, sizeof *a);                                                                \
        *a = (A_##name){crumb_, VKREC_UNPAREN packargs};                                                               \
        vkrec_publish();                                                                                               \
    }
VKREC_DRAWS(PACK_D)
#undef PACK_D

// ---------- commands with arrays
typedef struct {
    VkPipelineBindPoint bp;
    VkPipelineLayout layout;
    uint32_t first, nsets, ndyn;
    const VkDescriptorSet *sets;
    const uint32_t *dyn;
} A_BindDescriptorSets;
void vkrec_BindDescriptorSets(VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t first, uint32_t nsets,
                              const VkDescriptorSet *sets, uint32_t ndynamic, const uint32_t *dynamic) {
    size_t sb = sizeof *sets * nsets, db = sizeof *dynamic * ndynamic;
    A_BindDescriptorSets *a = vkrec_put(VKR_BindDescriptorSets, sizeof *a + A8(sb) + A8(db));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_BindDescriptorSets){bp, layout, first, nsets, ndynamic, NULL, NULL};
    a->sets = tail_copy(&at, sets, sb);
    a->dyn = tail_copy(&at, dynamic, db);
    vkrec_publish();
}

// Push descriptors: the writes, then for each the array its type reads (images, buffers or texel buffer views).
typedef struct {
    VkPipelineBindPoint bp;
    VkPipelineLayout layout;
    uint32_t set, n;
    const VkWriteDescriptorSet *writes;
} A_PushDescriptorSetKHR;
static size_t write_element(const VkWriteDescriptorSet *w) {
    switch (w->descriptorType) {
    case VK_DESCRIPTOR_TYPE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
    case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
    case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
    case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT: return sizeof(VkDescriptorImageInfo);
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
    case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
    case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: return sizeof(VkDescriptorBufferInfo);
    case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
    case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: return sizeof(VkBufferView);
    default: rt_fatal("gfx: vulkan: recorder: a push descriptor of type %d has no packet", (int)w->descriptorType);
    }
}
void vkrec_PushDescriptorSetKHR(VkPipelineBindPoint bp, VkPipelineLayout layout, uint32_t set, uint32_t nwrites,
                                const VkWriteDescriptorSet *writes) {
    size_t bytes = sizeof *writes * nwrites;
    for (uint32_t i = 0; i < nwrites; i++) {
        no_pnext(writes[i].pNext, "a push descriptor write");
        bytes += A8(write_element(&writes[i]) * writes[i].descriptorCount);
    }
    A_PushDescriptorSetKHR *a = vkrec_put(VKR_PushDescriptorSetKHR, sizeof *a + bytes);
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_PushDescriptorSetKHR){bp, layout, set, nwrites, NULL};
    VkWriteDescriptorSet *w = (VkWriteDescriptorSet *)tail_copy(&at, writes, sizeof *writes * nwrites);
    a->writes = w;
    for (uint32_t i = 0; i < nwrites; i++) {
        size_t n = write_element(&writes[i]) * writes[i].descriptorCount;
        const void *src = writes[i].pImageInfo;
        switch (writes[i].descriptorType) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: src = writes[i].pBufferInfo; break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: src = writes[i].pTexelBufferView; break;
        default: break;
        }
        const void *copy = tail_copy(&at, src, n);
        w[i].pImageInfo = NULL, w[i].pBufferInfo = NULL, w[i].pTexelBufferView = NULL;
        switch (writes[i].descriptorType) {
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER:
        case VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER_DYNAMIC:
        case VK_DESCRIPTOR_TYPE_STORAGE_BUFFER_DYNAMIC: w[i].pBufferInfo = copy; break;
        case VK_DESCRIPTOR_TYPE_UNIFORM_TEXEL_BUFFER:
        case VK_DESCRIPTOR_TYPE_STORAGE_TEXEL_BUFFER: w[i].pTexelBufferView = copy; break;
        default: w[i].pImageInfo = copy; break;
        }
    }
    vkrec_publish();
}

typedef struct {
    VkPipelineLayout layout;
    VkShaderStageFlags stages;
    uint32_t offset, size;
    const void *values;
} A_PushConstants;
void vkrec_PushConstants(VkPipelineLayout layout, VkShaderStageFlags stages, uint32_t offset, uint32_t size,
                         const void *values) {
    A_PushConstants *a = vkrec_put(VKR_PushConstants, sizeof *a + A8(size));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_PushConstants){layout, stages, offset, size, NULL};
    a->values = tail_copy(&at, values, size);
    vkrec_publish();
}

typedef struct {
    uint32_t first, n;
    const VkBuffer *buffers;
    const VkDeviceSize *offsets, *sizes, *strides;
} A_BindVertexBuffers2;
void vkrec_BindVertexBuffers2(uint32_t first, uint32_t n, const VkBuffer *buffers, const VkDeviceSize *offsets,
                              const VkDeviceSize *sizes, const VkDeviceSize *strides) {
    size_t bytes = A8(sizeof *buffers * n) + A8(sizeof *offsets * n) + (sizes ? A8(sizeof *sizes * n) : 0) +
                   (strides ? A8(sizeof *strides * n) : 0);
    A_BindVertexBuffers2 *a = vkrec_put(VKR_BindVertexBuffers2, sizeof *a + bytes);
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_BindVertexBuffers2){first, n, NULL, NULL, NULL, NULL};
    a->buffers = tail_copy(&at, buffers, sizeof *buffers * n);
    a->offsets = tail_copy(&at, offsets, sizeof *offsets * n);
    a->sizes = tail_copy(&at, sizes, sizeof *sizes * n);
    a->strides = tail_copy(&at, strides, sizeof *strides * n);
    vkrec_publish();
}

typedef struct {
    uint32_t n;
    const VkViewport *v;
} A_SetViewportWithCount;
void vkrec_SetViewportWithCount(uint32_t n, const VkViewport *v) {
    A_SetViewportWithCount *a = vkrec_put(VKR_SetViewportWithCount, sizeof *a + A8(sizeof *v * n));
    uint8_t *at = (uint8_t *)(a + 1);
    a->n = n;
    a->v = tail_copy(&at, v, sizeof *v * n);
    vkrec_publish();
}

typedef struct {
    uint32_t n;
    const VkRect2D *r;
} A_SetScissorWithCount;
void vkrec_SetScissorWithCount(uint32_t n, const VkRect2D *r) {
    A_SetScissorWithCount *a = vkrec_put(VKR_SetScissorWithCount, sizeof *a + A8(sizeof *r * n));
    uint8_t *at = (uint8_t *)(a + 1);
    a->n = n;
    a->r = tail_copy(&at, r, sizeof *r * n);
    vkrec_publish();
}

typedef struct {
    uint32_t nb, na;
    const VkVertexInputBindingDescription2EXT *b;
    const VkVertexInputAttributeDescription2EXT *at;
} A_SetVertexInputEXT;
void vkrec_SetVertexInputEXT(uint32_t nbindings, const VkVertexInputBindingDescription2EXT *bindings,
                             uint32_t nattributes, const VkVertexInputAttributeDescription2EXT *attributes) {
    for (uint32_t i = 0; i < nbindings; i++) no_pnext(bindings[i].pNext, "a vertex input binding");
    for (uint32_t i = 0; i < nattributes; i++) no_pnext(attributes[i].pNext, "a vertex input attribute");
    size_t bb = sizeof *bindings * nbindings, ab = sizeof *attributes * nattributes;
    A_SetVertexInputEXT *a = vkrec_put(VKR_SetVertexInputEXT, sizeof *a + A8(bb) + A8(ab));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_SetVertexInputEXT){nbindings, nattributes, NULL, NULL};
    a->b = tail_copy(&at, bindings, bb);
    a->at = tail_copy(&at, attributes, ab);
    vkrec_publish();
}

typedef struct {
    uint32_t first, n;
    const VkBool32 *v;
} A_SetColorBlendEnableEXT;
void vkrec_SetColorBlendEnableEXT(uint32_t first, uint32_t n, const VkBool32 *enable) {
    A_SetColorBlendEnableEXT *a = vkrec_put(VKR_SetColorBlendEnableEXT, sizeof *a + A8(sizeof *enable * n));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_SetColorBlendEnableEXT){first, n, NULL};
    a->v = tail_copy(&at, enable, sizeof *enable * n);
    vkrec_publish();
}

typedef struct {
    uint32_t first, n;
    const VkColorBlendEquationEXT *v;
} A_SetColorBlendEquationEXT;
void vkrec_SetColorBlendEquationEXT(uint32_t first, uint32_t n, const VkColorBlendEquationEXT *equations) {
    A_SetColorBlendEquationEXT *a = vkrec_put(VKR_SetColorBlendEquationEXT, sizeof *a + A8(sizeof *equations * n));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_SetColorBlendEquationEXT){first, n, NULL};
    a->v = tail_copy(&at, equations, sizeof *equations * n);
    vkrec_publish();
}

typedef struct {
    uint32_t first, n;
    const VkColorComponentFlags *v;
} A_SetColorWriteMaskEXT;
void vkrec_SetColorWriteMaskEXT(uint32_t first, uint32_t n, const VkColorComponentFlags *masks) {
    A_SetColorWriteMaskEXT *a = vkrec_put(VKR_SetColorWriteMaskEXT, sizeof *a + A8(sizeof *masks * n));
    uint8_t *at = (uint8_t *)(a + 1);
    *a = (A_SetColorWriteMaskEXT){first, n, NULL};
    a->v = tail_copy(&at, masks, sizeof *masks * n);
    vkrec_publish();
}

typedef struct {
    float c[4];
} A_SetBlendConstants;
void vkrec_SetBlendConstants(const float c[4]) {
    A_SetBlendConstants *a = vkrec_put(VKR_SetBlendConstants, sizeof *a);
    memcpy(a->c, c, sizeof a->c);
    vkrec_publish();
}

// A dependency: the structure, then its three arrays.
typedef struct {
    VkDependencyInfo di;
} A_PipelineBarrier2;
void vkrec_PipelineBarrier2(const VkDependencyInfo *info) {
    no_pnext(info->pNext, "a dependency");
    for (uint32_t i = 0; i < info->memoryBarrierCount; i++) no_pnext(info->pMemoryBarriers[i].pNext, "a memory barrier");
    for (uint32_t i = 0; i < info->bufferMemoryBarrierCount; i++)
        no_pnext(info->pBufferMemoryBarriers[i].pNext, "a buffer barrier");
    for (uint32_t i = 0; i < info->imageMemoryBarrierCount; i++)
        no_pnext(info->pImageMemoryBarriers[i].pNext, "an image barrier");
    size_t mb = sizeof(VkMemoryBarrier2) * info->memoryBarrierCount,
           bb = sizeof(VkBufferMemoryBarrier2) * info->bufferMemoryBarrierCount,
           ib = sizeof(VkImageMemoryBarrier2) * info->imageMemoryBarrierCount;
    A_PipelineBarrier2 *a = vkrec_put(VKR_PipelineBarrier2, sizeof *a + A8(mb) + A8(bb) + A8(ib));
    uint8_t *at = (uint8_t *)(a + 1);
    a->di = *info;
    a->di.pMemoryBarriers = tail_copy(&at, info->pMemoryBarriers, mb);
    a->di.pBufferMemoryBarriers = tail_copy(&at, info->pBufferMemoryBarriers, bb);
    a->di.pImageMemoryBarriers = tail_copy(&at, info->pImageMemoryBarriers, ib);
    vkrec_publish();
}

// A rendering: the structure, the colour attachments, then the depth and stencil attachments when there are any.
typedef struct {
    VkRenderingInfo ri;
} A_BeginRendering;
void vkrec_BeginRendering(const VkRenderingInfo *info) {
    no_pnext(info->pNext, "a rendering");
    for (uint32_t i = 0; i < info->colorAttachmentCount; i++)
        no_pnext(info->pColorAttachments[i].pNext, "a colour attachment");
    if (info->pDepthAttachment) no_pnext(info->pDepthAttachment->pNext, "a depth attachment");
    if (info->pStencilAttachment) no_pnext(info->pStencilAttachment->pNext, "a stencil attachment");
    size_t cb = sizeof(VkRenderingAttachmentInfo) * info->colorAttachmentCount,
           one = sizeof(VkRenderingAttachmentInfo);
    A_BeginRendering *a = vkrec_put(VKR_BeginRendering, sizeof *a + A8(cb) + 2 * one);
    uint8_t *at = (uint8_t *)(a + 1);
    a->ri = *info;
    a->ri.pColorAttachments = tail_copy(&at, info->pColorAttachments, cb);
    a->ri.pDepthAttachment = tail_copy(&at, info->pDepthAttachment, one);
    a->ri.pStencilAttachment = tail_copy(&at, info->pStencilAttachment, one);
    vkrec_publish();
}

void vkrec_EndRendering(void) {
    vkrec_put(VKR_EndRendering, 0);
    vkrec_publish();
}

// ---------- the executor
void vkrec_exec(const VkRecExec *x, uint32_t op, const void *p) {
    const VkRecCalls *calls = x->calls;
    VkCommandBuffer cb = x->cb;
    switch (op) {
#define EXEC_F(name, params, packargs, callargs, ...)                                                                  \
    case VKR_##name: {                                                                                                 \
        const A_##name *a = p;                                                                                         \
        (void)a;                                                                                                       \
        calls->name callargs;                                                                                          \
        return;                                                                                                        \
    }
        VKREC_FIXED(EXEC_F)
#undef EXEC_F
#define EXEC_D(name, kind, params, packargs, callargs, ...)                                                            \
    case VKR_##name: {                                                                                                 \
        const A_##name *a = p;                                                                                         \
        if (a->crumb && x->checkpoints)                                                                                \
            calls->SetCheckpointNV(cb, (const void *)(uintptr_t)GPU_VK_DRAW_MARKER(x->value, a->crumb));               \
        calls->name callargs;                                                                                          \
        return;                                                                                                        \
    }
        VKREC_DRAWS(EXEC_D)
#undef EXEC_D
    case VKR_BindDescriptorSets: {
        const A_BindDescriptorSets *a = p;
        calls->BindDescriptorSets(cb, a->bp, a->layout, a->first, a->nsets, a->sets, a->ndyn, a->dyn);
        return;
    }
    case VKR_PushDescriptorSetKHR: {
        const A_PushDescriptorSetKHR *a = p;
        calls->PushDescriptorSetKHR(cb, a->bp, a->layout, a->set, a->n, a->writes);
        return;
    }
    case VKR_PushConstants: {
        const A_PushConstants *a = p;
        calls->PushConstants(cb, a->layout, a->stages, a->offset, a->size, a->values);
        return;
    }
    case VKR_BindVertexBuffers2: {
        const A_BindVertexBuffers2 *a = p;
        calls->BindVertexBuffers2(cb, a->first, a->n, a->buffers, a->offsets, a->sizes, a->strides);
        return;
    }
    case VKR_SetViewportWithCount: {
        const A_SetViewportWithCount *a = p;
        calls->SetViewportWithCount(cb, a->n, a->v);
        return;
    }
    case VKR_SetScissorWithCount: {
        const A_SetScissorWithCount *a = p;
        calls->SetScissorWithCount(cb, a->n, a->r);
        return;
    }
    case VKR_SetVertexInputEXT: {
        const A_SetVertexInputEXT *a = p;
        calls->SetVertexInputEXT(cb, a->nb, a->b, a->na, a->at);
        return;
    }
    case VKR_SetColorBlendEnableEXT: {
        const A_SetColorBlendEnableEXT *a = p;
        calls->SetColorBlendEnableEXT(cb, a->first, a->n, a->v);
        return;
    }
    case VKR_SetColorBlendEquationEXT: {
        const A_SetColorBlendEquationEXT *a = p;
        calls->SetColorBlendEquationEXT(cb, a->first, a->n, a->v);
        return;
    }
    case VKR_SetColorWriteMaskEXT: {
        const A_SetColorWriteMaskEXT *a = p;
        calls->SetColorWriteMaskEXT(cb, a->first, a->n, a->v);
        return;
    }
    case VKR_SetBlendConstants: {
        const A_SetBlendConstants *a = p;
        calls->SetBlendConstants(cb, a->c);
        return;
    }
    case VKR_PipelineBarrier2: {
        const A_PipelineBarrier2 *a = p;
        calls->PipelineBarrier2(cb, &a->di);
        return;
    }
    case VKR_BeginRendering: {
        const A_BeginRendering *a = p;
        calls->BeginRendering(cb, &a->ri);
        return;
    }
    case VKR_EndRendering: calls->EndRendering(cb); return;
    default: rt_fatal("gfx: vulkan: recorder: a packet of operation %u that no command has", op);
    }
}

// ---------- the device's entry points
void vkrec_calls_device(VkRecCalls *out) {
    memset(out, 0, sizeof *out);
#define SET_F(name, ...) out->name = vkCmd##name;
#define SET_D(name, kind, ...) out->name = vkCmd##name;
#define SET_C(name) out->name = vkCmd##name;
    VKREC_FIXED(SET_F)
    VKREC_DRAWS(SET_D)
    VKREC_CUSTOM(SET_C)
#undef SET_F
#undef SET_D
#undef SET_C
    out->SetCheckpointNV = vkCmdSetCheckpointNV;
}
