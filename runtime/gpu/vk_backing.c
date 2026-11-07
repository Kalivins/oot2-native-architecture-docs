// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The device side of the backing map (oot2/backing.h, architecture D14, D15): VRAM blocks and host imports, each with
// one buffer over the whole allocation that every binding of the descriptor path can use (vertex, index, uniform,
// storage, indirect, copies) and whose device address the raw path will use, so its memory is allocated with the
// device-address flag and the buffer has the device-address usage (research 14 A2: neither of the bases did).
//
// A block takes a memory type that is device local and not host visible (the small host-visible VRAM heap is left to
// the back end's own staging). An import takes a type both the host pointer and the buffer allow; the device refuses an
// import over a page that is not committed (research 14 F12, research 48 B1), which comes back as 0 here, never as a
// lost device: the map retries once and then stops by name.
//
// Mirrors (oot2/backing.h, Mirrors). The budget figures are VK_EXT_memory_budget's for the heap blocks come from. Uploads
// and readbacks go through staging: host-visible, host-coherent memory (cached where the device has such a type, since
// readbacks are read by the CPU), in chunks of STAGE_CHUNK, a larger copy getting a chunk of its own. A chunk serves one
// submission at a time and is taken again once that submission's value is done; one larger than STAGE_CHUNK is freed
// then. An upload is copied into staging at once and recorded in the submission's prologue (gpu_vk_prologue: after a
// full barrier over earlier submissions' work, and before one over this submission's own); a readback is recorded in its
// epilogue (gpu_vk_epilogue: after the main command buffer), between a full barrier and one that makes the copy visible
// to the host.
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"

#define STAGE_CHUNK (16ull << 20)
#define STAGE_ALIGN 16ull

static const VkBufferUsageFlags k_usage =
    VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT |
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_VERTEX_BUFFER_BIT |
    VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;

// The lowest memory type in bits with every flag of want and none of avoid, or UINT32_MAX.
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags avoid) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((bits >> i & 1) && (f & want) == want && !(f & avoid)) return i;
    }
    return UINT32_MAX;
}

// The buffer over mem, bound, and its device address into out. 1, or 0 with nothing left behind when the allocation
// was refused for want of room (or, for an import, by the host pointer); a lost device stops the run.
static int finish(VkBuffer buf, const VkMemoryAllocateInfo *alloc, uint64_t size, const char *what, BackingMem *out) {
    VkDeviceMemory mem;
    VkResult r = vkAllocateMemory(gpu_vk.device, alloc, NULL, &mem);
    if (r == VK_ERROR_DEVICE_LOST) gpu_vk_failed(r, what);
    if (r != VK_SUCCESS) {
        rt_log("gfx: vulkan: %s of %llu KiB: %s", what, (unsigned long long)(size >> 10), gpu_vk_result_name(r));
        vkDestroyBuffer(gpu_vk.device, buf, NULL);
        return 0;
    }
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, buf, mem, 0), "vkBindBufferMemory (a backing)");
    VkBufferDeviceAddressInfo di = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = buf};
    *out = (BackingMem){.buffer = (uint64_t)(uintptr_t)buf, .memory = (uint64_t)(uintptr_t)mem,
                        .address = vkGetBufferDeviceAddress(gpu_vk.device, &di), .size = size};
    return 1;
}

static int vk_block_new(uint64_t size, BackingMem *out) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size, .usage = k_usage,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buf;
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &buf), "vkCreateBuffer (a VRAM block)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, buf, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT,
                                VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    if (type == UINT32_MAX)
        rt_fatal("gfx: vulkan: no memory type is device local and not host visible for a buffer (types %#x)",
                 mr.memoryTypeBits);
    VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fi,
                                  .allocationSize = mr.size, .memoryTypeIndex = type};
    return finish(buf, &alloc, size, "vkAllocateMemory (a VRAM block)", out);
}

static int vk_import(uint64_t host, uint64_t size, BackingMem *out) {
    VkExternalMemoryBufferCreateInfo eb = {.sType = VK_STRUCTURE_TYPE_EXTERNAL_MEMORY_BUFFER_CREATE_INFO,
                                           .handleTypes = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .pNext = &eb, .size = size,
                             .usage = k_usage, .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    VkBuffer buf;
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &buf), "vkCreateBuffer (a host import)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, buf, &mr);
    VkMemoryHostPointerPropertiesEXT hp = {.sType = VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    VkResult r = vkGetMemoryHostPointerPropertiesEXT(gpu_vk.device,
                                                     VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                                     (const void *)(uintptr_t)host, &hp);
    if (r == VK_ERROR_DEVICE_LOST) gpu_vk_failed(r, "vkGetMemoryHostPointerPropertiesEXT");
    uint32_t type = r == VK_SUCCESS ? memory_type(hp.memoryTypeBits & mr.memoryTypeBits, 0, 0) : UINT32_MAX;
    if (type == UINT32_MAX) {
        rt_log("gfx: vulkan: no memory type takes host memory at %#llx (%s, types %#x and %#x)",
               (unsigned long long)host, gpu_vk_result_name(r), hp.memoryTypeBits, mr.memoryTypeBits);
        vkDestroyBuffer(gpu_vk.device, buf, NULL);
        return 0;
    }
    VkImportMemoryHostPointerInfoEXT imp = {.sType = VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT,
                                            .handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT,
                                            .pHostPointer = (void *)(uintptr_t)host};
    VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO, .pNext = &imp,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fi,
                                  .allocationSize = size, .memoryTypeIndex = type};
    return finish(buf, &alloc, size, "vkAllocateMemory (a host import)", out);
}

static void vk_free(const BackingMem *m) {
    vkDestroyBuffer(gpu_vk.device, (VkBuffer)(uintptr_t)m->buffer, NULL);
    vkFreeMemory(gpu_vk.device, (VkDeviceMemory)(uintptr_t)m->memory, NULL);
}

// ---------- mirrors
// The device-local heap blocks come from: the heap of the memory type vk_block_new takes.
static uint32_t block_heap(void) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) && !(f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT))
            return mp.memoryTypes[i].heapIndex;
    }
    return UINT32_MAX;
}

static int vk_vram(uint64_t *used, uint64_t *budget) {
    uint32_t heap = block_heap();
    if (heap == UINT32_MAX) return 0;
    VkPhysicalDeviceMemoryBudgetPropertiesEXT b = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_BUDGET_PROPERTIES_EXT};
    VkPhysicalDeviceMemoryProperties2 mp = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MEMORY_PROPERTIES_2, .pNext = &b};
    vkGetPhysicalDeviceMemoryProperties2(gpu_vk.phys, &mp);
    *used = b.heapUsage[heap];
    *budget = b.heapBudget[heap];
    return *budget != 0;
}

typedef struct Stage {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    uint64_t size, used, value;   // value: the submission it serves now (free once done)
} Stage;
static Stage *g_stages;
static int g_nstages;

static void stage_free(Stage *s) {
    vkDestroyBuffer(gpu_vk.device, s->buf, NULL);
    vkFreeMemory(gpu_vk.device, s->mem, NULL);
}

static Stage *stage_new(uint64_t size) {
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    Stage s = {.size = size};
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &s.buf), "vkCreateBuffer (staging)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, s.buf, &mr);
    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    uint32_t type = memory_type(mr.memoryTypeBits, hv | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
                                VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) type = memory_type(mr.memoryTypeBits, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) type = memory_type(mr.memoryTypeBits, hv, 0);
    if (type == UINT32_MAX) rt_fatal("gfx: vulkan: no host-visible, host-coherent memory type for staging");
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                  .memoryTypeIndex = type};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &s.mem), "vkAllocateMemory (staging)");
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, s.buf, s.mem, 0), "vkBindBufferMemory (staging)");
    void *p;
    GPU_VK_OK(vkMapMemory(gpu_vk.device, s.mem, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory (staging)");
    s.map = p;
    Stage *t = realloc(g_stages, (size_t)(g_nstages + 1) * sizeof *t);
    if (!t) rt_fatal("gfx: vulkan: out of memory for %d staging chunks", g_nstages + 1);
    g_stages = t;
    g_stages[g_nstages] = s;
    return &g_stages[g_nstages++];
}

// n bytes of staging for submission value: its chunk, and the offset in *at. On the submission's thread.
static Stage *stage_take(uint64_t value, uint64_t n, uint64_t *at) {
    uint64_t done = gpu_sync_done(), need = (n + STAGE_ALIGN - 1) & ~(STAGE_ALIGN - 1);
    for (int i = 0; i < g_nstages;) {   // chunks of their own whose submission is done go
        if (g_stages[i].size > STAGE_CHUNK && g_stages[i].value <= done) {
            stage_free(&g_stages[i]);
            g_stages[i] = g_stages[--g_nstages];
        } else {
            i++;
        }
    }
    Stage *s = NULL;
    for (int i = 0; i < g_nstages && !s; i++)
        if (g_stages[i].value == value && g_stages[i].size - g_stages[i].used >= need) s = &g_stages[i];
    for (int i = 0; i < g_nstages && !s; i++)
        if (g_stages[i].value <= done && g_stages[i].size >= need) s = &g_stages[i], s->used = 0;
    if (!s) s = stage_new(need > STAGE_CHUNK ? need : STAGE_CHUNK);
    s->value = value;
    *at = s->used;
    s->used += need;
    return s;
}

static void full_barrier(VkCommandBuffer cb, VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = dst_stage,
                           .dstAccessMask = dst_access};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void vk_upload(uint64_t value, const BackingCopy *c, int n) {
    VkCommandBuffer cb = gpu_vk_prologue();
    for (int i = 0; i < n; i++) {
        uint64_t at;
        Stage *s = stage_take(value, c[i].size, &at);
        memcpy(s->map + at, c[i].host, c[i].size);
        VkBufferCopy r = {.srcOffset = at, .dstOffset = c[i].offset, .size = c[i].size};
        vkCmdCopyBuffer(cb, s->buf, (VkBuffer)(uintptr_t)c[i].mem->buffer, 1, &r);
    }
}

static void vk_readback(uint64_t value, BackingCopy *c, int n) {
    VkCommandBuffer cb = gpu_vk_epilogue();   // not the main command buffer: the recorder thread records that one
    full_barrier(cb, VK_PIPELINE_STAGE_2_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT);
    for (int i = 0; i < n; i++) {
        uint64_t at;
        Stage *s = stage_take(value, c[i].size, &at);
        VkBufferCopy r = {.srcOffset = c[i].offset, .dstOffset = at, .size = c[i].size};
        vkCmdCopyBuffer(cb, (VkBuffer)(uintptr_t)c[i].mem->buffer, s->buf, 1, &r);
        c[i].host = s->map + at;
    }
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_TRANSFER_BIT,
                           .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT, .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

static void vk_close(void) {
    for (int i = 0; i < g_nstages; i++) stage_free(&g_stages[i]);
    free(g_stages);
    g_stages = NULL;
    g_nstages = 0;
}

const BackingOps gpu_vk_backing_ops = {.block_new = vk_block_new, .import = vk_import, .free = vk_free,
                                       .vram = vk_vram, .upload = vk_upload, .readback = vk_readback,
                                       .close = vk_close};

// Mirrors on a discrete device, where imported host memory is read across the bus (D14; research 14 A7: 22.9 times
// slower than VRAM); imports on any other, where VRAM is host memory too.
static int want_mirror(const BackingAsk *ask) {
    (void)ask;
    return gpu_vk.facts.props.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
}

void gpu_vk_mirror_policy(void) { backing_set_mirror_hook(want_mirror); }
