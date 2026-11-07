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
#include <string.h>

#include "oot2/backing.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"

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

const BackingOps gpu_vk_backing_ops = {.block_new = vk_block_new, .import = vk_import, .free = vk_free};
