// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The raw path on the device (oot2/vk_raw.h, architecture D15): the copies of the page table, the out buffers of the
// write-back, the runtime shader raw_writeback.comp, and the effect that logs misses and writes raw stores into mirrors
// back to guest memory.
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "oot2/pagegen.h"
#include "oot2/runtime.h"
#include "oot2/vk_raw.h"
#include "plat.h"

#define PAGE 0x1000ull
#define OUT_HEAD 16u   // raw_writeback.comp: the table's head is copied from this word of the out buffer
#define HEAD_WORDS (RAW_MISS_LOG_WORD + RAW_MISS_ENTRIES * RAW_MISS_ENTRY_WORDS)
#define PAGE_WORDS 1024u

typedef unsigned long long ull;

// Memory with one buffer over it, mapped, and the buffer's device address.
typedef struct Mem {
    VkBuffer buf;
    VkDeviceMemory mem;
    uint8_t *map;
    uint64_t address, size;
} Mem;

typedef struct Copy {
    Mem m;
    uint64_t version;    // the map's version it was built at; 0 never built
    uint64_t last_use;   // the newest submission that read it
    RawLayout l;
    uint64_t pages_at;   // byte offset of the mirrored pages' list
    uint64_t *guest;     // the guest address of each mirrored page (host memory)
} Copy;

typedef struct Out {
    Mem m;
    uint32_t slots;
    struct Out *next;
} Out;

// What an end hands its effect.
typedef struct Done {
    Out *out;
    const uint64_t *guest;
    uint32_t slots;
} Done;

static Copy g_copy[RAW_COPIES_MAX];
static int g_ncopies;
static Copy *g_current;              // the copy the newest submission used
static uint64_t g_value;             // the submission that began and has not ended (one at a time: the gate)
static uint32_t g_push[4];
static uint32_t *g_image;            // the build's host image
static uint64_t g_image_bytes;
static RawMirrorPage *g_pages;
static uint64_t *g_guest_scratch;
static uint64_t g_pages_cap, g_guest_cap;
static VkShaderModule g_module;
static VkPipelineLayout g_layout;
static VkPipeline g_pipe;
static PlatMutex g_lock = PLAT_MUTEX_INIT;   // the out buffers' free list and the counters
static Out *g_free;
static GpuVkRawStats g_st;

// ---------- memory
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

// size bytes of memory of the first type that has every flag of one of the wants, in order, with a buffer the shaders
// reach by address; mapped.
static Mem mem_new(uint64_t size, const VkMemoryPropertyFlags *wants, int nwants, const char *what) {
    Mem m = {.size = size};
    VkBufferCreateInfo bi = {.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
                             .usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT |
                                      VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                             .sharingMode = VK_SHARING_MODE_EXCLUSIVE};
    GPU_VK_OK(vkCreateBuffer(gpu_vk.device, &bi, NULL, &m.buf), "vkCreateBuffer (raw path)");
    VkMemoryRequirements mr;
    vkGetBufferMemoryRequirements(gpu_vk.device, m.buf, &mr);
    VkMemoryAllocateFlagsInfo fi = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO,
                                    .flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT};
    VkResult r = VK_ERROR_OUT_OF_DEVICE_MEMORY;
    for (int i = 0; i < nwants && r != VK_SUCCESS; i++) {
        uint32_t type = memory_type(mr.memoryTypeBits, wants[i]);
        if (type == UINT32_MAX) continue;
        VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .pNext = &fi,
                                      .allocationSize = mr.size, .memoryTypeIndex = type};
        r = vkAllocateMemory(gpu_vk.device, &alloc, NULL, &m.mem);
        if (r == VK_ERROR_DEVICE_LOST) gpu_vk_failed(r, "vkAllocateMemory (raw path)");
    }
    if (r != VK_SUCCESS)
        rt_fatal("gfx: vulkan: raw path: no host-visible memory for %s of %llu KiB (%s)", what, (ull)(size >> 10),
                 gpu_vk_result_name(r));
    GPU_VK_OK(vkBindBufferMemory(gpu_vk.device, m.buf, m.mem, 0), "vkBindBufferMemory (raw path)");
    void *p;
    GPU_VK_OK(vkMapMemory(gpu_vk.device, m.mem, 0, VK_WHOLE_SIZE, 0, &p), "vkMapMemory (raw path)");
    m.map = p;
    VkBufferDeviceAddressInfo di = {.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO, .buffer = m.buf};
    m.address = vkGetBufferDeviceAddress(gpu_vk.device, &di);
    return m;
}

static void mem_free(Mem *m) {
    if (!m->buf) return;
    vkDestroyBuffer(gpu_vk.device, m->buf, NULL);
    vkFreeMemory(gpu_vk.device, m->mem, NULL);
    memset(m, 0, sizeof *m);
}

static const VkMemoryPropertyFlags k_table[] = {
    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};
static const VkMemoryPropertyFlags k_out[] = {
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT | VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT};

// ---------- the table's copies
static void wait_for(uint64_t value) {
    if (value <= gpu_sync_done()) return;
    int token = rt_kernel_block_begin();
    gpu_sync_wait(value, UINT64_MAX);
    rt_kernel_block_end(token);
}

// A copy no submission in flight reads, other than the current one: a done one, else one already issued (waited for),
// else a new one. The current copy is kept, since the submission being recorded may follow one that still reads it.
static Copy *copy_free(void) {
    uint64_t done = gpu_sync_done(), issued = gpu_sync_issued();
    Copy *wait = NULL;
    for (int i = 0; i < g_ncopies; i++) {
        Copy *c = &g_copy[i];
        if (c == g_current) continue;
        if (c->last_use <= done) return c;
        if (c->last_use <= issued && (!wait || c->last_use < wait->last_use)) wait = c;
    }
    if (g_ncopies < RAW_COPIES_MAX) return &g_copy[g_ncopies++];
    if (!wait)
        rt_fatal("gfx: vulkan: raw path: every one of the %d copies of the page table is read by a submission that is "
                 "not issued yet", RAW_COPIES_MAX);
    plat_mutex_lock(&g_lock);
    g_st.copy_waits++;
    plat_mutex_unlock(&g_lock);
    wait_for(wait->last_use);
    return wait;
}

static void *grow_to(void *p, uint64_t *cap, uint64_t need, const char *what) {
    if (need <= *cap) return p;
    uint64_t c = need + need / 4;
    void *q = realloc(p, (size_t)c);
    if (!q) rt_fatal("gfx: vulkan: raw path: out of memory for %s (%llu KiB)", what, (ull)(c >> 10));
    *cap = c;
    return q;
}

// Build the table of the map as it is now into a free copy, which becomes the current one.
static void build(void) {
    uint64_t t0 = plat_now_ns(), version;
    BackingRun *runs;
    int n = backing_raw_runs(&runs, &version);
    RawLayout l;
    char why[256];
    if (!raw_layout(runs, n, &l, why, sizeof why)) rt_fatal("gfx: vulkan: raw path: the page table: %s", why);
    uint64_t table_bytes = (l.words * 4 + 15) & ~15ull, bytes = table_bytes + l.mirror_pages * sizeof(RawMirrorPage);
    g_image = grow_to(g_image, &g_image_bytes, table_bytes, "the page table's image");
    g_pages = grow_to(g_pages, &g_pages_cap, (l.mirror_pages + 1) * sizeof *g_pages, "the mirrored pages");
    g_guest_scratch = grow_to(g_guest_scratch, &g_guest_cap, (l.mirror_pages + 1) * sizeof *g_guest_scratch,
                              "the mirrored pages");
    raw_write(runs, n, &l, g_image, g_pages, g_guest_scratch);
    free(runs);
    Copy *c = copy_free();
    if (c->m.size < bytes) {
        mem_free(&c->m);
        c->m = mem_new(bytes + bytes / 4, k_table, 2, "a copy of the page table");
        plat_mutex_lock(&g_lock);
        g_st.copies++;
        plat_mutex_unlock(&g_lock);
    }
    memcpy(c->m.map, g_image, (size_t)l.words * 4);
    memcpy(c->m.map + table_bytes, g_pages, (size_t)(l.mirror_pages * sizeof *g_pages));
    free(c->guest);
    c->guest = malloc((size_t)(l.mirror_pages + 1) * sizeof *c->guest);
    if (!c->guest) rt_fatal("gfx: vulkan: raw path: out of memory for %llu mirrored pages", (ull)l.mirror_pages);
    memcpy(c->guest, g_guest_scratch, (size_t)l.mirror_pages * sizeof *c->guest);
    c->l = l, c->version = version, c->pages_at = table_bytes;
    g_current = c;
    uint64_t ns = plat_now_ns() - t0;
    plat_mutex_lock(&g_lock);
    g_st.builds++, g_st.build_ns += ns;
    if (ns > g_st.build_ns_max) g_st.build_ns_max = ns;
    g_st.table_bytes = l.words * 4, g_st.pages = l.pages, g_st.mirror_pages = l.mirror_pages;
    g_st.blocks = l.blocks, g_st.span_count = l.span_count;
    plat_mutex_unlock(&g_lock);
}

void gpu_vk_raw_begin(uint64_t value, uint32_t push[4]) {
    if (g_value != value) {
        if (g_value) rt_fatal("gfx: vulkan: raw path: submission %llu began while %llu had not ended", (ull)value,
                              (ull)g_value);
        backing_raw_begin(value);
        if (!g_current || g_current->version != backing_version()) build();
        g_current->last_use = value;
        g_value = value;
        g_push[RAW_PUSH_TABLE_LOW] = (uint32_t)g_current->m.address;
        g_push[RAW_PUSH_TABLE_HIGH] = (uint32_t)(g_current->m.address >> 32);
        g_push[RAW_PUSH_SPAN_BASE] = g_current->l.span_base;
        g_push[RAW_PUSH_SPAN_COUNT] = g_current->l.span_count;
    }
    memcpy(push, g_push, sizeof g_push);
}

// ---------- the write-back
static void pipeline(void) {
    if (g_pipe) return;
    const GpuShader *s = gpu_shader("raw_writeback");
    if (!s) rt_fatal("gfx: vulkan: raw path: the runtime shader raw_writeback is not embedded");
    VkShaderModuleCreateInfo mi = {.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .codeSize = s->size,
                                   .pCode = s->code};
    GPU_VK_OK(vkCreateShaderModule(gpu_vk.device, &mi, NULL, &g_module), "vkCreateShaderModule (raw write-back)");
    VkPushConstantRange pr = {.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT, .offset = 0, .size = 44};
    VkPipelineLayoutCreateInfo li = {.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
                                     .pushConstantRangeCount = 1, .pPushConstantRanges = &pr};
    GPU_VK_OK(vkCreatePipelineLayout(gpu_vk.device, &li, NULL, &g_layout), "vkCreatePipelineLayout (raw write-back)");
    VkComputePipelineCreateInfo ci = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, .stage = VK_SHADER_STAGE_COMPUTE_BIT,
                  .module = g_module, .pName = "main"},
        .layout = g_layout};
    GPU_VK_OK(vkCreateComputePipelines(gpu_vk.device, VK_NULL_HANDLE, 1, &ci, NULL, &g_pipe),
              "vkCreateComputePipelines (raw write-back)");
}

static uint64_t data_word(uint32_t slots) {
    return (OUT_HEAD + HEAD_WORDS + slots + PAGE_WORDS - 1) & ~(uint64_t)(PAGE_WORDS - 1);
}

static Out *out_take(uint32_t slots) {
    plat_mutex_lock(&g_lock);
    Out **at = &g_free;
    while (*at && (*at)->slots < slots) at = &(*at)->next;
    Out *o = *at;
    if (o) *at = o->next;
    plat_mutex_unlock(&g_lock);
    if (!o) {
        o = calloc(1, sizeof *o);
        if (!o) rt_fatal("gfx: vulkan: raw path: out of memory for an out buffer");
        o->slots = slots;
        o->m = mem_new((data_word(slots) + (uint64_t)slots * PAGE_WORDS) * 4, k_out, 2, "the raw write-back");
    }
    memset(o->m.map, 0, (size_t)(OUT_HEAD + HEAD_WORDS) * 4);
    return o;
}

static void miss_log(const uint32_t *head, uint64_t value) {
    uint32_t count = head[RAW_HEADER_MISS_COUNT], kept = count < RAW_MISS_ENTRIES ? count : RAW_MISS_ENTRIES;
    static const char *const kinds[] = {"?", "load", "store", "atomic"};
    static const char *const reasons[] = {"?", "outside the span", "no chunk", "no page"};
    plat_mutex_lock(&g_lock);
    g_st.misses += count, g_st.miss_submissions++, g_st.last_miss_value = value, g_st.last_misses = kept;
    for (uint32_t i = 0; i < kept; i++) {
        const uint32_t *e = head + RAW_MISS_LOG_WORD + i * RAW_MISS_ENTRY_WORDS;
        g_st.last_miss[i] = (RawMiss){(uint64_t)e[RAW_MISS_ADDRESS_HIGH] << 32 | e[RAW_MISS_ADDRESS_LOW],
                                      e[RAW_MISS_KIND], e[RAW_MISS_REASON]};
    }
    plat_mutex_unlock(&g_lock);
    rt_log("gfx: vulkan: raw path: submission %llu missed the page table %u times (a miss reads 0 and drops its write)",
           (ull)value, count);
    for (uint32_t i = 0; i < kept; i++) {
        const uint32_t *e = head + RAW_MISS_LOG_WORD + i * RAW_MISS_ENTRY_WORDS;
        uint32_t k = e[RAW_MISS_KIND], r = e[RAW_MISS_REASON];
        rt_log("gfx: vulkan: raw path:   %s at %#llx: %s", kinds[k < 4 ? k : 0],
               (ull)((uint64_t)e[RAW_MISS_ADDRESS_HIGH] << 32 | e[RAW_MISS_ADDRESS_LOW]), reasons[r < 4 ? r : 0]);
    }
}

// The completion thread, once the device reached value and before it is done.
static void ended(void *arg, uint64_t value) {
    Done *d = arg;
    const uint32_t *w = (const uint32_t *)d->out->m.map;
    if (w[OUT_HEAD + RAW_HEADER_MISS_COUNT]) miss_log(w + OUT_HEAD, value);
    uint32_t taken = w[0];
    if (taken > d->slots)
        rt_fatal("gfx: vulkan: raw path: the raw stores of submission %llu reached %u mirrored pages, more than the %u "
                 "its write-back holds (RAW_WRITEBACK_PAGES)", (ull)value, taken, d->slots);
    uint64_t data = data_word(d->slots), bytes = 0, skipped = 0;
    for (uint32_t i = 0; i < taken; i++) {
        uint64_t guest = d->guest[w[OUT_HEAD + HEAD_WORDS + i]];
        if (!rt_mem_commit_io(guest, PAGE, 1)) {
            if (!skipped++)
                rt_log("gfx: vulkan: raw path: a raw store of submission %llu into the mirror of %#llx lands in memory "
                       "the guest no longer maps: left out", (ull)value, (ull)guest);
            continue;
        }
        memcpy((void *)(uintptr_t)guest, w + data + (uint64_t)i * PAGE_WORDS, PAGE);
        pagegen_mark(guest, guest + PAGE);
        bytes += PAGE;
    }
    plat_mutex_lock(&g_lock);
    g_st.writebacks += taken - skipped, g_st.writeback_bytes += bytes, g_st.writebacks_skipped += skipped;
    d->out->next = g_free, g_free = d->out;
    plat_mutex_unlock(&g_lock);
    free(d);
}

void gpu_vk_raw_end(uint64_t value) {
    if (!g_value || g_value != value) return;
    g_value = 0;
    Copy *c = g_current;
    uint32_t slots = c->l.mirror_pages < RAW_WRITEBACK_PAGES ? (uint32_t)c->l.mirror_pages : RAW_WRITEBACK_PAGES;
    Out *o = out_take(slots);
    Done *d = malloc(sizeof *d);
    if (!d) rt_fatal("gfx: vulkan: raw path: out of memory for a submission's end");
    *d = (Done){o, c->guest, slots};
    pipeline();
    VkCommandBuffer cb = gpu_vk_epilogue();   // after a full barrier over the submission's own work
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g_pipe);
    uint64_t pages = c->m.address + c->pages_at;
    uint32_t pc[11] = {(uint32_t)c->m.address, (uint32_t)(c->m.address >> 32), (uint32_t)pages,
                       (uint32_t)(pages >> 32), (uint32_t)o->m.address, (uint32_t)(o->m.address >> 32),
                       (uint32_t)c->l.mirror_pages, slots, RAW_STORE_BITS_WORD, RAW_HEADER_MISS_COUNT, HEAD_WORDS};
    vkCmdPushConstants(cb, g_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    uint64_t groups = c->l.mirror_pages ? c->l.mirror_pages : 1;
    vkCmdDispatch(cb, (uint32_t)(groups < 65535 ? groups : 65535), 1, 1);
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT,
                           .srcAccessMask = VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_HOST_BIT | VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_HOST_READ_BIT | VK_ACCESS_2_MEMORY_READ_BIT |
                                            VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
    gpu_sync_effect(ended, d);
    plat_mutex_lock(&g_lock);
    g_st.submissions++;
    plat_mutex_unlock(&g_lock);
}

void gpu_vk_raw_stats(GpuVkRawStats *out) {
    plat_mutex_lock(&g_lock);
    *out = g_st;
    plat_mutex_unlock(&g_lock);
}

void gpu_vk_raw_close(void) {
    for (int i = 0; i < g_ncopies; i++) {
        mem_free(&g_copy[i].m);
        free(g_copy[i].guest);
    }
    memset(g_copy, 0, sizeof g_copy);
    g_ncopies = 0, g_current = NULL, g_value = 0;
    while (g_free) {
        Out *o = g_free;
        g_free = o->next;
        mem_free(&o->m);
        free(o);
    }
    if (g_pipe) vkDestroyPipeline(gpu_vk.device, g_pipe, NULL);
    if (g_layout) vkDestroyPipelineLayout(gpu_vk.device, g_layout, NULL);
    if (g_module) vkDestroyShaderModule(gpu_vk.device, g_module, NULL);
    g_pipe = VK_NULL_HANDLE, g_layout = VK_NULL_HANDLE, g_module = VK_NULL_HANDLE;
    free(g_image), free(g_pages), free(g_guest_scratch);
    g_image = NULL, g_pages = NULL, g_guest_scratch = NULL, g_image_bytes = g_pages_cap = g_guest_cap = 0;
    memset(&g_st, 0, sizeof g_st);
}
