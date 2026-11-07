// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Submissions of the Vulkan back end on the timeline (oot2/gpu_vk.h, oot2/gpu.h), and its back-end table.
//
// Every submission, replayed or the back end's own, goes through gpu_vk_begin and gpu_vk_end, which wrap the timeline's
// gpu_sync_begin and gpu_sync_end around one vkQueueSubmit2 on the one queue: the command buffer, if one was asked for,
// after the caller's waits, signalling the submission's value on the timeline semaphore at all commands. The submission
// lock spans the whole, so the queue (which Vulkan requires to be used by one thread at a time) and the command pool
// are used by one thread at a time, and the values reach the semaphore in order. The completion thread waits on that
// semaphore (the source below) and runs each value's effects before it publishes the value (gpu_sync.c).
//
// Command buffers come from a set of slots: a slot is taken again once the value it was submitted with is done, so a
// slot never holds a command buffer the GPU may still read; the set grows to the number of submissions in flight. With
// the report on, each command buffer starts and ends with a diagnostic checkpoint naming its value and kind, so the
// last checkpoints a lost queue reached say which submission was running (per-draw checkpoints come with the recorder).
//
// The replay of this back end walks each recording and stops by name at the first record that is not its begin or its
// end: no op is replayed yet. A replayed submission still takes its value and a command buffer, so fences, finish and
// the checkpoints work on it as they will once ops are replayed.
//
// Adapted from DKB Recomp runtime/vk_sync.c at 9d8fa02 (GPL-3.0-or-later; there adapted from LAS Recomp
// runtime/nvn/vulkan/exec.c at da35b83, MIT, Copyright (c) 2026 Peiikai, licenses/lasrecomp-MIT.txt): a primary command
// buffer per submission taken from reusable slots, vkQueueSubmit2 signalling the submission's value at all commands, a
// failed wait treated as a lost device, a stand-in replay that accepts only empty recordings. Changed here: the back
// end's own work submits on the same timeline through the same calls, with effects of its own; checkpoints mark each
// command buffer; slots are freed by the published value; a recording holding an op stops with that op's name.
#include <stdlib.h>
#include <string.h>

#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "plat.h"

// ---------- the timeline's source: the device's timeline semaphore
static uint64_t semaphore_wait(void *ctx, uint64_t value, uint64_t timeout_ns) {
    (void)ctx;
    VkSemaphoreWaitInfo wi = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_WAIT_INFO, .semaphoreCount = 1,
                              .pSemaphores = &gpu_vk.timeline, .pValues = &value};
    VkResult r = vkWaitSemaphores(gpu_vk.device, &wi, timeout_ns);
    if (r != VK_SUCCESS && r != VK_TIMEOUT) gpu_vk_failed(r, "vkWaitSemaphores (the completion thread)");
    uint64_t counter = 0;
    GPU_VK_OK(vkGetSemaphoreCounterValue(gpu_vk.device, gpu_vk.timeline, &counter),
              "vkGetSemaphoreCounterValue (the completion thread)");
    return counter;
}

// ---------- command buffers (the submission lock's owner only)
typedef struct {
    VkCommandBuffer cb;
    uint64_t value;   // the submission that used it last; free once that value is done
} Slot;
static Slot *g_slots;
static int g_nslots;
static VkCommandBuffer g_cur;   // the open submission's command buffer, once asked for
static VkCommandBuffer g_pro;   // its prologue, once asked for
static uint64_t g_value;        // the open submission's value, 0 when none is open
static int g_kind;

void gpu_vk_start_timeline(uint64_t after) {
    GpuTimelineSource src = {.wait = semaphore_wait};
    gpu_sync_start(&src, after);
}

void gpu_vk_stop_timeline(void) {
    gpu_sync_stop();
    free(g_slots);   // their command buffers go with the pool
    g_slots = NULL;
    g_nslots = 0;
}

static void checkpoint(int end) {
    if (gpu_vk.checkpoints)
        vkCmdSetCheckpointNV(g_cur, (const void *)(uintptr_t)GPU_VK_MARKER(g_value, g_kind, end));
}

uint64_t gpu_vk_begin(int kind) {
    if (!gpu_vk.device) rt_fatal("gfx: vulkan: a submission before the device is open");
    uint64_t value = gpu_sync_begin();
    g_value = value;
    g_kind = kind;
    g_cur = g_pro = VK_NULL_HANDLE;
    return value;
}

// A command buffer from a free slot, begun for the open submission.
static VkCommandBuffer slot_begin(void) {
    uint64_t done = gpu_sync_done();
    int i = 0;
    while (i < g_nslots && g_slots[i].value > done) i++;
    if (i == g_nslots) {
        Slot *s = realloc(g_slots, (size_t)(g_nslots + 1) * sizeof *s);
        if (!s) rt_fatal("gfx: vulkan: out of memory for %d command buffers", g_nslots + 1);
        g_slots = s;
        VkCommandBufferAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO,
                                          .commandPool = gpu_vk.pool, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY,
                                          .commandBufferCount = 1};
        GPU_VK_OK(vkAllocateCommandBuffers(gpu_vk.device, &alloc, &g_slots[i].cb), "vkAllocateCommandBuffers");
        g_nslots++;
    }
    g_slots[i].value = g_value;
    VkCommandBufferBeginInfo bi = {.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO,
                                   .flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT};
    GPU_VK_OK(vkBeginCommandBuffer(g_slots[i].cb, &bi), "vkBeginCommandBuffer");
    return g_slots[i].cb;
}

VkCommandBuffer gpu_vk_cmd(void) {
    if (!g_value) rt_fatal("gfx: vulkan: a command buffer outside a submission");
    if (g_cur) return g_cur;
    g_cur = slot_begin();
    checkpoint(0);
    return g_cur;
}

// Every write before it made available and visible to every access after it, in submission order (D18's barrier).
static void full_barrier(VkCommandBuffer cb) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2,
                           .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT,
                           .dstStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .dstAccessMask = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb};
    vkCmdPipelineBarrier2(cb, &di);
}

VkCommandBuffer gpu_vk_prologue(void) {
    if (!g_value) rt_fatal("gfx: vulkan: a prologue outside a submission");
    if (g_pro) return g_pro;
    g_pro = slot_begin();
    full_barrier(g_pro);
    return g_pro;
}

void gpu_vk_effect(GpuEffect fn, void *arg) {
    if (!g_value) rt_fatal("gfx: vulkan: an effect outside a submission");
    gpu_sync_effect(fn, arg);
}

uint64_t gpu_vk_end(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits) {
    if (!g_value) rt_fatal("gfx: vulkan: the end of a submission that was not begun");
    uint64_t value = g_value;
    VkCommandBufferSubmitInfo cbi[2];
    uint32_t ncb = 0;
    if (g_pro) {
        full_barrier(g_pro);
        GPU_VK_OK(vkEndCommandBuffer(g_pro), "vkEndCommandBuffer (the prologue)");
        cbi[ncb++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                 .commandBuffer = g_pro};
    }
    if (g_cur) {
        checkpoint(1);
        GPU_VK_OK(vkEndCommandBuffer(g_cur), "vkEndCommandBuffer");
        cbi[ncb++] = (VkCommandBufferSubmitInfo){.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO,
                                                 .commandBuffer = g_cur};
    }
    VkSemaphoreSubmitInfo signal = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = gpu_vk.timeline,
                                    .value = value, .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkSubmitInfo2 si = {.sType = VK_STRUCTURE_TYPE_SUBMIT_INFO_2, .waitSemaphoreInfoCount = nwaits,
                        .pWaitSemaphoreInfos = waits, .commandBufferInfoCount = ncb, .pCommandBufferInfos = cbi,
                        .signalSemaphoreInfoCount = 1, .pSignalSemaphoreInfos = &signal};
    GPU_VK_OK(vkQueueSubmit2(gpu_vk.queue, 1, &si, VK_NULL_HANDLE), "vkQueueSubmit2");
    g_cur = g_pro = VK_NULL_HANDLE;
    g_value = 0;
    gpu_sync_end();
    return value;
}

// ---------- what this back end replays: empty recordings only
void gpu_check_recording(uint64_t value, int index, uint64_t handle) {
    const uint8_t *p = (const uint8_t *)(uintptr_t)handle;
    if (!p) rt_fatal("gfx: vulkan: recording %d of submission %llu is a null handle", index, (unsigned long long)value);
    for (size_t at = 0;;) {
        op_hdr h;
        memcpy(&h, p + at, sizeof h);
        if (h.size < sizeof h || (h.size & 7) || h.op >= OP_COUNT_ || h.op == OP_JUMP || h.op == OP_CALL ||
            (at == 0) != (h.op == OP_BEGIN))
            rt_fatal("gfx: vulkan: recording %d of submission %llu is not whole at +%zu (op %u, %u bytes)", index,
                     (unsigned long long)value, at, h.op, h.size);
        if (h.op == OP_END) return;
        if (h.op != OP_BEGIN)
            rt_fatal("gfx: vulkan: recording %d of submission %llu holds %s at +%zu, and the Vulkan back end replays no "
                     "op yet", index, (unsigned long long)value, nvn_op_name(h.op), at);
        at += h.size;
    }
}

// ---------- the back-end table
// Pools are the backing map's (oot2/backing.h): pool_fini returns once no submission in flight used the pool.
static int vk_pool_init(pool_t *p) { return backing_pool_init(p); }
static void vk_pool_fini(pool_t *p) { backing_pool_fini(p); }
static int vk_pool_map_virtual(pool_t *v, int n, const void *requests) {
    return backing_pool_map_virtual(v, n, requests);
}

// The replay resolves its bindings in the submission (a mirror's first resolve brings it current in the prologue) and
// declares its GPU writes (backing_gpu_wrote); its end reads back what it wrote into mirrors.
static uint64_t vk_submit(const uint64_t *handles, int n) {
    backing_sync();
    uint64_t value = gpu_vk_begin(GPU_VK_REPLAY);
    for (int i = 0; i < n; i++) gpu_check_recording(value, i, handles[i]);
    gpu_vk_cmd();
    backing_submit_end(value);
    return gpu_vk_end(NULL, 0);
}

static int vk_fence_wait(uint64_t value, uint64_t timeout_ns) { return gpu_sync_wait(value, timeout_ns); }
static uint64_t vk_fence_done(void) { return gpu_sync_done(); }
static void vk_finish(void) { gpu_sync_finish(); }

// What later tickets bring: a call stops by name.
__attribute__((noreturn)) static void not_built(const char *what) {
    rt_fatal("gfx: vulkan: %s is not built yet in the Vulkan back end", what);
}
static void vk_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v) {
    (void)pool;
    (void)id;
    (void)t;
    (void)v;
    not_built("a texture registration (texture_register)");
}
static void vk_sampler_register(const dpool_ref *pool, int id, const samp_par *s) {
    (void)pool;
    (void)id;
    (void)s;
    not_built("a sampler registration (sampler_register)");
}
static void vk_present(const tex_t *t, const int crop[4]) {
    (void)t;
    (void)crop;
    not_built("present");
}
static void vk_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                           uint64_t img) {
    (void)t;
    (void)v;
    (void)r;
    (void)dst;
    (void)row;
    (void)img;
    not_built("a texel read (read_texels)");
}
static void vk_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src,
                            uint64_t row, uint64_t img) {
    (void)t;
    (void)v;
    (void)r;
    (void)src;
    (void)row;
    (void)img;
    not_built("a texel write (write_texels)");
}
// Nothing is kept for textures or programs yet: these have nothing to do.
static void vk_texture_release(const tex_t *t) { (void)t; }
static void vk_program_release(const prog_t *p) { (void)p; }
// The CPU's view of mirrored pool storage (oot2/backing.h, Mirrors).
static void vk_guest_written(uint64_t addr, uint64_t size) { backing_guest_written(addr, size); }
static void vk_guest_read(uint64_t addr, uint64_t size) { backing_guest_read(addr, size); }
static void vk_cpu_synced(uint64_t fence) { backing_cpu_synced(fence); }

const GfxBackend gfx_vulkan_backend = {
    .name = "vulkan",
    .pool_init = vk_pool_init,
    .pool_fini = vk_pool_fini,
    .pool_map_virtual = vk_pool_map_virtual,
    .texture_register = vk_texture_register,
    .sampler_register = vk_sampler_register,
    .texture_release = vk_texture_release,
    .program_release = vk_program_release,
    .submit = vk_submit,
    .fence_wait = vk_fence_wait,
    .fence_done = vk_fence_done,
    .present = vk_present,
    .finish = vk_finish,
    .read_texels = vk_read_texels,
    .write_texels = vk_write_texels,
    .guest_written = vk_guest_written,
    .guest_read = vk_guest_read,
    .cpu_synced = vk_cpu_synced,
};
