// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The Vulkan back end (architecture section 6): its settings, the choice and creation of its device, its one queue,
// the submissions on the timeline of oot2/gpu.h, and the device-loss report (D17b). Every call goes through Volk
// (runtime/gpu/vk_volk.c), so a machine without a Vulkan loader starts and fails only when it asks for this back end.
//
// The back end opens once (gpu_vk_open, from gfx_select): the settings below are read then, the device is chosen and
// created with every item of oot2/vk_required.h enabled, and the completion thread starts over the device's timeline
// semaphore. One queue carries every submission, the replayed ones (GfxBackend.submit, every NVN2 queue in one order,
// as the null back end serialises them) and the back end's own (gpu_vk_begin to gpu_vk_end), in the order of their
// values.
//
// Any Vulkan call of the back end that fails stops the run by name (gpu_vk_failed). VK_ERROR_DEVICE_LOST first writes
// the device-loss report to the game folder's logs folder (gpu.h): the fault description, address and vendor
// information of VK_EXT_device_fault with the vendor's binary saved beside it, the last diagnostic checkpoints the queue
// reached, the timeline's values and the memory heaps; then the run stops with the report's path. A second thread that
// meets the loss waits for the first one's report and never returns.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/vk_required.h"

// ---- settings (read once, by gpu_vk_open; the settings store of the configuration menu will keep these names)
//   OOT2_VK_DEVICE       which device: unset or empty, the first discrete device with every item, else the first
//                        integrated, else the first other; digits below the device count, the device at that index of
//                        vkEnumeratePhysicalDevices; anything else, digits past the count included (3060), the first
//                        device whose name contains it, ignoring case
//   OOT2_VK_VALIDATION   off (the default), on (the Khronos validation layer), sync (with synchronization validation),
//                        gpu (with GPU-assisted validation); the layer must be installed when asked for
//   OOT2_VK_FAULT_REPORT on (the default): the device fault and checkpoint extensions are enabled and every submission
//                        carries checkpoints; off: neither extension is enabled and no checkpoint is recorded, so the
//                        report of a lost device holds the timeline and the heaps only
// A value outside these stops the start by name.
enum { GPU_VALIDATION_OFF, GPU_VALIDATION_ON, GPU_VALIDATION_SYNC, GPU_VALIDATION_GPU };
typedef struct GpuVkSettings {
    char device[256];
    int validation;     // GPU_VALIDATION_*
    int fault_report;   // 1 on, 0 off
} GpuVkSettings;
// The settings for these values of the three variables (NULL or "" for one not set): 1 with *out filled, or 0 with the
// reason in why.
int gpu_vk_settings(const char *device, const char *validation, const char *fault_report, GpuVkSettings *out,
                    char *why, size_t why_size);

// ---- the choice of device
typedef struct GpuVkCandidate {
    const char *name;
    VkPhysicalDeviceType type;
    unsigned char has[OOT2_VK_ITEM_COUNT];   // oot2_vk_judge
    const char *unusable;                    // NULL, or what else rules it out (no graphics and compute queue)
} GpuVkCandidate;
// The index of the device setting (GpuVkSettings.device) picks among c[0..n), or -1 with the reason in why: a device
// that lacks an item, or is unusable, is never picked, and why names every item it lacks.
int gpu_vk_choose(const GpuVkCandidate *c, int n, const char *setting, char *why, size_t why_size);

// ---- the device
typedef struct GpuVk {
    VkInstance instance;
    VkDebugUtilsMessengerEXT messenger;
    VkPhysicalDevice phys;
    int index;                  // in vkEnumeratePhysicalDevices
    Oot2VkFacts facts;          // what the device has (oot2_vk_query)
    VkDevice device;
    uint32_t family;            // the queue family of the one queue
    VkQueueFlags family_flags;
    VkQueue queue;
    VkSemaphore timeline;
    VkCommandPool pool;         // command buffers of submissions, used under the submission lock only
    GpuVkSettings settings;
    int fault, checkpoints;     // VK_EXT_device_fault (with its vendor binary when fault == 2), the checkpoints
} GpuVk;
extern GpuVk gpu_vk;            // valid once gpu_vk_open returned

void gpu_vk_open(void);         // once; later calls return at once
// Finish every submission, stop the completion thread and destroy the device and the instance (tests; a run ends
// without it). gpu_vk_open may follow.
void gpu_vk_close(void);
// Messages of the validation layer since gpu_vk_open: errors and warnings (performance warnings included).
void gpu_vk_validation_counts(uint64_t *errors, uint64_t *warnings);

// Stop the run by name for result r of the call named what: the device-loss report first when r is
// VK_ERROR_DEVICE_LOST.
__attribute__((noreturn)) void gpu_vk_failed(VkResult r, const char *what);
#define GPU_VK_OK(call, what)                                                                                          \
    do {                                                                                                               \
        VkResult gpu_vk_r_ = (call);                                                                                   \
        if (gpu_vk_r_ != VK_SUCCESS) gpu_vk_failed(gpu_vk_r_, what);                                                   \
    } while (0)
const char *gpu_vk_result_name(VkResult r);   // "VK_ERROR_DEVICE_LOST", ...; "VkResult <n>" for others

// ---- submissions on the timeline (vk_submit.c)
// The back end's own work (tests, uploads, conversions, page-table writes) and the replay both submit this way:
// gpu_vk_begin takes the submission lock and returns the value the submission signals; gpu_vk_cmd gives the command
// buffer it records into (begun at the first call, with a checkpoint when the report is on; none is submitted when it is
// never asked for); gpu_vk_effect queues fn(arg, value) for the completion thread, after the device reached the value
// and before the value is done; gpu_vk_end ends the command buffer, submits it to the one queue after the given waits,
// signalling the value at all commands, and releases the lock. Only the thread between begin and end may call the
// others. kind is GPU_VK_INTERNAL or GPU_VK_REPLAY, written in the checkpoints.
enum { GPU_VK_REPLAY = 1, GPU_VK_INTERNAL = 2 };
uint64_t gpu_vk_begin(int kind);
VkCommandBuffer gpu_vk_cmd(void);
void gpu_vk_effect(GpuEffect fn, void *arg);
uint64_t gpu_vk_end(const VkSemaphoreSubmitInfo *waits, uint32_t nwaits);

// A checkpoint's marker: the value, the kind and 1 at the end of the submission's command buffer (0 at its start).
#define GPU_VK_MARKER(value, kind, end) ((uint64_t)(value) << 4 | (uint64_t)(kind) << 1 | (uint64_t)(end))

// The back-end table (vk_submit.c): submit, fence_wait, fence_done and finish on the timeline; the operations later
// tickets bring stop by name.
extern const GfxBackend gfx_vulkan_backend;
