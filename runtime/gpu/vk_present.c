// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Present to a window on the Vulkan back end (oot2/vk_present.h has the design): the snapshot ring the guest thread fills,
// the present thread, the swapchain on FIFO, the scale into the window, the menu over it, and the shown time.
//
// Locks and the queue. One mutex (g_mu) guards the ring and the thread's mailbox; the guest thread holds it only to pick
// and to publish a slot, never across a submission. Every use of the one queue goes through gpu_vk_begin and gpu_vk_end
// (oot2/gpu_vk.h): the guest thread's snapshot, and the present thread's picture, whose gpu_vk_end_present also makes the
// vkQueuePresentKHR call while it still holds the submission, because that call needs the queue to itself too. Waiting on
// the display (vkAcquireNextImageKHR, which is where FIFO throttles) happens before the submission is begun, so a stalled
// display never holds the queue; waiting for a frame's resources is a wait on the timeline (gpu_sync_wait) with no lock
// held.
//
// Adapted from this repository's tools/presentcost/vkpresent.c (the present spike of tools/presentcost): the present id and the present timing
// request on every present, the time domain preferring the host's own counter, the calibration against it, and the
// results read with vkGetPastPresentationTimingEXT after each present; the swapchain made the way that harness makes it.
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif

#include "oot2/capture.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_image.h"
#include "oot2/vk_present.h"
#include "oot2/vsync.h"
#include "plat.h"
#include "vk_present_int.h"

#define SLOTS 4
#define FRAMES VKO_FRAMES
#define MAX_IMAGES 8
#define IDLE_NS 4000000ull            // the present thread looks round at least this often with nothing to show
#define REDRAW_NS 16000000ull         // a picture drawn again no sooner than this after the last
#define ACQUIRE_NS 100000000ull       // one wait for a swapchain image; the loop asks again until the window can stop it
#define SHOWN_RING 512
#define TIMING_QUEUE 64
#define TIMING_BATCH 16
#define RECALIBRATE_EVERY 120

enum { SLOT_FREE, SLOT_FILLING, SLOT_FILLED, SLOT_BUSY, SLOT_HELD };

typedef struct {
    VkImage image;
    VkDeviceMemory mem;
    uint32_t w, h;
    int state;
    uint64_t seq, value;   // the picture's number; the value of the snapshot's submission
    uint64_t last_use;     // the newest submission that touched the image: it goes once that is done
} Slot;

typedef struct {
    VkSwapchainKHR handle;
    VkExtent2D extent;
    VkFormat format;
    uint32_t n;
    VkImage images[MAX_IMAGES];
    VkImageView views[MAX_IMAGES];
    VkSemaphore done[MAX_IMAGES];
    int dirty, id2, timing;
    uint64_t domain_id;
    VkTimeDomainKHR domain_kind;
} Sc;

typedef struct {
    VkSemaphore acquire;
    uint64_t value;   // the submission that last used the frame's buffers and semaphore
} Frame;

typedef struct {
    uint64_t seq, returned_ns, timing_ns;
} Shown;

int vkw_no_feedback;
int vkw_no_recreate;

static const WinVulkan *g_win;
static const VkwUi *g_ui;
static VkSurfaceKHR g_surface;
static VkFormat g_surface_format;
static VkColorSpaceKHR g_surface_space;
static int g_active, g_stop, g_ended, g_timing_dev, g_ui_on;

static PlatMutex g_mu = PLAT_MUTEX_INIT;   // the ring, the mailbox, g_stop and g_ended
static PlatCond g_cv = PLAT_COND_INIT;
static Slot g_slots[SLOTS];
static int g_held = -1;                    // the slot the display shows now: the picture to draw again
static PlatThread *g_thread;
static uint64_t g_seq_fallback;
static _Thread_local uint64_t t_pace_seq;  // present_pace's seq, for the present() of the same call on the same thread

static Sc g_sc;
static Frame g_frames[FRAMES];
static uint64_t g_last_value;              // the present thread's newest submission
static uint64_t g_frame_no;

static PlatMutex g_shown_mu = PLAT_MUTEX_INIT;
static Shown g_shown[SHOWN_RING];
static uint64_t g_last_seq;
static int64_t g_cal_offset_ns;            // host time minus the swapchain domain's, from the last calibration
static int g_cal_ok;
static uint64_t g_since_cal;

static VkwStats g_stats;                   // written under g_mu or by the present thread alone; read through vkw_stats

// ---------- the hooks of gpu_vk_open
void vkw_window(const WinVulkan *win, const VkwUi *ui) {
    g_win = win;
    g_ui = ui;
}

int vkw_wanted(void) { return g_win != NULL; }
int vkw_active(void) { return __atomic_load_n(&g_active, __ATOMIC_ACQUIRE); }

uint32_t vkw_instance_extensions(const char **names, uint32_t max) {
    uint32_t n = 0, count = 0;
    const char *const *e = g_win ? g_win->instance_extensions(&count) : NULL;
    for (uint32_t i = 0; e && i < count && n < max; i++) names[n++] = e[i];
    uint32_t have = 0;
    vkEnumerateInstanceExtensionProperties(NULL, &have, NULL);
    VkExtensionProperties *p = calloc(have ? have : 1, sizeof *p);
    if (p && vkEnumerateInstanceExtensionProperties(NULL, &have, p) >= 0)
        for (uint32_t i = 0; i < have && n < max; i++)
            if (!strcmp(p[i].extensionName, VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME)) names[n++] = VK_KHR_GET_SURFACE_CAPABILITIES_2_EXTENSION_NAME;
    free(p);
    return n;
}

static int lists(const VkExtensionProperties *e, uint32_t n, const char *name) {
    for (uint32_t i = 0; i < n; i++)
        if (!strcmp(e[i].extensionName, name)) return 1;
    return 0;
}

static VkExtensionProperties *device_extensions(VkPhysicalDevice dev, uint32_t *n) {
    *n = 0;
    GPU_VK_OK(vkEnumerateDeviceExtensionProperties(dev, NULL, n, NULL), "vkEnumerateDeviceExtensionProperties (present)");
    VkExtensionProperties *e = calloc(*n ? *n : 1, sizeof *e);
    if (!e) rt_fatal("gfx: vulkan: out of memory for %u extensions", *n);
    GPU_VK_OK(vkEnumerateDeviceExtensionProperties(dev, NULL, n, e), "vkEnumerateDeviceExtensionProperties (present)");
    return e;
}

int vkw_can_present(VkPhysicalDevice phys, uint32_t family) {
    if (!g_win) return 1;
    uint32_t n;
    VkExtensionProperties *e = device_extensions(phys, &n);
    int ok = lists(e, n, VK_KHR_SWAPCHAIN_EXTENSION_NAME);
    free(e);
    return ok && g_win->can_present(gpu_vk.instance, phys, family);
}

static VkPhysicalDevicePresentId2FeaturesKHR g_id2_en = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR};
static VkPhysicalDevicePresentTimingFeaturesEXT g_pt_en = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT};

void vkw_device_extensions(VkPhysicalDevice phys, const char **names, uint32_t *count, VkPhysicalDeviceFeatures2 *features) {
    names[(*count)++] = VK_KHR_SWAPCHAIN_EXTENSION_NAME;
    uint32_t n;
    VkExtensionProperties *e = device_extensions(phys, &n);
    int ext = lists(e, n, "VK_KHR_present_id2") && lists(e, n, "VK_EXT_present_timing") &&
              lists(e, n, VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME);
    free(e);
    g_timing_dev = 0;
    if (!ext) return;
    VkPhysicalDevicePresentId2FeaturesKHR id2 = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_ID_2_FEATURES_KHR};
    VkPhysicalDevicePresentTimingFeaturesEXT pt = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PRESENT_TIMING_FEATURES_EXT,
                                                   .pNext = &id2};
    VkPhysicalDeviceFeatures2 q = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2, .pNext = &pt};
    vkGetPhysicalDeviceFeatures2(phys, &q);
    if (!id2.presentId2 || !pt.presentTiming) return;
    g_id2_en.presentId2 = VK_TRUE;
    g_pt_en.presentTiming = VK_TRUE;
    g_id2_en.pNext = NULL;
    g_pt_en.pNext = &g_id2_en;
    VkBaseOutStructure *tail = (VkBaseOutStructure *)features;   // the chain's last structure takes ours
    while (tail->pNext) tail = tail->pNext;
    tail->pNext = (VkBaseOutStructure *)&g_pt_en;
    names[(*count)++] = "VK_KHR_present_id2";
    names[(*count)++] = "VK_EXT_present_timing";
    names[(*count)++] = VK_KHR_CALIBRATED_TIMESTAMPS_EXTENSION_NAME;
    g_timing_dev = 1;
}

// ---------- time
#ifdef _WIN32
static uint64_t host_ns_of_counter(uint64_t counts) {   // the same arithmetic as plat_now_ns
    static LARGE_INTEGER freq;
    if (!freq.QuadPart) QueryPerformanceFrequency(&freq);
    uint64_t f = (uint64_t)freq.QuadPart;
    return counts / f * 1000000000ull + counts % f * 1000000000ull / f;
}
#define HOST_DOMAIN VK_TIME_DOMAIN_QUERY_PERFORMANCE_COUNTER_KHR
#else
static uint64_t host_ns_of_counter(uint64_t counts) { return counts; }
#define HOST_DOMAIN VK_TIME_DOMAIN_CLOCK_MONOTONIC_KHR
#endif

// The tick of the 60 Hz clock that host time ns falls in; never 0 (tick 0 is the origin, before any present).
static uint64_t tick_of_ns(uint64_t ns) {
    uint64_t tick = rt_sched_slot_of(rt_vsync_counter_at_ns(ns), RT_VSYNC_HZ);
    return tick ? tick : 1;
}

static void record_returned(uint64_t seq, uint64_t ns) {
    plat_mutex_lock(&g_shown_mu);
    g_shown[seq % SHOWN_RING] = (Shown){seq, ns, 0};
    g_last_seq = seq;
    plat_mutex_unlock(&g_shown_mu);
}

static void record_timing(uint64_t seq, uint64_t ns) {
    plat_mutex_lock(&g_shown_mu);
    Shown *s = &g_shown[seq % SHOWN_RING];
    if (s->seq == seq) s->timing_ns = ns;
    plat_mutex_unlock(&g_shown_mu);
}

uint64_t vkw_shown_ns(uint64_t seq) {
    uint64_t ns = 0;
    plat_mutex_lock(&g_shown_mu);
    const Shown *s = &g_shown[seq % SHOWN_RING];
    if (seq && s->seq == seq) ns = s->timing_ns;
    plat_mutex_unlock(&g_shown_mu);
    return ns;
}

uint64_t vkw_shown(uint64_t seq) {
    if (vkw_no_feedback || !seq) return 0;
    uint64_t ns = 0;
    plat_mutex_lock(&g_shown_mu);
    const Shown *s = &g_shown[seq % SHOWN_RING];
    if (s->seq == seq) ns = s->timing_ns ? s->timing_ns : (__atomic_load_n(&g_stats.timing, __ATOMIC_RELAXED) ? 0 : s->returned_ns);
    plat_mutex_unlock(&g_shown_mu);
    return ns ? tick_of_ns(ns) : 0;
}

uint64_t vkw_last_seq(void) {
    plat_mutex_lock(&g_shown_mu);
    uint64_t s = g_last_seq;
    plat_mutex_unlock(&g_shown_mu);
    return s;
}

void vkw_stats(VkwStats *out) {
    plat_mutex_lock(&g_mu);
    *out = g_stats;
    plat_mutex_unlock(&g_mu);
}

void vkw_pace(uint64_t seq, uint64_t due_tick, uint64_t due_count) {
    (void)due_tick;
    (void)due_count;
    t_pace_seq = seq;
}

// ---------- the snapshot ring (the guest thread)
static uint32_t memory_type(uint32_t bits, VkMemoryPropertyFlags want) {
    VkPhysicalDeviceMemoryProperties mp;
    vkGetPhysicalDeviceMemoryProperties(gpu_vk.phys, &mp);
    for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
        if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & want) == want) return i;
    return UINT32_MAX;
}

typedef struct { VkImage image; VkDeviceMemory mem; } OldImage;
static void old_image_gone(void *arg, uint64_t value) {
    (void)value;
    OldImage *o = arg;
    vkDestroyImage(gpu_vk.device, o->image, NULL);
    vkFreeMemory(gpu_vk.device, o->mem, NULL);
    free(o);
}

static void slot_image_drop(Slot *s) {
    if (!s->image) return;
    gpu_sync_wait(s->last_use, UINT64_MAX);   // the submissions that read or wrote it are done
    vkDestroyImage(gpu_vk.device, s->image, NULL);
    vkFreeMemory(gpu_vk.device, s->mem, NULL);
    s->image = VK_NULL_HANDLE, s->mem = VK_NULL_HANDLE, s->w = s->h = 0;
}

// An image of w x h in the slot (the slot is the caller's: FILLING; a submission is open); 1 when it is new and in the
// undefined layout. The image it replaces goes once the open submission is done, which is after every use of it.
static int slot_image(Slot *s, uint32_t w, uint32_t h) {
    if (s->image && s->w == w && s->h == h) return 0;
    if (s->image) {
        OldImage *o = malloc(sizeof *o);
        if (!o) rt_fatal("gfx: vulkan: out of memory for an old present snapshot");
        *o = (OldImage){s->image, s->mem};
        gpu_vk_effect(old_image_gone, o);
        s->image = VK_NULL_HANDLE, s->mem = VK_NULL_HANDLE, s->w = s->h = 0;
    }
    VkImageCreateInfo ci = {.sType = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO, .imageType = VK_IMAGE_TYPE_2D,
                            .format = VK_FORMAT_R8G8B8A8_UNORM, .extent = {w, h, 1}, .mipLevels = 1, .arrayLayers = 1,
                            .samples = VK_SAMPLE_COUNT_1_BIT, .tiling = VK_IMAGE_TILING_OPTIMAL,
                            .usage = VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                            .sharingMode = VK_SHARING_MODE_EXCLUSIVE, .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED};
    GPU_VK_OK(vkCreateImage(gpu_vk.device, &ci, NULL, &s->image), "vkCreateImage (a present snapshot)");
    VkMemoryRequirements mr;
    vkGetImageMemoryRequirements(gpu_vk.device, s->image, &mr);
    uint32_t type = memory_type(mr.memoryTypeBits, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    if (type == UINT32_MAX) type = memory_type(mr.memoryTypeBits, 0);
    VkMemoryAllocateInfo alloc = {.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = mr.size,
                                  .memoryTypeIndex = type};
    GPU_VK_OK(vkAllocateMemory(gpu_vk.device, &alloc, NULL, &s->mem), "vkAllocateMemory (a present snapshot)");
    GPU_VK_OK(vkBindImageMemory(gpu_vk.device, s->image, s->mem, 0), "vkBindImageMemory (a present snapshot)");
    s->w = w, s->h = h;
    return 1;
}

static int int_format(VkFormat f) {
    switch (f) {
    case VK_FORMAT_R8_UINT: case VK_FORMAT_R8_SINT: case VK_FORMAT_R8G8_UINT: case VK_FORMAT_R8G8_SINT:
    case VK_FORMAT_R8G8B8A8_UINT: case VK_FORMAT_R8G8B8A8_SINT: case VK_FORMAT_B8G8R8A8_UINT: case VK_FORMAT_B8G8R8A8_SINT:
    case VK_FORMAT_A2B10G10R10_UINT_PACK32: case VK_FORMAT_A2R10G10B10_UINT_PACK32: case VK_FORMAT_R16_UINT:
    case VK_FORMAT_R16_SINT: case VK_FORMAT_R16G16_UINT: case VK_FORMAT_R16G16_SINT: case VK_FORMAT_R16G16B16A16_UINT:
    case VK_FORMAT_R16G16B16A16_SINT: case VK_FORMAT_R32_UINT: case VK_FORMAT_R32_SINT: case VK_FORMAT_R32G32_UINT:
    case VK_FORMAT_R32G32_SINT: case VK_FORMAT_R32G32B32_UINT: case VK_FORMAT_R32G32B32_SINT:
    case VK_FORMAT_R32G32B32A32_UINT: case VK_FORMAT_R32G32B32A32_SINT:
        return 1;
    default: return 0;
    }
}

static void barrier(VkCommandBuffer cmd, VkPipelineStageFlags2 src, VkAccessFlags2 src_access, VkPipelineStageFlags2 dst,
                    VkAccessFlags2 dst_access, VkImage fresh_image) {
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = src, .srcAccessMask = src_access,
                           .dstStageMask = dst, .dstAccessMask = dst_access};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = src,
                                .dstStageMask = dst, .dstAccessMask = dst_access,
                                .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED, .newLayout = VK_IMAGE_LAYOUT_GENERAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = fresh_image, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = fresh_image ? 1u : 0u, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cmd, &di);
}

// A slot for the guest thread: a free one, else the oldest picture the present thread has not taken (dropped). g_mu held.
static int pick_slot(void) {
    for (int i = 0; i < SLOTS; i++)
        if (g_slots[i].state == SLOT_FREE) return i;
    int oldest = -1;
    for (int i = 0; i < SLOTS; i++)
        if (g_slots[i].state == SLOT_FILLED && (oldest < 0 || g_slots[i].seq < g_slots[oldest].seq)) oldest = i;
    if (oldest >= 0) g_stats.dropped_ring++;
    return oldest;
}

void vkw_present(const tex_t *t, const int crop[4]) {
    uint64_t seq = t_pace_seq;
    t_pace_seq = 0;
    if (!seq) seq = __atomic_add_fetch(&g_seq_fallback, 1, __ATOMIC_RELAXED);   // a present with no pace call: own numbers (never mixed with a pace's)
    VkbImage *img = vkb_image_get(t);
    if (!img)
        rt_fatal("gfx: vulkan: present of the texture at %p (NVN format %#x, %ux%u): no image stands on its storage "
                 "(storage the CPU writes is the texture arena's)", (const void *)t->mem, t->d.format, t->d.width, t->d.height);
    VkbShape sh;
    vkb_image_shape(img, &sh);
    VkFormatProperties fp;
    vkGetPhysicalDeviceFormatProperties(gpu_vk.phys, sh.format, &fp);
    if (sh.all_aspects != VK_IMAGE_ASPECT_COLOR_BIT || int_format(sh.format) ||
        !(fp.optimalTilingFeatures & VK_FORMAT_FEATURE_BLIT_SRC_BIT))
        rt_fatal("gfx: vulkan: present of the texture at %p (NVN format %#x, Vulkan format %d): the window cannot blit "
                 "from it", (const void *)t->mem, t->d.format, (int)sh.format);
    int64_t x0 = crop[0], y0 = crop[1], x1 = x0 + crop[2], y1 = y0 + crop[3];
    if (crop[2] <= 0 || crop[3] <= 0) x0 = 0, y0 = 0, x1 = sh.width, y1 = sh.height;
    if (x0 < 0) x0 = 0;
    if (y0 < 0) y0 = 0;
    if (x1 > (int64_t)sh.width) x1 = sh.width;
    if (y1 > (int64_t)sh.height) y1 = sh.height;
    if (x1 <= x0 || y1 <= y0) x0 = 0, y0 = 0, x1 = sh.width, y1 = sh.height;
    uint32_t w = (uint32_t)(x1 - x0), h = (uint32_t)(y1 - y0);

    plat_mutex_lock(&g_mu);
    int si = pick_slot();
    if (si < 0) {
        plat_mutex_unlock(&g_mu);
        rt_fatal("gfx: vulkan: a present found no snapshot slot (more than one window presenting?)");
    }
    Slot *s = &g_slots[si];
    s->state = SLOT_FILLING;
    g_stats.presents++;
    plat_mutex_unlock(&g_mu);

    uint64_t value = gpu_vk_begin(GPU_VK_INTERNAL);
    int fresh = slot_image(s, w, h);
    VkCommandBuffer cmd = gpu_vk_cmd();
    vkb_image_flush(cmd);
    // after every submission before this one: the first scope of a barrier holds the earlier commands of the queue
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT, VK_ACCESS_2_MEMORY_WRITE_BIT | VK_ACCESS_2_MEMORY_READ_BIT,
            VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT,
            fresh ? s->image : VK_NULL_HANDLE);
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffsets = {{(int32_t)x0, (int32_t)y0, 0}, {(int32_t)x1, (int32_t)y1, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .dstOffsets = {{0, 0, 0}, {(int32_t)w, (int32_t)h, 1}}};
    vkCmdBlitImage(cmd, vkb_image_handle(img), VK_IMAGE_LAYOUT_GENERAL, s->image, VK_IMAGE_LAYOUT_GENERAL, 1, &b,
                   VK_FILTER_NEAREST);
    barrier(cmd, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
            VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT, VK_NULL_HANDLE);
    vkb_image_use(img, value);
    gpu_vk_end(NULL, 0);

    plat_mutex_lock(&g_mu);
    s->seq = seq, s->value = value, s->last_use = value;
    s->state = SLOT_FILLED;
    plat_cond_broadcast(&g_cv);
    plat_mutex_unlock(&g_mu);
}

// ---------- the swapchain (the present thread)
static VkExtent2D window_extent(VkSurfaceCapabilitiesKHR *caps) {
    VkResult r = vkGetPhysicalDeviceSurfaceCapabilitiesKHR(gpu_vk.phys, g_surface, caps);
    if (r == VK_ERROR_SURFACE_LOST_KHR) return (VkExtent2D){0, 0};
    if (r != VK_SUCCESS) gpu_vk_failed(r, "vkGetPhysicalDeviceSurfaceCapabilitiesKHR");
    VkExtent2D e = caps->currentExtent;
    if (e.width == UINT32_MAX) {   // the surface leaves it to the swapchain: the window's own size
        WinState st;
        g_win->state(&st);
        e.width = (uint32_t)st.width, e.height = (uint32_t)st.height;
    }
    return e;
}

static void sc_destroy(Sc *sc) {
    for (uint32_t i = 0; i < sc->n; i++) {
        vkDestroyImageView(gpu_vk.device, sc->views[i], NULL);
        vkDestroySemaphore(gpu_vk.device, sc->done[i], NULL);
    }
    if (sc->handle) vkDestroySwapchainKHR(gpu_vk.device, sc->handle, NULL);
    memset(sc, 0, sizeof *sc);
}

static int surface_timing(void) {
    if (!g_timing_dev) return 0;
    VkPhysicalDeviceSurfaceInfo2KHR si = {.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_SURFACE_INFO_2_KHR, .surface = g_surface};
    VkPresentTimingSurfaceCapabilitiesEXT ptc = {.sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_SURFACE_CAPABILITIES_EXT};
    VkSurfaceCapabilitiesPresentId2KHR pid = {.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_PRESENT_ID_2_KHR,
                                              .pNext = &ptc};
    VkSurfaceCapabilities2KHR caps = {.sType = VK_STRUCTURE_TYPE_SURFACE_CAPABILITIES_2_KHR, .pNext = &pid};
    if (!vkGetPhysicalDeviceSurfaceCapabilities2KHR) return 0;
    if (vkGetPhysicalDeviceSurfaceCapabilities2KHR(gpu_vk.phys, &si, &caps) != VK_SUCCESS) return 0;
    return pid.presentId2Supported && ptc.presentTimingSupported && (ptc.presentStageQueries & VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT);
}

static void choose_domain(Sc *sc) {
    sc->timing = 0;
    if (vkSetSwapchainPresentTimingQueueSizeEXT(gpu_vk.device, sc->handle, TIMING_QUEUE) != VK_SUCCESS) return;
    uint64_t counter = 0;
    VkSwapchainTimeDomainPropertiesEXT dp = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_TIME_DOMAIN_PROPERTIES_EXT};
    if (vkGetSwapchainTimeDomainPropertiesEXT(gpu_vk.device, sc->handle, &dp, &counter) != VK_SUCCESS) return;
    VkTimeDomainKHR kinds[8];
    uint64_t ids[8];
    if (dp.timeDomainCount > 8) dp.timeDomainCount = 8;
    dp.pTimeDomains = kinds;
    dp.pTimeDomainIds = ids;
    VkResult r = vkGetSwapchainTimeDomainPropertiesEXT(gpu_vk.device, sc->handle, &dp, &counter);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) return;
    int best = -1, rank = 99;
    for (uint32_t i = 0; i < dp.timeDomainCount; i++) {   // the host's own counter needs no calibration; then swapchain-local
        int k = kinds[i] == HOST_DOMAIN ? 0 : kinds[i] == VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT ? 1
              : kinds[i] == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT ? 2 : 3;
        if (k < rank) rank = k, best = (int)i;
    }
    if (best < 0 || rank > 2) return;
    sc->domain_id = ids[best], sc->domain_kind = kinds[best];
    sc->timing = 1;
    g_cal_ok = 0;
}

// A swapchain made again is not destroyed on the spot: its images may still be on the display's way, and a present of
// them may still wait on its semaphores. It is retired, and goes once a submission made after it was retired is done (the
// queue runs in order, so everything before it is) and a few pictures have gone by since, with no wait and no hold of the
// submission gate, so a resize costs the game nothing. vkw_close destroys what is left after the device is idle.
#define RETIRED 32
#define RETIRE_FRAMES 4
static struct { Sc sc; uint64_t value, frame; } g_retired[RETIRED];
static int g_nretired;

static void reap_retired(int all) {
    int kept = 0;
    for (int i = 0; i < g_nretired; i++) {
        if (all || (gpu_sync_done() > g_retired[i].value && g_frame_no >= g_retired[i].frame + RETIRE_FRAMES))
            sc_destroy(&g_retired[i].sc);
        else
            g_retired[kept++] = g_retired[i];
    }
    g_nretired = kept;
}

static void retire(const Sc *old) {
    if (!old->handle) return;
    if (g_nretired == RETIRED) {   // more than a resize makes in the time it takes to retire one: the oldest goes now
        gpu_sync_wait(g_retired[0].value + 1, UINT64_MAX);
        sc_destroy(&g_retired[0].sc);
        memmove(&g_retired[0], &g_retired[1], (RETIRED - 1) * sizeof g_retired[0]);
        g_nretired--;
    }
    g_retired[g_nretired].sc = *old;
    g_retired[g_nretired].value = gpu_sync_handed();
    g_retired[g_nretired].frame = g_frame_no;
    g_nretired++;
}

// Make (or make again) the swapchain for the window's extent; 1 when there is one to present to, 0 while the window has
// no area. The old one is retired (retire above).
static int ensure_swapchain(void) {
    VkSurfaceCapabilitiesKHR caps;
    VkExtent2D e = window_extent(&caps);
    if (!e.width || !e.height) return 0;
    if (g_sc.handle && vkw_no_recreate) return 1;
    if (g_sc.handle && !g_sc.dirty && e.width == g_sc.extent.width && e.height == g_sc.extent.height) return 1;
    const VkImageUsageFlags need = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    if ((caps.supportedUsageFlags & need) != need)
        rt_fatal("gfx: vulkan: the window's surface does not allow images to be drawn into and blitted to (usage %#x)",
                 (unsigned)caps.supportedUsageFlags);
    uint32_t want = 3 < caps.minImageCount ? caps.minImageCount : 3;
    if (caps.maxImageCount && want > caps.maxImageCount) want = caps.maxImageCount;
    int id2 = surface_timing();
    VkSwapchainCreateInfoKHR ci = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR,
                                   .flags = id2 ? (VK_SWAPCHAIN_CREATE_PRESENT_ID_2_BIT_KHR | VK_SWAPCHAIN_CREATE_PRESENT_TIMING_BIT_EXT) : 0,
                                   .surface = g_surface, .minImageCount = want, .imageFormat = g_surface_format,
                                   .imageColorSpace = g_surface_space, .imageExtent = e, .imageArrayLayers = 1,
                                   .imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT,
                                   .imageSharingMode = VK_SHARING_MODE_EXCLUSIVE, .preTransform = caps.currentTransform,
                                   .compositeAlpha = (caps.supportedCompositeAlpha & VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR)
                                                         ? VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR : VK_COMPOSITE_ALPHA_INHERIT_BIT_KHR,
                                   .presentMode = VK_PRESENT_MODE_FIFO_KHR, .clipped = VK_TRUE,
                                   .oldSwapchain = g_sc.handle};
    Sc fresh = {0};
    VkResult r = vkCreateSwapchainKHR(gpu_vk.device, &ci, NULL, &fresh.handle);
    if (r == VK_ERROR_SURFACE_LOST_KHR) return 0;
    if (r != VK_SUCCESS) gpu_vk_failed(r, "vkCreateSwapchainKHR");
    Sc old = g_sc;
    fresh.extent = e, fresh.format = g_surface_format, fresh.id2 = id2;
    fresh.n = MAX_IMAGES;
    GPU_VK_OK(vkGetSwapchainImagesKHR(gpu_vk.device, fresh.handle, &fresh.n, fresh.images), "vkGetSwapchainImagesKHR");
    for (uint32_t i = 0; i < fresh.n; i++) {
        VkImageViewCreateInfo vi = {.sType = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO, .image = fresh.images[i],
                                    .viewType = VK_IMAGE_VIEW_TYPE_2D, .format = fresh.format,
                                    .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
        GPU_VK_OK(vkCreateImageView(gpu_vk.device, &vi, NULL, &fresh.views[i]), "vkCreateImageView (the swapchain)");
        VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        GPU_VK_OK(vkCreateSemaphore(gpu_vk.device, &sci, NULL, &fresh.done[i]), "vkCreateSemaphore (the swapchain)");
    }
    if (id2) choose_domain(&fresh);
    g_sc = fresh;
    retire(&old);
    plat_mutex_lock(&g_mu);
    g_stats.swapchains++;
    g_stats.width = e.width, g_stats.height = e.height, g_stats.images = fresh.n;
    g_stats.timing = g_sc.timing;
    g_stats.format = (int)g_sc.format;
    plat_mutex_unlock(&g_mu);
    rt_log("gfx: vulkan: swapchain %ux%u, %u images, format %d, FIFO, shown time from %s", e.width, e.height, fresh.n,
           (int)fresh.format, g_sc.timing ? "present timing (first pixel out)" : "the return of the present call");
    return 1;
}

// ---------- present timing
static int calibrate(void) {
    VkSwapchainCalibratedTimestampInfoEXT sc = {.sType = VK_STRUCTURE_TYPE_SWAPCHAIN_CALIBRATED_TIMESTAMP_INFO_EXT,
                                                .swapchain = g_sc.handle,
                                                .presentStage = g_sc.domain_kind == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT
                                                                    ? VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT : 0,
                                                .timeDomainId = g_sc.domain_id};
    VkCalibratedTimestampInfoKHR info[2] = {{.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR,
                                             .pNext = (g_sc.domain_kind == VK_TIME_DOMAIN_PRESENT_STAGE_LOCAL_EXT ||
                                                       g_sc.domain_kind == VK_TIME_DOMAIN_SWAPCHAIN_LOCAL_EXT) ? &sc : NULL,
                                             .timeDomain = g_sc.domain_kind},
                                            {.sType = VK_STRUCTURE_TYPE_CALIBRATED_TIMESTAMP_INFO_KHR, .timeDomain = HOST_DOMAIN}};
    uint64_t ts[2] = {0, 0}, dev = 0;
    if (vkGetCalibratedTimestampsKHR(gpu_vk.device, 2, info, ts, &dev) != VK_SUCCESS) return 0;
    g_cal_offset_ns = (int64_t)host_ns_of_counter(ts[1]) - (int64_t)ts[0];
    g_cal_ok = 1;
    g_since_cal = 0;
    return 1;
}

// The host time of a first-pixel-out value in the swapchain's time domain.
static uint64_t host_ns_of(uint64_t value) {
    if (g_sc.domain_kind == HOST_DOMAIN) return host_ns_of_counter(value);
    if (!g_cal_ok && !calibrate()) return 0;
    return (uint64_t)((int64_t)value + g_cal_offset_ns);
}

static void poll_timing(void) {
    if (!g_sc.handle || !g_sc.timing) return;
    static VkPastPresentationTimingEXT arr[TIMING_BATCH];
    static VkPresentStageTimeEXT stages[TIMING_BATCH][4];
    for (;;) {
        for (int i = 0; i < TIMING_BATCH; i++) {
            memset(&arr[i], 0, sizeof arr[i]);
            arr[i].sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_EXT;
            arr[i].presentStageCount = 4;
            arr[i].pPresentStages = stages[i];
        }
        VkPastPresentationTimingInfoEXT info = {.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_INFO_EXT,
                                                .swapchain = g_sc.handle};
        VkPastPresentationTimingPropertiesEXT props = {.sType = VK_STRUCTURE_TYPE_PAST_PRESENTATION_TIMING_PROPERTIES_EXT,
                                                       .presentationTimingCount = TIMING_BATCH,
                                                       .pPresentationTimings = arr};
        VkResult r = vkGetPastPresentationTimingEXT(gpu_vk.device, &info, &props);
        if (r != VK_SUCCESS && r != VK_INCOMPLETE) return;
        for (uint32_t i = 0; i < props.presentationTimingCount; i++) {
            const VkPastPresentationTimingEXT *t = &arr[i];
            if (!t->presentId || !t->reportComplete) continue;
            if (t->timeDomain != g_sc.domain_kind || t->timeDomainId != g_sc.domain_id) continue;
            for (uint32_t s = 0; s < t->presentStageCount && s < 4; s++) {
                if (t->pPresentStages[s].stage != VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT) continue;
                if (!t->pPresentStages[s].time) continue;   // 0: the picture was never shown
                if (++g_since_cal >= RECALIBRATE_EVERY) g_cal_ok = 0;
                uint64_t ns = host_ns_of(t->pPresentStages[s].time);
                if (ns) record_timing(t->presentId, ns);
                plat_mutex_lock(&g_mu);
                g_stats.timing_results++;
                plat_mutex_unlock(&g_mu);
            }
        }
        if (r == VK_SUCCESS) return;
    }
}

// ---------- one picture
// The largest rectangle of the snapshot's aspect that fits the extent, centred.
VkwRect vkw_fit(uint32_t sw, uint32_t sh, uint32_t ew, uint32_t eh) {
    VkwRect r = {0, 0, (int)ew, (int)eh};
    if (!sw || !sh) return r;
    if ((uint64_t)ew * sh >= (uint64_t)eh * sw) {   // the extent is wider than the picture: bars at the sides
        r.h = (int)eh;
        r.w = (int)(((uint64_t)eh * sw + sh / 2) / sh);
    } else {
        r.w = (int)ew;
        r.h = (int)(((uint64_t)ew * sh + sw / 2) / sw);
    }
    if (r.w < 1) r.w = 1;
    if (r.h < 1) r.h = 1;
    if (r.w > (int)ew) r.w = (int)ew;
    if (r.h > (int)eh) r.h = (int)eh;
    r.x = ((int)ew - r.w) / 2;
    r.y = ((int)eh - r.h) / 2;
    return r;
}

static void layout_barrier(VkCommandBuffer cmd, VkImage image, VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                           VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access, VkImageLayout from, VkImageLayout to) {
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2, .srcStageMask = src_stage,
                                .srcAccessMask = src_access, .dstStageMask = dst_stage, .dstAccessMask = dst_access,
                                .oldLayout = from, .newLayout = to, .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .image = image,
                                .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cmd, &di);
}

static uint64_t g_last_frame_ns, g_last_dt_ns = 16666667ull;

// The driver says the swapchain still presents but no longer matches the surface. A size change is seen on its own at the
// next picture (ensure_swapchain reads the extent); other reasons (a transform, an adapter) make it again, but no oftener
// than twice a second, so a driver that keeps saying it costs a swapchain per half second and not one per picture.
static void note_suboptimal(void) {
    static uint64_t at;
    plat_mutex_lock(&g_mu);
    g_stats.suboptimal++;
    plat_mutex_unlock(&g_mu);
    uint64_t now = plat_now_ns();
    if (now - at > 500000000ull) g_sc.dirty = 1, at = now;
}

// Draw slot s into a swapchain image and present it with game present number seq (0: a picture drawn again, no id, no
// timing). 1 when it was presented, 0 when the window or the swapchain would not take it, 2 when it was a picture drawn
// again (game_frame 0) with nothing to put on it and no reason (force 0) to draw it.
static int present_slot(Slot *s, uint64_t seq, int game_frame, int force) {
    if (!ensure_swapchain()) {
        plat_mutex_lock(&g_mu);
        g_stats.dropped_window++;
        plat_mutex_unlock(&g_mu);
        return 0;
    }
    Frame *fr = &g_frames[g_frame_no % FRAMES];
    int fi = (int)(g_frame_no % FRAMES);
    if (fr->value) gpu_sync_wait(fr->value, UINT64_MAX);   // the frame's buffers and its acquire semaphore are free
    // the menu's lists first: they are the CPU work, and the acquire is the wait
    UiDrawData dd = {0};
    int overlay = 0;
    if (g_ui_on && g_ui->frame) {
        uint64_t now = plat_now_ns();
        uint64_t dt_ns = game_frame ? (g_last_frame_ns ? now - g_last_frame_ns : g_last_dt_ns) : g_last_dt_ns;
        if (game_frame) g_last_frame_ns = now, g_last_dt_ns = dt_ns;
        UiFrame uf = {.width = (int)g_sc.extent.width, .height = (int)g_sc.extent.height, .dt = (double)dt_ns / 1e9};
        UiDraw built;
        if (g_ui->frame(&uf, &built) && g_ui->draw_data) {
            g_ui->draw_data(&dd);
            overlay = dd.cmd_count > 0;
        }
    } else if (game_frame) {
        uint64_t now = plat_now_ns();
        if (g_last_frame_ns) g_last_dt_ns = now - g_last_frame_ns;
        g_last_frame_ns = now;
    }
    if (!game_frame && !force && !overlay) return 2;
    if (overlay) vko_upload(fi, &dd);

    uint32_t idx = 0;
    for (int stale = 0;;) {
        VkResult r = vkAcquireNextImageKHR(gpu_vk.device, g_sc.handle, ACQUIRE_NS, fr->acquire, VK_NULL_HANDLE, &idx);
        if (r == VK_SUCCESS || r == VK_SUBOPTIMAL_KHR) {
            if (r == VK_SUBOPTIMAL_KHR) note_suboptimal();
            break;
        }
        if (r == VK_TIMEOUT || r == VK_NOT_READY) {
            plat_mutex_lock(&g_mu);
            int stop = g_stop;
            plat_mutex_unlock(&g_mu);
            if (stop) return 0;
            continue;
        }
        if (r == VK_ERROR_OUT_OF_DATE_KHR || r == VK_ERROR_SURFACE_LOST_KHR) {
            plat_mutex_lock(&g_mu);
            g_stats.out_of_date++;
            plat_mutex_unlock(&g_mu);
            g_sc.dirty = 1;
            if (stale++ >= 1 || vkw_no_recreate || !ensure_swapchain()) {
                plat_mutex_lock(&g_mu);
                g_stats.dropped_failed++;
                plat_mutex_unlock(&g_mu);
                return 0;
            }
            fr = &g_frames[g_frame_no % FRAMES];
            continue;
        }
        gpu_vk_failed(r, "vkAcquireNextImageKHR");
    }

    VkwRect rc = vkw_fit(s->w, s->h, g_sc.extent.width, g_sc.extent.height);
    VkImage dst = g_sc.images[idx];
    uint64_t value = gpu_vk_begin(GPU_VK_INTERNAL);
    VkCommandBuffer cmd = gpu_vk_cmd();
    // the image arrives from the presentation engine (the acquire semaphore) with nothing worth keeping; the snapshot's
    // writes, and every earlier submission's, are made visible to the blit by the first scope of this barrier
    VkMemoryBarrier2 mb = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
                           .srcAccessMask = VK_ACCESS_2_MEMORY_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                           .dstAccessMask = VK_ACCESS_2_TRANSFER_READ_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkImageMemoryBarrier2 ib = {.sType = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
                                .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, .srcAccessMask = 0,
                                .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .oldLayout = VK_IMAGE_LAYOUT_UNDEFINED,
                                .newLayout = VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL,
                                .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED, .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
                                .image = dst, .subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}};
    VkDependencyInfo di = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &mb,
                           .imageMemoryBarrierCount = 1, .pImageMemoryBarriers = &ib};
    vkCmdPipelineBarrier2(cmd, &di);
    VkClearColorValue black = {{0.0f, 0.0f, 0.0f, 1.0f}};
    VkImageSubresourceRange range = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    vkCmdClearColorImage(cmd, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &range);
    VkMemoryBarrier2 waw = {.sType = VK_STRUCTURE_TYPE_MEMORY_BARRIER_2, .srcStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                            .srcAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT, .dstStageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                            .dstAccessMask = VK_ACCESS_2_TRANSFER_WRITE_BIT};
    VkDependencyInfo wd = {.sType = VK_STRUCTURE_TYPE_DEPENDENCY_INFO, .memoryBarrierCount = 1, .pMemoryBarriers = &waw};
    vkCmdPipelineBarrier2(cmd, &wd);
    VkImageBlit b = {.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .srcOffsets = {{0, 0, 0}, {(int32_t)s->w, (int32_t)s->h, 1}},
                     .dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1},
                     .dstOffsets = {{rc.x, rc.y, 0}, {rc.x + rc.w, rc.y + rc.h, 1}}};
    vkCmdBlitImage(cmd, s->image, VK_IMAGE_LAYOUT_GENERAL, dst, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &b, VK_FILTER_LINEAR);
    if (overlay) {
        layout_barrier(cmd, dst, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL);
        vko_record(cmd, value, fi, g_sc.views[idx], g_sc.extent, &dd);
        layout_barrier(cmd, dst, VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_NONE, 0, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    } else {
        layout_barrier(cmd, dst, VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT, VK_ACCESS_2_TRANSFER_WRITE_BIT,
                       VK_PIPELINE_STAGE_2_NONE, 0, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR);
    }
    VkSemaphoreSubmitInfo wait = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = fr->acquire,
                                  .stageMask = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT};
    VkSemaphoreSubmitInfo signal = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO, .semaphore = g_sc.done[idx],
                                    .stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT};
    VkPresentId2KHR pid = {.sType = VK_STRUCTURE_TYPE_PRESENT_ID_2_KHR, .swapchainCount = 1, .pPresentIds = &seq};
    VkPresentTimingInfoEXT ti = {.sType = VK_STRUCTURE_TYPE_PRESENT_TIMING_INFO_EXT,
                                 .presentStageQueries = VK_PRESENT_STAGE_IMAGE_FIRST_PIXEL_OUT_BIT_EXT,
                                 .timeDomainId = g_sc.domain_id};
    VkPresentTimingsInfoEXT tis = {.sType = VK_STRUCTURE_TYPE_PRESENT_TIMINGS_INFO_EXT, .swapchainCount = 1, .pTimingInfos = &ti};
    VkPresentInfoKHR pi = {.sType = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR, .waitSemaphoreCount = 1,
                           .pWaitSemaphores = &g_sc.done[idx], .swapchainCount = 1, .pSwapchains = &g_sc.handle,
                           .pImageIndices = &idx};
    if (seq && g_sc.id2) {
        pid.pNext = g_sc.timing ? (void *)&tis : NULL;
        pi.pNext = &pid;
    }
    VkResult pr = VK_SUCCESS;
    uint64_t end_began = plat_now_ns();
    gpu_vk_end_present(&wait, 1, &signal, &pi, &pr);
    uint64_t returned = plat_now_ns();
    plat_mutex_lock(&g_mu);
    if (returned - end_began > g_stats.end_ns_max) g_stats.end_ns_max = returned - end_began;
    g_stats.end_ns_sum += returned - end_began;
    g_stats.end_count++;
    plat_mutex_unlock(&g_mu);
    fr->value = value;
    g_last_value = value;
    g_frame_no++;
    s->last_use = value;
    if (pr != VK_SUCCESS && pr != VK_SUBOPTIMAL_KHR) {
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_ERROR_SURFACE_LOST_KHR) {
            g_sc.dirty = 1;
            plat_mutex_lock(&g_mu);
            g_stats.out_of_date++;
            g_stats.dropped_failed++;
            plat_mutex_unlock(&g_mu);
            return 0;
        }
        if (pr == VK_ERROR_PRESENT_TIMING_QUEUE_FULL_EXT) {   // the results were not read fast enough: this picture is lost
            plat_mutex_lock(&g_mu);
            g_stats.dropped_failed++;
            plat_mutex_unlock(&g_mu);
            poll_timing();
            return 0;
        }
        gpu_vk_failed(pr, "vkQueuePresentKHR");
    }
    if (pr == VK_SUBOPTIMAL_KHR) note_suboptimal();
    if (seq) record_returned(seq, returned);
    plat_mutex_lock(&g_mu);
    if (seq) g_stats.shown++;
    else g_stats.repeats++;
    if (overlay) g_stats.overlay_frames++;
    plat_mutex_unlock(&g_mu);
    return 1;
}

// ---------- the present thread
static void present_thread(void *arg) {
    (void)arg;
    plat_thread_raise_priority();
    uint64_t gen = 0, last_draw = 0;
    for (;;) {
        reap_retired(0);
        plat_mutex_lock(&g_mu);
        if (!g_stop) {
            int filled = 0;
            for (int i = 0; i < SLOTS; i++) filled |= g_slots[i].state == SLOT_FILLED;
            if (!filled) plat_cond_wait_ns(&g_cv, &g_mu, IDLE_NS);
        }
        int stop = g_stop, take = -1;
        for (int i = 0; !stop && i < SLOTS; i++)
            if (g_slots[i].state == SLOT_FILLED && (take < 0 || g_slots[i].seq < g_slots[take].seq)) take = i;
        if (take >= 0) g_slots[take].state = SLOT_BUSY;
        int held = g_held;
        plat_mutex_unlock(&g_mu);
        if (stop) break;
        poll_timing();
        if (take >= 0) {
            Slot *s = &g_slots[take];
            int ok = present_slot(s, s->seq, 1, 0);
            last_draw = plat_now_ns();
            poll_timing();
            plat_mutex_lock(&g_mu);
            if (ok) {
                if (g_held >= 0 && g_held != take) g_slots[g_held].state = SLOT_FREE;
                g_held = take;
                s->state = SLOT_HELD;
            } else {
                s->state = SLOT_FREE;   // lost: the picture the display shows stays held
            }
            plat_mutex_unlock(&g_mu);
            continue;
        }
        // nothing new: draw the held picture again when the window changed or the menu and the overlay are up
        if (held < 0) continue;
        uint64_t now = plat_now_ns();
        if (now - last_draw < REDRAW_NS) continue;
        WinState st;
        g_win->state(&st);
        int changed = st.generation != gen;
        if (!changed && !g_ui_on) continue;
        plat_mutex_lock(&g_mu);
        g_slots[held].state = SLOT_BUSY;
        plat_mutex_unlock(&g_mu);
        // with the window unchanged, a redraw is made only when the menu or the overlay has something to show
        int ok = present_slot(&g_slots[held], 0, 0, changed);
        last_draw = plat_now_ns();
        if (ok) gen = st.generation;   // a picture that could not be drawn leaves the change to be seen again
        plat_mutex_lock(&g_mu);
        g_slots[held].state = SLOT_HELD;
        plat_mutex_unlock(&g_mu);
    }
    plat_mutex_lock(&g_mu);
    g_ended = 1;
    plat_cond_broadcast(&g_cv);
    plat_mutex_unlock(&g_mu);
}

// ---------- the capture's header (oot2/capture.h): what a run on this back end says about its machine
static void note_machine(void) {
    char text[600];
    const VkPhysicalDeviceProperties *p = &gpu_vk.facts.props;
    snprintf(text, sizeof text, "%s, driver %#x, Vulkan %u.%u.%u", p->deviceName, p->driverVersion,
             VK_API_VERSION_MAJOR(p->apiVersion), VK_API_VERSION_MINOR(p->apiVersion), VK_API_VERSION_PATCH(p->apiVersion));
    capture_note("driver", text);
    if (!g_win->display(text, sizeof text)) rt_log("gfx: vulkan: the display's mode is not known: %s", text);
    capture_note("display_mode", text);
    uint32_t n = 0;
    char layers[500] = "";
    if (vkEnumerateInstanceLayerProperties(&n, NULL) == VK_SUCCESS && n) {
        VkLayerProperties *lp = calloc(n, sizeof *lp);
        if (lp && vkEnumerateInstanceLayerProperties(&n, lp) == VK_SUCCESS)
            for (uint32_t i = 0; i < n && strlen(layers) + 60 < sizeof layers; i++)
                snprintf(layers + strlen(layers), sizeof layers - strlen(layers), "%s%s", i ? ", " : "", lp[i].layerName);
        free(lp);
    }
    snprintf(text, sizeof text, "the loader lists %u (which of them load is the process's)%s%s", n, n ? ": " : "", layers);
    capture_note("implicit_vulkan_layers", text);
    capture_note("gpu_timestamps", "none (report timestamps are the host clock at completion; no GPU timestamp is read)");
    capture_note("present_path", g_win ? "snapshot copy, scaled into a FIFO swapchain; shown time from the first pixel out of present timing where the device has it"
                                       : "offscreen capture only");
}

// ---------- open and close
void vkw_open(void) {
    if (!g_win) return;
    uint64_t surface = 0;
    if (!g_win->create_surface(gpu_vk.instance, &surface))
        rt_fatal("gfx: vulkan: the window's surface could not be made");
    g_surface = (VkSurfaceKHR)(uintptr_t)surface;
    VkBool32 ok = VK_FALSE;
    GPU_VK_OK(vkGetPhysicalDeviceSurfaceSupportKHR(gpu_vk.phys, gpu_vk.family, g_surface, &ok), "vkGetPhysicalDeviceSurfaceSupportKHR");
    if (!ok) rt_fatal("gfx: vulkan: the queue family %u cannot present to the window's surface", gpu_vk.family);
    uint32_t n = 0;
    GPU_VK_OK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_vk.phys, g_surface, &n, NULL), "vkGetPhysicalDeviceSurfaceFormatsKHR");
    VkSurfaceFormatKHR *f = calloc(n ? n : 1, sizeof *f);
    if (!f) rt_fatal("gfx: vulkan: out of memory for %u surface formats", n);
    GPU_VK_OK(vkGetPhysicalDeviceSurfaceFormatsKHR(gpu_vk.phys, g_surface, &n, f), "vkGetPhysicalDeviceSurfaceFormatsKHR");
    int pick = n ? 0 : -1;   // a plain 8-bit format: the picture is already display-encoded, and so is the menu
    for (int want = 0; want < 2 && n; want++) {
        VkFormat fmt = want ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_B8G8R8A8_UNORM;
        int found = -1;
        for (uint32_t i = 0; i < n && found < 0; i++)
            if (f[i].format == fmt && f[i].colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) found = (int)i;
        if (found >= 0) {
            pick = found;
            break;
        }
    }
    if (pick < 0 || (f[pick].format != VK_FORMAT_B8G8R8A8_UNORM && f[pick].format != VK_FORMAT_R8G8B8A8_UNORM))
        rt_fatal("gfx: vulkan: the window's surface offers no plain 8-bit format (the picture and the menu are already display-encoded, an sRGB format would encode them twice)");
    g_surface_format = f[pick].format, g_surface_space = f[pick].colorSpace;
    free(f);
    for (int i = 0; i < FRAMES; i++) {
        VkSemaphoreCreateInfo sci = {.sType = VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        GPU_VK_OK(vkCreateSemaphore(gpu_vk.device, &sci, NULL, &g_frames[i].acquire), "vkCreateSemaphore (acquire)");
        g_frames[i].value = 0;
    }
    note_machine();
    g_ui_on = vko_init(g_ui, g_surface_format);
    if (g_ui_on && g_ui->prepare) {
        WinState st;
        g_win->state(&st);
        g_ui->prepare(st.width, st.height);
    }
    memset(g_slots, 0, sizeof g_slots);
    memset(g_shown, 0, sizeof g_shown);
    g_last_seq = 0, g_cal_ok = 0, g_since_cal = 0, g_nretired = 0;
    g_held = -1, g_stop = g_ended = 0, g_frame_no = 0, g_last_value = 0;
    memset(&g_stats, 0, sizeof g_stats);
    g_stats.format = (int)g_surface_format;
    g_thread = plat_thread_start(present_thread, NULL, 1ull << 20);
    if (!g_thread) rt_fatal("gfx: vulkan: the present thread could not start");
    __atomic_store_n(&g_active, 1, __ATOMIC_RELEASE);
    rt_log("gfx: vulkan: present to the window: surface format %d, the menu %s, present timing %s on the device",
           (int)g_surface_format, g_ui_on ? "drawn over the picture" : "not built in", g_timing_dev ? "enabled" : "absent");
}

void vkw_close(void) {
    if (!g_active) return;
    __atomic_store_n(&g_active, 0, __ATOMIC_RELEASE);
    plat_mutex_lock(&g_mu);
    g_stop = 1;
    plat_cond_broadcast(&g_cv);
    while (!g_ended) plat_cond_wait_ns(&g_cv, &g_mu, 50000000ull);
    plat_mutex_unlock(&g_mu);
    plat_thread_free(g_thread);
    g_thread = NULL;
    gpu_sync_finish();
    GPU_VK_OK(vkDeviceWaitIdle(gpu_vk.device), "vkDeviceWaitIdle (the swapchain)");   // the last presents are shown
    sc_destroy(&g_sc);
    reap_retired(1);
    for (int i = 0; i < FRAMES; i++) {
        vkDestroySemaphore(gpu_vk.device, g_frames[i].acquire, NULL);
        g_frames[i].acquire = VK_NULL_HANDLE;
    }
    for (int i = 0; i < SLOTS; i++) slot_image_drop(&g_slots[i]);
    if (g_ui_on) vko_close();
    g_ui_on = 0;
    if (g_surface) g_win->destroy_surface(gpu_vk.instance, (uint64_t)(uintptr_t)g_surface);
    g_surface = VK_NULL_HANDLE;
}
