// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Present to a window on the Vulkan back end (architecture D22, D23, D24, D29; runtime/gpu/vk_present.c and
// vk_overlay.c). GfxBackend.present hands the game's window texture here. Two threads share the work:
//
//   the guest thread, in present(): one submission of the back end's own (oot2/gpu_vk.h) that copies the crop of the
//   texture into a snapshot image (RGBA8, the crop's size) behind a full barrier, so the picture is the work submitted
//   before the call and no later write of the game can reach it; then it marks the snapshot ready and returns. It never
//   waits for the display, the swapchain or the present thread. When the present thread is more than two pictures behind,
//   the oldest picture not taken yet is dropped for the new one (counted; the game is paced by the virtual clock, not by
//   this queue, so a stalled display costs pictures and never the game).
//
//   the present thread (above normal priority): takes the oldest ready snapshot, acquires a swapchain image, and in one
//   submission scales the snapshot into it with the aspect kept (black bars, a bilinear filter), draws the menu over the
//   scaled picture (D29: a second step on the same image, after the scale, so the snapshot, which the offscreen capture
//   shares the picture with, never holds the overlay), and presents it on FIFO. The swapchain is made again when the window's
//   size changed or the driver says out of date or suboptimal; nothing is presented while the window is minimised. With
//   nothing new to show and the menu or the overlay up, or the window changed, it draws the last picture again.
//
// The shown time (gfx.h present_shown): where the device has VK_EXT_present_timing, the first-pixel-out time of the
// present, calibrated to the host clock and given as the tick of the virtual clock it falls in; otherwise the tick at which
// vkQueuePresentKHR returned. It is read back by the present thread, never blocks a reader, and answers 0 until it is known.
//
// The window and the swapchain are optional: a program that never calls vkw_window (the tests, the offscreen capture
// runs) makes none of this, and its device has no swapchain extension.
#pragma once
#include <stdint.h>

#include "volk.h"

#include "oot2/nvn_obj.h"
#include "oot2/ui.h"
#include "oot2/window.h"

// What the present thread calls into the menu (runtime/ui), by pointer so that runtime/gpu links without it. A NULL
// member, or a NULL table, is no overlay.
typedef struct VkwUi {
    int (*frame)(const UiFrame *frame, UiDraw *out);
    void (*draw_data)(UiDrawData *out);
    void (*set_texture_ops)(const UiTextureOps *ops);
    int (*prepare)(int width, int height);
} VkwUi;

// Before gpu_vk_open: the window the device must be able to present to (win_vulkan, or a test's own) and the menu. The
// pointers are kept, not copied. Calling it with NULL as win goes back to no window.
void vkw_window(const WinVulkan *win, const VkwUi *ui);
int vkw_wanted(void);   // vkw_window was called with a window

// The hooks of gpu_vk_open (vk_device.c).
// The instance extensions the surface needs, with VK_KHR_get_surface_capabilities2 when the loader has it, in names (at
// most max); returns how many.
uint32_t vkw_instance_extensions(const char **names, uint32_t max);
// 1 when the queue family of the device can present to the window's surface and the device lists VK_KHR_swapchain.
int vkw_can_present(VkPhysicalDevice phys, uint32_t family);
// Append the device extensions to names (room for 8 more) and chain their feature structures to features: the swapchain,
// and present id 2, present timing and calibrated timestamps when the device has all three.
void vkw_device_extensions(VkPhysicalDevice phys, const char **names, uint32_t *count, VkPhysicalDeviceFeatures2 *features);
// After the device is made: the surface, the overlay's pipeline and font, the present thread. Stops the start by name when
// the window cannot be presented to.
void vkw_open(void);
// Before the device goes: finish the present thread, destroy the swapchain, the overlay, the snapshots and the surface.
void vkw_close(void);

// The GfxBackend operations (vk_submit.c). vkw_active is 1 between vkw_open and vkw_close.
int vkw_active(void);
void vkw_present(const tex_t *t, const int crop[4]);
void vkw_pace(uint64_t seq, uint64_t due_tick, uint64_t due_count);
uint64_t vkw_shown(uint64_t seq);

// The rectangle of a dst_w by dst_h image that the src_w by src_h picture is scaled into: the largest one of the picture's
// aspect that fits, centred, rounded to whole pixels and never empty (black bars take the rest).
typedef struct VkwRect { int x, y, w, h; } VkwRect;
VkwRect vkw_fit(uint32_t src_w, uint32_t src_h, uint32_t dst_w, uint32_t dst_h);

typedef struct VkwStats {
    uint64_t presents;         // vkw_present calls: snapshots made
    uint64_t shown;            // pictures presented with a game present's number
    uint64_t repeats;          // pictures drawn again with nothing new to show
    uint64_t dropped_ring;     // snapshots replaced before the present thread took them
    uint64_t dropped_window;   // snapshots taken while the window had no area (minimised)
    uint64_t dropped_failed;   // pictures lost to a present or an acquire that failed (out of date)
    uint64_t swapchains;       // swapchains made, the first included
    uint64_t out_of_date, suboptimal;
    uint64_t timing_results;   // present timing results read back
    uint64_t overlay_frames;   // pictures with the menu or the overlay drawn
    uint64_t end_ns_max, end_ns_sum, end_count;   // the present thread's submit-and-present call (gpu_vk_end_present), which holds
                                                  // the submission gate: the longest and the total of its calls, and their number
    uint32_t width, height;    // the swapchain's extent now (0 before the first)
    uint32_t images;
    int timing;                // 1 when the shown time comes from present timing
    int format;                // the swapchain's VkFormat
    int minimised;
} VkwStats;
void vkw_stats(VkwStats *out);

// The shown time of the pictures that have one: the host time (plat_now_ns's clock) of the first pixel out, or 0.
// For the tests, which compare it with the display's own vertical blanks.
uint64_t vkw_shown_ns(uint64_t seq);
// The seq of the newest picture presented (0 before the first) and whether its swapchain image has been handed to the
// display: for a test that waits for a frame to be on screen.
uint64_t vkw_last_seq(void);

// Tests only. While non-zero: the shown time is never fed back (vkw_shown answers 0); the swapchain is not made again when
// the window's size changes (the old extent is kept and the picture is stretched into it by the display).
extern int vkw_no_feedback;
extern int vkw_no_recreate;
