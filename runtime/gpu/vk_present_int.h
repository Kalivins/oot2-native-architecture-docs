// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What vk_present.c asks of vk_overlay.c (oot2/vk_present.h): the menu's pipeline, its textures and the per-frame buffers
// its draw lists go through. Private to the two files.
#pragma once
#include "oot2/vk_present.h"

#define VKO_FRAMES 3   // pictures in flight: each has buffers of its own

// The overlay's pipeline for a swapchain of colour_format, the sampler and the descriptor pool, the frames' buffers; and the
// menu's texture ops installed (ui->set_texture_ops), so the font atlas is made here when ui->prepare asks. 1, or 0 when
// there is no menu (ui is NULL or has no frame function) and nothing is made.
int vko_init(const VkwUi *ui, VkFormat color_format);
// Everything vko_init made, once no submission can use it (the caller waited).
void vko_close(void);
// The format the pipeline was made for changed: make it again (the caller waited).
void vko_format(VkFormat color_format);

// Copy the lists into the buffers of frame (0 to VKO_FRAMES - 1), which the caller has waited for. Growing a buffer is the
// only allocation on this path, and the caller waited so that the old one can go.
void vko_upload(int frame, const UiDrawData *dd);
// Record the texture uploads waiting since the last frame, then draw the lists with dynamic rendering into view (an image
// in COLOR_ATTACHMENT_OPTIMAL, load as the load operation) of the given extent, in the open submission `value`.
void vko_record(VkCommandBuffer cmd, uint64_t value, int frame, VkImageView view, VkExtent2D extent,
                const UiDrawData *dd);
// 1 when textures wait for a submission to carry their pixels.
int vko_uploads_waiting(void);
