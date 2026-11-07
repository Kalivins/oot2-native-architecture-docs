// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The offscreen capture of the Vulkan back end: GfxBackend.present until the window, its swapchain, the present thread
// and the pacing exist (D22, D23, D24 and Q4 stay with the present part of the map).
//
// present(t, crop) makes one submission of the back end's own (oot2/gpu_vk.h), after every submission before it on the
// one queue and behind a full barrier, so the picture is the work submitted before the call: a blit of the crop of t's
// image (level 0, layer 0) into the capture image, RGBA8 and the crop's size. It returns once the submission is on the
// queue, without waiting for the GPU. The crop is taken in the texture's rows as memory holds them, row 0 first: the row
// the replay's Y flip puts at the top of the picture. The whole texture when the crop is empty; a crop past the texture
// is clipped to it. A texture no image stands on (storage the CPU writes: the texture arena's), or whose image is not a
// colour image the device can blit from, stops by name.
//
// Settings, read at the first present:
//   OOT2_VK_CAPTURE         unset or empty: no file; a folder: every OOT2_VK_CAPTURE_EVERY-th present (the first
//                           included) is also written there as present-<number>.ppm (binary PPM, RGB, rows top first) by
//                           the completion thread
//   OOT2_VK_CAPTURE_EVERY   a count from 1 (60 when unset)
// A value outside these stops the run by name.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "oot2/nvn_obj.h"

void vkc_present(const tex_t *t, const int crop[4]);
// The newest present's picture into dst (*w x *h RGBA8 pixels, rows top first, when cap holds them): waits for that
// present. Returns the present's number (from 1), 0 when nothing was presented yet or dst is too small (*w and *h say
// what is needed).
uint64_t vkc_capture_read(uint8_t *dst, size_t cap, uint32_t *w, uint32_t *h);
// The newest present's picture as a binary PPM file at path: 1 written, 0 when nothing was presented or the file could not
// be written.
int vkc_capture_save(const char *path);
uint64_t vkc_presents(void);   // presents made
// Free the capture (gpu_vk_close, once every submission is done).
void vkc_close(void);
// Tests only: while non-zero, a present shows the whole texture whatever its crop (the counter-run of the capture).
extern int vkc_no_crop;
