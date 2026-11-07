// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The replay of the Vulkan back end (architecture D11, D12, D17a, D18; ticket 63): GfxBackend.submit walks the host
// recordings the front end hands it (nvn_obj.h, command records), on the submitting thread, and turns each record into
// the recorder's packets (oot2/vk_rec.h) for one frame: the frame's value is the fence value submit returns.
//
// What the walk keeps. The bound state of one GPU channel across records and across the handles of a submission, as
// the console's channel keeps it between submissions: render targets, viewports, scissors, depth ranges, state objects,
// vertex attributes, streams and buffers, each stage's uniform and storage buffers and texture, image and sampler
// handles, the program, push constants and the descriptor pools. The program's stages and the attachment formats come
// from the ledger's tracker (oot2/ledger.h), which the walk feeds every record, so the null back end and this replay
// share the one walk of bound types. State persists from one submission to the next, as it does on the console.
//
// What a draw or dispatch resolves, at the record (VkReplayOps below; the device's own in runtime/gpu/vk_replay_dev.c):
//   pipeline     the stages' job keys and the attachment formats, through the pipeline builder's blocking get
//                (oot2/vk_pipeline.h); one not built yet is the builder's named stop, until the miss ticket
//   buffers      vertex, index, uniform, storage and indirect ranges through the backing map's resolver
//                (oot2/backing.h), in the frame's submission
//   banks        per stage the driver bank c[0] (the storage buffer table at the stage's base, the base instance, the
//                largest point size; NVN2 compute: a 16-byte head with the group counts, the root tables, then the
//                uniform and storage tables in slot order, sized by the control section), the texture handle bank c[2],
//                the program's own constants c[1] (its code area), the root tables as bank 24, and the support buffer
//                (viewport inverse and size, render scales 1.0), written into the frame's scratch memory
//   storage      a storage buffer binds from the address and size words of its bank (address rounded down to 16 bytes,
//                range the size plus what was rounded off: research 19 B5); one the module may write (no NonWritable,
//                the description's `write`) is declared to the backing map (backing_gpu_wrote) and listed for the copies
//                ticket's write-back (vkr_writes)
//   textures     each push constant slot the module reads takes the handle bound at its stage's c[2] slot: the
//                texture ID mapped to its slot of set 0 for the view type the module declares (the bindless tables),
//                the sampler ID in bits 20 to 31
//   snapshot     a uniform range in host memory the GPU reads live (an import) is copied into the frame's scratch at
//                submit, and the draw binds the copy (D17a); a mirror is already a copy taken at submit (oot2/backing.h),
//                and VRAM is the GPU's. A GPU write of the same submission over a copied range, in either order, is a
//                named stop until the copies ticket brings the drop rule
//   modes        nvn_depth_mode NEAR_IS_MINUS_W sets depth clip control's -1 to 1 range (dynamic state); ZERO_TO_ONE
//                Vulkan's own. nvn_window_origin_mode LOWER_LEFT flips Y through the viewport (a negative height from the
//                viewport's top), the translator's convention (window origin upper left, Y flipped by the viewport)
//
// Passes (D18). Rendering begins lazily at the first draw or clear after a render target change, with every image in
// the general layout, loading and storing; a render target change or a barrier ends the pass, and a full memory barrier
// separates passes. A barrier inside a pass ends it.
//
// Named stops. Every op of nvn_obj.h's list is replayed or stops by name (a named fatal, rt_fatal: the replay has no
// guest CPU to stop): vkr_op_ticket names the ticket a stopped op waits for, and the stop's message names the op and
// the ticket. So does any state the replay cannot say yet (a masked clear, a storage image, a separate sampler).
//
// The recorder (oot2/vk_rec.h) opens with the first replayed submission unless a test opened it first; gpu_vk_close
// closes it.
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/backing.h"
#include "oot2/nvn_obj.h"
#include "oot2/vk_pipeline.h"

struct nvn_stage;

// ---- the op list: replayed, or the ticket a stopped op waits for
#define VKR_REPLAYED NULL
// NULL for an op this replay makes packets of (or ignores by contract, as OP_HINT), else the name of what it waits for
// ("copies (ticket 64)", ...). Past OP_COUNT_: "unknown".
const char *vkr_op_ticket(uint32_t op);

// ---- what the replay needs of a device (the device's, or a test's stand-ins: continuous integration checks the
// op-to-packet resolution without one)
typedef struct VkrTarget {
    VkImageView view;            // a view of one level and the layers the record names, in the attachment's format
    VkFormat format;
    uint32_t width, height, layers;
    VkImageAspectFlags aspect;   // colour, or depth and stencil as the format has them
} VkrTarget;

typedef struct VkReplayOps {
    // The pipeline builder (vkp_graphics, vkp_compute) and the layouts the replay binds against.
    VkPipeline (*graphics)(const VkpStage *vs, const VkpStage *ps, const VkpFormats *formats);
    VkPipeline (*compute)(const VkpStage *cs);
    VkPipelineLayout graphics_layout, compute_layout;
    // The description of a stage (oot2/shader_desc.h), parsed once per stage record and kept: NULL is a named stop.
    const struct ShaderBinds *(*binds)(const struct nvn_stage *s);
    // The backing map (backing_resolve, backing_gpu_wrote) in the frame's submission.
    void (*resolve)(uint64_t addr, uint64_t size, uint64_t value, BackingRef *out);
    void (*gpu_wrote)(uint64_t addr, uint64_t size, uint64_t value);
    // A render target or storage image: the texture as the record holds it, for the frame `value`; 0 when no image
    // stands on its storage (a named stop follows). The view lives until the frame is done.
    int (*target)(const nvn_tex_rec *t, int storage, uint64_t value, VkrTarget *out);
    // The bindless tables for the frame (vkb_acquire, recorded into the frame's prologue) and the slot of a texture ID
    // seen as a view type (vkb_image_slot); UINT32_MAX for none.
    VkDescriptorSet (*acquire)(uint64_t value, const dpool_ref *textures, const dpool_ref *samplers);
    uint32_t (*image_slot)(uint32_t id, int type);
    // `bytes` of host-visible memory the device reads, for the frame `value` (freed once it is done): the pointer, and
    // the buffer and offset that bind it. Offsets are aligned to `align`.
    void *(*scratch)(uint64_t value, uint32_t bytes, uint32_t align, VkBuffer *buffer, VkDeviceSize *offset);
    // Around a frame: before its value is taken (the backing map's sync), and after its last packet, before the frame
    // is handed over (the images made for it brought to the general layout, the backing map's end of submission).
    // Either may be NULL.
    void (*frame_begin)(void);
    void (*frame_end)(uint64_t value);
} VkReplayOps;

// Use these ops from now on (NULL: the device's, vkr_device_ops). A test's ops stay valid while it replays.
void vkr_set_ops(const VkReplayOps *ops);
const VkReplayOps *vkr_device_ops(void);
// Free what the device's ops hold (scratch memory, the parsed descriptions); gpu_vk_close, once every frame is done.
void vkr_device_close(void);

// ---- the replay
// GfxBackend.submit of the Vulkan back end: a frame through the recorder (vkrec_begin, the packets, the backing map's
// end of submission, vkrec_end); returns the frame's value at once.
uint64_t vkr_submit(const uint64_t *handles, int n);
// Forget the bound state (tests; a run never does: a channel keeps its state).
void vkr_reset(void);

// Tests only: while non-zero, uniform ranges are bound live and never copied (the counter-run of the snapshot).
extern int vkr_no_snapshot;
// Tests only: while non-zero, the window origin's Y flip is not applied (the counter-run of the origin mode).
extern int vkr_no_flip;

// ---- what the last frames wrote: the storage and image ranges a draw or dispatch may write (the copies ticket's
// write-back reads them), for the frame `value`; *n how many. Kept for the frames in flight.
typedef struct VkrWrite {
    uint64_t addr, size;
    uint32_t draw;     // the draw or dispatch's number in its frame, from 1
    uint8_t image;     // 1 for a storage image's storage, 0 for a storage buffer
} VkrWrite;
const VkrWrite *vkr_writes(uint64_t value, uint32_t *n);

typedef struct VkrStats {
    uint64_t frames, records, draws, dispatches, clears, passes, barriers;
    uint64_t snapshots, snapshot_bytes;             // uniform ranges copied at submit (D17a)
    uint64_t live_uniforms;                         // uniform ranges bound live (a mirror or VRAM)
    uint64_t storage_writes;                        // storage ranges declared written
    uint64_t unregistered_textures;                 // slots whose texture ID had no entry (the placeholder is read)
    uint64_t scratch_bytes;
} VkrStats;
void vkr_stats(VkrStats *out);
