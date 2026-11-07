// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The memory the texture uploads use (architecture D19, section 8): a bounded staging ring that the CPU's texel bytes
// are copied into, and the texture arena, the VRAM the images of CPU-written textures are bound into. No Vulkan type is
// here; runtime/gpu/vk_texture.c gives it a device through TexMemOps, and a test gives it stand-ins.
//
// The ring (TEXRING_CAP at most, 384 MiB). Chunks of TEXRING_CHUNK, host visible, which a submission fills from the
// front; a chunk serves one submission at a time (the one that filled it last) and is taken again, from its start, once
// that submission is done. A request that no chunk with room, and no chunk done, can serve makes a new chunk while the
// ring is under its cap; at the cap it waits for the oldest submission that holds a chunk, since that one finishes by
// itself. A request that only the open submission's own chunks stand in the way of cannot wait (the submission would
// wait for itself): it stops by name with the figures, as does a request larger than the cap.
//
// The arena (TEXARENA_CAP at most, 3.4 GiB). Blocks of TEXARENA_BLOCK (a request larger than that gets a block of its
// own), cut by first fit into extents with their alignment, which images bind into; a freed extent joins its
// neighbours, and a block with nothing in it goes back to the device when a request finds no room and the cap or the
// budget forbids a new block. The budget is VK_EXT_memory_budget's (section 8, the same eighth kept free as for
// mirrors, backing_mirror_fits): a block is made only when used + block + budget / 8 fits. A request nothing can serve
// stops by name with the figures; eviction by uploading again is D19's last resort and is not built.
//
// Locking: the ring and the arena each have a mutex; every call may come from any thread. The ring's waits and the
// ops' calls run with the ring's mutex released.
#pragma once
#include <stddef.h>
#include <stdint.h>

#define TEXRING_CAP (384ull << 20)
#define TEXRING_CHUNK (64ull << 20)
#define TEXRING_ALIGN 512ull        // a region of raw storage starts at this alignment in the ring (a GOB)
#define TEXARENA_CAP ((3ull << 30) + (2ull << 30) / 5)   // 3.4 GiB
#define TEXARENA_BLOCK (64ull << 20)
#define TEXARENA_RESERVE 8          // a block leaves an eighth of the budget free (section 8)

// A chunk of staging: host visible and coherent, mapped, with a device address for the compute pass.
typedef struct TexChunk {
    uint8_t *map;
    uint64_t address, size;
    uint64_t handle[2];   // the ops' own (a VkBuffer and its VkDeviceMemory)
} TexChunk;

typedef struct TexMemOps {
    int (*chunk_new)(uint64_t size, TexChunk *out);         // 1, or 0 when the device has no room
    void (*chunk_free)(const TexChunk *c);                  // no submission can use it any more
    uint32_t (*type_pick)(uint32_t type_bits);              // the memory type an image of these bits takes; UINT32_MAX: none
    int (*block_new)(uint32_t type, uint64_t size, uint64_t *handle);   // 1, or 0 when the device has no room
    void (*block_free)(uint64_t handle);
    int (*budget)(uint64_t *used, uint64_t *budget);        // the device-local heap's figures: 1, or 0 when unknown
    uint64_t (*done)(void);                                 // the newest submission value that is done
    void (*wait)(uint64_t value);                           // returns once that value is done
} TexMemOps;

// Open over ops. Close frees every chunk and block (no submission may be in flight); open may follow.
void texmem_open(const TexMemOps *ops);
void texmem_close(void);
int texmem_is_open(void);

// ---- the ring
// A region of `size` bytes for submission `value`, 16-byte aligned (TEXRING_ALIGN for a region of raw storage).
typedef struct TexSlot {
    const TexChunk *chunk;   // valid until the submission is done
    uint64_t offset;
    uint8_t *map;            // chunk->map + offset
    uint64_t address;        // chunk->address + offset
} TexSlot;
void texring_take(uint64_t value, uint64_t size, uint64_t align, TexSlot *out);

// ---- the arena
typedef struct TexExtent {
    uint64_t block;          // the ops' handle of the block it lies in
    uint64_t offset, size;   // inside the block
    uint32_t type;           // the block's memory type
    int blk;                 // the arena's own index of the block (for texarena_free)
    uint64_t id;             // which block it was when taken (a block's slot may be made again)
} TexExtent;
void texarena_alloc(uint64_t size, uint64_t align, uint32_t type_bits, TexExtent *out);
void texarena_free(const TexExtent *e);

typedef struct TexMemStats {
    uint64_t ring_chunks, ring_bytes, ring_peak;           // chunks live, their bytes, the most bytes the chunks held
    uint64_t ring_takes, ring_taken, ring_waits;           // regions handed out, their bytes, waits for a submission
    uint64_t ring_made, ring_oversize;                     // chunks made, and those larger than TEXRING_CHUNK
    uint64_t arena_blocks, arena_bytes, arena_used, arena_peak;   // blocks live, their bytes, bytes in extents, its peak
    uint64_t arena_bytes_peak;                             // the most bytes the blocks held
    uint64_t arena_allocs, arena_frees, arena_blocks_made, arena_blocks_freed;
    uint64_t arena_reclaims;                               // empty blocks given back to make room
} TexMemStats;
void texmem_stats(TexMemStats *out);
