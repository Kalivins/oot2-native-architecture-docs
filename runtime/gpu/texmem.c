// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The staging ring and the texture arena (oot2/texmem.h). The staging chunks serving one submission at a time and taken
// again once it is done follow the staging of DKB Recomp's runtime/vk_memory.c (GPL-3.0-or-later; the same idea as the
// mirrors' staging in vk_backing.c); changed here, the ring has a cap that holds (it waits for the oldest submission, or
// stops by name when only the open submission is in the way) and the arena is a first-fit allocator over blocks with a
// cap of its own and the device's budget behind it.
#include <stdlib.h>
#include <string.h>

#include "oot2/backing.h"
#include "oot2/runtime.h"
#include "oot2/texmem.h"
#include "plat.h"

typedef unsigned long long ull;

typedef struct RingChunk {
    TexChunk c;
    uint64_t used, value;   // bytes handed out; the newest submission it serves (free once that is done)
} RingChunk;

typedef struct Free {
    uint64_t off, size;
} Free;

typedef struct Block {
    uint64_t handle, size, id, used;
    uint32_t type;
    int live;
    Free *free;
    int nfree, capfree;
} Block;

static struct {
    int open;
    TexMemOps ops;
    PlatMutex ring_mu, arena_mu;
    RingChunk *chunks;
    int nchunks, capchunks;
    uint64_t ring_bytes;
    Block *blocks;
    int nblocks, capblocks;
    uint64_t arena_bytes, arena_used, next_id;
    TexMemStats st;
} g = {.ring_mu = PLAT_MUTEX_INIT, .arena_mu = PLAT_MUTEX_INIT};

static uint64_t up(uint64_t v, uint64_t a) { return (v + a - 1) / a * a; }

void texmem_open(const TexMemOps *ops) {
    if (g.open) rt_fatal("gfx: vulkan: the texture memory is opened twice");
    g.ops = *ops;
    g.open = 1;
}

int texmem_is_open(void) { return g.open; }

void texmem_close(void) {
    if (!g.open) return;
    plat_mutex_lock(&g.ring_mu);
    for (int i = 0; i < g.nchunks; i++) g.ops.chunk_free(&g.chunks[i].c);
    free(g.chunks);
    g.chunks = NULL, g.nchunks = g.capchunks = 0, g.ring_bytes = 0;
    plat_mutex_unlock(&g.ring_mu);
    plat_mutex_lock(&g.arena_mu);
    for (int i = 0; i < g.nblocks; i++) {
        if (g.blocks[i].live) g.ops.block_free(g.blocks[i].handle);
        free(g.blocks[i].free);
    }
    free(g.blocks);
    g.blocks = NULL, g.nblocks = g.capblocks = 0, g.arena_bytes = g.arena_used = 0;
    plat_mutex_unlock(&g.arena_mu);
    memset(&g.st, 0, sizeof g.st);
    g.open = 0;
}

// ---------- the ring

static void ring_take_from(RingChunk *c, uint64_t value, uint64_t at, uint64_t size, TexSlot *out) {
    c->value = value;
    c->used = at + size;
    out->chunk = &c->c, out->offset = at, out->map = c->c.map + at, out->address = c->c.address + at;
    g.st.ring_takes++, g.st.ring_taken += size;
}

void texring_take(uint64_t value, uint64_t size, uint64_t align, TexSlot *out) {
    if (!g.open) rt_fatal("gfx: vulkan: a staging region before the texture memory is open");
    uint64_t need = size ? size : 1;
    if (align < 16) align = 16;
    if (need + align > TEXRING_CAP)
        rt_fatal("gfx: vulkan: the staging ring holds at most %llu bytes and a region of %llu bytes was asked for (the "
                 "level of a texture the CPU wrote)", (ull)TEXRING_CAP, (ull)need);
    for (;;) {
        plat_mutex_lock(&g.ring_mu);
        uint64_t done = g.ops.done();
        // a chunk this submission fills already
        for (int i = 0; i < g.nchunks; i++) {
            RingChunk *c = &g.chunks[i];
            uint64_t at = up(c->used, align);
            if (c->value == value && at + need <= c->c.size) {
                ring_take_from(c, value, at, need, out);
                plat_mutex_unlock(&g.ring_mu);
                return;
            }
        }
        // a chunk whose submission is done
        for (int i = 0; i < g.nchunks; i++) {
            RingChunk *c = &g.chunks[i];
            if (c->value != value && c->value <= done && need + align <= c->c.size) {
                ring_take_from(c, value, 0, need, out);
                plat_mutex_unlock(&g.ring_mu);
                return;
            }
        }
        // a chunk of its own, while the ring is under its cap: finished chunks too small for the request go first
        uint64_t sz = need + align > TEXRING_CHUNK ? up(need + align, 1ull << 20) : TEXRING_CHUNK;
        for (int i = 0; i < g.nchunks && g.ring_bytes + sz > TEXRING_CAP;) {
            RingChunk *c = &g.chunks[i];
            if (c->value != value && c->value <= done) {
                g.ops.chunk_free(&c->c);
                g.ring_bytes -= c->c.size;
                *c = g.chunks[--g.nchunks];
            } else {
                i++;
            }
        }
        if (g.ring_bytes + sz <= TEXRING_CAP) {
            RingChunk n = {.value = 0};
            if (g.ops.chunk_new(sz, &n.c)) {
                if (g.nchunks == g.capchunks) {
                    g.capchunks = g.capchunks ? g.capchunks * 2 : 8;
                    g.chunks = realloc(g.chunks, (size_t)g.capchunks * sizeof *g.chunks);
                    if (!g.chunks) rt_fatal("gfx: vulkan: out of memory for the staging ring");
                }
                g.chunks[g.nchunks] = n;
                g.ring_bytes += sz;
                g.st.ring_made++;
                if (sz > TEXRING_CHUNK) g.st.ring_oversize++;
                if (g.ring_bytes > g.st.ring_peak) g.st.ring_peak = g.ring_bytes;
                ring_take_from(&g.chunks[g.nchunks++], value, 0, need, out);
                plat_mutex_unlock(&g.ring_mu);
                return;
            }
        }
        // at the cap: the oldest submission that holds a chunk finishes by itself
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < g.nchunks; i++)
            if (g.chunks[i].value < value && g.chunks[i].value > done && g.chunks[i].value < oldest) oldest = g.chunks[i].value;
        if (oldest == UINT64_MAX) {
            uint64_t mine = 0;
            for (int i = 0; i < g.nchunks; i++)
                if (g.chunks[i].value == value) mine += g.chunks[i].used;
            plat_mutex_unlock(&g.ring_mu);
            rt_fatal("gfx: vulkan: the staging ring is full inside one submission: submission %llu holds %llu bytes of its "
                     "%llu (cap %llu) and asks for %llu more, and no earlier submission is left to wait for; the lever is "
                     "to upload before the submission (the Vulkan replay) or a larger ring", (ull)value, (ull)mine,
                     (ull)g.ring_bytes, (ull)TEXRING_CAP, (ull)need);
        }
        g.st.ring_waits++;
        plat_mutex_unlock(&g.ring_mu);
        g.ops.wait(oldest);
    }
}

// ---------- the arena

static void free_add(Block *b, uint64_t off, uint64_t size) {
    int i = 0;
    while (i < b->nfree && b->free[i].off < off) i++;
    if (i > 0 && b->free[i - 1].off + b->free[i - 1].size == off) {   // joins the extent before
        b->free[i - 1].size += size;
        if (i < b->nfree && b->free[i - 1].off + b->free[i - 1].size == b->free[i].off) {
            b->free[i - 1].size += b->free[i].size;
            memmove(&b->free[i], &b->free[i + 1], (size_t)(b->nfree - i - 1) * sizeof *b->free);
            b->nfree--;
        }
        return;
    }
    if (i < b->nfree && off + size == b->free[i].off) {               // joins the one after
        b->free[i].off = off, b->free[i].size += size;
        return;
    }
    if (b->nfree == b->capfree) {
        b->capfree = b->capfree ? b->capfree * 2 : 8;
        b->free = realloc(b->free, (size_t)b->capfree * sizeof *b->free);
        if (!b->free) rt_fatal("gfx: vulkan: out of memory for the texture arena's free list");
    }
    memmove(&b->free[i + 1], &b->free[i], (size_t)(b->nfree - i) * sizeof *b->free);
    b->free[i] = (Free){off, size};
    b->nfree++;
}

static int cut(Block *b, uint64_t size, uint64_t align, uint64_t *at) {
    for (int i = 0; i < b->nfree; i++) {
        Free f = b->free[i];
        uint64_t a = up(f.off, align);
        if (a + size > f.off + f.size) continue;
        memmove(&b->free[i], &b->free[i + 1], (size_t)(b->nfree - i - 1) * sizeof *b->free);
        b->nfree--;
        if (a > f.off) free_add(b, f.off, a - f.off);
        if (a + size < f.off + f.size) free_add(b, a + size, f.off + f.size - a - size);
        b->used += size;
        *at = a;
        return 1;
    }
    return 0;
}

static uint64_t largest_free(void) {
    uint64_t best = 0;
    for (int k = 0; k < g.nblocks; k++)
        for (int i = 0; g.blocks[k].live && i < g.blocks[k].nfree; i++)
            if (g.blocks[k].free[i].size > best) best = g.blocks[k].free[i].size;
    return best;
}

// Give the blocks with nothing in them back to the device.
static int reclaim(void) {
    int n = 0;
    for (int k = 0; k < g.nblocks; k++) {
        Block *b = &g.blocks[k];
        if (!b->live || b->used) continue;
        g.ops.block_free(b->handle);
        g.arena_bytes -= b->size;
        free(b->free);
        b->free = NULL, b->nfree = b->capfree = 0, b->live = 0;
        g.st.arena_blocks_freed++, g.st.arena_reclaims++;
        n++;
    }
    return n;
}

void texarena_alloc(uint64_t size, uint64_t align, uint32_t type_bits, TexExtent *out) {
    if (!g.open) rt_fatal("gfx: vulkan: an arena extent before the texture memory is open");
    if (align < 1) align = 1;
    uint64_t need = size ? size : 1;
    plat_mutex_lock(&g.arena_mu);
    uint32_t type = g.ops.type_pick(type_bits);
    if (type == UINT32_MAX) {
        plat_mutex_unlock(&g.arena_mu);
        rt_fatal("gfx: vulkan: no device memory type serves an image of the texture arena (types %#x)", type_bits);
    }
    int reclaimed = 0;
    for (;;) {
        for (int k = 0; k < g.nblocks; k++) {
            Block *b = &g.blocks[k];
            uint64_t at;
            if (!b->live || !(type_bits >> b->type & 1) || !cut(b, need, align, &at)) continue;
            *out = (TexExtent){.block = b->handle, .offset = at, .size = need, .type = b->type, .blk = k, .id = b->id};
            g.arena_used += need;
            g.st.arena_allocs++;
            g.st.arena_used = g.arena_used;
            if (g.arena_used > g.st.arena_peak) g.st.arena_peak = g.arena_used;
            plat_mutex_unlock(&g.arena_mu);
            return;
        }
        // no room in the blocks: a new one, if the cap and the budget allow; the empty ones go back first when they do not
        uint64_t bsz = need + align > TEXARENA_BLOCK ? up(need + align, 1ull << 20) : TEXARENA_BLOCK;
        uint64_t used = 0, budget = 0;
        int known = g.ops.budget(&used, &budget);
        const char *why = NULL;
        if (g.arena_bytes + bsz > TEXARENA_CAP) why = "the arena's cap";
        else if (known && !backing_mirror_fits(used, budget, bsz)) why = "the device's memory budget";
        uint64_t handle = 0;
        if (!why && !g.ops.block_new(type, bsz, &handle)) why = "the device (it refused the block)";
        if (why && !reclaimed) {
            reclaimed = 1;
            if (reclaim()) continue;
        }
        if (why) {
            uint64_t live = g.arena_bytes, in_extents = g.arena_used, blocks = 0, best = largest_free();
            for (int k = 0; k < g.nblocks; k++) blocks += g.blocks[k].live;
            plat_mutex_unlock(&g.arena_mu);
            rt_fatal("gfx: vulkan: the texture arena is full (%s): an image needs %llu bytes at alignment %llu, the arena "
                     "holds %llu of its %llu bytes in %llu blocks with %llu in extents and %llu its largest free run; the "
                     "device heap uses %llu of a budget of %llu; evicting textures by uploading them again is not built",
                     why, (ull)need, (ull)align, (ull)live, (ull)TEXARENA_CAP, (ull)blocks, (ull)in_extents, (ull)best,
                     (ull)used, (ull)budget);
        }
        int slot = -1;
        for (int k = 0; k < g.nblocks && slot < 0; k++)
            if (!g.blocks[k].live) slot = k;
        if (slot < 0) {
            if (g.nblocks == g.capblocks) {
                g.capblocks = g.capblocks ? g.capblocks * 2 : 16;
                g.blocks = realloc(g.blocks, (size_t)g.capblocks * sizeof *g.blocks);
                if (!g.blocks) rt_fatal("gfx: vulkan: out of memory for the texture arena's blocks");
            }
            slot = g.nblocks++;
        }
        g.blocks[slot] = (Block){.handle = handle, .size = bsz, .id = ++g.next_id, .type = type, .live = 1};
        free_add(&g.blocks[slot], 0, bsz);
        g.arena_bytes += bsz;
        g.st.arena_blocks_made++;
        if (g.arena_bytes > g.st.arena_bytes_peak) g.st.arena_bytes_peak = g.arena_bytes;
    }
}

void texarena_free(const TexExtent *e) {
    plat_mutex_lock(&g.arena_mu);
    if (e->blk < 0 || e->blk >= g.nblocks || !g.blocks[e->blk].live || g.blocks[e->blk].id != e->id) {
        plat_mutex_unlock(&g.arena_mu);
        rt_fatal("gfx: vulkan: an arena extent at offset %llu of %llu bytes was freed after its block went", (ull)e->offset, (ull)e->size);
    }
    Block *b = &g.blocks[e->blk];
    free_add(b, e->offset, e->size);
    b->used -= e->size;
    g.arena_used -= e->size;
    g.st.arena_frees++;
    g.st.arena_used = g.arena_used;
    plat_mutex_unlock(&g.arena_mu);
}

void texmem_stats(TexMemStats *out) {
    plat_mutex_lock(&g.ring_mu);
    plat_mutex_lock(&g.arena_mu);
    *out = g.st;
    out->ring_chunks = (uint64_t)g.nchunks;
    out->ring_bytes = g.ring_bytes;
    out->arena_blocks = 0;
    for (int k = 0; k < g.nblocks; k++) out->arena_blocks += g.blocks[k].live;
    out->arena_bytes = g.arena_bytes;
    out->arena_used = g.arena_used;
    plat_mutex_unlock(&g.arena_mu);
    plat_mutex_unlock(&g.ring_mu);
}
