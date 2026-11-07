// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Texture and sampler pools, and the device's handles (architecture D10, D13).
//
// A pool is a table of 32-byte descriptors in a memory pool, which the game names by ID: its textures by texture ID,
// its samplers by sampler ID. Registering a texture or a sampler at an ID calls the back end once, for that ID and
// nothing else, so that the back end's bindless tables are updated one entry at a time and never rebuilt (D10; the
// game registers about 110 textures per frame in Kokiri Forest). A texture registration hands over the texture (its
// format, storage range and serial, nvn_obj.h) and the view; a sampler registration hands over the sampler's whole
// state, so that the back end can share one host sampler among equal ones (D13). Nothing is written into the pool's
// memory: the console's driver writes its descriptors there, but only its GPU reads them, and the back end keeps its own.
//
// A handle names descriptors by ID, packed as the console packs them (nvn_tex.h): the texture ID in bits 0 to 19 and
// the sampler ID in bits 20 to 31, which is what the translated shaders unpack from their constant banks. An image
// handle and a texel fetch handle are the texture ID alone. In Kokiri Forest the game asks for about 1,400 texture
// handles and 60 image handles per frame.
//
// Ported from oot-pc-recomp runtime/nvn_api_obj.c and nvn_api_core.c at 8e09aaa (GPL-3.0-or-later), as DKB Recomp
// carries them at 9d8fa02, ported there from MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c at
// b75be8cf (MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile; licenses/dkb-xlate-MIT.txt): pool initialization
// through nvn_dpool_snapshot, the registrations and the handle packing, which nxnative dm/gfx/nvn/hle_nvn.c at 865cc35
// (MIT, licenses/nxnative-MIT.txt) packs the same way. Changed here: the pool limits are the device's own answers; a
// registration outside its pool, of a texture that is not initialised, or with IDs that do not fit their bits is a
// named stop; samplers also register straight from a builder, and pools answer their storage back.
#include <string.h>

#include "nvn_api.h"
#include "oot2/ledger.h"
#include "oot2/nvn_tex.h"

// nvnDeviceGetInteger MAX_TEXTURE_POOL_SIZE (0x24) and MAX_SAMPLER_POOL_SIZE (0x25), nvn_device.c; the handle bits
// hold exactly these.
#define TEXTURE_POOL_MAX (1u << NVN_HANDLE_TEXTURE_BITS)
#define SAMPLER_POOL_MAX (1u << NVN_HANDLE_SAMPLER_BITS)
_Static_assert(TEXTURE_POOL_MAX == 1048576 && SAMPLER_POOL_MAX == 4096, "the device's pool limits");

// NVNboolean nvn{Texture,Sampler}PoolInitialize(NVN...Pool *dp, const NVNmemoryPool *p, ptrdiff_t offset, int n):
// false, and an empty pool, for a table outside an initialised pool or longer than the device allows
static void dpool_init(Cpu *c, uint32_t limit) {
    dpool_t p = {arg_p(c, 1), (int64_t)arg_x(c, 2), (int32_t)arg_x(c, 3)};
    dpool_ref ref;
    int ok = nvn_dpool_snapshot(&p, limit, &ref);
    *(dpool_t *)arg_p(c, 0) = ok ? p : (dpool_t){0};
    ret_x(c, ok);
}
NVN_API(nvnTexturePoolInitialize) { dpool_init(c, TEXTURE_POOL_MAX); }
NVN_API(nvnSamplerPoolInitialize) { dpool_init(c, SAMPLER_POOL_MAX); }
NVN_API(nvnTexturePoolFinalize) { memset(arg_p(c, 0), 0, sizeof(dpool_t)); }
NVN_API(nvnSamplerPoolFinalize) { memset(arg_p(c, 0), 0, sizeof(dpool_t)); }
NVN_API(nvnTexturePoolSetDebugLabel) { (void)c; }
NVN_API(nvnSamplerPoolSetDebugLabel) { (void)c; }
// const NVNmemoryPool *nvn{Texture,Sampler}PoolGetMemoryPool(const NVN...Pool *dp), ptrdiff_t ...GetMemoryOffset,
// int ...GetSize
#define DP(c) ((const dpool_t *)arg_p(c, 0))
NVN_API(nvnTexturePoolGetMemoryPool) { ret_x(c, (uintptr_t)DP(c)->pool); }
NVN_API(nvnTexturePoolGetMemoryOffset) { ret_x(c, (uint64_t)DP(c)->off); }
NVN_API(nvnTexturePoolGetSize) { ret_x(c, (uint32_t)DP(c)->n); }
NVN_API(nvnSamplerPoolGetMemoryPool) { ret_x(c, (uintptr_t)DP(c)->pool); }
NVN_API(nvnSamplerPoolGetMemoryOffset) { ret_x(c, (uint64_t)DP(c)->off); }
NVN_API(nvnSamplerPoolGetSize) { ret_x(c, (uint32_t)DP(c)->n); }

// The table a registration writes into, and its ID checked against it.
static dpool_ref registered(Cpu *c, uint32_t limit, const char *fn) {
    dpool_ref ref;
    if (!nvn_dpool_snapshot(DP(c), limit, &ref) || !ref.count)
        rt_fatal("nvn: %s into a pool that is not initialised (%p)", fn, arg_p(c, 0));
    int64_t id = (int32_t)arg_x(c, 1);
    if (id < 0 || id >= ref.count) rt_fatal("nvn: %s at ID %lld, outside the pool's %u", fn, (long long)id, ref.count);
    return ref;
}

// void nvnTexturePoolRegisterTexture(const NVNtexturePool *tp, int id, const NVNtexture *t, const NVNtextureView *v)
NVN_API(nvnTexturePoolRegisterTexture) {
    dpool_ref pool = registered(c, TEXTURE_POOL_MAX, "nvnTexturePoolRegisterTexture");
    const tex_t *t = arg_p(c, 2);
    if (!t || t->magic != NVN_TEX_MAGIC)
        rt_fatal("nvn: nvnTexturePoolRegisterTexture of a texture that is not initialised (%p)", (const void *)t);
    ledger_desc_register(&pool, (int32_t)arg_x(c, 1), t, arg_p(c, 3));
    if (texheap_on()) texheap_desc(&pool, (int32_t)arg_x(c, 1), t);
    gfx_backend()->texture_register(&pool, (int32_t)arg_x(c, 1), t, arg_p(c, 3));
}
// void nvnSamplerPoolRegisterSampler(const NVNsamplerPool *sp, int id, const NVNsampler *s)
NVN_API(nvnSamplerPoolRegisterSampler) {
    dpool_ref pool = registered(c, SAMPLER_POOL_MAX, "nvnSamplerPoolRegisterSampler");
    gfx_backend()->sampler_register(&pool, (int32_t)arg_x(c, 1), &((const samp_t *)arg_p(c, 2))->p);
}
// void nvnSamplerPoolRegisterSamplerBuilder(const NVNsamplerPool *sp, int id, const NVNsamplerBuilder *b): a
// sampler's state straight from its builder
NVN_API(nvnSamplerPoolRegisterSamplerBuilder) {
    dpool_ref pool = registered(c, SAMPLER_POOL_MAX, "nvnSamplerPoolRegisterSamplerBuilder");
    gfx_backend()->sampler_register(&pool, (int32_t)arg_x(c, 1), &((const samp_builder_t *)arg_p(c, 2))->p);
}

// ---- handles
static uint32_t id_arg(Cpu *c, int i, uint32_t limit, const char *fn) {
    int64_t id = (int32_t)arg_x(c, i);
    if (id < 0 || id >= limit) rt_fatal("nvn: %s: ID %lld does not fit below %u", fn, (long long)id, limit);
    return (uint32_t)id;
}
// NVNtextureHandle nvnDeviceGetTextureHandle(const NVNdevice *d, int texture_id, int sampler_id)
NVN_API(nvnDeviceGetTextureHandle) {
    uint32_t t = id_arg(c, 1, TEXTURE_POOL_MAX, "nvnDeviceGetTextureHandle");
    uint32_t s = id_arg(c, 2, SAMPLER_POOL_MAX, "nvnDeviceGetTextureHandle");
    ret_x(c, nvn_handle_pack(t, s));
}
// NVNtextureHandle nvnDeviceGetTexelFetchHandle(const NVNdevice *d, int texture_id): a fetch reads no sampler
NVN_API(nvnDeviceGetTexelFetchHandle) {
    ret_x(c, nvn_handle_pack(id_arg(c, 1, TEXTURE_POOL_MAX, "nvnDeviceGetTexelFetchHandle"), 0));
}
// NVNimageHandle nvnDeviceGetImageHandle(const NVNdevice *d, int id): an image is read through the texture pool's
// descriptor at its ID (the game registers no image descriptors of its own: research 34)
NVN_API(nvnDeviceGetImageHandle) {
    ret_x(c, nvn_handle_pack(id_arg(c, 1, TEXTURE_POOL_MAX, "nvnDeviceGetImageHandle"), 0));
}
// NVNseparateTextureHandle nvnDeviceGetSeparateTextureHandle(const NVNdevice *d, int id), NVNseparateSamplerHandle
// ...GetSeparateSamplerHandle(d, int id): the two halves of a combined handle, which a shader ORs together
NVN_API(nvnDeviceGetSeparateTextureHandle) {
    ret_x(c, nvn_handle_pack(id_arg(c, 1, TEXTURE_POOL_MAX, "nvnDeviceGetSeparateTextureHandle"), 0));
}
NVN_API(nvnDeviceGetSeparateSamplerHandle) {
    ret_x(c, nvn_handle_pack(0, id_arg(c, 1, SAMPLER_POOL_MAX, "nvnDeviceGetSeparateSamplerHandle")));
}
