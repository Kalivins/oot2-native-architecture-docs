// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the NVN2 front end's textures, samplers, their pools and handles (architecture D10, D13, D19),
// driven by tests/test_runtime_nvn_tex.py: `nvn_tex_test <case> [arguments]`. Each case runs in its own process. A
// passing case prints "ok <case>"; a failing case must end in one named fatal, which the driver checks.
//
// Entry points are called the way generated code calls them: rt_call at their synthetic address, arguments in the
// AAPCS64 registers (floats in v0 to v7), and the objects they read in guest memory. The cases install a recording back
// end, a copy of the null back end that also keeps what reaches it. Every texel and table is made here; nothing is read
// from a dump.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/gfx.h"
#include "oot2/memory.h"
#include "oot2/nvn_tex.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MiB (1ull << 20)
#define HEAP (32 * MiB)
#define PAGE GFX_GPU_PAGE
#define CPU_VISIBLE (NVN_MEMORY_POOL_FLAGS_CPU_UNCACHED | NVN_MEMORY_POOL_FLAGS_GPU_CACHED)
#define GPU_ONLY (NVN_MEMORY_POOL_FLAGS_CPU_NO_ACCESS | NVN_MEMORY_POOL_FLAGS_GPU_CACHED)

typedef unsigned long long ull;

static Cpu c;
static uint64_t heap, heap_used;
static uint64_t loader;   // nvnBootstrapLoader, bound as the module loader binds main's import

static void fake_fn(Cpu *cpu) { (void)cpu; }

// ---- the recording back end: what the texture and sampler entry points hand the back end
static struct {
    int textures, samplers, releases, written, writes, reads;
    dpool_ref pool;
    int id;
    tex_t tex;                 // a copy of the registered texture, as the back end may keep one
    view_t view;
    int has_view;
    samp_par samp;
    uint64_t released, addr, size, row, img;
} rec;
static void rec_texture_register(const dpool_ref *pool, int id, const tex_t *t, const view_t *v) {
    rec.textures++, rec.pool = *pool, rec.id = id, rec.tex = *t, rec.has_view = v != NULL;
    if (v) rec.view = *v;
}
static void rec_sampler_register(const dpool_ref *pool, int id, const samp_par *s) {
    rec.samplers++, rec.pool = *pool, rec.id = id, rec.samp = *s;
}
static void rec_texture_release(const tex_t *t) { rec.releases++, rec.released = t->serial; }
static void rec_write_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, const uint8_t *src,
                             uint64_t row, uint64_t img) {
    (void)t, (void)v, (void)r, (void)src;
    rec.writes++, rec.row = row, rec.img = img;
}
static void rec_read_texels(const tex_t *t, const view_t *v, const NVNcopyRegion *r, uint8_t *dst, uint64_t row,
                            uint64_t img) {
    (void)t, (void)v, (void)r, (void)dst;
    rec.reads++, rec.row = row, rec.img = img;
}
static void rec_guest_written(uint64_t addr, uint64_t size) { rec.written++, rec.addr = addr, rec.size = size; }
static GfxBackend recording;

// Guest memory, the recording back end, the entry points registered and the host-function registry sealed.
static void setup(void) {
    RtMemConfig cfg = {.pool_size = 256 * MiB, .heap_region_size = 64 * MiB, .alias_region_size = 1024 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    c.x[0] = 0, c.x[1] = HEAP;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    recording = gfx_null_backend;
    recording.name = "recording";
    recording.texture_register = rec_texture_register, recording.sampler_register = rec_sampler_register;
    recording.texture_release = rec_texture_release, recording.guest_written = rec_guest_written;
    recording.write_texels = rec_write_texels, recording.read_texels = rec_read_texels;
    gfx_set_backend(&recording);
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    nvn_procs_register();
    static const uint64_t addrs[] = {MOD_MAIN_BASE + 0x100};
    static const GuestFn fns[] = {fake_fn};
    static const RtModule mods[] = {{"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, addrs, fns, 1, TEST_END}};
    rt_dispatch_init(mods, 1);
    c.excl_addr = ~0ull;
}

// Guest memory for an object or a pool's storage, zeroed, as the game's allocator would give it.
static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}
static void *gp(uint64_t a) { return (void *)(uintptr_t)a; }

static uint64_t guest_str(const char *s) {
    uint64_t a = galloc(strlen(s) + 1, 16);
    memcpy(gp(a), s, strlen(s) + 1);
    return a;
}

// An entry point's address, asked for once through nvnBootstrapLoader and nvnDeviceGetProcAddress.
static uint64_t proc(const char *name) {
    static struct { const char *name; uint64_t addr; } cache[160];
    static int n;
    static uint64_t gpa;
    for (int i = 0; i < n; i++)
        if (!strcmp(cache[i].name, name)) return cache[i].addr;
    if (!gpa) {
        c.x[0] = guest_str("nvnDeviceGetProcAddress"), c.x[1] = 0;
        rt_call(&c, loader);
        gpa = c.x[0];
        CHECK(gpa != 0);
    }
    c.x[0] = 0, c.x[1] = guest_str(name);
    rt_call(&c, gpa);
    CHECK(c.x[0] != 0 && n < 160);
    cache[n].name = name, cache[n].addr = c.x[0];
    return cache[n++].addr;
}

// NVN(nvnX, arguments...): call nvnX with up to eight integer or pointer arguments in x0 to x7. NVNF(nvnX, floats,
// n, arguments...) also puts n floats in v0 to v(n-1).
static uint64_t nvn_vcall(uint64_t fn, const float *f, size_t nf, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    for (size_t i = 0; i < 8; i++) {
        uint32_t bits = 0;
        if (i < nf) memcpy(&bits, &f[i], 4);
        c.v[i] = bits;
    }
    rt_call(&c, fn);
    return c.x[0];
}
#define ARGS(...) (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t)
#define NVN(name, ...) nvn_vcall(proc(#name), NULL, 0, ARGS(__VA_ARGS__))
#define NVNF(name, floats, nf, ...) nvn_vcall(proc(#name), floats, nf, ARGS(__VA_ARGS__))
static float ret_float(void) {
    uint32_t bits = (uint32_t)c.v[0];
    float f;
    memcpy(&f, &bits, 4);
    return f;
}

// ---- objects made through the entry points
static uint64_t pool(uint64_t size, uint32_t flags) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x100, 16);
    uint64_t storage = galloc(size, PAGE);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetStorage, b, storage, size);
    NVN(nvnMemoryPoolBuilderSetFlags, b, flags);
    CHECK((NVN(nvnMemoryPoolInitialize, o, b) & 0xff) == 1);
    return o;
}
static uint64_t pool_addr(uint64_t p) { return NVN(nvnMemoryPoolGetBufferAddress, p); }

// A texture builder for a 2D RGBA8 texture of w x h and one level, with storage at offset off of pool p.
static uint64_t builder(uint32_t w, uint32_t h, uint64_t p, uint64_t off) {
    uint64_t b = galloc(0x80, 16);
    NVN(nvnTextureBuilderSetDevice, b, 0x1234);
    NVN(nvnTextureBuilderSetDefaults, b);
    NVN(nvnTextureBuilderSetSize2D, b, w, h);
    NVN(nvnTextureBuilderSetFormat, b, NVN_FORMAT_RGBA8);
    NVN(nvnTextureBuilderSetStorage, b, p, off);
    return b;
}
static uint64_t texture(uint64_t b) {
    uint64_t t = galloc(0x80, 16);
    CHECK((NVN(nvnTextureInitialize, t, b) & 0xff) == 1);
    return t;
}
static uint64_t texture_pool(uint64_t p, uint64_t off, int n) {
    uint64_t tp = galloc(0x20, 16);
    CHECK((NVN(nvnTexturePoolInitialize, tp, p, off, (uint64_t)n) & 0xff) == 1);
    return tp;
}
static uint64_t sampler_pool(uint64_t p, uint64_t off, int n) {
    uint64_t sp = galloc(0x20, 16);
    CHECK((NVN(nvnSamplerPoolInitialize, sp, p, off, (uint64_t)n) & 0xff) == 1);
    return sp;
}

// ---- storage: the console's sizes for the descriptions the driver passes
// Each argument is one texture: target:format:flags:width:height:depth:levels:stride:packaged:samples, where packaged
// is byte 0 of the packaged layout plus one (0: none). Prints the storage size and alignment the entry points answer,
// the block height and depth, and the offset of each level from 0 to `levels` through GetViewOffset.
static void case_layout(int argc, char **argv) {
    setup();
    for (int i = 0; i < argc; i++) {
        unsigned long v[10] = {0};
        char *s = argv[i];
        for (int k = 0; k < 10; k++) {
            v[k] = strtoul(s, &s, 0);
            CHECK(k == 9 ? *s == 0 : *s == ':');
            s++;
        }
        uint64_t b = galloc(0x80, 16), view = galloc(0x10, 16);
        NVN(nvnTextureBuilderSetDefaults, b);
        NVN(nvnTextureBuilderSetTarget, b, v[0]);
        NVN(nvnTextureBuilderSetFormat, b, v[1]);
        NVN(nvnTextureBuilderSetFlags, b, v[2]);
        NVN(nvnTextureBuilderSetWidth, b, v[3]);
        NVN(nvnTextureBuilderSetHeight, b, v[4]);
        NVN(nvnTextureBuilderSetDepth, b, v[5]);
        NVN(nvnTextureBuilderSetLevels, b, v[6]);
        NVN(nvnTextureBuilderSetStride, b, v[7]);
        if (v[8]) {
            uint64_t l = galloc(8, 8);
            *(uint8_t *)gp(l) = (uint8_t)(v[8] - 1);
            NVN(nvnTextureBuilderSetPackagedTextureLayout, b, l);
        }
        NVN(nvnTextureBuilderSetSamples, b, v[9]);
        uint64_t size = NVN(nvnTextureBuilderGetStorageSize, b), align = NVN(nvnTextureBuilderGetStorageAlignment, b);
        nvn_texlayout L;
        nvn_tex_layout(&((const tex_builder_t *)gp(b))->d, &L);
        printf("row %d size %#llx align %#llx tile %u %u offsets", i, (ull)size, (ull)align, L.tileH, L.tileD);
        for (unsigned lv = 0; lv <= v[6]; lv++) {
            NVN(nvnTextureViewSetDefaults, view);
            NVN(nvnTextureViewSetLevels, view, lv, 1);
            printf(" %#llx", (ull)NVN(nvnTextureBuilderGetViewOffset, b, view));
        }
        printf("\n");
    }
    printf("ok layout\n");
}

// The bytes per block and block dimensions of formats 0 to 0x9E, for the driver to compare with the format names.
static void case_formats(void) {
    for (int f = 0; f <= 0x9e; f++) {
        int bw, bh, bpb = nvn_fmt_bpb(f, &bw, &bh);
        printf("format %#x %d %d %d\n", f, bpb, bw, bh);
    }
    printf("ok formats\n");
}

// A format no source sizes stops the storage query with its number.
static void case_unknown_format(const char *format) {
    setup();
    uint64_t b = galloc(0x80, 16);
    NVN(nvnTextureBuilderSetDefaults, b);
    NVN(nvnTextureBuilderSetFormat, b, strtoul(format, NULL, 0));
    NVN(nvnTextureBuilderGetStorageSize, b);
}

// ---- textures and views
static void case_texture(void) {
    setup();
    uint64_t p = pool(4 * PAGE, CPU_VISIBLE), g = pool(4 * PAGE, GPU_ONLY);
    CHECK(gfx_gpu_only(pool_addr(g)) && !gfx_gpu_only(pool_addr(p)));

    // The builder keeps what it is given, and SetDefaults keeps the device.
    uint64_t b = builder(64, 32, p, 0x2000);
    CHECK(NVN(nvnTextureBuilderGetDevice, b) == 0x1234);
    CHECK(NVN(nvnTextureBuilderGetTarget, b) == NVN_TEXTURE_TARGET_2D);
    CHECK(NVN(nvnTextureBuilderGetWidth, b) == 64 && NVN(nvnTextureBuilderGetHeight, b) == 32);
    CHECK(NVN(nvnTextureBuilderGetDepth, b) == 1 && NVN(nvnTextureBuilderGetLevels, b) == 1);
    CHECK(NVN(nvnTextureBuilderGetFormat, b) == NVN_FORMAT_RGBA8);
    CHECK(NVN(nvnTextureBuilderGetMemoryPool, b) == p && NVN(nvnTextureBuilderGetMemoryOffset, b) == 0x2000);
    uint64_t sw = galloc(16, 16);
    NVN(nvnTextureBuilderGetSwizzle, b, sw, sw + 4, sw + 8, sw + 12);
    CHECK(!memcmp(gp(sw), (const int32_t[]){2, 3, 4, 5}, 16));
    NVN(nvnTextureBuilderSetSwizzle, b, NVN_TEXTURE_SWIZZLE_B, NVN_TEXTURE_SWIZZLE_G, NVN_TEXTURE_SWIZZLE_R,
        NVN_TEXTURE_SWIZZLE_ONE);
    NVN(nvnTextureBuilderSetDepthStencilMode, b, NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL);
    CHECK(NVN(nvnTextureBuilderGetDepthStencilMode, b) == 1);
    NVN(nvnTextureBuilderSetDepthStencilMode, b, NVN_TEXTURE_DEPTH_STENCIL_MODE_DEPTH);
    CHECK(NVN(nvnTextureBuilderGetStorageSize, b) == 0x2000 && NVN(nvnTextureBuilderGetStorageAlignment, b) == 512);
    CHECK(NVN(nvnTextureBuilderGetStorageClass, b) == 1);
    // ZCull: nxnative's estimate, 8 bytes per 32x32 tile, rounded up to 256
    CHECK(NVN(nvnTextureBuilderGetZCullStorageSize, b) == 0x100);
    NVN(nvnTextureBuilderSetSize2D, b, 1280, 720);
    CHECK(NVN(nvnTextureBuilderGetZCullStorageSize, b) == ((40 * 23 * 8 + 0xff) & ~0xffull));
    NVN(nvnTextureBuilderSetSize2D, b, 64, 32);

    // A texture: its storage is the pool's GPU address plus its offset, its description the builder's.
    uint64_t t = texture(b);
    CHECK(NVN(nvnTextureGetTextureAddress, t) == pool_addr(p) + 0x2000);
    CHECK(NVN(nvnTextureGetMemoryPool, t) == p && NVN(nvnTextureGetMemoryOffset, t) == 0x2000);
    CHECK(NVN(nvnTextureGetWidth, t) == 64 && NVN(nvnTextureGetHeight, t) == 32 && NVN(nvnTextureGetDepth, t) == 1);
    CHECK(NVN(nvnTextureGetFormat, t) == NVN_FORMAT_RGBA8 && NVN(nvnTextureGetTarget, t) == NVN_TEXTURE_TARGET_2D);
    CHECK(NVN(nvnTextureGetStorageSize, t) == 0x2000 && NVN(nvnTextureGetLevels, t) == 1);
    NVN(nvnTextureGetSwizzle, t, sw, sw + 4, sw + 8, sw + 12);
    CHECK(!memcmp(gp(sw), (const int32_t[]){4, 3, 2, 1}, 16));
    CHECK(NVN(nvnTextureGetDebugID, t) != 0);
    // In a pool the CPU cannot see, the texture's address is the pool's window address plus its offset.
    uint64_t tg = texture(builder(64, 32, g, 0x4000));
    CHECK(NVN(nvnTextureGetTextureAddress, tg) == pool_addr(g) + 0x4000);
    // Storage outside the pool, or no pool, is refused.
    uint64_t o = galloc(0x80, 16);
    CHECK((NVN(nvnTextureInitialize, o, builder(64, 32, p, 4 * PAGE - 0x1000)) & 0xff) == 0);
    CHECK((NVN(nvnTextureInitialize, o, builder(64, 32, 0, 0)) & 0xff) == 0);
    CHECK((NVN(nvnTextureInitialize, o, builder(64, 32, p, (uint64_t)-512)) & 0xff) == 0);

    // Packaged data is copied into storage the CPU sees, and the back end is told once.
    uint64_t data = galloc(0x2000, 16);
    for (int i = 0; i < 0x2000; i++) ((uint8_t *)gp(data))[i] = (uint8_t)(i * 7 + 1);
    uint64_t pb = builder(64, 32, p, 0x8000);
    NVN(nvnTextureBuilderSetPackagedTextureData, pb, data);
    CHECK(NVN(nvnTextureBuilderGetPackagedTextureData, pb) == data);
    rec.written = 0;
    texture(pb);
    CHECK(!memcmp(gp(pool_addr(p) + 0x8000), gp(data), 0x2000));
    CHECK(rec.written == 1 && rec.addr == pool_addr(p) + 0x8000 && rec.size == 0x2000);

    // Views: what they set, and only that, is read back and compared.
    uint64_t v = galloc(0x10, 16), w = galloc(0x10, 16), out = galloc(16, 16);
    NVN(nvnTextureViewSetDefaults, v);
    NVN(nvnTextureViewSetDefaults, w);
    CHECK(NVN(nvnTextureViewCompare, v, w) == 1);
    CHECK((NVN(nvnTextureViewGetLevels, v, out, out + 4) & 0xff) == 0);
    NVN(nvnTextureViewSetLevels, v, 2, 3);
    NVN(nvnTextureViewSetLayers, v, 4, 5);
    NVN(nvnTextureViewSetFormat, v, NVN_FORMAT_RGBA8_SRGB);
    NVN(nvnTextureViewSetTarget, v, NVN_TEXTURE_TARGET_2D_ARRAY);
    NVN(nvnTextureViewSetSwizzle, v, 5, 4, 3, 2);
    NVN(nvnTextureViewSetDepthStencilMode, v, NVN_TEXTURE_DEPTH_STENCIL_MODE_STENCIL);
    CHECK(NVN(nvnTextureViewCompare, v, w) == 0);
    CHECK(NVN(nvnTextureViewGetLevels, v, out, out + 4) == 1 && !memcmp(gp(out), (const int32_t[]){2, 3}, 8));
    CHECK(NVN(nvnTextureViewGetLayers, v, out, out + 4) == 1 && !memcmp(gp(out), (const int32_t[]){4, 5}, 8));
    CHECK(NVN(nvnTextureViewGetFormat, v, out) == 1 && *(int32_t *)gp(out) == NVN_FORMAT_RGBA8_SRGB);
    CHECK(NVN(nvnTextureViewGetTarget, v, out) == 1 && *(int32_t *)gp(out) == NVN_TEXTURE_TARGET_2D_ARRAY);
    CHECK(NVN(nvnTextureViewGetSwizzle, v, out, out + 4, out + 8, out + 12) == 1);
    CHECK(!memcmp(gp(out), (const int32_t[]){5, 4, 3, 2}, 16));
    CHECK(NVN(nvnTextureViewGetDepthStencilMode, v, out) == 1 && *(int32_t *)gp(out) == 1);
    memcpy(gp(w), gp(v), 16);
    CHECK(NVN(nvnTextureViewCompare, v, w) == 1);
    NVN(nvnTextureViewSetLayers, w, 4, 6);
    CHECK(NVN(nvnTextureViewCompare, v, w) == 0);

    // A view's offset: its base level's, then its first layer's (a 2D array of 3 layers of 64x64 RGBA8, 2 levels:
    // level 0 is 4 GOBs by one 8-GOB block, 0x4000 bytes; level 1 one 4-GOB block 2 GOBs wide, 0x1000).
    uint64_t ab = builder(64, 64, p, 0x10000);
    NVN(nvnTextureBuilderSetTarget, ab, NVN_TEXTURE_TARGET_2D_ARRAY);
    NVN(nvnTextureBuilderSetDepth, ab, 3);
    NVN(nvnTextureBuilderSetLevels, ab, 2);
    uint64_t at = texture(ab);
    nvn_texlayout L;
    nvn_tex_layout(&((const tex_t *)gp(at))->d, &L);
    NVN(nvnTextureViewSetDefaults, w);
    NVN(nvnTextureViewSetLevels, w, 1, 1);
    NVN(nvnTextureViewSetLayers, w, 2, 1);
    CHECK(NVN(nvnTextureGetViewOffset, at, w) == 0x4000 + 2 * L.layer_size && L.layer_size == 0x5000);
    CHECK(NVN(nvnTextureGetViewOffset, at, 0) == 0);

    // Texels: the region's tight strides, or the game's, reach the back end.
    NVNcopyRegion *r = gp(galloc(sizeof(NVNcopyRegion), 16));
    *r = (NVNcopyRegion){0, 0, 0, 16, 8, 1};
    uint64_t buf = galloc(0x1000, 16);
    NVN(nvnTextureWriteTexels, t, 0, (uintptr_t)r, buf);
    CHECK(rec.writes == 1 && rec.row == 64 && rec.img == 512);
    NVN(nvnTextureReadTexelsStrided, t, v, (uintptr_t)r, buf, 256, 4096);
    CHECK(rec.reads == 1 && rec.row == 256 && rec.img == 4096);
    NVN(nvnTextureFlushTexels, t, 0, (uintptr_t)r);
    NVN(nvnTextureInvalidateTexels, t, 0, (uintptr_t)r);

    // Finalize tells the back end once, with the texture's serial.
    uint64_t serial = NVN(nvnTextureGetDebugID, t);
    rec.releases = 0;
    NVN(nvnTextureFinalize, t);
    NVN(nvnTextureFinalize, t);
    CHECK(rec.releases == 1 && rec.released == serial);
    printf("ok texture\n");
}

// Two textures of the same description on the same storage, then copied by value elsewhere: each copy keeps naming its
// own texture, and a texture initialized again at the same address names a third one.
static void case_copies(void) {
    setup();
    uint64_t p = pool(4 * PAGE, CPU_VISIBLE);
    uint64_t tp = texture_pool(p, 0, 64);
    uint64_t b = builder(64, 64, p, PAGE);
    uint64_t a = texture(b), bt = texture(b);
    uint64_t sa = NVN(nvnTextureGetDebugID, a), sb = NVN(nvnTextureGetDebugID, bt);
    CHECK(sa && sb && sa != sb);
    CHECK(NVN(nvnTextureCompare, a, bt) == 1);   // the same texture to the console's eye
    uint64_t ca = galloc(0x80, 16), cb = galloc(0x80, 16);
    memcpy(gp(ca), gp(a), 0x80);
    memcpy(gp(cb), gp(bt), 0x80);
    NVN(nvnTexturePoolRegisterTexture, tp, 1, ca, 0);
    CHECK(rec.textures == 1 && rec.tex.serial == sa);
    NVN(nvnTexturePoolRegisterTexture, tp, 2, cb, 0);
    CHECK(rec.textures == 2 && rec.tex.serial == sb);
    // a initialized again without Finalize: its old serial is released, the copy still names it
    rec.releases = 0;
    CHECK((NVN(nvnTextureInitialize, a, b) & 0xff) == 1);
    uint64_t sa2 = NVN(nvnTextureGetDebugID, a);
    CHECK(sa2 != sa && sa2 != sb && rec.releases == 1 && rec.released == sa);
    NVN(nvnTexturePoolRegisterTexture, tp, 3, ca, 0);
    CHECK(rec.tex.serial == sa);
    NVN(nvnTexturePoolRegisterTexture, tp, 4, a, 0);
    CHECK(rec.tex.serial == sa2);
    printf("serials %llu %llu %llu\n", (ull)sa, (ull)sb, (ull)sa2);
    printf("ok copies\n");
}

// ---- pools and registration
static void case_register(void) {
    setup();
    uint64_t p = pool(4 * PAGE, CPU_VISIBLE), g = pool(4 * PAGE, GPU_ONLY);
    // A table outside its memory pool, misaligned, empty or past the device's limit is refused.
    uint64_t bad = galloc(0x20, 16);
    CHECK((NVN(nvnTexturePoolInitialize, bad, p, 4 * PAGE - 32 * 8, 9) & 0xff) == 0);
    CHECK((NVN(nvnTexturePoolInitialize, bad, p, 16, 4) & 0xff) == 0);
    CHECK((NVN(nvnTexturePoolInitialize, bad, p, 0, 0) & 0xff) == 0);
    CHECK((NVN(nvnSamplerPoolInitialize, bad, p, 0, 4097) & 0xff) == 0);
    uint64_t tp = texture_pool(p, 0x400, 256), sp = sampler_pool(p, 0x4000, 4096), tpg = texture_pool(g, 0, 16);
    CHECK(NVN(nvnTexturePoolGetMemoryPool, tp) == p && NVN(nvnTexturePoolGetMemoryOffset, tp) == 0x400);
    CHECK(NVN(nvnTexturePoolGetSize, tp) == 256 && NVN(nvnSamplerPoolGetSize, sp) == 4096);
    CHECK(NVN(nvnSamplerPoolGetMemoryPool, sp) == p && NVN(nvnSamplerPoolGetMemoryOffset, sp) == 0x4000);

    // One registration, one call into the back end, for that ID, with the view's parameters.
    uint64_t b = builder(64, 64, p, PAGE);
    NVN(nvnTextureBuilderSetLevels, b, 3);
    uint64_t t = texture(b), v = galloc(0x10, 16);
    NVN(nvnTextureViewSetDefaults, v);
    NVN(nvnTextureViewSetLevels, v, 1, 2);
    NVN(nvnTextureViewSetFormat, v, NVN_FORMAT_RGBA8_SRGB);
    NVN(nvnTexturePoolRegisterTexture, tp, 255, t, v);
    CHECK(rec.textures == 1 && rec.id == 255);
    CHECK(rec.pool.addr == pool_addr(p) + 0x400 && rec.pool.count == 256);
    CHECK(rec.tex.serial == NVN(nvnTextureGetDebugID, t) && rec.tex.d.format == NVN_FORMAT_RGBA8);
    CHECK((uintptr_t)rec.tex.mem == pool_addr(p) + PAGE && rec.tex.L.size == NVN(nvnTextureGetStorageSize, t));
    CHECK(rec.has_view && rec.view.set_levels && rec.view.base_level == 1 && rec.view.nlevels == 2);
    CHECK(rec.view.set_format && rec.view.format == NVN_FORMAT_RGBA8_SRGB && !rec.view.set_layers);
    NVN(nvnTexturePoolRegisterTexture, tpg, 0, t, 0);
    CHECK(rec.textures == 2 && rec.id == 0 && !rec.has_view && rec.pool.addr == pool_addr(g) && rec.pool.count == 16);

    // A sampler reaches the back end with its whole state, from a sampler or straight from its builder.
    uint64_t sb = galloc(0x60, 16), s = galloc(0x30, 16), border = galloc(16, 16);
    NVN(nvnSamplerBuilderSetDefaults, sb);
    NVN(nvnSamplerBuilderSetMinMagFilter, sb, NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST, NVN_MAG_FILTER_NEAREST);
    NVN(nvnSamplerBuilderSetWrapMode, sb, NVN_WRAP_MODE_CLAMP_TO_EDGE, NVN_WRAP_MODE_MIRRORED_REPEAT,
        NVN_WRAP_MODE_CLAMP_TO_BORDER);
    NVNF(nvnSamplerBuilderSetLodClamp, ((const float[]){0.5f, 7.25f}), 2, sb);
    NVNF(nvnSamplerBuilderSetLodBias, ((const float[]){-1.5f}), 1, sb);
    NVNF(nvnSamplerBuilderSetMaxAnisotropy, ((const float[]){8.f}), 1, sb);
    NVN(nvnSamplerBuilderSetCompare, sb, NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE, NVN_COMPARE_FUNC_GEQUAL);
    NVN(nvnSamplerBuilderSetReductionFilter, sb, NVN_SAMPLER_REDUCTION_MAX);
    memcpy(gp(border), (const float[]){0.25f, 0.5f, 0.75f, 1.f}, 16);
    NVN(nvnSamplerBuilderSetBorderColor, sb, border);
    CHECK((NVN(nvnSamplerInitialize, s, sb) & 0xff) == 1);
    NVN(nvnSamplerPoolRegisterSampler, sp, 4095, s);
    const samp_par want = {NVN_MIN_FILTER_LINEAR_MIPMAP_NEAREST, NVN_MAG_FILTER_NEAREST, NVN_WRAP_MODE_CLAMP_TO_EDGE,
                           NVN_WRAP_MODE_MIRRORED_REPEAT, NVN_WRAP_MODE_CLAMP_TO_BORDER,
                           NVN_COMPARE_MODE_COMPARE_R_TO_TEXTURE, NVN_COMPARE_FUNC_GEQUAL, NVN_SAMPLER_REDUCTION_MAX,
                           0.5f, 7.25f, -1.5f, 8.f, {0.25f, 0.5f, 0.75f, 1.f}, 0.f};
    CHECK(rec.samplers == 1 && rec.id == 4095 && rec.pool.addr == pool_addr(p) + 0x4000 && rec.pool.count == 4096);
    CHECK(!memcmp(&rec.samp, &want, sizeof want));
    NVN(nvnSamplerBuilderSetReductionFilter, sb, NVN_SAMPLER_REDUCTION_MIN);
    NVN(nvnSamplerPoolRegisterSamplerBuilder, sp, 7, sb);
    CHECK(rec.samplers == 2 && rec.id == 7 && rec.samp.reduction == NVN_SAMPLER_REDUCTION_MIN);
    CHECK(rec.textures == 2);

    NVN(nvnTexturePoolFinalize, tp);
    CHECK(NVN(nvnTexturePoolGetSize, tp) == 0);
    printf("ok register\n");
}

// The failing registrations: outside the pool, of a finalized texture, into a finalized pool.
static void case_register_bad(const char *what) {
    setup();
    uint64_t p = pool(4 * PAGE, CPU_VISIBLE);
    uint64_t tp = texture_pool(p, 0, 8), t = texture(builder(64, 64, p, PAGE));
    if (!strcmp(what, "outside")) NVN(nvnTexturePoolRegisterTexture, tp, 8, t, 0);
    if (!strcmp(what, "negative")) NVN(nvnTexturePoolRegisterTexture, tp, (uint64_t)-1, t, 0);
    if (!strcmp(what, "finalized")) {
        NVN(nvnTextureFinalize, t);
        NVN(nvnTexturePoolRegisterTexture, tp, 0, t, 0);
    }
    if (!strcmp(what, "pool")) {
        NVN(nvnTexturePoolFinalize, tp);
        NVN(nvnTexturePoolRegisterTexture, tp, 0, t, 0);
    }
    if (!strcmp(what, "packaged")) {   // packaged data into storage the CPU cannot see
        uint64_t b = builder(64, 64, pool(4 * PAGE, GPU_ONLY), 0);
        NVN(nvnTextureBuilderSetPackagedTextureData, b, galloc(0x4000, 16));
        texture(b);
    }
    printf("no stop for %s\n", what);
}

// ---- handles
// Each argument is texture:sampler. Prints the combined handle, the two separate handles, and the texel fetch and image
// handles of the texture ID, as the game receives them.
static void case_handles(int argc, char **argv) {
    setup();
    for (int i = 0; i < argc; i++) {
        char *s = argv[i];
        uint64_t t = strtoull(s, &s, 0);
        CHECK(*s == ':');
        uint64_t sm = strtoull(s + 1, NULL, 0);
        uint64_t h = NVN(nvnDeviceGetTextureHandle, 0, t, sm);
        uint64_t st = NVN(nvnDeviceGetSeparateTextureHandle, 0, t), ss = NVN(nvnDeviceGetSeparateSamplerHandle, 0, sm);
        uint64_t fetch = NVN(nvnDeviceGetTexelFetchHandle, 0, t), image = NVN(nvnDeviceGetImageHandle, 0, t);
        CHECK(nvn_handle_texture_id(h) == t && nvn_handle_sampler_id(h) == sm && (st | ss) == h);
        printf("handle %llu %llu %#llx %#llx %#llx %#llx %#llx\n", (ull)t, (ull)sm, (ull)h, (ull)st, (ull)ss,
               (ull)fetch, (ull)image);
    }
    printf("unpack %#x %d\n", (1u << NVN_HANDLE_TEXTURE_BITS) - 1, NVN_HANDLE_TEXTURE_BITS);
    printf("ok handles\n");
}

static void case_handle_range(const char *which) {
    setup();
    if (!strcmp(which, "sampler")) NVN(nvnDeviceGetTextureHandle, 0, 0, 4096);
    if (!strcmp(which, "texture")) NVN(nvnDeviceGetTextureHandle, 0, 1u << 20, 0);
    if (!strcmp(which, "image")) NVN(nvnDeviceGetImageHandle, 0, (uint64_t)-1);
    printf("no stop for %s\n", which);
}

// ---- samplers
static void case_sampler(void) {
    setup();
    uint64_t sb = galloc(0x60, 16), s = galloc(0x30, 16), t = galloc(0x30, 16), out = galloc(16, 16);
    NVN(nvnSamplerBuilderSetDevice, sb, 0x5678);
    NVN(nvnSamplerBuilderSetDefaults, sb);
    CHECK(NVN(nvnSamplerBuilderGetDevice, sb) == 0x5678);
    // OpenGL's defaults
    NVN(nvnSamplerBuilderGetMinMagFilter, sb, out, out + 4);
    CHECK(!memcmp(gp(out), (const int32_t[]){NVN_MIN_FILTER_NEAREST_MIPMAP_LINEAR, NVN_MAG_FILTER_LINEAR}, 8));
    NVN(nvnSamplerBuilderGetWrapMode, sb, out, out + 4, out + 8);
    CHECK(!memcmp(gp(out), (const int32_t[]){1, 1, 1}, 12));
    NVN(nvnSamplerBuilderGetLodClamp, sb, out, out + 4);
    CHECK(!memcmp(gp(out), (const float[]){-1000.f, 1000.f}, 8));
    NVN(nvnSamplerBuilderGetCompare, sb, out, out + 4);
    CHECK(!memcmp(gp(out), (const int32_t[]){NVN_COMPARE_MODE_NONE, NVN_COMPARE_FUNC_LESS}, 8));
    NVN(nvnSamplerBuilderGetMaxAnisotropy, sb);
    CHECK(ret_float() == 1.f);
    // Integer border colours keep their 32 bits, past what a float holds.
    memcpy(gp(out), (const uint32_t[]){0xffffffffu, 0x80000001u, 16777217u, 0}, 16);
    NVN(nvnSamplerBuilderSetBorderColorui, sb, out);
    memset(gp(out), 0, 16);
    NVN(nvnSamplerBuilderGetBorderColorui, sb, out);
    CHECK(!memcmp(gp(out), (const uint32_t[]){0xffffffffu, 0x80000001u, 16777217u, 0}, 16));
    NVNF(nvnSamplerBuilderSetLodSnap, ((const float[]){0.75f}), 1, sb);
    NVN(nvnSamplerBuilderGetLodSnap, sb);
    CHECK(ret_float() == 0.75f);
    NVNF(nvnSamplerBuilderSetLodBias, ((const float[]){2.5f}), 1, sb);
    NVN(nvnSamplerBuilderGetLodBias, sb);
    CHECK(ret_float() == 2.5f);
    NVN(nvnSamplerBuilderSetReductionFilter, sb, NVN_SAMPLER_REDUCTION_MIN);
    CHECK(NVN(nvnSamplerBuilderGetReductionFilter, sb) == NVN_SAMPLER_REDUCTION_MIN);

    // The sampler holds the builder's state; two samplers compare by state.
    CHECK((NVN(nvnSamplerInitialize, s, sb) & 0xff) == 1 && (NVN(nvnSamplerInitialize, t, sb) & 0xff) == 1);
    CHECK(NVN(nvnSamplerCompare, s, t) == 1);
    NVN(nvnSamplerGetBorderColori, s, out);
    CHECK(!memcmp(gp(out), (const uint32_t[]){0xffffffffu, 0x80000001u, 16777217u, 0}, 16));
    NVN(nvnSamplerGetLodBias, s);
    CHECK(ret_float() == 2.5f);
    CHECK(NVN(nvnSamplerGetReductionFilter, s) == NVN_SAMPLER_REDUCTION_MIN);
    NVN(nvnSamplerBuilderSetWrapMode, sb, NVN_WRAP_MODE_CLAMP, NVN_WRAP_MODE_REPEAT, NVN_WRAP_MODE_CLAMP_TO_EDGE);
    CHECK((NVN(nvnSamplerInitialize, t, sb) & 0xff) == 1 && NVN(nvnSamplerCompare, s, t) == 0);
    NVN(nvnSamplerGetWrapMode, t, out, out + 4, out + 8);
    CHECK(!memcmp(gp(out), (const int32_t[]){0, 1, 7}, 12));
    NVN(nvnSamplerFinalize, s);
    printf("ok sampler\n");
}

// A value past NVN's last wrap mode stops by name.
static void case_sampler_bad(void) {
    setup();
    uint64_t sb = galloc(0x60, 16);
    NVN(nvnSamplerBuilderSetDefaults, sb);
    NVN(nvnSamplerBuilderSetWrapMode, sb, 0, 8, 0);
    printf("no stop\n");
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: nvn_tex_test <case> [arguments]\n");
        return 2;
    }
    const char *cs = argv[1], *arg = argc > 2 ? argv[2] : "";
    if (!strcmp(cs, "layout")) case_layout(argc - 2, argv + 2);
    else if (!strcmp(cs, "formats")) case_formats();
    else if (!strcmp(cs, "unknown_format")) case_unknown_format(arg);
    else if (!strcmp(cs, "texture")) case_texture();
    else if (!strcmp(cs, "copies")) case_copies();
    else if (!strcmp(cs, "register")) case_register();
    else if (!strcmp(cs, "register_bad")) case_register_bad(arg);
    else if (!strcmp(cs, "handles")) case_handles(argc - 2, argv + 2);
    else if (!strcmp(cs, "handle_range")) case_handle_range(arg);
    else if (!strcmp(cs, "sampler")) case_sampler();
    else if (!strcmp(cs, "sampler_bad")) case_sampler_bad();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
