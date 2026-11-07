// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the NVN2 front end's programs and state objects (architecture D10, D16), driven by
// tests/test_runtime_nvn_program.py: `nvn_program_test <case> [arguments]`. Each case runs in its own process. A
// passing case prints "ok <case>"; a failing case must end in one named fatal, which the driver checks.
//
// Entry points are called the way generated code calls them: rt_call at their synthetic address, arguments in the
// AAPCS64 registers, the objects in guest memory (the heap of rt_mem_init). Control sections and code areas are
// synthetic bytes, written by the driver or assembled here; nothing is read from a dump. The harness repeats the one of
// tests/runtime/nvn_test.c, with a back end that counts the programs released.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/gfx.h"
#include "oot2/memory.h"
#include "oot2/nvn_program.h"
#include "plat.h"

#define CHECK(cond)                                                                    \
    do {                                                                               \
        if (!(cond)) {                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);        \
            exit(1);                                                                   \
        }                                                                              \
    } while (0)

#define MiB (1ull << 20)
#define HEAP (16 * MiB)
#define CODE_POOL (4 * MiB)

// NVNmemoryPoolFlags
#define CPU_NO_ACCESS 0x1u
#define CPU_UNCACHED 0x2u
#define GPU_CACHED 0x20u

static Cpu c;
static uint64_t heap, heap_used;
static uint64_t loader;

static void fake_fn(Cpu *cpu) { (void)cpu; }

// ---- the back end: the null one, counting the programs released and the newest serial
static struct { int releases; uint64_t serial; } rec;
static void rec_program_release(const prog_t *p) { rec.releases++, rec.serial = p->serial; }
static GfxBackend recording;

static void setup(void) {
    RtMemConfig cfg = {.pool_size = 256 * MiB, .heap_region_size = 64 * MiB, .alias_region_size = 1024 * MiB,
                       .stack_region_size = 64 * MiB, .system_resource_size = MiB};
    rt_mem_init(&cfg);
    c.x[0] = 0, c.x[1] = HEAP;
    rt_svc(&c, 0x01, 0);   // SetHeapSize
    CHECK(c.x[0] == 0);
    heap = c.x[1];
    loader = rt_hostfn_bind("nvnBootstrapLoader");
    CHECK(loader != 0);
    recording = gfx_null_backend;
    recording.name = "recording";
    recording.program_release = rec_program_release;
    gfx_set_backend(&recording);
    nvn_procs_register();
    static const uint64_t addrs[] = {MOD_MAIN_BASE + 0x100};
    static const GuestFn fns[] = {fake_fn};
    static const RtModule mods[] = {{"main", MOD_MAIN_BASE, MOD_MAIN_SIZE, addrs, fns, 1, TEST_END}};
    rt_dispatch_init(mods, 1);
    c.excl_addr = ~0ull;
}

// Guest memory, zeroed, as the game's allocator would give it.
static uint64_t galloc(uint64_t size, uint64_t align) {
    uint64_t a = (heap + heap_used + align - 1) & ~(align - 1);
    CHECK(a + size <= heap + HEAP);
    heap_used = a + size - heap;
    memset((void *)(uintptr_t)a, 0, size);
    return a;
}

static uint64_t guest_str(const char *s) {
    size_t n = strlen(s) + 1;
    uint64_t a = galloc(n, 16);
    memcpy((void *)(uintptr_t)a, s, n);
    return a;
}

static uint64_t call(uint64_t fn, uint64_t x0, uint64_t x1) {
    c.x[0] = x0, c.x[1] = x1;
    rt_call(&c, fn);
    return c.x[0];
}

static uint64_t proc(const char *name) {
    static struct { const char *name; uint64_t addr; } cache[160];
    static int n;
    static uint64_t gpa;
    for (int i = 0; i < n; i++)
        if (!strcmp(cache[i].name, name)) return cache[i].addr;
    if (!gpa) gpa = call(loader, guest_str("nvnDeviceGetProcAddress"), 0);
    CHECK(gpa != 0 && n < 160);
    uint64_t a = call(gpa, 0, guest_str(name));
    CHECK(a != 0);
    cache[n].name = name, cache[n].addr = a;
    return cache[n++].addr;
}

// NVN(nvnX, arguments...): call entry point nvnX with up to eight integer or pointer arguments, as x0 to x7.
static uint64_t nvn_vcall(uint64_t fn, const uint64_t *a, size_t n) {
    for (size_t i = 0; i < 8; i++) c.x[i] = i < n ? a[i] : 0;
    rt_call(&c, fn);
    return c.x[0];
}
#define NVN(name, ...)                                                                                               \
    nvn_vcall(proc(#name), (const uint64_t[]){__VA_ARGS__}, sizeof((const uint64_t[]){__VA_ARGS__}) / sizeof(uint64_t))

static uint64_t device(void) {
    uint64_t b = galloc(0x40, 16), d = galloc(0x3000, 16);
    NVN(nvnDeviceBuilderSetDefaults, b);
    CHECK((NVN(nvnDeviceInitialize, d, b) & 0xff) == 1);
    return d;
}

static uint64_t pool(uint64_t storage, uint64_t size, uint32_t flags) {
    uint64_t b = galloc(0x40, 16), o = galloc(0x100, 16);
    NVN(nvnMemoryPoolBuilderSetDefaults, b);
    NVN(nvnMemoryPoolBuilderSetStorage, b, storage, size);
    NVN(nvnMemoryPoolBuilderSetFlags, b, flags);
    CHECK((NVN(nvnMemoryPoolInitialize, o, b) & 0xff) == 1);
    return o;
}

// ---- programs

// The code pool: CPU-visible, so a code area's GPU address is its guest address. Code areas are 256-byte aligned in it.
static uint64_t code_base, code_used;
static uint64_t place_code(const uint8_t *code, uint32_t size) {
    uint64_t a = (code_base + code_used + 255) & ~255ull;
    CHECK(a + size <= code_base + CODE_POOL);
    code_used = a + size - code_base;
    memcpy((void *)(uintptr_t)a, code, size);
    return a;
}
static uint64_t place_ctl(const uint8_t *ctl, uint32_t size) {
    uint64_t a = galloc(size, 16);
    memcpy((void *)(uintptr_t)a, ctl, size);
    return a;
}

static uint64_t g_dev;
static uint64_t program(void) {
    uint64_t o = galloc(0xC0, 16);
    CHECK((NVN(nvnProgramInitialize, o, g_dev) & 0xff) == 1);
    return o;
}
// NVNshaderData[n] in guest memory: {code address, control section}
static uint64_t shader_data(int n, const uint64_t *code, const uint64_t *ctl) {
    uint64_t sd = galloc(16 * (uint64_t)(n ? n : 1), 16);
    for (int i = 0; i < n; i++) {
        memcpy((void *)(uintptr_t)(sd + 16 * i), &code[i], 8);
        memcpy((void *)(uintptr_t)(sd + 16 * i + 8), &ctl[i], 8);
    }
    return sd;
}

static int hexval(char ch) { return ch >= '0' && ch <= '9' ? ch - '0' : ch >= 'a' && ch <= 'f' ? ch - 'a' + 10 : -1; }
// Hex digits from *p up to a byte in stop (or NUL) into a new buffer; *p moves past them.
static uint8_t *unhex(const char **p, const char *stop, uint32_t *size) {
    const char *s = *p;
    size_t n = strcspn(s, stop);
    CHECK(n % 2 == 0 && n > 0);
    uint8_t *out = malloc(n / 2);
    CHECK(out != NULL);
    for (size_t i = 0; i < n / 2; i++) {
        int hi = hexval(s[2 * i]), lo = hexval(s[2 * i + 1]);
        CHECK(hi >= 0 && lo >= 0);
        out[i] = (uint8_t)(hi << 4 | lo);
    }
    *size = (uint32_t)(n / 2), *p = s + n;
    return out;
}

typedef struct { int n; uint8_t *ctl[6], *code[6]; uint32_t ctl_size[6], code_size[6]; uint64_t ctl_at[6], code_at[6]; } Prog;

// Each line of the file is a program: up to six stages "<control hex>:<code hex>", separated by spaces.
static int read_programs(const char *path, Prog **out) {
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);
    char *text = malloc((size_t)size + 1);
    CHECK(text && fread(text, 1, (size_t)size, f) == (size_t)size);
    fclose(f);
    text[size] = 0;
    int cap = 16, n = 0;
    Prog *progs = malloc(sizeof *progs * (size_t)cap);
    for (const char *p = text; *p;) {
        if (*p == '\n' || *p == '\r') { p++; continue; }
        if (n == cap) progs = realloc(progs, sizeof *progs * (size_t)(cap *= 2));
        CHECK(progs != NULL);
        Prog *g = &progs[n++];
        memset(g, 0, sizeof *g);
        while (*p && *p != '\n' && *p != '\r') {
            CHECK(g->n < 6);
            g->ctl[g->n] = unhex(&p, ":", &g->ctl_size[g->n]);
            CHECK(*p == ':');
            p++;
            g->code[g->n] = unhex(&p, " \r\n", &g->code_size[g->n]);
            g->n++;
            if (*p == ' ') p++;
        }
    }
    free(text);
    *out = progs;
    return n;
}

// The stages of g set on program o: NVNshaderData through SetShaders (path 0), the builder (path 1) or
// NVNshaderDataExt through SetShadersExt (path 2). Returns SetShaders' answer.
static uint64_t set_stages(uint64_t o, const Prog *g, int path) {
    if (path == 1) {
        uint64_t b = galloc(0x80, 16);
        memset((void *)(uintptr_t)b, 0xCC, 0x80);   // the game's stack
        NVN(nvnProgramBuilderSetDevice, b, g_dev);
        NVN(nvnProgramBuilderSetDefaults, b);
        CHECK(NVN(nvnProgramBuilderGetDevice, b) == g_dev && NVN(nvnProgramBuilderGetShaderCount, b) == 0);
        NVN(nvnProgramBuilderSetShaders, b, (uint64_t)g->n, shader_data(g->n, g->code_at, g->ctl_at));
        CHECK(NVN(nvnProgramBuilderGetShaderCount, b) == (uint64_t)g->n);
        return NVN(nvnProgramInitializeFromBuilder, o, b) & 0xff;
    }
    if (path == 2) {
        uint64_t sd = galloc(0x18 * (uint64_t)(g->n ? g->n : 1), 16);
        for (int i = 0; i < g->n; i++) {
            memcpy((void *)(uintptr_t)(sd + 0x18 * i), &g->code_at[i], 8);
            memcpy((void *)(uintptr_t)(sd + 0x18 * i + 8), &g->ctl_at[i], 8);
        }
        return NVN(nvnProgramSetShadersExt, o, (uint64_t)g->n, sd) & 0xff;
    }
    return NVN(nvnProgramSetShaders, o, (uint64_t)g->n, shader_data(g->n, g->code_at, g->ctl_at)) & 0xff;
}

// The program's stages are the ones g names: their code addresses, the kinds and sizes, the control bytes copied.
static void check_program(uint64_t o, const Prog *g, uint64_t after_serial) {
    const prog_t *p = (const prog_t *)(uintptr_t)o;
    CHECK(p->nstage == (uint32_t)g->n && p->serial > after_serial);
    for (int i = 0; i < 6; i++) {
        const nvn_stage *s = nvn_program_stage(p, i);
        if (i >= g->n) {
            CHECK(s == NULL && p->st[i].code == 0 && p->st[i].id == NULL);
            continue;
        }
        CHECK(s != NULL && p->st[i].code == g->code_at[i]);
        CHECK(s->code_size == g->code_size[i] && s->ctl_size <= g->ctl_size[i]);
        CHECK(!memcmp(s->ctl, g->ctl[i], s->ctl_size) && strlen(s->key) == 25 && s->key[16] == '.');
        CHECK(nvn_stage_kind_name(s->kind) != NULL);
    }
}

// Every program of the file set through one of the three paths in turn; each stage printed as
// "stage <program> <index> <kind> <key>" for the driver to compare with split.py. Then: a stage set again from other
// addresses gives the same record, the records outlive the control sections, and serials go to the back end.
static void case_keys(const char *path) {
    setup();
    g_dev = device();
    code_base = galloc(CODE_POOL, 0x10000);
    pool(code_base, CODE_POOL, CPU_UNCACHED | GPU_CACHED);
    Prog *progs;
    int n = read_programs(path, &progs);
    uint64_t serial = 0, *objs = malloc(sizeof *objs * (size_t)n);
    CHECK(objs != NULL);
    for (int k = 0; k < n; k++) {
        Prog *g = &progs[k];
        for (int i = 0; i < g->n; i++) {
            g->code_at[i] = place_code(g->code[i], g->code_size[i]);
            g->ctl_at[i] = place_ctl(g->ctl[i], g->ctl_size[i]);
        }
        objs[k] = program();
        CHECK(set_stages(objs[k], g, k % 3) == 1);
        check_program(objs[k], g, serial);
        const prog_t *p = (const prog_t *)(uintptr_t)objs[k];
        serial = p->serial;
        for (int i = 0; i < g->n; i++) {
            const nvn_stage *s = nvn_program_stage(p, i);
            printf("stage %d %d %s %s\n", k, i, nvn_stage_kind_name(s->kind), s->key);
        }
    }
    CHECK(rec.releases == 0);
    int records = nvn_stage_records();
    printf("records %d\n", records);

    // The first program's stages from new addresses: the same records, a new serial.
    Prog again = progs[0];
    for (int i = 0; i < again.n; i++) {
        again.code_at[i] = place_code(again.code[i], again.code_size[i]);
        again.ctl_at[i] = place_ctl(again.ctl[i], again.ctl_size[i]);
    }
    uint64_t o = program();
    CHECK(set_stages(o, &again, 0) == 1);
    check_program(o, &again, serial);
    for (int i = 0; i < again.n; i++)
        CHECK(nvn_program_stage((const prog_t *)(uintptr_t)o, i) ==
              nvn_program_stage((const prog_t *)(uintptr_t)objs[0], i));
    CHECK(nvn_stage_records() == records);

    // The game frees the control sections once SetShaders returns: the records keep their copies.
    for (int i = 0; i < progs[0].n; i++) memset((void *)(uintptr_t)progs[0].ctl_at[i], 0xFF, progs[0].ctl_size[i]);
    check_program(objs[0], &progs[0], 0);

    // Setting stages again releases the old serial; Finalize releases the newest once; nothing is left to release.
    uint64_t first = ((const prog_t *)(uintptr_t)o)->serial;
    CHECK(set_stages(o, &again, 0) == 1);
    CHECK(rec.releases == 1 && rec.serial == first);
    uint64_t second = ((const prog_t *)(uintptr_t)o)->serial;
    CHECK(second > first);
    NVN(nvnProgramFinalize, o);
    CHECK(rec.releases == 2 && rec.serial == second);
    NVN(nvnProgramFinalize, o);
    CHECK(rec.releases == 2);

    // No stages is a program; a stage count past six, or stages without an array, is refused.
    o = program();
    CHECK((NVN(nvnProgramSetShaders, o, 0, 0) & 0xff) == 1 && ((const prog_t *)(uintptr_t)o)->nstage == 0);
    CHECK((NVN(nvnProgramSetShaders, o, 7, shader_data(1, progs[0].code_at, progs[0].ctl_at)) & 0xff) == 0);
    CHECK((NVN(nvnProgramSetShaders, o, 1, 0) & 0xff) == 0);
    CHECK((NVN(nvnProgramSetShadersExt, o, (uint64_t)-1, 0) & 0xff) == 0);
    NVN(nvnProgramSetDebugLabel, o, guest_str("program"));
    printf("ok keys %d\n", n);
}

// ---- one synthetic vertex stage for the failing cases: an NVN2 control section and a code area that starts with a
// vertex SPH, 0x90 bytes of instructions in 0x200
static uint8_t f_ctl[0xE0], f_code[0x200];
static void vertex_stage(void) {
    memset(f_ctl, 0, sizeof f_ctl);
    memset(f_code, 0, sizeof f_code);
    uint32_t words[][2] = {{0, 0x98761234u}, {4, 0xC9}, {0x38, 0x90}, {0x40, 0x100}, {0x44, sizeof f_code}};
    for (size_t i = 0; i < sizeof words / sizeof *words; i++) memcpy(f_ctl + words[i][0], &words[i][1], 4);
    uint32_t sph = 1u << 10 | 1;
    memcpy(f_code, &sph, 4);
}
static uint64_t set_one(uint64_t code_at) {
    uint64_t ctl = place_ctl(f_ctl, sizeof f_ctl);
    return NVN(nvnProgramSetShaders, program(), 1, shader_data(1, &code_at, &ctl));
}
static uint64_t cpu_code(void) {
    code_base = galloc(CODE_POOL, 0x10000);
    pool(code_base, CODE_POOL, CPU_UNCACHED | GPU_CACHED);
    return place_code(f_code, sizeof f_code);
}

static void case_code_out_of_reach(void) {   // code in a pool the CPU cannot see
    setup();
    g_dev = device();
    vertex_stage();
    uint64_t p = pool(galloc(0x10000, 0x10000), 0x10000, CPU_NO_ACCESS | GPU_CACHED);
    set_one(NVN(nvnMemoryPoolGetBufferAddress, p) + 0x100);
}
static void case_code_in_no_pool(void) {
    setup();
    g_dev = device();
    vertex_stage();
    uint64_t a = galloc(0x400, 256);
    memcpy((void *)(uintptr_t)a, f_code, sizeof f_code);
    set_one(a);
}
static void case_geometry_stage(void) {   // an SPH of type 4
    setup();
    g_dev = device();
    vertex_stage();
    uint32_t sph = 4u << 10 | 1;
    memcpy(f_code, &sph, 4);
    set_one(cpu_code());
}
static void case_no_job(void) {   // an instruction length that is not a multiple of 16
    setup();
    g_dev = device();
    vertex_stage();
    uint32_t inst = 0x88;
    memcpy(f_ctl + 0x38, &inst, 4);
    set_one(cpu_code());
}
static void case_two_vertex_stages(void) {
    setup();
    g_dev = device();
    vertex_stage();
    uint64_t code[2], ctl[2];
    code[0] = cpu_code();   // makes the code pool, so before the second placement
    code[1] = place_code(f_code, sizeof f_code);
    ctl[0] = place_ctl(f_ctl, sizeof f_ctl), ctl[1] = place_ctl(f_ctl, sizeof f_ctl);
    NVN(nvnProgramSetShaders, program(), 2, shader_data(2, code, ctl));
}
static void case_ext_field(void) {   // NVNshaderDataExt with its third field set
    setup();
    g_dev = device();
    vertex_stage();
    uint64_t sd = galloc(0x18, 16), code = cpu_code(), ctl = place_ctl(f_ctl, sizeof f_ctl), one = 1;
    memcpy((void *)(uintptr_t)sd, &code, 8), memcpy((void *)(uintptr_t)(sd + 8), &ctl, 8);
    memcpy((void *)(uintptr_t)(sd + 16), &one, 8);
    NVN(nvnProgramSetShadersExt, program(), 1, sd);
}

// ---- state objects

static uint64_t rng = 0x9E3779B97F4A7C15ull;
static uint64_t rnd(void) {
    rng ^= rng << 13, rng ^= rng >> 7, rng ^= rng << 17;
    return rng;
}
static uint32_t pick(uint32_t lo, uint32_t hi) {   // lo or hi a quarter of the time each, else between
    uint64_t r = rnd();
    return r % 4 == 0 ? lo : r % 4 == 1 ? hi : lo + (uint32_t)((r >> 8) % ((uint64_t)hi - lo + 1));
}
// An int or enum argument with junk above its 32 bits; a boolean with junk above its 8 (true when its byte is not 0).
static uint64_t a32(uint32_t v) { return rnd() << 32 | v; }
static uint64_t abool(int *truth) {
    uint64_t r = rnd();
    uint8_t b = r % 3 == 0 ? 0 : (uint8_t)(1 + (r >> 8) % 255);
    *truth = b != 0;
    return (rnd() & ~0xffull) | b;
}

#define GUARD 16
static uint64_t g_obj, g_size, g_out;
// A state object of the SDK's size with guard bytes after it, filled with junk as the game's memory would be.
static uint64_t state_object(uint64_t size) {
    g_size = size;
    g_obj = galloc(size + GUARD, 16);
    memset((void *)(uintptr_t)g_obj, 0xCC, size);
    memset((void *)(uintptr_t)(g_obj + size), 0xA5, GUARD);
    if (!g_out) g_out = galloc(64, 16);
    return g_obj;
}
static void check_guard(void) {
    for (uint64_t i = 0; i < GUARD; i++) CHECK(((const uint8_t *)(uintptr_t)(g_obj + g_size))[i] == 0xA5);
}
// Output slots for getters that write through pointers: slot k at g_out + 8k, filled with 0xEE first; read() takes
// width bytes of slot k and checks that the getter wrote nothing past them.
static uint64_t slot(int k) { return g_out + 8 * (uint64_t)k; }
static void clear_slots(void) { memset((void *)(uintptr_t)g_out, 0xEE, 64); }
static uint64_t read_slot(int k, int width) {
    const uint8_t *p = (const uint8_t *)(uintptr_t)slot(k);
    for (int i = width; i < 8; i++) CHECK(p[i] == 0xEE);
    uint64_t v = 0;
    memcpy(&v, p, (size_t)width);
    return v;
}
#define LOW32(x) ((uint32_t)(x))
#define LOW8(x) ((uint8_t)(x))

// Blend state
typedef struct { uint32_t target, f[4], eq[2], adv, ovl, pm, nd; } Blend;
static void blend_check(uint64_t s, const Blend *m) {
    CHECK(LOW32(NVN(nvnBlendStateGetBlendTarget, s)) == m->target);
    clear_slots();
    NVN(nvnBlendStateGetBlendFunc, s, slot(0), slot(1), slot(2), slot(3));
    for (int i = 0; i < 4; i++) CHECK(read_slot(i, 4) == m->f[i]);
    clear_slots();
    NVN(nvnBlendStateGetBlendEquation, s, slot(0), slot(1));
    CHECK(read_slot(0, 4) == m->eq[0] && read_slot(1, 4) == m->eq[1]);
    CHECK(LOW32(NVN(nvnBlendStateGetAdvancedMode, s)) == m->adv);
    CHECK(LOW32(NVN(nvnBlendStateGetAdvancedOverlap, s)) == m->ovl);
    CHECK(LOW8(NVN(nvnBlendStateGetAdvancedPremultipliedSrc, s)) == m->pm);
    CHECK(LOW8(NVN(nvnBlendStateGetAdvancedNormalizedDst, s)) == m->nd);
    check_guard();
}
static void blend_set(uint64_t s, Blend *m, int op) {
    int t;
    switch (op) {
    case 0: m->target = pick(0, 7), NVN(nvnBlendStateSetBlendTarget, s, a32(m->target)); break;
    case 1:
        for (int i = 0; i < 4; i++) m->f[i] = pick(0, NVN_BLEND_FUNC_ONE_MINUS_CONSTANT_ALPHA);
        NVN(nvnBlendStateSetBlendFunc, s, a32(m->f[0]), a32(m->f[1]), a32(m->f[2]), a32(m->f[3]));
        break;
    case 2:
        m->eq[0] = pick(0, NVN_BLEND_EQUATION_MAX), m->eq[1] = pick(0, NVN_BLEND_EQUATION_MAX);
        NVN(nvnBlendStateSetBlendEquation, s, a32(m->eq[0]), a32(m->eq[1]));
        break;
    case 3: m->adv = pick(0, NVN_BLEND_ADVANCED_MODE_HSL_LUMINOSITY), NVN(nvnBlendStateSetAdvancedMode, s, a32(m->adv)); break;
    case 4: m->ovl = pick(0, NVN_BLEND_ADVANCED_OVERLAP_CONJOINT), NVN(nvnBlendStateSetAdvancedOverlap, s, a32(m->ovl)); break;
    case 5: NVN(nvnBlendStateSetAdvancedPremultipliedSrc, s, abool(&t)), m->pm = (uint32_t)t; break;
    default: NVN(nvnBlendStateSetAdvancedNormalizedDst, s, abool(&t)), m->nd = (uint32_t)t; break;
    }
}

// Color state
typedef struct { uint32_t en, logic, alpha; } Color;
static void color_check(uint64_t s, const Color *m) {
    for (uint32_t i = 0; i < 10; i++)
        CHECK(LOW8(NVN(nvnColorStateGetBlendEnable, s, a32(i))) == (i < 8 ? m->en >> i & 1 : 0));
    CHECK(LOW32(NVN(nvnColorStateGetLogicOp, s)) == m->logic && LOW32(NVN(nvnColorStateGetAlphaTest, s)) == m->alpha);
    check_guard();
}
static void color_set(uint64_t s, Color *m, int op) {
    int t;
    switch (op) {
    case 0: {
        uint32_t i = pick(0, 9);   // 8 and 9 are past the targets: ignored
        NVN(nvnColorStateSetBlendEnable, s, a32(i), abool(&t));
        if (i < 8) m->en = t ? m->en | 1u << i : m->en & ~(1u << i);
        break;
    }
    case 1: m->logic = pick(0, NVN_LOGIC_OP_SET), NVN(nvnColorStateSetLogicOp, s, a32(m->logic)); break;
    default: m->alpha = pick(0, NVN_ALPHA_FUNC_ALWAYS), NVN(nvnColorStateSetAlphaTest, s, a32(m->alpha)); break;
    }
}

// Channel mask state
typedef struct { uint32_t mask; } Mask;
static void mask_check(uint64_t s, const Mask *m) {
    for (uint32_t i = 0; i < 10; i++) {
        clear_slots();
        NVN(nvnChannelMaskStateGetChannelMask, s, a32(i), slot(0), slot(1), slot(2), slot(3));
        for (int ch = 0; ch < 4; ch++) CHECK(read_slot(ch, 1) == (i < 8 ? m->mask >> (4 * i + ch) & 1 : 0));
    }
    check_guard();
}
static void mask_set(uint64_t s, Mask *m, int op) {
    (void)op;
    uint32_t i = pick(0, 9);
    int t[4];
    uint64_t a[4];
    for (int ch = 0; ch < 4; ch++) a[ch] = abool(&t[ch]);
    NVN(nvnChannelMaskStateSetChannelMask, s, a32(i), a[0], a[1], a[2], a[3]);
    if (i < 8)
        for (int ch = 0; ch < 4; ch++) m->mask = t[ch] ? m->mask | 1u << (4 * i + ch) : m->mask & ~(1u << (4 * i + ch));
}

// Multisample state
typedef struct { uint32_t enable, samples, a2c, dither, raster, mod, trunc, c2c, c2c_out, sl, grid; } Ms;
static void ms_check(uint64_t s, const Ms *m) {
    CHECK(LOW8(NVN(nvnMultisampleStateGetMultisampleEnable, s)) == m->enable);
    CHECK(LOW32(NVN(nvnMultisampleStateGetSamples, s)) == m->samples);
    CHECK(LOW8(NVN(nvnMultisampleStateGetAlphaToCoverageEnable, s)) == m->a2c);
    CHECK(LOW8(NVN(nvnMultisampleStateGetAlphaToCoverageDither, s)) == m->dither);
    CHECK(LOW32(NVN(nvnMultisampleStateGetRasterSamples, s)) == m->raster);
    CHECK(LOW32(NVN(nvnMultisampleStateGetCoverageModulationMode, s)) == m->mod);
    CHECK(LOW8(NVN(nvnMultisampleStateGetCoverageTruncationEnable, s)) == m->trunc);
    CHECK(LOW8(NVN(nvnMultisampleStateGetCoverageToColorEnable, s)) == m->c2c);
    CHECK(LOW32(NVN(nvnMultisampleStateGetCoverageToColorOutput, s)) == m->c2c_out);
    CHECK(LOW8(NVN(nvnMultisampleStateGetSampleLocationsEnable, s)) == m->sl);
    CHECK(LOW8(NVN(nvnMultisampleStateGetSampleLocationsGridEnable, s)) == m->grid);
    check_guard();
}
static void ms_set(uint64_t s, Ms *m, int op) {
    int t;
    switch (op) {
    case 0: NVN(nvnMultisampleStateSetMultisampleEnable, s, abool(&t)), m->enable = (uint32_t)t; break;
    case 1: m->samples = pick(0, 16), NVN(nvnMultisampleStateSetSamples, s, a32(m->samples)); break;
    case 2: NVN(nvnMultisampleStateSetAlphaToCoverageEnable, s, abool(&t)), m->a2c = (uint32_t)t; break;
    case 3: NVN(nvnMultisampleStateSetAlphaToCoverageDither, s, abool(&t)), m->dither = (uint32_t)t; break;
    case 4: m->raster = pick(0, 16), NVN(nvnMultisampleStateSetRasterSamples, s, a32(m->raster)); break;
    case 5:
        m->mod = pick(0, NVN_COVERAGE_MODULATION_MODE_RGBA);
        NVN(nvnMultisampleStateSetCoverageModulationMode, s, a32(m->mod));
        break;
    case 6: NVN(nvnMultisampleStateSetCoverageTruncationEnable, s, abool(&t)), m->trunc = (uint32_t)t; break;
    case 7: NVN(nvnMultisampleStateSetCoverageToColorEnable, s, abool(&t)), m->c2c = (uint32_t)t; break;
    case 8: m->c2c_out = pick(0, 7), NVN(nvnMultisampleStateSetCoverageToColorOutput, s, a32(m->c2c_out)); break;
    case 9: NVN(nvnMultisampleStateSetSampleLocationsEnable, s, abool(&t)), m->sl = (uint32_t)t; break;
    default: NVN(nvnMultisampleStateSetSampleLocationsGridEnable, s, abool(&t)), m->grid = (uint32_t)t; break;
    }
}

// Polygon state
typedef struct { uint32_t cull, front, mode, offset; } Poly;
static void poly_check(uint64_t s, const Poly *m) {
    CHECK(LOW32(NVN(nvnPolygonStateGetCullFace, s)) == m->cull && LOW32(NVN(nvnPolygonStateGetFrontFace, s)) == m->front);
    CHECK(LOW32(NVN(nvnPolygonStateGetPolygonMode, s)) == m->mode);
    CHECK(LOW32(NVN(nvnPolygonStateGetPolygonOffsetEnables, s)) == m->offset);
    check_guard();
}
static void poly_set(uint64_t s, Poly *m, int op) {
    switch (op) {
    case 0: m->cull = pick(0, NVN_FACE_FRONT_AND_BACK), NVN(nvnPolygonStateSetCullFace, s, a32(m->cull)); break;
    case 1: m->front = pick(0, NVN_FRONT_FACE_CCW), NVN(nvnPolygonStateSetFrontFace, s, a32(m->front)); break;
    case 2: m->mode = pick(0, NVN_POLYGON_MODE_FILL), NVN(nvnPolygonStateSetPolygonMode, s, a32(m->mode)); break;
    default: m->offset = pick(0, 7), NVN(nvnPolygonStateSetPolygonOffsetEnables, s, a32(m->offset)); break;
    }
}

// Depth-stencil state: per face [front, back] a stencil function and the fail, depth-fail and pass ops
typedef struct { uint32_t dt, dw, df, st, sf[2], op[2][3]; } Ds;
static void ds_check(uint64_t s, const Ds *m) {
    CHECK(LOW8(NVN(nvnDepthStencilStateGetDepthTestEnable, s)) == m->dt);
    CHECK(LOW8(NVN(nvnDepthStencilStateGetDepthWriteEnable, s)) == m->dw);
    CHECK(LOW32(NVN(nvnDepthStencilStateGetDepthFunc, s)) == m->df);
    CHECK(LOW8(NVN(nvnDepthStencilStateGetStencilTestEnable, s)) == m->st);
    uint32_t faces[3] = {NVN_FACE_FRONT, NVN_FACE_BACK, NVN_FACE_FRONT_AND_BACK};
    for (int f = 0; f < 3; f++) {
        int b = faces[f] == NVN_FACE_BACK;   // FRONT_AND_BACK reads the front face
        CHECK(LOW32(NVN(nvnDepthStencilStateGetStencilFunc, s, a32(faces[f]))) == m->sf[b]);
        clear_slots();
        NVN(nvnDepthStencilStateGetStencilOp, s, a32(faces[f]), slot(0), slot(1), slot(2));
        for (int k = 0; k < 3; k++) CHECK(read_slot(k, 4) == m->op[b][k]);
    }
    check_guard();
}
static void ds_set(uint64_t s, Ds *m, int op) {
    int t;
    uint32_t face = pick(0, NVN_FACE_FRONT_AND_BACK);
    switch (op) {
    case 0: NVN(nvnDepthStencilStateSetDepthTestEnable, s, abool(&t)), m->dt = (uint32_t)t; break;
    case 1: NVN(nvnDepthStencilStateSetDepthWriteEnable, s, abool(&t)), m->dw = (uint32_t)t; break;
    case 2: m->df = pick(0, NVN_DEPTH_FUNC_ALWAYS), NVN(nvnDepthStencilStateSetDepthFunc, s, a32(m->df)); break;
    case 3: NVN(nvnDepthStencilStateSetStencilTestEnable, s, abool(&t)), m->st = (uint32_t)t; break;
    case 4: {
        uint32_t fn = pick(0, NVN_STENCIL_FUNC_ALWAYS);
        NVN(nvnDepthStencilStateSetStencilFunc, s, a32(face), a32(fn));
        for (int b = 0; b < 2; b++)
            if (face & (b ? NVN_FACE_BACK : NVN_FACE_FRONT)) m->sf[b] = fn;
        break;
    }
    default: {
        uint32_t v[3];
        for (int k = 0; k < 3; k++) v[k] = pick(0, NVN_STENCIL_OP_DECR_WRAP);
        NVN(nvnDepthStencilStateSetStencilOp, s, a32(face), a32(v[0]), a32(v[1]), a32(v[2]));
        for (int b = 0; b < 2; b++)
            if (face & (b ? NVN_FACE_BACK : NVN_FACE_FRONT)) memcpy(m->op[b], v, sizeof v);
        break;
    }
    }
}

// Vertex attribute state: every format the packed state holds (nvn.h's last is 0x9C), offsets below 2^19, 32 streams
typedef struct { uint32_t fmt, offset, stream; } Va;
static void va_check(uint64_t s, const Va *m) {
    clear_slots();
    NVN(nvnVertexAttribStateGetFormat, s, slot(0), slot(1));
    CHECK(read_slot(0, 4) == m->fmt && read_slot(1, 8) == m->offset);
    CHECK(LOW32(NVN(nvnVertexAttribStateGetStreamIndex, s)) == m->stream);
    check_guard();
}
static void va_set(uint64_t s, Va *m, int op) {
    if (op == 0) {
        m->fmt = pick(0, 0xff), m->offset = pick(0, (1u << 19) - 1);
        NVN(nvnVertexAttribStateSetFormat, s, a32(m->fmt), m->offset);
    } else {
        m->stream = pick(0, 31), NVN(nvnVertexAttribStateSetStreamIndex, s, a32(m->stream));
    }
}

// Vertex stream state
typedef struct { uint32_t stride, divisor; } Vs;
static void vs_check(uint64_t s, const Vs *m) {
    CHECK(NVN(nvnVertexStreamStateGetStride, s) == m->stride);
    CHECK(LOW32(NVN(nvnVertexStreamStateGetDivisor, s)) == m->divisor);
    check_guard();
}
static void vs_set(uint64_t s, Vs *m, int op) {
    if (op == 0) m->stride = pick(0, 0xffffffffu), NVN(nvnVertexStreamStateSetStride, s, m->stride);
    else m->divisor = pick(0, 0xffffffffu), NVN(nvnVertexStreamStateSetDivisor, s, a32(m->divisor));
}

// One state object: SetDefaults over junk gives the defaults; every setter at its least and greatest values, then
// random setters, each followed by every getter against the model and the guard bytes.
#define ROUND_TRIP(T, sdk, defaults_proc, model, nops, set, check)                                                   \
    do {                                                                                                             \
        uint64_t s = state_object(sizeof(sdk));                                                                      \
        T m = model;                                                                                                 \
        NVN(defaults_proc, s);                                                                                       \
        check(s, &m);                                                                                                \
        for (int op = 0; op < (nops); op++)                                                                          \
            for (int k = 0; k < 8; k++) set(s, &m, op), check(s, &m);                                                \
        for (int k = 0; k < 4000; k++) set(s, &m, (int)(rnd() % (nops))), check(s, &m);                              \
        printf("state %s\n", #sdk);                                                                                  \
    } while (0)

static void case_states(void) {
    setup();
    ROUND_TRIP(Blend, NVNblendState, nvnBlendStateSetDefaults,
               ((Blend){0, {NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO, NVN_BLEND_FUNC_ONE, NVN_BLEND_FUNC_ZERO},
                        {NVN_BLEND_EQUATION_ADD, NVN_BLEND_EQUATION_ADD}, 0, 0, 1, 1}),
               7, blend_set, blend_check);
    ROUND_TRIP(Color, NVNcolorState, nvnColorStateSetDefaults, ((Color){0, NVN_LOGIC_OP_COPY, NVN_ALPHA_FUNC_ALWAYS}), 3,
               color_set, color_check);
    ROUND_TRIP(Mask, NVNchannelMaskState, nvnChannelMaskStateSetDefaults, ((Mask){0xffffffffu}), 1, mask_set,
               mask_check);
    ROUND_TRIP(Ms, NVNmultisampleState, nvnMultisampleStateSetDefaults, ((Ms){1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0}), 11,
               ms_set, ms_check);
    ROUND_TRIP(Poly, NVNpolygonState, nvnPolygonStateSetDefaults,
               ((Poly){NVN_FACE_NONE, NVN_FRONT_FACE_CCW, NVN_POLYGON_MODE_FILL, 0}), 4, poly_set, poly_check);
    ROUND_TRIP(Ds, NVNdepthStencilState, nvnDepthStencilStateSetDefaults,
               ((Ds){0, 1, NVN_DEPTH_FUNC_LESS, 0, {NVN_STENCIL_FUNC_ALWAYS, NVN_STENCIL_FUNC_ALWAYS},
                     {{NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP},
                      {NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP, NVN_STENCIL_OP_KEEP}}}),
               6, ds_set, ds_check);
    ROUND_TRIP(Va, NVNvertexAttribState, nvnVertexAttribStateSetDefaults, ((Va){0, 0, 0}), 2, va_set, va_check);
    ROUND_TRIP(Vs, NVNvertexStreamState, nvnVertexStreamStateSetDefaults, ((Vs){0, 0}), 2, vs_set, vs_check);
    printf("ok states\n");
}

// A vertex attribute past what the packed state holds stops with its values.
static void case_attribute_offset(void) {
    setup();
    uint64_t s = state_object(sizeof(NVNvertexAttribState));
    NVN(nvnVertexAttribStateSetDefaults, s);
    NVN(nvnVertexAttribStateSetFormat, s, 0x2a, 1u << 19);
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: nvn_program_test <case> [arguments]\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "keys") && argc > 2) case_keys(argv[2]);
    else if (!strcmp(cs, "states")) case_states();
    else if (!strcmp(cs, "code_out_of_reach")) case_code_out_of_reach();
    else if (!strcmp(cs, "code_in_no_pool")) case_code_in_no_pool();
    else if (!strcmp(cs, "geometry_stage")) case_geometry_stage();
    else if (!strcmp(cs, "no_job")) case_no_job();
    else if (!strcmp(cs, "two_vertex_stages")) case_two_vertex_stages();
    else if (!strcmp(cs, "ext_field")) case_ext_field();
    else if (!strcmp(cs, "attribute_offset")) case_attribute_offset();
    else {
        fprintf(stderr, "unknown case %s\n", cs);
        return 2;
    }
    return 0;
}
