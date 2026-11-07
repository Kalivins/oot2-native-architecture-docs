// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Programs (architecture D10, D16, D21): ProgramInitialize, Finalize, SetShaders and SetShadersExt, and NVN2's program
// builder. A program keeps, per stage, the GPU address of its code area and the stage's identity (oot2/nvn_program.h).
// Nothing is translated and no pipeline is made here: a back end does that from the identity.
//
// The identity is the key the shader step of prepare gives the same stage (tools/shaders/split.py), so a back end looks
// a stage up in prepare's cache with no second scheme: nxnative's dm_ctl_sha16, the first 16 hex digits of SHA-256 over
// the control section's first 0x48 bytes (+0x30 hashed as zero: some containers hold the section's own length there,
// others 0) followed by the code area, then a dot and the first 8 hex digits of SHA-256 over the canonical JSON of the
// job fields that set control variants of one key16 apart (the stage, a compute stage's thread group and shared memory,
// a task or mesh stage's stage memory and mesh facts). Both come from the same bytes as split.py's ctl_key, stage_of,
// build_job and JobSpec.variant_hash take them; tests/test_runtime_nvn_program.py runs both on the same synthetic bytes.
//
// Stage kind and sizes come from the control section (D16): its stage byte (+0x56: 5 compute, 6 task, 7 mesh), the
// code area's length (+0x44), the instructions' length (+0x38), the thread group and the stage memory fields. A task or
// mesh stage's program header (SPH) reads as a vertex or domain shader, and classifying by it is the likely cause of the
// oracle's 64 failed mesh programs. Only a stage whose stage byte names none of those three takes its kind from the SPH
// type at the start of its code area, as split.py and nxnative do: vertex and pixel stages.
//
// The code area is read through the pool index (nvn_pool_find): a pool the CPU can see holds it at its GPU address,
// which is its guest address. A stage whose code lies in no pool or in a pool the CPU cannot see, and a stage prepare
// makes no job for (geometry or tessellation, a length outside the code area, an empty thread group: split.py's
// JobError cases), stop as a named fatal with the stage's shape, since no prepared module could ever answer it.
//
// The control section is read only as far as the key and the job need: 0x57 bytes (through the stage byte), 0x6C for
// a task or mesh stage, 0xBC for a compute stage. Those bytes are copied into the stage's record, which is made once per
// distinct key and kept for the life of the process, so a program copied by value never points at a freed record. When
// two stages share a key, the first one's control bytes are kept, as split.py keeps the first record of a job.
//
// Ported from oot-pc-recomp runtime/nvn_api_obj.c at 8e09aaa (GPL-3.0-or-later; ported there from MacNativeGames
// games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c at b75be8cf, MIT, Copyright (c) 2026 jhaizhou; origin d2r-mobile;
// licenses/dkb-xlate-MIT.txt): the program entry points, NVN2's program builder and NVNshaderDataExt. The key, the
// lengths in the control section and the stage rule are adapted from nxnative dm/gfx/nvn/hle_nvn.c (keys_hash,
// nvn_ctl_lens) at 865cc35 and from tools/shaders/split.py, itself adapted from nxnative dm/gfx/tools/shader_split.py,
// dmsh_meshstage.py and dmsh_build.py (MIT, licenses/nxnative-MIT.txt). Changed here: no NVN2-to-NVN1 lowering and no
// control sections copied per program (a stage keeps its identity instead), the stage kind from the control section,
// code the CPU cannot read is a named fatal, and the trace lines are gone.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nvn_api.h"
#include "oot2/loader.h"
#include "oot2/nvn_program.h"
#include "plat.h"

#define PROG_MAGIC NVN_PROG_MAGIC
#define PROG_BUILDER_MAGIC MAGIC('2', 'N', 'B', 'P')
#define MAX_STAGES 6

// The NVN2 control section, as split.py reads it.
#define CTL_MAGIC 0x98761234u
#define CTL_VERSION 0xC9u
#define KEY_CTL 0x48           // the control bytes in key16
#define CTL_HEAD 0x57          // through the stage byte at +0x56
#define CTL_MESH 0x6C          // a task or mesh stage's fields
#define SPH_BYTES 0x80         // the shader program header that vertex, pixel, task and mesh code starts with
#define MAX_CODE (1u << 24)    // nxnative's bound on a code area

typedef unsigned long long ull;

static uint32_t rd32(const uint8_t *p, uint32_t off) {
    uint32_t v;
    memcpy(&v, p + off, 4);
    return v;
}
static uint16_t rd16(const uint8_t *p, uint32_t off) {
    uint16_t v;
    memcpy(&v, p + off, 2);
    return v;
}

static const char *const KIND_NAME[NVN_STAGE_KINDS] = {"vs", "ps", "cs", "ts", "ms"};
const char *nvn_stage_kind_name(int kind) { return kind >= 0 && kind < NVN_STAGE_KINDS ? KIND_NAME[kind] : NULL; }

// ---- the stage's kind and job fields (split.py stage_of, build_job, JobSpec.variant_hash)

// The kind: the control section's stage byte for compute, task and mesh; else the SPH type in the code area's first
// word. -1 when no job form exists, with *other naming a geometry or tessellation stage ("hs", "ds", "gs") or NULL.
static int stage_kind(const uint8_t *ctl, const uint8_t *code, uint32_t code_size, const char **other) {
    *other = NULL;
    switch (ctl[0x56]) {
    case 5: return NVN_STAGE_CS;
    case 6: return NVN_STAGE_TS;
    case 7: return NVN_STAGE_MS;
    }
    if (code_size < 4) return -1;
    uint32_t word = rd32(code, 0), type = word >> 10 & 0xF;
    if (type == 0 && (word & 0x1F) == 1) return NVN_STAGE_VS;
    switch (type) {
    case 1: return NVN_STAGE_VS;
    case 5: return NVN_STAGE_PS;
    case 2: *other = "hs"; break;
    case 3: *other = "ds"; break;
    case 4: *other = "gs"; break;
    }
    return -1;
}

typedef struct { char s[512]; size_t n; } Text;
static void put(Text *t, const char *fmt, ...) __attribute__((format(printf, 2, 3)));
static void put(Text *t, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    int k = vsnprintf(t->s + t->n, sizeof t->s - t->n, fmt, ap);
    va_end(ap);
    if (k < 0 || (size_t)k >= sizeof t->s - t->n) rt_fatal("nvn: a stage's job fields do not fit their text");
    t->n += (size_t)k;
}
static const char *flag(int on) { return on ? "true" : "false"; }

// Whether instructions in [start, end) use shared memory of their own: an Ampere opcode (low 12 bits of the first
// word) of LDS, STS, ATOMS or their variants. A task or mesh stage keeps its stage memory there.
static int own_shared_memory(const uint8_t *code, uint32_t start, uint32_t end) {
    static const uint16_t ops[] = {0x984, 0x388, 0x988, 0x38c, 0x98c, 0xf8c};
    for (uint32_t o = start; o + 16 <= end; o += 16)
        for (size_t i = 0; i < sizeof ops / sizeof *ops; i++)
            if ((rd16(code, o) & 0xFFF) == ops[i]) return 1;
    return 0;
}

// The output components an SPH enables: generic attributes 0 to 31 (a nibble each from byte 0x36), and the position
// (byte 0x35, bits 4 to 7) when with_position is set.
static uint32_t sph_components(const uint8_t *sph, int with_position) {
    uint32_t n = with_position ? (uint32_t)__builtin_popcount(sph[0x35] >> 4) : 0;
    for (int i = 0; i < 16; i++) n += (uint32_t)__builtin_popcount(sph[0x36 + i]);
    return n;
}

// The job fields that set control variants of one key16 apart, as JSON with sorted keys and no spaces; NULL, or the
// reason prepare makes no job for the stage (split.py's JobError). ctl holds at least the bytes the kind reads.
static const char *job_text(int kind, const uint8_t *ctl, const uint8_t *code, uint32_t code_size, Text *t) {
    uint32_t inst = rd32(ctl, 0x38);
    if (!inst || inst % 16 || inst > code_size)
        return "the instruction length is not a positive multiple of 16 inside the code area";
    if (kind == NVN_STAGE_VS || kind == NVN_STAGE_PS) {
        put(t, "{\"stage\":\"%s\"}", KIND_NAME[kind]);
        return NULL;
    }
    if (kind == NVN_STAGE_CS) {
        uint32_t l[3] = {rd32(ctl, 0x58), rd32(ctl, 0x5C), rd32(ctl, 0x60)};
        if (!l[0] || !l[1] || !l[2] || (uint64_t)l[0] * l[1] > 1024 || (uint64_t)l[0] * l[1] * l[2] > 1024)
            return "the thread group is outside 1 to 1024 threads";
        put(t, "{\"local\":[%u,%u,%u],\"nvn2_cs\":[%u,%u,%u,%u,%u,%u,%u],\"shared\":%llu,\"stage\":\"cs\"}", l[0], l[1],
            l[2], (unsigned)ctl[0xB1], (unsigned)ctl[0xB2], (unsigned)ctl[0xB3], (unsigned)rd16(ctl, 0xBA),
            ctl[0xA9] * 4u, (unsigned)ctl[0xB4], (unsigned)ctl[0xB5],
            (ull)(((uint64_t)rd32(ctl, 0x64) + 15) / 16 * 16));
        return NULL;
    }
    if (inst <= SPH_BYTES) return "no instructions follow the program header";
    uint32_t threads = rd16(ctl, 0x58), payload = rd16(ctl, 0x5A), max_v = 0, max_p = 0, nv = 0, np = 0;
    if (!threads) return "the workgroup has no threads";
    if (kind == NVN_STAGE_MS) {
        max_v = rd16(ctl, 0x60), max_p = rd16(ctl, 0x62);
        nv = sph_components(code, 1);
        uint32_t second = ctl[0x68] & 0x80 ? rd32(ctl, 0x64) : 0;   // the per-primitive SPH
        if (second) {
            if ((uint64_t)second + SPH_BYTES > code_size)
                return "the per-primitive program header lies outside the code area";
            np = sph_components(code + second, 0);
        }
    }
    int own = own_shared_memory(code, SPH_BYTES, inst);
    // Stage memory: the workgroup number, the input the task wrote, the outputs, the primitive bytes (dmsh_meshstage.py)
    uint64_t out = 0x10 + ((payload + 15) & ~15u), mem[4] = {0, 0x10, out, out + payload};
    uint64_t size = out + ((payload + 3) & ~3u);
    if (kind == NVN_STAGE_MS) {
        uint64_t most = max_v > max_p ? max_v : max_p, blocks = ((most ? most : 1) + 31) / 32;
        mem[3] = out + blocks * ((nv + np) ? (nv + np) : 1) * 32 * 4;
        size = mem[3] + ((4 + 3 * (uint64_t)(max_p ? max_p : 1) + 3) & ~3ull);
    }
    put(t, "{\"local\":[%u,1,1],\"mesh\":{", threads);
    if (kind == NVN_STAGE_MS)
        put(t,
            "\"has_task\":%s,\"kind\":\"ms\",\"max_primitives\":%u,\"max_vertices\":%u,\"own_shared\":%s,"
            "\"payload\":%u,\"primitive_components\":%u,\"threads\":%u,\"vertex_components\":%u",
            flag(ctl[0x5E] != 0), max_p, max_v, flag(own), payload, np, threads, nv);
    else
        put(t, "\"kind\":\"ts\",\"own_shared\":%s,\"payload\":%u,\"threads\":%u", flag(own), payload, threads);
    put(t, "},\"shared\":%llu,\"stage\":\"%s\",\"stage_memory\":[%llu,%llu,%llu,%llu]}", (ull)size, KIND_NAME[kind],
        (ull)mem[0], (ull)mem[1], (ull)mem[2], (ull)mem[3]);
    return NULL;
}

static void hex(char *out, const uint8_t *b, int n) {
    static const char digit[] = "0123456789abcdef";
    for (int i = 0; i < n; i++) out[2 * i] = digit[b[i] >> 4], out[2 * i + 1] = digit[b[i] & 15];
}

// ---- stage records, one per distinct key, never freed (an open-addressing table under one lock)

static PlatLock g_lock = PLAT_LOCK_INIT;
static nvn_stage **g_table;
static uint32_t g_cap, g_count;

static uint32_t key_slot(const char *key, uint32_t mask) {
    uint32_t h = 2166136261u;   // FNV-1a
    for (; *key; key++) h = (h ^ (uint8_t)*key) * 16777619u;
    return h & mask;
}

static void grow(void) {
    uint32_t cap = g_cap ? 2 * g_cap : 256;
    nvn_stage **t = calloc(cap, sizeof *t);
    if (!t) rt_fatal("nvn: out of memory for %u stage records", cap);
    for (uint32_t i = 0; i < g_cap; i++)
        if (g_table[i]) {
            uint32_t j = key_slot(g_table[i]->key, cap - 1);
            while (t[j]) j = (j + 1) & (cap - 1);
            t[j] = g_table[i];
        }
    free(g_table);
    g_table = t, g_cap = cap;
}

static const nvn_stage *intern(const nvn_stage *rec) {
    plat_lock(&g_lock);
    if (2 * (g_count + 1) > g_cap) grow();
    uint32_t i = key_slot(rec->key, g_cap - 1);
    for (; g_table[i]; i = (i + 1) & (g_cap - 1))
        if (!strcmp(g_table[i]->key, rec->key)) {
            const nvn_stage *s = g_table[i];
            plat_unlock(&g_lock);
            return s;
        }
    nvn_stage *s = malloc(sizeof *s);
    if (!s) rt_fatal("nvn: out of memory for a stage record");
    *s = *rec;
    g_table[i] = s, g_count++;
    plat_unlock(&g_lock);
    return s;
}

int nvn_stage_records(void) {
    plat_lock_shared(&g_lock);
    int n = (int)g_count;
    plat_unlock_shared(&g_lock);
    return n;
}

// The identity of stage i of a call to fn, from the control section and the code area its NVNshaderData names.
static const nvn_stage *stage_identity(Cpu *c, const char *fn, int i, const NVNshaderData *sd) {
    const uint8_t *ctl = sd->control;
    uint64_t at = sd->data;
    if (!ctl) rt_unimplemented(c, "nvn", "%s: stage %d has no control section (code %#llx)", fn, i, (ull)at);
    if (rd32(ctl, 0) != CTL_MAGIC || rd32(ctl, 4) != CTL_VERSION)
        rt_unimplemented(c, "nvn", "%s: stage %d's control section at %p is not NVN2's (magic %#x, version %#x)", fn,
                         i, (const void *)ctl, rd32(ctl, 0), rd32(ctl, 4));
    uint32_t code_size = rd32(ctl, 0x44);
    if (!code_size || code_size >= MAX_CODE)
        rt_unimplemented(c, "nvn", "%s: stage %d's control section gives a code area of %#x bytes", fn, i, code_size);
    const pool_t *p = nvn_pool_find(at, code_size);
    if (!p)
        rt_unimplemented(c, "nvn", "%s: stage %d's code %#llx+%#x lies inside no memory pool the GPU can reach", fn, i,
                         (ull)at, code_size);
    if (gfx_gpu_only((uintptr_t)p->mem))
        rt_unimplemented(c, "nvn",
                         "%s: stage %d's code %#llx+%#x lies in a pool the CPU cannot see (flags %#x), so its key "
                         "cannot be read",
                         fn, i, (ull)at, code_size, p->flags);
    const uint8_t *code = (const uint8_t *)(uintptr_t)at;
    const char *other;
    int kind = stage_kind(ctl, code, code_size, &other);
    if (kind < 0)
        rt_unimplemented(c, "nvn",
                         "%s: stage %d is %s%s, which prepare makes no translator job for (control +0x56 %u, SPH word "
                         "%#x, code %#llx+%#x)",
                         fn, i, other ? "a stage of kind " : "of no known kind", other ? other : "", (unsigned)ctl[0x56],
                         code_size >= 4 ? rd32(code, 0) : 0, (ull)at, code_size);
    nvn_stage rec = {.kind = (uint8_t)kind, .code_size = code_size, .inst_size = rd32(ctl, 0x38)};
    rec.ctl_size = kind == NVN_STAGE_CS ? NVN_STAGE_CTL_MAX : kind == NVN_STAGE_TS || kind == NVN_STAGE_MS ? CTL_MESH
                                                                                                           : CTL_HEAD;
    memcpy(rec.ctl, ctl, rec.ctl_size);
    Text t = {.n = 0};
    const char *bad = job_text(kind, rec.ctl, code, code_size, &t);
    if (bad)
        rt_unimplemented(c, "nvn", "%s: stage %d (%s, code %#llx+%#x) has no translator job: %s", fn, i,
                         KIND_NAME[kind], (ull)at, code_size, bad);
    uint8_t *bytes = malloc(KEY_CTL + (size_t)code_size), d[32], v[32];
    if (!bytes) rt_fatal("nvn: out of memory to key a code area of %#x bytes", code_size);
    memcpy(bytes, rec.ctl, KEY_CTL);
    memset(bytes + 0x30, 0, 4);
    memcpy(bytes + KEY_CTL, code, code_size);
    rt_sha256(bytes, KEY_CTL + (uint64_t)code_size, d);
    free(bytes);
    rt_sha256(t.s, t.n, v);
    hex(rec.key, d, 8);
    rec.key[16] = '.';
    hex(rec.key + 17, v, 4);
    return intern(&rec);
}

// ---- programs

static uint64_t g_serial;

static void prog_release(prog_t *o) {
    if (o->serial) gfx_backend()->program_release(o);
    o->serial = 0;
}

// Every stage's identity first (a stage that has none stops the run before the program changes), then the old stages
// go and the new ones take a new serial.
static int set_shaders(Cpu *c, const char *fn, prog_t *o, int n, const NVNshaderData *sd) {
    if (n < 0 || n > MAX_STAGES || (n && !sd)) {
        rt_log("nvn: %s with %d stages at %p", fn, n, (const void *)sd);
        return 0;
    }
    if (o->magic != PROG_MAGIC) rt_fatal("nvn: %s on %p, which ProgramInitialize did not initialise", fn, (void *)o);
    prog_stage_t next[MAX_STAGES] = {{0, NULL}};
    unsigned kinds = 0;
    for (int i = 0; i < n; i++) {
        next[i] = (prog_stage_t){sd[i].data, stage_identity(c, fn, i, &sd[i])};
        unsigned bit = 1u << next[i].id->kind;
        if (kinds & bit)
            rt_unimplemented(c, "nvn", "%s: stage %d is a second %s stage", fn, i, KIND_NAME[next[i].id->kind]);
        kinds |= bit;
    }
    if ((kinds & 1u << NVN_STAGE_CS) && kinds != 1u << NVN_STAGE_CS)
        rt_unimplemented(c, "nvn", "%s: a compute stage beside graphics stages (%d stages)", fn, n);
    prog_release(o);
    o->nstage = (uint32_t)n;
    memcpy(o->st, next, sizeof next);
    o->serial = __atomic_add_fetch(&g_serial, 1, __ATOMIC_RELAXED);
    return 1;
}

// NVNboolean nvnProgramInitialize(NVNprogram *p, NVNdevice *d): a program with no stages
NVN_API(nvnProgramInitialize) {
    prog_t *o = arg_p(c, 0);
    memset(o, 0, sizeof *o);
    o->magic = PROG_MAGIC;
    ret_x(c, 1);
}
// void nvnProgramFinalize(NVNprogram *p)
NVN_API(nvnProgramFinalize) {
    prog_t *o = arg_p(c, 0);
    prog_release(o);
    o->magic = 0;
}
// void nvnProgramSetDebugLabel(NVNprogram *p, const char *label): for NVN's debugging tools only
NVN_API(nvnProgramSetDebugLabel) { (void)c; }
// NVNboolean nvnProgramSetShaders(NVNprogram *p, int n, const NVNshaderData *sd)
NVN_API(nvnProgramSetShaders) {
    ret_x(c, set_shaders(c, "ProgramSetShaders", arg_p(c, 0), (int)arg_x(c, 1), arg_p(c, 2)));
}

// NVNboolean nvnProgramSetShadersExt(NVNprogram *p, int n, const NVNshaderDataExt *sd): no prototype in nvn.h; the
// entries are {data, control, reserved}, 0x18 bytes, reserved seen 0 (oot-pc-recomp).
typedef struct { NVNbufferAddress data; const void *control; uint64_t reserved; } shader_data_ext;
_Static_assert(sizeof(shader_data_ext) == 0x18, "NVNshaderDataExt stride");
NVN_API(nvnProgramSetShadersExt) {
    int n = (int)arg_x(c, 1);
    const shader_data_ext *sd = arg_p(c, 2);
    if (n < 0 || n > MAX_STAGES || (n && !sd)) {
        rt_log("nvn: ProgramSetShadersExt with %d stages at %p", n, (const void *)sd);
        ret_x(c, 0);
        return;
    }
    NVNshaderData plain[MAX_STAGES];
    for (int i = 0; i < n; i++) {
        if (sd[i].reserved)
            rt_unimplemented(c, "nvn", "ProgramSetShadersExt: stage %d has the unknown field %#llx", i,
                             (ull)sd[i].reserved);
        plain[i] = (NVNshaderData){sd[i].data, sd[i].control};
    }
    ret_x(c, set_shaders(c, "ProgramSetShadersExt", arg_p(c, 0), n, plain));
}

// ---- NVN2's program builder. The game builds on its stack: SetDefaults, SetDevice, SetShaders(b, n, sd), then
// nvnProgramInitializeFromBuilder (oot-pc-recomp, for Donkey Kong Bananza, and the same in this game, which it runs).
// The builder keeps its own copy of the stage list; the control sections must stay readable until the program is made.
typedef struct { uint32_t magic, nstage; NVNdevice *dev; NVNshaderData sd[MAX_STAGES]; } prog_builder_t;
_Static_assert(sizeof(prog_builder_t) <= 0x80, "NVNprogramBuilder is 0x80 bytes (Donkey Kong Bananza)");

// void nvnProgramBuilderSetDefaults(NVNprogramBuilder *b): keeps the device, as the other builders do
NVN_API(nvnProgramBuilderSetDefaults) {
    prog_builder_t *b = arg_p(c, 0);
    NVNdevice *dev = b->dev;
    memset(b, 0, sizeof *b);
    b->magic = PROG_BUILDER_MAGIC;
    b->dev = dev;
}
// void nvnProgramBuilderSetDevice(NVNprogramBuilder *b, NVNdevice *d), NVNdevice *...GetDevice(const ... *b)
NVN_API(nvnProgramBuilderSetDevice) { ((prog_builder_t *)arg_p(c, 0))->dev = arg_p(c, 1); }
NVN_API(nvnProgramBuilderGetDevice) { ret_x(c, (uintptr_t)((const prog_builder_t *)arg_p(c, 0))->dev); }
// void nvnProgramBuilderSetShaders(NVNprogramBuilder *b, int n, const NVNshaderData *sd)
NVN_API(nvnProgramBuilderSetShaders) {
    prog_builder_t *b = arg_p(c, 0);
    int n = (int)arg_x(c, 1);
    const NVNshaderData *sd = arg_p(c, 2);
    if (n < 0 || n > MAX_STAGES || (n && !sd))
        rt_fatal("nvn: ProgramBuilderSetShaders with %d stages at %p", n, (const void *)sd);
    b->nstage = (uint32_t)n;
    if (n) memcpy(b->sd, sd, (size_t)n * sizeof *sd);
}
// int nvnProgramBuilderGetShaderCount(const NVNprogramBuilder *b)
NVN_API(nvnProgramBuilderGetShaderCount) { ret_x(c, ((const prog_builder_t *)arg_p(c, 0))->nstage); }
// NVNboolean nvnProgramInitializeFromBuilder(NVNprogram *p, const NVNprogramBuilder *b): Initialize, then SetShaders
NVN_API(nvnProgramInitializeFromBuilder) {
    prog_t *o = arg_p(c, 0);
    const prog_builder_t *b = arg_p(c, 1);
    if (!o || !b || b->magic != PROG_BUILDER_MAGIC)
        rt_fatal("nvn: ProgramInitializeFromBuilder(%p, %p) without ProgramBuilderSetDefaults", (void *)o,
                 (const void *)b);
    memset(o, 0, sizeof *o);
    o->magic = PROG_MAGIC;
    ret_x(c, set_shaders(c, "ProgramInitializeFromBuilder", o, (int)b->nstage, b->sd));
}
