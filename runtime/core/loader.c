// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The module loader (the loader half of architecture D9): plain modules at their layout bases, relocated and bound.
// Adapted from DKB Recomp runtime/loader.c at 9d8fa02 (GPL-3.0-or-later; adapted there from LAS Recomp and
// MacNativeGames, MIT): the NSO header, MOD0 and the dynamic section, the four relocation types and the DT_RELR walk,
// and the rule that relocations write only the data segment. Changed here: plain modules only, so the runtime holds no
// decompressor, checked against the segment hashes of their own header; a symbol binds through rt_hostfn_bind first,
// every definition is offered to it too, and a strong import defined nowhere binds to rt_hostfn_missing; bss is
// committed on first touch; the counts are kept per module for the report.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "core.h"
#include "oot2/loader.h"
#include "plat.h"

#define PAGE 0x1000ull
#define NSO_HEADER 0x100

enum { DT_NULL = 0, DT_PLTRELSZ = 2, DT_STRTAB = 5, DT_SYMTAB = 6, DT_RELA = 7, DT_RELASZ = 8, DT_STRSZ = 10,
       DT_INIT = 12, DT_FINI = 13, DT_JMPREL = 23, DT_INIT_ARRAY = 25, DT_FINI_ARRAY = 26, DT_INIT_ARRAYSZ = 27,
       DT_FINI_ARRAYSZ = 28, DT_RELRSZ = 35, DT_RELR = 36 };
enum { R_ABS64 = 257, R_GLOB_DAT = 1025, R_JUMP_SLOT = 1026, R_RELATIVE = 1027 };
enum { STB_GLOBAL = 1, STB_WEAK = 2, STT_FUNC = 2, SHN_ABS = 0xfff1 };
// How a symbol was bound: to a host function, to a module's definition (BOUND_MODULE + its index), weak to 0, missing.
enum { BOUND_NONE, BOUND_HOST, BOUND_WEAK, BOUND_MISSING, BOUND_MODULE };

typedef struct { uint32_t name; uint8_t info, other; uint16_t shndx; uint64_t value, size; } Sym;
typedef struct { uint64_t off, info; int64_t addend; } Rela;
_Static_assert(sizeof(Sym) == 24 && sizeof(Rela) == 24, "ELF64 layouts");

typedef struct Mod {
    RtLoaded pub;
    RtLoadStats st;
    const Sym *syms;
    uint32_t nsyms;
    const char *strtab;
    uint64_t strsz;
    const Rela *rela, *jmprel;
    uint64_t nrela, njmprel;
    const uint64_t *relr;
    uint64_t nrelr;
    uint32_t *hash, hash_mask;   // defined, exported symbols by name
    uint64_t *value;             // per symbol, once bound
    uint8_t *how;                // per symbol: BOUND_*
} Mod;

static Mod mods[RT_MAX_LOADED];
static int nmods;
static RtReplaced *replaced;
static int nreplaced, capreplaced;

static uint64_t align_up(uint64_t v, uint64_t a) { return (v + a - 1) & ~(a - 1); }
static uint32_t rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }

int rt_loaded_count(void) { return nmods; }
const RtLoaded *rt_loaded(int i) { return i >= 0 && i < nmods ? &mods[i].pub : NULL; }
const RtLoadStats *rt_load_stats(int i) { return i >= 0 && i < nmods ? &mods[i].st : NULL; }
int rt_load_replaced_count(void) { return nreplaced; }
const RtReplaced *rt_load_replaced(int i) { return i >= 0 && i < nreplaced ? &replaced[i] : NULL; }

// ---- the file
static void hex(const uint8_t *d, int n, char *out) {
    for (int i = 0; i < n; i++) snprintf(out + 2 * i, 3, "%02x", d[i]);
}

// Map the file, check it, commit the image up to the end of its data and copy the segments in. bss is left to the
// first touch.
static void place(Mod *m, const RtModuleFile *f) {
    uint64_t len = 0;
    const uint8_t *raw = plat_map_file(f->path, &len);
    if (!raw) rt_fatal("loader: cannot read module %s from %s", f->name, f->path);
    char got[65], want[65];
    if (f->sha256) {
        uint8_t d[32];
        rt_sha256(raw, len, d);
        hex(d, 32, got);
        if (strcmp(got, f->sha256))
            rt_fatal("loader: %s has SHA-256 %s, the build expects %s: prepare again from the same dump", f->path, got,
                     f->sha256);
    }
    if (len < NSO_HEADER || memcmp(raw, "NSO0", 4)) rt_fatal("loader: %s is not an NSO0 module", f->path);
    uint32_t flags = rd32(raw + 0xC);
    if (flags & 0x87)
        rt_fatal("loader: %s is compressed (flags %#x); the runtime loads plain modules, which prepare writes", f->path,
                 flags);
    static const char *const seg_name[3] = {"text", "rodata", "data"};
    uint32_t off[3], mem[3], size[3];
    for (int i = 0; i < 3; i++) {
        off[i] = rd32(raw + 0x10 + 0x10 * i);
        mem[i] = rd32(raw + 0x14 + 0x10 * i);
        size[i] = rd32(raw + 0x18 + 0x10 * i);
        if ((uint64_t)off[i] + size[i] > len) rt_fatal("loader: %s: the %s segment lies past the end of the file", f->path, seg_name[i]);
        if ((flags >> (3 + i)) & 1) {
            uint8_t d[32];
            rt_sha256(raw + off[i], size[i], d);
            if (memcmp(d, raw + 0xA0 + 32 * i, 32)) {
                hex(d, 32, got);
                hex(raw + 0xA0 + 32 * i, 32, want);
                rt_fatal("loader: %s: the %s segment has SHA-256 %s, its header says %s", f->path, seg_name[i], got, want);
            }
        }
    }
    uint32_t bss = rd32(raw + 0x3C);
    if (mem[0] || (mem[1] & (PAGE - 1)) || (mem[2] & (PAGE - 1)) || mem[1] < size[0] || mem[2] < (uint64_t)mem[1] + size[1])
        rt_fatal("loader: %s: text, rodata and data are not page-aligned and in order", f->path);
    uint64_t data_end = (uint64_t)mem[2] + size[2], image = align_up(data_end + bss, PAGE);
    if (image != f->size)
        rt_fatal("loader: %s is a %#llx-byte image, the layout says %#llx: the build and this module differ", f->path,
                 (unsigned long long)image, (unsigned long long)f->size);
    if (f->base & (PAGE - 1)) rt_fatal("loader: %s base %#llx is not page-aligned", f->name, (unsigned long long)f->base);

    mem_module_commit(f->base, align_up(data_end, PAGE));
    for (int i = 0; i < 3; i++) memcpy((void *)(uintptr_t)(f->base + mem[i]), raw + off[i], size[i]);
    m->pub = (RtLoaded){.name = f->name, .base = f->base, .size = image, .text_size = size[0], .rodata_off = mem[1],
                        .rodata_size = size[1], .data_off = mem[2], .data_size = size[2], .bss_off = data_end,
                        .bss_size = image - data_end};
    memcpy(m->pub.build_id, raw + 0x40, 32);
    // The dynamic symbol table's extent, relative to rodata, from the header; its count is not in the dynamic section.
    uint32_t dynsym_off = rd32(raw + 0x98), dynsym_size = rd32(raw + 0x9C);
    if ((uint64_t)dynsym_off + dynsym_size > size[1] || dynsym_size % sizeof(Sym))
        rt_fatal("loader: %s: the header's dynamic symbol table %#x+%#x lies outside rodata", f->path, dynsym_off,
                 dynsym_size);
    m->nsyms = dynsym_size / (uint32_t)sizeof(Sym);
    plat_unmap_file(raw, len);
}

// ---- MOD0 and the dynamic section, read from the placed image
static uint64_t in_image(const Mod *m, uint64_t off, uint64_t size, const char *what) {
    if (off > m->pub.size || size > m->pub.size - off)
        rt_fatal("loader: %s: %s at +%#llx, %#llx bytes, lies outside the image", m->pub.name, what,
                 (unsigned long long)off, (unsigned long long)size);
    return m->pub.base + off;
}

static void parse_dynamic(Mod *m) {
    uint64_t base = m->pub.base;
    uint64_t mod0 = base + LD32(base + 4);
    in_image(m, mod0 - base, 28, "MOD0");
    if (memcmp((const void *)(uintptr_t)mod0, "MOD0", 4)) rt_fatal("loader: %s has no MOD0 header", m->pub.name);
    uint64_t dyn = mod0 + (uint64_t)(int64_t)(int32_t)LD32(mod0 + 4);
    uint64_t symtab = 0, strtab = 0, rela = 0, relasz = 0, jmprel = 0, pltsz = 0, relr = 0, relrsz = 0;
    for (uint64_t d = in_image(m, dyn - base, 16, "the dynamic section");; d += 16) {
        in_image(m, d - base, 16, "the dynamic section");
        uint64_t tag = LD64(d), v = LD64(d + 8);
        if (tag == DT_NULL) break;
        switch (tag) {
        case DT_SYMTAB: symtab = v; break;
        case DT_STRTAB: strtab = v; break;
        case DT_STRSZ: m->strsz = v; break;
        case DT_RELA: rela = v; break;
        case DT_RELASZ: relasz = v; break;
        case DT_JMPREL: jmprel = v; break;
        case DT_PLTRELSZ: pltsz = v; break;
        case DT_RELR: relr = v; break;
        case DT_RELRSZ: relrsz = v; break;
        case DT_INIT: m->pub.init = base + v; break;
        case DT_FINI: m->pub.fini = base + v; break;
        case DT_INIT_ARRAY: m->pub.init_array = base + v; break;
        case DT_INIT_ARRAYSZ: m->pub.init_array_size = v; break;
        case DT_FINI_ARRAY: m->pub.fini_array = base + v; break;
        case DT_FINI_ARRAYSZ: m->pub.fini_array_size = v; break;
        }
    }
    if (!symtab || !strtab || !m->strsz) rt_fatal("loader: %s lacks DT_SYMTAB, DT_STRTAB or DT_STRSZ", m->pub.name);
    m->syms = (const Sym *)(uintptr_t)in_image(m, symtab, (uint64_t)m->nsyms * sizeof(Sym), "the symbol table");
    m->strtab = (const char *)(uintptr_t)in_image(m, strtab, m->strsz, "the string table");
    if (m->strtab[m->strsz - 1]) rt_fatal("loader: %s: the string table does not end with a NUL", m->pub.name);
    m->rela = (const Rela *)(uintptr_t)in_image(m, rela, relasz, "DT_RELA");
    m->nrela = relasz / sizeof(Rela);
    m->jmprel = (const Rela *)(uintptr_t)in_image(m, jmprel, pltsz, "DT_JMPREL");
    m->njmprel = pltsz / sizeof(Rela);
    m->relr = (const uint64_t *)(uintptr_t)in_image(m, relr, relrsz, "DT_RELR");
    m->nrelr = relrsz / 8;
}

// ---- symbols
static const char *sym_name(const Mod *m, uint32_t i) {
    uint32_t n = m->syms[i].name;
    if (n >= m->strsz) rt_fatal("loader: %s: symbol %u names string %#x, past the string table", m->pub.name, i, n);
    return m->strtab + n;
}

static int exported(const Mod *m, uint32_t i) {
    unsigned bind = m->syms[i].info >> 4;
    return m->syms[i].shndx != 0 && (bind == STB_GLOBAL || bind == STB_WEAK) && *sym_name(m, i);
}

static uint32_t fnv(const char *s) {
    uint32_t h = 2166136261u;
    for (; *s; s++) h = (h ^ (uint8_t)*s) * 16777619u;
    return h;
}

static uint64_t sym_value(const Mod *m, uint32_t i) {
    return m->syms[i].shndx == SHN_ABS ? m->syms[i].value : m->pub.base + m->syms[i].value;
}

static void make_hash(Mod *m) {
    uint32_t cap = 16;
    while (cap < 2 * m->nsyms) cap <<= 1;
    m->hash = calloc(cap, sizeof *m->hash);
    m->value = calloc(m->nsyms ? m->nsyms : 1, sizeof *m->value);
    m->how = calloc(m->nsyms ? m->nsyms : 1, 1);
    if (!m->hash || !m->value || !m->how) rt_fatal("loader: symbol tables of %s", m->pub.name);
    m->hash_mask = cap - 1;
    for (uint32_t i = 1; i < m->nsyms; i++) {
        if (!exported(m, i)) continue;
        const char *s = sym_name(m, i);
        uint32_t p = fnv(s) & m->hash_mask;
        while (m->hash[p] && strcmp(s, sym_name(m, m->hash[p]))) p = (p + 1) & m->hash_mask;
        if (!m->hash[p]) m->hash[p] = i;
    }
}

static uint64_t module_symbol(const Mod *m, const char *name) {
    for (uint32_t p = fnv(name) & m->hash_mask; m->hash[p]; p = (p + 1) & m->hash_mask)
        if (!strcmp(name, sym_name(m, m->hash[p]))) return sym_value(m, m->hash[p]);
    return 0;
}

static uint64_t find_symbol(const char *name, int *which) {
    for (int i = 0; i < nmods; i++) {
        uint64_t a = module_symbol(&mods[i], name);
        if (a) {
            if (which) *which = i;
            return a;
        }
    }
    return 0;
}

uint64_t rt_symbol(const char *name) { return find_symbol(name, NULL); }

// A HOST_OVERRIDE of the name, else the module's own definition, else the first module in load order that defines it,
// else 0 for a weak symbol, else the named fatal of a missing import. Once per symbol.
static uint64_t bind(Mod *m, uint32_t i) {
    if (m->how[i] != BOUND_NONE) return m->value[i];
    const char *name = sym_name(m, i);
    uint64_t v = *name ? rt_hostfn_bind(name) : 0;
    int which = (int)(m - mods);
    uint8_t how = BOUND_HOST;
    if (!v && m->syms[i].shndx != 0) v = sym_value(m, i), how = (uint8_t)(BOUND_MODULE + which);
    else if (!v && *name && (v = find_symbol(name, &which))) how = (uint8_t)(BOUND_MODULE + which);
    else if (!v && (m->syms[i].info >> 4) == STB_WEAK) how = BOUND_WEAK;
    else if (!v) {
        if (!*name) rt_fatal("loader: %s: a relocation names symbol %u, which is undefined and unnamed", m->pub.name, i);
        v = rt_hostfn_missing(name), how = BOUND_MISSING;
    }
    m->value[i] = v;
    m->how[i] = how;
    return v;
}

// Every function a module defines is offered to the host-function table: the sdk defines longjmp and the C++ throw
// entry points itself, and they must be replaced there too.
static void offer_definitions(Mod *m) {
    for (uint32_t i = 1; i < m->nsyms; i++) {
        if (!exported(m, i) || (m->syms[i].info & 0xF) != STT_FUNC) continue;
        uint64_t host = rt_hostfn_bind(sym_name(m, i));
        if (!host) continue;
        if (nreplaced == capreplaced) {
            capreplaced = capreplaced ? 2 * capreplaced : 16;
            replaced = realloc(replaced, (size_t)capreplaced * sizeof *replaced);
            if (!replaced) rt_fatal("loader: replaced definitions");
        }
        replaced[nreplaced++] = (RtReplaced){sym_name(m, i), sym_value(m, i), host};
        m->value[i] = host;
        m->how[i] = BOUND_HOST;
        m->st.replaced++;
    }
}

static void count_imports(Mod *m) {
    for (uint32_t i = 1; i < m->nsyms; i++) {
        if (m->syms[i].shndx != 0 || !*sym_name(m, i)) continue;
        bind(m, i);
        m->st.imports++;
        uint8_t how = m->how[i];
        if (how == BOUND_HOST) m->st.imports_to_host++;
        else if (how == BOUND_WEAK) m->st.imports_weak++;
        else if (how == BOUND_MISSING) m->st.imports_missing++;
        else m->st.imports_to[how - BOUND_MODULE]++;
    }
}

// ---- relocation
static uint64_t target(const Mod *m, uint64_t off, const char *what, uint64_t i) {
    if (off < m->pub.data_off || off > m->pub.size - 8)
        rt_fatal("loader: %s %s %llu writes +%#llx, outside the data segment", m->pub.name, what, (unsigned long long)i,
                 (unsigned long long)off);
    return m->pub.base + off;
}

static void relocate_table(Mod *m, const Rela *r, uint64_t n, const char *what) {
    for (uint64_t i = 0; i < n; i++) {
        uint32_t type = (uint32_t)r[i].info, si = (uint32_t)(r[i].info >> 32);
        uint64_t at = target(m, r[i].off, what, i), v;
        if (type == R_RELATIVE) {
            v = m->pub.base + (uint64_t)r[i].addend;
            m->st.relative++;
        } else if (type == R_ABS64 || type == R_GLOB_DAT || type == R_JUMP_SLOT) {
            if (si == 0 || si >= m->nsyms)
                rt_fatal("loader: %s %s %llu names symbol %u of %u", m->pub.name, what, (unsigned long long)i, si, m->nsyms);
            uint64_t s = bind(m, si);
            uint8_t how = m->how[si];
            if (r[i].addend && (how == BOUND_HOST || how == BOUND_MISSING))
                rt_fatal("loader: %s %s %llu adds %#llx to host function %s", m->pub.name, what, (unsigned long long)i,
                         (unsigned long long)r[i].addend, sym_name(m, si));
            v = s + (uint64_t)r[i].addend;
            if (type == R_ABS64) m->st.abs64++;
            else if (type == R_GLOB_DAT) m->st.glob_dat++;
            else m->st.jump_slot++;
        } else {
            rt_fatal("loader: %s %s %llu has relocation type %u, which the loader does not apply", m->pub.name, what,
                     (unsigned long long)i, type);
        }
        ST64(at, v);
    }
}

// DT_RELR: an even entry is an offset to relocate, and the next bitmap starts after it; an odd entry is a bitmap of the
// 63 words that follow. Each relocated word holds its addend, as R_AARCH64_RELATIVE with an implicit addend.
static void relocate_relr(Mod *m) {
    uint64_t base = m->pub.base, where = 0;
    for (uint64_t i = 0; i < m->nrelr; i++) {
        uint64_t e = m->relr[i];
        if (!(e & 1)) {
            uint64_t at = target(m, e, "DT_RELR entry", i);
            ST64(at, base + LD64(at));
            m->st.relr++;
            where = e + 8;
            continue;
        }
        for (int k = 0; k < 63; k++)
            if ((e >> (k + 1)) & 1) {
                uint64_t at = target(m, where + 8ull * (uint64_t)k, "DT_RELR entry", i);
                ST64(at, base + LD64(at));
                m->st.relr++;
            }
        where += 63 * 8;
    }
}

void rt_load_modules(const RtModuleFile *files, int count) {
    if (nmods) rt_fatal("rt_load_modules called twice");
    if (!rt_mem_ready()) rt_fatal("rt_load_modules before rt_mem_init");
    if (rt_sealed()) rt_fatal("rt_load_modules after rt_dispatch_init: imports bind through the host-function table");
    if (count <= 0 || count > RT_MAX_LOADED) rt_fatal("rt_load_modules: %d modules", count);
    for (int i = 0; i < count; i++) {
        place(&mods[i], &files[i]);
        nmods = i + 1;
        mem_module_add(&mods[i].pub);
        parse_dynamic(&mods[i]);
        make_hash(&mods[i]);
    }
    for (int i = 0; i < nmods; i++) offer_definitions(&mods[i]);
    for (int i = 0; i < nmods; i++) {
        Mod *m = &mods[i];
        count_imports(m);
        relocate_table(m, m->rela, m->nrela, "DT_RELA entry");
        relocate_table(m, m->jmprel, m->njmprel, "DT_JMPREL entry");
        relocate_relr(m);
    }
    for (int i = 0; i < nmods; i++) mem_module_seal(mods[i].pub.base, mods[i].pub.data_off);
}

void rt_load_report(void) {
    for (int i = 0; i < nmods; i++) {
        const Mod *m = &mods[i];
        const RtLoadStats *s = &m->st;
        char to[256] = "";
        size_t n = 0;
        for (int j = 0; j < nmods && n < sizeof to; j++)
            n += (size_t)snprintf(to + n, sizeof to - n, "%s%s %u", j ? ", " : "", mods[j].pub.name, s->imports_to[j]);
        rt_log("loaded %s at %#llx, %#llx bytes: relocations %llu RELATIVE, %llu RELR, %llu ABS64, %llu GLOB_DAT, "
               "%llu JUMP_SLOT; %u imports: %s, host %u, weak %u, missing %u; %u definitions replaced",
               m->pub.name, (unsigned long long)m->pub.base, (unsigned long long)m->pub.size,
               (unsigned long long)s->relative, (unsigned long long)s->relr, (unsigned long long)s->abs64,
               (unsigned long long)s->glob_dat, (unsigned long long)s->jump_slot, s->imports, to, s->imports_to_host,
               s->imports_weak, s->imports_missing, s->replaced);
        for (uint32_t k = 1; k < m->nsyms; k++)
            if (m->how[k] == BOUND_MISSING) rt_log("  %s: import %s is defined nowhere", m->pub.name, sym_name(m, k));
    }
    for (int i = 0; i < nreplaced; i++)
        rt_log("replaced %s at %#llx by host function %#llx", replaced[i].symbol, (unsigned long long)replaced[i].guest,
               (unsigned long long)replaced[i].host);
}
