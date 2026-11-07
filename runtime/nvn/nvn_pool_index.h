// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Memory pools sorted by GPU address (architecture D10, D14): the index nvn_pool.c keeps and nvn_pool_find reads.
//
// Taken from oot-pc-recomp runtime/nvn_pool_index.h at 8e09aaa (identical in DKB Recomp at 9d8fa02), ported there from
// MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/nvn_pool_index.h at b75be8cf (MIT, Copyright (c) 2026 jhaizhou;
// origin d2r-mobile; licenses/dkb-xlate-MIT.txt). Changed here: this comment; the exact-query cache and pidx_overlaps
// are gone (nothing here uses them, and the cache wrote on lookup, which readers sharing a lock cannot do). The code
// that remains is unchanged.
#pragma once
/* TOTK: memory pools sorted by guest address.
   The world scene keeps about 3,200 pools alive, and the replay resolves several dozen buffer addresses per draw
   (vertex / index / uniform / storage bindings). A linear scan of all pools for each of them was the largest part of
   the ~87k instructions per draw. Lookups are a binary search on the range start plus a backwards walk that is
   bounded by a running maximum of the range ends, so a pool nested inside or overlapping another is still found.
   Not thread-safe: the caller holds the pool lock for every call (shared for pidx_find). */
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

typedef struct { uintptr_t lo, hi, base; void *buf; uint64_t gpu; const void *key; int res; } pidx_ent;   // buf, res: back-end data
typedef struct { pidx_ent *e; uintptr_t *maxhi; int n, cap; } pidx;   // maxhi[i] = largest hi among e[0..i]

static inline void pidx_fix_maxhi(pidx *x, int from) {
    for (int i = from > 0 ? from : 0; i < x->n; i++) {
        uintptr_t prev = i ? x->maxhi[i - 1] : 0;
        x->maxhi[i] = x->e[i].hi > prev ? x->e[i].hi : prev;
    }
}
// Number of entries whose range starts at or below addr (index of the first start above it).
static inline int pidx_upper(const pidx *x, uintptr_t addr) {
    int a = 0, b = x->n;
    while (a < b) { int m = (a + b) >> 1; if (x->e[m].lo <= addr) a = m + 1; else b = m; }
    return a;
}
// Returns 0 when memory runs out (the index is left unchanged).
static inline int pidx_insert(pidx *x, const pidx_ent *v) {
    if (x->n == x->cap) {
        int cap = x->cap ? x->cap * 2 : 64;
        pidx_ent *e = realloc(x->e, (size_t)cap * sizeof *e); if (!e) return 0; x->e = e;
        uintptr_t *mh = realloc(x->maxhi, (size_t)cap * sizeof *mh); if (!mh) return 0; x->maxhi = mh;
        x->cap = cap;
    }
    int at = pidx_upper(x, v->lo);
    memmove(x->e + at + 1, x->e + at, (size_t)(x->n - at) * sizeof *x->e);
    x->e[at] = *v; x->n++;
    pidx_fix_maxhi(x, at);
    return 1;
}
// Removes the entry registered for key; returns 0 when there is none.
static inline int pidx_remove(pidx *x, const void *key, pidx_ent *out) {
    for (int i = 0; i < x->n; i++) {
        if (x->e[i].key != key) continue;
        if (out) *out = x->e[i];
        memmove(x->e + i, x->e + i + 1, (size_t)(x->n - i - 1) * sizeof *x->e); x->n--;
        pidx_fix_maxhi(x, i);
        return 1;
    }
    return 0;
}
// A pool whose range [lo, hi] contains [addr, addr + len], or NULL (same containment test as the old linear scan).
static inline const pidx_ent *pidx_find(const pidx *x, uintptr_t addr, uint64_t len) {
    uintptr_t end = addr + (uintptr_t)len;
    if (end < addr) return NULL;
    for (int i = pidx_upper(x, addr) - 1; i >= 0 && x->maxhi[i] >= end; i--)
        if (x->e[i].hi >= end) return &x->e[i];
    return NULL;
}
