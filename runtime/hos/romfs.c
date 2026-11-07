// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The RomFS pack, read-only: the index (romfs-index.bin) mapped once, the image (romfs.bin, tens of gigabytes: every
// offset is 64 bits) opened once, a path found by binary search over the index's sorted rows, a directory listing
// derived from the sorted paths. Both files come from the program glue (hos_game_file); tools/prepare/extract.py
// writes them and documents the layout.
//
// Adapted from oot-pc-recomp runtime/romfs.c at 8e09aaa39c2cff5c3ca7c2a4ee36345745eebe17 (GPL-3.0-or-later; the file's
// own header says it was adapted from MacNativeGames games/totk/runtime/romfs.c at
// b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026 MacNativeGames contributors): the lookup by
// binary search, the directory listing from sorted paths, the handle table, the checks on the index at start.
// Changed here: the index is this project's OOT2RFS1 format, mapped instead of read; the image is opened through
// hostfile.c and read at an offset into the buffer the caller owns; the pack opens on the first use, with its reason
// returned when it cannot; a path is normalised by fs_path_norm (a trailing slash and the root are handled); no
// overlays, no read cache, no whole-image hash, no environment settings.
#include <stdio.h>
#include <string.h>

#include "fs.h"
#include "plat.h"

#define INDEX_MAGIC "OOT2RFS1"
#define HEADER_BYTES 64u
#define ROW_BYTES 56u
#define HANDLE_SLOTS 65536u   // slot 0 is never used
#define PATH_CAP 1024

typedef struct Row { uint64_t offset, size; uint32_t name, len; } Row;

static const uint8_t *index_map;       // the whole index, mapped read-only for the life of the process
static uint64_t row_count, names_at;   // the rows start at HEADER_BYTES; the names at names_at
static HostFile *image;

static PlatMutex pack_lock = PLAT_MUTEX_INIT;
static int pack_ready;                 // 1 once everything above is set

static PlatMutex handle_lock = PLAT_MUTEX_INIT;
static uint32_t handles[HANDLE_SLOTS]; // row + 1 of an open file; 0 is free
static uint32_t handle_hint = 1;       // where the next search for a free slot starts

// The index is little-endian, as the host is.
static uint32_t get32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static uint64_t get64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

static Row row_at(uint64_t i) {
    const uint8_t *p = index_map + HEADER_BYTES + i * ROW_BYTES;
    return (Row){get64(p), get64(p + 8), get32(p + 16), get32(p + 20)};
}

static const char *row_name(Row r) { return (const char *)index_map + names_at + r.name; }

// ---- opening the pack
// Failures that are the player's set-up (a missing file) return 0 with the reason; a pack that is there but damaged is
// a fatal, since nothing the guest can do changes it.
static int pack_load(char *why, size_t cap) {
    char ipath[1024], mpath[1024];
    const char *ix = hos_game_file("RomFS index", ipath, sizeof ipath);
    const char *im = hos_game_file("RomFS image", mpath, sizeof mpath);
    if (!ix || !im) {
        snprintf(why, cap, "this build has no RomFS pack (oot2 prepare writes the index and the image)");
        return 0;
    }
    uint64_t size = 0;
    const uint8_t *map = plat_map_file(ix, &size);
    if (!map) {
        snprintf(why, cap, "cannot map the RomFS index %s", ix);
        return 0;
    }
    if (size < HEADER_BYTES || memcmp(map, INDEX_MAGIC, 8))
        rt_fatal("RomFS index %s is not an %s index", ix, INDEX_MAGIC);
    uint32_t version = get32(map + 8), row_size = get32(map + 12);
    uint64_t count = get64(map + 16), rows_at = get64(map + 24), nat = get64(map + 32), names_size = get64(map + 40),
             image_size = get64(map + 48);
    if (version != 1 || row_size != ROW_BYTES || rows_at != HEADER_BYTES || count >= 0xffffffffull ||
        count > (size - HEADER_BYTES) / ROW_BYTES || nat != rows_at + count * ROW_BYTES || names_size != size - nat)
        rt_fatal("RomFS index %s has a bad header", ix);

    HostFile *img = hf_open(im, 0, NULL);
    if (!img) {
        snprintf(why, cap, "cannot open the RomFS image %s", im);
        return 0;
    }
    uint64_t have = 0;
    if (!hf_size(img, &have) || have != image_size)
        rt_fatal("RomFS image %s has %llu bytes, its index was built for %llu (run oot2 prepare again)", im,
                 (unsigned long long)have, (unsigned long long)image_size);

    index_map = map;
    row_count = count;
    names_at = nat;
    image = img;
    // Every row, once: inside the image, a name inside the names and ended by its NUL, strictly ascending by bytes.
    const char *prev = NULL;
    for (uint64_t i = 0; i < count; i++) {
        Row r = row_at(i);
        if (r.size > image_size || r.offset > image_size - r.size)
            rt_fatal("RomFS index %s: row %llu lies outside the image", ix, (unsigned long long)i);
        if (!r.len || (uint64_t)r.name + r.len >= names_size || row_name(r)[r.len] || strlen(row_name(r)) != r.len)
            rt_fatal("RomFS index %s: the name of row %llu is not inside the names", ix, (unsigned long long)i);
        if (prev && strcmp(prev, row_name(r)) >= 0)
            rt_fatal("RomFS index %s: row %llu is not in ascending path order", ix, (unsigned long long)i);
        prev = row_name(r);
    }
    rt_log("RomFS pack: %llu files, image of %llu bytes", (unsigned long long)count, (unsigned long long)image_size);
    return 1;
}

// The whole slow path runs without the virtual core, the wait for pack_lock included: a thread that waited for it while
// holding a core could keep the loading thread from getting one back, with every core held by a waiter.
int romfs_ready(char *why, size_t cap) {
    if (__atomic_load_n(&pack_ready, __ATOMIC_ACQUIRE)) return 1;
    int tok = rt_kernel_block_begin();
    plat_mutex_lock(&pack_lock);
    int ok = __atomic_load_n(&pack_ready, __ATOMIC_ACQUIRE) || pack_load(why, cap);
    if (ok) __atomic_store_n(&pack_ready, 1, __ATOMIC_RELEASE);
    plat_mutex_unlock(&pack_lock);
    rt_kernel_block_end(tok);
    return ok;
}

// ---- lookup
// The first row whose path is not below name.
static uint64_t lower(const char *name) {
    uint64_t lo = 0, hi = row_count;
    while (lo < hi) {
        uint64_t mid = lo + (hi - lo) / 2;
        if (strcmp(row_name(row_at(mid)), name) < 0) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}

// The row of the file rel, or UINT64_MAX.
static uint64_t find(const char *rel) {
    uint64_t i = lower(rel);
    return i < row_count && !strcmp(row_name(row_at(i)), rel) ? i : UINT64_MAX;
}

// Whether a file lies under the folder rel.
static int folder_exists(const char *rel) {
    char prefix[PATH_CAP + 2];
    int n = snprintf(prefix, sizeof prefix, "%s/", rel);
    if (n < 0 || (size_t)n >= sizeof prefix) return 0;
    uint64_t i = lower(prefix);
    return i < row_count && !strncmp(row_name(row_at(i)), prefix, (size_t)n);
}

// ---- handles
int romfs_handle(uint64_t h) { return h > ROMFS_HANDLE_BASE && h < ROMFS_HANDLE_BASE + HANDLE_SLOTS; }

// The row an open handle names; 0 when h is not open.
static int handle_row(uint64_t h, Row *out) {
    if (!romfs_handle(h)) return 0;
    plat_mutex_lock(&handle_lock);
    uint32_t i = handles[h - ROMFS_HANDLE_BASE];
    plat_mutex_unlock(&handle_lock);
    if (!i) return 0;
    *out = row_at(i - 1);
    return 1;
}

uint32_t romfs_open(uint64_t *out, const char *rest) {
    char rel[PATH_CAP];
    if (!fs_path_norm(rest, rel, sizeof rel, 0) || !*rel) return FS_R_PATH_NOT_FOUND;
    uint64_t i = find(rel);
    if (i == UINT64_MAX) return FS_R_PATH_NOT_FOUND;
    plat_mutex_lock(&handle_lock);
    for (uint32_t k = 0; k < HANDLE_SLOTS - 1; k++) {
        uint32_t j = 1 + (handle_hint - 1 + k) % (HANDLE_SLOTS - 1);
        if (!handles[j]) {
            handles[j] = (uint32_t)i + 1;
            handle_hint = j + 1;
            plat_mutex_unlock(&handle_lock);
            *out = ROMFS_HANDLE_BASE + j;
            return 0;
        }
    }
    plat_mutex_unlock(&handle_lock);
    return FS_R_NO_SLOT;
}

uint32_t romfs_size(int64_t *out, uint64_t h) {
    Row r;
    if (!handle_row(h, &r)) return FS_R_INVALID;
    *out = (int64_t)r.size;
    return 0;
}

int64_t romfs_pread(uint64_t h, void *buf, uint32_t n, uint64_t off) {
    Row r;
    if (!handle_row(h, &r)) return -1;
    if (off >= r.size) return 0;
    if (n > r.size - off) n = (uint32_t)(r.size - off);
    return hf_pread(image, buf, n, r.offset + off);
}

int romfs_close(uint64_t h) {
    if (!romfs_handle(h)) return 0;
    plat_mutex_lock(&handle_lock);
    uint32_t was = handles[h - ROMFS_HANDLE_BASE];
    handles[h - ROMFS_HANDLE_BASE] = 0;
    plat_mutex_unlock(&handle_lock);
    return was != 0;
}

// ---- entries
uint32_t romfs_entry_type(int *out, const char *rest) {
    char rel[PATH_CAP];
    if (!fs_path_norm(rest, rel, sizeof rel, 0)) return FS_R_PATH_NOT_FOUND;
    if (!*rel || (find(rel) == UINT64_MAX && folder_exists(rel))) {   // the root exists whatever the pack holds
        *out = FS_ENTRY_DIRECTORY;
        return 0;
    }
    if (find(rel) == UINT64_MAX) return FS_R_PATH_NOT_FOUND;
    *out = FS_ENTRY_FILE;
    return 0;
}

uint32_t romfs_list(const char *rest, int (*add)(const char *name, int kind, uint64_t size, void *ctx), void *ctx) {
    char rel[PATH_CAP], prefix[PATH_CAP + 2], last[PATH_CAP] = "";
    if (!fs_path_norm(rest, rel, sizeof rel, 0)) return FS_R_PATH_NOT_FOUND;
    size_t n = (size_t)snprintf(prefix, sizeof prefix, *rel ? "%s/" : "%s", rel);   // the root has an empty prefix
    int found = !*rel;
    for (uint64_t i = lower(prefix); i < row_count; i++) {
        Row r = row_at(i);
        const char *name = row_name(r);
        if (strncmp(name, prefix, n)) break;
        const char *tail = name + n;
        found = 1;
        const char *slash = strchr(tail, '/');
        if (!slash) {
            if (!add(tail, HF_FILE, r.size, ctx)) return 0;
            continue;
        }
        size_t k = (size_t)(slash - tail);   // a subdirectory: its files are contiguous, so it is named once
        if (k >= sizeof last) return FS_R_INVALID;
        if (strlen(last) == k && !strncmp(last, tail, k)) continue;
        memcpy(last, tail, k);
        last[k] = 0;
        if (!add(last, HF_DIR, 0, ctx)) return 0;
    }
    return found ? 0 : FS_R_PATH_NOT_FOUND;
}
