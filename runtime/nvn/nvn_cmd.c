// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The NVN2 front end's command buffers (architecture D11): the object and its command and control memory, recordings,
// the host copies behind command handles, CallCommands and CopyCommands, and what a submission reads when it is made
// (nvn_submit_recordings, gfx.h). The commands themselves are recorded in nvn_cmd_ops.c.
//
// Records go to a growable host buffer per command buffer, never into the game's command memory: they are larger than
// the console's methods, and the game budgets its memory for the console. EndRecording copies the recording into a
// host block of its exact size, and the handle is that copy. A copy lives until the game adds the control memory its
// header went to again (on the console the handle lives there, and adding that memory again overwrites it), or until
// a new recording puts its header at the same address. CallCommands and CopyCommands inline the called recording, so a
// copy holds no call.
//
// How a recording starts. The game's NVN2 header inlines BeginRecording, at twelve sites of main: when 16 bytes of
// control memory are left it writes the command buffer's recording fields and a 16-byte header itself, and calls
// nvnCommandBufferBeginRecording only when they are not, which is why the NVN2 call census counts one BeginRecording
// a frame against 131 EndRecording. The console driver's own BeginRecording does the same after calling the memory
// callback for control memory. So the object keeps the driver's layout (nvn_obj.h cmd_t), and the front end takes up
// a recording the game began itself at the next call on that command buffer: +0x11 says one is open, +0x50 says where
// its header went, and the control write pointer must have moved by exactly the header.
//
// Memory. Control memory is accounted the way the driver fills it, entry by entry (a header, a command segment
// closed when command memory changes or the recording ends, an end, a jump to new control memory), so the game's
// inline BeginRecording, its GetControlMemoryUsed and its memory callback see what they would see on the console; the
// front end writes nothing there. The driver extends a segment entry when the new segment continues the last one in
// command memory; the front end never does, so it may count 0x18 bytes more than the console, never fewer, and the
// game's control memory never runs ahead of a live handle. Command memory is accounted only: GetCommandMemoryUsed
// answers the bytes recorded since the game last added some, the estimate of what the console would have written,
// never more than the game added, and the memory callback for command memory is called only where the console's
// certainly would be: a command recorded with no command memory added at all.
//
// Adapted from oot-pc-recomp runtime/nvn_api_cmd.c at 8e09aaa (GPL-3.0-or-later), DKB Recomp's runtime/nvn_api_cmd.c
// at 9d8fa02 with additions, which ports MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c at b75be8cf
// (MIT, Copyright (c) 2026 jhaizhou; licenses/dkb-xlate-MIT.txt): the recording buffer, copies retired when their
// control memory is added again, called recordings inlined, deferred binds read at submit, the memory callback called
// through the guest, the op names. Changed here: the object, the recording start and the control memory accounting
// are the NVN2 driver's; copies are keyed by their header's address and pinned while a submission reads them; the
// copy index is a table of its own; textures are recorded by value (nvn_cmd_ops.c), so no texture is pinned.
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "nvn_cmd.h"
#include "oot2/ledger.h"
#include "plat.h"

typedef unsigned long long ull;

#define CMD_MAGIC MAGIC('C', 'M', 'D', 'B')
#define CMD_NO_ROOM (~0ull)            // cmd_t.cmd_cur: no inline fast path finds room past it
#define REC_FIRST_SIZE 4096u           // a recording buffer's first size; it doubles
// The driver's control entries: a recording's header, a command segment (written with 0x20 bytes free, so the end
// still fits), a recording's end, and a jump to new control memory (in the 16 bytes kept back at the end).
#define CTL_HEADER 0x10u
#define CTL_SEGMENT 0x18u
#define CTL_SEGMENT_ROOM 0x20u
#define CTL_END 8u
#define CTL_JUMP 0x10u
#define CTL_KEPT 0x10u
#define CTL_CALLBACK_SIZE 0x100u       // what the driver asks its memory callback for, for control memory
#define CMD_CALLBACK_SIZE 0x200u       // and at least this for command memory
#define COPY_PAGE_SHIFT 16             // copies are indexed by the 64 KiB page of their header
#define SUBMIT_MAX 4096                // handles in one submission

static int64_t g_host_bytes;           // host memory the command buffers, their copies and the index hold
static void host_bytes(int64_t n) { __atomic_fetch_add(&g_host_bytes, n, __ATOMIC_RELAXED); }
uint64_t nvn_cmd_host_bytes(void) { return (uint64_t)__atomic_load_n(&g_host_bytes, __ATOMIC_RELAXED); }

// ---------- the front end's state of one command buffer (cmd_t.host)
typedef struct nvn_cmd_host {
    uint8_t *buf, *cur, *end;          // records since the last start of a recording, OP_BEGIN first while one is open
    uint8_t *rec_ctl;                  // the open recording's header in control memory; NULL when none is open
    uint8_t *ctl_seen;                 // the control write pointer where the front end last left it
    uint64_t cmd_given, cmd_used;      // command memory the game added last, and what the console would have written
    uint64_t ctl_inner;                // control bytes of the open recording's entries between its header and its end
    int pending;                       // a record since the last segment entry: the console's next entry covers it
    int beginning;                     // BeginRecording runs the memory callback: its recording is not taken up yet
    int deferred;                      // the open recording holds a deferred bind or push
} nvn_cmd_host;

static void reserve(nvn_cmd_host *h, size_t need) {
    size_t used = (size_t)(h->cur - h->buf), size = (size_t)(h->end - h->buf);
    if (size - used >= need) return;
    size_t grown = size ? size : REC_FIRST_SIZE;
    while (grown - used < need) grown *= 2;
    uint8_t *b = realloc(h->buf, grown);
    if (!b) rt_fatal("nvn: a command recording buffer of %zu bytes", grown);
    host_bytes((int64_t)grown - (int64_t)size);
    h->buf = b, h->cur = b + used, h->end = b + grown;
}

static void put_record(nvn_cmd_host *h, uint32_t op, size_t size) {
    reserve(h, size);
    op_hdr hd = {op, (uint32_t)size};
    memcpy(h->cur, &hd, sizeof hd);
    memset(h->cur + sizeof hd, 0, size - sizeof hd);
    h->cur += size;
}

// A recording starts with its header at ctl. Records made since the last start (commands outside a recording, or a
// recording begun again before it ended) are left out of every handle, as the console's driver leaves them out.
// What the run did with recordings (the null back end boot, the NVN2 call census): counters only, read by nvn_cmd_counts.
static uint64_t g_begin_inline, g_begin_slow, g_end_handle, g_end_empty, g_cb_control, g_cb_command, g_left_out;
void nvn_cmd_counts(NvnCmdCounts *out) {
    *out = (NvnCmdCounts){__atomic_load_n(&g_begin_inline, __ATOMIC_RELAXED), __atomic_load_n(&g_begin_slow, __ATOMIC_RELAXED),
                          __atomic_load_n(&g_end_handle, __ATOMIC_RELAXED), __atomic_load_n(&g_end_empty, __ATOMIC_RELAXED),
                          __atomic_load_n(&g_cb_control, __ATOMIC_RELAXED), __atomic_load_n(&g_cb_command, __ATOMIC_RELAXED),
                          __atomic_load_n(&g_left_out, __ATOMIC_RELAXED)};
}

static void open_recording(nvn_cmd_host *h, uint8_t *ctl) {
    if (h->cur != h->buf) {
        __atomic_fetch_add(&g_left_out, 1, __ATOMIC_RELAXED);
        static int logged;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
            rt_log("nvn: %zu bytes of commands recorded %s are left out of every handle, as the console leaves them",
                   (size_t)(h->cur - h->buf), h->rec_ctl ? "before a recording began again" : "outside a recording");
    }
    h->cur = h->buf;
    put_record(h, OP_BEGIN, sizeof(op_hdr));
    h->rec_ctl = ctl, h->ctl_inner = 0, h->pending = 0, h->deferred = 0;
}

// ---------- the game's memory callback, called through the guest as the driver calls it: (cmd, event, size, data)
static void memory_callback(Cpu *c, cmd_t *o, int event, uint64_t size) {
    const char *what = event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY ? "command" : "control";
    __atomic_fetch_add(event == NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY ? &g_cb_command : &g_cb_control, 1,
                       __ATOMIC_RELAXED);
    if (!o->cb)
        rt_unimplemented(c, "nvn", "command buffer %p needs %s memory, and its memory callback is on but not set "
                         "(the console's driver aborts)", (void *)o, what);
    uint64_t x[8], pc = c->pc, lr = c->x[30];
    __uint128_t v[8];
    memcpy(x, c->x, sizeof x);
    memcpy(v, c->v, sizeof v);
    o->host->ctl_seen = o->ctl_cur;
    c->x[0] = (uintptr_t)o, c->x[1] = (uint64_t)event, c->x[2] = size, c->x[3] = (uintptr_t)o->cb_data;
    rt_call(c, (uintptr_t)o->cb);
    memcpy(c->x, x, sizeof x);
    memcpy(c->v, v, sizeof v);
    c->pc = pc, c->x[30] = lr;
}

// Control memory for an entry of `bytes` with `room` free first, as the driver takes it: the memory callback when the
// room is not left and the callback is on, and a stop when it added none. Returns where the entry goes.
static uint8_t *ctl_take(Cpu *c, cmd_t *o, uint64_t room, uint64_t bytes, const char *fn) {
    if ((uintptr_t)o->ctl_cur + room > (uintptr_t)o->ctl_end) {
        if (o->ctl_cb_on) {
            memory_callback(c, o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_CONTROL_MEMORY, CTL_CALLBACK_SIZE);
            if ((uintptr_t)o->ctl_cur + room > (uintptr_t)o->ctl_end)
                rt_unimplemented(c, "nvn", "%s: command buffer %p: its memory callback left %lld bytes of control memory "
                                 "where %llu are needed (the console's driver aborts)", fn, (void *)o,
                                 (long long)((uintptr_t)o->ctl_end - (uintptr_t)o->ctl_cur), (ull)room);
        } else if (!o->ctl_cur) {
            rt_unimplemented(c, "nvn", "%s: command buffer %p has no control memory and its control memory callback is "
                             "off: the console would write at address 0", fn, (void *)o);
        }
    }
    uint8_t *at = o->ctl_cur;
    o->ctl_cur += bytes;
    o->host->ctl_seen = o->ctl_cur;
    return at;
}

// The driver closes the commands written since its last entry with a segment entry.
static void segment_entry(Cpu *c, cmd_t *o, const char *fn) {
    uint8_t *at = ctl_take(c, o, CTL_SEGMENT_ROOM, CTL_SEGMENT, fn);
    o->last_entry = (uintptr_t)at;
    o->host->pending = 0;
    if (o->host->rec_ctl) o->host->ctl_inner += CTL_SEGMENT;
}

// The command buffer of entry point c, checked, with a recording the game began itself taken up.
static cmd_t *cmd_sync(Cpu *c, const char *fn) {
    cmd_t *o = arg_p(c, 0);
    if (!o || o->magic != CMD_MAGIC || !o->host)
        rt_unimplemented(c, "nvn", "%s on %p, which CommandBufferInitialize did not initialise", fn, (void *)o);
    nvn_cmd_host *h = o->host;
    if (o->cmd_cur != CMD_NO_ROOM || o->cmd_end || o->cmd_limit)
        rt_unimplemented(c, "nvn", "%s: command buffer %p: game code moved its command write pointer (+0x00 %#llx, "
                         "+0x08 %#llx, +0x20 %#llx): it writes GPU methods into command memory itself, which the front "
                         "end cannot replay", fn, (void *)o, (ull)o->cmd_cur, (ull)o->cmd_end, (ull)o->cmd_limit);
    if (o->recording && o->rec_ctl != h->rec_ctl && !h->beginning) {
        // The game's inline BeginRecording: its header where the control write pointer stood, the pointer moved past.
        if (o->rec_ctl != h->ctl_seen || o->ctl_cur != h->ctl_seen + CTL_HEADER)
            rt_unimplemented(c, "nvn", "%s: command buffer %p: a recording began outside the front end with its header "
                             "at %p and the control write pointer at %p, which the front end left at %p", fn, (void *)o,
                             (void *)o->rec_ctl, (void *)o->ctl_cur, (void *)h->ctl_seen);
        __atomic_fetch_add(&g_begin_inline, 1, __ATOMIC_RELAXED);
        open_recording(h, o->rec_ctl);
        h->ctl_seen = o->ctl_cur;
    } else if (!o->recording && h->rec_ctl) {
        rt_unimplemented(c, "nvn", "%s: command buffer %p: its recording (header at %p) ended outside the front end", fn,
                         (void *)o, (void *)h->rec_ctl);
    }
    if (o->ctl_cur != h->ctl_seen)
        rt_unimplemented(c, "nvn", "%s: command buffer %p: game code moved its control write pointer from %p to %p: "
                         "it writes control memory itself, which the front end cannot follow", fn, (void *)o,
                         (void *)h->ctl_seen, (void *)o->ctl_cur);
    return o;
}

static nvn_cmd_host *cmd_host(Cpu *c, const char *fn) { return cmd_sync(c, fn)->host; }

// The console writes every command into command memory. With none added, its first command calls the memory callback
// for some, and the driver aborts when the callback added none: the one place the front end calls it for commands.
static void command_memory(Cpu *c, cmd_t *o, size_t size) {
    if (!o->cmd_cb_on) return;
    memory_callback(c, o, NVN_COMMAND_BUFFER_MEMORY_EVENT_OUT_OF_COMMAND_MEMORY,
                    size > CMD_CALLBACK_SIZE ? size : CMD_CALLBACK_SIZE);
    if (!o->host->cmd_given)
        rt_unimplemented(c, "nvn", "command buffer %p: its memory callback added no command memory (the console's driver "
                         "aborts)", (void *)o);
}

void *nvn_rec(Cpu *c, uint32_t op, size_t payload) {
    const char *fn = nvn_op_name(op);
    cmd_t *o = cmd_sync(c, fn);
    nvn_cmd_host *h = o->host;
    size_t size = (sizeof(op_hdr) + payload + 7) & ~(size_t)7;
    if (size > UINT32_MAX) rt_unimplemented(c, "nvn", "command %s of %zu bytes", fn, size);
    if (!h->cmd_given) command_memory(c, o, size);
    put_record(h, op, size);
    h->cmd_used += size;
    h->pending = 1;
    return h->cur - size + sizeof(op_hdr);
}

void nvn_rec_deferred(const Cpu *c) { ((cmd_t *)arg_p(c, 0))->host->deferred = 1; }

// ---------- copies, the index behind command handles
// A copy: this prefix, then its records from OP_BEGIN to OP_END. Its handle points just past the prefix.
typedef struct nvn_copy {
    struct nvn_copy *next;             // the next copy whose header lies in the same page
    uintptr_t key;                     // its header's address in control memory
    uint64_t bytes, ctl_inner;         // its records' bytes; the control bytes of its entries, for a caller to inline
    uint32_t refs, retired;            // calls and submissions reading it now; out of the index, freed at the last one
    uint32_t deferred, pad;            // it holds a deferred bind or push
} nvn_copy;
_Static_assert(sizeof(nvn_copy) % 8 == 0, "records stay 8-byte aligned");

// Open addressing on 64-bit keys (never 0), linear probing, deletion by backward shift.
typedef struct { uint64_t *key; void **val; uint32_t size, n; } u64map;
static uint32_t home(const u64map *m, uint64_t k) {
    k ^= k >> 33, k *= 0xff51afd7ed558ccdull, k ^= k >> 33;
    return (uint32_t)k & (m->size - 1);
}
static void **map_get(const u64map *m, uint64_t k) {
    if (!m->size) return NULL;
    for (uint32_t i = home(m, k);; i = (i + 1) & (m->size - 1)) {
        if (m->key[i] == k) return &m->val[i];
        if (!m->key[i]) return NULL;
    }
}
static void map_put(u64map *m, uint64_t k, void *v);
static void map_grow(u64map *m) {
    u64map g = {calloc(m->size ? 2 * m->size : 64, sizeof(uint64_t)), NULL, m->size ? 2 * m->size : 64, 0};
    g.val = calloc(g.size, sizeof(void *));
    if (!g.key || !g.val) rt_fatal("nvn: command copy index of %u entries", g.size);
    host_bytes((int64_t)(g.size - m->size) * (int64_t)(sizeof(uint64_t) + sizeof(void *)));
    for (uint32_t i = 0; i < m->size; i++)
        if (m->key[i]) map_put(&g, m->key[i], m->val[i]);
    free(m->key), free(m->val);
    *m = g;
}
static void map_put(u64map *m, uint64_t k, void *v) {
    if (2 * (m->n + 1) > m->size) map_grow(m);
    uint32_t i = home(m, k);
    while (m->key[i] && m->key[i] != k) i = (i + 1) & (m->size - 1);
    if (!m->key[i]) m->n++;
    m->key[i] = k, m->val[i] = v;
}
static void map_del(u64map *m, uint64_t k) {
    if (!m->size) return;
    uint32_t mask = m->size - 1, i = home(m, k);
    while (m->key[i] != k) {
        if (!m->key[i]) return;
        i = (i + 1) & mask;
    }
    for (uint32_t j = i;;) {
        m->key[i] = 0;
        for (;;) {
            j = (j + 1) & mask;
            if (!m->key[j]) {
                m->n--;
                return;
            }
            uint32_t h = home(m, m->key[j]);
            if (i <= j ? (h <= i || h > j) : (h <= i && h > j)) break;   // j may move back to i
        }
        m->key[i] = m->key[j], m->val[i] = m->val[j];
        i = j;
    }
}

static PlatMutex copy_lock = PLAT_MUTEX_INIT;
static u64map by_handle, by_page;      // handle -> copy; header page + 1 -> its copies
static uint64_t ncopies;

static uint64_t handle_of(const nvn_copy *k) { return (uintptr_t)(k + 1); }
static void copy_free(nvn_copy *k) {
    host_bytes(-(int64_t)(sizeof *k + k->bytes));
    free(k);
}
// Out of the index (copy_lock held): freed now, or by the last call or submission that reads it.
static void copy_drop(nvn_copy *k) {
    map_del(&by_handle, handle_of(k));
    ncopies--;
    if (k->refs) k->retired = 1;
    else copy_free(k);
}

// Adding control memory again overwrites, on the console, the handles recorded in it: their copies go.
static void retire_copies(uintptr_t lo, uintptr_t hi) {
    if (hi <= lo) return;
    plat_mutex_lock(&copy_lock);
    for (uintptr_t pg = lo >> COPY_PAGE_SHIFT; pg <= (hi - 1) >> COPY_PAGE_SHIFT && by_page.n; pg++) {
        void **slot = map_get(&by_page, pg + 1);
        if (!slot) continue;
        nvn_copy *k = *slot, *kept = NULL;
        while (k) {
            nvn_copy *next = k->next;
            if (k->key >= lo && k->key < hi) copy_drop(k);
            else k->next = kept, kept = k;
            k = next;
        }
        if (kept) *slot = kept;
        else map_del(&by_page, pg + 1);
    }
    plat_mutex_unlock(&copy_lock);
}

// EndRecording: the open recording, OP_END appended, into a copy indexed by its header's address. A copy whose header
// was at the same address is gone: on the console its control memory has been written again.
static uint64_t copy_publish(nvn_cmd_host *h) {
    put_record(h, OP_END, sizeof(op_hdr));
    size_t bytes = (size_t)(h->cur - h->buf);
    nvn_copy *k = malloc(sizeof *k + bytes);
    if (!k) rt_fatal("nvn: a command recording copy of %zu bytes", bytes);
    *k = (nvn_copy){NULL, (uintptr_t)h->rec_ctl, bytes, h->ctl_inner, 0, 0, (uint32_t)h->deferred, 0};
    memcpy(k + 1, h->buf, bytes);
    host_bytes((int64_t)(sizeof *k + bytes));
    retire_copies(k->key, k->key + 1);
    uint64_t page = (k->key >> COPY_PAGE_SHIFT) + 1;
    plat_mutex_lock(&copy_lock);
    void **slot = map_get(&by_page, page);
    k->next = slot ? *slot : NULL;
    map_put(&by_page, page, k);
    map_put(&by_handle, handle_of(k), k);
    ncopies++;
    plat_mutex_unlock(&copy_lock);
    return handle_of(k);
}

static nvn_copy *copy_pin(Cpu *c, uint64_t handle, const char *fn) {
    plat_mutex_lock(&copy_lock);
    void **slot = handle ? map_get(&by_handle, handle) : NULL;
    nvn_copy *k = slot ? *slot : NULL;
    if (k) k->refs++;
    plat_mutex_unlock(&copy_lock);
    if (!k)
        rt_unimplemented(c, "nvn", "%s: command handle %#llx is not a live recording (never recorded, or the control "
                         "memory it was recorded in was added again)", fn, (ull)handle);
    return k;
}
static void copy_unpin(nvn_copy *k) {
    plat_mutex_lock(&copy_lock);
    if (!--k->refs && k->retired) copy_free(k);
    plat_mutex_unlock(&copy_lock);
}

uint64_t nvn_cmd_copies(void) {
    plat_mutex_lock(&copy_lock);
    uint64_t n = ncopies;
    plat_mutex_unlock(&copy_lock);
    return n;
}

// ---------- CommandBuffer: object and memory
// NVNboolean nvnCommandBufferInitialize(NVNcommandBuffer *cmd, NVNdevice *device): the driver's fields zeroed, both
// memory callbacks on, the command write pointer where nothing fits
NVN_API(nvnCommandBufferInitialize) {
    cmd_t *o = arg_p(c, 0);
    // Over an object that was not finalized its host state is dropped, not freed: the old bytes may be a copy of a
    // command buffer that still owns it.
    if (o->magic == CMD_MAGIC && o->host) {
        static int logged;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
            rt_log("nvn: CommandBufferInitialize over the command buffer %p, which was not finalized: its host state is "
                   "dropped", (void *)o);
    }
    nvn_cmd_host *h = calloc(1, sizeof *h);
    if (!h) rt_fatal("nvn: command buffer host state");
    host_bytes(sizeof *h);
    memset(o, 0, sizeof *o);
    o->cmd_cur = CMD_NO_ROOM;
    o->cmd_cb_on = o->ctl_cb_on = 1;
    o->dev = arg_p(c, 1);
    o->host = h;
    o->magic = CMD_MAGIC;
    ret_x(c, 1);
}
// void nvnCommandBufferFinalize(NVNcommandBuffer *cmd): handles outlive the command buffer that recorded them
NVN_API(nvnCommandBufferFinalize) {
    cmd_t *o = cmd_sync(c, "CommandBufferFinalize");
    nvn_cmd_host *h = o->host;
    host_bytes(-(int64_t)(h->end - h->buf) - (int64_t)sizeof *h);
    free(h->buf);
    free(h);
    o->host = NULL;
    o->magic = 0;
}
// void nvnCommandBufferSetDebugLabel(NVNcommandBuffer *cmd, const char *label): kept by no one
NVN_API(nvnCommandBufferSetDebugLabel) { cmd_sync(c, "CommandBufferSetDebugLabel"); }
// void nvnCommandBufferSetMemoryCallback(NVNcommandBuffer *cmd, PFNNVNCOMMANDBUFFERMEMORYCALLBACKPROC callback),
// SetMemoryCallbackData(cmd, void *data), and their getters
NVN_API(nvnCommandBufferSetMemoryCallback) { cmd_sync(c, "CommandBufferSetMemoryCallback")->cb = arg_p(c, 1); }
NVN_API(nvnCommandBufferSetMemoryCallbackData) { cmd_sync(c, "CommandBufferSetMemoryCallbackData")->cb_data = arg_p(c, 1); }
NVN_API(nvnCommandBufferGetMemoryCallback) { ret_x(c, (uintptr_t)cmd_sync(c, "CommandBufferGetMemoryCallback")->cb); }
NVN_API(nvnCommandBufferGetMemoryCallbackData) {
    ret_x(c, (uintptr_t)cmd_sync(c, "CommandBufferGetMemoryCallbackData")->cb_data);
}
// void nvnCommandBufferSet{Command,Control}MemoryCallbackEnabled(NVNcommandBuffer *cmd, NVNboolean enabled)
NVN_API(nvnCommandBufferSetCommandMemoryCallbackEnabled) {
    cmd_sync(c, "CommandBufferSetCommandMemoryCallbackEnabled")->cmd_cb_on = nvn_arg_bool(c, 1) != 0;
}
NVN_API(nvnCommandBufferSetControlMemoryCallbackEnabled) {
    cmd_sync(c, "CommandBufferSetControlMemoryCallbackEnabled")->ctl_cb_on = nvn_arg_bool(c, 1) != 0;
}
// void nvnCommandBufferSetRecordingMode(NVNcommandBuffer *cmd, int mode): kept where the driver keeps it. The driver
// gives recordings made in modes 3 and 4 a handle of another form; every handle here is a host copy.
NVN_API(nvnCommandBufferSetRecordingMode) { cmd_sync(c, "CommandBufferSetRecordingMode")->rec_mode = (int32_t)arg_x(c, 1); }

// void nvnCommandBufferAddCommandMemory(NVNcommandBuffer *cmd, const NVNmemoryPool *pool, ptrdiff_t offset, size_t
// size): accounted only. The driver closes the commands written so far with a segment entry first.
NVN_API(nvnCommandBufferAddCommandMemory) {
    const char *fn = "CommandBufferAddCommandMemory";
    cmd_t *o = cmd_sync(c, fn);
    const pool_t *p = arg_p(c, 1);
    int64_t off = (int64_t)arg_x(c, 2);
    uint64_t size = arg_x(c, 3);
    if (p && p->magic != NVN_POOL_MAGIC)
        rt_unimplemented(c, "nvn", "%s: command buffer %p: the memory pool %p is not initialised", fn, (void *)o,
                         (const void *)p);
    if (o->host->pending) segment_entry(c, o, fn);
    o->host->cmd_given = size, o->host->cmd_used = 0;
    o->cmd_gpu = (p ? (uintptr_t)p->mem : 0) + (uint64_t)off;
}
// void nvnCommandBufferAddControlMemory(NVNcommandBuffer *cmd, void *memory, size_t size): the handles recorded in that
// memory go; inside a recording the driver jumps to the new memory from the 16 bytes it kept back.
NVN_API(nvnCommandBufferAddControlMemory) {
    const char *fn = "CommandBufferAddControlMemory";
    cmd_t *o = cmd_sync(c, fn);
    uint8_t *m = arg_p(c, 1);
    uint64_t n = arg_x(c, 2);
    if (!m) rt_unimplemented(c, "nvn", "%s: command buffer %p: control memory at address 0 (%llu bytes)", fn, (void *)o, (ull)n);
    retire_copies((uintptr_t)m, (uintptr_t)m + n);
    if (o->ctl_start && o->recording) o->ctl_cur += CTL_JUMP, o->ctl_moved = 1;
    else o->rec_ctl = m;
    o->ctl_start = m, o->last_entry = 0;
    o->ctl_cur = m, o->ctl_end = m + n - CTL_KEPT;
    o->host->ctl_seen = m;
}
// size_t nvnCommandBufferGet{Command,Control}Memory{Size,Used,Free}(const NVNcommandBuffer *cmd). Control memory: the
// driver's own arithmetic on the fields it keeps. Command memory: what the console would have written since the game
// added some, never more than it added.
static uint64_t cmd_used(const nvn_cmd_host *h) { return h->cmd_used < h->cmd_given ? h->cmd_used : h->cmd_given; }
NVN_API(nvnCommandBufferGetCommandMemorySize) { ret_x(c, cmd_host(c, "CommandBufferGetCommandMemorySize")->cmd_given); }
NVN_API(nvnCommandBufferGetCommandMemoryUsed) { ret_x(c, cmd_used(cmd_host(c, "CommandBufferGetCommandMemoryUsed"))); }
NVN_API(nvnCommandBufferGetCommandMemoryFree) {
    const nvn_cmd_host *h = cmd_host(c, "CommandBufferGetCommandMemoryFree");
    ret_x(c, h->cmd_given - cmd_used(h));
}
NVN_API(nvnCommandBufferGetControlMemorySize) {
    const cmd_t *o = cmd_sync(c, "CommandBufferGetControlMemorySize");
    ret_x(c, (uintptr_t)o->ctl_end - (uintptr_t)o->ctl_start + CTL_KEPT);
}
NVN_API(nvnCommandBufferGetControlMemoryUsed) {
    const cmd_t *o = cmd_sync(c, "CommandBufferGetControlMemoryUsed");
    ret_x(c, (uintptr_t)o->ctl_cur - (uintptr_t)o->ctl_start);
}
NVN_API(nvnCommandBufferGetControlMemoryFree) {
    const cmd_t *o = cmd_sync(c, "CommandBufferGetControlMemoryFree");
    ret_x(c, (uintptr_t)o->ctl_end - (uintptr_t)o->ctl_cur + CTL_KEPT);
}

// ---------- CommandBuffer: recording
// void nvnCommandBufferBeginRecording(NVNcommandBuffer *cmd): the game's slow path, which the driver takes the same
// way as the inline one after the memory callback for control memory
NVN_API(nvnCommandBufferBeginRecording) {
    const char *fn = "CommandBufferBeginRecording";
    cmd_t *o = cmd_sync(c, fn);
    nvn_cmd_host *h = o->host;
    o->last_entry = 0, o->rec_cmd = o->cmd_cur, o->rec_ctl = o->ctl_cur;
    o->recording = 1, o->ctl_moved = 0;
    h->beginning = 1;
    o->rec_ctl = ctl_take(c, o, CTL_HEADER, CTL_HEADER, fn);
    h->beginning = 0;
    __atomic_fetch_add(&g_begin_slow, 1, __ATOMIC_RELAXED);
    open_recording(h, o->rec_ctl);
}
// NVNcommandHandle nvnCommandBufferEndRecording(NVNcommandBuffer *cmd): the copy's handle; 0 when no recording is
// open, as the driver answers
NVN_API(nvnCommandBufferEndRecording) {
    const char *fn = "CommandBufferEndRecording";
    cmd_t *o = cmd_sync(c, fn);
    nvn_cmd_host *h = o->host;
    if (!o->recording) {
        __atomic_fetch_add(&g_end_empty, 1, __ATOMIC_RELAXED);
        ret_x(c, 0);
        return;
    }
    __atomic_fetch_add(&g_end_handle, 1, __ATOMIC_RELAXED);
    if (h->pending) segment_entry(c, o, fn);
    if (o->ctl_cur == o->rec_ctl + CTL_HEADER) o->ctl_moved = 1, o->flags13 |= 1;
    else o->flags13 &= (uint8_t)~1u;
    ctl_take(c, o, CTL_END, CTL_END, fn);
    o->last_entry = 0;
    uint64_t handle = copy_publish(h);
    o->recording = 0, o->ctl_moved = 0;
    o->rec_ctl = o->ctl_cur;
    h->rec_ctl = NULL;
    h->cur = h->buf;
    ret_x(c, handle);
}
// NVNboolean nvnCommandBufferIsRecording(const NVNcommandBuffer *cmd)
NVN_API(nvnCommandBufferIsRecording) { ret_x(c, cmd_sync(c, "CommandBufferIsRecording")->recording != 0); }

// void nvnCommandBuffer{Call,Copy}Commands(NVNcommandBuffer *cmd, int n, const NVNcommandHandle *handles): each called
// recording's records go into the caller at the call position, since the game may add its control memory again before
// the caller is submitted. The driver closes the caller's commands with a segment entry and copies the called
// recording's control entries after it; its commands are not copied, so command memory does not grow.
static void call_commands(Cpu *c, const char *fn) {
    cmd_t *o = cmd_sync(c, fn);
    nvn_cmd_host *h = o->host;
    int n = (int)arg_x(c, 1);
    const uint64_t *handles = arg_p(c, 2);
    if (n < 0 || (n && !handles))
        rt_unimplemented(c, "nvn", "%s: command buffer %p: %d handles at %p", fn, (void *)o, n, (const void *)handles);
    for (int i = 0; i < n; i++) {
        nvn_copy *k = copy_pin(c, handles[i], fn);
        if (h->pending) segment_entry(c, o, fn);
        if (k->ctl_inner) {
            ctl_take(c, o, k->ctl_inner, k->ctl_inner, fn);
            if (h->rec_ctl) h->ctl_inner += k->ctl_inner;
        }
        size_t body = (size_t)k->bytes - 2 * sizeof(op_hdr);   // without its OP_BEGIN and OP_END
        reserve(h, body);
        memcpy(h->cur, (const uint8_t *)(k + 1) + sizeof(op_hdr), body);
        h->cur += body;
        h->deferred |= (int)k->deferred;
        copy_unpin(k);
    }
}
NVN_API(nvnCommandBufferCallCommands) { call_commands(c, "CommandBufferCallCommands"); }
NVN_API(nvnCommandBufferCopyCommands) { call_commands(c, "CommandBufferCopyCommands"); }
// void nvnDeviceFinalizeCommandHandle(NVNdevice *device, NVNcommandHandle handle): the console's driver returns at
// once; a copy goes with its control memory
NVN_API(nvnDeviceFinalizeCommandHandle) { (void)c; }

// ---------- submit
// What the console's driver reads when a submission is made: the deferred arrays. Done on the submitting thread, into
// the copy; a copy two threads submit at once would be resolved by both (the game is not seen to do it).
static void resolve_deferred(nvn_copy *k) {
    const uint8_t *p = (const uint8_t *)(k + 1), *end = p + k->bytes;
    for (; p < end; p += ((const op_hdr *)(const void *)p)->size) {
        const op_hdr *hd = (const op_hdr *)(const void *)p;
        void *payload = (uint8_t *)(uintptr_t)p + sizeof *hd;
        if (hd->op == OP_BIND_TEX || hd->op == OP_BIND_IMG || hd->op == OP_BIND_SAMP) {
            p_bind_handles *b = payload;
            if (b->src) memcpy(b->h, (const void *)(uintptr_t)b->src, sizeof b->h[0] * (size_t)b->n);
        } else if (hd->op == OP_PUSH_CONST) {
            p_push_const *b = payload;
            if (b->src) memcpy(b->w, (const void *)(uintptr_t)b->src, (size_t)b->bytes);
        }
    }
}

uint64_t nvn_submit_recordings(Cpu *c, const uint64_t *handles, int n) {
    if (n < 0 || n > SUBMIT_MAX || (n && !handles))
        rt_unimplemented(c, "nvn", "QueueSubmitCommands: %d handles at %p", n, (const void *)handles);
    nvn_copy *pins[SUBMIT_MAX];
    for (int i = 0; i < n; i++) pins[i] = copy_pin(c, handles[i], "QueueSubmitCommands");
    for (int i = 0; i < n; i++)
        if (pins[i]->deferred) resolve_deferred(pins[i]);
    ledger_frame_enter();
    uint64_t fence = gfx_backend()->submit(handles, n);
    ledger_frame_leave();
    for (int i = 0; i < n; i++) copy_unpin(pins[i]);
    return fence;
}

const char *nvn_op_name(uint32_t op) {
    static const char *const names[] = {
        "END", "JUMP", "BEGIN", "CALL", "SIGNAL_EVENT", "REPORT_COUNTER", "COPY_B2B", "UPDATE_UB", "SET_RT",
        "CLEAR_COLOR", "CLEAR_DS", "VIEWPORTS", "SCISSOR", "DEPTH_RANGE", "BIND_PROGRAM", "BIND_STATE", "BIND_VATTR",
        "BIND_VSTREAM", "BIND_VBUF", "BIND_UBO", "BIND_SSBO", "BIND_TEX", "BIND_SAMP", "STENCIL", "POLY_OFFSET", "DRAW",
        "DRAW_IDX", "DISPATCH", "BARRIER", "SET_TEX_POOL", "SET_SAMP_POOL", "COPY_B2T", "COPY_STRIDE", "BLEND_COLOR",
        "COPY_T2B", "BIND_IMG", "COPY_T2T", "RESET_COUNTER", "CLEAR_TEX", "DEPTH_CLAMP", "RENDER_ENABLE",
        "RASTER_DISCARD", "RENDER_COND", "PRIM_RESTART", "DRAW_IDX_IND", "DISPATCH_IND", "CLEAR_BUF", "PUSH_CONST",
        "WAIT_SYNC", "COPY_IMAGE_STRIDE", "DRAW_MESH", "DRAW_MESH_IND", "GEOMETRY_MODE", "WAIT_EVENT", "SAMPLE_MASK",
        "DEPTH_BOUNDS", "CONSERVATIVE", "SHADING_RATE_IMAGE", "SHADING_RATE_PALETTE", "SHADING_RATE_MODE", "SCRATCH",
        "BIND_ACCEL", "HINT",
    };
    _Static_assert(sizeof names / sizeof *names == OP_COUNT_, "one name per op");
    return op < OP_COUNT_ ? names[op] : "?";
}
