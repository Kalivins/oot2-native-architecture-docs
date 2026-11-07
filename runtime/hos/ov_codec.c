// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The video decoder service under the SDK's movie library (architecture D24b, the movie player): the AV1 entry points of
// nn::codec, answered here on the host with the decoder seam of runtime/video (dav1d).
//
// What the SDK does on the console. The movie library (movie::VideoDecoder, in the game's SDK module) reaches the video
// decoder service through nn::codec. It makes one image pool for each output picture (MakeDecodeImagePreference, then
// CalculateDecoderImagePoolBuffer and InitializeDecoderImagePool) and registers each with the decoder under an index
// (RegisterAv1DecoderBuffer). Registering puts the index in the decoder's queue of acquirable buffers. Acquire takes the
// oldest index and a sync handle; Release gives the buffer to the decoder to fill and puts it back in the queue at once,
// with a new sync handle that is reached when the decoder has written a picture of that release into it. The library
// polls nn::gs::TryWaitNativeSync on that handle, then reads the picture's properties and the picture itself
// (ReadDecoderOutputImageNv12). Submit takes one temporal unit with its time and an end-of-stream flag, and answers
// "busy" when the decoder cannot take more: the library then waits for the submission-ready event.
//
// Where this comes from. The protocol above was read from the behaviour of the SDK's own code in the movie ticket's runs,
// and its host side follows nxnative's codec service (dm/host/hle_codec_video.c, MIT, Copyright (c) 2026 seedds; the
// notices file has the row and licenses/nxnative-MIT.txt the text): the record kept for each buffer, the sync handle, the
// queue of acquirable buffers, the queue of released buffers and the order in which a picture goes into one, the
// preference record, and the two thresholds READY and HOLD are that file's logic, rewritten here. This is adapted logic,
// not a copy of the file. What differs is ours: the decode runs on a worker thread, the pictures wait in host memory as
// NV12, the events are kernel events signalled from any thread, and a generation makes Stop and Flush drop a picture
// that is still being decoded.
//
// What this file does with it.
//  - Submit copies the unit into a queue and returns at once: no decode runs on a guest thread. One worker thread for
//    each decoder slot decodes with the seam, converts the picture to NV12 in host memory, and hands it to the oldest
//    buffer that was released. The picture waits in host memory in the pool slot; ReadDecoderOutputImageNv12 copies it
//    into the guest's buffer at the strides it asks for.
//  - A buffer's sync handle is reached only when that buffer holds a picture of that release: a service that answered
//    "reached" for a buffer still waiting would never end the library's Flush loop, so the rule is tested. Stop and
//    Flush drop every unit and picture not yet delivered and bump a generation, so a picture the worker was still
//    decoding is discarded when it ends; buffers stay reserved, and a picture that was already in a buffer when the call
//    came stays in it until the library releases the buffer.
//  - Events are real: the submission-ready event is a kernel event attached to a system event of the SDK with
//    hos_bind_system_event, signalled from any thread when the decoder has room for a unit.
//
// Locking: one lock (g_mu) guards every table; the worker holds it only to move state, never across the decode. A guest
// thread that waits for the worker gives its virtual core up (rt_kernel_block_begin). No guest call is made with the
// lock held.
#include <stdlib.h>
#include <string.h>

#include "../video/av1.h"
#include "core.h"
#include "hos.h"
#include "plat.h"

// Results of module 111 (nn::codec): description 3603, the decoder cannot take the unit now; 3604, nothing to acquire or
// no picture in the buffer yet; 2004, a bad argument.
#define CODEC_R(desc) (0x6fu | ((uint32_t)(desc) << 9))
#define R_BUSY CODEC_R(3603)
#define R_NONE CODEC_R(3604)
#define R_BAD CODEC_R(2004)

#define MAX_DEC 8        // decoders at once: the movie library makes one for the movie playing, and one more during a change
#define MAX_BUF 32       // registered buffers of one decoder
#define ACQ_DEPTH 8      // acquirable buffers at once
#define HOLD 12          // units queued plus pictures waiting for a released buffer, at which a unit is refused (busy)
#define READY 4          // below this the decoder says it has room (submission-ready event)
#define HELD_MAX 16      // pictures waiting for a released buffer (a unit shows at most one; room for a surprise)
#define MAX_POOL 64
#define MAX_POOL_IMG 32
#define SYNC_MAGIC 0x56434d53u   // 'SMCV': a sync handle this service made

typedef struct { uint32_t magic, dec, idx, gen; } SyncHandle;   // 16 bytes, as nn::gs::NativeSyncHandle is passed

typedef struct Pool {
    uint64_t key;                  // the guest address of the VideoDecoderImagePoolType
    int used;
    uint32_t w, h, count;
    uint8_t *img[MAX_POOL_IMG];    // the NV12 picture in each image of the pool (host memory), or NULL
    uint16_t iw[MAX_POOL_IMG], ih[MAX_POOL_IMG];   // its size: the decoded picture's, which the pool's own size may not be
} Pool;

typedef struct Buf {
    int reg;
    Pool *pool;
    uint32_t pidx;
    int filled;
    uint32_t gen, filled_gen;
    uint16_t w, h;
    int64_t ts;
    uint8_t eos;
} Buf;

typedef struct Unit { uint8_t *data; size_t size; int64_t ts; int eos; uint32_t gen; } Unit;
typedef struct Pic { uint8_t *nv12; int w, h; int64_t ts; int eos; } Pic;

typedef struct Dec {
    int used, live, started, thread;
    uint64_t key;                  // the guest address of the Av1DecoderType
    uint32_t id;                   // slot + 1
    uint16_t w, h;
    uint64_t ev_addr;              // the SDK's SystemEventType (guest memory, one block for the slot)
    uint32_t ev_w;                 // its writable end
    Av1Decoder *av1;
    Buf buf[MAX_BUF];
    uint32_t fifo[ACQ_DEPTH], fhead, fcount;
    uint32_t pend[MAX_BUF], phead, pcount;
    Unit in[HOLD];
    int ihead, icount;
    Pic held[HELD_MAX];
    int nheld, eos_pending;
    uint32_t gen;                  // bumped by Stop, Flush and Finalize
    int reset_pending, busy;
    uint64_t submits, pictures, damaged, dropped;
    uint64_t stops, flushes, unregs;   // calls so far (the first of each is logged, the counts at Finalize)
    PlatCond work, idle;
} Dec;

// Sync point events (nn::syncpt::sdk): the movie library waits, in a multi-wait, on one for the sync handle of the picture it
// expects, and the SDK's own implementation sits on a graphics library the host replaces. Each is a system event of its own
// (manual clear: the library clears it itself after a hit) that is signalled when the buffer it was asked about holds the
// picture of that release. The event type's memory is never touched here: events are found by its guest address.
#define MAX_SPT 64
typedef struct SpEvent {
    uint64_t key;          // the guest address of the SyncPointEventType; 0 when free
    uint64_t ev_addr;      // its system event (a guest page of the slot's own, kept for the life of the process)
    uint32_t ev_w;
    int armed;
    SyncHandle want;
} SpEvent;
static SpEvent g_spt[MAX_SPT];

static PlatMutex g_mu = PLAT_MUTEX_INIT;
static Dec g_dec[MAX_DEC];
static Pool g_pool[MAX_POOL];
static uint32_t (*g_bind_event)(Cpu *, uint64_t, int) = hos_bind_system_event;
static void (*g_clear_event)(Cpu *, uint64_t);
static void (*g_init_holder)(Cpu *, uint64_t, uint64_t);

void hos_codec_set_event_binder(uint32_t (*bind)(Cpu *, uint64_t, int)) { g_bind_event = bind ? bind : hos_bind_system_event; }
void hos_codec_set_event_hooks(void (*clear)(Cpu *, uint64_t), void (*init_holder)(Cpu *, uint64_t, uint64_t)) {
    g_clear_event = clear;
    g_init_holder = init_holder;
}

// The SDK's own nn::os functions, called from a guest thread (an override runs on one), unless a test gave hooks.
static void os_clear_event(Cpu *c, uint64_t ev) {
    if (g_clear_event) g_clear_event(c, ev);
    else hos_call(c, hos_sdk(c, "_ZN2nn2os16ClearSystemEventEPNS0_15SystemEventTypeE"), 1, ev);
}
static void os_init_holder(Cpu *c, uint64_t holder, uint64_t ev) {
    if (g_init_holder) g_init_holder(c, holder, ev);
    else hos_call(c, hos_sdk(c, "_ZN2nn2os25InitializeMultiWaitHolderEPNS0_19MultiWaitHolderTypeEPNS0_15SystemEventTypeE"), 2, holder, ev);
}

// ---- lookups (g_mu held)
static Dec *dec_of(uint64_t key) {
    for (int i = 0; i < MAX_DEC; i++)
        if (g_dec[i].live && g_dec[i].key == key) return &g_dec[i];
    return NULL;
}
static Pool *pool_of(uint64_t key) {
    for (int i = 0; i < MAX_POOL; i++)
        if (g_pool[i].used && g_pool[i].key == key) return &g_pool[i];
    return NULL;
}

static int room(const Dec *d) { return d->icount + d->nheld < READY; }

// Sync point events armed for a buffer of this decoder stop waiting (g_mu held): an event is keyed by decoder id and
// buffer index, so one left armed could be signalled by a later decoder that takes the same id.
static void disarm_events(const Dec *d) {
    for (int k = 0; k < MAX_SPT; k++)
        if (g_spt[k].key && g_spt[k].armed && g_spt[k].want.dec == d->id) g_spt[k].armed = 0;
}

static void free_pic(Pic *p) {
    free(p->nv12);
    p->nv12 = NULL;
}

// Drop what has not been delivered: queued units, pictures waiting for a buffer, a pending end of stream (g_mu held).
static void drop_undelivered(Dec *d) {
    for (; d->icount; d->icount--, d->ihead = (d->ihead + 1) % HOLD) free(d->in[d->ihead].data);
    for (int i = 0; i < d->nheld; i++) free_pic(&d->held[i]);
    d->nheld = 0;
    d->eos_pending = 0;
}

static void fifo_push(Dec *d, uint32_t i) {
    if (d->fcount == ACQ_DEPTH) {
        rt_log("codec: decoder %u: the acquirable queue is full at buffer %u", d->id, i);
        return;
    }
    d->fifo[(d->fhead + d->fcount++) % ACQ_DEPTH] = i;
}
static void queue_drop(uint32_t *q, uint32_t *head, uint32_t *count, uint32_t cap, uint32_t i) {   // i leaves, the order stays
    uint32_t n = 0, tmp[MAX_BUF];
    for (uint32_t k = 0; k < *count; k++) {
        uint32_t v = q[(*head + k) % cap];
        if (v != i) tmp[n++] = v;
    }
    memcpy(q, tmp, n * sizeof *tmp);
    *head = 0;
    *count = n;
}

// Pictures and end of stream into the released buffers, in release order (g_mu held).
static void drain(Dec *d) {
    while (d->pcount && (d->nheld || d->eos_pending)) {
        uint32_t i = d->pend[d->phead];
        d->phead = (d->phead + 1) % MAX_BUF;
        d->pcount--;
        Buf *b = &d->buf[i];
        if (d->nheld) {
            Pic p = d->held[0];
            memmove(d->held, d->held + 1, (size_t)(HELD_MAX - 1) * sizeof d->held[0]);
            d->nheld--;
            b->w = (uint16_t)p.w;
            b->h = (uint16_t)p.h;
            b->ts = p.ts;
            b->eos = (uint8_t)p.eos;
            if (b->pool && b->pidx < b->pool->count) {
                free(b->pool->img[b->pidx]);
                b->pool->img[b->pidx] = p.nv12;
                b->pool->iw[b->pidx] = (uint16_t)p.w;
                b->pool->ih[b->pidx] = (uint16_t)p.h;
            } else {
                free(p.nv12);
            }
            if (!d->pictures++) rt_log("codec: av1 decoder %u: first picture %dx%d into buffer %u", d->id, p.w, p.h, i);
        } else {   // an end of stream with no picture of its own
            b->w = b->h = 0;
            b->ts = 0;
            b->eos = 1;
            d->eos_pending = 0;
        }
        b->filled = 1;
        b->filled_gen = b->gen;
        for (int k = 0; k < MAX_SPT; k++) {   // the sync point events waiting for this release of this buffer
            SpEvent *e = &g_spt[k];
            if (e->key && e->armed && e->want.dec == d->id && e->want.idx == i && b->filled_gen >= e->want.gen) {
                e->armed = 0;
                hos_signal(e->ev_w);
            }
        }
    }
}

// ---- the worker: one thread for each decoder slot, started at the slot's first decoder and never joined
static void *convert_nv12(const Av1Picture *p) {
    size_t w = (size_t)p->w, h = (size_t)p->h, cw = (w + 1) / 2, ch = (h + 1) / 2;
    uint8_t *out = malloc(w * h + 2 * cw * ch);
    if (!out) return NULL;
    for (size_t y = 0; y < h; y++) memcpy(out + y * w, p->plane[0] + (ptrdiff_t)y * p->stride[0], w);
    uint8_t *uv = out + w * h;
    for (size_t y = 0; y < ch; y++) {
        const uint8_t *u = p->plane[1] + (ptrdiff_t)y * p->stride[1], *v = p->plane[2] + (ptrdiff_t)y * p->stride[2];
        for (size_t x = 0; x < cw; x++) {
            uv[y * 2 * cw + 2 * x] = u[x];
            uv[y * 2 * cw + 2 * x + 1] = v[x];
        }
    }
    return out;
}

static void worker(void *arg) {
    Dec *d = arg;
    plat_mutex_lock(&g_mu);
    for (;;) {
        while (!(d->live && (d->reset_pending || (d->icount && d->nheld < HELD_MAX - 2)))) plat_cond_wait(&d->work, &g_mu);
        Av1Decoder *a = d->av1;
        if (d->reset_pending) {   // Stop or Flush: the decoder starts a new stream (the units it was sent are already gone)
            d->reset_pending = 0;
            d->busy = 1;
            plat_mutex_unlock(&g_mu);
            av1_flush(a);
            plat_mutex_lock(&g_mu);
            d->busy = 0;
            plat_cond_broadcast(&d->idle);
            continue;
        }
        Unit u = d->in[d->ihead];
        d->ihead = (d->ihead + 1) % HOLD;
        d->icount--;
        int was_room = d->icount + 1 + d->nheld < READY;
        if (!was_room && room(d)) hos_signal(d->ev_w);
        d->busy = 1;
        plat_mutex_unlock(&g_mu);

        Pic got[4];
        int ngot = 0, r = AV1_OK;
        if (u.size) {
            r = av1_send(a, u.data, u.size, u.ts);
            Av1Picture pic;
            while (ngot < 4 && av1_receive(a, &pic) == AV1_PICTURE) {
                void *nv = convert_nv12(&pic);
                if (!nv) continue;
                got[ngot++] = (Pic){nv, pic.w, pic.h, pic.ts, 0};
            }
        }
        free(u.data);

        plat_mutex_lock(&g_mu);
        d->busy = 0;
        if (d->live && u.gen == d->gen) {
            if (r != AV1_OK && !d->damaged++) rt_log("codec: av1 decoder %u: the first damaged unit (%zu bytes) was refused", d->id, u.size);
            for (int i = 0; i < ngot; i++) {
                if (d->nheld == HELD_MAX) {
                    d->dropped++;
                    free_pic(&got[i]);
                } else {
                    d->held[d->nheld++] = got[i];
                }
            }
            if (u.eos) {
                if (ngot && d->nheld) d->held[d->nheld - 1].eos = 1;
                else d->eos_pending = 1;
            }
            drain(d);
        } else {
            for (int i = 0; i < ngot; i++) free_pic(&got[i]);   // decoded before a Stop, a Flush or a Finalize: not delivered
        }
        plat_cond_broadcast(&d->idle);
    }
}

// ---- nn::codec::video
static void ov_video_init(Cpu *c) { ret_x(c, HOS_R_OK); }
HOS_OVERRIDE("_ZN2nn5codec5video10InitializeEv", ov_video_init, HOS_IMPLEMENTED,
             "the decoder service is the host's (ov_codec.c): nothing to open");
HOS_OVERRIDE("_ZN2nn5codec5video8FinalizeEv", ov_video_init, HOS_IMPLEMENTED, "the decoder service is the host's: nothing to close");

// ---- image pools. The preference is the host's own record (nothing else reads it): width, height, output format,
// tiling mode and a tag, in the first 24 bytes of the caller's result block.
typedef struct { uint16_t w, h; uint32_t fmt, tiling, tag, pad[2]; } Pref;
#define PREF_TAG 0x76705054u   // 'TPpv'

// VideoDecoderImagePreference MakeDecodeImagePreference(ushort w, ushort h, VideoDecoderOutputFormat, ImageTilingMode):
// the structure comes back through x8
static void ov_make_pref(Cpu *c) {
    Pref p = {(uint16_t)arg_x(c, 0), (uint16_t)arg_x(c, 1), (uint32_t)arg_x(c, 2), (uint32_t)arg_x(c, 3), PREF_TAG, {0, 0}};
    if (!hos_out(c->x[8], &p, sizeof p)) rt_guest_stop(c, "nn::codec::MakeDecodeImagePreference: the result block is not writable");
}
HOS_OVERRIDE("_ZN2nn5codec25MakeDecodeImagePreferenceEttNS0_24VideoDecoderOutputFormatENS0_15ImageTilingModeE", ov_make_pref,
             HOS_IMPLEMENTED, "records the size and the format in the result block; the pool needs no tiling");

// Result CalculateDecoderImagePoolBuffer(size_t *size, size_t *alignment, const Preference &, uint count): the pictures
// are kept in host memory, so the pool's own memory is a page of the caller's and no more
static void ov_pool_size(Cpu *c) {
    uint64_t size = 0x1000, align = 0x1000;
    if (arg_x(c, 0)) hos_out(arg_x(c, 0), &size, sizeof size);
    if (arg_x(c, 1)) hos_out(arg_x(c, 1), &align, sizeof align);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec31CalculateDecoderImagePoolBufferEPmS1_RKNS0_27VideoDecoderImagePreferenceEj", ov_pool_size,
             HOS_IMPLEMENTED, "one page: the decoded pictures stay in host memory");

// Result InitializeDecoderImagePool(VideoDecoderImagePoolType *pool, void *memory, size_t size, const Preference &, uint count)
static void ov_pool_init(Cpu *c) {
    uint64_t key = arg_x(c, 0);
    Pref p;
    memset(&p, 0, sizeof p);
    if (!key || !arg_x(c, 3) || !hos_in(&p, arg_x(c, 3), sizeof p) || p.tag != PREF_TAG)
        rt_guest_stop(c, "nn::codec::InitializeDecoderImagePool: no image preference made by MakeDecodeImagePreference");
    uint32_t count = (uint32_t)arg_x(c, 4);
    if (!count) count = 1;
    if (count > MAX_POOL_IMG) hos_gap(c, "movie", "nn::codec::InitializeDecoderImagePool: %u images in one pool, the host keeps %d", count, MAX_POOL_IMG);
    plat_mutex_lock(&g_mu);
    Pool *pl = pool_of(key);
    if (!pl)
        for (int i = 0; i < MAX_POOL && !pl; i++)
            if (!g_pool[i].used) pl = &g_pool[i];
    if (!pl) {
        plat_mutex_unlock(&g_mu);
        hos_gap(c, "movie", "nn::codec::InitializeDecoderImagePool: more than %d image pools", MAX_POOL);
    }
    for (uint32_t i = 0; i < MAX_POOL_IMG; i++) free(pl->img[i]);
    memset(pl, 0, sizeof *pl);
    pl->used = 1;
    pl->key = key;
    pl->w = p.w;
    pl->h = p.h;
    pl->count = count;
    plat_mutex_unlock(&g_mu);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec26InitializeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeEPvmRKNS0_27VideoDecoderImagePreferenceEj",
             ov_pool_init, HOS_IMPLEMENTED, "a host table of pictures, one for each image of the pool");

// void FinalizeDecoderImagePool(pool *): the decoders that still hold a buffer of the pool keep it, with no pool to fill
static void ov_pool_fini(Cpu *c) {
    plat_mutex_lock(&g_mu);
    Pool *pl = pool_of(arg_x(c, 0));
    if (pl) {
        for (int k = 0; k < MAX_DEC; k++)
            for (int i = 0; i < MAX_BUF; i++)
                if (g_dec[k].buf[i].pool == pl) g_dec[k].buf[i].pool = NULL;
        for (uint32_t i = 0; i < MAX_POOL_IMG; i++) free(pl->img[i]);
        memset(pl, 0, sizeof *pl);
    }
    plat_mutex_unlock(&g_mu);
}
HOS_OVERRIDE("_ZN2nn5codec24FinalizeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeE", ov_pool_fini, HOS_IMPLEMENTED,
             "frees the host pictures of the pool");

// Result ReadDecoderOutputImageNv12(const Pool *, uint index, ushort x, y, w, h, void *y_dst, size_t y_stride, size_t y_size,
// void *uv_dst, size_t uv_stride, size_t uv_size): the picture of the image, as NV12 rows at the strides asked for; mid
// grey when the image holds none
static void ov_read_nv12(Cpu *c) {
    uint64_t key = arg_x(c, 0);
    uint32_t idx = (uint32_t)arg_x(c, 1);
    unsigned x = (uint16_t)arg_x(c, 2), y = (uint16_t)arg_x(c, 3), w = (uint16_t)arg_x(c, 4), h = (uint16_t)arg_x(c, 5);
    uint64_t ydst = arg_x(c, 6), ystride = arg_x(c, 7), ysize = arg_x(c, 8), uvdst = arg_x(c, 9), uvstride = arg_x(c, 10),
             uvsize = arg_x(c, 11);
    unsigned cw = (w + 1) / 2, ch = (h + 1) / 2;
    if (!w || !h || (x & 1) || (y & 1) || ystride < w || uvstride < 2 * (uint64_t)cw || ysize < ystride * (h - 1) + w ||
        uvsize < uvstride * (ch - 1) + 2 * (uint64_t)cw) {
        ret_x(c, R_BAD);
        return;
    }
    uint8_t *ybuf = malloc((size_t)w * h), *uvbuf = malloc((size_t)2 * cw * ch);
    if (!ybuf || !uvbuf) rt_fatal("codec: out of memory for a %ux%u picture", w, h);
    plat_mutex_lock(&g_mu);
    Pool *pl = pool_of(key);
    const uint8_t *img = pl && idx < pl->count ? pl->img[idx] : NULL;
    int ok = 1;
    memset(ybuf, 16, (size_t)w * h);   // black where the picture is smaller than the region asked for, and all of it for an empty image
    memset(uvbuf, 128, (size_t)2 * cw * ch);
    if (img) {
        unsigned iw = pl->iw[idx], ih = pl->ih[idx], icw = (iw + 1) / 2, ich = (ih + 1) / 2;
        unsigned ncols = x < iw ? (w < iw - x ? w : iw - x) : 0, nrows = y < ih ? (h < ih - y ? h : ih - y) : 0;
        for (unsigned r = 0; r < nrows; r++) memcpy(ybuf + (size_t)r * w, img + (size_t)(y + r) * iw + x, ncols);
        const uint8_t *uv = img + (size_t)iw * ih;
        unsigned ccols = x / 2 < icw ? (cw < icw - x / 2 ? cw : icw - x / 2) : 0, crows = y / 2 < ich ? (ch < ich - y / 2 ? ch : ich - y / 2) : 0;
        for (unsigned r = 0; r < crows; r++) memcpy(uvbuf + (size_t)r * 2 * cw, uv + (size_t)(y / 2 + r) * 2 * icw + x, 2 * (size_t)ccols);
    }
    plat_mutex_unlock(&g_mu);
    if (ok) {
        for (unsigned r = 0; r < h && ok; r++) ok = hos_out(ydst + (uint64_t)r * ystride, ybuf + (size_t)r * w, w);
        for (unsigned r = 0; r < ch && ok; r++) ok = hos_out(uvdst + (uint64_t)r * uvstride, uvbuf + (size_t)r * 2 * cw, 2 * (size_t)cw);
    }
    free(ybuf);
    free(uvbuf);
    ret_x(c, ok ? HOS_R_OK : R_BAD);
}
HOS_OVERRIDE("_ZN2nn5codec26ReadDecoderOutputImageNv12EPKNS0_25VideoDecoderImagePoolTypeEjttttPvmmS4_mm", ov_read_nv12,
             HOS_IMPLEMENTED, "copies the host's NV12 picture into the caller's two planes at its strides");

// ---- the AV1 decoder
// Result InitializeAv1DecoderByRemoteMemory(Av1DecoderType *, size_t, ushort w, ushort h, uint refs, VideoDecoderInputFormat,
// VideoDecoderOutputFormat, VideoDecoderOutputMode)
// Result InitializeAv1DecoderByTransferMemory(Av1DecoderType *, void *, size_t, size_t, ushort w, ushort h, uint refs, ...)
static void decoder_init(Cpu *c, uint64_t key, unsigned w, unsigned h) {
    if (!key) rt_guest_stop(c, "nn::codec::InitializeAv1Decoder: null decoder");
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(key);
    if (d) {   // the same object initialised again: the library finalised it by dropping its memory
        d->live = 0;
        d->gen++;
        drop_undelivered(d);
        disarm_events(d);
        d->started = 0;
        int token = rt_kernel_block_begin();   // the wait is bounded by one decode, and the core is not kept for it
        while (d->busy) plat_cond_wait(&d->idle, &g_mu);
        rt_kernel_block_end(token);
        Av1Decoder *old = d->av1;
        d->av1 = NULL;
        plat_mutex_unlock(&g_mu);
        av1_close(old);
        plat_mutex_lock(&g_mu);
    } else {
        for (int i = 0; i < MAX_DEC && !d; i++)
            if (!g_dec[i].used) d = &g_dec[i];
    }
    if (!d) {
        plat_mutex_unlock(&g_mu);
        hos_gap(c, "movie", "nn::codec::InitializeAv1Decoder: more than %d decoders at once", MAX_DEC);
    }
    d->used = 1;   // taken, not yet live
    d->key = key;
    d->id = (uint32_t)(d - g_dec) + 1;
    plat_mutex_unlock(&g_mu);

    Av1Decoder *a = av1_open();
    if (!a) hos_gap(c, "movie", "nn::codec::InitializeAv1Decoder: the host has no AV1 decoder (%s): the movie ticket's decoder needs the dav1d kit", av1_describe());
    if (!d->ev_addr) d->ev_addr = mem_kernel_alloc(0x1000);   // the SDK's system event lives in its own page, for the life of the slot
    d->ev_w = g_bind_event(c, d->ev_addr, 1);
    plat_mutex_lock(&g_mu);
    d->w = (uint16_t)w;
    d->h = (uint16_t)h;
    d->av1 = a;
    d->started = d->reset_pending = d->busy = 0;
    d->icount = d->ihead = d->nheld = d->eos_pending = 0;
    d->fcount = d->fhead = d->pcount = d->phead = 0;
    d->submits = d->pictures = d->damaged = d->dropped = d->stops = d->flushes = d->unregs = 0;
    memset(d->buf, 0, sizeof d->buf);
    d->live = 1;
    int start = !d->thread;
    d->thread = 1;
    plat_mutex_unlock(&g_mu);
    if (start && !plat_thread_start(worker, d, 4ull << 20)) rt_fatal("codec: cannot start the AV1 decoder's worker thread");
    rt_log("codec: av1 decoder %u: %ux%u, %s", d->id, w, h, av1_describe());
}
static void ov_av1_init_remote(Cpu *c) {
    decoder_init(c, arg_x(c, 0), (uint16_t)arg_x(c, 2), (uint16_t)arg_x(c, 3));
    ret_x(c, HOS_R_OK);
}
static void ov_av1_init_transfer(Cpu *c) {
    decoder_init(c, arg_x(c, 0), (uint16_t)arg_x(c, 4), (uint16_t)arg_x(c, 5));
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec34InitializeAv1DecoderByRemoteMemoryEPNS0_14Av1DecoderTypeEmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE",
             ov_av1_init_remote, HOS_IMPLEMENTED, "an AV1 decoder on the host: a worker thread and the decoder seam (runtime/video)");
HOS_OVERRIDE("_ZN2nn5codec36InitializeAv1DecoderByTransferMemoryEPNS0_14Av1DecoderTypeEPvmmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE",
             ov_av1_init_transfer, HOS_IMPLEMENTED, "as the remote-memory form: the transfer memory is not needed");

// Result FinalizeAv1Decoder(Av1DecoderType *): the worker is idle when it returns, the decoder closed, no picture kept
static void ov_av1_fini(Cpu *c) {
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 0));
    if (!d) {
        plat_mutex_unlock(&g_mu);
        ret_x(c, HOS_R_OK);
        return;
    }
    d->live = 0;
    d->gen++;
    drop_undelivered(d);
    disarm_events(d);
    int token = rt_kernel_block_begin();
    while (d->busy) plat_cond_wait(&d->idle, &g_mu);
    Av1Decoder *a = d->av1;
    d->av1 = NULL;
    uint64_t pics = d->pictures, bad = d->damaged, stops = d->stops, flushes = d->flushes, unregs = d->unregs;
    uint32_t id = d->id;
    d->used = 0;
    plat_mutex_unlock(&g_mu);
    rt_kernel_block_end(token);
    av1_close(a);
    rt_log("codec: av1 decoder %u finalised after %llu pictures (%llu units refused as damaged, %llu Stop, %llu Flush, %llu Unregister)", id,
           (unsigned long long)pics, (unsigned long long)bad, (unsigned long long)stops, (unsigned long long)flushes,
           (unsigned long long)unregs);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec18FinalizeAv1DecoderEPNS0_14Av1DecoderTypeE", ov_av1_fini, HOS_IMPLEMENTED,
             "waits for the worker, closes the decoder, drops every unit and picture");

// Result RegisterAv1DecoderBuffer(type, uint index, pool, uint pool_index): the buffer joins the acquirable queue
static void ov_av1_register(Cpu *c) {
    uint32_t i = (uint32_t)arg_x(c, 1);
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 0));
    if (!d || i >= MAX_BUF) {
        plat_mutex_unlock(&g_mu);
        ret_x(c, R_BAD);
        return;
    }
    Buf *b = &d->buf[i];
    memset(b, 0, sizeof *b);
    b->pool = pool_of(arg_x(c, 2));
    b->pidx = (uint32_t)arg_x(c, 3);
    b->reg = 1;
    fifo_push(d, i);
    plat_mutex_unlock(&g_mu);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec24RegisterAv1DecoderBufferEPNS0_14Av1DecoderTypeEjPNS0_25VideoDecoderImagePoolTypeEj", ov_av1_register,
             HOS_IMPLEMENTED, "a buffer of the host decoder, acquirable at once");

// Result UnregisterAv1DecoderBuffer(type, uint index): the buffer is neither acquirable nor awaited any more
static void ov_av1_unregister(Cpu *c) {
    uint32_t i = (uint32_t)arg_x(c, 1);
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 0));
    if (!d || i >= MAX_BUF) {
        plat_mutex_unlock(&g_mu);
        ret_x(c, R_BAD);
        return;
    }
    if (!d->unregs++) rt_log("codec: av1 decoder %u: the first UnregisterAv1DecoderBuffer (buffer %u)", d->id, i);
    d->buf[i].reg = 0;
    d->buf[i].filled = 0;
    d->buf[i].pool = NULL;
    d->buf[i].gen++;
    queue_drop(d->fifo, &d->fhead, &d->fcount, ACQ_DEPTH, i);
    queue_drop(d->pend, &d->phead, &d->pcount, MAX_BUF, i);
    plat_mutex_unlock(&g_mu);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec26UnregisterAv1DecoderBufferEPNS0_14Av1DecoderTypeEj", ov_av1_unregister, HOS_IMPLEMENTED,
             "the buffer leaves the host decoder");

// Result StartAv1Decoder(type): ready for units from here on, whatever became of an earlier signal
static void ov_av1_start(Cpu *c) {
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 0));
    if (d) {
        d->started = 1;
        if (room(d)) hos_signal(d->ev_w);
    }
    plat_mutex_unlock(&g_mu);
    ret_x(c, d ? HOS_R_OK : R_BAD);
}
HOS_OVERRIDE("_ZN2nn5codec15StartAv1DecoderEPNS0_14Av1DecoderTypeE", ov_av1_start, HOS_IMPLEMENTED,
             "the decoder takes units; the submission-ready event is signalled while there is room");

// Result StopAv1Decoder(type), FlushAv1Decoder(type): every unit and picture not yet delivered is dropped, the decoder
// starts a new stream at the next key frame; a picture the worker still decodes is discarded when it ends (generation).
// A picture already in a buffer stays there, readable, until the library releases the buffer. The first call of each kind
// is logged, and the counts come with the Finalize line.
static void reset_decoder(Cpu *c, int flush) {
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 0));
    if (d) {
        if (!(flush ? d->flushes++ : d->stops++))
            rt_log("codec: av1 decoder %u: the first %s (%d units queued, %d pictures waiting)", d->id, flush ? "FlushAv1Decoder" : "StopAv1Decoder",
                   d->icount, d->nheld);
        d->gen++;
        drop_undelivered(d);
        d->reset_pending = 1;
        plat_cond_signal(&d->work);
        if (room(d)) hos_signal(d->ev_w);
    }
    plat_mutex_unlock(&g_mu);
    ret_x(c, d ? HOS_R_OK : R_BAD);
}
static void ov_av1_stop(Cpu *c) { reset_decoder(c, 0); }
static void ov_av1_flush(Cpu *c) { reset_decoder(c, 1); }
HOS_OVERRIDE("_ZN2nn5codec14StopAv1DecoderEPNS0_14Av1DecoderTypeE", ov_av1_stop, HOS_IMPLEMENTED,
             "drops every unit and picture not delivered; the buffers stay reserved, a picture already in one stays until its release");
HOS_OVERRIDE("_ZN2nn5codec15FlushAv1DecoderEPNS0_14Av1DecoderTypeE", ov_av1_flush, HOS_IMPLEMENTED,
             "drops every unit and picture not delivered; the buffers stay reserved, a picture already in one stays until its release");

// SystemEventType *GetAv1DecoderSubmissionReadyEvent(type)
static void ov_av1_event(Cpu *c) {
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 0));
    uint64_t ev = d ? d->ev_addr : 0;
    plat_mutex_unlock(&g_mu);
    ret_x(c, ev);
}
HOS_OVERRIDE("_ZN2nn5codec33GetAv1DecoderSubmissionReadyEventEPNS0_14Av1DecoderTypeE", ov_av1_event, HOS_IMPLEMENTED,
             "the decoder's own system event, signalled while it has room for a unit");

// Result SubmitAv1DecoderInputBuffer(uint64_t *id, type, const void *data, size_t size, TimeSpan time, bool end_of_stream):
// the unit is copied and queued and the call returns; busy (3603) when the queue is full
static void ov_av1_submit(Cpu *c) {
    uint64_t data = arg_x(c, 2), size = arg_x(c, 3);
    int64_t ts = (int64_t)arg_x(c, 4);
    int eos = (arg_x(c, 5) & 0xff) != 0;
    uint8_t *copy = NULL;
    if (size) {
        if (size > (64u << 20)) {
            ret_x(c, R_BAD);
            return;
        }
        copy = malloc((size_t)size);
        if (!copy) rt_fatal("codec: out of memory for a %llu byte unit", (unsigned long long)size);
        if (!hos_in(copy, data, size)) {
            free(copy);
            ret_x(c, R_BAD);
            return;
        }
    }
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 1));
    if (!d) {
        plat_mutex_unlock(&g_mu);
        free(copy);
        ret_x(c, R_BAD);
        return;
    }
    if (d->icount + d->nheld >= HOLD) {
        plat_mutex_unlock(&g_mu);
        free(copy);
        ret_x(c, R_BUSY);
        return;
    }
    d->in[(d->ihead + d->icount++) % HOLD] = (Unit){copy, (size_t)size, ts, eos, d->gen};
    uint64_t n = ++d->submits;
    plat_cond_signal(&d->work);
    if (room(d)) hos_signal(d->ev_w);
    plat_mutex_unlock(&g_mu);
    if (arg_x(c, 0)) hos_out(arg_x(c, 0), &n, sizeof n);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec27SubmitAv1DecoderInputBufferEPmPNS0_14Av1DecoderTypeEPKvmNS_8TimeSpanEb", ov_av1_submit, HOS_IMPLEMENTED,
             "copies the unit into the worker's queue and returns; busy when the queue is full");

// Result SubmitEncryptedAv1DecoderInputBuffer(...): protected streams are not served
static void ov_av1_submit_encrypted(Cpu *c) { ret_x(c, R_BAD); }
HOS_OVERRIDE("_ZN2nn5codec36SubmitEncryptedAv1DecoderInputBufferEPmPNS0_14Av1DecoderTypeEPKNS0_18EncryptionPropertyEPKvmNS_8TimeSpanEb",
             ov_av1_submit_encrypted, HOS_ABSENT, "answers a bad argument: no protected stream is served");

// Result AcquireAv1DecoderOutputBuffer(uint *index, NativeSyncHandle *sync, type): the oldest acquirable buffer, whether
// or not it holds its picture yet (the sync handle says when)
static void ov_av1_acquire(Cpu *c) {
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 2));
    if (!d) {
        plat_mutex_unlock(&g_mu);
        ret_x(c, R_BAD);
        return;
    }
    if (!d->fcount) {
        plat_mutex_unlock(&g_mu);
        ret_x(c, R_NONE);
        return;
    }
    uint32_t i = d->fifo[d->fhead];
    d->fhead = (d->fhead + 1) % ACQ_DEPTH;
    d->fcount--;
    SyncHandle s = {SYNC_MAGIC, d->id, i, d->buf[i].gen};
    plat_mutex_unlock(&g_mu);
    if (arg_x(c, 0)) hos_out(arg_x(c, 0), &i, sizeof i);
    if (arg_x(c, 1)) hos_out(arg_x(c, 1), &s, sizeof s);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec29AcquireAv1DecoderOutputBufferEPjPNS_2gs16NativeSyncHandleEPNS0_14Av1DecoderTypeE", ov_av1_acquire,
             HOS_IMPLEMENTED, "the oldest acquirable buffer and its sync handle; 3604 when there is none");

// void ReleaseAv1DecoderOutputBuffer(type, uint index, NativeSyncHandle sync): the buffer waits for the next picture and is
// acquirable again at once, under a new generation
static void ov_av1_release(Cpu *c) {
    uint32_t i = (uint32_t)arg_x(c, 1);
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 0));
    if (d && i < MAX_BUF && d->buf[i].reg) {
        int was = room(d);
        Buf *b = &d->buf[i];
        b->gen++;
        b->filled = 0;
        queue_drop(d->pend, &d->phead, &d->pcount, MAX_BUF, i);   // a buffer waits for one picture, not two
        if (d->pcount < MAX_BUF) d->pend[(d->phead + d->pcount++) % MAX_BUF] = i;
        fifo_push(d, i);
        drain(d);
        plat_cond_signal(&d->work);   // a held picture was delivered: the worker may decode on
        if (!was && room(d)) hos_signal(d->ev_w);
    } else {
        rt_log("codec: ReleaseAv1DecoderOutputBuffer: buffer %u is not registered", i);
    }
    plat_mutex_unlock(&g_mu);
}
HOS_OVERRIDE("_ZN2nn5codec29ReleaseAv1DecoderOutputBufferEPNS0_14Av1DecoderTypeEjNS_2gs16NativeSyncHandleE", ov_av1_release,
             HOS_IMPLEMENTED, "the buffer waits for the next picture and is acquirable again, under a new generation");

// Result GetAv1DecoderOutputBufferProperties(ushort *w, ushort *h, ushort *chroma_w, ushort *chroma_h, TimeSpan *time,
// bool *end_of_stream, type, uint index): the picture's size, time and end of stream; 3604 until the buffer holds it
static void ov_av1_props(Cpu *c) {
    uint32_t i = (uint32_t)arg_x(c, 7);
    plat_mutex_lock(&g_mu);
    Dec *d = dec_of(arg_x(c, 6));
    int ok = d && i < MAX_BUF && d->buf[i].filled;
    Buf b = ok ? d->buf[i] : (Buf){0};
    plat_mutex_unlock(&g_mu);
    if (!d || i >= MAX_BUF) {
        ret_x(c, R_BAD);
        return;
    }
    if (!ok) {
        ret_x(c, R_NONE);
        return;
    }
    uint16_t cw = (uint16_t)((b.w + 1) / 2), ch = (uint16_t)((b.h + 1) / 2);
    uint8_t eos = b.eos;
    if (arg_x(c, 0)) hos_out(arg_x(c, 0), &b.w, sizeof b.w);
    if (arg_x(c, 1)) hos_out(arg_x(c, 1), &b.h, sizeof b.h);
    if (arg_x(c, 2)) hos_out(arg_x(c, 2), &cw, sizeof cw);
    if (arg_x(c, 3)) hos_out(arg_x(c, 3), &ch, sizeof ch);
    if (arg_x(c, 4)) hos_out(arg_x(c, 4), &b.ts, sizeof b.ts);
    if (arg_x(c, 5)) hos_out(arg_x(c, 5), &eos, sizeof eos);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn5codec35GetAv1DecoderOutputBufferPropertiesEPtS1_S1_S1_PNS_8TimeSpanEPbPNS0_14Av1DecoderTypeEj", ov_av1_props,
             HOS_IMPLEMENTED, "the filled buffer's picture size, time and end of stream; 3604 until it is filled");

// Whether the buffer of a sync handle holds a picture of the release the handle was made for (g_mu held). A handle this service did
// not make, and a handle of a buffer's first life (nothing was waited for), read as reached.
static int sync_reached(const SyncHandle *s) {
    if (s->magic != SYNC_MAGIC || !s->gen) return 1;
    Dec *d = s->dec && s->dec <= MAX_DEC && g_dec[s->dec - 1].live ? &g_dec[s->dec - 1] : NULL;
    if (d && s->idx < MAX_BUF) return d->buf[s->idx].filled && d->buf[s->idx].filled_gen >= s->gen;
    return 1;
}

// bool nn::gs::TryWaitNativeSync(const NativeSyncHandle &): reached when the buffer of the handle holds a picture of that release
static void ov_try_wait_sync(Cpu *c) {
    SyncHandle s;
    memset(&s, 0, sizeof s);
    if (!arg_x(c, 0) || !hos_in(&s, arg_x(c, 0), sizeof s)) {
        ret_x(c, 1);
        return;
    }
    plat_mutex_lock(&g_mu);
    int reached = sync_reached(&s);
    plat_mutex_unlock(&g_mu);
    ret_x(c, reached ? 1 : 0);
}
HOS_OVERRIDE("_ZN2nn2gs17TryWaitNativeSyncERKNS0_16NativeSyncHandleE", ov_try_wait_sync, HOS_IMPLEMENTED,
             "reached when the buffer holds a picture of the release the handle was made for");

// ---- sync point events
static SpEvent *spt_find(uint64_t key) {
    for (int i = 0; i < MAX_SPT; i++)
        if (g_spt[i].key == key) return &g_spt[i];
    return NULL;
}

// Result CreateSyncPointEvent(SyncPointEventType *): a system event of its own, bound the first time the slot is used
static void ov_spt_create(Cpu *c) {
    uint64_t key = arg_x(c, 0);
    if (!key) rt_guest_stop(c, "nn::syncpt::sdk::CreateSyncPointEvent: null event");
    plat_mutex_lock(&g_mu);
    SpEvent *e = spt_find(key);
    for (int i = 0; i < MAX_SPT && !e; i++)
        if (!g_spt[i].key && g_spt[i].ev_addr) e = &g_spt[i];   // a slot with its event already made first
    for (int i = 0; i < MAX_SPT && !e; i++)
        if (!g_spt[i].key) e = &g_spt[i];
    if (!e) {
        plat_mutex_unlock(&g_mu);
        hos_gap(c, "movie", "nn::syncpt::sdk::CreateSyncPointEvent: more than %d sync point events at once", MAX_SPT);
    }
    e->key = key;
    e->armed = 0;
    int bind = !e->ev_addr;
    plat_mutex_unlock(&g_mu);
    if (bind) {
        uint64_t page = mem_kernel_alloc(0x1000);
        uint32_t w = g_bind_event(c, page, 0);
        plat_mutex_lock(&g_mu);
        e->ev_addr = page;
        e->ev_w = w;
        plat_mutex_unlock(&g_mu);
    }
    os_clear_event(c, e->ev_addr);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn6syncpt3sdk20CreateSyncPointEventEPNS1_18SyncPointEventTypeE", ov_spt_create, HOS_IMPLEMENTED,
             "a manual-clear system event of the host's, signalled when the decoder fills the buffer asked about");

// void InitializeMultiWaitHolder(MultiWaitHolderType *, SyncPointEventType *): the holder waits on the event's system event
static void ov_spt_holder(Cpu *c) {
    plat_mutex_lock(&g_mu);
    SpEvent *e = spt_find(arg_x(c, 1));
    uint64_t ev = e ? e->ev_addr : 0;
    plat_mutex_unlock(&g_mu);
    if (!ev) rt_guest_stop(c, "nn::syncpt::sdk::InitializeMultiWaitHolder: an event CreateSyncPointEvent never made");
    os_init_holder(c, arg_x(c, 0), ev);
}
HOS_OVERRIDE("_ZN2nn6syncpt3sdk25InitializeMultiWaitHolderEPNS_2os19MultiWaitHolderTypeEPNS1_18SyncPointEventTypeE", ov_spt_holder,
             HOS_IMPLEMENTED, "links the event's system event to the multi-wait holder with the SDK's own nn::os call");

// Result RequestSyncPointEvent(SyncPointEventType *, const SyncPointHandle &): signalled when the awaited buffer holds the
// picture of that release (at once if it does); the event is cleared first
static void ov_spt_request(Cpu *c) {
    uint64_t key = arg_x(c, 0);   // read first: the guest call below may change x0..x18
    SyncHandle h;
    memset(&h, 0, sizeof h);
    int have = arg_x(c, 1) && hos_in(&h, arg_x(c, 1), sizeof h);
    plat_mutex_lock(&g_mu);
    SpEvent *e = spt_find(key);
    uint64_t ev = e ? e->ev_addr : 0;
    if (e) e->armed = 0;
    plat_mutex_unlock(&g_mu);
    if (!ev) rt_guest_stop(c, "nn::syncpt::sdk::RequestSyncPointEvent: an event CreateSyncPointEvent never made");
    os_clear_event(c, ev);
    plat_mutex_lock(&g_mu);
    e = spt_find(key);
    if (e) {
        e->want = h;
        if (!have || sync_reached(&h)) hos_signal(e->ev_w);
        else e->armed = 1;
    }
    plat_mutex_unlock(&g_mu);
    ret_x(c, HOS_R_OK);
}
HOS_OVERRIDE("_ZN2nn6syncpt3sdk21RequestSyncPointEventEPNS1_18SyncPointEventTypeERKNS1_15SyncPointHandleE", ov_spt_request,
             HOS_IMPLEMENTED, "signalled when the decoder has filled the buffer of the handle with that release's picture");

// void CancelSyncPointEvent(SyncPointEventType *): no longer awaited, the event cleared
static void ov_spt_cancel(Cpu *c) {
    plat_mutex_lock(&g_mu);
    SpEvent *e = spt_find(arg_x(c, 0));
    uint64_t ev = e ? e->ev_addr : 0;
    if (e) e->armed = 0;
    plat_mutex_unlock(&g_mu);
    if (ev) os_clear_event(c, ev);
}
HOS_OVERRIDE("_ZN2nn6syncpt3sdk20CancelSyncPointEventEPNS1_18SyncPointEventTypeE", ov_spt_cancel, HOS_IMPLEMENTED,
             "the event is no longer awaited and is cleared");

// void DestroySyncPointEvent(SyncPointEventType *): the slot is free for another event (its system event is kept)
static void ov_spt_destroy(Cpu *c) {
    plat_mutex_lock(&g_mu);
    SpEvent *e = spt_find(arg_x(c, 0));
    if (e) {
        e->armed = 0;
        e->key = 0;
    }
    plat_mutex_unlock(&g_mu);
}
HOS_OVERRIDE("_ZN2nn6syncpt3sdk21DestroySyncPointEventEPNS1_18SyncPointEventTypeE", ov_spt_destroy, HOS_IMPLEMENTED,
             "the slot is free for another event");
