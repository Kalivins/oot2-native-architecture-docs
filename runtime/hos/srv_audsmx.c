// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// audsmx:u, the sound mixer service of the console's newer SDKs (architecture D9): the PCM output streams its audio library
// opens beside the renderer. A stream's buffer is memory the game shares with the service; the service takes frames out of
// it on a 5 ms grid and hands them to an audio sink of its own. Commands (id: request -> reply):
//   audsmx:u      0(copy handle: the process): the user session
//   user session  1(u32 device, u32 flags, u64 applet resource user id, u64 buffer, u64 size; a name buffer) -> u32 id and the
//                 stream      2 and 3(a buffer of u32 ids): engage and disengage streams
//   stream        0(u32 sample type, channels, rate, frames, prefill, flags): open   1: close   2: u32 state (0 closed, 1 open,
//                 2 engaged)   3: f32 gain   4(f32 gain, u32 pad, s64 ramp ns)   5 and 6: acquire and release the state
//                 event   7(u32 mode, s32 threshold) and 8: acquire and release the data event   9 to 11: clock, prime, purge
//                 (not supported)   12: rename
// The device manager (srv_auddmg.c) engages a stream when the game attaches it to the output device, and disengages it when
// it detaches it.
//
// The shared buffer: the u32 read index at +0x44 (the service advances it), the u32 write index at +0x1004 (the game advances
// it) and, from +0x1040, a ring of `frames + 1` stereo frames of 16-bit or float PCM at 48 kHz. Each 5 ms tick takes up to 240
// frames of each engaged stream, applies its gain, writes them to the stream's sink and moves the read index with a
// compare-and-swap (the game may be moving it too: that is a fault, named in the log, and the stream stops). The data event is
// signalled (mode 2) while at least `threshold` ring slots are free, which is how the game learns it may write more. A sink that
// is full leaves the frames in the game's ring for the next tick.
//
// Adapted from oot-pc-recomp runtime/audio_sysmix.c at f4656d3b31a1348fe04ab2b989f15c6047e681ae (no licence header: the
// repository's GPL-3.0-or-later; the file says its wire format was read from Lichii/Eden
// src/core/hle/service/audio/nx2_sysmix.cpp, GPL-3.0-or-later, for the command layout only, and that it is an independent
// implementation). Changed here: handlers get the calling Cpu; the SDL stream per stream is an audio sink (oot2/audio_sink.h)
// chosen by the audio_sink setting; the ticks come from the renderer's grid clock (a test drives it); the stream's memory is
// checked for the host's use when the stream is created; the events are kernel events made per stream, a second acquire
// making a new pair; a close of the stream releases its events and sink.
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hos.h"
#include "ipc.h"
#include "oot2/audio.h"
#include "oot2/audio_sink.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "plat.h"

enum { MAX_STREAMS = 16, BLOCK_FRAMES = 240 };
enum { READ_OFF = 0x44, WRITE_OFF = 0x1004, PAYLOAD_OFF = 0x1040 };
enum { FMT_S16 = 2, FMT_F32 = 5 };   // PcmInt16, PcmFloat
enum { R_NOT_SUPPORTED = 0x40299u, R_INVALID_ARG = 0x20099u, R_BAD_ADDRESS = 0x1a099u, R_FAILED = 0x00c99u,
       R_INSUFFICIENT = 0x04099u, R_NOT_FOUND = 0x14099u, R_OUT_OF_SESSIONS = 0x3c699u };

typedef struct { uint32_t w, r; int handed; } Ev;   // the host's writable end, the guest's readable end, whether it left

typedef struct {
    int used, open, engaged, faulted;
    uint32_t id, flags;
    uint8_t *buf;
    uint64_t size;
    uint32_t format, frame_bytes, slots, consumer, data_mode;
    int32_t threshold;
    float gain;
    Ev state, data;
    AudioSink *sink;
} Stream;

static Stream g_streams[MAX_STREAMS];
static PlatMutex g_lock = PLAT_MUTEX_INIT;   // streams, their rings' bookkeeping, and the ticker
static uint32_t g_next_id = 1;
static AudioTicker *g_ticker;
static int g_live;   // streams that exist

static void ev_make(Ev *e) { e->w = rt_kernel_event_create(&e->r), e->handed = 0; }
static void ev_signal(const Ev *e) { if (e->w) rt_kernel_event_signal(e->w); }
// Hand the readable end to the guest: the first time the one made with the stream, afterwards a new pair (the guest closed
// the old readable end with its own handle, and the host's writable end of the old pair goes).
static uint32_t ev_acquire(Ev *e) {
    if (e->handed) {
        rt_kernel_event_close(e->w);
        ev_make(e);
    }
    e->handed = 1;
    return e->r;
}
static void ev_free(Ev *e) {
    if (e->w) rt_kernel_event_close(e->w);
    if (e->r && !e->handed) rt_kernel_event_close(e->r);   // never handed over: nobody else holds it
    *e = (Ev){0};
}

static uint32_t ld32(const Stream *s, uint64_t off) { return __atomic_load_n((const uint32_t *)(s->buf + off), __ATOMIC_ACQUIRE); }

// the data event, when the game may write at least threshold more frames (lock held)
static void notify(Stream *s, uint32_t filled) {
    if (s->data_mode == 2 && s->slots - 1 - filled >= (uint32_t)s->threshold) ev_signal(&s->data);
}

static int occupancy(const Stream *s, uint32_t *read, uint32_t *write, uint32_t *filled) {
    *read = ld32(s, READ_OFF);
    *write = ld32(s, WRITE_OFF);
    if (*read != s->consumer || *read >= s->slots || *write >= s->slots) return 0;
    *filled = (uint32_t)(((uint64_t)*write + s->slots - *read) % s->slots);
    return 1;
}

static void fault(Stream *s, const char *why) {
    if (!s->faulted) rt_log("audio: sysmix stream %u fault: %s", s->id, why);
    s->faulted = 1, s->engaged = 0;
    ev_signal(&s->state);
}

// one tick of one stream (lock held)
static void pump_one(Stream *s) {
    uint32_t read, write, filled;
    if (!occupancy(s, &read, &write, &filled)) return fault(s, "the ring's indexes are out of range or the read index moved");
    if (!s->engaged) return notify(s, filled);
    uint32_t taken = filled < BLOCK_FRAMES ? filled : BLOCK_FRAMES;
    if (taken) {
        float out[BLOCK_FRAMES * 2];
        for (uint32_t i = 0; i < taken; i++) {
            const uint8_t *p = s->buf + PAYLOAD_OFF + (uint64_t)((read + i) % s->slots) * s->frame_bytes;
            for (int ch = 0; ch < 2; ch++) {
                float v;
                if (s->format == FMT_S16) {
                    int16_t x;
                    memcpy(&x, p + ch * 2, 2);
                    v = (float)x / 32768.0f;
                } else {
                    memcpy(&v, p + ch * 4, 4);
                }
                v = isfinite(v) ? v * s->gain : 0.0f;
                out[i * 2 + ch] = v > 1.0f ? 1.0f : v < -1.0f ? -1.0f : v;
            }
        }
        if (!audio_sink_write(s->sink, out, taken)) return;   // the sink is full: the frames stay in the game's ring
        uint32_t next = (read + taken) % s->slots, expected = read;
        if (!__atomic_compare_exchange_n((uint32_t *)(s->buf + READ_OFF), &expected, next, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE))
            return fault(s, "the game moved the read index");
        s->consumer = next;
    }
    if (occupancy(s, &read, &write, &filled)) notify(s, filled);
}

static void tick(void *ctx) {
    (void)ctx;
    plat_mutex_lock(&g_lock);
    for (int i = 0; i < MAX_STREAMS; i++)
        if (g_streams[i].used && g_streams[i].open && !g_streams[i].faulted) pump_one(&g_streams[i]);
    plat_mutex_unlock(&g_lock);
}

static void close_locked(Stream *s) {
    s->open = s->engaged = 0;
    if (s->sink) {
        audio_sink_close(s->sink);
        s->sink = NULL;
    }
    ev_signal(&s->state);
}

// ---- the stream object
static void stream_closed(void *state) {
    Stream *s = state;
    AudioTicker *stop = NULL;
    plat_mutex_lock(&g_lock);
    close_locked(s);
    ev_free(&s->state);
    ev_free(&s->data);
    s->used = 0;
    if (--g_live == 0) {
        stop = g_ticker;
        g_ticker = NULL;
    }
    plat_mutex_unlock(&g_lock);
    if (stop) audio_ticker_stop(stop);   // outside the lock: the ticker's thread takes it
}

static void stream_handler(IpcSession *sess, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    Stream *s = sess->state;
    switch (rq->cmd) {
    case 0: {   // OpenStream
        uint32_t a[6];
        memcpy(a, ipc_arg(c, sess, rq, 0, 24), 24);
        uint32_t type = a[0], ch = a[1], rate = a[2], frames = a[3], prefill = a[4], flags = a[5];
        plat_mutex_lock(&g_lock);
        uint32_t r = 0;
        uint64_t slots = (uint64_t)frames + 1;
        uint32_t fb = ch * (type == FMT_S16 ? 2u : 4u);
        if (s->open) r = R_FAILED;
        else if ((type != FMT_S16 && type != FMT_F32) || ch != 2 || rate != 48000 || prefill != 0 || flags != 0) r = R_NOT_SUPPORTED;
        else if (frames == 0 || frames >= 0x7fffffffu) r = R_INVALID_ARG;
        else if (PAYLOAD_OFF + slots * fb > s->size) r = R_INSUFFICIENT;
        if (!r) {
            s->slots = (uint32_t)slots, s->format = type, s->frame_bytes = fb;
            uint32_t rd = ld32(s, READ_OFF), wr = ld32(s, WRITE_OFF);
            if (rd >= s->slots || wr >= s->slots) {
                r = R_BAD_ADDRESS;
            } else {
                char path[1100];
                snprintf(path, sizeof path, "%s/audio-capture-stream%u.wav", hos_game_folder(), s->id);
                s->sink = audio_sink_open((AudioSinkKind)rt_setting(RT_SET_audio_sink), 48000, path);
                if (!s->sink) {
                    rt_log("audio: sysmix stream %u: the platform stream cannot open (%s); using the null sink", s->id, audio_sink_why());
                    s->sink = audio_sink_open(AUDIO_SINK_NULL, 48000, NULL);
                }
                s->consumer = rd, s->faulted = 0, s->open = 1, s->engaged = 0, s->gain = 1.0f;
                ev_signal(&s->state);
                notify(s, (uint32_t)(((uint64_t)wr + s->slots - rd) % s->slots));
                rt_log("audio: sysmix stream %u open: %s stereo 48000, %u frames, sink %s (%s)", s->id,
                       type == FMT_S16 ? "int16" : "float", frames, audio_sink_kind_name(s->sink), audio_sink_device(s->sink));
            }
        }
        plat_mutex_unlock(&g_lock);
        rs->result = r;
        return;
    }
    case 1:   // CloseStream
        plat_mutex_lock(&g_lock);
        close_locked(s);
        plat_mutex_unlock(&g_lock);
        return;
    case 2: {   // GetStreamState
        plat_mutex_lock(&g_lock);
        uint32_t v = s->open && !s->faulted ? (s->engaged ? 2u : 1u) : 0u;
        plat_mutex_unlock(&g_lock);
        ipc_out(rs, &v, 4);
        return;
    }
    case 3: {   // GetStreamGain
        plat_mutex_lock(&g_lock);
        float g = s->gain;
        plat_mutex_unlock(&g_lock);
        ipc_out(rs, &g, 4);
        return;
    }
    case 4: {   // SetStreamGain(f32 gain, u32 pad, s64 ramp ns)
        float g;
        int64_t ramp;
        memcpy(&g, ipc_arg(c, sess, rq, 0, 4), 4);
        memcpy(&ramp, ipc_arg(c, sess, rq, 8, 8), 8);
        if (ramp != 0) {
            rs->result = R_NOT_SUPPORTED;
        } else if (!isfinite(g) || g < 0.0f || g > 128.0f) {
            rs->result = R_INVALID_ARG;
        } else {
            plat_mutex_lock(&g_lock);
            s->gain = g;
            plat_mutex_unlock(&g_lock);
        }
        return;
    }
    case 5:   // AcquireStreamStateChangeEvent
        plat_mutex_lock(&g_lock);
        rs->copy[rs->ncopy++] = ev_acquire(&s->state);
        plat_mutex_unlock(&g_lock);
        return;
    case 6: return;   // ReleaseStreamStateChangeEvent
    case 7: {   // AcquireStreamDataEvent(u32 mode, s32 threshold)
        uint32_t mode;
        int32_t th;
        memcpy(&mode, ipc_arg(c, sess, rq, 0, 4), 4);
        memcpy(&th, ipc_arg(c, sess, rq, 4, 4), 4);
        plat_mutex_lock(&g_lock);
        if (mode != 0 && mode != 2) {
            rs->result = R_NOT_SUPPORTED;
        } else if (!s->open || s->faulted || th < 0 || (uint32_t)th >= s->slots) {
            rs->result = R_INVALID_ARG;
        } else {
            uint32_t rd, wr, filled;
            if (!occupancy(s, &rd, &wr, &filled)) {
                fault(s, "the ring's indexes are out of range");
                rs->result = R_BAD_ADDRESS;
            } else {
                s->data_mode = mode, s->threshold = th;
                rs->copy[rs->ncopy++] = ev_acquire(&s->data);
                notify(s, filled);
            }
        }
        plat_mutex_unlock(&g_lock);
        return;
    }
    case 8:   // ReleaseStreamDataEvent
        plat_mutex_lock(&g_lock);
        s->data_mode = 0, s->threshold = 0;
        plat_mutex_unlock(&g_lock);
        return;
    case 9: case 10: case 11: rs->result = R_NOT_SUPPORTED; return;   // ClockStream, PrimeStream, PurgeStream
    case 12: return;                                                    // RenameStream
    default: ipc_unknown_command(c, sess, rq);
    }
}

// ---- engaging (lock held)
static Stream *find_stream(uint32_t id) {
    for (int i = 0; i < MAX_STREAMS; i++)
        if (g_streams[i].used && g_streams[i].id == id) return &g_streams[i];
    return NULL;
}

static uint32_t engage_locked(Stream *s, int engage) {
    if (!s) return R_NOT_FOUND;
    if (!s->open || s->faulted) return R_FAILED;
    if (s->engaged == engage) return 0;
    if (s->sink) audio_sink_pause(s->sink, !engage);
    s->engaged = engage;
    ev_signal(&s->state);
    return 0;
}

uint32_t sysmix_set_engaged(uint32_t id, int engage) {
    plat_mutex_lock(&g_lock);
    uint32_t r = engage_locked(find_stream(id), engage);
    plat_mutex_unlock(&g_lock);
    return r;
}

AudioSink *sysmix_stream_sink(uint32_t id) {
    plat_mutex_lock(&g_lock);
    Stream *s = find_stream(id);
    AudioSink *k = s ? s->sink : NULL;
    plat_mutex_unlock(&g_lock);
    return k;
}

// ---- the user session
static void user_session_handler(IpcSession *sess, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 1: {   // CreateStream
        uint32_t device, flags;
        uint64_t buffer, size;
        memcpy(&device, ipc_arg(c, sess, rq, 0, 4), 4);
        memcpy(&flags, ipc_arg(c, sess, rq, 4, 4), 4);
        memcpy(&buffer, ipc_arg(c, sess, rq, 16, 8), 8);
        memcpy(&size, ipc_arg(c, sess, rq, 24, 8), 8);
        if (device != 1) { rs->result = R_INVALID_ARG; return; }
        if (flags != 0x24) { rs->result = R_NOT_SUPPORTED; return; }
        if (!buffer || (buffer & 0xfff) || (size & 0xfff) || size < 0x2000 || size > UINT64_MAX - buffer) {
            rs->result = R_BAD_ADDRESS;
            return;
        }
        if (!rt_mem_commit_io(buffer, size, 1)) { rs->result = R_BAD_ADDRESS; return; }   // the host reads and writes it
        plat_mutex_lock(&g_lock);
        Stream *s = NULL;
        for (int i = 0; i < MAX_STREAMS && !s; i++)
            if (!g_streams[i].used) s = &g_streams[i];
        if (!s) {
            plat_mutex_unlock(&g_lock);
            rs->result = R_OUT_OF_SESSIONS;
            return;
        }
        *s = (Stream){.used = 1, .id = g_next_id++, .flags = flags, .buf = (uint8_t *)(uintptr_t)buffer, .size = size, .gain = 1.0f};
        ev_make(&s->state);
        ev_make(&s->data);
        uint32_t id = s->id;
        if (g_live++ == 0) {
            g_ticker = audio_ticker_start(BLOCK_FRAMES, 48000, tick, NULL);
            if (!g_ticker) rt_fatal("audio: cannot start the sound mixer's clock");
        }
        plat_mutex_unlock(&g_lock);
        rt_log("audio: sysmix stream %u created: buffer %#llx size %#llx", id, (unsigned long long)buffer, (unsigned long long)size);
        ipc_out(rs, &id, 4);
        ipc_answer_session(rs, ipc_session_new("ISysmixStream", stream_handler, s, stream_closed));
        return;
    }
    case 2: case 3: {   // EngageStreams, DisengageStreams: a buffer of u32 ids
        IpcBuffer b = rq->na && rq->a[0].size ? rq->a[0] : rq->nx ? rq->x[0] : (IpcBuffer){0};
        if (!b.size || b.size % 4 || b.size > 0x1000) { rs->result = R_INVALID_ARG; return; }
        uint32_t ids[0x400];
        if (!ipc_read(&b, 0, ids, b.size)) { rs->result = R_BAD_ADDRESS; return; }
        int engage = rq->cmd == 2;
        plat_mutex_lock(&g_lock);
        for (uint64_t i = 0; i < b.size / 4 && !rs->result; i++) rs->result = engage_locked(find_stream(ids[i]), engage);
        plat_mutex_unlock(&g_lock);
        return;
    }
    default: ipc_unknown_command(c, sess, rq);
    }
}

static void service_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    if (rq->cmd == 0) {   // OpenUserSession(the process handle)
        ipc_answer_session(rs, ipc_session_new("ISysmixUserSession", user_session_handler, NULL, NULL));
        return;
    }
    ipc_unknown_command(c, s, rq);
}

void srv_audsmx_register(void) { ipc_service_add("audsmx:u", service_handler); }
