// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The audio sinks (oot2/audio_sink.h): null, capture, and the platform's stream behind the same calls.
//
// The null sink models a device that takes frames at the output rate on the renderer's clock (the driven clock in a test):
// it starts playing when a short prefill has queued, drains continuously, and counts an underrun when the queue runs
// empty while it plays. The queue is bounded; a write that does not fit is refused whole and counted. The capture sink
// takes everything at once, keeps the first AUDIO_CAPTURE_KEEP_FRAMES frames and writes a 16-bit stereo WAV file as it
// goes (the sizes in the header are rewritten about once a second of audio, so a run that is killed leaves a file that
// plays).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "frameclock.h"
#include "oot2/audio_sink.h"
#include "oot2/runtime.h"
#include "plat.h"

#define PREFILL_MS 20    // queued before the null device starts to play, as after an underrun
#define CAPACITY_MS 100  // the null sink's queue: a write past this is an overrun
#define WAV_PATCH_FRAMES 48000u

struct AudioSink {
    AudioSinkKind kind;
    int rate;
    PlatMutex lock;
    char device[512];
    uint64_t written, played, underruns, overruns;
    // null
    uint64_t queued, credit, last_ns;   // credit: frames x 1e9 owed to the drain
    int playing, paused;
    // capture
    FILE *file;
    uint64_t data_bytes, unpatched;
    float *keep;
    size_t keep_frames;
    // platform stream
    const AudioSinkDriver *drv;
    void *dctx;
};

static char g_why[512];
const char *audio_sink_why(void) { return g_why; }

static const char *const KIND_NAMES[] = {"auto", "null", "capture", "sdl3"};

// ---- WAV
static void put32(uint8_t *p, uint32_t v) { for (int i = 0; i < 4; i++) p[i] = (uint8_t)(v >> (8 * i)); }
static void put16(uint8_t *p, uint32_t v) { p[0] = (uint8_t)v, p[1] = (uint8_t)(v >> 8); }

static void wav_header(AudioSink *s) {
    uint8_t h[44] = "RIFF\0\0\0\0WAVEfmt \x10\0\0\0";
    put32(h + 4, (uint32_t)(36 + s->data_bytes));
    put16(h + 20, 1);                          // PCM
    put16(h + 22, 2);                          // channels
    put32(h + 24, (uint32_t)s->rate);
    put32(h + 28, (uint32_t)s->rate * 4);      // bytes a second
    put16(h + 32, 4);                          // bytes a frame
    put16(h + 34, 16);
    memcpy(h + 36, "data", 4);
    put32(h + 40, (uint32_t)s->data_bytes);
    fwrite(h, 1, sizeof h, s->file);
}

static void wav_patch(AudioSink *s) {
    if (!s->file) return;
    fflush(s->file);
    if (fseek(s->file, 0, SEEK_SET) == 0) {
        wav_header(s);
        fflush(s->file);
        fseek(s->file, 0, SEEK_END);
    }
    s->unpatched = 0;
}

static void wav_write(AudioSink *s, const float *lr, uint32_t frames) {
    uint8_t buf[4096];
    uint32_t done = 0;
    while (done < frames) {
        uint32_t n = frames - done < 1024 ? frames - done : 1024;
        for (uint32_t i = 0; i < 2 * n; i++) {
            float x = lr[2 * done + i];
            if (!(x == x)) x = 0;   // a NaN is silence
            x = x > 1 ? 1 : x < -1 ? -1 : x;
            int v = (int)lrintf(x * 32767.0f);
            put16(buf + 2 * i, (uint32_t)(uint16_t)(int16_t)v);
        }
        fwrite(buf, 4, n, s->file);
        done += n;
    }
    s->data_bytes += 4ull * frames;
    s->unpatched += frames;
    if (s->unpatched >= WAV_PATCH_FRAMES) wav_patch(s);
}

// ---- the null device
static void null_drain(AudioSink *s, uint64_t now) {
    if (!s->playing || s->paused) return;
    uint64_t dt = now > s->last_ns ? now - s->last_ns : 0;
    s->last_ns = now;
    s->credit += dt * (uint64_t)s->rate;
    uint64_t frames = s->credit / 1000000000ull;
    s->credit %= 1000000000ull;
    if (frames >= s->queued) {
        s->played += s->queued;
        s->queued = 0;
        s->underruns++;
        s->playing = 0;
        s->credit = 0;
    } else {
        s->queued -= frames;
        s->played += frames;
    }
}

// ---- open and close
AudioSink *audio_sink_open(AudioSinkKind kind, int rate, const char *capture_path) {
    g_why[0] = 0;
    AudioSink *s = calloc(1, sizeof *s);
    if (!s) rt_fatal("audio: sink allocation");
    s->lock = (PlatMutex)PLAT_MUTEX_INIT;
    s->rate = rate;
    if (kind == AUDIO_SINK_AUTO || kind == AUDIO_SINK_SDL3) {
        const AudioSinkDriver *d = NULL;
#ifdef OOT2_AUDIO_SDL3   // not OOT2_WITH_SDL3: a build with the kit but without the driver file must still link
        d = audio_sdl3_driver();
#endif
        if (!d) {
            snprintf(g_why, sizeof g_why, "this build has no SDL3 audio (tools/fetch_sdl3.py, SDL3_DIR)");
        } else {
            s->dctx = d->open(rate, s->device, sizeof s->device);
            if (s->dctx) {
                s->drv = d;
                s->kind = AUDIO_SINK_SDL3;
                return s;
            }
            snprintf(g_why, sizeof g_why, "%s", s->device);
        }
        if (kind == AUDIO_SINK_SDL3) {
            free(s);
            return NULL;
        }
        snprintf(s->device, sizeof s->device, "no device: %s", g_why);   // auto: the null sink, and why
        kind = AUDIO_SINK_NULL;
    }
    s->kind = kind;
    if (kind == AUDIO_SINK_NULL) {
        if (!s->device[0]) snprintf(s->device, sizeof s->device, "no device, a queue drained on the renderer's clock");
    } else {
        snprintf(s->device, sizeof s->device, "kept in memory");
        s->keep = malloc((size_t)AUDIO_CAPTURE_KEEP_FRAMES * 2 * sizeof(float));
        if (!s->keep) rt_fatal("audio: capture buffer allocation");
        if (capture_path && *capture_path) {
            s->file = fopen(capture_path, "wb");
            if (!s->file) {
                rt_log("audio: sink: cannot create %s, the capture stays in memory", capture_path);
            } else {
                wav_header(s);
                snprintf(s->device, sizeof s->device, "%s", capture_path);
            }
        }
    }
    return s;
}

void audio_sink_close(AudioSink *s) {
    if (!s) return;
    if (s->drv) s->drv->close(s->dctx);
    if (s->file) {
        wav_patch(s);
        fclose(s->file);
    }
    free(s->keep);
    free(s);
}

// ---- the calls
uint32_t audio_sink_write(AudioSink *s, const float *lr, uint32_t frames) {
    if (!frames) return 0;
    uint32_t accepted = 0;
    if (s->drv) {
        accepted = s->drv->write(s->dctx, lr, frames);
        plat_mutex_lock(&s->lock);
        s->written += accepted;
        plat_mutex_unlock(&s->lock);
        return accepted;
    }
    uint64_t now = frameclock_now_ns();
    plat_mutex_lock(&s->lock);
    if (s->kind == AUDIO_SINK_CAPTURE) {
        if (s->keep_frames < AUDIO_CAPTURE_KEEP_FRAMES) {
            size_t n = AUDIO_CAPTURE_KEEP_FRAMES - s->keep_frames;
            if (n > frames) n = frames;
            memcpy(s->keep + 2 * s->keep_frames, lr, n * 2 * sizeof(float));
            s->keep_frames += n;
        }
        if (s->file) wav_write(s, lr, frames);
        s->written += frames;
        s->played += frames;
        accepted = frames;
    } else {
        null_drain(s, now);
        uint64_t cap = (uint64_t)s->rate * CAPACITY_MS / 1000, prefill = (uint64_t)s->rate * PREFILL_MS / 1000;
        if (s->paused || s->queued + frames > cap) {
            if (!s->paused) s->overruns++;
        } else {
            s->queued += frames;
            s->written += frames;
            accepted = frames;
            if (!s->playing && s->queued >= prefill) {
                s->playing = 1;
                s->last_ns = now;
                s->credit = 0;
            }
        }
    }
    plat_mutex_unlock(&s->lock);
    return accepted;
}

void audio_sink_stats(const AudioSink *cs, AudioSinkStats *out) {
    AudioSink *s = (AudioSink *)cs;
    memset(out, 0, sizeof *out);
    if (s->drv) {
        s->drv->stats(s->dctx, out);
        plat_mutex_lock(&s->lock);
        out->written = s->written;
        plat_mutex_unlock(&s->lock);
        return;
    }
    uint64_t now = frameclock_now_ns();
    plat_mutex_lock(&s->lock);
    if (s->kind == AUDIO_SINK_NULL) null_drain(s, now);
    out->written = s->written;
    out->played = s->played;
    out->queued = s->queued;
    out->underruns = s->underruns;
    out->overruns = s->overruns;
    plat_mutex_unlock(&s->lock);
}

uint32_t audio_sink_queued(const AudioSink *s) {
    AudioSinkStats st;
    audio_sink_stats(s, &st);
    return (uint32_t)st.queued;
}

void audio_sink_pause(AudioSink *s, int paused) {
    if (s->drv) {
        s->drv->pause(s->dctx, paused);
        return;
    }
    plat_mutex_lock(&s->lock);
    s->paused = paused;
    if (paused && s->kind == AUDIO_SINK_NULL) {
        s->queued = 0;
        s->playing = 0;
        s->credit = 0;
    }
    plat_mutex_unlock(&s->lock);
}

AudioSinkKind audio_sink_kind(const AudioSink *s) { return s->kind; }
const char *audio_sink_kind_name(const AudioSink *s) { return KIND_NAMES[s->kind]; }
const char *audio_sink_device(const AudioSink *s) { return s->device; }

const float *audio_sink_capture_data(const AudioSink *s, size_t *frames) {
    if (s->kind != AUDIO_SINK_CAPTURE) {
        *frames = 0;
        return NULL;
    }
    *frames = s->keep_frames;
    return s->keep;
}
