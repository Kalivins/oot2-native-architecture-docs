// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The audio sink's platform stream (oot2/audio_sink.h, AudioSinkDriver): one SDL3 audio device stream for the default playback
// device, fed stereo float frames at the renderer's output rate; SDL converts to what the device wants. Compiled with the SDL3
// kit (OOT2_WITH_SDL3) and, for the sink to call it, OOT2_AUDIO_SDL3 (tools/build_game.py defines both with a kit).
//
// The queue is held short on purpose: at most QUEUE_CAP_MS of frames wait in the stream, so the sound is late by about that and
// the device's own buffer (a hint of DEVICE_BUFFER_MS asks for a small one). The renderer writes a frame of 5 ms on its own
// clock and the device drains on its own: a write that would take the queue past the cap is refused whole and counted as an
// overrun (the renderer drops that frame's samples, which is the right way to lose the difference between two clocks); a
// write that finds the queue empty while the device runs counts an underrun, pauses the device and waits for PREFILL_MS of
// frames before it runs again, as the null sink does. Pausing the sink (the renderer stopped) holds no queue and counts nothing.
//
// The calls the sink makes come from the renderer's thread (its lock held) except stats, which any thread may ask: counters
// are atomics. SDL_Quit of the window thread (runtime/platform/window.c) ends every subsystem when the window closes: the sink
// goes silent from then on (the clock may still write to a stream that no longer plays), and a stream still open at the
// process's end is lost with it. The device-buffer hint is process-wide and is not put back.
//
// What the counters can and cannot say. An underrun is "the stream's queue is empty at a write", which leaves out what SDL has
// already moved into the device buffer (about DEVICE_BUFFER_MS): a queue that SDL's pull emptied while the device still holds
// audio counts as an underrun and costs a PREFILL_MS pause that is not a true starvation, so the count is an upper bound on
// audible gaps. `played` is frames written minus frames still queued, so it runs ahead of the device by up to its buffer and
// is not evidence that the device took the samples. A write SDL refuses counts as an overrun, like one past the cap.
#include "oot2/audio_sink.h"

#ifdef OOT2_AUDIO_SDL3
#ifdef OOT2_WITH_SDL3
#include <SDL3/SDL.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define QUEUE_CAP_MS 30       // frames the stream may hold: a write past it is refused
#define PREFILL_MS 20         // queued before the device runs, at the start and after an underrun
#define DEVICE_BUFFER_MS 10   // the device buffer the hint asks for
#define FRAME_BYTES 8         // stereo float

typedef struct SdlSink {
    SDL_AudioStream *stream;
    int rate;
    int playing;                  // the device runs (renderer thread only)
    uint64_t written;             // frames accepted
    uint64_t underruns, overruns; // atomics
} SdlSink;

static uint64_t queued_frames(SdlSink *s) {
    int bytes = SDL_GetAudioStreamQueued(s->stream);
    return bytes > 0 ? (uint64_t)bytes / FRAME_BYTES : 0;
}

static void *sdl_open(int rate, char *device, size_t device_size) {
    if (rate < 8000 || rate > 192000) {
        snprintf(device, device_size, "an output rate of %d Hz", rate);
        return NULL;
    }
    char frames[16];
    snprintf(frames, sizeof frames, "%d", rate * DEVICE_BUFFER_MS / 1000);
    SDL_SetHint(SDL_HINT_AUDIO_DEVICE_SAMPLE_FRAMES, frames);
    if (!SDL_InitSubSystem(SDL_INIT_AUDIO)) {
        snprintf(device, device_size, "SDL_InitSubSystem(audio): %s", SDL_GetError());
        return NULL;
    }
    SdlSink *s = calloc(1, sizeof *s);
    if (!s) {
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        snprintf(device, device_size, "out of memory");
        return NULL;
    }
    s->rate = rate;
    const SDL_AudioSpec spec = {SDL_AUDIO_F32, 2, rate};
    s->stream = SDL_OpenAudioDeviceStream(SDL_AUDIO_DEVICE_DEFAULT_PLAYBACK, &spec, NULL, NULL);
    if (!s->stream) {
        snprintf(device, device_size, "SDL_OpenAudioDeviceStream: %s", SDL_GetError());
        free(s);
        SDL_QuitSubSystem(SDL_INIT_AUDIO);
        return NULL;
    }
    const char *name = SDL_GetAudioDeviceName(SDL_GetAudioStreamDevice(s->stream));
    const char *driver = SDL_GetCurrentAudioDriver();
    snprintf(device, device_size, "%s (SDL audio driver %s), queue held to %d ms", name ? name : "the default playback device",
             driver ? driver : "unknown", QUEUE_CAP_MS);
    return s;   // the device is paused until a prefill has queued
}

static uint32_t sdl_write(void *ctx, const float *lr, uint32_t frames) {
    SdlSink *s = ctx;
    uint64_t cap = (uint64_t)s->rate * QUEUE_CAP_MS / 1000, prefill = (uint64_t)s->rate * PREFILL_MS / 1000;
    uint64_t queued = queued_frames(s);
    if (s->playing && queued == 0) {   // the device took everything and nothing came: re-prefill
        __atomic_fetch_add(&s->underruns, 1, __ATOMIC_RELAXED);
        SDL_PauseAudioStreamDevice(s->stream);
        s->playing = 0;
    }
    if (queued + frames > cap) {
        __atomic_fetch_add(&s->overruns, 1, __ATOMIC_RELAXED);
        return 0;
    }
    if (!SDL_PutAudioStreamData(s->stream, lr, (int)(frames * FRAME_BYTES))) {
        __atomic_fetch_add(&s->overruns, 1, __ATOMIC_RELAXED);
        return 0;
    }
    __atomic_fetch_add(&s->written, frames, __ATOMIC_RELAXED);
    if (!s->playing && queued + frames >= prefill) {
        SDL_ResumeAudioStreamDevice(s->stream);
        s->playing = 1;
    }
    return frames;
}

static void sdl_stats(void *ctx, AudioSinkStats *out) {
    SdlSink *s = ctx;
    uint64_t written = __atomic_load_n(&s->written, __ATOMIC_RELAXED), queued = queued_frames(s);
    if (queued > written) queued = written;   // SDL may count what it converted for the device
    out->written = written;
    out->queued = queued;
    out->played = written - queued;
    out->underruns = __atomic_load_n(&s->underruns, __ATOMIC_RELAXED);
    out->overruns = __atomic_load_n(&s->overruns, __ATOMIC_RELAXED);
}

static void sdl_pause(void *ctx, int paused) {
    SdlSink *s = ctx;
    if (!paused) return;   // a resume starts as at the first write: the prefill decides
    SDL_PauseAudioStreamDevice(s->stream);
    SDL_ClearAudioStream(s->stream);
    s->playing = 0;
}

static void sdl_close(void *ctx) {
    SdlSink *s = ctx;
    SDL_DestroyAudioStream(s->stream);   // closes the device it opened
    free(s);
    SDL_QuitSubSystem(SDL_INIT_AUDIO);
}

static const AudioSinkDriver g_driver = {sdl_open, sdl_write, sdl_stats, sdl_pause, sdl_close};

const AudioSinkDriver *audio_sdl3_driver(void) { return &g_driver; }

#else   // the sink's flag without the kit's: no driver, and the sink says so

const AudioSinkDriver *audio_sdl3_driver(void) { return NULL; }

#endif
#endif
