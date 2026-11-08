// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Where the mixed audio goes: a sink takes stereo float frames, interleaved, at the renderer's output rate (48000 or 32000
// Hz). The renderer never waits for a sink: a write that does not fit is refused whole and counted as an overrun, and the
// renderer drops that frame. Four kinds:
//   null     drains its queue on the renderer's clock, as a device would, so a run with no device keeps its cadence
//   capture  accepts everything at once; keeps the first AUDIO_CAPTURE_KEEP_FRAMES frames in memory for a test, and writes
//            a WAV file when it is given a path (for listening: the path is under the game folder or a work folder,
//            never a repository)
//   sdl3     the platform's audio stream (runtime/platform/audio_sdl3.c), in a build that compiles it
//   auto     sdl3 when this build has it and a device opens, else null with the reason logged; never a stop
#pragma once
#include <stddef.h>
#include <stdint.h>

// The values are the indices of the audio_sink setting's names, "auto|null|capture|sdl3" (settings.h).
typedef enum AudioSinkKind { AUDIO_SINK_AUTO, AUDIO_SINK_NULL, AUDIO_SINK_CAPTURE, AUDIO_SINK_SDL3 } AudioSinkKind;

typedef struct AudioSink AudioSink;

typedef struct AudioSinkStats {
    uint64_t written;      // frames accepted
    uint64_t played;       // frames the device has taken (written less queued)
    uint64_t queued;       // frames waiting
    uint64_t underruns;    // times the queue ran empty while the sink was playing
    uint64_t overruns;     // writes refused for want of room
} AudioSinkStats;

// How many frames a capture sink keeps in memory (30 s at 48000 Hz); the WAV file takes all.
#define AUDIO_CAPTURE_KEEP_FRAMES (48000u * 30u)

// Open a sink for rate Hz. capture_path is used by the capture kind only (NULL: memory only). NULL when kind is sdl3 and
// the platform's stream cannot be opened, with the reason in audio_sink_why(); the other kinds always open (a capture
// file that cannot be created is logged and the sink keeps its memory).
AudioSink *audio_sink_open(AudioSinkKind kind, int rate, const char *capture_path);
const char *audio_sink_why(void);   // the reason of the last open that returned NULL, or why auto fell back
void audio_sink_close(AudioSink *s);

// frames stereo frames (2 floats each) in lr: all of them accepted (the return value is frames), or none (0, an overrun).
uint32_t audio_sink_write(AudioSink *s, const float *lr, uint32_t frames);
uint32_t audio_sink_queued(const AudioSink *s);
void audio_sink_stats(const AudioSink *s, AudioSinkStats *out);
// A paused sink holds no queue and counts no underrun: the renderer is stopped. Resuming starts as at the first write.
void audio_sink_pause(AudioSink *s, int paused);
AudioSinkKind audio_sink_kind(const AudioSink *s);   // the kind that opened: never auto
const char *audio_sink_kind_name(const AudioSink *s);
const char *audio_sink_device(const AudioSink *s);   // the device's name, or the file, or why the sink is not a device
// The frames a capture sink kept (up to AUDIO_CAPTURE_KEEP_FRAMES), interleaved; NULL for the other kinds.
const float *audio_sink_capture_data(const AudioSink *s, size_t *frames);

// The platform's stream (runtime/platform/audio_sdl3.c), behind audio_sink_open: a table of the calls the sink makes. open
// returns NULL on failure with the reason in device. A build that compiles that file defines OOT2_AUDIO_SDL3 for the sink and
// for it; a build that merely has the SDL3 kit (OOT2_WITH_SDL3) does not, so no call to a driver that is not there is made.
typedef struct AudioSinkDriver {
    void *(*open)(int rate, char *device, size_t device_size);   // NULL on failure, with the reason in device
    uint32_t (*write)(void *ctx, const float *lr, uint32_t frames);
    void (*stats)(void *ctx, AudioSinkStats *out);
    void (*pause)(void *ctx, int paused);
    void (*close)(void *ctx);
} AudioSinkDriver;
#ifdef OOT2_AUDIO_SDL3
const AudioSinkDriver *audio_sdl3_driver(void);
#endif
