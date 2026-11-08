// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The renderer's server-side state, shared by the renderer (audren.c: the update, the voice lifetime, the frame) and the
// mixer (mixer.c: what a started voice sounds like this frame). Nothing outside runtime/audio includes it.
//
// Adapted from DKB Recomp runtime/audren.h at c5d5e7c (MIT, Copyright (c) 2026 MacNativeGames contributors; see the notice
// of audren.c): the voice and renderer records, split from the wire layouts (audren_wire.h).
#pragma once
#include <stdint.h>

#include "audren_wire.h"
#include "frameclock.h"
#include "oot2/audio.h"
#include "oot2/audio_sink.h"
#include "plat.h"

typedef struct {
    AudrenVoiceIn in;
    AudrenBiquadFloat float_biquads[2];
    uint8_t use_float_biquads;
    AudrenWaveBuffer buffers[4];
    uint8_t valid[4];
    uint8_t pending[4];          // sent while the voice was not playing: valid once it plays
    uint8_t state;               // SV_*: the client's Start, Stop and Pause, plus Stopping until the next frame
    uint32_t flush, flush_from;  // FlushWaveBuffers since the last frame, from the client's current buffer
    unsigned index;
    uint64_t offset;
    uint64_t acc;                // samples owed to the voice, in units of 1 / engine_rate sample
    int32_t loops_done;
    uint64_t played;
    uint32_t consumed, drops;
    // for the stall detector (stall.c), on its clock: when each slot's buffer was submitted, by which guest thread, after how
    // many rendered frames, and when the voice last started
    uint64_t sub_ns[4], sub_frame[4], start_ns;
    uint32_t sub_thread[4];
} ServerVoice;
enum { SV_STOPPED, SV_STARTED, SV_PAUSED, SV_STOPPING };

typedef struct Mixer Mixer;

struct AudioRenderer {
    AudioConfig cfg;
    PlatMutex ctl;               // Start, Stop and Close, one at a time (Stop joins the clock thread)
    PlatMutex lock;              // everything below: an update and a frame never overlap
    FrameClock *clock;
    uint32_t event;              // writable end of the frame event
    int running;                 // written under the lock with an atomic store, so the watchdog can read it without the lock
    int revision;                // from the last update
    int logged;
    uint64_t behavior_flags, elapsed_frames, skipped_frames, signals, released, updates;
    uint64_t last_signal_ns, last_update_ns, signal_to_update_max, update_gap_max;
    int signal_pending;
    uint64_t late_max;           // the frame clock's, copied at Stop and Close
    unsigned diag_new, diag_sent;   // wave buffer lines logged: with the sent flag clear, and a real buffer with it set
    ServerVoice *voices;
    AudrenPoolIn *pools;
    uint32_t pool_gen;           // changes when a memory pool's address, size or state does
    AudrenMixIn *mixes;
    AudrenSinkIn *sinks;
    AudrenEffectIn *effects;
    AudrenChannelIn *channels;
    int mix_count;
    Mixer *mixer;
    AudioSink *sink;
    float *frame_buf;            // one frame of stereo output
    uint64_t dropped_frames;     // frames the sink refused (its queue was full)
    uint32_t readable;           // the client's end of the frame event: who waits on it is the stall detector's question
    // on the stall detector's clock, and the guest thread that made the last update: written under the lock with atomic
    // stores, read by the watchdog without it (elapsed_frames and running too), so a stuck frame cannot silence the watchdog
    uint64_t last_frame_wd, last_update_wd;
    uint32_t last_update_thread;
};

// What a started voice consumes of its own samples in this frame, in units of 1 / engine_rate sample, added to acc:
// sample_count x voice_rate x pitch, rounded. The renderer advances by it and the mixer reads the same quantity, so the
// samples a frame sounds are the samples it releases.
uint64_t voice_owed(const AudioRenderer *r, const ServerVoice *v);
// The pitch the renderer uses: the client's, when it is finite and above 0, else 1, and at most AUDIO_PITCH_MAX (a client
// that asks for more would make a frame walk for ever). The stall detector's durations read it as well.
#define AUDIO_PITCH_MAX 4.0
double voice_pitch(const ServerVoice *v);
// A wave buffer with no samples: no memory, no size, or an empty range.
int buffer_empty(const AudrenWaveBuffer *b);
