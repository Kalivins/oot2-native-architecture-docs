// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The mixing stage of the audio renderer: each frame it decodes every started voice (PCM8, 16, 24, 32, float and
// DSP-ADPCM), resamples it linearly by its rate and pitch, runs the voice's two biquads, scales it into its destination
// mix (or a splitter's destinations), folds the sub mixes in dependency order, passes the effects through dry, and maps
// the device sink's channels to stereo float frames. It runs under the renderer's lock, on the frame clock's thread.
#pragma once
#include <stdint.h>

#include "audren_state.h"

Mixer *mixer_open(AudioRenderer *r);
void mixer_close(Mixer *m);
// The splitter section, read after audren.c has walked it clean (the same bytes, the same revision).
void mixer_set_splitters(Mixer *m, const uint8_t *p, uint32_t n, int rev);
// A voice starts over (new, or stopped): its filter and decoder history is cleared.
void mixer_voice_reset(Mixer *m, unsigned id);
// One frame: sample_count stereo float frames into lr (interleaved). Returns 1 when a device sink took part, 0 when the
// client has none (lr is silence).
int mixer_render(AudioRenderer *r, float *lr);
// What the mixer has done since the last call with reset: voice frames decoded, the largest sample of a decoded voice (before any
// gain) and of the output. A run that is silent shows here whether the voices carry samples at all.
typedef struct MixerStats {
    uint64_t voice_frames;
    float input_peak, output_peak;
} MixerStats;
MixerStats mixer_stats(Mixer *m, int reset);
