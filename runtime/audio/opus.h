// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The seam under the hardware Opus decoder (runtime/hos/ov_opus.c): what the override needs of an Opus decoder, as a table
// of calls, so that the override's own rules (the frame header, what a decode consumes, the refusals, the state of each
// decoder) are tested with a fake and the decoder itself is libopus in a build that has it (runtime/audio/opus_libopus.c,
// a pinned release built into a kit by tools/fetch_opus.py). A build without libopus has no table: the override then
// consumes its input and makes no sample, and says so once.
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct OpusApi {
    const char *name;   // what decodes, for the log ("libopus 1.5.2")
    // A decoder for rate Hz and channels (1 or 2); NULL on failure, with the reason in why.
    void *(*create)(int rate, int channels, char *why, size_t why_size);
    void (*destroy)(void *dec);
    int (*reset)(void *dec);   // 0 on success
    // Decode one packet of len bytes into pcm, room for max_frames frames of the decoder's channels, interleaved. Returns the
    // frames per channel made, or a negative number when the packet is refused; *final_range is the decoder's final range
    // after the packet.
    int (*decode)(void *dec, const uint8_t *packet, int len, int16_t *pcm, int max_frames, uint32_t *final_range);
} OpusApi;

// The decoder this build has: the one installed, else libopus's when the build has it, else NULL.
const OpusApi *audio_opus(void);
// Install a table (a test's fake); NULL goes back to the build's own.
void audio_opus_install(const OpusApi *api);
#ifdef OOT2_OPUS
const OpusApi *audio_opus_libopus(void);
#endif
