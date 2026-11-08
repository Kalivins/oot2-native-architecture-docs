// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The Opus decoder behind the seam of opus.h: libopus (BSD-3-Clause, built from Xiph's pinned release by tools/fetch_opus.py
// into a kit, never kept in this repository). Compiled with -DOOT2_OPUS=1 and the kit's headers when tools/build_game.py
// found a kit; without it this file is empty and audio_opus() has no table of its own.
//
// What the hardware decoder's override (runtime/hos/ov_opus.c) asks of it: one decoder object for each guest decoder, made at
// the guest's rate and channel count, decoding 16-bit interleaved samples, one packet for each call, with the final range of
// the range coder reported back (the stream's header carries it, the override compares). A packet of no bytes is no packet:
// libopus would conceal a whole output buffer of lost audio for it, so it makes no samples here.
#include "opus.h"

#ifdef OOT2_OPUS
#include <stdio.h>

#include "../core/plat.h"

#include <opus/opus.h>

static void *lo_create(int rate, int channels, char *why, size_t why_size) {
    int err = OPUS_OK;
    OpusDecoder *d = opus_decoder_create(rate, channels, &err);
    if (!d && why && why_size) snprintf(why, why_size, "opus_decoder_create(%d Hz, %d channels): %s", rate, channels, opus_strerror(err));
    return d;
}

static void lo_destroy(void *dec) { opus_decoder_destroy(dec); }

static int lo_reset(void *dec) { return opus_decoder_ctl((OpusDecoder *)dec, OPUS_RESET_STATE) == OPUS_OK ? 0 : -1; }

static int lo_decode(void *dec, const uint8_t *packet, int len, int16_t *pcm, int max_frames, uint32_t *final_range) {
    if (len <= 0) {
        *final_range = 0;
        return 0;
    }
    int frames = opus_decode((OpusDecoder *)dec, packet, len, pcm, max_frames, 0);
    if (frames < 0) return frames;
    opus_uint32 range = 0;
    opus_decoder_ctl((OpusDecoder *)dec, OPUS_GET_FINAL_RANGE(&range));
    *final_range = range;
    return frames;
}

// The table's name carries the library's own version string ("libopus 1.5.2"), known only at run time: the first caller writes
// it, and a caller that meets the first one in the middle waits for it, so no one has the table before its name is there.
static OpusApi g_api = {"libopus", lo_create, lo_destroy, lo_reset, lo_decode};

const OpusApi *audio_opus_libopus(void) {
    static char text[48];
    static int32_t named;   // 0 not yet, 1 being written, 2 written
    int32_t none = 0;
    if (__atomic_compare_exchange_n(&named, &none, 1, 0, __ATOMIC_ACQ_REL, __ATOMIC_ACQUIRE)) {
        snprintf(text, sizeof text, "%s", opus_get_version_string());
        g_api.name = text;
        __atomic_store_n(&named, 2, __ATOMIC_RELEASE);
    } else {
        while (__atomic_load_n(&named, __ATOMIC_ACQUIRE) != 2) plat_pause();
    }
    return &g_api;
}
#endif
