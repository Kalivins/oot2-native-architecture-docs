// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// No video decoder yet (ticket 38). After its startup logo the game builds its movie player, whose first step is
// movie::VideoDecoder::Initialize (subsdk0, over the SDK's nn::codec::video and the console's video decoder service).
// The run stops there by name, so the next hole is the movie player and not a null pointer inside it: the player also
// needs nn::movie::MediaExtractor (a container reader) and the decoders' events and frames, and the AV1 decoder (research
// 07's gap) is the movie ticket's work.
#include "hos.h"

// Result movie::VideoDecoder::Initialize(void *work, size_t size, nn::movie::MediaMimeType, VideoDecoderOutputFormat,
// VideoCodecMode)
static void ov_video_decoder_initialize(Cpu *c) {
    hos_gap(c, "movie", "movie::VideoDecoder::Initialize: the game's movie player starts, and the host has no video "
            "decoder and no media extractor yet");
}
HOS_OVERRIDE("_ZN5movie12VideoDecoder10InitializeEPvmN2nn5movie13MediaMimeTypeENS_24VideoDecoderOutputFormatENS_14VideoCodecModeE",
             ov_video_decoder_initialize, HOS_GAP, "the movie player's first step: a named fatal until the movie ticket");
