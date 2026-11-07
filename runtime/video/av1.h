// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The AV1 decoder seam (architecture D24b, ticket 98). The video decoder service of runtime/hos (ov_codec.c) hands a
// compressed temporal unit to a decoder and takes a picture back, all on host buffers: nothing here touches guest
// memory or the SDK. The real decoder is dav1d (av1_dav1d.c); a test links a fake of its own, so the service's events,
// frame ownership and Stop/Flush rules are tested with no decoder, and the decoder with no service.
//
// One Av1Decoder is used from one thread at a time (the service's worker for that decoder).
#pragma once
#include <stddef.h>
#include <stdint.h>

typedef struct Av1Decoder Av1Decoder;

// A decoded picture: 8-bit 4:2:0, three planes (luma, then the two chroma planes), each with its own row stride in bytes,
// the stride of a row of the decoder's own buffer and not the picture's width. The planes belong to the decoder: they
// stay valid until the next av1_send, av1_receive, av1_flush or av1_close on that decoder.
typedef struct Av1Picture {
    int w, h;                    // luma size; the chroma planes are (w + 1) / 2 by (h + 1) / 2
    const uint8_t *plane[3];     // Y, U, V
    ptrdiff_t stride[3];
    int64_t ts;                  // the time stamp given to av1_send for the temporal unit this picture shows
} Av1Picture;

enum { AV1_OK = 0, AV1_NONE = 0, AV1_PICTURE = 1, AV1_BAD = -1 };

// A new decoder, or NULL when this build has none (no dav1d kit) or the decoder cannot start.
Av1Decoder *av1_open(void);
// What the decoder is, for the log ("dav1d 1.5.4", "none").
const char *av1_describe(void);
// One temporal unit (the bytes of a container's AV1 block: OBUs, the sequence header included when the stream has one
// there). Returns AV1_OK, or AV1_BAD when the bytes are not a decodable unit (damaged, a layout or bit depth this
// seam does not carry): the decoder stays usable and the next unit may be fine. Pictures it makes are held for
// av1_receive; there may be none, one or more per unit.
int av1_send(Av1Decoder *d, const void *data, size_t size, int64_t ts);
// The next picture in decode order: AV1_PICTURE with *out filled, AV1_NONE when there is none yet.
int av1_receive(Av1Decoder *d, Av1Picture *out);
// Forget everything not yet received and everything the stream carried over (references): the next unit starts a
// new stream and must be a key frame with its sequence header.
void av1_flush(Av1Decoder *d);
void av1_close(Av1Decoder *d);
