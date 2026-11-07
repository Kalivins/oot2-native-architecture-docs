// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// A fake AV1 decoder for the tests of the video decoder service (runtime/hos/ov_codec.c), written against the seam of
// runtime/video/av1.h. It decodes nothing: a unit whose first byte is T makes one picture of g_fake_w x g_fake_h whose
// planes are generated from T, so a test can tell which unit a picture shows and whether its planes were swapped or
// shifted; a unit that starts with 0xFF is damaged. The knobs let a test make the decode slow (the service must still
// return at once) or refuse to open. Included once by each test program that links runtime/hos.
#pragma once
#include <stdlib.h>
#include <string.h>

#include "plat.h"
#include "../../runtime/video/av1.h"

int g_fake_w = 6, g_fake_h = 4;           // picture size (odd sizes are allowed: the chroma planes round up)
uint64_t g_fake_delay_ns;                 // each av1_send takes this long
int g_fake_refuse_open;                   // av1_open fails
volatile int g_fake_sends, g_fake_flushes, g_fake_bad, g_fake_closes;
volatile int g_fake_inflight, g_fake_close_busy;   // decodes running now; closes that came while one ran
volatile int g_fake_threads_differ;       // av1_send ran on a thread other than the one that opened the decoder

#define FAKE_YSTRIDE_PAD 16
#define FAKE_UVSTRIDE_PAD 8
#define FAKE_Y(t, x, y) ((uint8_t)((t) + (x) + 3 * (y)))
#define FAKE_U(t, x, y) ((uint8_t)((t) + 7 + (x) + (y)))
#define FAKE_V(t, x, y) ((uint8_t)((t) + 13 + 2 * (x) + (y)))

struct Av1Decoder {
    uint8_t *y, *u, *v;
    int w, h, have;
    int64_t ts;
    int opened_on;
};

static int fake_thread_id(void) {
    static __thread int id;
    return id ? id : (id = (int)(plat_now_ns() & 0x7fffffff) | 1);
}

Av1Decoder *av1_open(void) {
    if (g_fake_refuse_open) return NULL;
    Av1Decoder *d = calloc(1, sizeof *d);
    d->opened_on = fake_thread_id();
    return d;
}
const char *av1_describe(void) { return "fake"; }
static int fake_send(Av1Decoder *d, const void *data, size_t size, int64_t ts) {
    g_fake_sends++;
    if (fake_thread_id() == d->opened_on) g_fake_threads_differ = 0;
    else g_fake_threads_differ = 1;
    if (g_fake_delay_ns) plat_sleep_ns(g_fake_delay_ns);
    const uint8_t *b = data;
    if (!size || b[0] == 0xFF) {
        g_fake_bad++;
        return AV1_BAD;
    }
    int t = b[0], w = g_fake_w, h = g_fake_h, cw = (w + 1) / 2, ch = (h + 1) / 2;
    free(d->y);
    free(d->u);
    free(d->v);
    d->y = calloc(1, (size_t)(w + FAKE_YSTRIDE_PAD) * h);
    d->u = calloc(1, (size_t)(cw + FAKE_UVSTRIDE_PAD) * ch);
    d->v = calloc(1, (size_t)(cw + FAKE_UVSTRIDE_PAD) * ch);
    for (int y = 0; y < h; y++)
        for (int x = 0; x < w; x++) d->y[y * (w + FAKE_YSTRIDE_PAD) + x] = FAKE_Y(t, x, y);
    for (int y = 0; y < ch; y++)
        for (int x = 0; x < cw; x++) {
            d->u[y * (cw + FAKE_UVSTRIDE_PAD) + x] = FAKE_U(t, x, y);
            d->v[y * (cw + FAKE_UVSTRIDE_PAD) + x] = FAKE_V(t, x, y);
        }
    d->w = w;
    d->h = h;
    d->ts = ts;
    d->have = 1;
    return AV1_OK;
}
int av1_send(Av1Decoder *d, const void *data, size_t size, int64_t ts) {
    g_fake_inflight++;
    int r = fake_send(d, data, size, ts);
    g_fake_inflight--;
    return r;
}
int av1_receive(Av1Decoder *d, Av1Picture *out) {
    if (!d->have) return AV1_NONE;
    d->have = 0;
    out->w = d->w;
    out->h = d->h;
    out->plane[0] = d->y;
    out->plane[1] = d->u;
    out->plane[2] = d->v;
    out->stride[0] = d->w + FAKE_YSTRIDE_PAD;
    out->stride[1] = out->stride[2] = (d->w + 1) / 2 + FAKE_UVSTRIDE_PAD;
    out->ts = d->ts;
    return AV1_PICTURE;
}
void av1_flush(Av1Decoder *d) {
    g_fake_flushes++;
    d->have = 0;
}
void av1_close(Av1Decoder *d) {
    g_fake_closes++;
    if (g_fake_inflight) g_fake_close_busy++;
    free(d->y);
    free(d->u);
    free(d->v);
    free(d);
}
