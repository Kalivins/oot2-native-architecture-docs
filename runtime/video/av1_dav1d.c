// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The AV1 decoder seam over dav1d (BSD-2-Clause, built from the pinned release by tools/fetch_dav1d.py, never kept in
// this repository). Built with -DOOT2_DAV1D=1 and the kit's headers when tools/build_game.py found a kit; otherwise
// av1_open refuses, and the movie service says so by name at the game's first movie.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "av1.h"

#if OOT2_DAV1D
#include <dav1d/dav1d.h>
#include <errno.h>

#define HELD_MAX 16   // pictures made by a unit and not yet received: more than a unit makes (a unit shows at most one picture)

struct Av1Decoder {
    Dav1dContext *ctx;
    Dav1dPicture held[HELD_MAX];
    int head, count;
    Dav1dPicture lent;   // the picture av1_receive last handed out
    int has_lent;
};

static void drop_lent(Av1Decoder *d) {
    if (d->has_lent) dav1d_picture_unref(&d->lent);
    d->has_lent = 0;
}

static void drop_held(Av1Decoder *d) {
    for (; d->count; d->count--, d->head = (d->head + 1) % HELD_MAX) dav1d_picture_unref(&d->held[d->head]);
}

const char *av1_describe(void) {
    static char text[48];
    snprintf(text, sizeof text, "dav1d %s", dav1d_version());
    return text;
}

Av1Decoder *av1_open(void) {
    Av1Decoder *d = calloc(1, sizeof *d);
    if (!d) return NULL;
    Dav1dSettings s;
    dav1d_default_settings(&s);
    // One picture for each unit sent, tiles and the post filters on four threads: the decode runs on the service's
    // worker and the picture is wanted at once, so no frame-parallel queue delays it.
    s.n_threads = 4;
    s.max_frame_delay = 1;
    if (dav1d_open(&d->ctx, &s) < 0) {
        free(d);
        return NULL;
    }
    return d;
}

// Pull every picture dav1d has ready into the held ring; 0, or a negative dav1d error other than "none yet".
static int pull(Av1Decoder *d) {
    for (;;) {
        Dav1dPicture p;
        memset(&p, 0, sizeof p);
        int r = dav1d_get_picture(d->ctx, &p);
        if (r == DAV1D_ERR(EAGAIN)) return 0;
        if (r < 0) return r;
        if (d->count == HELD_MAX) {   // cannot happen with a unit shown once; drop the oldest rather than the newest
            dav1d_picture_unref(&d->held[d->head]);
            d->head = (d->head + 1) % HELD_MAX;
            d->count--;
        }
        d->held[(d->head + d->count++) % HELD_MAX] = p;
    }
}

int av1_send(Av1Decoder *d, const void *data, size_t size, int64_t ts) {
    if (!d || !data || !size) return AV1_BAD;
    Dav1dData in;
    uint8_t *buf = dav1d_data_create(&in, size);
    if (!buf) return AV1_BAD;
    memcpy(buf, data, size);
    in.m.timestamp = ts;
    int bad = 0;
    while (in.sz) {
        int r = dav1d_send_data(d->ctx, &in);
        if (r == DAV1D_ERR(EAGAIN)) {   // dav1d wants its pictures taken before it takes more
            if (pull(d) < 0) {
                bad = 1;
                break;
            }
            continue;
        }
        if (r < 0) {
            bad = 1;
            break;
        }
    }
    if (in.sz) dav1d_data_unref(&in);
    if (pull(d) < 0) bad = 1;
    // A picture this seam does not carry is refused whole: only 8-bit 4:2:0 reaches the service.
    for (int i = 0; i < d->count; i++) {
        const Dav1dPicture *p = &d->held[(d->head + i) % HELD_MAX];
        if (p->p.layout != DAV1D_PIXEL_LAYOUT_I420 || p->p.bpc != 8) bad = 1;
    }
    if (bad) drop_held(d);
    return bad ? AV1_BAD : AV1_OK;
}

int av1_receive(Av1Decoder *d, Av1Picture *out) {
    if (!d) return AV1_NONE;
    drop_lent(d);
    if (!d->count) return AV1_NONE;
    d->lent = d->held[d->head];
    d->has_lent = 1;
    d->head = (d->head + 1) % HELD_MAX;
    d->count--;
    const Dav1dPicture *p = &d->lent;
    out->w = p->p.w;
    out->h = p->p.h;
    for (int i = 0; i < 3; i++) out->plane[i] = p->data[i];
    out->stride[0] = p->stride[0];
    out->stride[1] = out->stride[2] = p->stride[1];
    out->ts = p->m.timestamp;
    return AV1_PICTURE;
}

void av1_flush(Av1Decoder *d) {
    if (!d) return;
    drop_lent(d);
    drop_held(d);
    dav1d_flush(d->ctx);
}

void av1_close(Av1Decoder *d) {
    if (!d) return;
    drop_lent(d);
    drop_held(d);
    dav1d_close(&d->ctx);
    free(d);
}

#else   // no dav1d kit: the build has no AV1 decoder

const char *av1_describe(void) { return "none (built without a dav1d kit)"; }
Av1Decoder *av1_open(void) { return NULL; }
int av1_send(Av1Decoder *d, const void *data, size_t size, int64_t ts) { (void)d; (void)data; (void)size; (void)ts; return AV1_BAD; }
int av1_receive(Av1Decoder *d, Av1Picture *out) { (void)d; (void)out; return AV1_NONE; }
void av1_flush(Av1Decoder *d) { (void)d; }
void av1_close(Av1Decoder *d) { (void)d; }

#endif
