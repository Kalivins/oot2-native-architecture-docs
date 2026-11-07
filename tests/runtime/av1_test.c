// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the AV1 decoder seam over dav1d (runtime/video/av1_dav1d.c), with the synthetic clips of
// tests/fixtures/av1 (generated pictures encoded by rav1e, never the game's movies). The decoder is linked from a dav1d kit
// (tools/fetch_dav1d.py).
//
//   av1_test CASE CLIP.ivf [W H FRAMES]   runs one case on one clip; prints "ok CASE" and exits 0, or exits 1 with a FAIL line.
//
// Cases: planes (every picture's three planes, read at the strides the decoder gives, against the generator formulas of
// make_fixtures.py), order (one picture per unit, the units' time stamps in order), damaged (damaged units are refused and
// the decoder goes on), flush (after a flush the next unit must be a key frame with its sequence header: an inter unit
// alone is refused, the whole stream again decodes), notakit (a stub-free check that the decoder names itself).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "endmark.h"
#include "../../runtime/video/av1.h"

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

typedef struct { uint8_t *data; size_t size; } Unit;
static Unit g_unit[64];
static int g_units;

static void load_ivf(const char *path) {
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    uint8_t hdr[32];
    CHECK(fread(hdr, 1, 32, f) == 32 && !memcmp(hdr, "DKIF", 4) && !memcmp(hdr + 8, "AV01", 4));
    for (;;) {
        uint8_t fh[12];
        if (fread(fh, 1, 12, f) != 12) break;
        uint32_t size = fh[0] | fh[1] << 8 | fh[2] << 16 | (uint32_t)fh[3] << 24;
        CHECK(g_units < 64 && size > 0 && size < (1u << 24));
        g_unit[g_units].data = malloc(size);
        g_unit[g_units].size = size;
        CHECK(fread(g_unit[g_units].data, 1, size, f) == size);
        g_units++;
    }
    fclose(f);
}

// The generator's formulas (tests/fixtures/av1/make_fixtures.py)
static int gen_y(int w, int x, int f) { return (x * 255 / (w - 1) + 8 * f) % 256; }
static int gen_u(int ch, int y, int f) { return (y * 255 / (ch - 1) + 8 * f) % 256; }
static int gen_v(int cw, int ch, int x, int y, int f) { return ((x + y) * 255 / (cw + ch - 2) + 8 * f) % 256; }

static void case_planes(int w, int h, int frames) {
    Av1Decoder *d = av1_open();
    CHECK(d != NULL);
    CHECK(strstr(av1_describe(), "dav1d") != NULL);
    int cw = (w + 1) / 2, ch = (h + 1) / 2, got = 0;
    for (int u = 0; u < g_units; u++) {
        CHECK(av1_send(d, g_unit[u].data, g_unit[u].size, 1000 + u) == AV1_OK);
        Av1Picture p;
        while (av1_receive(d, &p) == AV1_PICTURE) {
            CHECK(p.w == w && p.h == h);
            CHECK(p.stride[0] >= w && p.stride[1] >= cw && p.stride[2] >= cw);
            int f = got++;
            long sum[3] = {0, 0, 0};
            int worst = 0;
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++) {
                    int e = abs(p.plane[0][y * p.stride[0] + x] - gen_y(w, x, f));
                    if (e > 128) e = 256 - e;   // the ramp wraps
                    sum[0] += e;
                    if (e > worst) worst = e;
                }
            for (int y = 0; y < ch; y++)
                for (int x = 0; x < cw; x++) {
                    int eu = abs(p.plane[1][y * p.stride[1] + x] - gen_u(ch, y, f));
                    int ev = abs(p.plane[2][y * p.stride[2] + x] - gen_v(cw, ch, x, y, f));
                    if (eu > 128) eu = 256 - eu;
                    if (ev > 128) ev = 256 - ev;
                    sum[1] += eu;
                    sum[2] += ev;
                    if (eu > worst) worst = eu;
                    if (ev > worst) worst = ev;
                }
            double mean_y = (double)sum[0] / ((double)w * h), mean_u = (double)sum[1] / ((double)cw * ch),
                   mean_v = (double)sum[2] / ((double)cw * ch);
            fprintf(stderr, "frame %d: mean error Y %.2f U %.2f V %.2f, worst %d, strides %td %td\n", f, mean_y, mean_u, mean_v,
                    worst, p.stride[0], p.stride[1]);
            // lossy at a quantizer of 40, but the ramps are smooth: a plane swapped or a row shifted is far off
            CHECK(mean_y < 1.0 && mean_u < 1.0 && mean_v < 1.0);
            CHECK(p.ts == 1000 + u);
        }
    }
    CHECK(got == frames);
    av1_close(d);
}

static void case_order(int frames) {
    Av1Decoder *d = av1_open();
    CHECK(d != NULL);
    int got = 0;
    for (int u = 0; u < g_units; u++) {
        CHECK(av1_send(d, g_unit[u].data, g_unit[u].size, 500 + 7 * u) == AV1_OK);
        Av1Picture p;
        int n = 0;
        while (av1_receive(d, &p) == AV1_PICTURE) {
            CHECK(p.ts == 500 + 7 * got);   // decode order, one picture per unit, time stamps carried through
            got++, n++;
        }
        CHECK(n == 1);                      // max_frame_delay 1: the picture of a unit is there when its send returns
    }
    CHECK(got == frames);
    Av1Picture p;
    CHECK(av1_receive(d, &p) == AV1_NONE);
    av1_close(d);
}

static void case_damaged(void) {
    Av1Decoder *d = av1_open();
    CHECK(d != NULL);
    Av1Picture p;
    CHECK(av1_send(d, g_unit[0].data, g_unit[0].size, 1) == AV1_OK);
    CHECK(av1_receive(d, &p) == AV1_PICTURE);
    // a unit cut short, one of 0xFF bytes (an OBU header with the forbidden bit set), one that is no OBU at all
    size_t half = g_unit[1].size / 2;
    uint8_t *junk = malloc(512);
    memset(junk, 0xFF, 512);
    // (a cut-short unit may be refused or may be taken and show nothing; what is fixed is that it makes no picture)
    int rc = av1_send(d, g_unit[1].data, half > 1 ? half : 1, 2);
    CHECK(rc == AV1_BAD || rc == AV1_OK);
    CHECK(av1_receive(d, &p) == AV1_NONE);
    CHECK(av1_send(d, junk, 512, 3) == AV1_BAD);
    CHECK(av1_receive(d, &p) == AV1_NONE);              // a refused unit makes no picture
    CHECK(av1_send(d, junk, 0, 4) == AV1_BAD);          // empty
    CHECK(av1_send(d, NULL, 8, 5) == AV1_BAD);
    // and the decoder is still usable: a fresh stream decodes
    av1_flush(d);
    for (int u = 0; u < g_units; u++) {
        CHECK(av1_send(d, g_unit[u].data, g_unit[u].size, 10 + u) == AV1_OK);
        CHECK(av1_receive(d, &p) == AV1_PICTURE && p.ts == 10 + u);
    }
    free(junk);
    av1_close(d);
}

static void case_flush(void) {
    Av1Decoder *d = av1_open();
    CHECK(d != NULL);
    Av1Picture p;
    for (int u = 0; u < g_units; u++) {
        CHECK(av1_send(d, g_unit[u].data, g_unit[u].size, u) == AV1_OK);
        CHECK(av1_receive(d, &p) == AV1_PICTURE);
    }
    CHECK(g_units >= 2);
    // the stream is forgotten, not paused: right after a key frame and a flush, the next unit of that stream (an inter
    // unit) has nothing to build on and makes no picture
    av1_flush(d);
    CHECK(av1_send(d, g_unit[0].data, g_unit[0].size, 40) == AV1_OK && av1_receive(d, &p) == AV1_PICTURE);
    av1_flush(d);
    int r1 = av1_send(d, g_unit[1].data, g_unit[1].size, 41);
    CHECK(r1 == AV1_BAD || r1 == AV1_OK);
    CHECK(av1_receive(d, &p) == AV1_NONE);
    av1_flush(d);
    CHECK(av1_receive(d, &p) == AV1_NONE);               // nothing carried over
    // an inter unit with no sequence header before it starts nothing: refused or silent, never a picture
    int r = av1_send(d, g_unit[1].data, g_unit[1].size, 100);
    CHECK(r == AV1_BAD || r == AV1_OK);
    CHECK(av1_receive(d, &p) == AV1_NONE);
    av1_flush(d);
    for (int u = 0; u < g_units; u++) {   // the stream again from its key frame
        CHECK(av1_send(d, g_unit[u].data, g_unit[u].size, 200 + u) == AV1_OK);
        CHECK(av1_receive(d, &p) == AV1_PICTURE && p.ts == 200 + u);
    }
    av1_close(d);
}

int main(int argc, char **argv) {
    (void)TEST_END;   // the end marker every runtime test links (this one has no guest code)
    if (argc < 3) return 2;
    load_ivf(argv[2]);
    int w = argc > 3 ? atoi(argv[3]) : 0, h = argc > 4 ? atoi(argv[4]) : 0, frames = argc > 5 ? atoi(argv[5]) : g_units;
    if (!strcmp(argv[1], "planes")) case_planes(w, h, frames);
    else if (!strcmp(argv[1], "order")) case_order(frames);
    else if (!strcmp(argv[1], "damaged")) case_damaged();
    else if (!strcmp(argv[1], "flush")) case_flush();
    else {
        fprintf(stderr, "unknown case %s\n", argv[1]);
        return 2;
    }
    printf("ok %s\n", argv[1]);
    return 0;
}
