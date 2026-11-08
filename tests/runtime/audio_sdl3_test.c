// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the audio sink's platform stream (runtime/platform/audio_sdl3.c, through runtime/audio/sink.c), linked from
// an SDL3 kit (tools/fetch_sdl3.py). The device is SDL's own dummy driver, or its disk driver that writes what the device
// would play to a file: no sound card is needed or touched.
//
//   audio_sdl3_test CASE   runs one case; prints "ok CASE" and exits 0, or exits 1 with a FAIL line.
//
// Cases: open (the sink opens on the dummy driver and says so, a driver that does not exist is refused by sdl3 and falls back
// to the null sink for auto), queue_cap (writes without waiting are accepted up to about 30 ms and then refused whole, and
// the refusals are counted), underrun (a queue the device drained is counted at the next write, which waits for a prefill
// again), pause (a paused sink holds no queue and counts no underrun), drain (the device takes what was written, at the
// device's rate), samples (the disk driver's file holds the frames the sink was given, in order, to the last bit).
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <SDL3/SDL.h>

#include "oot2/audio_sink.h"
#include "oot2/runtime.h"
#include "plat.h"

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define RATE 48000
#define CHUNK 240   // 5 ms, the renderer's frame

// The two calls of the runtime's fatal path that live in other files than the ones this test links: a stop is not expected here.
int rt_census_on(void) { return 0; }
const char *rt_guest_name(uint64_t pc, char *buf, int size) {
    snprintf(buf, (size_t)size, "%#llx", (unsigned long long)pc);
    return buf;
}

static void sleep_ms(int ms) { plat_sleep_ns((uint64_t)ms * 1000000ull); }

static void tone(float *lr, uint32_t first, uint32_t frames) {
    for (uint32_t i = 0; i < frames; i++) {
        double t = (double)(first + i) / RATE;
        lr[2 * i] = (float)(0.5 * sin(6.283185307179586 * 440.0 * t));
        lr[2 * i + 1] = (float)(0.25 * sin(6.283185307179586 * 880.0 * t));
    }
}

static AudioSink *open_sdl3(void) {
    AudioSink *s = audio_sink_open(AUDIO_SINK_SDL3, RATE, NULL);
    if (!s) fprintf(stderr, "open: %s\n", audio_sink_why());
    CHECK(s != NULL);
    return s;
}

static void case_open(void) {
    AudioSink *s = open_sdl3();
    CHECK(audio_sink_kind(s) == AUDIO_SINK_SDL3 && !strcmp(audio_sink_kind_name(s), "sdl3"));
    CHECK(strstr(audio_sink_device(s), "dummy") != NULL || strstr(audio_sink_device(s), "disk") != NULL);
    CHECK(strstr(audio_sink_device(s), "30 ms") != NULL);
    audio_sink_close(s);
    AudioSink *a = audio_sink_open(AUDIO_SINK_AUTO, RATE, NULL);   // auto picks the platform stream when it opens
    CHECK(a != NULL && audio_sink_kind(a) == AUDIO_SINK_SDL3);
    audio_sink_close(a);
    CHECK(audio_sink_open(AUDIO_SINK_SDL3, 100, NULL) == NULL);   // a rate no device takes
    CHECK(strstr(audio_sink_why(), "100 Hz") != NULL);
    SDL_SetHintWithPriority(SDL_HINT_AUDIO_DRIVER, "no-such-driver", SDL_HINT_OVERRIDE);   // the environment names the real one
    CHECK(audio_sink_open(AUDIO_SINK_SDL3, RATE, NULL) == NULL);   // asked for by name: refused, with the reason
    CHECK(audio_sink_why()[0] != 0);
    AudioSink *n = audio_sink_open(AUDIO_SINK_AUTO, RATE, NULL);   // auto: the null sink, and why
    CHECK(n != NULL && audio_sink_kind(n) == AUDIO_SINK_NULL && !strncmp(audio_sink_device(n), "no device: ", 11));
    audio_sink_close(n);
}

static void case_queue_cap(void) {
    AudioSink *s = open_sdl3();
    float lr[2 * CHUNK];
    tone(lr, 0, CHUNK);
    uint64_t accepted = 0, refused = 0;
    for (int i = 0; i < 400; i++) {   // 2 s of audio in no time at all
        uint32_t got = audio_sink_write(s, lr, CHUNK);
        CHECK(got == 0 || got == CHUNK);   // whole or nothing
        if (got) accepted += got;
        else refused++;
    }
    AudioSinkStats st;
    audio_sink_stats(s, &st);
    uint64_t cap = RATE * 30ull / 1000;
    CHECK(accepted >= cap - CHUNK);                 // the queue was filled to its cap
    CHECK(accepted <= cap + 2ull * RATE / 100);     // and not past it, but for what the device may have taken meanwhile
    CHECK(refused >= 300 && st.overruns == refused);
    CHECK(st.written == accepted && st.queued <= cap);
    audio_sink_close(s);
}

static void case_underrun(void) {
    AudioSink *s = open_sdl3();
    float lr[2 * CHUNK];
    tone(lr, 0, CHUNK);
    for (int i = 0; i < 5; i++) CHECK(audio_sink_write(s, lr, CHUNK) == CHUNK);   // 25 ms: past the prefill, the device runs
    AudioSinkStats st;
    audio_sink_stats(s, &st);
    CHECK(st.underruns == 0);
    sleep_ms(300);   // far longer than the queue lasts
    audio_sink_stats(s, &st);
    CHECK(st.queued == 0 && st.played == st.written);
    CHECK(audio_sink_write(s, lr, CHUNK) == CHUNK);   // the next write finds it empty
    audio_sink_stats(s, &st);
    CHECK(st.underruns == 1);
    sleep_ms(100);   // 5 ms queued is under the prefill: the device is held, and the queue stays
    audio_sink_stats(s, &st);
    CHECK(st.queued == CHUNK && st.underruns == 1);
    audio_sink_close(s);
}

static void case_pause(void) {
    AudioSink *s = open_sdl3();
    float lr[2 * CHUNK];
    tone(lr, 0, CHUNK);
    for (int i = 0; i < 5; i++) CHECK(audio_sink_write(s, lr, CHUNK) == CHUNK);
    audio_sink_pause(s, 1);
    AudioSinkStats st;
    audio_sink_stats(s, &st);
    CHECK(st.queued == 0);   // a paused sink holds no queue
    sleep_ms(200);
    audio_sink_pause(s, 0);
    CHECK(audio_sink_write(s, lr, CHUNK) == CHUNK);
    audio_sink_stats(s, &st);
    CHECK(st.underruns == 0 && st.queued == CHUNK);   // no underrun for the stop, and the prefill starts again
    audio_sink_close(s);
}

static void case_drain(void) {
    AudioSink *s = open_sdl3();
    float lr[2 * CHUNK];
    uint32_t first = 0;
    for (int i = 0; i < 100; i++) {   // 500 ms written at the renderer's own pace
        tone(lr, first, CHUNK);
        first += audio_sink_write(s, lr, CHUNK);
        sleep_ms(5);
    }
    AudioSinkStats st;
    audio_sink_stats(s, &st);
    CHECK(st.written >= 100ull * CHUNK * 8 / 10);   // nearly every frame fit at that pace
    CHECK(st.played >= st.written / 2);             // and the device took the most of them while it ran
    CHECK(st.queued <= RATE * 30ull / 1000);
    audio_sink_close(s);
}

// The disk driver writes what the device is given as the device's format; the sink's frames come out unchanged.
static void case_samples(void) {
    const char *path = getenv("OOT2_TEST_DEVICE_FILE");   // the runner names the file
    CHECK(path != NULL);
    SDL_SetHintWithPriority(SDL_HINT_AUDIO_DRIVER, "disk", SDL_HINT_OVERRIDE);
    SDL_SetHint(SDL_HINT_AUDIO_DISK_OUTPUT_FILE, path);
    SDL_SetHint(SDL_HINT_AUDIO_DISK_TIMESCALE, "0.5");   // a device slower than the feed: no underrun, whatever the machine does
    AudioSink *s = open_sdl3();
    CHECK(strstr(audio_sink_device(s), "disk") != NULL);
    enum { TOTAL = RATE / 5 };   // 200 ms
    static float all[2 * TOTAL];
    tone(all, 0, TOTAL);
    uint32_t sent = 0;
    while (sent < TOTAL) {
        uint32_t n = audio_sink_write(s, all + 2 * sent, CHUNK);
        sent += n;
        if (!n) sleep_ms(2);   // the queue is at its cap: wait for the device
    }
    sleep_ms(300);   // the device takes the rest
    audio_sink_close(s);
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    static float got[2 * (TOTAL + 4800)];
    size_t n = fread(got, sizeof(float), 2 * (TOTAL + 4800), f);
    fclose(f);
    // The device runs silent from its opening until the prefill has queued (a few hundred frames of zeros), then plays the
    // frames in order. The tone's first frame is zero, so the first frame that is not zero is the second of the stream.
    size_t at = 0;
    while (at < n / 2 && got[2 * at] == 0 && got[2 * at + 1] == 0) at++;
    CHECK(at >= 1 && at - 1 <= 2400);   // at most 50 ms of lead-in
    size_t lead = at - 1;
    CHECK(n >= 2 * (lead + TOTAL));     // the device took every frame
    CHECK(memcmp(got + 2 * lead, all, sizeof all) == 0);   // and they are the frames the sink was given, to the last bit
}

int main(int argc, char **argv) {
    if (argc != 2) return 2;
    const char *c = argv[1];
    if (!strcmp(c, "open")) case_open();
    else if (!strcmp(c, "queue_cap")) case_queue_cap();
    else if (!strcmp(c, "underrun")) case_underrun();
    else if (!strcmp(c, "pause")) case_pause();
    else if (!strcmp(c, "drain")) case_drain();
    else if (!strcmp(c, "samples")) case_samples();
    else {
        fprintf(stderr, "unknown case %s\n", c);
        return 2;
    }
    printf("ok %s\n", c);
    return 0;
}
