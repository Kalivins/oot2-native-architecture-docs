// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the Opus decoder behind the seam of runtime/audio/opus.h (runtime/audio/opus_libopus.c), linked from a
// libopus kit (tools/fetch_opus.py). The streams are made here, by libopus's own encoder, from generated tones: nothing
// comes from a game.
//
//   opus_test CASE   runs one case; prints "ok CASE" and exits 0, or exits 1 with a FAIL line.
//
// Cases: stereo_tones (a stereo stream at 48 kHz decodes to the frames the encoder was given, each channel carries its own
// tone at its own level and nothing of the other's, and the final range of every packet is the encoder's), mono_16k,
// all_formats (every rate and channel count the hardware decoder takes, the final ranges again), frame_sizes (2.5 ms to 60
// ms packets), garbage (an invalid packet is refused and the decoder goes on), too_small (an output buffer too small for the
// packet is refused), empty (a packet of no bytes makes no sample), reset (a reset gives the first packet's samples again,
// where a decoder that was not reset gives others), create (the rates and channel counts the decoder refuses), table (the
// build's table is this one, and an installed one wins). Built without -DOOT2_OPUS the one case is nokit: the build has no
// table, an installed one is the build's, and removing it leaves none.
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../../runtime/audio/opus.h"

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#ifdef OOT2_OPUS
#include <opus/opus.h>

#define MAX_PACKETS 64
#define MAX_PACKET_BYTES 4000
#define LEVEL 12000.0   // the tone's amplitude on the 16-bit scale

typedef struct {
    int rate, channels, frame, count;   // frame: frames per channel in a packet
    uint8_t data[MAX_PACKETS][MAX_PACKET_BYTES];
    int len[MAX_PACKETS];
    uint32_t range[MAX_PACKETS];        // the encoder's final range after each packet
    int16_t *pcm;                       // what the encoder was given, interleaved
} Stream;

static double tone_hz(int channel) { return channel == 0 ? 500.0 : 1000.0; }

// An encoded stream of `count` packets of `frame` frames, channel c carrying tone_hz(c) (a mono stream carries the first).
static Stream *make_stream(int rate, int channels, int frame, int count, int bitrate) {
    CHECK(count <= MAX_PACKETS);
    Stream *s = calloc(1, sizeof *s);
    CHECK(s);
    s->rate = rate, s->channels = channels, s->frame = frame, s->count = count;
    size_t frames = (size_t)frame * (size_t)count;
    s->pcm = malloc(frames * (size_t)channels * sizeof(int16_t));
    CHECK(s->pcm);
    for (size_t i = 0; i < frames; i++)
        for (int c = 0; c < channels; c++)
            s->pcm[i * (size_t)channels + (size_t)c] = (int16_t)lrint(LEVEL * sin(6.283185307179586 * tone_hz(c) * (double)i / rate));
    int err = 0;
    OpusEncoder *enc = opus_encoder_create(rate, channels, OPUS_APPLICATION_AUDIO, &err);
    CHECK(enc && err == OPUS_OK);
    CHECK(opus_encoder_ctl(enc, OPUS_SET_BITRATE(bitrate)) == OPUS_OK);
    CHECK(opus_encoder_ctl(enc, OPUS_SET_COMPLEXITY(10)) == OPUS_OK);
    for (int p = 0; p < count; p++) {
        int n = opus_encode(enc, s->pcm + (size_t)p * (size_t)frame * (size_t)channels, frame, s->data[p], MAX_PACKET_BYTES);
        CHECK(n > 0);
        s->len[p] = n;
        opus_uint32 range = 0;
        CHECK(opus_encoder_ctl(enc, OPUS_GET_FINAL_RANGE(&range)) == OPUS_OK);
        s->range[p] = range;
    }
    opus_encoder_destroy(enc);
    return s;
}

static void free_stream(Stream *s) {
    free(s->pcm);
    free(s);
}

// The amplitude of the tone at hz in channel c of pcm over frames [from, from + n), by correlation with a sine and a cosine.
static double tone_level(const int16_t *pcm, int channels, int c, int rate, double hz, int from, int n) {
    double re = 0, im = 0;
    for (int i = from; i < from + n; i++) {
        double x = pcm[(size_t)i * (size_t)channels + (size_t)c], a = 6.283185307179586 * hz * (double)i / rate;
        re += x * cos(a);
        im += x * sin(a);
    }
    return 2.0 * sqrt(re * re + im * im) / n;
}

// Decode every packet of s into one buffer; the frames per packet and the final range of each are checked on the way.
static int16_t *decode_all(const OpusApi *api, void *dec, const Stream *s) {
    int16_t *out = malloc((size_t)s->frame * (size_t)s->count * (size_t)s->channels * sizeof(int16_t));
    CHECK(out);
    for (int p = 0; p < s->count; p++) {
        uint32_t range = 0xdeadbeef;
        int got = api->decode(dec, s->data[p], s->len[p], out + (size_t)p * (size_t)s->frame * (size_t)s->channels, s->frame, &range);
        CHECK(got == s->frame);
        CHECK(range == s->range[p]);
    }
    return out;
}

// What the decoder gives is the stream's tones, each in its own channel: the level of the tone in its channel is the one
// that was encoded and the other channel's tone is not there. The first 100 ms are skipped (the encoder's look-ahead and the
// decoder's first frames), and the window holds whole periods of every tone.
static void check_tones(const int16_t *pcm, const Stream *s) {
    int from = s->rate / 10, n = s->rate / 5;
    for (int c = 0; c < s->channels; c++) {
        double own = tone_level(pcm, s->channels, c, s->rate, tone_hz(c), from, n);
        CHECK(own > 0.9 * LEVEL && own < 1.1 * LEVEL);
        if (s->channels == 2) CHECK(tone_level(pcm, s->channels, c, s->rate, tone_hz(1 - c), from, n) < 0.05 * LEVEL);
    }
}

static const OpusApi *table(void) {
    const OpusApi *api = audio_opus();
    CHECK(api != NULL);
    return api;
}

static void *make_decoder(const OpusApi *api, int rate, int channels) {
    char why[160] = "";
    void *d = api->create(rate, channels, why, sizeof why);
    if (!d) fprintf(stderr, "create(%d, %d): %s\n", rate, channels, why);
    CHECK(d != NULL);
    return d;
}

static void case_stereo_tones(void) {
    const OpusApi *api = table();
    Stream *s = make_stream(48000, 2, 960, 25, 128000);
    void *dec = make_decoder(api, 48000, 2);
    int16_t *pcm = decode_all(api, dec, s);
    check_tones(pcm, s);
    free(pcm);
    api->destroy(dec);
    free_stream(s);
}

static void case_mono_16k(void) {
    const OpusApi *api = table();
    Stream *s = make_stream(16000, 1, 320, 25, 32000);
    void *dec = make_decoder(api, 16000, 1);
    int16_t *pcm = decode_all(api, dec, s);
    check_tones(pcm, s);
    free(pcm);
    api->destroy(dec);
    free_stream(s);
}

static void case_all_formats(void) {
    const OpusApi *api = table();
    static const int rates[] = {8000, 12000, 16000, 24000, 48000};
    for (int r = 0; r < 5; r++)
        for (int channels = 1; channels <= 2; channels++) {
            int frame = rates[r] / 50;   // 20 ms
            Stream *s = make_stream(rates[r], channels, frame, 25, 48000 * channels);
            void *dec = make_decoder(api, rates[r], channels);
            int16_t *pcm = decode_all(api, dec, s);
            check_tones(pcm, s);
            free(pcm);
            api->destroy(dec);
            free_stream(s);
        }
}

static void case_frame_sizes(void) {
    const OpusApi *api = table();
    static const int sizes[] = {120, 240, 480, 960, 1920, 2880};   // 2.5, 5, 10, 20, 40 and 60 ms at 48 kHz
    for (int i = 0; i < 6; i++) {
        Stream *s = make_stream(48000, 2, sizes[i], 6, 96000);
        void *dec = make_decoder(api, 48000, 2);
        int16_t *pcm = decode_all(api, dec, s);   // frames and final range of every packet
        free(pcm);
        api->destroy(dec);
        free_stream(s);
    }
}

static void case_garbage(void) {
    const OpusApi *api = table();
    Stream *s = make_stream(48000, 2, 960, 6, 128000);
    void *dec = make_decoder(api, 48000, 2);
    int16_t pcm[960 * 2];
    uint32_t range;
    const uint8_t bad_count[2] = {0x03, 0x00};   // a code 3 packet that says it holds no frame
    CHECK(api->decode(dec, bad_count, 2, pcm, 960, &range) < 0);
    int16_t *all = decode_all(api, dec, s);   // the decoder goes on: every packet decodes, to the final range of the encoder
    free(all);
    const uint8_t odd[2] = {0x05, 0xff};   // a code 1 packet (two frames of one size) with an odd number of bytes after the header
    CHECK(api->decode(dec, odd, 2, pcm, 960, &range) < 0);
    CHECK(api->decode(dec, s->data[0], s->len[0], pcm, 960, &range) == 960);
    api->destroy(dec);
    free_stream(s);
}

static void case_too_small(void) {
    const OpusApi *api = table();
    Stream *s = make_stream(48000, 2, 960, 2, 128000);
    void *dec = make_decoder(api, 48000, 2);
    int16_t pcm[960 * 2];
    uint32_t range;
    CHECK(api->decode(dec, s->data[0], s->len[0], pcm, 480, &range) < 0);   // 10 ms of room for a 20 ms packet
    CHECK(api->decode(dec, s->data[0], s->len[0], pcm, 960, &range) == 960 && range == s->range[0]);
    api->destroy(dec);
    free_stream(s);
}

static void case_empty(void) {
    const OpusApi *api = table();
    void *dec = make_decoder(api, 48000, 2);
    int16_t pcm[5760 * 2];
    memset(pcm, 0x55, sizeof pcm);
    uint32_t range = 7;
    const uint8_t none[1] = {0};
    CHECK(api->decode(dec, none, 0, pcm, 5760, &range) == 0);   // no packet, no sample, and not 120 ms of concealment
    CHECK(range == 0 && pcm[0] == 0x5555);
    api->destroy(dec);
}

static void case_reset(void) {
    const OpusApi *api = table();
    Stream *s = make_stream(48000, 2, 960, 8, 128000);
    void *dec = make_decoder(api, 48000, 2);
    int16_t *first = decode_all(api, dec, s);
    size_t bytes = (size_t)s->frame * 2 * sizeof(int16_t);
    int16_t again[960 * 2];
    uint32_t range;
    // not reset: the first packet again lands on the state the others left, and the samples differ from the first time
    CHECK(api->decode(dec, s->data[0], s->len[0], again, 960, &range) == 960);
    CHECK(memcmp(again, first, bytes) != 0);
    CHECK(api->reset(dec) == 0);
    CHECK(api->decode(dec, s->data[0], s->len[0], again, 960, &range) == 960);
    CHECK(memcmp(again, first, bytes) == 0 && range == s->range[0]);
    free(first);
    api->destroy(dec);
    free_stream(s);
}

static void case_create(void) {
    const OpusApi *api = table();
    char why[160] = "";
    CHECK(api->create(44100, 2, why, sizeof why) == NULL);
    CHECK(strstr(why, "44100") != NULL);
    why[0] = 0;
    CHECK(api->create(48000, 3, why, sizeof why) == NULL);
    CHECK(strstr(why, "3 channels") != NULL);
    CHECK(api->create(48000, 0, NULL, 0) == NULL);   // no room for a reason is no crash
    static const int rates[] = {8000, 12000, 16000, 24000, 48000};
    for (int r = 0; r < 5; r++)
        for (int channels = 1; channels <= 2; channels++) api->destroy(make_decoder(api, rates[r], channels));
}

static OpusApi g_other = {"another decoder", NULL, NULL, NULL, NULL};

static void case_table(void) {
    const OpusApi *api = table();
    CHECK(api == audio_opus_libopus());
    CHECK(!strcmp(api->name, opus_get_version_string()) && !strncmp(api->name, "libopus ", 8));
    CHECK(api->create && api->destroy && api->reset && api->decode);
    audio_opus_install(&g_other);
    CHECK(audio_opus() == &g_other);
    audio_opus_install(NULL);
    CHECK(audio_opus() == api);
}

#else   // no kit: the build has no decoder of its own

static OpusApi g_other = {"another decoder", NULL, NULL, NULL, NULL};

static void case_nokit(void) {
    CHECK(audio_opus() == NULL);
    audio_opus_install(&g_other);
    CHECK(audio_opus() == &g_other);
    audio_opus_install(NULL);
    CHECK(audio_opus() == NULL);
}
#endif

int main(int argc, char **argv) {
    if (argc != 2) return 2;
#ifndef OOT2_OPUS
    if (!strcmp(argv[1], "nokit")) {
        case_nokit();
        printf("ok nokit\n");
        return 0;
    }
    fprintf(stderr, "unknown case %s\n", argv[1]);
    return 2;
#else
    static const struct { const char *name; void (*run)(void); } cases[] = {
        {"stereo_tones", case_stereo_tones}, {"mono_16k", case_mono_16k}, {"all_formats", case_all_formats},
        {"frame_sizes", case_frame_sizes}, {"garbage", case_garbage}, {"too_small", case_too_small}, {"empty", case_empty},
        {"reset", case_reset}, {"create", case_create}, {"table", case_table}};
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++)
        if (!strcmp(argv[1], cases[i].name)) {
            cases[i].run();
            printf("ok %s\n", argv[1]);
            return 0;
        }
    fprintf(stderr, "unknown case %s\n", argv[1]);
    return 2;
#endif
}
