// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Builds the update blocks the audio renderer's client sends, and reads the blocks it answers with, for the tests of the
// renderer. Every offset is written out here as a number, from the public description of the block's layout, and none
// is taken from the runtime's own structures, so that a fault in the reader's layout cannot make the test agree with
// itself. A block is built in revisions 15 and 16 (voice records of 0x188 bytes) or before 15 (0x170); the sections
// follow the header in the order the renderer reads them: behavior, memory pools, voice channel resources, voices,
// effects, splitters, mixes, sinks, performance, render info.
#pragma once
#include <stdint.h>
#include <string.h>

typedef struct BkWave {
    uint64_t address, size;
    int32_t start, end;
    uint8_t loop, end_of_stream, sent;
    int32_t loop_count;
    int32_t loop_start;                   // the sample a loop starts again from (0: the buffer's start)
} BkWave;

typedef struct BkVoice {
    uint32_t id;
    uint8_t is_new, in_use, play_state;   // play_state: 0 started, 1 stopped, 2 paused
    uint8_t format;                       // 2: PCM16
    uint32_t sample_rate;
    float pitch, volume;
    uint16_t wave_buffer_index;
    uint8_t flush;
    BkWave wave[4];
    uint8_t channels;                     // channel count of the samples (a mono voice: 1)
    int32_t mix_id;                       // the mix the voice feeds
    float gain[24];                       // channel 0's gain into each buffer of that mix
    uint64_t extra_addr, extra_size;      // the ADPCM coefficient table
    float gain1[24];                      // channel 1's gain into each buffer (the second channel resource of the voice)
    struct { uint8_t enable; float num[3], den[2]; } biquad[2];   // the two voice filters, as floats (revision 15 and later)
    uint8_t via_splitter;                 // 1: the voice feeds splitter splitter_id, not a mix
    uint32_t splitter_id;
} BkVoice;

typedef struct BkMix { float volume; int32_t buffer_count; uint8_t in_use; int32_t dest_mix_id; float matrix[4][4]; } BkMix;
typedef struct BkSink { uint8_t type, in_use, input_count; int8_t inputs[6]; uint8_t downmix_enabled; float downmix[4]; } BkSink;   // type 1: the device

typedef struct BkBlock {
    uint32_t tag;                  // the revision tag as the client writes it ('REV' and a byte)
    int rev;                       // the layout that picks the voice stride
    int voices, effects, sinks, mixes;
    BkVoice voice[8];
    uint32_t pool_state[40];       // the state of memory pool i, as the client sets it
    uint64_t pool_addr[40], pool_size[40];
    BkMix mix[4];                  // mix i, when in_use
    BkSink sink[2];
    uint32_t voice_stride;         // not 0: overrides the stride the section is built with
    uint32_t pools_extra, voices_extra, effects_extra, sinks_extra, behavior_size, total_extra, trailing;
    const uint8_t *splitters;      // splitter section bytes, and their number
    uint32_t splitter_bytes;
    uint32_t mix_bytes;            // 0: every mix in full
    uint8_t second_channel;        // 1: a second channel resource for each voice (id voice x 6 + 1, gains gain1)
} BkBlock;

#define BK_TAG(n) (0x30564552u + ((uint32_t)(n) << 24))   // "REV" and the byte 0x30 + n

static inline void bk_u32(uint8_t *p, uint32_t v) { memcpy(p, &v, 4); }
static inline void bk_u64(uint8_t *p, uint64_t v) { memcpy(p, &v, 8); }
static inline void bk_f32(uint8_t *p, float v) { memcpy(p, &v, 4); }
static inline uint32_t bk_rd32(const uint8_t *p) { uint32_t v; memcpy(&v, p, 4); return v; }
static inline uint64_t bk_rd64(const uint8_t *p) { uint64_t v; memcpy(&v, p, 8); return v; }

static inline uint32_t bk_stride(const BkBlock *b) { return b->voice_stride ? b->voice_stride : b->rev >= 15 ? 0x188 : 0x170; }

// One voice record. The wave buffer array sits at 0x60 before revision 15 and 0x78 from it (the filters are 0x18 bytes
// longer); the fields before the filters are the same.
static inline void bk_voice(uint8_t *p, const BkVoice *v, int rev) {
    uint32_t shift = rev >= 15 ? 0x18 : 0;
    bk_u32(p + 0, v->id);
    p[8] = v->is_new, p[9] = v->in_use, p[10] = v->play_state, p[11] = v->format;
    bk_u32(p + 12, v->sample_rate);
    bk_u32(p + 24, v->channels);
    bk_f32(p + 28, v->pitch);
    bk_f32(p + 32, v->volume);
    bk_u32(p + 0x3c + shift, 4);                          // wave buffer count
    memcpy(p + 0x40 + shift, &v->wave_buffer_index, 2);
    bk_u64(p + 0x48 + shift, v->extra_addr);
    bk_u64(p + 0x50 + shift, v->extra_size);
    bk_u32(p + 0x58 + shift, (uint32_t)v->mix_id);
    bk_u32(p + 0x5c + shift, v->via_splitter ? v->splitter_id : 0xffffffffu);
    if (rev >= 15)
        for (int f = 0; f < 2; f++) {                     // the filters sit before the shifted part, 0x18 bytes each
            uint8_t *q = p + 0x24 + 0x18 * f;
            q[0] = v->biquad[f].enable;
            for (int k = 0; k < 3; k++) bk_f32(q + 4 + 4 * k, v->biquad[f].num[k]);
            for (int k = 0; k < 2; k++) bk_f32(q + 16 + 4 * k, v->biquad[f].den[k]);
        }
    for (int c = 0; c < 6; c++) bk_u32(p + 0x140 + shift + 4 * c, c < v->channels ? v->id * 6 + (uint32_t)c : 0xffffffffu);
    for (int i = 0; i < 4; i++) {
        uint8_t *w = p + 0x60 + shift + 0x38 * i;
        bk_u64(w + 0, v->wave[i].address);
        bk_u64(w + 8, v->wave[i].size);
        bk_u32(w + 16, (uint32_t)v->wave[i].start);
        bk_u32(w + 20, (uint32_t)v->wave[i].end);
        w[24] = v->wave[i].loop, w[25] = v->wave[i].end_of_stream, w[26] = v->wave[i].sent;
        bk_u32(w + 28, (uint32_t)v->wave[i].loop_count);
        bk_u32(w + 48, (uint32_t)v->wave[i].loop_start);
    }
    p[0x159 + shift] = v->flush;                           // flush wave buffer count: after the 6 channel ids (0x140 + 24) and the clear-drop byte
}

// The whole block into buf; returns its size. Header: revision, behavior, memory pools, voices, voice resources,
// effects, mixes, sinks, performance, splitters, render info, 4 reserved words, total.
static inline uint32_t bk_build(uint8_t *buf, const BkBlock *b) {
    const uint32_t stride = bk_stride(b), npools = (uint32_t)b->effects + 4u * (uint32_t)b->voices;
    const uint32_t behavior = b->behavior_size ? b->behavior_size : 0x10, pools = npools * 0x20 + b->pools_extra,
                   resources = (uint32_t)b->voices * 0x70 * (1u + b->second_channel), voices = (uint32_t)b->voices * stride + b->voices_extra,
                   effects = (uint32_t)b->effects * 0xc0 + b->effects_extra, splitters = b->splitter_bytes,
                   mixes = b->mix_bytes ? b->mix_bytes : (uint32_t)b->mixes * 0x930,
                   sinks = (uint32_t)b->sinks * 0x140 + b->sinks_extra, performance = 0, render = 0x10;
    memset(buf, 0, 0x40);
    bk_u32(buf + 0, b->tag);
    bk_u32(buf + 4, behavior);
    bk_u32(buf + 8, pools);
    bk_u32(buf + 12, voices);
    bk_u32(buf + 16, resources);
    bk_u32(buf + 20, effects);
    bk_u32(buf + 24, mixes);
    bk_u32(buf + 28, sinks);
    bk_u32(buf + 32, performance);
    bk_u32(buf + 36, splitters);
    bk_u32(buf + 40, render);
    uint32_t total = 0x40 + behavior + pools + resources + voices + effects + splitters + mixes + sinks + performance + render;
    uint8_t *p = buf + 0x40;
    memset(p, 0, total - 0x40 + b->trailing);
    p += behavior;
    for (uint32_t i = 0; i < npools; i++) {   // memory pools: address, size, state
        bk_u64(p + 0, i < 40 ? b->pool_addr[i] : 0);
        bk_u64(p + 8, i < 40 ? b->pool_size[i] : 0);
        bk_u32(p + 16, i < 40 ? b->pool_state[i] : 0);
        p += 0x20;
    }
    p += pools - npools * 0x20;
    for (int i = 0; i < b->voices; i++, p += 0x70) {   // channel resources: the id of the voice's channel 0, its gains
        bk_u32(p, (uint32_t)i * 6);
        for (int g = 0; g < 24; g++) bk_f32(p + 4 + 4 * g, b->voice[i].gain[g]);
        p[100] = 1;
    }
    if (b->second_channel)
        for (int i = 0; i < b->voices; i++, p += 0x70) {   // channel 1 of each voice
            bk_u32(p, (uint32_t)i * 6 + 1);
            for (int g = 0; g < 24; g++) bk_f32(p + 4 + 4 * g, b->voice[i].gain1[g]);
            p[100] = 1;
        }
    for (int i = 0; i < b->voices; i++, p += stride) bk_voice(p, &b->voice[i], b->rev);
    p += voices - (uint32_t)b->voices * stride;
    p += effects;
    memcpy(p, b->splitters, splitters);
    p += splitters;
    if (!b->mix_bytes) {   // the mixes in full: volume, rate, buffer count, in use, id, the matrix, the destination
        for (int i = 0; i < b->mixes; i++) {
            uint8_t *m = p + 0x930 * i;
            bk_f32(m + 0, b->mix[i].volume);
            bk_u32(m + 4, 48000);
            bk_u32(m + 8, (uint32_t)b->mix[i].buffer_count);
            m[12] = b->mix[i].in_use;
            bk_u32(m + 16, (uint32_t)i);
            for (int r = 0; r < 4; r++)
                for (int c = 0; c < 4; c++) bk_f32(m + 36 + 4 * (24 * r + c), b->mix[i].matrix[r][c]);
            bk_u32(m + 0x924, (uint32_t)b->mix[i].dest_mix_id);
            bk_u32(m + 0x928, 0xffffffffu);
        }
    }
    p += mixes;
    for (int i = 0; i < b->sinks; i++, p += 0x140) {   // sinks: type, in use, the device's name, the input channels
        p[0] = b->sink[i].type, p[1] = b->sink[i].in_use;
        memcpy(p + 32, "MainAudioOut", 13);
        bk_u32(p + 288, b->sink[i].input_count);
        memcpy(p + 292, b->sink[i].inputs, 6);
        p[299] = b->sink[i].downmix_enabled;
        for (int k = 0; k < 4; k++) bk_f32(p + 300 + 4 * k, b->sink[i].downmix[k]);
    }
    p += sinks - (uint32_t)b->sinks * 0x140 + performance + render;
    bk_u32(buf + 60, total + b->total_extra);
    return total + b->trailing;
}

// One splitter, number 0, with ndest destinations 0 to ndest - 1, in the layout of revision 15 and later: a header (SNDH, one
// info, ndest destinations) of 0x20 bytes, the info (SNDI, the id, the destination count) with the destination ids and 12
// reserved bytes, then each destination (SNDD, its id, 24 volumes at 8, the mix it feeds at 0x68, the filters, and the flag
// "used" at 0x9c) at the stride 0xa8; the section is padded to 16 bytes. Returns its size.
typedef struct BkSplitDest { float vol[24]; int32_t mix; } BkSplitDest;
static inline uint32_t bk_splitter(uint8_t *buf, int ndest, const BkSplitDest *d) {
    uint32_t size = 0x20 + 16 + 4u * (uint32_t)ndest + 12 + 0xa8u * (uint32_t)ndest;
    size = (size + 15) & ~15u;
    memset(buf, 0, size);
    bk_u32(buf + 0, 0x48444e53u);   // SNDH
    bk_u32(buf + 4, 1);
    bk_u32(buf + 8, (uint32_t)ndest);
    uint8_t *p = buf + 0x20;
    bk_u32(p + 0, 0x49444e53u);     // SNDI
    bk_u32(p + 4, 0);
    bk_u32(p + 12, (uint32_t)ndest);
    p += 16;
    for (int i = 0; i < ndest; i++) bk_u32(p + 4 * i, (uint32_t)i);
    p += 4u * (uint32_t)ndest + 12;
    for (int i = 0; i < ndest; i++, p += 0xa8) {
        bk_u32(p + 0, 0x44444e53u);   // SNDD
        bk_u32(p + 4, (uint32_t)i);
        for (int g = 0; g < 24; g++) bk_f32(p + 8 + 4 * g, d[i].vol[g]);
        bk_u32(p + 0x68, (uint32_t)d[i].mix);
        p[0x9c] = 1;
    }
    return size;
}

// ---- the answer: header 0x40, memory pools (0x10 each), voices (0x10 each), effects (0x90 each from revision 9),
// sinks (0x20 each), performance 0x10, behavior 0xb0, render info 0x10
typedef struct BkAnswer {
    uint32_t tag, total;
    uint32_t pool_state[40];
    uint64_t played[8];
    uint32_t consumed[8], drops[8];
    uint8_t effect_status[4];
    uint64_t elapsed_frames;
} BkAnswer;

static inline void bk_read(const uint8_t *out, const BkBlock *b, BkAnswer *a) {
    memset(a, 0, sizeof *a);
    a->tag = bk_rd32(out);
    a->total = bk_rd32(out + 60);
    const uint32_t npools = (uint32_t)b->effects + 4u * (uint32_t)b->voices, eo = b->rev >= 9 ? 0x90 : 0x10;
    const uint8_t *p = out + 0x40;
    for (uint32_t i = 0; i < npools && i < 40; i++) a->pool_state[i] = bk_rd32(p + 0x10 * i);
    p += npools * 0x10;
    for (int i = 0; i < b->voices; i++) {
        a->played[i] = bk_rd64(p + 0x10 * i);
        a->consumed[i] = bk_rd32(p + 0x10 * i + 8);
        a->drops[i] = bk_rd32(p + 0x10 * i + 12);
    }
    p += (uint32_t)b->voices * 0x10;
    for (int i = 0; i < b->effects && i < 4; i++) a->effect_status[i] = p[eo * (uint32_t)i];
    p += (uint32_t)b->effects * eo + (uint32_t)b->sinks * 0x20 + 0x10 + 0xb0;
    a->elapsed_frames = bk_rd64(p);
}
