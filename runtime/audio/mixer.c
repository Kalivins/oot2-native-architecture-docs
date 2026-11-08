// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The mixing stage of the audio renderer (mixer.h). Per frame, under the renderer's lock: every started voice is decoded
// along the sample path its own clock walks (PCM8, 16, 24, 32, float; DSP-ADPCM with its history carried across frames),
// resampled linearly, filtered by its two biquads, and scaled into its destination mix (channel gain, then the voice's
// volume) or into a splitter's destinations; the sub mixes are folded into their destinations in dependency order (a
// mix's volume applies when it is folded), the final mix is mapped through the device sink's inputs (down to stereo when
// it has more than two, with the sink's downmix coefficients or the usual 0.7071) and scaled by the final mix's volume
// and the device volume.
//
// Effects are accepted and passed through dry: a delay, a reverb, a biquad or a compressor copies its input buffers to
// its output buffers, a buffer mix adds them scaled, and an aux, limiter, capture or 3D reverb effect (and a circular
// sink) does nothing; the first time a type is seen enabled, one log line says so.
//
// Wave buffer memory is guest memory. A range is read only after it is found inside an attached memory pool and
// rt_mem_commit_io has accepted it for reading; a range that fails is silent, and the first few are named in the log
// (voice, slot, address, size), never read blind.
//
// Adapted from DKB Recomp runtime/audio_mixer.c at c5d5e7c (MIT; the file's header: adapted from MacNativeGames
// games/totk/runtime/audio_mixer.c at b75be8c, MIT, Copyright (c) 2026 MacNativeGames contributors; the algorithms
// follow Ryujinx.Audio, MIT, Copyright (c) Ryujinx Team and Contributors). Changed here: no device, ring or pacing (the
// frame goes to the sink of oot2/audio_sink.h and the renderer's own clock paces it); the sample position of the walk
// comes from the voice's integer accumulator, the same one the renderer advances by, so the samples a frame sounds are the
// samples it releases; scratch buffers are sized from the renderer's sample count; wave buffer reads are checked against
// the attached pools; the effects are passed through dry; the splitter section is stored here after audren.c has walked it;
// the voice volume ramp, the level meter and the TOTK settings are left out.
#include <math.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "mixer.h"
#include "oot2/memory.h"
#include "oot2/runtime.h"

#define SPLITTER_UNUSED 0xffffffffu

typedef struct { float vol[24]; int32_t dest; uint8_t used; AudrenBiquadFloat biquads[2]; float state[2][2]; } SplitDest;
typedef struct { int32_t ndest; int32_t *ids; } SplitInfo;

// A range of a wave buffer (or a voice's decoder tables) found readable, kept while the buffer and the pools stay the same.
typedef struct { uint64_t addr, size; const uint8_t *ptr; uint32_t pool_gen; uint8_t state; } Span;   // state: 0 unchecked, 1 readable, 2 refused

typedef struct {
    int16_t h1, h2;           // DSP-ADPCM history
    int32_t abuf;             // the slot the history belongs to
    uint64_t aat;             // the next sample the decoder has not produced
    float bq[2][6][2];        // the two voice biquads, per channel
    Span span[4], context[4], coef;   // the slots' samples, their ADPCM contexts, the coefficient table
} MixVoice;

struct Mixer {
    int frame, nbuf, mixes;
    float *buf;               // nbuf mix buffers of frame samples
    int *offset;              // first mix buffer of each mix id
    SplitInfo *split; int nsplit;
    SplitDest *dest; int ndest;
    MixVoice *voice;
    float *scratch[6], *filtered;   // one voice's decoded channels, and a filter's output
    unsigned reported, effect_reported;
    unsigned refused;         // wave buffer ranges named in the log so far
    unsigned routed;          // routing lines logged so far
    uint32_t edges[256][8];   // mix_order's scratch: which sub mix feeds which (one frame at a time, under the renderer's lock)
    MixerStats stats;
};

static _Atomic uint32_t g_device_volume_bits = 0x3f800000u;   // 1.0f
void audio_device_volume_set(float v) {
    uint32_t b;
    if (!(v == v)) v = 0.0f;
    memcpy(&b, &v, 4);
    atomic_store_explicit(&g_device_volume_bits, b, memory_order_relaxed);
}
float audio_device_volume(void) {
    uint32_t b = atomic_load_explicit(&g_device_volume_bits, memory_order_relaxed);
    float v;
    memcpy(&v, &b, 4);
    return v;
}

static void report_once(Mixer *m, unsigned bit, const char *msg, int value) {
    if (m->reported & (1u << bit)) return;
    m->reported |= 1u << bit;
    rt_log("audio: %s (%d)", msg, value);
}

// ---- open and close
Mixer *mixer_open(AudioRenderer *r) {
    Mixer *m = calloc(1, sizeof *m);
    if (!m) rt_fatal("audio: mixer allocation");
    m->frame = r->cfg.sample_count;
    m->nbuf = r->cfg.mix_buffer_count;
    m->mixes = r->cfg.submix_count + 1;
    m->nsplit = r->cfg.splitter_count;
    m->ndest = r->cfg.splitter_destination_count;
    m->buf = calloc((size_t)m->nbuf * (size_t)m->frame, sizeof(float));
    m->offset = calloc((size_t)m->mixes, sizeof(int));
    m->split = calloc((size_t)(m->nsplit ? m->nsplit : 1), sizeof *m->split);
    m->dest = calloc((size_t)(m->ndest ? m->ndest : 1), sizeof *m->dest);
    m->voice = calloc((size_t)r->cfg.voice_count, sizeof *m->voice);
    m->filtered = calloc((size_t)m->frame, sizeof(float));
    for (int c = 0; c < 6; c++) m->scratch[c] = calloc((size_t)m->frame, sizeof(float));
    if (!m->buf || !m->offset || !m->split || !m->dest || !m->voice || !m->filtered) rt_fatal("audio: mixer tables");
    for (int c = 0; c < 6; c++)
        if (!m->scratch[c]) rt_fatal("audio: mixer scratch");
    return m;
}

void mixer_close(Mixer *m) {
    if (!m) return;
    for (int i = 0; i < m->nsplit; i++) free(m->split[i].ids);
    for (int c = 0; c < 6; c++) free(m->scratch[c]);
    free(m->filtered);
    free(m->voice);
    free(m->dest);
    free(m->split);
    free(m->offset);
    free(m->buf);
    free(m);
}

MixerStats mixer_stats(Mixer *m, int reset) {
    MixerStats s = m->stats;
    if (reset) m->stats = (MixerStats){0};
    return s;
}

void mixer_voice_reset(Mixer *m, unsigned id) {
    MixVoice *v = &m->voice[id];
    v->h1 = v->h2 = 0;
    v->abuf = -1;
    v->aat = 0;
    memset(v->bq, 0, sizeof v->bq);
}

// ---- splitters: infos (the destination ids of each splitter) and destinations (volumes, target mix, filters)
void mixer_set_splitters(Mixer *m, const uint8_t *p, uint32_t n, int rev) {
    if (!n) return;
    uint32_t hdr[3];
    memcpy(hdr, p, 12);
    uint32_t ni = hdr[1], nd = hdr[2], at = 0x20;
    for (uint32_t i = 0; i < ni; i++) {
        uint32_t h[4];
        memcpy(h, p + at, 16);
        at += 16;
        SplitInfo *s = &m->split[h[1]];
        free(s->ids);
        s->ndest = (int32_t)h[3];
        s->ids = calloc(h[3] ? h[3] : 1, sizeof(int32_t));
        if (!s->ids) rt_fatal("audio: splitter destinations");
        memcpy(s->ids, p + at, 4u * h[3]);
        at += 4u * h[3] + 12;   // 12 reserved bytes follow the ids
        for (uint32_t k = 0; k < h[3]; k++)
            if (s->ids[k] < 0 || s->ids[k] >= m->ndest) s->ids[k] = -1;   // a destination that does not exist is skipped
    }
    const uint32_t stride = rev >= 15 ? 0xa8 : rev >= 12 ? 0x90 : 0x70;
    for (uint32_t i = 0; i < nd; i++) {
        uint8_t e[0xa8] = {0};
        memcpy(e, p + at, stride);
        at += stride;
        uint32_t id;
        int32_t dst;
        memcpy(&id, e + 4, 4);
        memcpy(&dst, e + 0x68, 4);
        SplitDest *d = &m->dest[id];
        memcpy(d->vol, e + 8, sizeof d->vol);
        d->dest = dst;
        d->used = e[rev >= 15 ? 0x9c : rev >= 12 ? 0x84 : 0x6c];
        for (unsigned f = 0; f < 2; f++) {
            AudrenBiquadFloat b = {0};
            if (rev >= 15) {
                memcpy(&b, e + 0x6c + f * sizeof b, sizeof b);
            } else if (rev >= 12) {
                AudrenBiquad q;
                memcpy(&q, e + 0x6c + f * sizeof q, sizeof q);
                b.enable = q.enable;
                for (unsigned c = 0; c < 3; c++) b.numerator[c] = q.numerator[c] / 16384.0f;
                for (unsigned c = 0; c < 2; c++) b.denominator[c] = q.denominator[c] / 16384.0f;
            }
            if (b.enable) {   // a filter that is not finite is dropped, not run
                for (unsigned c = 0; c < 3; c++)
                    if (!isfinite(b.numerator[c])) b.enable = 0;
                for (unsigned c = 0; c < 2; c++)
                    if (!isfinite(b.denominator[c])) b.enable = 0;
            }
            if (!b.enable || !d->biquads[f].enable || !d->used) memset(d->state[f], 0, sizeof d->state[f]);
            d->biquads[f] = b;
        }
    }
}

// ---- guest memory: a range is read only when an attached pool holds it and the host accepts it for reading
static int in_pool(const AudioRenderer *r, uint64_t addr, uint64_t size) {
    const uint32_t n = (uint32_t)r->cfg.effect_count + 4u * (uint32_t)r->cfg.voice_count;
    for (uint32_t i = 0; i < n; i++) {
        const AudrenPoolIn *p = &r->pools[i];
        if (p->state == POOL_ATTACHED && p->size && addr >= p->address && size <= p->size && addr - p->address <= p->size - size)
            return 1;
    }
    return 0;
}

static const uint8_t *guest_span(AudioRenderer *r, Span *sp, uint64_t addr, uint64_t size, unsigned voice, const char *what,
                                 unsigned slot) {
    if (sp->state && sp->addr == addr && sp->size == size && sp->pool_gen == r->pool_gen) return sp->state == 1 ? sp->ptr : NULL;
    sp->addr = addr, sp->size = size, sp->pool_gen = r->pool_gen;
    const char *why = NULL;
    if (!addr || !size || addr + size < addr) why = "an empty or wrapping range";
    else if (!in_pool(r, addr, size)) why = "not inside an attached memory pool";
    else if (!rt_mem_commit_io(addr, size, 0)) why = "not guest memory the host may read";
    sp->state = why ? 2 : 1;
    sp->ptr = why ? NULL : (const uint8_t *)(uintptr_t)addr;
    if (why && r->mixer->refused < 8) {
        r->mixer->refused++;
        rt_log("audio: %s of voice %u (slot %u) at %#llx, %llu bytes, is not read: %s", what, voice, slot,
               (unsigned long long)addr, (unsigned long long)size, why);
    }
    return sp->ptr;
}

// ---- voice decoding along the voice's own sample path
typedef struct {
    AudioRenderer *r;
    const ServerVoice *v;
    MixVoice *mv;
    const uint8_t *data[4];     // the slots' sample bytes, NULL when a slot is not readable
    const int16_t *coef;        // ADPCM coefficients, NULL when not readable
    unsigned id;
    unsigned index;
    uint64_t offset;
    int32_t loops;
    uint8_t valid[4];
    int ended;
    int16_t h1, h2;
    int32_t abuf;
    uint64_t aat;
} Walk;

static int fmt_bytes(int f) {
    return f == FMT_PCM8 ? 1 : f == FMT_PCM16 ? 2 : f == FMT_PCM24 ? 3 : f == FMT_PCM32 || f == FMT_PCMF32 ? 4 : 0;
}

// DSP-ADPCM: 8-byte frames of a predictor and scale byte and 14 4-bit samples; the history runs across samples.
static int16_t adpcm_next(Walk *w, const uint8_t *d, uint64_t size, uint64_t i) {
    uint64_t frame = i / 14, nib = i % 14;
    if ((frame + 1) * 8 > size) return 0;
    uint8_t ps = d[frame * 8];
    int scale = 1 << (ps & 15), pred = (ps >> 4) & 7;
    int n = (d[frame * 8 + 1 + nib / 2] >> ((nib & 1) ? 0 : 4)) & 15;
    if (n >= 8) n -= 16;
    int64_t s = (int64_t)n * scale * 2048 + 1024 + (int64_t)w->coef[pred * 2] * w->h1 + (int64_t)w->coef[pred * 2 + 1] * w->h2;
    s >>= 11;
    if (s > 32767) s = 32767;
    if (s < -32768) s = -32768;
    w->h2 = w->h1;
    w->h1 = (int16_t)s;
    return (int16_t)s;
}

static float sample_at(Walk *w, int ch) {
    if (w->ended || !w->valid[w->index]) return 0;
    const ServerVoice *v = w->v;
    const AudrenWaveBuffer *b = &v->buffers[w->index];
    const uint8_t *d = w->data[w->index];
    if (!d) return 0;
    const uint64_t i = (uint64_t)b->start + w->offset;
    const int f = v->in.sample_format, cc = (int)v->in.channel_count;
    if (f == FMT_ADPCM) {
        if (cc != 1 || !w->coef) return 0;
        if (w->abuf != (int32_t)w->index || w->aat > i + 1) {   // (re)start this buffer: from its context, else from the frame boundary
            w->abuf = (int32_t)w->index;
            const uint8_t *ctx = b->context_address && b->context_size >= 6
                                     ? guest_span(w->r, &w->mv->context[w->index], b->context_address, 6, w->id, "ADPCM context", w->index)
                                     : NULL;
            if (ctx) {   // the context holds the history the buffer starts from
                int16_t c[3];
                memcpy(c, ctx, 6);
                w->h1 = c[1], w->h2 = c[2];
                w->aat = (uint64_t)b->start;
            } else {
                w->aat = (uint64_t)b->start - (uint64_t)b->start % 14;   // from the frame boundary, with the carried history
            }
            if (i < w->aat) w->aat = i;
        }
        while (w->aat <= i) adpcm_next(w, d, b->size, w->aat++);
        return w->h1 / 32768.0f;
    }
    const int bytes = fmt_bytes(f);
    if (!bytes || ch >= cc) return 0;
    const uint64_t at = (i * (uint64_t)cc + (uint64_t)ch) * (uint64_t)bytes;
    if (at + (uint64_t)bytes > b->size) return 0;
    const uint8_t *p = d + at;
    switch (f) {
    case FMT_PCM8: return (int8_t)p[0] / 128.0f;
    case FMT_PCM16: { int16_t s; memcpy(&s, p, 2); return s / 32768.0f; }
    case FMT_PCM24: { uint32_t word = (uint32_t)p[0] | (uint32_t)p[1] << 8 | (uint32_t)p[2] << 16; return ((int32_t)(word << 8) >> 8) / 8388608.0f; }
    case FMT_PCM32: { int32_t s; memcpy(&s, p, 4); return (float)(s / 2147483648.0); }
    default: { float s; memcpy(&s, p, 4); return isfinite(s) ? s : 0.0f; }
    }
}

// The same transitions as skip_empty and advance_voice of audren.c: a buffer with no samples is played on arrival.
static void walk_skip_empty(Walk *w) {
    while (!w->ended && w->valid[w->index] && buffer_empty(&w->v->buffers[w->index])) {
        const AudrenWaveBuffer *b = &w->v->buffers[w->index];
        if (b->loop && b->loop_count < 0) break;
        w->valid[w->index] = 0;
        w->offset = 0;
        w->loops = 0;
        w->index = (w->index + 1) & 3;
        if (b->end_of_stream) w->ended = 1;
    }
}

static void walk_step(Walk *w) {
    if (w->ended || !w->valid[w->index]) {
        w->ended = 1;
        return;
    }
    const AudrenWaveBuffer *b = &w->v->buffers[w->index];
    uint64_t length = (uint64_t)((int64_t)b->end - b->start);
    if (++w->offset < length) return;
    if (b->loop && (b->loop_count < 0 || w->loops < b->loop_count)) {
        w->loops++;
        w->offset = b->loop_start > b->start && b->loop_start < b->end ? (uint64_t)(b->loop_start - b->start) : 0;
        return;
    }
    w->valid[w->index] = 0;
    w->offset = 0;
    w->loops = 0;
    w->index = (w->index + 1) & 3;
    if (b->end_of_stream) w->ended = 1;
    walk_skip_empty(w);
}

// Decodes frame samples per channel into m->scratch[c][k]. The renderer advances the voice by the same amount afterwards:
// output sample k stands at (acc + owed x k / frame) / engine_rate input samples from the frame's start.
static void voice_render(AudioRenderer *r, Mixer *m, ServerVoice *v, unsigned id) {
    MixVoice *mv = &m->voice[id];
    const uint64_t engine = (uint64_t)r->cfg.sample_rate, owed = voice_owed(r, v), frames = (uint64_t)m->frame;
    Walk w = {.r = r, .v = v, .mv = mv, .id = id, .index = v->index, .offset = v->offset, .loops = v->loops_done,
              .h1 = mv->h1, .h2 = mv->h2, .abuf = mv->abuf, .aat = mv->aat};
    memcpy(w.valid, v->valid, sizeof w.valid);
    for (unsigned i = 0; i < 4; i++)
        if (v->valid[i] && !buffer_empty(&v->buffers[i]))
            w.data[i] = guest_span(r, &mv->span[i], v->buffers[i].address, v->buffers[i].size, id, "wave buffer", i);
    if (v->in.sample_format == FMT_ADPCM && v->in.extra_address && v->in.extra_size >= 32)
        w.coef = (const int16_t *)guest_span(r, &mv->coef, v->in.extra_address, 32, id, "ADPCM coefficients", 0);
    walk_skip_empty(&w);
    const int cc = (int)v->in.channel_count;
    uint64_t cur = 0;
    for (uint64_t k = 0; k < frames; k++) {
        const uint64_t num = v->acc + owed * k / frames, idx = num / engine;
        const float frac = (float)((double)(num % engine) / (double)engine);
        while (cur < idx) {
            walk_step(&w);
            cur++;
        }
        Walk next = w;
        walk_step(&next);
        for (int c = 0; c < cc; c++) {
            float s0 = sample_at(&w, c), s1 = sample_at(&next, c);
            m->scratch[c][k] = s0 + (s1 - s0) * frac;
        }
    }
    for (int c = 0; c < cc; c++)
        for (uint64_t k = 0; k < frames; k++) {
            float a = fabsf(m->scratch[c][k]);
            if (a > m->stats.input_peak) m->stats.input_peak = a;
        }
    m->stats.voice_frames++;
    mv->h1 = w.h1;
    mv->h2 = w.h2;
    mv->abuf = w.abuf;
    mv->aat = w.aat;
}

// ---- mixing
static float *mixbuf(Mixer *m, int i) { return m->buf + (size_t)i * (size_t)m->frame; }
static void accumulate(Mixer *m, int buffer, const float *src, float vol) {
    if (vol == 0 || buffer < 0 || buffer >= m->nbuf) return;
    float *d = mixbuf(m, buffer);
    for (int k = 0; k < m->frame; k++) d[k] += src[k] * vol;
}
static int mix_valid(const AudioRenderer *r, int32_t id) { return id >= 0 && id <= r->cfg.submix_count && r->mixes[id].in_use; }

static const float *splitter_filter(SplitDest *d, const float *input, float *output, int frames) {
    if (!d->biquads[0].enable && !d->biquads[1].enable) return input;
    memcpy(output, input, (size_t)frames * sizeof *output);
    for (unsigned f = 0; f < 2; f++)
        if (d->biquads[f].enable) {
            const AudrenBiquadFloat *b = &d->biquads[f];
            float *z = d->state[f];
            for (int k = 0; k < frames; k++) {
                float x = output[k], y = b->numerator[0] * x + z[0];
                z[0] = b->numerator[1] * x + b->denominator[0] * y + z[1];
                z[1] = b->numerator[2] * x + b->denominator[1] * y;
                output[k] = y;
            }
        }
    return output;
}

static void mix_order_edge(const AudioRenderer *r, int source, int destination, uint32_t (*edges)[8], uint16_t incoming[256]) {
    if (destination <= 0 || !mix_valid(r, destination)) return;   // the final mix is processed separately, last
    uint32_t bit = 1u << (destination & 31);
    if (!(edges[source][destination >> 5] & bit)) {
        edges[source][destination >> 5] |= bit;
        incoming[destination]++;
    }
}

// The sub mixes in an order where every mix comes after the mixes that feed it, through a mix or a splitter's destinations.
// Returns how many, or -1 when the routes make a cycle.
static int mix_order(const AudioRenderer *r, Mixer *m, int order[256]) {
    const int count = r->cfg.submix_count + 1;
    if (count < 1 || count > 256) return -1;
    uint32_t (*edges)[8] = m->edges;
    uint16_t incoming[256] = {0};
    memset(m->edges, 0, sizeof m->edges);
    int active = 0, n = 0;
    for (int i = 1; i < count; i++)
        if (r->mixes[i].in_use) {
            active++;
            const AudrenMixIn *x = &r->mixes[i];
            if (x->dest_splitter_id != SPLITTER_UNUSED && x->dest_splitter_id < (uint32_t)m->nsplit) {
                const SplitInfo *s = &m->split[x->dest_splitter_id];
                for (int k = 0; k < s->ndest; k++) {
                    if (s->ids[k] < 0) continue;
                    const SplitDest *d = &m->dest[s->ids[k]];
                    if (d->used) mix_order_edge(r, i, d->dest, edges, incoming);
                }
            } else {
                mix_order_edge(r, i, x->dest_mix_id, edges, incoming);
            }
        }
    for (int i = 1; i < count; i++)
        if (r->mixes[i].in_use && !incoming[i]) order[n++] = i;
    for (int at = 0; at < n; at++) {
        int source = order[at];
        for (int destination = 1; destination < count; destination++)
            if ((edges[source][destination >> 5] & (1u << (destination & 31))) && !--incoming[destination]) order[n++] = destination;
    }
    return n == active ? n : -1;
}

// ---- effects: passed through dry (see the top of the file)
typedef struct { uint8_t input[24], output[24]; float volume[24]; uint32_t count; } EffectBufferMix;
_Static_assert(sizeof(EffectBufferMix) == 148, "buffer mix effect parameter");
enum { EFFECT_BUFFER_MIX = 1, EFFECT_DELAY = 3, EFFECT_REVERB = 4, EFFECT_BIQUAD = 6, EFFECT_COMPRESSOR = 9 };

static void effect_copy(Mixer *m, int mix, const AudioRenderer *r, const uint8_t *in, const uint8_t *out, unsigned channels) {
    if (channels > 6) channels = 6;
    for (unsigned c = 0; c < channels; c++) {
        if (in[c] == out[c] || in[c] >= r->mixes[mix].buffer_count || out[c] >= r->mixes[mix].buffer_count) continue;
        memcpy(mixbuf(m, m->offset[mix] + out[c]), mixbuf(m, m->offset[mix] + in[c]), (size_t)m->frame * sizeof(float));
    }
}

static void mix_effects(AudioRenderer *r, Mixer *m, int mix) {
    int order[64], count = 0;
    for (int i = 0; i < r->cfg.effect_count; i++) {
        if (!r->effects[i].type || r->effects[i].mix_id != mix || count == 64) continue;
        int j = count++;
        while (j > 0 && r->effects[order[j - 1]].order > r->effects[i].order) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = i;
    }
    for (int n = 0; n < count; n++) {
        const AudrenEffectIn *e = &r->effects[order[n]];
        if (e->type == EFFECT_BUFFER_MIX) {
            EffectBufferMix p;
            memcpy(&p, e->specific, sizeof p);
            if (!e->enabled || p.count > 24) continue;
            for (unsigned c = 0; c < p.count; c++) {
                if (p.input[c] >= r->mixes[mix].buffer_count || p.output[c] >= r->mixes[mix].buffer_count) continue;
                const float *src = mixbuf(m, m->offset[mix] + p.input[c]);
                float *dst = mixbuf(m, m->offset[mix] + p.output[c]);
                for (int k = 0; k < m->frame; k++) dst[k] += ceilf(src[k] * 32768 * p.volume[c]) / 32768;
            }
            continue;
        }
        if (e->enabled && e->type < 32 && !(m->effect_reported & (1u << e->type))) {
            m->effect_reported |= 1u << e->type;
            rt_log("audio: effect type %u is enabled and is passed through", e->type);
        }
        if (e->type == EFFECT_DELAY || e->type == EFFECT_REVERB || e->type == EFFECT_COMPRESSOR) {
            uint16_t channels;
            memcpy(&channels, e->specific + 14, 2);
            effect_copy(m, mix, r, e->specific, e->specific + 6, channels);
        } else if (e->type == EFFECT_BIQUAD) {
            effect_copy(m, mix, r, e->specific, e->specific + 6, e->specific[r->revision >= 15 ? 36 : 22]);
        }
    }
}

// ---- the frame
static void voice_filters(Mixer *m, const ServerVoice *v, unsigned id) {
    const int cc = (int)v->in.channel_count;
    MixVoice *mv = &m->voice[id];
    for (int f = 0; f < 2; f++) {
        if (!v->in.biquads[f].enable) continue;
        const AudrenBiquad *b = &v->in.biquads[f];
        const AudrenBiquadFloat *bf = &v->float_biquads[f];
        const float a0 = v->use_float_biquads ? bf->numerator[0] : b->numerator[0] / 16384.0f;
        const float a1 = v->use_float_biquads ? bf->numerator[1] : b->numerator[1] / 16384.0f;
        const float a2 = v->use_float_biquads ? bf->numerator[2] : b->numerator[2] / 16384.0f;
        const float b1 = v->use_float_biquads ? bf->denominator[0] : b->denominator[0] / 16384.0f;
        const float b2 = v->use_float_biquads ? bf->denominator[1] : b->denominator[1] / 16384.0f;
        for (int c = 0; c < cc; c++) {
            float *z = mv->bq[f][c], *x = m->scratch[c];
            for (int k = 0; k < m->frame; k++) {
                float in = x[k], y = a0 * in + z[0];
                z[0] = a1 * in + b1 * y + z[1];
                z[1] = a2 * in + b2 * y;
                x[k] = y;
            }
        }
    }
}

// The first few started voices, with where they go: the mixes in use, the splitter they feed, and the sinks. One look at what
// the client built, in the log, when a run is silent or loud.
static void log_routing(const AudioRenderer *r, Mixer *m, const ServerVoice *v, int id) {
    m->routed++;
    rt_log("audio: voice %d is mixed: format %u, %u channel(s), %u Hz, pitch %.3f, volume %.3f, mix %d, splitter %#x, channel ids "
           "%d %d, first gains %.3f %.3f", id, v->in.sample_format, v->in.channel_count, v->in.sample_rate, v->in.pitch,
           v->in.volume, v->in.mix_id, v->in.splitter_id, v->in.channel_ids[0], v->in.channel_ids[1],
           v->in.channel_ids[0] >= 0 && v->in.channel_ids[0] < r->cfg.voice_count * 6 ? r->channels[v->in.channel_ids[0]].mix[0] : -1.0f,
           v->in.channel_ids[0] >= 0 && v->in.channel_ids[0] < r->cfg.voice_count * 6 ? r->channels[v->in.channel_ids[0]].mix[1] : -1.0f);
    if (m->routed > 1) return;
    for (int i = 0; i < m->mixes; i++) {
        const AudrenMixIn *x = &r->mixes[i];
        if (x->in_use)
            rt_log("audio: mix %d: %d buffers, volume %.3f, destination mix %d, splitter %#x", i, x->buffer_count, x->volume,
                   x->dest_mix_id, x->dest_splitter_id);
    }
    for (int i = 0; i < r->cfg.sink_count; i++) {
        const AudrenSinkIn *s = &r->sinks[i];
        rt_log("audio: sink %d: type %u, in use %u, %u inputs: %d %d %d %d %d %d", i, s->type, s->in_use, s->input_count, s->inputs[0],
               s->inputs[1], s->inputs[2], s->inputs[3], s->inputs[4], s->inputs[5]);
    }
    if (v->in.splitter_id != SPLITTER_UNUSED && v->in.splitter_id < (uint32_t)m->nsplit) {
        const SplitInfo *s = &m->split[v->in.splitter_id];
        for (int k = 0; k < s->ndest && k < 6; k++) {
            if (s->ids[k] < 0) continue;
            const SplitDest *d = &m->dest[s->ids[k]];
            rt_log("audio: splitter %u destination %d (id %d): mix %d, used %u, volumes %.3f %.3f %.3f %.3f", v->in.splitter_id, k,
                   s->ids[k], d->dest, d->used, d->vol[0], d->vol[1], d->vol[2], d->vol[3]);
        }
    }
}

int mixer_render(AudioRenderer *r, float *lr) {
    Mixer *m = r->mixer;
    const int frames = m->frame, mc = m->mixes;
    memset(m->buf, 0, (size_t)m->nbuf * (size_t)frames * sizeof(float));
    int at = 0;
    for (int i = 0; i < mc; i++) {
        m->offset[i] = at;
        if (r->mixes[i].in_use) at += r->mixes[i].buffer_count;
    }
    if (at > m->nbuf) {
        report_once(m, 0, "the mixes use more buffers than the renderer has", at);
        memset(lr, 0, sizeof(float) * 2 * (size_t)frames);
        return 0;
    }

    for (int i = 0; i < r->cfg.voice_count; i++) {
        ServerVoice *v = &r->voices[i];
        if (!v->in.in_use || v->state != SV_STARTED || !v->in.sample_rate || !v->valid[v->index]) continue;
        const int cc = (int)v->in.channel_count;
        if (cc < 1 || cc > 6) {
            report_once(m, 2, "voice channel count unsupported", cc);
            continue;
        }
        if (m->routed < 6) log_routing(r, m, v, i);
        voice_render(r, m, v, (unsigned)i);
        voice_filters(m, v, (unsigned)i);
        if (v->in.splitter_id != SPLITTER_UNUSED && v->in.splitter_id < (uint32_t)m->nsplit) {
            const SplitInfo *s = &m->split[v->in.splitter_id];
            for (int c = 0; c < cc; c++)
                for (int k = c; k < s->ndest; k += cc) {
                    if (s->ids[k] < 0) continue;
                    SplitDest *d = &m->dest[s->ids[k]];
                    if (!d->used || !mix_valid(r, d->dest)) continue;
                    const float *source = splitter_filter(d, m->scratch[c], m->filtered, frames);
                    for (int b = 0; b < r->mixes[d->dest].buffer_count && b < 24; b++)
                        accumulate(m, m->offset[d->dest] + b, source, d->vol[b] * v->in.volume);
                }
        } else if (mix_valid(r, v->in.mix_id)) {
            for (int c = 0; c < cc; c++) {
                int32_t id = v->in.channel_ids[c];
                if (id < 0 || id >= r->cfg.voice_count * 6) continue;
                const AudrenChannelIn *ch = &r->channels[id];
                for (int b = 0; b < r->mixes[v->in.mix_id].buffer_count && b < 24; b++)
                    accumulate(m, m->offset[v->in.mix_id] + b, m->scratch[c], ch->mix[b] * v->in.volume);
            }
        }
    }

    // every source precedes its destination sub mix
    int order[256], n = mix_order(r, m, order);
    if (n < 0) {
        report_once(m, 3, "the sub mixes route in a cycle; they are not mixed", 0);
        n = 0;
    }
    for (int a = 0; a < n; a++) {
        mix_effects(r, m, order[a]);
        const AudrenMixIn *x = &r->mixes[order[a]];
        const int src = m->offset[order[a]];
        if (x->dest_splitter_id != SPLITTER_UNUSED && x->dest_splitter_id < (uint32_t)m->nsplit) {
            const SplitInfo *s = &m->split[x->dest_splitter_id];
            for (int c = 0; c < x->buffer_count && c < 24; c++)
                for (int k = c; k < s->ndest; k += x->buffer_count) {
                    if (s->ids[k] < 0) continue;
                    SplitDest *d = &m->dest[s->ids[k]];
                    if (!d->used || !mix_valid(r, d->dest)) continue;
                    const float *source = splitter_filter(d, mixbuf(m, src + c), m->filtered, frames);
                    for (int b = 0; b < r->mixes[d->dest].buffer_count && b < 24; b++)
                        accumulate(m, m->offset[d->dest] + b, source, d->vol[b] * x->volume);
                }
        } else if (mix_valid(r, x->dest_mix_id)) {
            for (int c = 0; c < x->buffer_count && c < 24; c++)
                for (int b = 0; b < r->mixes[x->dest_mix_id].buffer_count && b < 24; b++)
                    accumulate(m, m->offset[x->dest_mix_id] + b, mixbuf(m, src + c), x->mix[c][b] * x->volume);
        }
    }
    if (r->mixes[0].in_use) mix_effects(r, m, 0);

    // the device sink: the first one in use takes the final mix; a circular sink is accepted and not written
    const AudrenSinkIn *sink = NULL;
    for (int i = 0; i < r->cfg.sink_count; i++) {
        if (!r->sinks[i].in_use) continue;
        if (r->sinks[i].type == SINK_DEVICE && !sink) sink = &r->sinks[i];
        else if (r->sinks[i].type == SINK_CIRCULAR && !(m->effect_reported & (1u << 31))) {
            m->effect_reported |= 1u << 31;
            rt_log("audio: a circular sink is in use and is passed through (nothing is written to its buffer)");
        }
    }
    memset(lr, 0, sizeof(float) * 2 * (size_t)frames);
    if (!sink || sink->input_count < 1 || sink->input_count > 6) {
        if (!sink && r->updates) report_once(m, 5, "no device sink in use", r->cfg.sink_count);
        return 0;
    }
    const float gain = audio_device_volume() * (r->mixes[0].in_use ? r->mixes[0].volume : 1.0f);
    const float *ch[6] = {0};
    for (uint32_t c = 0; c < sink->input_count; c++)
        if (sink->inputs[c] >= 0 && sink->inputs[c] < m->nbuf) ch[c] = mixbuf(m, sink->inputs[c]);
    for (int k = 0; k < frames; k++) {
        float l, rr;
        if (sink->input_count >= 6) {   // front left, front right, centre, low frequency, rear left, rear right
            float ff = sink->downmix_enabled ? sink->downmix[0] : 1.0f, cf = sink->downmix_enabled ? sink->downmix[1] : 0.7071f,
                  sf = sink->downmix_enabled ? sink->downmix[2] : 0.7071f;
            float cen = ch[2] ? ch[2][k] : 0;
            l = ff * (ch[0] ? ch[0][k] : 0) + cf * cen + sf * (ch[4] ? ch[4][k] : 0);
            rr = ff * (ch[1] ? ch[1][k] : 0) + cf * cen + sf * (ch[5] ? ch[5][k] : 0);
        } else if (sink->input_count >= 2) {
            l = ch[0] ? ch[0][k] : 0;
            rr = ch[1] ? ch[1][k] : 0;
        } else {
            l = rr = ch[0] ? ch[0][k] : 0;
        }
        l *= gain;
        rr *= gain;
        if (!(l == l)) l = 0;
        if (!(rr == rr)) rr = 0;
        lr[2 * k] = l > 1 ? 1 : l < -1 ? -1 : l;
        lr[2 * k + 1] = rr > 1 ? 1 : rr < -1 ? -1 : rr;
        if (fabsf(lr[2 * k]) > m->stats.output_peak) m->stats.output_peak = fabsf(lr[2 * k]);
        if (fabsf(lr[2 * k + 1]) > m->stats.output_peak) m->stats.output_peak = fabsf(lr[2 * k + 1]);
    }
    return 1;
}
