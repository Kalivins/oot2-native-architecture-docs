// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The hardware Opus decoder at the nn::codec level (architecture D9, D24a). The SDK's own hardware decoder client would
// need the hwopus service, a transfer memory and a sub-session per decoder, and the references override at this level as
// well, so these seven entry points are overrides. The decoder itself is behind the seam of runtime/audio/opus.h: libopus
// in a build that has it, a fake in a test. A build without one still answers every call, consumes the input of a decode and
// makes no sample, and says so once in the log.
//
// The contract (what the SDK's client and the references agree on): the input of a decode is Nintendo's frame, an 8-byte
// big-endian header {payload size, final range} and the Opus packet; `consumed` is 8 plus the payload size (a call may be
// given several frames in a row), `samples` is the count per channel; a final range that differs from the header's is logged
// once and the samples are kept (a header value of 0 is not a recorded value, so it is not checked); a refusal returns the
// codec result 0x1fcd1; the work buffer the guest is told to reserve is 0x9000 bytes, of which the host uses none.
//
// Adapted from oot-pc-recomp runtime/ov_media.c at f4656d3b31a1348fe04ab2b989f15c6047e681ae (MIT; the file's header:
// adapted from MacNativeGames games/totk/runtime/opus_codec.c at b75be8cf, MIT, Copyright (c) 2026 MacNativeGames
// contributors; the notices file has the row and licenses/macnativegames-MIT.txt the text). Changed here: the decoder is
// behind the seam, so the override's rules are tested without libopus and a build without it degrades instead of stopping;
// the packet and the pcm are copied through host buffers with hos_in and hos_out; the names, logging and locking are this
// project's.
#include <limits.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "../audio/opus.h"
#include "hos.h"
#include "plat.h"

#define CODEC_R_INVALID 0x1fcd1u   // the nn::codec "invalid argument" result
#define OPUS_WORK_SIZE 0x9000u     // what the game is told to reserve
#define MAX_FRAMES 5760            // 120 ms at 48 kHz: the most one Opus packet decodes to

// One record per initialised decoder object (the guest's HardwareOpusDecoderType). The registry lock guards the list; a
// decoder's own lock serialises its state, so different streams decode in parallel. Lock order: registry, then decoder (a
// caller takes the decoder's lock before it drops the registry's, so Finalize cannot free a decoder another thread uses).
typedef struct Decoder {
    uint64_t guest;
    void *dec;   // the seam's decoder, NULL in a build without one
    int rate, channels;
    PlatMutex use;
    struct Decoder *next;
} Decoder;
static Decoder *g_decoders;
static PlatMutex g_lock = PLAT_MUTEX_INIT;

static int valid_format(int rate, int channels) {
    return (rate == 48000 || rate == 24000 || rate == 16000 || rate == 12000 || rate == 8000) && (channels == 1 || channels == 2);
}

// The decoder of a guest object, with its lock held.
static Decoder *acquire(uint64_t guest, const char *fn) {
    plat_mutex_lock(&g_lock);
    Decoder *d = g_decoders;
    while (d && d->guest != guest) d = d->next;
    if (d) plat_mutex_lock(&d->use);
    plat_mutex_unlock(&g_lock);
    if (!d) rt_fatal("nn::codec::%s: unknown hardware Opus decoder %#llx", fn, (unsigned long long)guest);
    return d;
}
static void release(Decoder *d) { plat_mutex_unlock(&d->use); }

static void tell_without_decoder(void) {
    static int32_t told;
    if (!__atomic_exchange_n(&told, 1, __ATOMIC_RELAXED))
        rt_log("audio: hardware Opus: this build has no decoder (tools/fetch_opus.py, OPUS_DIR); a decode consumes its input and makes no sample");
}

// size_t GetHardwareOpusDecoderWorkBufferSize(int rate, int channels, int option)
static void ov_opus_work(Cpu *c) { ret_x(c, valid_format((int)arg_x(c, 0), (int)arg_x(c, 1)) ? OPUS_WORK_SIZE : 0); }

// Result InitializeHardwareOpusDecoder(HardwareOpusDecoderType *d, int rate, int channels, void *work, size_t size, int option).
// The state lives on the host; the guest's work buffer is reserved and unused. Initialising the same object again without
// Finalize replaces its decoder.
static void ov_opus_init(Cpu *c) {
    uint64_t guest = arg_x(c, 0), size = arg_x(c, 4);
    int rate = (int)arg_x(c, 1), channels = (int)arg_x(c, 2);
    static int32_t told;
    if (told < 12) {
        told++;
        rt_log("audio: hardware Opus decoder %#llx: %d Hz, %d ch, work buffer %llu bytes, option %d", (unsigned long long)guest, rate,
               channels, (unsigned long long)size, (int)arg_x(c, 5));
    }
    if (!guest || !valid_format(rate, channels) || size < OPUS_WORK_SIZE) {
        rt_log("audio: hardware Opus: that initialize is refused (object %#llx, %d Hz, %d ch, work buffer %llu bytes)",
               (unsigned long long)guest, rate, channels, (unsigned long long)size);
        ret_x(c, CODEC_R_INVALID);
        return;
    }
    const OpusApi *api = audio_opus();
    void *fresh = NULL;
    if (api) {
        char why[160] = "";
        fresh = api->create(rate, channels, why, sizeof why);
        if (!fresh) rt_fatal("audio: the Opus decoder for %d Hz, %d channels cannot be made: %s", rate, channels, why);
    } else {
        tell_without_decoder();
    }
    plat_mutex_lock(&g_lock);
    Decoder *d = g_decoders;
    while (d && d->guest != guest) d = d->next;
    if (d) {
        plat_mutex_lock(&d->use);   // waits for a decode in flight
        void *old = d->dec;
        d->dec = fresh;
        d->rate = rate;
        d->channels = channels;
        plat_mutex_unlock(&d->use);
        plat_mutex_unlock(&g_lock);
        if (old && api) api->destroy(old);
    } else {
        d = calloc(1, sizeof *d);
        if (!d) rt_fatal("audio: Opus decoder allocation");
        d->use = (PlatMutex)PLAT_MUTEX_INIT;
        d->guest = guest;
        d->dec = fresh;
        d->rate = rate;
        d->channels = channels;
        d->next = g_decoders;
        g_decoders = d;
        plat_mutex_unlock(&g_lock);
    }
    ret_x(c, HOS_R_OK);
}

// void FinalizeHardwareOpusDecoder(HardwareOpusDecoderType *d)
static void ov_opus_finalize(Cpu *c) {
    uint64_t guest = arg_x(c, 0);
    plat_mutex_lock(&g_lock);
    Decoder **p = &g_decoders;
    while (*p && (*p)->guest != guest) p = &(*p)->next;
    if (!*p) {
        plat_mutex_unlock(&g_lock);
        rt_fatal("nn::codec::FinalizeHardwareOpusDecoder: unknown decoder %#llx", (unsigned long long)guest);
    }
    Decoder *d = *p;
    plat_mutex_lock(&d->use);   // a decode in flight finishes first; the registry lock keeps new ones out
    *p = d->next;
    plat_mutex_unlock(&g_lock);
    const OpusApi *api = audio_opus();
    if (d->dec && api) api->destroy(d->dec);
    plat_mutex_unlock(&d->use);
    free(d);
    ret_x(c, HOS_R_OK);
}

// int GetHardwareOpusDecoderSampleRate(const HardwareOpusDecoderType *), int GetHardwareOpusDecoderChannelCount(...)
static void ov_opus_get(Cpu *c, int channels) {
    Decoder *d = acquire(arg_x(c, 0), channels ? "GetHardwareOpusDecoderChannelCount" : "GetHardwareOpusDecoderSampleRate");
    int v = channels ? d->channels : d->rate;
    release(d);
    ret_x(c, (uint64_t)(uint32_t)v);
}
static void ov_opus_rate(Cpu *c) { ov_opus_get(c, 0); }
static void ov_opus_channels(Cpu *c) { ov_opus_get(c, 1); }

// Result ResetHardwareOpusDecoder(HardwareOpusDecoderType *d)
static void ov_opus_reset(Cpu *c) {
    Decoder *d = acquire(arg_x(c, 0), "ResetHardwareOpusDecoder");
    const OpusApi *api = audio_opus();
    int bad = d->dec && api && api->reset(d->dec) != 0;
    release(d);
    ret_x(c, bad ? CODEC_R_INVALID : HOS_R_OK);
}

static uint32_t be32(const uint8_t *p) { return (uint32_t)p[0] << 24 | (uint32_t)p[1] << 16 | (uint32_t)p[2] << 8 | p[3]; }

// Result DecodeOpusInterleavedWithHardware(HardwareOpusDecoderType *d, size_t *consumed, int *samples, int16_t *out,
// size_t out_size, const void *in, size_t in_size)
static void ov_opus_decode(Cpu *c) {
    static int32_t warned, told_bad;
    uint64_t guest = arg_x(c, 0), consumed_at = arg_x(c, 1), samples_at = arg_x(c, 2), out_at = arg_x(c, 3), out_size = arg_x(c, 4),
             in_at = arg_x(c, 5), in_size = arg_x(c, 6);
    if (!consumed_at || !samples_at || !out_at || !in_at || in_size < 8 || in_size > INT_MAX) {
        if (__atomic_fetch_add(&told_bad, 1, __ATOMIC_RELAXED) < 10)
            rt_log("audio: hardware Opus: a decode is refused: a null argument, or %llu bytes in", (unsigned long long)in_size);
        ret_x(c, CODEC_R_INVALID);
        return;
    }
    uint8_t head[8];
    if (!hos_in(head, in_at, sizeof head)) {
        ret_x(c, CODEC_R_INVALID);
        return;
    }
    uint32_t n = be32(head), range = be32(head + 4);
    if (n > in_size - 8 || n > INT_MAX) {
        if (__atomic_fetch_add(&told_bad, 1, __ATOMIC_RELAXED) < 10)
            rt_log("audio: hardware Opus: a decode is refused: the packet is %u bytes, the input %llu", n, (unsigned long long)in_size);
        ret_x(c, CODEC_R_INVALID);
        return;
    }
    Decoder *d = acquire(guest, "DecodeOpusInterleavedWithHardware");
    const OpusApi *api = audio_opus();
    const int channels = d->channels;
    int32_t got = 0;
    uint32_t final = 0;
    int16_t *pcm = NULL;
    if (d->dec && api) {
        size_t capacity = out_size / (sizeof(int16_t) * (size_t)channels);
        if (capacity > MAX_FRAMES) capacity = MAX_FRAMES;
        uint8_t *packet = malloc(n ? n : 1);
        pcm = malloc((capacity ? capacity : 1) * (size_t)channels * sizeof(int16_t));
        if (!packet || !pcm) rt_fatal("audio: Opus decode buffers");
        int ok = n == 0 || hos_in(packet, in_at + 8, n);
        got = ok ? api->decode(d->dec, packet, (int)n, pcm, (int)capacity, &final) : -1;
        free(packet);
        if (got < 0) {
            release(d);
            free(pcm);
            if (__atomic_fetch_add(&told_bad, 1, __ATOMIC_RELAXED) < 10)
                rt_log("audio: hardware Opus: %s refused a packet of %u bytes (%d)", api->name, n, got);
            ret_x(c, CODEC_R_INVALID);
            return;
        }
    } else {
        tell_without_decoder();
    }
    release(d);
    if (pcm) {
        int ok = got == 0 || hos_out(out_at, pcm, (uint64_t)got * (uint64_t)channels * sizeof(int16_t));
        free(pcm);
        if (!ok) {
            ret_x(c, CODEC_R_INVALID);
            return;
        }
        if (n && range && final != range && !__atomic_exchange_n(&warned, 1, __ATOMIC_RELAXED))
            rt_log("audio: hardware Opus: the final range %#x is not the stream's %#x (logged once)", final, range);
    }
    uint64_t consumed = 8 + (uint64_t)n;
    if (!hos_out(consumed_at, &consumed, sizeof consumed) || !hos_out(samples_at, &got, sizeof got)) {
        ret_x(c, CODEC_R_INVALID);
        return;
    }
    ret_x(c, HOS_R_OK);
}

#define WHY_OPUS "decodes with the build's Opus decoder (runtime/audio/opus.h); a build without one makes no sample"
HOS_OVERRIDE("_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii", ov_opus_work, HOS_IMPLEMENTED,
             "0x9000 for a valid format, 0 otherwise, as the reference decoder asks");
HOS_OVERRIDE("_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi", ov_opus_init, HOS_IMPLEMENTED,
             "a host decoder for the object; the guest's work buffer is reserved and unused");
HOS_OVERRIDE("_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE", ov_opus_rate, HOS_IMPLEMENTED,
             "the rate the object was initialised with");
HOS_OVERRIDE("_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE", ov_opus_channels, HOS_IMPLEMENTED,
             "the channel count the object was initialised with");
HOS_OVERRIDE("_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm", ov_opus_decode,
             HOS_IMPLEMENTED, "one frame (an 8-byte header and the packet) per call: " WHY_OPUS);
HOS_OVERRIDE("_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE", ov_opus_reset, HOS_IMPLEMENTED,
             "resets the host decoder's state");
HOS_OVERRIDE("_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE", ov_opus_finalize, HOS_IMPLEMENTED,
             "frees the host decoder");
