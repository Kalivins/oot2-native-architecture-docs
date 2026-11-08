// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The audio services the SDK's own client talks to (architecture D9, D24a): aud:u, the user client session of the
// console's newer SDKs, whose command 2 hands out the renderer manager; audren:u, the manager itself (open a renderer,
// ask the work buffer size, get the device service); each renderer as an IAudioRenderer session; and IAudioDevice with
// one output device. The renderer is runtime/audio/audren.c; this file is its wire: commands, buffers, handles and the
// sessions' lifetimes. The client's code (nn::audio) runs unchanged on top of it, so a command no reference documents
// is the named fatal of the IPC layer, with its number, and the census row that names it.
//
// Adapted from DKB Recomp runtime/audren.c at c5d5e7c (the service parts: aud_user_handler, audren_manager_handler,
// renderer_handler; MIT, Copyright (c) 2026 MacNativeGames contributors, as audren.c's header says) and runtime/
// audio_mixer.c at c5d5e7c (audio_device_handler; the same MIT chain), whose IAudioDevice commands 15 to 18 oot-pc-recomp
// added for this SDK family. Changed here: handlers get the calling thread's Cpu, so a named fatal names the SDK
// function; sessions carry their state and a close that ends the renderer; the work buffer answer is one constant the
// open checks the transfer memory against; the frame event is made here (its readable end goes to the client once, its
// writable end belongs to the renderer); the open refuses what the renderer does not run (manual or CPU rendering, an
// output rate the renderer lacks); the update is read into host memory and the answer written back whole.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "hos.h"
#include "ipc.h"
#include "oot2/audio.h"
#include "plat.h"

#define WORK_BUFFER_BYTES 0x10000u   // the host keeps the renderer's state in host memory; the client only reserves this much
#define DEVICE_NAME "MainAudioOut"

typedef struct RendererState {
    AudioRenderer *r;
    uint32_t readable;   // the client's end of the frame event, until QuerySystemEvent hands it over
    int event_taken;
    float voice_drop;
    uint32_t time_limit;
    int logged_update;
} RendererState;

static void tag_text(uint32_t tag, char out[5]) {
    for (int i = 0; i < 4; i++) {
        char ch = (char)(tag >> (8 * i));
        out[i] = ch >= 0x20 && ch < 0x7f ? ch : '?';
    }
    out[4] = 0;
}

// The buffer the client gave for something it reads in (A, else X) or for something the answer fills (B, else C).
static IpcBuffer pick(const IpcBuffer *a, unsigned na, const IpcBuffer *b, unsigned nb, unsigned i) {
    IpcBuffer x = i < na ? a[i] : (IpcBuffer){0}, y = i < nb ? b[i] : (IpcBuffer){0};
    return x.size ? x : y;
}

// ---- IAudioRenderer
static void renderer_closed(void *p) {
    RendererState *st = p;
    audio_renderer_close(st->r);
    if (!st->event_taken) rt_kernel_event_close(st->readable);   // never handed over: nobody else holds it
    free(st);
}

static void request_update(Cpu *c, IpcSession *s, const IpcRequest *rq, IpcResponse *rs) {
    RendererState *st = s->state;
    IpcBuffer in = pick(rq->a, rq->na, rq->x, rq->nx, 0);
    IpcBuffer out = pick(rq->b, rq->nb, rq->c, rq->nc, 0), perf = pick(rq->b, rq->nb, rq->c, rq->nc, 1);
    int out_pointer = !(rq->nb && rq->b[0].size) && rq->nc && rq->c[0].size;
    if (!in.size || !out.size)
        hos_gap(c, "audio", "RequestUpdate command %u: buffers of %llu bytes in and %llu out (a%u b%u x%u c%u)", rq->cmd,
                (unsigned long long)in.size, (unsigned long long)out.size, rq->na, rq->nb, rq->nx, rq->nc);
    if (!st->logged_update) {
        st->logged_update = 1;
        rt_log("audio: RequestUpdate command %u: input %llu bytes (%s), output %llu bytes (%s), performance %llu bytes",
               rq->cmd, (unsigned long long)in.size, rq->na && rq->a[0].size ? "map-alias" : "pointer",
               (unsigned long long)out.size, out_pointer ? "pointer" : "map-alias", (unsigned long long)perf.size);
    }
    uint8_t *buf = malloc((size_t)(in.size + out.size));
    if (!buf) rt_fatal("audio: update buffers of %llu bytes", (unsigned long long)(in.size + out.size));
    uint8_t *input = buf, *output = buf + in.size;
    if (!ipc_read(&in, 0, input, in.size)) {
        free(buf);
        hos_gap(c, "audio", "RequestUpdate: the input buffer %#llx+%llu is not guest memory the guest may read",
                (unsigned long long)in.addr, (unsigned long long)in.size);
    }
    char why[512];
    uint32_t used = 0;
    if (!audio_renderer_update(st->r, input, in.size, output, out.size, &used, why, sizeof why)) {
        free(buf);
        hos_gap(c, "audio", "RequestUpdate: %s", why);
    }
    int ok;
    if (out_pointer) {
        ok = ipc_answer_pointer(rs, rq, 0, output, out.size);
        if (ok && perf.size) {   // the performance buffer's own descriptor, empty: no performance frames are written
            static const uint8_t zeros[0x100];
            uint64_t n = perf.size < sizeof zeros ? perf.size : sizeof zeros;
            ok = ipc_answer_pointer(rs, rq, 1, zeros, n);
        }
    } else {
        ok = ipc_write(&out, 0, output, out.size);
    }
    free(buf);
    if (!ok)
        hos_gap(c, "audio", "RequestUpdate: the output buffer %#llx+%llu is not guest memory the guest may write",
                (unsigned long long)out.addr, (unsigned long long)out.size);
}

static void renderer_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    RendererState *st = s->state;
    const AudioConfig *cfg = audio_renderer_config(st->r);
    switch (rq->cmd) {
    case 0: ipc_out(rs, &cfg->sample_rate, 4); return;         // GetSampleRate
    case 1: ipc_out(rs, &cfg->sample_count, 4); return;        // GetSampleCount
    case 2: ipc_out(rs, &cfg->mix_buffer_count, 4); return;    // GetMixBufferCount
    case 3: {                                                  // GetState: 0 started, 1 stopped
        uint32_t v = audio_renderer_running(st->r) ? 0 : 1;
        ipc_out(rs, &v, 4);
        return;
    }
    case 4: case 10: request_update(c, s, rq, rs); return;     // RequestUpdate, RequestUpdateAuto
    case 5: audio_renderer_start(st->r); return;               // Start
    case 6: audio_renderer_stop(st->r); return;                // Stop
    case 7:                                                    // QuerySystemEvent: the readable end of the frame event, once
        if (st->event_taken) hos_gap(c, "audio", "IAudioRenderer QuerySystemEvent twice: one readable end per event");
        st->event_taken = 1;
        rs->copy[rs->ncopy++] = st->readable;
        return;
    case 8: memcpy(&st->time_limit, ipc_arg(c, s, rq, 0, 4), 4); return;   // SetRenderingTimeLimit
    case 9: {                                                              // GetRenderingTimeLimit
        uint32_t v = st->time_limit ? st->time_limit : 100;
        ipc_out(rs, &v, 4);
        return;
    }
    case 12: memcpy(&st->voice_drop, ipc_arg(c, s, rq, 0, 4), 4); return;  // SetVoiceDropParameter (a float)
    case 13: ipc_out(rs, &st->voice_drop, 4); return;                      // GetVoiceDropParameter
    case 14: return;   // SetAsyncMemoryPoolControlEnabled(bool): memory pools are always synchronous here
    default: ipc_unknown_command(c, s, rq);
    }
}

// ---- IAudioDevice: one stereo output device
static void put_device_name(const IpcBuffer *b, IpcResponse *rs, const IpcRequest *rq, int pointer) {
    if (!b->size) return;
    uint8_t name[0x100] = {0};
    memcpy(name, DEVICE_NAME, sizeof DEVICE_NAME);
    uint64_t n = b->size < sizeof name ? b->size : sizeof name;
    if (pointer) ipc_answer_pointer(rs, rq, 0, name, n);
    else ipc_write(b, 0, name, n);
}

static void device_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    int pointer = !(rq->nb && rq->b[0].size) && rq->nc && rq->c[0].size;
    IpcBuffer b = pointer ? rq->c[0] : rq->nb ? rq->b[0] : (IpcBuffer){0};
    switch (rq->cmd) {
    case 0: case 6: case 14: {   // ListAudioDeviceName(Auto), ListAudioOutputDeviceName
        int32_t n = b.size >= 0x100 ? 1 : 0;
        if (n) put_device_name(&b, rs, rq, pointer);
        ipc_out(rs, &n, 4);
        return;
    }
    case 1: case 7: {   // SetAudioDeviceOutputVolume(Auto)
        float v;
        memcpy(&v, ipc_arg(c, s, rq, 0, 4), 4);
        audio_device_volume_set(v);   // read back as set (one device, one value); a NaN is silence; the mixer applies it
        return;
    }
    case 2: case 8: {   // GetAudioDeviceOutputVolume(Auto)
        float v = audio_device_volume();
        ipc_out(rs, &v, 4);
        return;
    }
    case 3: case 10: case 13:   // GetActiveAudioDeviceName(Auto), GetActiveAudioOutputDeviceName
        put_device_name(&b, rs, rq, pointer);
        return;
    case 5: {   // GetActiveChannelCount
        int32_t n = 2;
        ipc_out(rs, &n, 4);
        return;
    }
    case 19: return;   // SetAudioDeviceOutputVolumeAutoTuneEnabled: the host output has no auto tune
    case 20: case 21: {   // IsAudioDeviceOutputVolumeAutoTuneEnabled, IsActiveOutputDeviceEstimatedLowLatency: false
        uint8_t v = 0;
        ipc_out(rs, &v, 1);
        return;
    }
    case 4: case 11: case 12: case 15: case 17: {
        // QueryAudioDeviceSystemEvent, ...InputEvent, ...OutputEvent, and the notifications of the newer SDKs
        // (AcquireAudioInputDeviceNotification, AcquireAudioOutputDeviceNotification, which carry 16 bytes of arguments the
        // device does not need): one event that never signals, the device never changes. The host keeps no end of it.
        uint32_t readable, writable = rt_kernel_event_create(&readable);
        rt_kernel_event_close(writable);
        rs->copy[rs->ncopy++] = readable;
        return;
    }
    case 16: case 18: return;   // ReleaseAudioInputDeviceNotification, ReleaseAudioOutputDeviceNotification: nothing to release
    default: ipc_unknown_command(c, s, rq);
    }
}

// ---- IAudioRendererManager
static void manager_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 0: {   // OpenAudioRenderer(parameter, work buffer size, applet resource user id; copied: transfer memory, process)
        AudioConfig cfg;
        const char *why = audio_config_read(ipc_arg(c, s, rq, 0, 0x34), 0x34, &cfg);
        if (why) hos_gap(c, "audio", "OpenAudioRenderer: %s", why);
        uint64_t work;
        memcpy(&work, ipc_arg(c, s, rq, 0x38, 8), 8);
        IpcTransferMemory tm = {0};
        int found = 0;
        for (unsigned i = 0; i < rq->ncopy && !found; i++) found = ipc_transfer_memory(rq->copy[i], &tm);
        if (!found)
            hos_gap(c, "audio", "OpenAudioRenderer: no transfer memory handle came with the request, and the work buffer states "
                    "%llu bytes", (unsigned long long)work);
        if (tm.size < work)
            hos_gap(c, "audio", "OpenAudioRenderer: the transfer memory is %llu bytes, smaller than the %llu the work buffer "
                    "states", (unsigned long long)tm.size, (unsigned long long)work);
        if (cfg.rendering_device != 0 || cfg.execution_mode != 0)
            hos_gap(c, "audio", "OpenAudioRenderer: rendering device %d and execution mode %d; the renderer runs on the "
                    "DSP in automatic mode only (anything else means the client renders itself)", cfg.rendering_device,
                    cfg.execution_mode);
        RendererState *st = calloc(1, sizeof *st);
        if (!st) rt_fatal("audio: renderer session allocation");
        char capture[1100];   // the capture sink's file, in the game folder (audio_sink = capture only)
        snprintf(capture, sizeof capture, "%s/audio-capture.wav", hos_game_folder());
        audio_set_capture_path(capture);
        uint32_t writable = rt_kernel_event_create(&st->readable);
        st->r = audio_renderer_open(&cfg, writable, st->readable);
        st->voice_drop = 1.0f;
        char tag[5];
        tag_text(cfg.revision_tag, tag);
        rt_log("audio: renderer open: %d Hz x%d, %d mix buffers, %d voices, %d effects, %d submixes, %d sinks, %d/%d splitters, "
               "tag %s, REV%d, device %d, mode %d, work buffer %u (asked) %llu (stated)", cfg.sample_rate, cfg.sample_count,
               cfg.mix_buffer_count, cfg.voice_count, cfg.effect_count, cfg.submix_count, cfg.sink_count,
               cfg.splitter_count, cfg.splitter_destination_count, tag, audio_revision_number(cfg.revision_tag),
               cfg.rendering_device, cfg.execution_mode, WORK_BUFFER_BYTES, (unsigned long long)work);
        ipc_answer_session(rs, ipc_session_new("IAudioRenderer", renderer_handler, st, renderer_closed));
        return;
    }
    case 1: {   // GetWorkBufferSize(parameter) -> u64: one constant; the host keeps the state, the client only reserves it
        AudioConfig cfg;
        const char *why = audio_config_read(ipc_arg(c, s, rq, 0, 0x34), 0x34, &cfg);
        if (why) hos_gap(c, "audio", "GetAudioRendererWorkBufferSize: %s", why);
        uint64_t v = WORK_BUFFER_BYTES;
        rt_log("audio: work buffer size asked for %d Hz x%d, %d voices, %d effects, %d submixes, %d sinks, %d splitters, "
               "REV%d: answered %llu bytes", cfg.sample_rate, cfg.sample_count, cfg.voice_count, cfg.effect_count,
               cfg.submix_count, cfg.sink_count, cfg.splitter_count, audio_revision_number(cfg.revision_tag),
               (unsigned long long)v);
        ipc_out(rs, &v, 8);
        return;
    }
    case 2: case 4:   // GetAudioDeviceService(applet resource user id), ...WithRevisionInfo(revision, applet resource user id)
        ipc_answer_session(rs, ipc_session_new("IAudioDevice", device_handler, NULL, NULL));
        return;
    default: ipc_unknown_command(c, s, rq);
    }
}

// ---- aud:u, IUserClientSession: the newer SDKs reach the audio managers through it. Command 2 returns an
// IAudioRendererManager; 0 (the input manager), 1 (the output manager), 3 and 4 and the output-device notification
// command have no reference, and stay named fatals until the game meets one.
static void user_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    switch (rq->cmd) {
    case 2: ipc_answer_session(rs, ipc_session_new("audren:u", manager_handler, NULL, NULL)); return;
    default: ipc_unknown_command(c, s, rq);
    }
}

void srv_audio_register(void) {
    ipc_service_add("aud:u", user_handler);
    ipc_service_add("audren:u", manager_handler);
}
