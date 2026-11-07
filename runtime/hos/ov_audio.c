// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// A silent audio renderer at the nn::audio level (architecture D9, the null back end boot). The game's audio client builds its
// RequestUpdate blocks with the SDK's own code, which runs unchanged; the entry points that would open a session to
// aud:u, audren:u, auddmg:u or audsmx:u are answered here without one: a renderer is a handle and a host thread that
// signals the renderer's event once per rendered frame (sample count / sample rate seconds), and nothing is mixed or
// played. The real services (IPC with a mixer behind them) are the audio ticket's work, M2; this stand-in lets the
// game start and present frames, and every override is a stub in the census so the stand-in cannot be taken for
// the service.
#include <stdlib.h>
#include <string.h>

#include "hos.h"
#include "plat.h"

#define RENDERER_HANDLE 0x41554431ull   // the handle value: one renderer at a time here
#define WORK_BUFFER_BYTES 0x80000ull    // answered for any parameter

typedef struct { int32_t sample_rate, sample_count; } RendererParam;   // the first two fields of nn::audio::AudioRendererParameter

typedef struct {
    uint32_t event;        // writable end of the renderer's event (atomic: the clock thread reads it while a reopen writes it)
    uint64_t frame_ns;     // atomic, as event
    volatile int running, open;
} Renderer;
static Renderer g_renderer;
static int g_clock_threads;   // clock threads started: 1 once the first renderer is open, and never more

int hos_audio_clock_threads(void) { return __atomic_load_n(&g_clock_threads, __ATOMIC_ACQUIRE); }

static void renderer_thread(void *arg) {
    Renderer *r = arg;
    uint64_t next = plat_now_ns();
    for (;;) {
        next += __atomic_load_n(&r->frame_ns, __ATOMIC_RELAXED);
        uint64_t now = plat_now_ns();
        if (next > now) plat_sleep_ns(next - now);
        else next = now;   // behind: no burst of frames to catch up
        if (__atomic_load_n(&r->running, __ATOMIC_ACQUIRE)) hos_signal(__atomic_load_n(&r->event, __ATOMIC_RELAXED));
    }
}

// int64 GetAudioRendererWorkBufferSize(const AudioRendererParameter &)
static void ov_work_size(Cpu *c) { ret_x(c, WORK_BUFFER_BYTES); }

// The stand-in's open, once the writable end of the frame event is known (hos_audio_renderer_open, hos.h): the override
// below binds the game's SystemEvent first, and a test gives the writable end of a kernel event it made itself.
void hos_audio_renderer_open(Cpu *c, uint32_t event, uint64_t handle, uint64_t param) {
    RendererParam p = {48000, 240};
    hos_in(&p, param, sizeof p);
    if (p.sample_rate <= 0 || p.sample_count <= 0 || p.sample_count > 48000) p = (RendererParam){48000, 240};
    if (__atomic_exchange_n(&g_renderer.open, 1, __ATOMIC_ACQ_REL))
        hos_gap(c, "audio", "nn::audio::OpenAudioRenderer: a second renderer; the stand-in keeps one");
    uint64_t frame_ns = 1000000000ull * (uint64_t)p.sample_count / (uint64_t)p.sample_rate;
    __atomic_store_n(&g_renderer.event, event, __ATOMIC_RELAXED);
    __atomic_store_n(&g_renderer.frame_ns, frame_ns, __ATOMIC_RELAXED);
    uint64_t h = RENDERER_HANDLE;
    hos_out(handle, &h, sizeof h);
    static int started;
    if (!__atomic_exchange_n(&started, 1, __ATOMIC_ACQ_REL)) {   // one clock thread for every renderer opened
        if (!plat_thread_start(renderer_thread, &g_renderer, 1ull << 20))
            rt_fatal("audio: cannot start the renderer's clock thread");
        __atomic_fetch_add(&g_clock_threads, 1, __ATOMIC_RELEASE);
    }
    rt_log("audio: stand-in renderer opened: %d Hz, %d samples a frame (%.2f ms), nothing is mixed", p.sample_rate,
           p.sample_count, (double)frame_ns / 1e6);
}

// Result OpenAudioRenderer(AudioRendererHandle *handle, os::SystemEvent *event, const AudioRendererParameter &, void *work, size_t size)
static void ov_open(Cpu *c) {
    uint64_t handle = arg_x(c, 0), ev = arg_x(c, 1);
    if (!handle || !ev) rt_guest_stop(c, "nn::audio::OpenAudioRenderer: null handle or event");
    hos_audio_renderer_open(c, hos_bind_system_event(c, ev, 1), handle, arg_x(c, 2));
    ret_x(c, HOS_R_OK);
}

// Result StartAudioRenderer(AudioRendererHandle), void StopAudioRenderer(AudioRendererHandle)
static void ov_start(Cpu *c) {
    __atomic_store_n(&g_renderer.running, 1, __ATOMIC_RELEASE);
    ret_x(c, HOS_R_OK);
}
static void ov_stop(Cpu *c) {
    __atomic_store_n(&g_renderer.running, 0, __ATOMIC_RELEASE);
    ret_x(c, HOS_R_OK);
}
// Result RequestUpdateAudioRenderer(AudioRendererHandle, const AudioRendererConfig *): accepted; the output block
// the client reads back stays as the SDK left it (no voice has played, no wave buffer is released by the stand-in: a
// game thread that waits for a wave buffer to be consumed stalls here once the title runs, which the audio ticket's real
// renderer answers and a stall detector would name)
static void ov_update(Cpu *c) { ret_x(c, HOS_R_OK); }
// void CloseAudioRenderer(AudioRendererHandle)
static void ov_close(Cpu *c) {
    __atomic_store_n(&g_renderer.running, 0, __ATOMIC_RELEASE);
    __atomic_store_n(&g_renderer.open, 0, __ATOMIC_RELEASE);
    ret_x(c, HOS_R_OK);
}

#define WHY "stand-in: no session to the audio services, nothing is mixed (the audio ticket's real services replace it)"
HOS_OVERRIDE("_ZN2nn5audio30GetAudioRendererWorkBufferSizeERKNS0_22AudioRendererParameterE", ov_work_size, HOS_STUB,
             "answers 512 KiB for any parameter; " WHY);
HOS_OVERRIDE("_ZN2nn5audio17OpenAudioRendererEPNS0_19AudioRendererHandleEPNS_2os11SystemEventERKNS0_22AudioRendererParameterEPvm",
             ov_open, HOS_STUB, "a handle, a bound event signalled once per frame by a host thread; " WHY);
HOS_OVERRIDE("_ZN2nn5audio18StartAudioRendererENS0_19AudioRendererHandleE", ov_start, HOS_STUB,
             "the frame event starts; " WHY);
HOS_OVERRIDE("_ZN2nn5audio17StopAudioRendererENS0_19AudioRendererHandleE", ov_stop, HOS_STUB,
             "the frame event stops; " WHY);
HOS_OVERRIDE("_ZN2nn5audio26RequestUpdateAudioRendererENS0_19AudioRendererHandleEPKNS0_19AudioRendererConfigE", ov_update,
             HOS_STUB, "accepted, the output block is not written; " WHY);
HOS_OVERRIDE("_ZN2nn5audio18CloseAudioRendererENS0_19AudioRendererHandleE", ov_close, HOS_STUB, "the renderer closes; " WHY);

// float GetAudioRendererVoiceDropParameter(AudioRendererHandle), void SetAudioRendererVoiceDropParameter(handle, float),
// Result ExecuteAudioRendererRendering(handle) (manual rendering: nothing to render here)
static float g_voice_drop = 1.0f;   // the SDK's default
static void ov_get_drop(Cpu *c) {
    __uint128_t v = 0;
    memcpy(&v, &g_voice_drop, sizeof g_voice_drop);
    c->v[0] = v;
}
static void ov_set_drop(Cpu *c) {
    float f;
    memcpy(&f, &c->v[0], sizeof f);
    g_voice_drop = f;
    ret_x(c, 0);
}
HOS_OVERRIDE("_ZN2nn5audio34GetAudioRendererVoiceDropParameterENS0_19AudioRendererHandleE", ov_get_drop, HOS_STUB,
             "the value last set, 1.0 at first; " WHY);
HOS_OVERRIDE("_ZN2nn5audio34SetAudioRendererVoiceDropParameterENS0_19AudioRendererHandleEf", ov_set_drop, HOS_STUB,
             "kept for the getter; " WHY);
HOS_OVERRIDE("_ZN2nn5audio29ExecuteAudioRendererRenderingENS0_19AudioRendererHandleE", ov_update, HOS_STUB,
             "accepted, nothing is rendered; " WHY);

// int ListAudioOutputDeviceName(AudioDeviceName *names, int count): one device. An AudioDeviceName is a NUL-terminated
// name in a fixed array; only the name's own bytes are written.
static void ov_list_output(Cpu *c) {
    static const char name[] = "MainAudioOut";
    if (arg_x(c, 0) && (int32_t)arg_x(c, 1) > 0) hos_out(arg_x(c, 0), name, sizeof name);
    ret_x(c, 1);
}
// int ListVirtualInputDevices(AudioDeviceName *names, int count): none (SDK 22; the console answers 0 without a
// virtual input device)
static void ov_list_virtual_input(Cpu *c) { ret_x(c, 0); }
HOS_OVERRIDE("_ZN2nn5audio25ListAudioOutputDeviceNameEPNS0_15AudioDeviceNameEi", ov_list_output, HOS_STUB,
             "one device, MainAudioOut; " WHY);
HOS_OVERRIDE("_ZN2nn5audio23ListVirtualInputDevicesEPNS0_15AudioDeviceNameEi", ov_list_virtual_input, HOS_STUB,
             "no virtual input device; nxnative hle_audio.c:569 answers the same; " WHY);

// ---- the output device (auddmg:u and aud:u in the console's SDK): one device, MainAudioOut, at unit volume
static void device_name(uint64_t at) {
    static const char name[] = "MainAudioOut";
    if (at) hos_out(at, name, sizeof name);
}
// float GetAudioDeviceOutputVolume(const AudioDeviceName &), void SetAudioDeviceOutputVolume(const AudioDeviceName &, float)
static void ov_dev_get_volume(Cpu *c) {
    float one = 1.0f;
    __uint128_t v = 0;
    memcpy(&v, &one, sizeof one);
    c->v[0] = v;
}
static void ov_dev_nothing(Cpu *c) { ret_x(c, 0); }
// Result GetDefaultVirtualOutputDevice(AudioDeviceName *), void GetActiveAudioOutputDeviceName(AudioDeviceName *)
static void ov_dev_name0(Cpu *c) {
    device_name(arg_x(c, 0));
    ret_x(c, HOS_R_OK);
}
// void AcquireAudioOutputDeviceNotification(os::SystemEvent *): a bound event that never fires: the device never changes
static void ov_dev_notification(Cpu *c) {
    if (arg_x(c, 0)) hos_bind_system_event(c, arg_x(c, 0), 0);
    ret_x(c, HOS_R_OK);
}
// int GetActiveAudioDeviceChannelCountForOutput(): stereo
static void ov_dev_channels(Cpu *c) { ret_x(c, 2); }
HOS_OVERRIDE("_ZN2nn5audio26GetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameE", ov_dev_get_volume, HOS_STUB,
             "unit volume; " WHY);
HOS_OVERRIDE("_ZN2nn5audio26SetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameEf", ov_dev_nothing, HOS_STUB,
             "accepted, nothing is played; " WHY);
HOS_OVERRIDE("_ZN2nn5audio26GetTrueRawAudioInputDeviceEPNS0_15AudioDeviceNameE", ov_dev_nothing, HOS_STUB,
             "answers 0 without a name: there is no input device; " WHY);
HOS_OVERRIDE("_ZN2nn5audio29GetDefaultVirtualOutputDeviceEPNS0_15AudioDeviceNameE", ov_dev_name0, HOS_STUB,
             "MainAudioOut; " WHY);
HOS_OVERRIDE("_ZN2nn5audio30GetActiveAudioOutputDeviceNameEPNS0_15AudioDeviceNameE", ov_dev_name0, HOS_STUB,
             "MainAudioOut; " WHY);
HOS_OVERRIDE("_ZN2nn5audio32GetVirtualOutputDevicePropertiesEPNS0_23VirtualDevicePropertiesERKNS0_15AudioDeviceNameE",
             ov_dev_nothing, HOS_STUB, "answers 0 and leaves the properties alone; " WHY);
HOS_OVERRIDE("_ZN2nn5audio36AcquireAudioOutputDeviceNotificationEPNS_2os11SystemEventE", ov_dev_notification, HOS_STUB,
             "a bound event that never fires; " WHY);
HOS_OVERRIDE("_ZN2nn5audio36ReleaseAudioOutputDeviceNotificationEPNS_2os11SystemEventE", ov_dev_nothing, HOS_STUB,
             "accepted; " WHY);
HOS_OVERRIDE("_ZN2nn5audio39IsDeviceOutputVolumeAutoTuneModeEnabledEv", ov_dev_nothing, HOS_STUB, "off; " WHY);
HOS_OVERRIDE("_ZN2nn5audio40SetDeviceOutputVolumeAutoTuneModeEnabledEb", ov_dev_nothing, HOS_STUB, "accepted; " WHY);
HOS_OVERRIDE("_ZN2nn5audio41GetActiveAudioDeviceChannelCountForOutputEv", ov_dev_channels, HOS_STUB, "stereo; " WHY);

// ---- the hardware Opus decoder (hwopus): sample rate and channel count are kept in the decoder's first 8 bytes, and a
// decode consumes its input and produces no samples: the game's streams run, silent
// size_t GetHardwareOpusDecoderWorkBufferSize(int rate, int channels, int): the stand-in needs none; 36 KiB is answered
static void ov_opus_work(Cpu *c) { ret_x(c, 0x9000); }
// Result InitializeHardwareOpusDecoder(HardwareOpusDecoderType *, int rate, int channels, void *work, size_t size, int)
static void ov_opus_init(Cpu *c) {
    int32_t v[2] = {(int32_t)arg_x(c, 1), (int32_t)arg_x(c, 2)};
    if (arg_x(c, 0)) hos_out(arg_x(c, 0), v, sizeof v);
    ret_x(c, HOS_R_OK);
}
// int GetHardwareOpusDecoderSampleRate(const HardwareOpusDecoderType *), int GetHardwareOpusDecoderChannelCount(...)
static void ov_opus_get(Cpu *c, int i) {
    int32_t v[2] = {48000, 2};
    if (arg_x(c, 0)) hos_in(v, arg_x(c, 0), sizeof v);
    ret_x(c, (uint32_t)v[i]);
}
static void ov_opus_rate(Cpu *c) { ov_opus_get(c, 0); }
static void ov_opus_channels(Cpu *c) { ov_opus_get(c, 1); }
// OpusResult DecodeOpusInterleavedWithHardware(decoder, size_t *consumed, int *samples, int16_t *out, size_t out_size,
// const void *in, size_t in_size): the input is consumed, no sample is made
static void ov_opus_decode(Cpu *c) {
    uint64_t consumed = arg_x(c, 6);   // in_size is the seventh argument: x6
    int32_t samples = 0;
    if (arg_x(c, 1)) hos_out(arg_x(c, 1), &consumed, sizeof consumed);
    if (arg_x(c, 2)) hos_out(arg_x(c, 2), &samples, sizeof samples);
    ret_x(c, HOS_R_OK);
}
#define WHY_OPUS "stand-in: no hwopus session, no sample is decoded (the audio ticket's real service replaces it)"
HOS_OVERRIDE("_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii", ov_opus_work, HOS_STUB, "36 KiB, as the reference decoder asks; " WHY_OPUS);
HOS_OVERRIDE("_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi", ov_opus_init, HOS_STUB,
             "rate and channels kept in the decoder's first 8 bytes; " WHY_OPUS);
HOS_OVERRIDE("_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE", ov_opus_rate, HOS_STUB,
             "the kept rate; " WHY_OPUS);
HOS_OVERRIDE("_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE", ov_opus_channels,
             HOS_STUB, "the kept channel count; " WHY_OPUS);
HOS_OVERRIDE("_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm", ov_opus_decode,
             HOS_STUB, "consumes the input, makes no sample; " WHY_OPUS);
HOS_OVERRIDE("_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE", ov_dev_nothing, HOS_STUB,
             "accepted; " WHY_OPUS);
HOS_OVERRIDE("_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE", ov_dev_nothing, HOS_STUB,
             "accepted; " WHY_OPUS);
