// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// auddmg:u, the audio device manager of the console's newer SDKs (architecture D9): the virtual audio devices the SDK
// lists and asks about before it opens an output stream. The game lists the virtual input devices when its audio
// starts; the host has no microphone, so that list is empty, and the one virtual output device is named for the
// calls that ask for it. Command 0 of the service opens the manager object; the manager's commands are (id: request ->
// reply):
//   0(u32 kind): u32 count, the names in the out buffer (0x100 bytes each)      100(): the default output device's name
//   300(4 x u32: sample type, channels, rate, functionality): result only        401(name buffer): the 0x228-byte property block
//   500(16 bytes, name buffer): copy handle of a state-change event
//   700(u32 stream, u32 functionality, name buffer): attach a sound mixer stream to the output device, which engages it
//   800(u32 stream, ...): detach it, which disengages it
//
// Adapted from oot-pc-recomp runtime/audio_nx2.c at f4656d3b31a1348fe04ab2b989f15c6047e681ae (no licence header: the
// repository's GPL-3.0-or-later; the file says its wire format was read from Lichii/Eden
// src/core/hle/service/audio/nx2_audio_device_manager.h, GPL-3.0-or-later, for the command layout only). Changed here:
// handlers get the calling Cpu; the list answers by its argument (the donor answered one output device whatever was
// asked, where the game asks for input devices and must hear none); the names are this runtime's; the state event is
// made per request, with no host end, since it never signals.
#include <stdint.h>
#include <string.h>

#include "hos.h"
#include "ipc.h"

#define VIRTUAL_DEVICE "AudioVirtualOutput0"
#define PHYSICAL_DEVICE "MainAudioOut"
#define NAME_SIZE 0x100
#define PROPS_SIZE 0x228
// The list's kind argument. The game's own start-up call, which lists the input devices, sends this value (the first run logged
// it: 0, with room for eight names); any other value is taken to ask for the output devices.
#define LIST_INPUT 0u

// The first buffer the caller gave for the answer: B or W, or the C (pointer) buffer the answer goes through. The buffer
// is returned by value, with whether it is the pointer kind.
static IpcBuffer answer_buffer(const IpcRequest *rq, int *pointer) {
    *pointer = !(rq->nb && rq->b[0].size) && !(rq->nw && rq->w[0].size) && rq->nc && rq->c[0].size;
    if (*pointer) return rq->c[0];
    if (rq->nb && rq->b[0].size) return rq->b[0];
    return rq->nw ? rq->w[0] : (IpcBuffer){0};
}

static int put_answer(IpcResponse *rs, const IpcRequest *rq, const IpcBuffer *b, int pointer, const void *data, uint64_t n) {
    return pointer ? ipc_answer_pointer(rs, rq, 0, data, n) : ipc_write(b, 0, data, n);
}

static void put_text(uint8_t *dst, const char *s) {
    size_t n = strlen(s);
    memcpy(dst, s, n < NAME_SIZE - 1 ? n : NAME_SIZE - 1);
}

static void manager_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    int pointer;
    IpcBuffer b = answer_buffer(rq, &pointer);
    uint8_t name[NAME_SIZE];
    switch (rq->cmd) {
    case 0: {   // ListVirtualDevices(kind): the input list is empty, the output list is the one virtual device
        uint32_t kind;
        memcpy(&kind, ipc_arg(c, s, rq, 0, 4), 4);
        static int logged;
        if (!__atomic_exchange_n(&logged, 1, __ATOMIC_RELAXED))
            rt_log("audio: auddmg:u list of virtual devices, kind %u (%s), room for %llu bytes", kind,
                   kind == LIST_INPUT ? "input: none" : "output: one", (unsigned long long)b.size);
        uint32_t n = kind != LIST_INPUT && b.size >= NAME_SIZE ? 1 : 0;
        if (n) {
            memset(name, 0, sizeof name);
            put_text(name, VIRTUAL_DEVICE);
            put_answer(rs, rq, &b, pointer, name, NAME_SIZE);
        }
        ipc_out(rs, &n, 4);
        return;
    }
    case 100:   // GetDefaultVirtualOutputDevice
        if (b.size >= NAME_SIZE) {
            memset(name, 0, sizeof name);
            put_text(name, VIRTUAL_DEVICE);
            put_answer(rs, rq, &b, pointer, name, NAME_SIZE);
        }
        return;
    case 300: return;   // IsVirtualOutputDeviceStreamFormatSupported: the mixer takes any PCM format, result 0
    case 401: {         // GetVirtualOutputDeviceProperties
        if (b.size < PROPS_SIZE) {
            rs->result = 0xCA01u;   // invalid size
            return;
        }
        uint8_t props[PROPS_SIZE] = {0};
        // name at 0, type 3 at 0x100, 2 at 0x104, stereo at 0x108, 48 kHz at 0x10c, the physical device's name at 0x110, its
        // latency in ns at 0x218 and a nonzero device id at 0x220 (the SDK passes that id to the system mixer's CreateStream,
        // which refuses 0)
        put_text(props, VIRTUAL_DEVICE);
        const uint32_t v100 = 3, v104 = 2, channels = 2, rate = 48000, dev_id = 1;
        const uint64_t latency_ns = 10000000;
        memcpy(props + 0x100, &v100, 4), memcpy(props + 0x104, &v104, 4), memcpy(props + 0x108, &channels, 4);
        memcpy(props + 0x10c, &rate, 4);
        put_text(props + 0x110, PHYSICAL_DEVICE);
        memcpy(props + 0x218, &latency_ns, 8), memcpy(props + 0x220, &dev_id, 4);
        put_answer(rs, rq, &b, pointer, props, PROPS_SIZE);
        return;
    }
    case 700: {   // AttachStream: functionality 1 is output
        uint32_t id, functionality;
        memcpy(&id, ipc_arg(c, s, rq, 0, 4), 4);
        memcpy(&functionality, ipc_arg(c, s, rq, 4, 4), 4);
        rs->result = functionality == 1 ? sysmix_set_engaged(id, 1) : 0x40299u;   // anything else is not supported
        return;
    }
    case 800: {   // DetachStream
        uint32_t id;
        memcpy(&id, ipc_arg(c, s, rq, 0, 4), 4);
        rs->result = sysmix_set_engaged(id, 0);
        return;
    }
    case 500: {   // AcquireVirtualOutputDeviceStateEvent: never signalled, the device does not change
        uint32_t readable, writable = rt_kernel_event_create(&readable);
        rt_kernel_event_close(writable);
        rs->copy[rs->ncopy++] = readable;
        return;
    }
    default: ipc_unknown_command(c, s, rq);
    }
}

static void service_handler(IpcSession *s, Cpu *c, const IpcRequest *rq, IpcResponse *rs) {
    if (rq->cmd == 0) {   // OpenVirtualDeviceManager (16 bytes and the process id)
        ipc_answer_session(rs, ipc_session_new("IAudioVirtualDM", manager_handler, NULL, NULL));
        return;
    }
    ipc_unknown_command(c, s, rq);
}

void srv_auddmg_register(void) { ipc_service_add("auddmg:u", service_handler); }
