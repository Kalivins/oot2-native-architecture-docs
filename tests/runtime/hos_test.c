// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the Horizon layer (runtime/hos): the service SVCs and sm: through hand-built messages in a guest
// thread's message buffer, the host-object handles they make, the census file and its rows, the override notes, and
// the boot driver's stop when no module defines nn::init::Start. Synthetic guest programs written by hand in the
// lifter's shape run on the kernel's guest threads; no game code is involved. Every file of runtime/hos is linked in,
// so this build is also the compile check of the overrides.
//
//   hos_test CASE DIR   runs one case with DIR as the game folder (census.tsv goes there); prints "ok CASE" and exits
//                       0, or exits 1 with a FAIL line. The cases that end in a named fatal exit with the runtime's
//                       code (tests/test_runtime_hos.py).
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "layout.h"
#include "endmark.h"
#include "oot2/boot.h"
#include "oot2/gfx.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "plat.h"
#include "../../runtime/hos/hos.h"

#define CHECK(cond)                                                                        \
    do {                                                                                   \
        if (!(cond)) {                                                                     \
            fprintf(stderr, "FAIL %s:%d: %s\n", __FILE__, __LINE__, #cond);                \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

#define MB MOD_MAIN_BASE
#define G(i) (MB + 0x100ull * (i))
#define PC 0x7000000040ull
#define R_INVALID_SIZE 0xCA01u
#define R_INVALID_HANDLE 0xE401u
#define SFCI 0x49434653u
#define SFCO 0x4f434653u

static uint64_t HEAP, g_bound;
// the audio stand-ins (runtime/hos/ov_audio.c), bound as the loader binds an import: every override of the file
static const char *const AUDIO_SYMBOLS[] = {
    "_ZN2nn5audio30GetAudioRendererWorkBufferSizeERKNS0_22AudioRendererParameterE",
    "_ZN2nn5audio25ListAudioOutputDeviceNameEPNS0_15AudioDeviceNameEi",
    "_ZN2nn5audio34GetAudioRendererVoiceDropParameterENS0_19AudioRendererHandleE",
    "_ZN2nn5codec36GetHardwareOpusDecoderWorkBufferSizeEiii",
    "_ZN2nn5codec29InitializeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeEiiPvmi",
    "_ZN2nn5codec32GetHardwareOpusDecoderSampleRateEPKNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec34GetHardwareOpusDecoderChannelCountEPKNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec33DecodeOpusInterleavedWithHardwareEPNS0_23HardwareOpusDecoderTypeEPmPiPsmPKvm",
    "_ZN2nn5audio17OpenAudioRendererEPNS0_19AudioRendererHandleEPNS_2os11SystemEventERKNS0_22AudioRendererParameterEPvm",
    "_ZN2nn5audio18StartAudioRendererENS0_19AudioRendererHandleE",
    "_ZN2nn5audio17StopAudioRendererENS0_19AudioRendererHandleE",
    "_ZN2nn5audio26RequestUpdateAudioRendererENS0_19AudioRendererHandleEPKNS0_19AudioRendererConfigE",
    "_ZN2nn5audio18CloseAudioRendererENS0_19AudioRendererHandleE",
    "_ZN2nn5audio34SetAudioRendererVoiceDropParameterENS0_19AudioRendererHandleEf",
    "_ZN2nn5audio29ExecuteAudioRendererRenderingENS0_19AudioRendererHandleE",
    "_ZN2nn5audio23ListVirtualInputDevicesEPNS0_15AudioDeviceNameEi",
    "_ZN2nn5audio26GetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameE",
    "_ZN2nn5audio26SetAudioDeviceOutputVolumeEPKNS0_15AudioDeviceNameEf",
    "_ZN2nn5audio26GetTrueRawAudioInputDeviceEPNS0_15AudioDeviceNameE",
    "_ZN2nn5audio29GetDefaultVirtualOutputDeviceEPNS0_15AudioDeviceNameE",
    "_ZN2nn5audio30GetActiveAudioOutputDeviceNameEPNS0_15AudioDeviceNameE",
    "_ZN2nn5audio32GetVirtualOutputDevicePropertiesEPNS0_23VirtualDevicePropertiesERKNS0_15AudioDeviceNameE",
    "_ZN2nn5audio36AcquireAudioOutputDeviceNotificationEPNS_2os11SystemEventE",
    "_ZN2nn5audio36ReleaseAudioOutputDeviceNotificationEPNS_2os11SystemEventE",
    "_ZN2nn5audio39IsDeviceOutputVolumeAutoTuneModeEnabledEv",
    "_ZN2nn5audio40SetDeviceOutputVolumeAutoTuneModeEnabledEb",
    "_ZN2nn5audio41GetActiveAudioDeviceChannelCountForOutputEv",
    "_ZN2nn5codec24ResetHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE",
    "_ZN2nn5codec27FinalizeHardwareOpusDecoderEPNS0_23HardwareOpusDecoderTypeE",
};
enum { A_WORK, A_LIST, A_DROP, A_OPUS_WORK, A_OPUS_INIT, A_OPUS_RATE, A_OPUS_CH, A_OPUS_DECODE, A_OPEN, A_START, A_STOP,
       A_UPDATE, A_CLOSE, A_SET_DROP, A_EXECUTE, A_LIST_VIRTUAL, A_GET_VOLUME, A_SET_VOLUME, A_TRUE_RAW, A_DEFAULT_OUT,
       A_ACTIVE_OUT, A_VIRTUAL_PROPS, A_ACQUIRE, A_RELEASE, A_AUTOTUNE_GET, A_AUTOTUNE_SET, A_CHANNELS, A_OPUS_RESET,
       A_OPUS_FINALIZE, A_COUNT };
static uint64_t g_audio[A_COUNT];
// the movie player's first step (runtime/hos/ov_movie.c)
static const char MOVIE_SYMBOL[] =
    "_ZN5movie12VideoDecoder10InitializeEPvmN2nn5movie13MediaMimeTypeENS_24VideoDecoderOutputFormatENS_14VideoCodecModeE";
static uint64_t g_movie;
static char g_nvn_name[96];
static const char *g_case;

// An override with a note, which the test binds as the loader would, and one no module names.
static void host_bound(Cpu *c) { ret_x(c, 7); }
HOS_OVERRIDE("hos_test_bound", host_bound, HOS_OFFLINE, "seven, as a console without the service answers");
static void host_unbound(Cpu *c) { (void)c; }
HOS_OVERRIDE("hos_test_unbound", host_unbound, HOS_STUB, "never bound");

// ---- guest helpers
static inline uint64_t sv(Cpu *c, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3;
    rt_svc(c, imm, PC);
    return c->x[0];
}
static inline uint32_t *msg(Cpu *c) { return (uint32_t *)(uintptr_t)c->tpidrro; }

static uint32_t open_port(Cpu *c, const char *port) {
    memcpy((void *)(uintptr_t)(HEAP + 0x100), port, strlen(port) + 1);
    CHECK(sv(c, 0x1F, 0, HEAP + 0x100, 0, 0) == 0);
    return (uint32_t)c->x[1];
}

// A TIPC request of command cmd with data bytes (and the client's process id when pid is set): the result.
static uint32_t tipc(Cpu *c, uint32_t h, uint32_t cmd, const void *data, uint32_t size, int pid) {
    uint32_t *m = msg(c), *p = m + 2;
    memset(m, 0, 0x100);
    m[0] = 16 + cmd;
    m[1] = (size + 3) / 4 | (pid ? 1u << 31 : 0);
    if (pid) {
        *p++ = 1;
        p += 2;
    }
    memcpy(p, data, size);
    CHECK(sv(c, 0x21, h, 0, 0, 0) == 0);
    CHECK(!(m[1] >> 31));
    return m[2];
}

// A CMIF control request of command cmd: the result; *moved gets the moved handle, *word the first data word.
static uint32_t control(Cpu *c, uint32_t h, uint32_t cmd, uint32_t *moved, uint32_t *word) {
    uint32_t *m = msg(c);
    memset(m, 0, 0x100);
    m[0] = 5;
    m[1] = 8;   // the 16-byte header 16-byte aligned, with libnx's slack
    m[4] = SFCI, m[6] = cmd;
    CHECK(sv(c, 0x21, h, 0, 0, 0) == 0);
    unsigned at = 2;
    if (m[1] >> 31) {
        CHECK((m[2] >> 5 & 0xf) == 1);
        *moved = m[3];
        at = 4;
    }
    at = (at + 3) & ~3u;   // the payload starts 16-byte aligned
    CHECK(m[at] == SFCO);
    if (word) *word = m[at + 4];
    return m[at + 2];
}

// ---- the audio stand-ins, called from a host Cpu as rt_call does
static uint64_t acall(Cpu *a, int i, uint64_t x0, uint64_t x1) {
    a->x[0] = x0, a->x[1] = x1;
    rt_call(a, g_audio[i]);
    return a->x[0];
}
static int audio_case(void) { return !strncmp(g_case, "audio", 5); }
// An NVN2 entry point called from the host thread by its synthetic address.
static uint64_t nvn(Cpu *a, const char *name, uint64_t x0, uint64_t x1, uint64_t x2) {
    a->x[0] = x0, a->x[1] = x1, a->x[2] = x2;
    rt_call(a, nvn_proc_address(name));
    return a->x[0];
}

// ---- guest programs (each ends its thread)
static void g_sm(Cpu *c) {
    uint32_t h = open_port(c, "sm:");
    CHECK(h != 0);
    CHECK(tipc(c, h, 0, NULL, 0, 1) == 0);   // RegisterClient
    CHECK((msg(c)[1] & 0x3ff) == 1);         // one data word, the result: TIPC puts no header before the data
    uint32_t size = 0, clone = 0;
    CHECK(control(c, h, 3, &clone, &size) == 0 && (size & 0xffff) == 0x8000);   // QueryPointerBufferSize
    CHECK(control(c, h, 2, &clone, NULL) == 0 && clone && clone != h);         // CloneCurrentObject
    CHECK(tipc(c, clone, 0, NULL, 0, 1) == 0);   // the clone is the same service
    CHECK(sv(c, 0x16, clone, 0, 0, 0) == 0);
    CHECK(sv(c, 0x16, h, 0, 0, 0) == 0);
    CHECK(sv(c, 0x16, h, 0, 0, 0) == R_INVALID_HANDLE);   // closed
    CHECK(sv(c, 0x21, h, 0, 0, 0) == R_INVALID_HANDLE);
    // transfer memory: a record with a handle; a size that is no whole page is refused
    CHECK(sv(c, 0x15, 0, HEAP + 0x10000, 0x2000, 0) == 0);
    uint32_t t = (uint32_t)c->x[1];
    CHECK(t != 0);
    CHECK(sv(c, 0x21, t, 0, 0, 0) == R_INVALID_HANDLE);   // no session
    CHECK(sv(c, 0x16, t, 0, 0, 0) == 0);
    CHECK(sv(c, 0x15, 0, HEAP + 0x10000, 0x1234, 0) == R_INVALID_SIZE);
    // a host function called from main+0x120: the census names that call site
    c->x[30] = G(1) + 0x24;
    c->pc = g_bound;
    rt_dispatch(c);
    CHECK(c->x[0] == 7);
    sv(c, 0x0A, 0, 0, 0, 0);
}

static void g_fatal(Cpu *c) {
    if (!strcmp(g_case, "port")) open_port(c, "xyz:");
    uint32_t h = open_port(c, "sm:");
    if (!strcmp(g_case, "service") || !strcmp(g_case, "service_census")) {
        uint64_t name = 0;
        memcpy(&name, "lm", 2);
        uint32_t r = tipc(c, h, 1, &name, 8, 0);   // GetServiceHandle
        if (!strcmp(g_case, "service_census")) {
            CHECK(r == 0x415);   // census mode: sm: refuses the name, the run goes on
            CHECK(tipc(c, h, 1, &name, 8, 0) == 0x415);   // the same name again: refused again, listed once
            // 40 more names: refused every time, but the log keeps 32 names in all
            for (int i = 0; i < 40; i++) {
                char n[9];
                snprintf(n, sizeof n, "svc%05d", i);
                memcpy(&name, n, 8);
                CHECK(tipc(c, h, 1, &name, 8, 0) == 0x415);
            }
        }
    } else if (!strcmp(g_case, "command")) {
        tipc(c, h, 7, NULL, 0, 0);
    } else if (!strcmp(g_case, "domain")) {
        uint32_t moved = 0;
        control(c, h, 0, &moved, NULL);
    } else if (!strcmp(g_case, "nvn") || !strcmp(g_case, "nvn_census")) {
        // the first NVN2 entry point of the list with no body, called from main+0x250 as the game calls its table
        const char *name = NULL;
        for (int i = 0; !name && i < nvn_proc_count(); i++)
            if (!nvn_proc_has_body(i)) name = nvn_proc_name(i);
        CHECK(name != NULL);
        snprintf(g_nvn_name, sizeof g_nvn_name, "%s", name);
        fprintf(stderr, "case nvn: %s\n", name);
        c->x[30] = G(2) + 0x54;
        c->pc = nvn_proc_address(name);
        rt_dispatch(c);
    } else if (!strcmp(g_case, "movie")) {
        c->x[30] = G(2) + 0x54;
        c->pc = g_movie;
        rt_dispatch(c);
    } else if (!strcmp(g_case, "fault")) {
        // a read of unmapped memory by guest code, with a frame record the fault names: x30 and the x29 chain
        uint64_t frame[2] = {0, G(1) + 0x44};
        memcpy((void *)(uintptr_t)(HEAP + 0x800), frame, sizeof frame);
        c->x[29] = HEAP + 0x800;
        c->x[30] = G(1) + 0x24;
        printf("read %llx\n", (unsigned long long)*(volatile uint64_t *)(uintptr_t)0x7f00000000ull);
    } else if (!strcmp(g_case, "fault_chain")) {
        // the same read, with an x29 that is aligned and below the address limit but not mapped: the report's own read of the
        // frame chain faults, and the first fault stays the reason (one FATAL line)
        c->x[29] = 0x7e00000000ull;
        c->x[30] = G(1) + 0x24;
        printf("read %llx\n", (unsigned long long)*(volatile uint64_t *)(uintptr_t)0x7f00000000ull);
    } else if (!strcmp(g_case, "audio_open_binds") || !strcmp(g_case, "audio_open_null") ||
               !strcmp(g_case, "audio_notification_binds")) {
        // the overrides that bind the game's SystemEvent through the SDK, which this module does not define: the named
        // fatal says it was reached; with a null handle or event OpenAudioRenderer stops the guest itself
        memcpy((void *)(uintptr_t)(HEAP + 0x340), &(int32_t[2]){48000, 480}, 8);
        c->x[30] = G(2) + 0x54;
        if (!strcmp(g_case, "audio_notification_binds")) {
            c->x[0] = HEAP + 0x400;
            c->pc = g_audio[A_ACQUIRE];
        } else {
            c->x[0] = !strcmp(g_case, "audio_open_null") ? 0 : HEAP + 0x300, c->x[1] = HEAP + 0x400,
            c->x[2] = HEAP + 0x340;
            c->pc = g_audio[A_OPEN];
        }
        rt_dispatch(c);
    } else if (!strcmp(g_case, "audio_second")) {
        // one renderer at a time: a second open without a close is the named fatal
        memcpy((void *)(uintptr_t)(HEAP + 0x340), &(int32_t[2]){48000, 480}, 8);
        uint32_t readable;
        uint32_t w = rt_kernel_event_create(&readable);
        hos_audio_renderer_open(c, w, HEAP + 0x300, HEAP + 0x340);
        hos_audio_renderer_open(c, w, HEAP + 0x300, HEAP + 0x340);
    }
    sv(c, 0x0A, 0, 0, 0, 0);
}

// The renderer's frame event, seen from a guest thread: nothing before StartAudioRenderer, one signal a frame while it
// runs (a frame is 10 ms here), nothing after StopAudioRenderer, and a fresh renderer after CloseAudioRenderer.
static uint64_t wait_event(Cpu *c, uint32_t readable, uint64_t ns) {
    memcpy((void *)(uintptr_t)(HEAP + 0x380), &readable, 4);
    uint64_t r = sv(c, 0x18, 0, HEAP + 0x380, 1, ns);   // WaitSynchronization: 0 when the event is signalled
    if (r == 0) CHECK(sv(c, 0x12, readable, 0, 0, 0) == 0);   // ClearEvent, as the SDK does for an auto-clear event
    return r;
}
static void open_renderer(Cpu *c, uint32_t w, int32_t rate, int32_t count) {
    memcpy((void *)(uintptr_t)(HEAP + 0x340), &(int32_t[2]){rate, count}, 8);
    memset((void *)(uintptr_t)(HEAP + 0x300), 0, 8);
    hos_audio_renderer_open(c, w, HEAP + 0x300, HEAP + 0x340);
    uint64_t handle;
    memcpy(&handle, (const void *)(uintptr_t)(HEAP + 0x300), 8);
    CHECK(handle == 0x41554431ull);   // the renderer's handle
}
static void g_audio_events(Cpu *c) {
    uint32_t readable, w = rt_kernel_event_create(&readable);
    open_renderer(c, w, 48000, 480);
    CHECK(wait_event(c, readable, 60000000) == 0xEA01u);   // not started: six frames' time, no signal
    CHECK(acall(c, A_START, 0x41554431ull, 0) == 0);
    uint64_t t0 = plat_now_ns();
    for (int i = 0; i < 5; i++) CHECK(wait_event(c, readable, 1000000000ull) == 0);   // one signal a frame
    uint64_t span = plat_now_ns() - t0;
    CHECK(span >= 30000000ull && span < 5000000000ull);   // five frames of 10 ms: not 5 ms (the lower bound is the point), not 100 s
    CHECK(acall(c, A_STOP, 0x41554431ull, 0) == 0);
    while (wait_event(c, readable, 0) == 0) {}   // a frame that was in flight when the clock stopped
    CHECK(wait_event(c, readable, 60000000) == 0xEA01u);   // stopped: nothing
    CHECK(acall(c, A_CLOSE, 0x41554431ull, 0) == 0);
    // closed: a second open is no gap, and the parameters it is given set the frame: 16 kHz and 160 samples is 10 ms;
    // a rate or count that is zero, negative or past 48000 samples is the SDK's default, 48 kHz and 240 samples
    uint32_t r2, w2 = rt_kernel_event_create(&r2);
    open_renderer(c, w2, 16000, 160);
    // a renderer closed while it runs stops signalling (Close ends it as Stop does)
    CHECK(acall(c, A_START, 0x41554431ull, 0) == 0);
    CHECK(wait_event(c, r2, 1000000000ull) == 0);
    CHECK(acall(c, A_CLOSE, 0x41554431ull, 0) == 0);
    while (wait_event(c, r2, 0) == 0) {}
    CHECK(wait_event(c, r2, 60000000) == 0xEA01u);
    // and a renderer opened after that does not run until it is started: nothing is signalled
    open_renderer(c, w2, 16000, 160);
    CHECK(wait_event(c, r2, 60000000) == 0xEA01u);
    CHECK(acall(c, A_CLOSE, 0x41554431ull, 0) == 0);
    open_renderer(c, w2, 0, 240);
    CHECK(acall(c, A_CLOSE, 0x41554431ull, 0) == 0);
    open_renderer(c, w2, 48000, -5);
    CHECK(acall(c, A_CLOSE, 0x41554431ull, 0) == 0);
    open_renderer(c, w2, 48000, 48001);
    CHECK(acall(c, A_CLOSE, 0x41554431ull, 0) == 0);
    CHECK(hos_audio_clock_threads() == 1);   // seven opens, one clock thread
    sv(c, 0x0A, 0, 0, 0, 0);
}

static void guest_end(Cpu *c) { (void)c; }

// ---- host side
static const uint32_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D,
                                    0x0E, 0x0F, 0x10, 0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A,
                                    0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20, 0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27,
                                    0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

static void boot(const char *folder) {
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x3, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512, .is_application = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++) k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    g_bound = rt_hostfn_bind("hos_test_bound");   // as the loader binds an import
    CHECK(g_bound != 0);
    if (audio_case())
        for (int i = 0; i < A_COUNT; i++) CHECK((g_audio[i] = rt_hostfn_bind(AUDIO_SYMBOLS[i])) != 0);
    if (!strcmp(g_case, "movie")) CHECK((g_movie = rt_hostfn_bind(MOVIE_SYMBOL)) != 0);
    if (!strcmp(g_case, "nvn") || !strcmp(g_case, "nvn_census") || !strcmp(g_case, "census_files"))
        nvn_procs_register();   // as main.c does, before the layer
    hos_init(folder, NULL, 0);
    static const uint64_t addrs[] = {G(1), G(2), G(3), G(4)};
    static const GuestFn fns[] = {g_sm, g_fatal, g_audio_events, guest_end};
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, addrs, fns, 4, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 16ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
}

static char *read_census(const char *folder) {
    char path[1024];
    snprintf(path, sizeof path, "%s/census.tsv", folder);
    FILE *f = fopen(path, "rb");
    CHECK(f != NULL);
    static char buf[1 << 20];
    size_t n = fread(buf, 1, sizeof buf - 1, f);
    fclose(f);
    buf[n] = 0;
    return buf;
}

int main(int argc, char **argv) {
    if (argc != 3) {
        fprintf(stderr, "usage: hos_test CASE DIR\n");
        return 2;
    }
    g_case = argv[1];
    const char *dir = argv[2];
    if (!strcmp(g_case, "sm")) {
        boot(dir);
        rt_kernel_run(G(1), 0);
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        static const char header[] = "kind\tkey\tstatus\tcalls\tfirst_at\twhy\n";
        CHECK(!strncmp(t, header, sizeof header - 1));
        CHECK(strstr(t, "\nsvc\t0x1f ConnectToNamedPort\timplemented\t1\t0x7000000040 (main+0x40)\t\n"));
        CHECK(strstr(t, "\nsvc\t0x15 CreateTransferMemory\timplemented\t2\t"));
        CHECK(strstr(t, "\ncommand\tsm: tipc 0\timplemented\t2\t"));
        CHECK(strstr(t, "\ncommand\tsm: control 3\timplemented\t1\t"));
        CHECK(strstr(t, "\ncommand\tsm: control 2\timplemented\t1\t"));
        CHECK(strstr(t, "\nimport\thos_test_bound\toffline\t1\t0x7000000120 (main+0x120)\tseven, as a console without "
                        "the service answers\n"));
        CHECK(strstr(t, "\nd3\tsetjmp\timplemented\t0\t"));
        CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n"));
    } else if (!strcmp(g_case, "notes")) {
        boot(dir);
        const HosNote *n = hos_note("hos_test_bound");
        CHECK(n && !strcmp(n->status, HOS_OFFLINE));
        CHECK(!hos_note("hos_test_absent"));
        int found = 0;
        for (int i = 0; rt_hostfn_unbound(i); i++) {
            CHECK(strcmp(rt_hostfn_unbound(i), "hos_test_bound"));
            found += !strcmp(rt_hostfn_unbound(i), "hos_test_unbound");
        }
        CHECK(found == 1);
    } else if (!strcmp(g_case, "audio")) {
        boot(dir);
        Cpu a;
        memset(&a, 0, sizeof a);
        a.excl_addr = ~0ull;
        // the renderer's work buffer: a size the game can allocate from its own heap, more than the service's 64 KiB
        rt_call(&a, g_audio[A_WORK]);
        CHECK(a.x[0] == 0x80000);
        // one output device, whose name is written NUL-terminated and nothing else is touched past it
        memset((void *)(uintptr_t)(HEAP + 0x200), 0x7f, 64);
        a.x[0] = HEAP + 0x200, a.x[1] = 4;
        rt_call(&a, g_audio[A_LIST]);
        CHECK(a.x[0] == 1);
        CHECK(!strcmp((const char *)(uintptr_t)(HEAP + 0x200), "MainAudioOut"));
        CHECK(((const uint8_t *)(uintptr_t)(HEAP + 0x200))[13] == 0x7f);
        a.x[0] = HEAP + 0x200, a.x[1] = 0;   // no room asked for: nothing is written
        memset((void *)(uintptr_t)(HEAP + 0x200), 0x7f, 16);
        rt_call(&a, g_audio[A_LIST]);
        CHECK(((const uint8_t *)(uintptr_t)(HEAP + 0x200))[0] == 0x7f);
        // the voice drop parameter, 1.0 as the SDK's default
        rt_call(&a, g_audio[A_DROP]);
        float drop;
        memcpy(&drop, &a.v[0], sizeof drop);
        CHECK(drop == 1.0f);
        // the hardware Opus decoder: its format is kept, a decode consumes its input and makes no sample
        a.x[0] = 48000, a.x[1] = 2, a.x[2] = 0;
        rt_call(&a, g_audio[A_OPUS_WORK]);
        CHECK(a.x[0] == 0x9000);
        a.x[0] = HEAP + 0x400, a.x[1] = 32000, a.x[2] = 1, a.x[3] = 0, a.x[4] = 0, a.x[5] = 0;
        rt_call(&a, g_audio[A_OPUS_INIT]);
        CHECK(a.x[0] == 0);
        a.x[0] = HEAP + 0x400;
        rt_call(&a, g_audio[A_OPUS_RATE]);
        CHECK(a.x[0] == 32000);
        a.x[0] = HEAP + 0x400;
        rt_call(&a, g_audio[A_OPUS_CH]);
        CHECK(a.x[0] == 1);
        memset((void *)(uintptr_t)(HEAP + 0x500), 0xff, 16);
        a.x[0] = HEAP + 0x400, a.x[1] = HEAP + 0x500, a.x[2] = HEAP + 0x508, a.x[3] = HEAP + 0x600, a.x[4] = 0x1000;
        a.x[5] = HEAP + 0x700, a.x[6] = 123;
        rt_call(&a, g_audio[A_OPUS_DECODE]);
        CHECK(a.x[0] == 0);
        uint64_t consumed;
        uint32_t samples;
        memcpy(&consumed, (const void *)(uintptr_t)(HEAP + 0x500), 8);
        memcpy(&samples, (const void *)(uintptr_t)(HEAP + 0x508), 4);
        CHECK(consumed == 123 && samples == 0);
        // the renderer's other calls are accepted; the voice drop parameter is kept for the getter
        CHECK(acall(&a, A_START, 0x41554431ull, 0) == 0);
        CHECK(acall(&a, A_STOP, 0x41554431ull, 0) == 0);
        CHECK(acall(&a, A_UPDATE, 0x41554431ull, HEAP + 0x800) == 0);
        CHECK(acall(&a, A_EXECUTE, 0x41554431ull, 0) == 0);
        CHECK(acall(&a, A_CLOSE, 0x41554431ull, 0) == 0);
        float quarter = 0.25f;
        a.v[0] = 0;
        memcpy(&a.v[0], &quarter, sizeof quarter);
        CHECK(acall(&a, A_SET_DROP, 0x41554431ull, 0) == 0);
        rt_call(&a, g_audio[A_DROP]);
        memcpy(&drop, &a.v[0], sizeof drop);
        CHECK(drop == 0.25f);
        // the output device: MainAudioOut, unit volume, stereo, no virtual input and no input device
        CHECK(acall(&a, A_LIST_VIRTUAL, HEAP + 0x200, 4) == 0);
        a.v[0] = 0;
        acall(&a, A_GET_VOLUME, HEAP + 0x200, 0);   // the volume is a float in v0
        memcpy(&drop, &a.v[0], sizeof drop);
        CHECK(drop == 1.0f);
        CHECK(acall(&a, A_SET_VOLUME, HEAP + 0x200, 0) == 0);
        memset((void *)(uintptr_t)(HEAP + 0x200), 0x7f, 64);
        CHECK(acall(&a, A_TRUE_RAW, HEAP + 0x200, 0) == 0);
        CHECK(((const uint8_t *)(uintptr_t)(HEAP + 0x200))[0] == 0x7f);   // no input device: no name
        CHECK(acall(&a, A_VIRTUAL_PROPS, HEAP + 0x200, HEAP + 0x240) == 0);
        CHECK(((const uint8_t *)(uintptr_t)(HEAP + 0x200))[0] == 0x7f);   // the properties are left alone
        static const int NAMED[] = {A_DEFAULT_OUT, A_ACTIVE_OUT};
        for (unsigned i = 0; i < sizeof NAMED / sizeof *NAMED; i++) {
            memset((void *)(uintptr_t)(HEAP + 0x200), 0x7f, 64);
            CHECK(acall(&a, NAMED[i], HEAP + 0x200, 0) == 0);
            CHECK(!strcmp((const char *)(uintptr_t)(HEAP + 0x200), "MainAudioOut"));
            CHECK(((const uint8_t *)(uintptr_t)(HEAP + 0x200))[13] == 0x7f);
            CHECK(acall(&a, NAMED[i], 0, 0) == 0);   // no place for the name: nothing is written
        }
        CHECK(acall(&a, A_ACQUIRE, 0, 0) == 0);   // no event to bind
        CHECK(acall(&a, A_RELEASE, HEAP + 0x400, 0) == 0);
        CHECK(acall(&a, A_AUTOTUNE_GET, 0, 0) == 0);
        CHECK(acall(&a, A_AUTOTUNE_SET, 1, 0) == 0);
        CHECK(acall(&a, A_CHANNELS, 0, 0) == 2);
        CHECK(acall(&a, A_OPUS_RESET, HEAP + 0x400, 0) == 0);
        CHECK(acall(&a, A_OPUS_FINALIZE, HEAP + 0x400, 0) == 0);
        // a stand-in says so in the census: every override of the file is a stub with its reason
        for (int i = 0; i < A_COUNT; i++) {
            const HosNote *n = hos_note(AUDIO_SYMBOLS[i]);
            CHECK(n && !strcmp(n->status, HOS_STUB) && strstr(n->why, "stand-in"));
        }
    } else if (!strcmp(g_case, "audio_events")) {
        boot(dir);
        rt_kernel_run(G(3), 0);
    } else if (!strcmp(g_case, "service_census")) {
        boot(dir);
        rt_kernel_run(G(2), 0);
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        CHECK(!strncmp(t, "#census-mode OOT2_CENSUS=1", 26));   // marked first
        CHECK(strstr(t, "\nservice\tlm\tcensus-gap\t2\t"));   // met twice, one row
        CHECK(strstr(t, "\nservice\tsvc00039\tcensus-gap\t1\t"));   // every name has its row, listed in the log or not
        CHECK(strstr(t, "\nkind\tkey\tstatus\tcalls\tfirst_at\twhy\n"));
    } else if (!strcmp(g_case, "nvn_census")) {
        // census mode: an NVN2 entry point with no body is passed and the census row says so
        boot(dir);
        rt_kernel_run(G(2), 0);
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        char want[400];
        snprintf(want, sizeof want, "\nnvn\t%s\tcensus-gap\t1\t0x7000000250 (main+0x250)\t[census mode] an NVN2 entry point "
                 "with no body: the run answered 0 and went on\n", g_nvn_name);
        CHECK(strstr(t, want));
        CHECK(!strstr(t, "named-fatal\t1\t0x7000000250"));
        CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n"));
    } else if (!strcmp(g_case, "census_files")) {
        // census.tsv in every mode carries the front end's own counters; census mode also keeps census-nvn.tsv and
        // rewrites both while the run goes on (rt_census_flush), never after the run's own end has been written
        boot(dir);
        char path[1100];
        snprintf(path, sizeof path, "%s/census-nvn.tsv", dir);
        // one recording through the entry points (device, command buffer with control memory, begin, end with a handle,
        // end with nothing open): the front end's counters the census carries are not all zero
        Cpu a;
        memset(&a, 0, sizeof a);
        a.excl_addr = ~0ull;
        const uint64_t b = HEAP + 0x1000, d = HEAP + 0x2000, o = HEAP + 0x6000;
        nvn(&a, "nvnDeviceBuilderSetDefaults", b, 0, 0);
        CHECK((nvn(&a, "nvnDeviceInitialize", d, b, 0) & 0xff) == 1);
        CHECK((nvn(&a, "nvnCommandBufferInitialize", o, d, 0) & 0xff) == 1);
        nvn(&a, "nvnCommandBufferAddControlMemory", o, HEAP + 0x8000, 0x1000);
        nvn(&a, "nvnCommandBufferBeginRecording", o, 0, 0);
        CHECK(nvn(&a, "nvnCommandBufferEndRecording", o, 0, 0) != 0);
        CHECK(nvn(&a, "nvnCommandBufferEndRecording", o, 0, 0) == 0);
        if (rt_census_on()) {
            rt_census_flush();
            const char *t = read_census(dir);
            CHECK(!strncmp(t, "#census-mode OOT2_CENSUS=1", 26));
            CHECK(strstr(t, "\nstop\trunning\tcensus\t1\t\t[census mode] the run was still going when this was written\n"));
            FILE *f = fopen(path, "rb");
            CHECK(f != NULL);
            char head[256] = {0};
            CHECK(fread(head, 1, sizeof head - 1, f) > 0);
            fclose(f);
            CHECK(!strncmp(head, "#census-mode OOT2_CENSUS=1: not a default run\n#presents 0\n", 56));
        }
        hos_census_write("the case ended");
        const char *t = read_census(dir);
        CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n"));
        CHECK(strstr(t, "\nnvn-stat\trecordings\timplemented\t1\t\tinline 0, slow 1, ended with a handle 1, ended empty 1\n"));
        CHECK(strstr(t, "\nnvn-stat\tcallbacks\timplemented\t0\t\tmemory callbacks: control 0, command 0; recordings begun "
                        "over left-out records 0\n"));
        CHECK(strstr(t, "\nnvn-stat\twaitevents\timplemented\t0\t\tWaitEvents met 0, unsatisfied at completion 0\n"));
        CHECK(strstr(t, "\nnvn-stat\twaitclasses\timplemented\t0\t\tby a GPU signal in order 0, by a CPU signal before 0, "
                        "already satisfied 0, unsatisfied 0 (later by GPU 0, later by CPU 0, not satisfied by the end 0, "
                        "dropped from the full list 0)\n"));
        CHECK(strstr(t, "\nnvn-stat\tpools\timplemented\t0\t\tmemory pools made 0, live 0 (most at once 0), 0 bytes live "
                        "(most 0); made with GPU_NO_ACCESS 0; GPU address window in use 0 bytes (most 0 of 68719476736)\n"));
        CHECK(strstr(t, "\nnvn-stat\tpools-virtual\timplemented\t0\t\tmade 0, live 0 (most at once 0), 0 bytes live "
                        "(most 0)\n"));
        CHECK(strstr(t, "\nnvn-stat\tpresents\timplemented\t0\t\tnvnQueuePresentTexture calls that reached the back end\n"));
        if (rt_census_on()) {
            rt_census_flush();   // the run's end is written: a later flush leaves it alone
            t = read_census(dir);
            CHECK(strstr(t, "\nstop\texit 0\timplemented\t1\t\tthe case ended\n") && !strstr(t, "\nstop\trunning"));
        } else {
            CHECK(t[0] != '#');   // no census-mode mark in a default run
            FILE *f = fopen(path, "rb");
            CHECK(f == NULL);     // and no calls-per-frame file
        }
    } else if (!strcmp(g_case, "census_end")) {
        // a census-mode run ends with its own exit code after the census has its stop row
        boot(dir);
        rt_census_end(RT_EXIT_CENSUS, "[census mode] the case ended at its frame limit");
    } else if (!strcmp(g_case, "service") || !strcmp(g_case, "port") || !strcmp(g_case, "command") ||
               !strcmp(g_case, "domain") || !strcmp(g_case, "nvn") || !strcmp(g_case, "movie") ||
               !strcmp(g_case, "fault") || !strcmp(g_case, "fault_chain") || !strcmp(g_case, "audio_open_binds") || !strcmp(g_case, "audio_open_null") ||
               !strcmp(g_case, "audio_notification_binds") || !strcmp(g_case, "audio_second")) {
        boot(dir);
        rt_kernel_run(G(2), 0);
    } else if (!strcmp(g_case, "boot")) {
        boot(dir);
        rt_kernel_run(hos_boot_entry(), 0);
    } else {
        fprintf(stderr, "unknown case %s\n", g_case);
        return 2;
    }
    printf("ok %s\n", g_case);
    return 0;
}
