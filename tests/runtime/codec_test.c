// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract test of the video decoder service under the SDK's movie library (runtime/hos/ov_codec.c, the movie player), with a
// fake AV1 decoder (av1_fake.h) behind the seam of runtime/video. Synthetic guest threads written by hand call the
// service's nn::codec entry points as the SDK would, by their mangled names, and wait on its events with the kernel's
// own WaitSynchronization. No game code, no SDK and no decoder is involved.
//
//   codec_test CASE   runs one case; prints "ok CASE" and exits 0, or exits 1 with a FAIL line.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "endmark.h"
#include "layout.h"
#include "oot2/kernel.h"
#include "oot2/memory.h"
#include "plat.h"
#include "av1_fake.h"
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
#define R_OK 0u
#define R_BUSY (0x6fu | 3603u << 9)
#define R_NONE (0x6fu | 3604u << 9)
#define R_TIMEOUT 0xEA01u
#define MS 1000000ull

#define SYM(name) "_ZN2nn5codec" name
static const char *const SYMBOLS[] = {
    SYM("25MakeDecodeImagePreferenceEttNS0_24VideoDecoderOutputFormatENS0_15ImageTilingModeE"),
    SYM("31CalculateDecoderImagePoolBufferEPmS1_RKNS0_27VideoDecoderImagePreferenceEj"),
    SYM("26InitializeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeEPvmRKNS0_27VideoDecoderImagePreferenceEj"),
    SYM("24FinalizeDecoderImagePoolEPNS0_25VideoDecoderImagePoolTypeE"),
    SYM("26ReadDecoderOutputImageNv12EPKNS0_25VideoDecoderImagePoolTypeEjttttPvmmS4_mm"),
    SYM("34InitializeAv1DecoderByRemoteMemoryEPNS0_14Av1DecoderTypeEmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE"),
    SYM("36InitializeAv1DecoderByTransferMemoryEPNS0_14Av1DecoderTypeEPvmmttjNS0_23VideoDecoderInputFormatENS0_24VideoDecoderOutputFormatENS0_22VideoDecoderOutputModeE"),
    SYM("18FinalizeAv1DecoderEPNS0_14Av1DecoderTypeE"),
    SYM("24RegisterAv1DecoderBufferEPNS0_14Av1DecoderTypeEjPNS0_25VideoDecoderImagePoolTypeEj"),
    SYM("26UnregisterAv1DecoderBufferEPNS0_14Av1DecoderTypeEj"),
    SYM("15StartAv1DecoderEPNS0_14Av1DecoderTypeE"),
    SYM("14StopAv1DecoderEPNS0_14Av1DecoderTypeE"),
    SYM("15FlushAv1DecoderEPNS0_14Av1DecoderTypeE"),
    SYM("33GetAv1DecoderSubmissionReadyEventEPNS0_14Av1DecoderTypeE"),
    SYM("27SubmitAv1DecoderInputBufferEPmPNS0_14Av1DecoderTypeEPKvmNS_8TimeSpanEb"),
    SYM("29AcquireAv1DecoderOutputBufferEPjPNS_2gs16NativeSyncHandleEPNS0_14Av1DecoderTypeE"),
    SYM("29ReleaseAv1DecoderOutputBufferEPNS0_14Av1DecoderTypeEjNS_2gs16NativeSyncHandleE"),
    SYM("35GetAv1DecoderOutputBufferPropertiesEPtS1_S1_S1_PNS_8TimeSpanEPbPNS0_14Av1DecoderTypeEj"),
    "_ZN2nn2gs17TryWaitNativeSyncERKNS0_16NativeSyncHandleE",
    "_ZN2nn5codec5video10InitializeEv",
    "_ZN2nn6syncpt3sdk20CreateSyncPointEventEPNS1_18SyncPointEventTypeE",
    "_ZN2nn6syncpt3sdk25InitializeMultiWaitHolderEPNS_2os19MultiWaitHolderTypeEPNS1_18SyncPointEventTypeE",
    "_ZN2nn6syncpt3sdk21RequestSyncPointEventEPNS1_18SyncPointEventTypeERKNS1_15SyncPointHandleE",
    "_ZN2nn6syncpt3sdk20CancelSyncPointEventEPNS1_18SyncPointEventTypeE",
    "_ZN2nn6syncpt3sdk21DestroySyncPointEventEPNS1_18SyncPointEventTypeE",
};
enum { F_PREF, F_POOLSIZE, F_POOLINIT, F_POOLFINI, F_READ, F_INIT_REMOTE, F_INIT_TRANSFER, F_FINI, F_REGISTER, F_UNREGISTER,
       F_START, F_STOP, F_FLUSH, F_EVENT, F_SUBMIT, F_ACQUIRE, F_RELEASE, F_PROPS, F_TRYWAIT, F_VIDEO_INIT, F_SPT_CREATE,
       F_SPT_HOLDER, F_SPT_REQUEST, F_SPT_CANCEL, F_SPT_DESTROY, F_COUNT };
static uint64_t g_fn[F_COUNT];
static uint64_t HEAP;
static const char *g_case;

// the event binder the service is given in place of the SDK's attach function: a kernel event for each system event
static struct { uint64_t ev; uint32_t writable, readable; int autoclear; } g_events[16];
static int g_nevents;
static uint32_t test_bind(Cpu *c, uint64_t ev, int autoclear) {
    (void)c;
    for (int i = 0; i < g_nevents; i++)
        if (g_events[i].ev == ev) return g_events[i].writable;   // as hos_bind_system_event: the same event gives the same end
    CHECK(g_nevents < 16);
    uint32_t readable, w = rt_kernel_event_create(&readable);
    g_events[g_nevents].ev = ev;
    g_events[g_nevents].writable = w;
    g_events[g_nevents].autoclear = autoclear;
    g_events[g_nevents++].readable = readable;
    return w;
}
static uint32_t readable_of(uint64_t ev) {
    for (int i = 0; i < g_nevents; i++)
        if (g_events[i].ev == ev) return g_events[i].readable;
    CHECK(!"an event the service never bound");
    return 0;
}

// the two nn::os calls the sync point events make, which the synthetic module has no SDK for
static uint64_t g_holder, g_holder_ev;
static void test_clear(Cpu *c, uint64_t ev) {
    c->x[0] = readable_of(ev), c->x[1] = c->x[2] = c->x[3] = 0;
    rt_svc(c, 0x12, PC);   // ClearEvent
}
static void test_holder(Cpu *c, uint64_t holder, uint64_t ev) {
    (void)c;
    g_holder = holder;
    g_holder_ev = ev;
}

// ---- guest helpers
static inline uint64_t sv(Cpu *c, uint32_t imm, uint64_t a0, uint64_t a1, uint64_t a2, uint64_t a3) {
    c->x[0] = a0, c->x[1] = a1, c->x[2] = a2, c->x[3] = a3;
    rt_svc(c, imm, PC);
    return c->x[0];
}

// A service entry called as the SDK would: n integer arguments, the ninth and later on the stack. The arguments are
// converted to 64 bits by an array, so an int literal is not read through varargs as a wider type.
#define call(c, fn, n, ...) call_((c), (fn), (n), (const uint64_t[]){0, ##__VA_ARGS__} + 1)
static uint64_t call_(Cpu *c, int fn, int n, const uint64_t *args) {
    uint64_t stack[8] = {0};
    for (int i = 0; i < n; i++) {
        if (i < 8) c->x[i] = args[i];
        else stack[i - 8] = args[i];
    }
    uint64_t saved = c->sp;
    c->sp = (saved - 128) & ~15ull;
    memcpy((void *)(uintptr_t)c->sp, stack, sizeof stack);
    rt_call(c, g_fn[fn]);
    c->sp = saved;
    return c->x[0];
}

// WaitSynchronization on handles: the index of the one signalled (x1), or R_TIMEOUT. A signalled auto-clear event is cleared
// by the waiter, as the SDK does.
static uint64_t wait_any(Cpu *c, const uint32_t *h, int n, uint64_t ns, int *index) {
    memcpy((void *)(uintptr_t)(HEAP + 0x380), h, (size_t)n * 4);
    uint64_t r = sv(c, 0x18, 0, HEAP + 0x380, (uint64_t)n, ns);
    if (r == 0) {
        *index = (int)c->x[1];
        CHECK(sv(c, 0x12, h[*index], 0, 0, 0) == 0);
    }
    return r;
}
static uint64_t wait_event(Cpu *c, uint64_t ev, uint64_t ns) {
    uint32_t h = readable_of(ev);
    int i;
    return wait_any(c, &h, 1, ns, &i);
}

// ---- the decoder, its pool and buffers
#define DEC(i) (HEAP + 0x1000 + 0x100ull * (i))     // the guest's Av1DecoderType
#define POOL(i) (HEAP + 0x2000 + 0x100ull * (i))    // the guest's VideoDecoderImagePoolType
#define PREF (HEAP + 0x3000)
#define SYNC(i) (HEAP + 0x3100 + 0x10ull * (i))     // a NativeSyncHandle the service wrote
#define UNIT (HEAP + 0x4000)                        // the compressed bytes the guest submits
#define OUT (HEAP + 0x10000)                        // the planes the guest reads a picture into

static uint64_t g_ev[4];   // the submission-ready event of each decoder

static void make_decoder(Cpu *c, int i, int w, int h) {   // the decoder, its event; the pool comes with make_buffers
    CHECK(call(c, F_INIT_REMOTE, 8, DEC(i), 0x1000, (uint64_t)w, (uint64_t)h, 8, 0, 0, 0) == R_OK);
    g_ev[i] = call(c, F_EVENT, 1, DEC(i));
    CHECK(g_ev[i] != 0);
}
// A pool of count images and the buffers 0..count-1 of decoder i registered over it
static void make_buffers(Cpu *c, int i, int w, int h, int count) {
    c->x[8] = PREF;
    c->x[0] = (uint64_t)w, c->x[1] = (uint64_t)h, c->x[2] = 3, c->x[3] = 1;
    rt_call(c, g_fn[F_PREF]);
    uint64_t size = 0, align = 0;
    CHECK(call(c, F_POOLSIZE, 4, HEAP + 0x3200, HEAP + 0x3208, PREF, (uint64_t)count) == R_OK);
    memcpy(&size, (void *)(uintptr_t)(HEAP + 0x3200), 8);
    memcpy(&align, (void *)(uintptr_t)(HEAP + 0x3208), 8);
    CHECK(size >= 0x1000 && align == 0x1000);
    CHECK(call(c, F_POOLINIT, 5, POOL(i), HEAP + 0x20000, size, PREF, (uint64_t)count) == R_OK);
    for (int b = 0; b < count; b++) CHECK(call(c, F_REGISTER, 4, DEC(i), (uint64_t)b, POOL(i), (uint64_t)b) == R_OK);
}
static void put_unit(uint8_t tag, size_t size) {
    uint8_t *u = (uint8_t *)(uintptr_t)UNIT;
    memset(u, 0xA5, size);
    u[0] = tag;
}
static uint64_t submit(Cpu *c, int i, uint8_t tag, int64_t ts, int eos) {
    put_unit(tag, 64);
    return call(c, F_SUBMIT, 6, HEAP + 0x3300, DEC(i), UNIT, 64, (uint64_t)ts, (uint64_t)eos);
}
// Acquire: the index, the handle in SYNC(slot)
static uint64_t acquire(Cpu *c, int i, int slot, uint32_t *idx) {
    uint64_t r = call(c, F_ACQUIRE, 3, HEAP + 0x3340, SYNC(slot), DEC(i));
    if (r == R_OK) memcpy(idx, (void *)(uintptr_t)(HEAP + 0x3340), 4);
    return r;
}
static void release(Cpu *c, int i, uint32_t idx, int slot) {
    uint64_t h[2];
    memcpy(h, (void *)(uintptr_t)SYNC(slot), 16);
    call(c, F_RELEASE, 4, DEC(i), idx, h[0], h[1]);
}
static int reached(Cpu *c, int slot) { return (int)call(c, F_TRYWAIT, 1, SYNC(slot)) & 1; }
typedef struct { uint16_t w, h, cw, ch; int64_t ts; uint8_t eos; } Props;
static uint64_t props(Cpu *c, int i, uint32_t idx, Props *p) {
    uint64_t r = call(c, F_PROPS, 8, HEAP + 0x3380, HEAP + 0x3382, HEAP + 0x3384, HEAP + 0x3386, HEAP + 0x3388, HEAP + 0x3390,
                      DEC(i), (uint64_t)idx);
    if (r == R_OK) {
        memcpy(&p->w, (void *)(uintptr_t)(HEAP + 0x3380), 2);
        memcpy(&p->h, (void *)(uintptr_t)(HEAP + 0x3382), 2);
        memcpy(&p->cw, (void *)(uintptr_t)(HEAP + 0x3384), 2);
        memcpy(&p->ch, (void *)(uintptr_t)(HEAP + 0x3386), 2);
        memcpy(&p->ts, (void *)(uintptr_t)(HEAP + 0x3388), 8);
        memcpy(&p->eos, (void *)(uintptr_t)(HEAP + 0x3390), 1);
    }
    return r;
}
// Poll until the buffer holds its picture
static int wait_filled(Cpu *c, int i, uint32_t idx, uint64_t ms, Props *p) {
    for (uint64_t t = 0; t <= ms; t++) {
        if (props(c, i, idx, p) == R_OK) return 1;
        plat_sleep_ns(MS);
    }
    return 0;
}
// Read the picture of pool image idx (width w, height h) into OUT with chosen strides, and check every byte against the unit
// tag it must show; the padding of each row stays what the guest wrote there
static void check_picture(Cpu *c, int pool, uint32_t idx, int w, int h, int tag, int ystride, int uvstride) {
    int cw = (w + 1) / 2, ch = (h + 1) / 2;
    uint8_t *yd = (uint8_t *)(uintptr_t)OUT, *uvd = (uint8_t *)(uintptr_t)(OUT + 0x8000);
    memset(yd, 0xEE, (size_t)ystride * h);
    memset(uvd, 0xEE, (size_t)uvstride * ch);
    CHECK(call(c, F_READ, 12, POOL(pool), (uint64_t)idx, 0, 0, (uint64_t)w, (uint64_t)h, OUT, (uint64_t)ystride,
               (uint64_t)ystride * h, OUT + 0x8000, (uint64_t)uvstride, (uint64_t)uvstride * ch) == R_OK);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) CHECK(yd[y * ystride + x] == FAKE_Y(tag, x, y));
        for (int x = w; x < ystride; x++) CHECK(yd[y * ystride + x] == 0xEE);
    }
    for (int y = 0; y < ch; y++) {
        for (int x = 0; x < cw; x++) {
            CHECK(uvd[y * uvstride + 2 * x] == FAKE_U(tag, x, y));       // U first, then V, interleaved
            CHECK(uvd[y * uvstride + 2 * x + 1] == FAKE_V(tag, x, y));
        }
        for (int x = 2 * cw; x < uvstride; x++) CHECK(uvd[y * uvstride + x] == 0xEE);
    }
}

// ---- the cases (guest programs)
// The submission-ready event of a decoder is its own: it fires when the decoder has room and not before, and a WaitAny over
// the events of two decoders returns the one that fired.
static void case_events(Cpu *c) {
    make_decoder(c, 0, 64, 64);
    make_decoder(c, 1, 64, 64);
    CHECK(g_ev[0] != g_ev[1]);
    for (int i = 0; i < g_nevents; i++) CHECK(g_events[i].autoclear == 1);   // the submission-ready events are auto-clear
    CHECK(wait_event(c, g_ev[0], 50 * MS) == R_TIMEOUT);   // not started: no room signalled
    CHECK(call(c, F_START, 1, DEC(1)) == R_OK);
    uint32_t h[2] = {readable_of(g_ev[0]), readable_of(g_ev[1])};
    int idx = -1;
    CHECK(wait_any(c, h, 2, 1000 * MS, &idx) == 0);
    CHECK(idx == 1);                                       // decoder 1 started, decoder 0 did not
    CHECK(wait_any(c, h, 2, 50 * MS, &idx) == R_TIMEOUT);  // consumed: auto-clear
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    CHECK(wait_any(c, h, 2, 1000 * MS, &idx) == 0 && idx == 0);
    CHECK(wait_any(c, h, 2, 50 * MS, &idx) == R_TIMEOUT);
    // a WaitAny with two other holders: the event fires among foreign ones
    uint32_t other_r, other_w = rt_kernel_event_create(&other_r);
    uint32_t h3[2] = {other_r, readable_of(g_ev[1])};
    (void)other_w;
    CHECK(submit(c, 1, 1, 0, 0) == R_OK);                  // a unit taken, room left: the event again
    CHECK(wait_any(c, h3, 2, 1000 * MS, &idx) == 0 && idx == 1);
    // the service's event is a Horizon event the SDK can wait on again after Stop
    CHECK(call(c, F_STOP, 1, DEC(0)) == R_OK);
    CHECK(wait_any(c, h, 2, 1000 * MS, &idx) == 0 && idx == 0);
}

// DecodeAsync's contract at the codec level: Submit returns at once, whatever the decode costs, and the decode runs on the
// service's worker and not on the caller.
static void case_submit_async(Cpu *c) {
    g_fake_delay_ns = 600 * MS;
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 64, 64, 2);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t i0, i1;
    CHECK(acquire(c, 0, 0, &i0) == R_OK && acquire(c, 0, 1, &i1) == R_OK);
    release(c, 0, i0, 0);
    release(c, 0, i1, 1);
    uint64_t t0 = plat_now_ns();
    for (int k = 0; k < 3; k++) CHECK(submit(c, 0, (uint8_t)(10 + k), k, 0) == R_OK);
    uint64_t took = plat_now_ns() - t0;
    CHECK(took < 400 * MS);                                // three submits of a 600 ms decode each, in less time than one decode
    Props p;
    CHECK(wait_filled(c, 0, i0, 5000, &p));
    CHECK(g_fake_threads_differ == 1);                     // the decode ran on another thread than the one that opened it
    CHECK(g_fake_sends >= 1);
}

// A buffer is not given to the decoder again before it is released: both buffers acquired, nothing to acquire; pictures go to
// the buffers in release order, a third picture waits for a release, and a picture is not overwritten while its buffer is held.
static void case_no_reuse(Cpu *c) {
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 2);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a, b, n;
    CHECK(acquire(c, 0, 0, &a) == R_OK && acquire(c, 0, 1, &b) == R_OK);
    CHECK(a != b);
    CHECK(acquire(c, 0, 2, &n) == R_NONE);                 // both are out: nothing to acquire, no frame reused
    release(c, 0, a, 0);
    release(c, 0, b, 1);
    CHECK(acquire(c, 0, 2, &n) == R_OK && n == a);         // back in the queue at once, in release order
    CHECK(acquire(c, 0, 3, &n) == R_OK && n == b);
    // a handle of a release is not reached before its picture is in the buffer
    CHECK(!reached(c, 2) && !reached(c, 3));
    CHECK(submit(c, 0, 20, 100, 0) == R_OK);
    CHECK(submit(c, 0, 21, 200, 0) == R_OK);
    CHECK(submit(c, 0, 22, 300, 0) == R_OK);               // the third has no released buffer to go into
    Props p;
    CHECK(wait_filled(c, 0, a, 2000, &p) && p.ts == 100 && wait_filled(c, 0, b, 2000, &p) && p.ts == 200);
    CHECK(reached(c, 2) && reached(c, 3));
    plat_sleep_ns(100 * MS);                               // time for the worker to overwrite a held picture, if it would
    check_picture(c, 0, a, 6, 4, 20, 6, 6);
    check_picture(c, 0, b, 6, 4, 21, 6, 6);
    CHECK(props(c, 0, a, &p) == R_OK && p.ts == 100 && props(c, 0, b, &p) == R_OK && p.ts == 200);
    // releasing a buffer gives the waiting picture to it, and only then
    release(c, 0, a, 2);                                   // buffer a is released: the waiting picture goes into it
    CHECK(acquire(c, 0, 4, &n) == R_OK && n == a);
    CHECK(wait_filled(c, 0, a, 2000, &p) && p.ts == 300);
    check_picture(c, 0, a, 6, 4, 22, 6, 6);
}

// Waits until the fake decoder has taken n units (the worker is then inside the n-th decode), polling instead of a
// fixed sleep so a loaded machine cannot run the check before the worker starts; gives up after ms milliseconds.
static int wait_sends(int n, int ms) {
    for (int i = 0; i < ms && g_fake_sends < n; i++) plat_sleep_ns(MS);
    return g_fake_sends >= n;
}

// Stop and Flush, units in flight: nothing sent before the call reaches a buffer, the units queued behind the one being
// decoded are dropped and never decoded, the handles taken before the call and after it stay not reached (their buffers
// hold no picture), buffers stay reserved, and the decoder takes a new stream after.
static void stop_or_flush(Cpu *c, int fn) {
    g_fake_delay_ns = 150 * MS;
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 2);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a, b;
    CHECK(acquire(c, 0, 0, &a) == R_OK && acquire(c, 0, 1, &b) == R_OK);
    release(c, 0, a, 0);
    release(c, 0, b, 1);
    uint32_t n;
    CHECK(acquire(c, 0, 2, &n) == R_OK && n == a && acquire(c, 0, 3, &n) == R_OK && n == b);   // handles taken before the call
    CHECK(!reached(c, 2) && !reached(c, 3));
    for (int k = 0; k < 4; k++) CHECK(submit(c, 0, (uint8_t)(30 + k), k, 0) == R_OK);
    CHECK(wait_sends(1, 2000));                            // the worker is inside the first unit's decode
    CHECK(call(c, fn, 1, DEC(0)) == R_OK);
    plat_sleep_ns(900 * MS);                               // far past every decode in flight
    Props p;
    CHECK(props(c, 0, a, &p) == R_NONE && props(c, 0, b, &p) == R_NONE);   // no frame after it
    CHECK(!reached(c, 2) && !reached(c, 3));               // the handles taken before the call are still not reached
    CHECK(g_fake_sends == 1);                              // the three units queued behind the first were dropped, not decoded
    CHECK(g_fake_flushes >= 1);                            // the decoder was told to start a new stream
    // the library releases what it holds: the buffers are still reserved, and each waits for one picture
    release(c, 0, a, 2);
    release(c, 0, b, 3);
    CHECK(acquire(c, 0, 4, &n) == R_OK && n == a && acquire(c, 0, 5, &n) == R_OK && n == b);
    CHECK(!reached(c, 4) && !reached(c, 5));
    // a new stream: three units sent after the call go to the buffers in release order, and the third waits for a release
    // (a buffer that was released twice across the call waits once, so it is not filled twice)
    g_fake_delay_ns = 0;
    CHECK(submit(c, 0, 99, 5000, 0) == R_OK);
    CHECK(submit(c, 0, 98, 5001, 0) == R_OK);
    CHECK(submit(c, 0, 97, 5002, 0) == R_OK);
    CHECK(wait_filled(c, 0, a, 2000, &p) && p.ts == 5000);
    CHECK(wait_filled(c, 0, b, 2000, &p) && p.ts == 5001);
    plat_sleep_ns(100 * MS);                               // time for the third picture to overwrite one of them, if it would
    CHECK(props(c, 0, a, &p) == R_OK && p.ts == 5000 && props(c, 0, b, &p) == R_OK && p.ts == 5001);
    check_picture(c, 0, a, 6, 4, 99, 6, 6);
    check_picture(c, 0, b, 6, 4, 98, 6, 6);
    CHECK(reached(c, 4) && reached(c, 5));
}
static void case_stop(Cpu *c) { stop_or_flush(c, F_STOP); }
static void case_flush(Cpu *c) { stop_or_flush(c, F_FLUSH); }

// Stop and Flush, pictures already decoded: one is in a buffer and one waits for a buffer when the call comes. The one in a
// buffer stays there, readable, with its handle reached, until the library releases that buffer; the one that waited is
// dropped, so the release delivers nothing; the decoder takes a new stream after.
static void stop_or_flush_after(Cpu *c, int fn) {
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 2);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a, b, n;
    CHECK(acquire(c, 0, 0, &a) == R_OK && acquire(c, 0, 1, &b) == R_OK);
    release(c, 0, a, 0);
    release(c, 0, b, 1);
    CHECK(acquire(c, 0, 2, &n) == R_OK && n == a && acquire(c, 0, 3, &n) == R_OK && n == b);
    CHECK(submit(c, 0, 31, 1, 0) == R_OK);
    CHECK(submit(c, 0, 32, 2, 0) == R_OK);
    CHECK(submit(c, 0, 33, 3, 0) == R_OK);                 // no released buffer is left for this one: it waits
    Props p;
    CHECK(wait_filled(c, 0, a, 2000, &p) && wait_filled(c, 0, b, 2000, &p));
    plat_sleep_ns(100 * MS);                               // the third is decoded and held
    CHECK(reached(c, 2) && reached(c, 3));
    CHECK(call(c, fn, 1, DEC(0)) == R_OK);
    plat_sleep_ns(60 * MS);
    CHECK(props(c, 0, a, &p) == R_OK && p.ts == 1 && props(c, 0, b, &p) == R_OK && p.ts == 2);   // stay until released
    CHECK(reached(c, 2) && reached(c, 3));
    check_picture(c, 0, a, 6, 4, 31, 6, 6);
    check_picture(c, 0, b, 6, 4, 32, 6, 6);
    release(c, 0, a, 2);                                   // the picture that waited is not delivered by this
    plat_sleep_ns(100 * MS);
    CHECK(props(c, 0, a, &p) == R_NONE);
    CHECK(acquire(c, 0, 4, &n) == R_OK && n == a);
    CHECK(!reached(c, 4));
    CHECK(props(c, 0, b, &p) == R_OK && p.ts == 2);        // and the other buffer is untouched
    check_picture(c, 0, b, 6, 4, 32, 6, 6);
    CHECK(submit(c, 0, 34, 7000, 0) == R_OK);              // a new stream goes into the released buffer
    CHECK(wait_filled(c, 0, a, 2000, &p) && p.ts == 7000);
    check_picture(c, 0, a, 6, 4, 34, 6, 6);
    CHECK(reached(c, 4));
}
static void case_stop_after(Cpu *c) { stop_or_flush_after(c, F_STOP); }
static void case_flush_after(Cpu *c) { stop_or_flush_after(c, F_FLUSH); }

// The planes and strides: odd sizes, the strides the guest asks for (wider than the picture), U and V interleaved after the luma
static void case_planes(Cpu *c) {
    g_fake_w = 7;
    g_fake_h = 5;
    make_decoder(c, 0, 7, 5);
    make_buffers(c, 0, 7, 5, 1);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a;
    CHECK(acquire(c, 0, 0, &a) == R_OK);
    release(c, 0, a, 0);
    CHECK(submit(c, 0, 77, 7, 0) == R_OK);
    Props p;
    CHECK(wait_filled(c, 0, a, 2000, &p));
    CHECK(p.w == 7 && p.h == 5 && p.cw == 4 && p.ch == 3 && p.ts == 7 && p.eos == 0);
    check_picture(c, 0, a, 7, 5, 77, 7, 8);                // tight luma rows and the minimum chroma rows
    check_picture(c, 0, a, 7, 5, 77, 12, 13);              // padded rows: the padding is untouched
    // a region larger than the picture: the picture where it is, black (luma 16, chroma 128) outside it
    uint8_t *yd = (uint8_t *)(uintptr_t)OUT, *uvd = (uint8_t *)(uintptr_t)(OUT + 0x8000);
    memset(yd, 0xEE, 128);
    memset(uvd, 0xEE, 128);
    CHECK(call(c, F_READ, 12, POOL(0), (uint64_t)a, 0, 0, 10, 8, OUT, 10, 80, OUT + 0x8000, 10, 40) == R_OK);
    for (int y = 0; y < 8; y++)
        for (int x = 0; x < 10; x++) CHECK(yd[y * 10 + x] == (x < 7 && y < 5 ? FAKE_Y(77, x, y) : 16));
    for (int y = 0; y < 4; y++)
        for (int x = 0; x < 5; x++) {
            int in = x < 4 && y < 3;
            CHECK(uvd[y * 10 + 2 * x] == (in ? FAKE_U(77, x, y) : 128));
            CHECK(uvd[y * 10 + 2 * x + 1] == (in ? FAKE_V(77, x, y) : 128));
        }
    // a region of the picture: even origin, the rows of the region only
    memset(yd, 0xEE, 64);
    memset(uvd, 0xEE, 64);
    CHECK(call(c, F_READ, 12, POOL(0), (uint64_t)a, 2, 2, 4, 2, OUT, 4, 8, OUT + 0x8000, 4, 4) == R_OK);
    for (int y = 0; y < 2; y++)
        for (int x = 0; x < 4; x++) CHECK(yd[y * 4 + x] == FAKE_Y(77, x + 2, y + 2));
    for (int x = 0; x < 2; x++) {
        CHECK(uvd[2 * x] == FAKE_U(77, x + 1, 1) && uvd[2 * x + 1] == FAKE_V(77, x + 1, 1));
    }
    // a read with a stride narrower than the row, and an odd origin, are refused and write nothing
    memset(yd, 0xEE, 64);
    CHECK(call(c, F_READ, 12, POOL(0), (uint64_t)a, 0, 0, 7, 5, OUT, 6, 64, OUT + 0x8000, 8, 24) == (0x6fu | 2004u << 9));
    CHECK(yd[0] == 0xEE);
    CHECK(call(c, F_READ, 12, POOL(0), (uint64_t)a, 1, 0, 4, 2, OUT, 4, 8, OUT + 0x8000, 4, 4) == (0x6fu | 2004u << 9));
}

// A damaged unit: the call returns a result, the picture is not made, the next unit decodes, nothing crashes
static void case_damaged(Cpu *c) {
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 3);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t i[3];
    for (int k = 0; k < 3; k++) CHECK(acquire(c, 0, k, &i[k]) == R_OK);
    for (int k = 0; k < 3; k++) release(c, 0, i[k], k);
    CHECK(submit(c, 0, 40, 1, 0) == R_OK);
    CHECK(submit(c, 0, 0xFF, 2, 0) == R_OK);               // damaged: the worker finds out; the guest's call has its result
    CHECK(submit(c, 0, 41, 3, 0) == R_OK);
    Props p;
    CHECK(wait_filled(c, 0, i[0], 2000, &p) && p.ts == 1);
    CHECK(wait_filled(c, 0, i[1], 2000, &p) && p.ts == 3);   // the damaged unit made no picture: the next one took its place
    plat_sleep_ns(100 * MS);
    CHECK(props(c, 0, i[2], &p) == R_NONE);
    CHECK(g_fake_bad == 1);
}

// Finalize: the worker is idle and the decoder closed when it returns, nothing is kept, and the slot takes a new decoder
static void case_finalize(Cpu *c) {
    g_fake_delay_ns = 100 * MS;
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 1);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a;
    CHECK(acquire(c, 0, 0, &a) == R_OK);
    release(c, 0, a, 0);
    CHECK(submit(c, 0, 50, 1, 0) == R_OK);
    CHECK(submit(c, 0, 51, 2, 0) == R_OK);
    CHECK(wait_sends(1, 2000));                            // the worker is inside a decode (100 ms) when Finalize comes
    uint64_t ev0 = g_ev[0];
    CHECK(call(c, F_FINI, 1, DEC(0)) == R_OK);
    CHECK(g_fake_closes == 1);                             // closed with the worker idle (a busy worker would use a freed decoder)
    CHECK(g_fake_close_busy == 0);                         // ... the fake saw no decode in flight at the close
    uint32_t n;
    CHECK(acquire(c, 0, 1, &n) == (0x6fu | 2004u << 9));   // no such decoder any more
    CHECK(submit(c, 0, 52, 3, 0) == (0x6fu | 2004u << 9));
    CHECK(call(c, F_FINI, 1, DEC(0)) == R_OK);             // twice: nothing
    // a new decoder in the same memory works, in the slot that was given back, with the same system event
    g_fake_delay_ns = 0;
    make_decoder(c, 0, 64, 64);
    CHECK(g_ev[0] == ev0);
    make_buffers(c, 0, 6, 4, 1);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    CHECK(acquire(c, 0, 0, &a) == R_OK);
    release(c, 0, a, 0);
    CHECK(submit(c, 0, 53, 4, 0) == R_OK);
    Props p;
    CHECK(wait_filled(c, 0, a, 2000, &p) && p.ts == 4);
    check_picture(c, 0, a, 6, 4, 53, 6, 6);
}

// End of stream: a unit with the flag marks its picture; a flagged empty unit makes an empty buffer with the flag
static void case_eos(Cpu *c) {
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 2);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a, b;
    CHECK(acquire(c, 0, 0, &a) == R_OK && acquire(c, 0, 1, &b) == R_OK);
    release(c, 0, a, 0);
    release(c, 0, b, 1);
    CHECK(submit(c, 0, 60, 1, 1) == R_OK);
    Props p;
    CHECK(wait_filled(c, 0, a, 2000, &p) && p.eos == 1 && p.ts == 1 && p.w == 6);
    CHECK(call(c, F_SUBMIT, 6, HEAP + 0x3300, DEC(0), UNIT, 0, 9, 1) == R_OK);   // no data, end of stream
    CHECK(wait_filled(c, 0, b, 2000, &p) && p.eos == 1 && p.w == 0 && p.h == 0);
}

// A full queue answers busy (3603), and the event comes back when the pictures are taken and there is room
static void case_busy(Cpu *c) {
    g_fake_delay_ns = 100 * MS;
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 2);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    CHECK(wait_event(c, g_ev[0], 1000 * MS) == 0);
    uint32_t i[2];
    CHECK(acquire(c, 0, 0, &i[0]) == R_OK && acquire(c, 0, 1, &i[1]) == R_OK);
    release(c, 0, i[0], 0);
    release(c, 0, i[1], 1);
    int ok = 0, busy = 0;
    for (int k = 0; k < 30; k++) {
        uint64_t r = submit(c, 0, (uint8_t)k, k, 0);
        if (r == R_OK) ok++;
        else {
            CHECK(r == R_BUSY);
            busy++;
        }
    }
    CHECK(ok >= 11 && ok <= 13 && busy == 30 - ok);
    (void)wait_event(c, g_ev[0], 0);
    // the player takes the pictures out, two at a time: room comes back once the queue is down, and the event says so
    int seen = 0;
    for (int cycle = 0; cycle < 8 && !seen; cycle++) {
        Props p;
        CHECK(acquire(c, 0, 2, &i[0]) == R_OK && acquire(c, 0, 3, &i[1]) == R_OK);
        CHECK(wait_filled(c, 0, i[0], 3000, &p) && wait_filled(c, 0, i[1], 3000, &p));
        release(c, 0, i[0], 2);
        release(c, 0, i[1], 3);
        seen = wait_event(c, g_ev[0], 20 * MS) == 0;
    }
    CHECK(seen);
    CHECK(submit(c, 0, 1, 1, 0) == R_OK);                  // and a unit is taken again
}

// The service's other entries: nn::codec::video::Initialize answers 0, a sync handle it did not make reads as reached
static void case_misc(Cpu *c) {
    CHECK(call(c, F_VIDEO_INIT, 0) == R_OK);
    memset((void *)(uintptr_t)SYNC(0), 0x5A, 16);
    CHECK(reached(c, 0));
    make_decoder(c, 0, 64, 64);
    CHECK(call(c, F_INIT_TRANSFER, 10, DEC(1), HEAP + 0x20000, 0x1000, 0x1000, 64, 64, 8, 0, 0, 0) == R_OK);   // the other form
    CHECK(g_ev[0] != 0);
    CHECK(call(c, F_FINI, 1, DEC(1)) == R_OK);
}

// Sync point events: the event of a request is signalled when the buffer it asks about holds the picture of that release, not
// before, and not after a cancel; a destroyed event's slot is reused with its system event.
#define SPT(i) (HEAP + 0x5000 + 0x100ull * (i))
static void case_syncpoint(Cpu *c) {
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 2);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a, b;
    CHECK(acquire(c, 0, 0, &a) == R_OK && acquire(c, 0, 1, &b) == R_OK);
    release(c, 0, a, 0);
    release(c, 0, b, 1);
    CHECK(acquire(c, 0, 2, &a) == R_OK);                   // the handle of this release: not reached yet
    int nev = g_nevents;
    CHECK(call(c, F_SPT_CREATE, 1, SPT(0)) == R_OK);
    CHECK(g_nevents == nev + 1);
    uint64_t ev = g_events[g_nevents - 1].ev;
    CHECK(g_events[g_nevents - 1].autoclear == 0);         // manual clear: the library clears it after a hit
    call(c, F_SPT_HOLDER, 2, HEAP + 0x5800, SPT(0));       // void: what it did is the holder the test double was given
    CHECK(g_holder == HEAP + 0x5800 && g_holder_ev == ev);   // the holder waits on the event's own system event
    CHECK(call(c, F_SPT_REQUEST, 2, SPT(0), SYNC(2)) == R_OK);
    CHECK(wait_event(c, ev, 60 * MS) == R_TIMEOUT);        // the picture is not there
    CHECK(submit(c, 0, 70, 1, 0) == R_OK);
    CHECK(wait_event(c, ev, 2000 * MS) == 0);              // it is: the event fires
    CHECK(wait_event(c, ev, 30 * MS) == R_TIMEOUT);
    CHECK(call(c, F_SPT_REQUEST, 2, SPT(0), SYNC(2)) == R_OK);   // asked after the fact: at once
    CHECK(wait_event(c, ev, 500 * MS) == 0);
    // a cancelled request is not signalled
    release(c, 0, a, 3);
    CHECK(acquire(c, 0, 4, &a) == R_OK && acquire(c, 0, 5, &b) == R_OK);
    CHECK(call(c, F_SPT_REQUEST, 2, SPT(0), SYNC(4)) == R_OK);   // the buffer the next unit fills
    call(c, F_SPT_CANCEL, 1, SPT(0));
    CHECK(submit(c, 0, 71, 2, 0) == R_OK);
    plat_sleep_ns(150 * MS);
    CHECK(wait_event(c, ev, 60 * MS) == R_TIMEOUT);
    // destroyed: the slot, and its system event, serve another event
    call(c, F_SPT_DESTROY, 1, SPT(0));
    CHECK(call(c, F_SPT_CREATE, 1, SPT(1)) == R_OK);
    CHECK(g_nevents == nev + 1);
}

// An event armed for a buffer of a decoder is disarmed when the decoder is finalised: a decoder made later in the same slot
// (the same id, the same buffer indices) must not signal the old request.
static void case_syncpoint_finalize(Cpu *c) {
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 1);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a;
    CHECK(acquire(c, 0, 0, &a) == R_OK);
    release(c, 0, a, 0);
    CHECK(acquire(c, 0, 1, &a) == R_OK);                   // the handle of this release: not reached
    CHECK(call(c, F_SPT_CREATE, 1, SPT(0)) == R_OK);
    uint64_t ev = g_events[g_nevents - 1].ev;
    CHECK(call(c, F_SPT_REQUEST, 2, SPT(0), SYNC(1)) == R_OK);   // armed, waiting for a picture that never comes
    CHECK(call(c, F_FINI, 1, DEC(0)) == R_OK);
    make_decoder(c, 0, 64, 64);                            // the same slot, so the same id, and the same buffer index
    make_buffers(c, 0, 6, 4, 1);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    CHECK(acquire(c, 0, 2, &a) == R_OK);
    release(c, 0, a, 2);
    CHECK(submit(c, 0, 62, 1, 0) == R_OK);
    Props p;
    CHECK(wait_filled(c, 0, a, 2000, &p));                 // the new decoder filled the buffer the old request named
    CHECK(wait_event(c, ev, 150 * MS) == R_TIMEOUT);       // and the old request was not signalled by it
}

// The same object initialised again while a decode is in flight (the library finalises a decoder by dropping its memory):
// the call comes back once the worker is idle, with the old decoder closed and none of its pictures kept.
static void case_reinit(Cpu *c) {
    g_fake_delay_ns = 100 * MS;
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 1);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a;
    CHECK(acquire(c, 0, 0, &a) == R_OK);
    release(c, 0, a, 0);
    CHECK(submit(c, 0, 50, 1, 0) == R_OK);
    CHECK(submit(c, 0, 51, 2, 0) == R_OK);
    CHECK(wait_sends(1, 2000));                            // inside a decode of 100 ms
    make_decoder(c, 0, 64, 64);
    CHECK(g_fake_closes == 1 && g_fake_close_busy == 0);   // the old decoder was closed with no decode running
    g_fake_delay_ns = 0;
    make_buffers(c, 0, 6, 4, 1);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    CHECK(acquire(c, 0, 1, &a) == R_OK);
    release(c, 0, a, 1);
    CHECK(submit(c, 0, 52, 3, 0) == R_OK);
    Props p;
    CHECK(wait_filled(c, 0, a, 2000, &p) && p.ts == 3);    // the first stream's units are gone
    check_picture(c, 0, a, 6, 4, 52, 6, 6);
}

// Room that comes back by a release alone: four decoded pictures wait for a buffer (no unit is queued), so the decoder has no
// room; releasing one buffer delivers one picture, and the event says so (the worker has nothing to dequeue to say it).
static void case_release_room(Cpu *c) {
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 4);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t i[4];
    for (int k = 0; k < 4; k++) CHECK(acquire(c, 0, k, &i[k]) == R_OK);
    for (int k = 0; k < 4; k++) CHECK(submit(c, 0, (uint8_t)(80 + k), k, 0) == R_OK);
    plat_sleep_ns(300 * MS);                               // all four decoded, held for a buffer; the queue is empty
    while (wait_event(c, g_ev[0], 0) == 0) {}              // whatever was signalled on the way
    CHECK(wait_event(c, g_ev[0], 60 * MS) == R_TIMEOUT);   // no room, no event
    release(c, 0, i[0], 0);
    CHECK(wait_event(c, g_ev[0], 1000 * MS) == 0);         // one picture left the queue: room
    CHECK(submit(c, 0, 90, 9, 0) == R_OK);
}

// Several guest threads on one decoder (a submitter, a consumer that acquires, reads and releases, and the main thread that
// stops, flushes and finalises it), against a slow decoder. Nothing may hang, crash or free a decoder a decode still uses,
// and a decoder made afterwards in the same slot works. The case has its own deadline (20 s), inside the runner's timeout.
#define STRESS_AREA(t) (HEAP + 0x30000 + 0x1000ull * (t))
static uint64_t g_st_stop, g_st_done, g_st_ops;
static void stress_pause(Cpu *c, uint64_t ms) { sv(c, 0x0B, ms * MS, 0, 0, 0); }   // SleepThread: the core is given up
static void stress_end(Cpu *c) {
    __atomic_add_fetch(&g_st_done, 1, __ATOMIC_RELEASE);
    sv(c, 0x0A, 0, 0, 0, 0);   // ExitThread
}
static void g_submitter(Cpu *c) {
    uint64_t a = STRESS_AREA(0);
    for (int k = 0; !__atomic_load_n(&g_st_stop, __ATOMIC_ACQUIRE); k++) {
        memset((void *)(uintptr_t)(a + 0x100), 0xA5, 64);
        *(uint8_t *)(uintptr_t)(a + 0x100) = (uint8_t)(k & 0x7f);
        (void)call(c, F_SUBMIT, 6, a, DEC(0), a + 0x100, 64, (uint64_t)k, (uint64_t)(k % 17 == 0));
        __atomic_add_fetch(&g_st_ops, 1, __ATOMIC_RELAXED);
        stress_pause(c, 1);
    }
    stress_end(c);
}
static void g_consumer(Cpu *c) {
    uint64_t a = STRESS_AREA(1);
    while (!__atomic_load_n(&g_st_stop, __ATOMIC_ACQUIRE)) {
        if (call(c, F_ACQUIRE, 3, a, a + 0x10, DEC(0)) == R_OK) {
            uint32_t idx;
            uint64_t h[2];
            memcpy(&idx, (void *)(uintptr_t)a, 4);
            memcpy(h, (void *)(uintptr_t)(a + 0x10), 16);
            (void)call(c, F_TRYWAIT, 1, a + 0x10);
            if (call(c, F_PROPS, 8, a + 0x40, a + 0x42, a + 0x44, a + 0x46, a + 0x48, a + 0x50, DEC(0), (uint64_t)idx) == R_OK)
                (void)call(c, F_READ, 12, POOL(0), (uint64_t)idx, 0, 0, 6, 4, a + 0x200, 6, 24, a + 0x300, 6, 12);
            call(c, F_RELEASE, 4, DEC(0), (uint64_t)idx, h[0], h[1]);
        }
        __atomic_add_fetch(&g_st_ops, 1, __ATOMIC_RELAXED);
        stress_pause(c, 1);
    }
    stress_end(c);
}
static void stress_thread(Cpu *c, int fn, int core, int index) {
    c->x[1] = G(fn), c->x[2] = 0, c->x[3] = HEAP + 0x200000 + 0x10000ull * (uint64_t)(index + 1);
    c->x[4] = 44, c->x[5] = (uint64_t)core;
    rt_svc(c, 0x08, PC);   // CreateThread
    CHECK(c->x[0] == 0);
    CHECK(sv(c, 0x09, (uint32_t)c->x[1], 0, 0, 0) == 0);   // StartThread
}
static void case_stress(Cpu *c) {
    g_fake_delay_ns = 3 * MS;
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 6);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    stress_thread(c, 2, 0, 0);
    stress_thread(c, 3, 1, 1);
    for (int k = 0; k < 60; k++) {
        CHECK(call(c, k & 1 ? F_STOP : F_FLUSH, 1, DEC(0)) == R_OK);
        stress_pause(c, 4);
    }
    CHECK(call(c, F_UNREGISTER, 2, DEC(0), 5) == R_OK);    // the buffer leaves while the others may hold it
    CHECK(call(c, F_FINI, 1, DEC(0)) == R_OK);   // the other two threads are still calling: they get an answer, not a crash
    __atomic_store_n(&g_st_stop, 1, __ATOMIC_RELEASE);
    for (int i = 0; __atomic_load_n(&g_st_done, __ATOMIC_ACQUIRE) < 2; i++) {
        CHECK(i < 20000);                                  // 20 s: a thread that does not end is a hang in the service
        stress_pause(c, 1);
    }
    CHECK(__atomic_load_n(&g_st_ops, __ATOMIC_RELAXED) > 100);
    CHECK(g_fake_closes == 1 && g_fake_close_busy == 0);   // closed once, with no decode in flight
    g_fake_delay_ns = 0;                                   // and the slot serves a new decoder
    make_decoder(c, 0, 64, 64);
    make_buffers(c, 0, 6, 4, 1);
    CHECK(call(c, F_START, 1, DEC(0)) == R_OK);
    uint32_t a;
    CHECK(acquire(c, 0, 0, &a) == R_OK);
    release(c, 0, a, 0);
    CHECK(submit(c, 0, 61, 8, 0) == R_OK);
    Props p;
    CHECK(wait_filled(c, 0, a, 2000, &p) && p.ts == 8);
    check_picture(c, 0, a, 6, 4, 61, 6, 6);
}

static void g_main(Cpu *c) {
    CHECK(HEAP != 0);
#define CASE(n)                      \
    if (!strcmp(g_case, #n)) {       \
        case_##n(c);                 \
        sv(c, 0x0A, 0, 0, 0, 0);     \
    }
    CASE(events) CASE(submit_async) CASE(no_reuse) CASE(stop) CASE(flush) CASE(stop_after) CASE(flush_after) CASE(planes) CASE(damaged)
    CASE(finalize) CASE(stress) CASE(syncpoint_finalize) CASE(reinit)
    CASE(eos) CASE(busy) CASE(misc) CASE(syncpoint) CASE(release_room)
    fprintf(stderr, "unknown case %s\n", g_case);
    exit(2);
}

static const uint32_t APP_SVCS[] = {0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0A, 0x0B, 0x0C, 0x0D, 0x0E, 0x0F, 0x10,
                                    0x11, 0x12, 0x13, 0x14, 0x15, 0x16, 0x17, 0x18, 0x19, 0x1A, 0x1B, 0x1C, 0x1D, 0x1E, 0x1F, 0x20,
                                    0x21, 0x22, 0x23, 0x24, 0x25, 0x26, 0x27, 0x28, 0x29, 0x2C, 0x2D, 0x32, 0x33, 0x34, 0x35, 0x36, 0x80};

int main(int argc, char **argv) {
    if (argc < 2) return 2;
    g_case = argv[1];
    RtMemConfig m = {.pool_size = 1ull << 30, .heap_region_size = 512ull << 20, .alias_region_size = 1ull << 30,
                     .stack_region_size = 256ull << 20};
    rt_mem_init(&m);
    RtKernelConfig k = {.core_mask = 0x3, .highest_priority = 28, .lowest_priority = 59, .main_priority = 44,
                        .main_core = 0, .main_stack_size = 0x100000, .handle_table_size = 512, .is_application = 1};
    for (unsigned i = 0; i < sizeof APP_SVCS / sizeof *APP_SVCS; i++) k.svc_mask[APP_SVCS[i] >> 6] |= 1ull << (APP_SVCS[i] & 63);
    rt_kernel_init(&k);
    for (int i = 0; i < F_COUNT; i++) CHECK((g_fn[i] = rt_hostfn_bind(SYMBOLS[i])) != 0);
    hos_init(".", NULL, 0);
    hos_codec_set_event_binder(test_bind);
    hos_codec_set_event_hooks(test_clear, test_holder);
    static const uint64_t addrs[] = {G(1), G(2), G(3)};
    static const GuestFn fns[] = {g_main, g_submitter, g_consumer};
    static RtModule mods[1];
    mods[0] = (RtModule){"main", MB, MOD_MAIN_SIZE, addrs, fns, 3, TEST_END};
    rt_dispatch_init(mods, 1);
    Cpu h;
    memset(&h, 0, sizeof h);
    h.x[1] = 16ull << 20;
    rt_svc(&h, 0x01, 0);   // SetHeapSize, from the host before the run
    CHECK(h.x[0] == 0);
    HEAP = h.x[1];
    rt_kernel_run(G(1), 0);
    printf("ok %s\n", g_case);
    return 0;
}
