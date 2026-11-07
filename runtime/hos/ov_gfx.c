// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Graphics bring-up, answered as the console's graphics service and its display layer would for a docked console:
// nv::* records the game's graphics allocator callbacks and work memory for the NVN layer, nn::vi::* gives the display,
// the layer and the native window their identities, answers the display queries (60 Hz, SDR, no variable refresh rate)
// and binds the vsync event, which the virtual 60 Hz clock signals (oot2/vsync.h). nvnBootstrapLoader and every NVN
// entry point belong to runtime/nvn, not to this file.
//
// What the NVN layer and a presenter read from here is declared next to the state it reads (hos_nv_state,
// hos_vi_native_window, hos_vsync_events); the types and the three declarations move into hos.h with the notices row.
//
// Adapted from oot-pc-recomp runtime/ov_gfx.c at 8e09aaa39c2cff5c3ca7c2a4ee36345745eebe17 (GPL-3.0-or-later as a
// repository; the file's own header says it is adapted from MacNativeGames games/totk/runtime/frontend.m and
// gfx_bridge.c at b75be8cfa704df3701e86130c8ced268e79d6b11, MIT, Copyright (c) 2026 MacNativeGames contributors; DKB
// Recomp at 9d8fa02 carries the same file). Changed here: host functions follow runtime.h and carry a HOS_OVERRIDE
// note; out parameters are written with hos_out and a pointer the guest may not write answers the invalid result;
// the three identities are one kernel-region allocation; the vsync event is bound with hos_bind_system_event (no lock
// held while guest code runs) and its writable end goes to the virtual 60 Hz clock (rt_vsync_bind, runtime/core) and is
// kept for the NVN layer, instead of a gfx_vsync_bind call; the
// allocator record is read through an accessor, not a shared header; nv::InitCheck checks the host's own record of
// nv::InitializeGraphics; nvnBootstrapLoader and its start-up constructor are left out (runtime/nvn has both); new:
// nn::vi::QueryDisplayCurrentRefreshRate (60 Hz), SetDisplay120HzRefreshRateCapability, and the gamut and tone-mapping
// setters, and the two new display calls check the display they are given.
#include <stdint.h>

#include "core.h"
#include "hos.h"
#include "oot2/vsync.h"
#include "plat.h"

#define VI_R_INVALID 0xf601u         // MacNativeGames' result for a call on a display or layer not handed out
#define DISPLAY_HZ 60u               // the docked console's display mode this project targets
#define HDR_PAPER_WHITE_NITS 203u    // BT.2408 reference white
#define IDENTITY_BYTES 64u           // one zeroed block per identity, in case SDK code reads through the pointer
#define VSYNC_EVENTS 4

static PlatMutex lock = PLAT_MUTEX_INIT;
static HosNvState nv;                         // lock for writes
static int vi_ready;                          // lock: nn::vi::Initialize ran
static uint64_t display, layer, window;       // lock: the identities
static uint32_t vsync[VSYNC_EVENTS];          // lock: writable ends of the bound events
static int nvsync;                            // lock

const HosNvState *hos_nv_state(void) { return &nv; }

uint64_t hos_vi_native_window(void) {
    plat_mutex_lock(&lock);
    uint64_t w = window;
    plat_mutex_unlock(&lock);
    return w;
}

int hos_vsync_events(uint32_t *writable, int max) {
    plat_mutex_lock(&lock);
    int n = nvsync < max ? nvsync : max;
    for (int i = 0; i < n; i++) writable[i] = vsync[i];
    plat_mutex_unlock(&lock);
    return n;
}

// ---- nv::*
static void set_allocator(Cpu *c, HosNvAllocator *a) {
    plat_mutex_lock(&lock);
    *a = (HosNvAllocator){arg_x(c, 0), arg_x(c, 1), arg_x(c, 2), arg_x(c, 3)};
    plat_mutex_unlock(&lock);
}

// void nv::SetGraphicsAllocator(AllocateFunction, FreeFunction, ReallocateFunction, void *user)
static void ov_graphics_allocator(Cpu *c) { set_allocator(c, &nv.graphics); }
// void nv::SetGraphicsDevtoolsAllocator(AllocateFunction, FreeFunction, ReallocateFunction, void *user)
static void ov_devtools_allocator(Cpu *c) { set_allocator(c, &nv.devtools); }

// void nv::InitializeGraphics(void *memory, size_t size). A game may hand over a work buffer or (0, 0) and leave the
// driver to its allocator callbacks; nothing on the host reads the buffer, so both are fine, and only one of the two
// being zero is not.
static void ov_initialize_graphics(Cpu *c) {
    uint64_t mem = arg_x(c, 0), size = arg_x(c, 1);
    if (!mem != !size)
        rt_guest_stop(c, "nv::InitializeGraphics: work memory %#llx with size %#llx", (unsigned long long)mem,
                      (unsigned long long)size);
    plat_mutex_lock(&lock);
    nv.memory = mem;
    nv.memory_size = size;
    nv.initialized = 1;
    plat_mutex_unlock(&lock);
    rt_log("nv: graphics work memory %#llx+%#llx recorded", (unsigned long long)mem, (unsigned long long)size);
}

// void nv::InitCheck(): the SDK aborts unless nv::InitializeGraphics ran. That call is the host's (above), so the SDK's
// own flag is never set and the host answers from its own record.
static void ov_init_check(Cpu *c) {
    plat_mutex_lock(&lock);
    int ran = nv.initialized;
    plat_mutex_unlock(&lock);
    if (!ran) rt_guest_stop(c, "nv::InitCheck before nv::InitializeGraphics: the SDK aborts here");
}

// ---- nn::vi: identities, as small zeroed blocks in the guest-visible kernel region
static uint64_t identity(const uint64_t *which) {
    plat_mutex_lock(&lock);
    uint64_t v = vi_ready ? *which : 0;
    plat_mutex_unlock(&lock);
    return v;
}

// Result code of an out parameter: a null pointer is skipped (a caller that wants no answer), one the guest may not
// write is the invalid result.
static uint32_t put_opt(uint64_t addr, const void *src, uint64_t size) {
    return !addr || hos_out(addr, src, size) ? HOS_R_OK : VI_R_INVALID;
}

// Result: *out = id, or invalid when nn::vi::Initialize has not run or the guest cannot take the pointer.
static void hand_out(Cpu *c, uint64_t id) {
    ret_x(c, id && hos_out(arg_x(c, 0), &id, sizeof id) ? HOS_R_OK : VI_R_INVALID);
}

// void nn::vi::Initialize()
static void ov_vi_initialize(Cpu *c) {
    (void)c;
    plat_mutex_lock(&lock);
    if (!display) {
        uint64_t block = mem_kernel_alloc(3 * IDENTITY_BYTES);
        display = block;
        layer = block + IDENTITY_BYTES;
        window = block + 2 * IDENTITY_BYTES;
    }
    vi_ready = 1;
    plat_mutex_unlock(&lock);
}

// Result nn::vi::OpenDefaultDisplay(Display **out)
static void ov_open_display(Cpu *c) { hand_out(c, identity(&display)); }

// Result nn::vi::CreateLayer(Layer **out, Display *display)
static void ov_create_layer(Cpu *c) {
    uint64_t d = identity(&display);
    hand_out(c, d && arg_x(c, 1) == d ? identity(&layer) : 0);
}

// Result nn::vi::GetNativeWindow(void **out, Layer *layer): the placeholder the NVN layer recognises
static void ov_native_window(Cpu *c) {
    uint64_t l = identity(&layer);
    hand_out(c, l && arg_x(c, 1) == l ? identity(&window) : 0);
}

// Result nn::vi::GetDisplayVsyncEvent(os::SystemEventType *event, Display *display): an auto-clear system event that
// the virtual 60 Hz clock signals at every tick from the next one (rt_vsync_bind); the same event object gives the
// same writable end, bound once.
static void ov_vsync_event(Cpu *c) {
    uint64_t ev = arg_x(c, 0), d = identity(&display);
    if (!ev || !d || arg_x(c, 1) != d) {
        ret_x(c, VI_R_INVALID);
        return;
    }
    uint32_t w = hos_bind_system_event(c, ev, 1);   // runs guest code: no lock held
    int known = 0, full = 0;
    plat_mutex_lock(&lock);
    for (int i = 0; i < nvsync; i++) known |= vsync[i] == w;
    full = !known && nvsync == VSYNC_EVENTS;
    if (!known && !full && w) vsync[nvsync++] = w;
    plat_mutex_unlock(&lock);
    if (full) hos_gap(c, "vi", "nn::vi::GetDisplayVsyncEvent: more than %d vsync events asked for", VSYNC_EVENTS);
    if (!known && w) rt_vsync_bind(w);
    ret_x(c, HOS_R_OK);
}

// ---- display queries and settings (Switch 2 SDK): SDR at a fixed 60 Hz with no variable refresh rate
// Result QueryDisplayHdrStatus(bool *enabled, HdrOutputFormat *format, u16 *, const Display *): HDR off. The format is
// left alone, as a console with HDR off leaves it.
static void ov_hdr_status(Cpu *c) {
    static const uint8_t off = 0;
    ret_x(c, put_opt(arg_x(c, 0), &off, sizeof off));
}
// Result QueryDisplayHdrPaperWhite(u16 *nits, const Display *)
static void ov_hdr_paper_white(Cpu *c) {
    uint16_t nits = HDR_PAPER_WHITE_NITS;
    ret_x(c, put_opt(arg_x(c, 0), &nits, sizeof nits));
}
// Result QueryDisplayVrrSupported(bool *out, const Display *): no
static void ov_vrr_supported(Cpu *c) {
    static const uint8_t no = 0;
    ret_x(c, put_opt(arg_x(c, 0), &no, sizeof no));
}
// Result QueryDisplayCurrentRefreshRate(RefreshRate *rate, bool *vrr, const Display *display): a RefreshRate is
// written as a 32-bit hertz value (60 and 120 are the modes the console has), and no variable refresh rate is active.
static void ov_refresh_rate(Cpu *c) {
    uint64_t d = identity(&display);
    if (!d || arg_x(c, 2) != d) {
        ret_x(c, VI_R_INVALID);
        return;
    }
    uint32_t hz = DISPLAY_HZ;
    static const uint8_t vrr = 0;
    uint32_t r = put_opt(arg_x(c, 0), &hz, sizeof hz);
    ret_x(c, r != HOS_R_OK ? r : put_opt(arg_x(c, 1), &vrr, sizeof vrr));
}
// Result SetDisplay120HzRefreshRateCapability(Display *, bool): accepted, and the display stays at 60 Hz
static void ov_set_120hz(Cpu *c) {
    uint64_t d = identity(&display);
    ret_x(c, d && arg_x(c, 0) == d ? HOS_R_OK : VI_R_INVALID);
}
// Result SetDisplayVrrCapability(Display *, bool), Set/ClearLayerExtendedSwapInterval(Layer *[, int, int]),
// SetLayerNextHdr{Extended,Standard}ContentLuminanceRangeMetadata(Layer *, u16, u16),
// SetLayerNextHdrContentGamutMetadata(Layer *, HdrContentGamut), SetLayerNextHdrToSdrToneMappingMetadata(Layer *,
// HdrToneMapping): accepted, nothing changes
static void ov_vi_accept(Cpu *c) { ret_x(c, HOS_R_OK); }

HOS_OVERRIDE("_ZN2nv20SetGraphicsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_", ov_graphics_allocator,
             HOS_IMPLEMENTED,
             "records the game's graphics allocator callbacks for the NVN layer; donor ov_gfx.c:22-29");
HOS_OVERRIDE("_ZN2nv28SetGraphicsDevtoolsAllocatorEPFPvmmS0_EPFvS0_S0_EPFS0_S0_mS0_ES0_", ov_devtools_allocator,
             HOS_IMPLEMENTED, "records the devtools allocator callbacks the same way; donor ov_gfx.c:22-31");
HOS_OVERRIDE("_ZN2nv18InitializeGraphicsEPvm", ov_initialize_graphics, HOS_IMPLEMENTED,
             "records the work memory (or none) and that the host did the initialisation; donor ov_gfx.c:35-46");
HOS_OVERRIDE("_ZN2nv9InitCheckEv", ov_init_check, HOS_IMPLEMENTED,
             "passes once nv::InitializeGraphics ran, else stops as the SDK aborts; donor ov_gfx.c:50 accepted always");
HOS_OVERRIDE("_ZN2nn2vi10InitializeEv", ov_vi_initialize, HOS_IMPLEMENTED,
             "allocates the display, layer and window identities in the kernel region; donor ov_gfx.c:57-67");
HOS_OVERRIDE("_ZN2nn2vi18OpenDefaultDisplayEPPNS0_7DisplayE", ov_open_display, HOS_IMPLEMENTED,
             "hands out the one display identity; donor ov_gfx.c:77-82");
HOS_OVERRIDE("_ZN2nn2vi11CreateLayerEPPNS0_5LayerEPNS0_7DisplayE", ov_create_layer, HOS_IMPLEMENTED,
             "hands out the one layer identity for the default display; donor ov_gfx.c:85-90");
HOS_OVERRIDE("_ZN2nn2vi15GetNativeWindowEPPvPNS0_5LayerE", ov_native_window, HOS_IMPLEMENTED,
             "hands out the window placeholder the NVN layer recognises; donor ov_gfx.c:93-98");
HOS_OVERRIDE("_ZN2nn2vi20GetDisplayVsyncEventEPNS_2os15SystemEventTypeEPNS0_7DisplayE", ov_vsync_event,
             HOS_IMPLEMENTED, "binds an auto-clear system event that the virtual 60 Hz clock signals at every tick "
             "(runtime/core/vsync.c); donor ov_gfx.c:101-107");
HOS_OVERRIDE("_ZN2nn2vi21QueryDisplayHdrStatusEPbPNS0_15HdrOutputFormatEPtPKNS0_7DisplayE", ov_hdr_status,
             HOS_IMPLEMENTED, "HDR is off: the host presents SDR; donor ov_gfx.c:111-115");
HOS_OVERRIDE("_ZN2nn2vi25QueryDisplayHdrPaperWhiteEPtPKNS0_7DisplayE", ov_hdr_paper_white, HOS_IMPLEMENTED,
             "BT.2408 reference white, 203 nits; donor ov_gfx.c:117-121");
HOS_OVERRIDE("_ZN2nn2vi24QueryDisplayVrrSupportedEPbPKNS0_7DisplayE", ov_vrr_supported, HOS_IMPLEMENTED,
             "no variable refresh rate: the host paces a fixed interval; donor ov_gfx.c:123-127");
HOS_OVERRIDE("_ZN2nn2vi30QueryDisplayCurrentRefreshRateEPNS0_11RefreshRateEPbPKNS0_7DisplayE", ov_refresh_rate,
             HOS_IMPLEMENTED,
             "60 Hz with no variable refresh rate, the docked console's display mode this project targets; no donor "
             "covers it");
HOS_OVERRIDE("_ZN2nn2vi36SetDisplay120HzRefreshRateCapabilityEPNS0_7DisplayEb", ov_set_120hz, HOS_STUB,
             "accepted, the display stays at 60 Hz; no donor covers it");
HOS_OVERRIDE("_ZN2nn2vi23SetDisplayVrrCapabilityEPNS0_7DisplayEb", ov_vi_accept, HOS_STUB,
             "accepted, there is no variable refresh rate to enable; donor ov_gfx.c:128-135");
HOS_OVERRIDE("_ZN2nn2vi28SetLayerExtendedSwapIntervalEPNS0_5LayerEii", ov_vi_accept, HOS_STUB,
             "accepted, the host presents at a fixed interval; donor ov_gfx.c:128-136");
HOS_OVERRIDE("_ZN2nn2vi30ClearLayerExtendedSwapIntervalEPNS0_5LayerE", ov_vi_accept, HOS_STUB,
             "accepted, the host presents at a fixed interval; donor ov_gfx.c:128-137");
HOS_OVERRIDE("_ZN2nn2vi52SetLayerNextHdrExtendedContentLuminanceRangeMetadataEPNS0_5LayerEtt", ov_vi_accept, HOS_STUB,
             "accepted, the host presents SDR; donor ov_gfx.c:128-138");
HOS_OVERRIDE("_ZN2nn2vi52SetLayerNextHdrStandardContentLuminanceRangeMetadataEPNS0_5LayerEtt", ov_vi_accept, HOS_STUB,
             "accepted, the host presents SDR; donor ov_gfx.c:128-139");
HOS_OVERRIDE("_ZN2nn2vi35SetLayerNextHdrContentGamutMetadataEPNS0_5LayerENS0_15HdrContentGamutE", ov_vi_accept,
             HOS_STUB, "accepted, the host presents SDR; no donor covers it, answered like its HDR siblings");
HOS_OVERRIDE("_ZN2nn2vi39SetLayerNextHdrToSdrToneMappingMetadataEPNS0_5LayerENS0_14HdrToneMappingE", ov_vi_accept,
             HOS_STUB, "accepted, the host presents SDR; no donor covers it, answered like its HDR siblings");
