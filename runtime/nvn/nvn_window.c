// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The NVN2 front end's windows and QueuePresentTexture (architecture section 5; D22 for what is left out).
//
// A window holds the game's textures, 2 to 4 of them, and rotates through the active ones: AcquireTexture hands out
// the next index and gives its sync the fence value the queue had reached when that texture was last presented, so a
// wait on the sync ends once the work drawn before that present is done (0, done at once, for a texture never
// presented). QueuePresentTexture hands the texture and the crop to the back end's present and returns at once. The
// crop is what the game set, or the whole first texture before it sets one, as the console starts it. The swap
// interval is stored as the game writes it, with no clamping, and answered back.
//
// The acquire gate (D22; oot2/nvn_pace.h has the rule). AcquireTexture, after choosing the index, blocks until the
// virtual 60 Hz clock (runtime/core/vsync.c) has reached the previous present's due tick plus the interval, read when
// it is called, with the guest thread's virtual core given up for the wait (rt_kernel_block_begin/end): another guest
// thread runs meanwhile. The wait is never between the return of an acquire and the next present, and
// QueuePresentTexture never waits: it takes the tick in progress as its due tick, writes its record, tells the back
// end (the optional present_pace of gfx.h) and returns at once. win_t.pace holds the window's last due tick. The
// vsync events belong to the clock thread (runtime/core); present timing and the swapchain belong to the Vulkan back
// end. The null back end counts presents. With the M1 capture on (oot2/capture.h), the present also marks the frame
// boundary and appends the frame's records to the capture's buffer in memory; the capture measures what that costs.
//
// Adapted from oot-pc-recomp runtime/nvn_api_obj.c at 8e09aaa (GPL-3.0-or-later), as DKB Recomp carries it at 9d8fa02,
// which ports MacNativeGames games/totk/third_party/d2r_gfx/gfx/nvn/hle_nvn.c at b75be8cf (MIT, Copyright (c) 2026
// jhaizhou; licenses/dkb-xlate-MIT.txt): the builder and its defaults (the device kept, interval 1), the textures and
// active count, the window object, the crop and the interval as stored, the rotation of AcquireTexture, and the present
// handed to the back end. The crop answered before any SetCrop (the whole first texture) is nxnative's
// dm/gfx/nvn/hle_nvn.c at 865cc35 (MIT, licenses/nxnative-MIT.txt). Changed here: the pacing is the gate above, in
// the clock's ticks, where the donors sleep until a display time on a present thread, and there is no movie hand-over
// in either call (D22); the acquire sync takes the fence value of the texture's last
// present instead of being done at once; window shapes the record cannot hold, textures not initialised and a present of
// a texture the window does not have stop by name.
#include <stdlib.h>
#include <string.h>

#include "nvn_queue.h"
#include "oot2/capture.h"
#include "oot2/kernel.h"
#include "oot2/nvn_pace.h"
#include "oot2/nvn_tex.h"
#include "oot2/vsync.h"
#include "plat.h"

#define WIN_MAGIC MAGIC('W', 'I', 'N', 'D')
#define WIN_TEXTURES 4   // what win_t holds

static win_t *win_arg(Cpu *c, int i, const char *fn) {
    win_t *w = arg_p(c, i);
    if (!w || w->magic != WIN_MAGIC)
        rt_unimplemented(c, "nvn", "%s on the window %p, which WindowInitialize did not initialise", fn, (void *)w);
    return w;
}

// The crop a present shows: the game's, or the whole first texture before it set one.
static void crop_of(const win_t *w, int v[4]) {
    if (w->crop[2] > 0 && w->crop[3] > 0) {
        memcpy(v, w->crop, 4 * sizeof *v);
        return;
    }
    const tex_t *t = (const tex_t *)w->tex[0];
    v[0] = v[1] = 0, v[2] = (int)t->d.width, v[3] = (int)t->d.height;
}

// ---------- the gate's state: the records of the presents and the experiment hold
static PlatMutex g_pace_lock = PLAT_MUTEX_INIT;   // the ring, the count, the hold, and every window's pace and presents
static NvnPresentRecord g_ring[NVN_PRESENT_RING];
static uint64_t g_count;
static int g_hold, g_hold_set;                    // g_hold_set: nvn_present_hold was called (the environment is not read)
static int g_hold_log;                            // the environment hold was read and is not logged yet

static int env_hold(void) {
    const char *e = getenv("OOT2_HOLD_TICKS");
    long v = e && *e ? strtol(e, NULL, 10) : 0;
    return v > 0 && v < 1000000 ? (int)v : 0;
}

// g_pace_lock held
static int hold_locked(void) {
    if (!g_hold_set) {
        g_hold_set = 1;
        g_hold = env_hold();
        g_hold_log = g_hold;
    }
    return g_hold;
}

// Called with no lock held: the log sink never runs under g_pace_lock.
static void hold_log(void) {
    plat_mutex_lock(&g_pace_lock);
    int h = g_hold_log;
    g_hold_log = 0;
    plat_mutex_unlock(&g_pace_lock);
    if (h) rt_log("nvn: experiment E2: every present is held for %d ticks, whatever the game asks for", h);
}

void nvn_present_hold(int ticks) {
    if (ticks < 0) rt_fatal("nvn_present_hold: %d ticks", ticks);
    plat_mutex_lock(&g_pace_lock);
    g_hold_set = 1;
    g_hold = ticks;
    plat_mutex_unlock(&g_pace_lock);
}

int nvn_present_hold_ticks(void) {
    plat_mutex_lock(&g_pace_lock);
    int h = hold_locked();
    plat_mutex_unlock(&g_pace_lock);
    hold_log();
    return h;
}

uint64_t nvn_present_count(void) {
    plat_mutex_lock(&g_pace_lock);
    uint64_t n = g_count;
    plat_mutex_unlock(&g_pace_lock);
    return n;
}

int nvn_present_record(uint64_t seq, NvnPresentRecord *out) {
    plat_mutex_lock(&g_pace_lock);
    int ok = seq && seq <= g_count && g_count - seq < NVN_PRESENT_RING;
    if (ok) *out = g_ring[seq % NVN_PRESENT_RING];
    plat_mutex_unlock(&g_pace_lock);
    if (ok && !out->shown && gfx_backend()->present_shown) {
        uint64_t shown = gfx_backend()->present_shown(seq);
        if (shown) {
            out->shown = shown;
            plat_mutex_lock(&g_pace_lock);
            if (g_ring[seq % NVN_PRESENT_RING].seq == seq) g_ring[seq % NVN_PRESENT_RING].shown = shown;
            plat_mutex_unlock(&g_pace_lock);
        }
    }
    return ok;
}

// ---------- window builder
#define WB(c) ((win_builder_t *)arg_p(c, 0))
// void nvnWindowBuilderSetDefaults(NVNwindowBuilder *b): the device stays, interval 1, no textures
NVN_API(nvnWindowBuilderSetDefaults) {
    win_builder_t *b = WB(c);
    NVNdevice *d = b->dev;
    memset(b, 0, sizeof *b);
    b->dev = d, b->interval = 1;
}
// void nvnWindowBuilderSetDevice(NVNwindowBuilder *b, NVNdevice *d), SetNativeWindow(b, NVNnativeWindow w)
NVN_API(nvnWindowBuilderSetDevice) { WB(c)->dev = arg_p(c, 1); }
NVN_API(nvnWindowBuilderSetNativeWindow) { WB(c)->native = arg_p(c, 1); }
// void nvnWindowBuilderSetTextures(NVNwindowBuilder *b, int n, NVNtexture *const *textures)
NVN_API(nvnWindowBuilderSetTextures) {
    win_builder_t *b = WB(c);
    int n = (int)arg_x(c, 1);
    NVNtexture *const *t = arg_p(c, 2);
    if (n < 0 || n > WIN_TEXTURES || (n && !t))
        rt_unimplemented(c, "nvn", "WindowBuilderSetTextures: %d textures at %p, past the %d a window holds here", n,
                         (const void *)t, WIN_TEXTURES);
    b->ntex = n;
    for (int i = 0; i < WIN_TEXTURES; i++) b->tex[i] = i < n ? t[i] : NULL;
}
// void nvnWindowBuilderSetNumActiveTextures(NVNwindowBuilder *b, int n) (NVN2), SetPresentInterval(b, int interval)
NVN_API(nvnWindowBuilderSetNumActiveTextures) { WB(c)->active = (int32_t)arg_x(c, 1); }
NVN_API(nvnWindowBuilderSetPresentInterval) { WB(c)->interval = (int32_t)arg_x(c, 1); }

// ---------- windows
// NVNboolean nvnWindowInitialize(NVNwindow *w, const NVNwindowBuilder *b): all the builder's textures active unless it
// set fewer
NVN_API(nvnWindowInitialize) {
    const win_builder_t *b = arg_p(c, 1);
    int active = b->active ? b->active : b->ntex;
    if (b->ntex < 2 || active < 2 || active > b->ntex)
        rt_unimplemented(c, "nvn", "WindowInitialize: %d textures, %d active; a window rotates 2 to %d", b->ntex, active,
                         WIN_TEXTURES);
    for (int i = 0; i < b->ntex; i++) {
        const tex_t *t = (const tex_t *)b->tex[i];
        if (!t || t->magic != NVN_TEX_MAGIC || !t->serial)
            rt_unimplemented(c, "nvn", "WindowInitialize: texture %d (%p) is not initialised", i, (const void *)t);
    }
    win_t *w = arg_p(c, 0);
    memset(w, 0, sizeof *w);
    w->magic = WIN_MAGIC, w->native = b->native, w->ntex = b->ntex, w->interval = b->interval, w->active = active;
    memcpy(w->tex, b->tex, sizeof w->tex);
    rt_log("nvn: WindowInitialize: %d textures, %d active, interval %d, the first %ux%u", w->ntex, w->active,
           w->interval, ((const tex_t *)w->tex[0])->d.width, ((const tex_t *)w->tex[0])->d.height);
    ret_x(c, 1);
}
// void nvnWindowFinalize(NVNwindow *w)
NVN_API(nvnWindowFinalize) { win_arg(c, 0, "WindowFinalize")->magic = 0; }
// void nvnWindowSetCrop(NVNwindow *w, int x, int y, int width, int height)
NVN_API(nvnWindowSetCrop) {
    win_t *w = win_arg(c, 0, "WindowSetCrop");
    for (int i = 0; i < 4; i++) w->crop[i] = (int32_t)arg_x(c, i + 1);
}
// void nvnWindowGetCrop(const NVNwindow *w, NVNrectangle *rect)
NVN_API(nvnWindowGetCrop) {
    const win_t *w = win_arg(c, 0, "WindowGetCrop");
    NVNrectangle *r = arg_p(c, 1);
    int v[4];
    crop_of(w, v);
    r->x = v[0], r->y = v[1], r->width = v[2], r->height = v[3];
}
// void nvnWindowSetNumActiveTextures(NVNwindow *w, int n) (NVN2): from the next acquire
NVN_API(nvnWindowSetNumActiveTextures) {
    win_t *w = win_arg(c, 0, "WindowSetNumActiveTextures");
    int n = (int)arg_x(c, 1);
    if (n < 2 || n > w->ntex)
        rt_unimplemented(c, "nvn", "WindowSetNumActiveTextures: %d active of the window's %d textures", n, w->ntex);
    w->active = n;
}
// void nvnWindowSetPresentInterval(NVNwindow *w, int interval), int nvnWindowGetPresentInterval(const NVNwindow *w)
NVN_API(nvnWindowSetPresentInterval) {
    win_t *w = win_arg(c, 0, "WindowSetPresentInterval");
    int v = (int)arg_x(c, 1);
    if (v != w->interval) rt_log("nvn: the game sets present interval %d (was %d)", v, w->interval);
    w->interval = v;
}
NVN_API(nvnWindowGetPresentInterval) {
    ret_x(c, (uint32_t)win_arg(c, 0, "WindowGetPresentInterval")->interval);
}
// NVNwindowAcquireTextureResult nvnWindowAcquireTexture(NVNwindow *w, NVNsync *sync, int *index): the gate. The
// interval is the window's now, so a change applies here; the hold (experiment E2) raises it, never lowers it.
NVN_API(nvnWindowAcquireTexture) {
    win_t *w = win_arg(c, 0, "WindowAcquireTexture");
    int *index = arg_p(c, 2);
    if (!index) rt_unimplemented(c, "nvn", "WindowAcquireTexture: the index goes to address 0");
    int i = w->next % w->active;
    w->next = (i + 1) % w->active;
    *index = i;
    plat_mutex_lock(&g_pace_lock);
    int interval = w->interval, hold = hold_locked(), presented = w->presents != 0;
    uint64_t due = w->pace;
    plat_mutex_unlock(&g_pace_lock);
    hold_log();
    // With the hold on, a game that asks for interval 0 is held too: effective = max(interval, hold).
    if (hold > interval) interval = hold;
    if (presented && interval > 0) {
        uint64_t target = due + (uint64_t)interval;
        if (rt_vsync_tick() < target) {
            int token = rt_kernel_block_begin();
            rt_vsync_wait_tick(target);
            rt_kernel_block_end(token);
        }
    }
    if (arg_p(c, 1)) nvn_sync_fence(nvn_sync_arg(c, 1, "WindowAcquireTexture"), 2, w->fence[i]);
    ret_x(c, NVN_WINDOW_ACQUIRE_TEXTURE_RESULT_SUCCESS);
}
// void nvnQueuePresentTexture(NVNqueue *q, NVNwindow *w, int index): shown after the work the queue submitted before it
NVN_API(nvnQueuePresentTexture) {
    queue_t *q = nvn_queue_arg(c, 0, "QueuePresentTexture");
    win_t *w = win_arg(c, 1, "QueuePresentTexture");
    int i = (int)arg_x(c, 2);
    if (i < 0 || i >= w->ntex)
        rt_unimplemented(c, "nvn", "QueuePresentTexture: texture %d of the window %p, which has %d", i, (void *)w,
                         w->ntex);
    int crop[4];
    crop_of(w, crop);
    uint64_t tick = rt_vsync_tick();
    plat_mutex_lock(&g_pace_lock);
    // max(previous due, tick): the previous due is a tick the clock already reached, so on a monotonic clock this is
    // just tick; it is kept as a guard (no test can reach it) so a due tick never goes back.
    uint64_t due = w->presents && w->pace > tick ? w->pace : tick, seq = ++g_count;
    w->pace = due;
    w->presents++;
    w->ticket[i] = w->presents;
    g_ring[seq % NVN_PRESENT_RING] = (NvnPresentRecord){.seq = seq, .due = due, .requested = w->interval,
                                                         .held = hold_locked(), .texture = i};
    plat_mutex_unlock(&g_pace_lock);
    hold_log();
    w->fence[i] = __atomic_load_n(&q->last, __ATOMIC_ACQUIRE);
    const GfxBackend *b = gfx_backend();
    if (b->present_pace) b->present_pace(seq, due);
    b->present((const tex_t *)w->tex[i], crop);
    capture_present(seq);   // before the census, whose frame limit may end the run here
    nvn_census_present(seq);
}
