// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What the front end counts per present (the null back end boot). Every present logs itself the first time and then every 60th, so a run
// stopped by a time limit still says how many frames it presented. In census mode (OOT2_CENSUS=1, runtime.h) the table of
// host-function calls is also copied at present 1 and at every 60th present, and nvn_census_write turns the copies into
// calls per frame in the NVN2 call census's six columns (phase, frames, slot, function, calls_per_frame, calls): phase "boot" is
// everything before present 1 (frames 0), "wNNN" is one window of presents, "all" the whole run after present 1. The
// slot column is left empty; an offline join to the game's slot map fills it. Nothing here includes the hos layer.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nvn_api.h"
#include "oot2/nvn_pace.h"
#include "plat.h"

#define WINDOW 60            // presents per window
#define FLUSH_EVERY 600      // a census-mode run rewrites its file this often (a killed run writes nothing at its end)
#define MAX_SNAPS 1024       // 1 + 1023 windows: about 17 minutes at 60 Hz; later windows are not kept

typedef struct {
    uint64_t present, ns;
    uint64_t *calls;
} Snap;
static Snap g_snap[MAX_SNAPS];
static int g_nsnap, g_nslots, g_full_said;
static PlatMutex g_lock = PLAT_MUTEX_INIT;
static uint64_t g_first_ns;

static void take(uint64_t seq) {
    int n = rt_hostfn_count();
    uint64_t *calls = malloc((size_t)n * sizeof *calls);
    if (!calls) rt_fatal("nvn census: out of memory for a snapshot of %d slots", n);
    rt_hostfn_snapshot(calls, n);
    plat_mutex_lock(&g_lock);
    if (g_nsnap == MAX_SNAPS) {
        int say = !g_full_said;
        g_full_said = 1;
        plat_mutex_unlock(&g_lock);
        free(calls);
        if (say) rt_log("nvn: %d snapshots kept; later windows are not recorded", MAX_SNAPS);
        return;
    }
    g_nslots = n;
    g_snap[g_nsnap++] = (Snap){seq, plat_now_ns(), calls};
    plat_mutex_unlock(&g_lock);
}

// The title, as the log shows it (the movie player, the title pass capture). The movie's frame is 14 compute dispatches (13 of them tensor
// dispatches of the neural upscaler, one the conversion of the decoded planes to RGBA) and 10 nvnCommandBufferSetRenderTargets
// calls a present; the menu then appears over the looping movie by itself. A window of 60 presents whose calls a present
// reach both counts is a "movie window"; two in a row is the title's movie: the log says so once. The menu's own draws are
// not part of that line: each window's line also carries the draws a present, whose step up over the movie's own marks the
// menu fading in (the movie's full-screen triangle and the window pass may sit in command memory the front end does not
// see, so the draw count is a floor).
#define TITLE_DISPATCHES 14
#define TITLE_TARGETS 10
static uint64_t g_prev_calls[3], g_prev_present;
static int g_movie_windows, g_title_said;

static uint64_t named_calls(const char *name) { return rt_hostfn_calls(nvn_proc_address(name)); }

// The calls a present of the last window, once per window, for the log line and the title.
static void window_line(uint64_t seq, double secs) {
    uint64_t cur[3] = {named_calls("nvnCommandBufferSetRenderTargets"), named_calls("nvnCommandBufferDispatchCompute"),
                       named_calls("nvnCommandBufferDrawElementsInstanced")};
    uint64_t frames = seq - g_prev_present;
    if (seq == 1 || !frames) {
        g_prev_present = seq;
        memcpy(g_prev_calls, cur, sizeof cur);
        rt_log("nvn: present %llu, %.2f s after the first", (unsigned long long)seq, secs);
        return;
    }
    double rt = (double)(cur[0] - g_prev_calls[0]) / (double)frames, dp = (double)(cur[1] - g_prev_calls[1]) / (double)frames,
           dr = (double)(cur[2] - g_prev_calls[2]) / (double)frames;
    g_prev_present = seq;
    memcpy(g_prev_calls, cur, sizeof cur);
    rt_log("nvn: present %llu, %.2f s after the first; a present: %.1f SetRenderTargets, %.1f DispatchCompute, %.1f DrawElementsInstanced",
           (unsigned long long)seq, secs, rt, dp, dr);
    g_movie_windows = (rt >= TITLE_TARGETS && dp >= TITLE_DISPATCHES) ? g_movie_windows + 1 : 0;
    if (g_movie_windows >= 2 && !g_title_said) {
        g_title_said = 1;
        rt_log("nvn: title: the movie's frame (%d dispatches and %d SetRenderTargets a present) in two windows in a row, at "
               "present %llu", TITLE_DISPATCHES, TITLE_TARGETS, (unsigned long long)seq);
    }
}

// Called by nvnQueuePresentTexture after present number seq (1 on) went to the back end.
void nvn_census_present(uint64_t seq) {
    uint64_t now = plat_now_ns();
    if (seq == 1) g_first_ns = now;
    if (seq == 1 || seq % WINDOW == 0) window_line(seq, seq == 1 ? 0.0 : (double)(now - g_first_ns) / 1e9);
    if (!rt_census_on()) return;
    if (seq == 1 || seq % WINDOW == 0) take(seq);
    if (seq % FLUSH_EVERY == 0) rt_census_flush();
    uint64_t limit = rt_census_frames();
    if (limit && seq >= limit) {
        char line[96];
        snprintf(line, sizeof line, "[census mode] ended at its frame limit, present %llu", (unsigned long long)seq);
        rt_log("ended at its frame limit, present %llu", (unsigned long long)seq);
        rt_census_end(RT_EXIT_CENSUS, line);
    }
}

// What the front end counted, as rows of the census (census.c writes them as kind "nvn-stat"): numbers only. emit(ctx, key,
// count, why) is called once per key: recordings, callbacks, waitevents, waitclasses, presents.
void nvn_census_stats(NvnStatRow emit, void *ctx) {
    typedef unsigned long long ull;
    NvnCmdCounts k;
    nvn_cmd_counts(&k);
    uint64_t unsat = 0, waits = gfx_null_wait_events(&unsat);
    char why[400];
    snprintf(why, sizeof why, "inline %llu, slow %llu, ended with a handle %llu, ended empty %llu", (ull)k.begin_inline,
             (ull)k.begin_slow, (ull)k.end_handle, (ull)k.end_empty);
    emit(ctx, "recordings", k.begin_inline + k.begin_slow, why);
    snprintf(why, sizeof why, "memory callbacks: control %llu, command %llu; recordings begun over left-out records %llu",
             (ull)k.callback_control, (ull)k.callback_command, (ull)k.left_out);
    emit(ctx, "callbacks", k.callback_control + k.callback_command, why);
    snprintf(why, sizeof why, "WaitEvents met %llu, unsatisfied at completion %llu", (ull)waits, (ull)unsat);
    emit(ctx, "waitevents", waits, why);
    GfxNullWaitClasses w;
    gfx_null_wait_classes(&w);
    snprintf(why, sizeof why, "by a GPU signal in order %llu, by a CPU signal before %llu, already satisfied %llu, "
             "unsatisfied %llu (later by GPU %llu, later by CPU %llu, not satisfied by the end %llu, dropped from the "
             "full list %llu)", (ull)w.by_gpu, (ull)w.by_cpu, (ull)w.initial, (ull)w.unsatisfied, (ull)w.later_gpu,
             (ull)w.later_cpu, (ull)w.pending, (ull)w.dropped);
    emit(ctx, "waitclasses", w.by_gpu + w.by_cpu + w.initial + w.unsatisfied, why);
    NvnPoolStats ps;
    nvn_pool_stats(&ps);
    snprintf(why, sizeof why, "memory pools made %llu, live %llu (most at once %llu), %llu bytes live (most %llu); made with "
             "GPU_NO_ACCESS %llu; GPU address window in use %llu bytes (most %llu of %llu)", (ull)ps.all.created,
             (ull)ps.all.live, (ull)ps.all.peak, (ull)ps.all.bytes, (ull)ps.all.peak_bytes, (ull)ps.gpu_no_access,
             (ull)ps.window_bytes, (ull)ps.window_peak, (ull)GFX_GPU_WINDOW_SIZE);
    emit(ctx, "pools", ps.all.created, why);
    static const char *const names[NVN_POOL_CLASSES] = {"pools-cpu", "pools-cpu-no-access", "pools-physical",
                                                        "pools-virtual"};
    for (int i = 0; i < NVN_POOL_CLASSES; i++) {
        const NvnPoolClass *k = &ps.cls[i];
        snprintf(why, sizeof why, "made %llu, live %llu (most at once %llu), %llu bytes live (most %llu)", (ull)k->created,
                 (ull)k->live, (ull)k->peak, (ull)k->bytes, (ull)k->peak_bytes);
        emit(ctx, names[i], k->created, why);
    }
    emit(ctx, "presents", nvn_present_count(), "nvnQueuePresentTexture calls that reached the back end");
}

static void rows(FILE *f, const char *phase, uint64_t frames, const uint64_t *now, const uint64_t *before, int n) {
    for (int i = 0; i < n; i++) {
        uint64_t d = now[i] - (before ? before[i] : 0);
        if (!d) continue;
        const char *name = rt_hostfn_name(RT_HOSTFN_BASE + 16ull * (uint64_t)i);
        if (!name || strncmp(name, "nvn", 3)) continue;
        if (frames) fprintf(f, "%s\t%llu\t\t%s\t%.3f\t%llu\n", phase, (unsigned long long)frames, name,
                            (double)d / (double)frames, (unsigned long long)d);
        else fprintf(f, "%s\t0\t\t%s\t%llu\t%llu\n", phase, name, (unsigned long long)d, (unsigned long long)d);
    }
}

// The census-mode file: window comment lines first (#window name presents a-b seconds), then the table.
void nvn_census_write(FILE *f) {
    int n = rt_hostfn_count();
    uint64_t *cur = malloc((size_t)n * sizeof *cur);
    if (!cur) return;
    rt_hostfn_snapshot(cur, n);
    uint64_t seq = nvn_present_count(), now_ns = plat_now_ns();
    plat_mutex_lock(&g_lock);
    fprintf(f, "#census-mode OOT2_CENSUS=1: not a default run\n#presents %llu\n", (unsigned long long)seq);
    for (int k = 1; k < g_nsnap; k++)
        fprintf(f, "#window w%03d presents %llu-%llu seconds %.3f\n", k, (unsigned long long)g_snap[k - 1].present,
                (unsigned long long)g_snap[k].present, (double)(g_snap[k].ns - g_snap[k - 1].ns) / 1e9);
    if (g_nsnap && seq > g_snap[g_nsnap - 1].present)
        fprintf(f, "#window tail presents %llu-%llu seconds %.3f\n", (unsigned long long)g_snap[g_nsnap - 1].present,
                (unsigned long long)seq, (double)(now_ns - g_snap[g_nsnap - 1].ns) / 1e9);
    fprintf(f, "phase\tframes\tslot\tfunction\tcalls_per_frame\tcalls\n");
    if (!g_nsnap) {
        rows(f, "boot", 0, cur, NULL, n);
    } else {
        rows(f, "boot", 0, g_snap[0].calls, NULL, g_nslots < n ? g_nslots : n);
        char name[16];
        for (int k = 1; k < g_nsnap; k++) {
            snprintf(name, sizeof name, "w%03d", k);
            rows(f, name, g_snap[k].present - g_snap[k - 1].present, g_snap[k].calls, g_snap[k - 1].calls, g_nslots);
        }
        const Snap *last = &g_snap[g_nsnap - 1];
        if (seq > last->present) rows(f, "tail", seq - last->present, cur, last->calls, g_nslots);
        if (seq > 1) rows(f, "all", seq - 1, cur, g_snap[0].calls, g_nslots);
    }
    plat_mutex_unlock(&g_lock);
    free(cur);
}
