// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The pacing trace (the acquire sync hold), for the game measurement the ticket asks for. Never part of a default run: it is off unless
// the environment variable OOT2_PACE_TRACE names a file, and then the run says so in the file's first line and in the log.
// AcquireTexture, SyncWait and QueuePresentTexture each write one line (event, wall nanoseconds since the first line, the
// virtual clock's tick, then what the event knows: the window's interval, the hold, the placement, the timeout, the result,
// the time the call took) and, for the acquire and the present, the values of the probes.
//
// A probe is a word of the guest's memory read at run time, from a file the run names (OOT2_PROBE_FILE). The file and its
// offsets belong to whoever measures; this file knows nothing of any game. One probe per line, '#' starts a comment:
//     label  base  [*+N ...]  =+N  type
// base is <module>+0xOFFSET (the module's load address from the dispatch table plus the offset) or @0xADDRESS (an absolute
// guest address). Each *+N adds N to the address so far and replaces it with the 64-bit word found there; =+N adds N and
// reads a value of the type u8, u16, u32, u64, i32, i64, f32 or f64 at the result. Every address is checked with
// rt_mem_query before it is read; a word that cannot be read prints "?". A line that does not parse stops the run by name.
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nvn_queue.h"
#include "oot2/memory.h"
#include "oot2/vsync.h"
#include "plat.h"

#define MAX_PROBES 32
#define MAX_DEREFS 8
typedef struct {
    char label[32];
    char module[16];
    uint64_t off;            // the offset from the module's base, or the absolute address when module is empty
    int nderef;
    uint64_t deref[MAX_DEREFS];
    uint64_t field;
    char type[4];
    int width;
} Probe;

static PlatMutex g_lock = PLAT_MUTEX_INIT;
static FILE *g_file;
static int g_on = -1;   // -1: the environment is not read yet
static uint64_t g_first_ns;
static Probe g_probe[MAX_PROBES];
static int g_nprobe;

static int type_width(const char *t) {
    if (!strcmp(t, "u8")) return 1;
    if (!strcmp(t, "u16")) return 2;
    if (!strcmp(t, "u32") || !strcmp(t, "i32") || !strcmp(t, "f32")) return 4;
    if (!strcmp(t, "u64") || !strcmp(t, "i64") || !strcmp(t, "f64")) return 8;
    return 0;
}

static uint64_t hex_arg(const char *s, const char *line) {
    char *end;
    if (s[0] != '0' || (s[1] != 'x' && s[1] != 'X')) rt_fatal("OOT2_PROBE_FILE: \"%s\": %s is not a 0x number", line, s);
    uint64_t v = strtoull(s, &end, 16);
    if (*end) rt_fatal("OOT2_PROBE_FILE: \"%s\": %s is not a 0x number", line, s);
    return v;
}

static void parse_probe(char *line) {
    char text[200];
    snprintf(text, sizeof text, "%s", line);
    size_t n = strlen(text);
    while (n && (text[n - 1] == '\n' || text[n - 1] == '\r')) text[--n] = 0;
    char *save, *tok = strtok_r(line, " \t\r\n", &save);
    if (!tok || tok[0] == '#') return;
    if (g_nprobe == MAX_PROBES) rt_fatal("OOT2_PROBE_FILE: more than %d probes", MAX_PROBES);
    Probe *p = &g_probe[g_nprobe];
    memset(p, 0, sizeof *p);
    snprintf(p->label, sizeof p->label, "%s", tok);
    char *base = strtok_r(NULL, " \t\r\n", &save);
    if (!base) rt_fatal("OOT2_PROBE_FILE: \"%s\": no base", text);
    if (base[0] == '@') {
        p->off = hex_arg(base + 1, text);
    } else {
        char *plus = strchr(base, '+');
        if (!plus || plus == base || (size_t)(plus - base) >= sizeof p->module)
            rt_fatal("OOT2_PROBE_FILE: \"%s\": the base is <module>+0xOFFSET or @0xADDRESS", text);
        memcpy(p->module, base, (size_t)(plus - base));
        p->off = hex_arg(plus + 1, text);
    }
    int have_field = 0;
    while ((tok = strtok_r(NULL, " \t\r\n", &save)) != NULL) {
        if (!strncmp(tok, "*+", 2)) {
            if (have_field || p->nderef == MAX_DEREFS) rt_fatal("OOT2_PROBE_FILE: \"%s\": misplaced or too many *+N", text);
            p->deref[p->nderef++] = hex_arg(tok + 2, text);
        } else if (!strncmp(tok, "=+", 2)) {
            if (have_field) rt_fatal("OOT2_PROBE_FILE: \"%s\": two =+N", text);
            p->field = hex_arg(tok + 2, text), have_field = 1;
        } else if (have_field && !p->width) {
            snprintf(p->type, sizeof p->type, "%s", tok);
            p->width = type_width(p->type);
            if (!p->width) rt_fatal("OOT2_PROBE_FILE: \"%s\": %s is not u8 u16 u32 u64 i32 i64 f32 f64", text, tok);
        } else {
            rt_fatal("OOT2_PROBE_FILE: \"%s\": cannot read \"%s\"", text, tok);
        }
    }
    if (!have_field || !p->width) rt_fatal("OOT2_PROBE_FILE: \"%s\": no =+N and type", text);
    g_nprobe++;
}

// g_lock held
static int init_locked(void) {
    const char *path = getenv("OOT2_PACE_TRACE");
    if (!path || !*path) return 0;
    g_file = fopen(path, "w");
    if (!g_file) rt_fatal("OOT2_PACE_TRACE: cannot write %s", path);
    fprintf(g_file, "#pace-trace OOT2_PACE_TRACE: not a default run\n");
    const char *pf = getenv("OOT2_PROBE_FILE");
    if (pf && *pf) {
        FILE *f = fopen(pf, "r");
        if (!f) rt_fatal("OOT2_PROBE_FILE: cannot read %s", pf);
        char line[200];
        while (fgets(line, sizeof line, f)) parse_probe(line);
        fclose(f);
        for (int i = 0; i < g_nprobe; i++) fprintf(g_file, "#probe %s\n", g_probe[i].label);
    }
    fflush(g_file);
    g_first_ns = plat_now_ns();
    return 1;
}

int nvn_trace_on(void) {
    int on = __atomic_load_n(&g_on, __ATOMIC_ACQUIRE);
    if (on >= 0) return on;
    int fresh = 0;
    plat_mutex_lock(&g_lock);
    if (g_on < 0) {
        int v = init_locked();
        __atomic_store_n(&g_on, v, __ATOMIC_RELEASE);
        fresh = v;
    }
    on = g_on;
    plat_mutex_unlock(&g_lock);
    if (fresh) rt_log("nvn: OOT2_PACE_TRACE: the pacing trace is on; this is not a default run");
    return on;
}

static int readable(uint64_t a, int w) {
    RtMemInfo m = rt_mem_query(a);
    return m.size && (m.perm & 1) && a >= m.base && a - m.base + (uint64_t)w <= m.size;
}

// The probe's value as text, or "?".
static void probe_text(const Probe *p, char *out, size_t n) {
    uint64_t cur = p->off;
    if (p->module[0]) {
        cur = rt_module_base(p->module);
        if (!cur) {
            snprintf(out, n, "?");
            return;
        }
        cur += p->off;
    }
    for (int i = 0; i < p->nderef; i++) {
        if (!readable(cur + p->deref[i], 8)) {
            snprintf(out, n, "?");
            return;
        }
        cur = *(const uint64_t *)(uintptr_t)(cur + p->deref[i]);
    }
    cur += p->field;
    if (!readable(cur, p->width)) {
        snprintf(out, n, "?");
        return;
    }
    const void *a = (const void *)(uintptr_t)cur;
    uint64_t u = 0;
    memcpy(&u, a, (size_t)p->width);
    if (p->type[0] == 'f' && p->width == 4) {
        float f;
        memcpy(&f, a, 4);
        snprintf(out, n, "%.9g", (double)f);
    } else if (p->type[0] == 'f') {
        double d;
        memcpy(&d, a, 8);
        snprintf(out, n, "%.17g", d);
    } else if (p->type[0] == 'i') {
        snprintf(out, n, "%lld", p->width == 4 ? (long long)(int32_t)u : (long long)u);
    } else {
        snprintf(out, n, "%llu", (unsigned long long)u);
    }
}

void nvn_trace(const char *event, int probes, const char *fmt, ...) {
    if (!nvn_trace_on()) return;
    char body[256], val[48];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(body, sizeof body, fmt, ap);
    va_end(ap);
    uint64_t now = plat_now_ns(), tick = rt_vsync_tick();
    plat_mutex_lock(&g_lock);
    fprintf(g_file, "%s %llu tick %llu %s", event, (unsigned long long)(now - g_first_ns), (unsigned long long)tick, body);
    if (probes && g_nprobe) {
        fprintf(g_file, " |");
        for (int i = 0; i < g_nprobe; i++) {
            probe_text(&g_probe[i], val, sizeof val);
            fprintf(g_file, " %s=%s", g_probe[i].label, val);
        }
    }
    fputc('\n', g_file);
    fflush(g_file);
    plat_mutex_unlock(&g_lock);
}
