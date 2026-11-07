// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The settings store (oot2/settings.h; research 93 D1, D5; ticket 103). Written here, after the shape of the stores
// research 93 read: Dusklight's layers of default, value and override (F4), TotK Recomp's per-key live-or-restart
// table and its line editing of the file (F8). No code is taken from either.
//
// One lock guards the layers, the file's bytes and the hooks; a published value is an atomic word that readers load
// without it. Every change (load, set, poll) recomputes each key from its layers and publishes under the lock, so two
// changes never interleave their hooks.
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#include <io.h>
#include <windows.h>
#else
#include <unistd.h>
#endif

#include "core.h"
#include "oot2/settings.h"
#include "plat.h"

#define FILE_MAX (1u << 20)   // a larger file is refused: a settings file holds a few hundred bytes
#define PATH_BYTES 4096
#define WHY_BYTES 600
#define MAX_HOOKS 32
#define WATCH_STACK (256u << 10)
#define DEFAULT_FROM "the default"

static RtSettingRow g_rows[RT_SETTING_COUNT] = {
#define RT_SETTING_ROW(key, group, type, apply, def, min, max, names, what)                                            \
    {#key, RT_GROUP_##group, RT_SETTING_##type, RT_APPLY_##apply, def, min, max, names, what, NULL},
    RT_SETTINGS(RT_SETTING_ROW)
#undef RT_SETTING_ROW
};
static char g_env[RT_SETTING_COUNT][48];   // OOT2_<KEY>
static PlatMutex g_table_lock = PLAT_MUTEX_INIT;
static int g_table_done;

typedef struct Layer {
    int set;
    int64_t v;
    const char *from;
} Layer;

static PlatMutex g_lock = PLAT_MUTEX_INIT;   // everything below but g_value and g_changes
static Layer g_layers[RT_SETTING_COUNT][RT_SOURCE_COUNT];
static Layer g_disk[RT_SETTING_COUNT];     // what the file held when it was last read, applied or saved
static int g_dirty[RT_SETTING_COUNT];      // rt_settings_set changed the file layer since then
static int64_t g_value[RT_SETTING_COUNT];  // published: atomic, read without the lock
static RtSettingSource g_src[RT_SETTING_COUNT];
static const char *g_from[RT_SETTING_COUNT];
static int g_pending_set[RT_SETTING_COUNT];   // a restart or prepare key's change was logged, with this value
static int64_t g_pending[RT_SETTING_COUNT];
static uint64_t g_changes;                 // atomic
static int g_ready;                        // atomic: the layers are read (rt_settings_load, or the first read)
static int g_watching;                     // atomic
static int g_fail_rename;
static char g_path[PATH_BYTES];

typedef struct Bytes {
    char *p;
    size_t n;
    int exists;
} Bytes;
static Bytes g_seen;       // the file as last applied (or loaded, or saved)
static Bytes g_next;       // the previous poll's read, which differed from g_seen
static int g_has_next, g_read_failed;

static struct {
    int id;
    RtSettingHook fn;
    void *ctx;
} g_hooks[MAX_HOOKS];
static int g_nhooks;

// ---- the table
static void table_init(void) {
    if (__atomic_load_n(&g_table_done, __ATOMIC_ACQUIRE)) return;
    plat_mutex_lock(&g_table_lock);
    if (!g_table_done) {
        for (int id = 0; id < RT_SETTING_COUNT; id++) {
            RtSettingRow *r = &g_rows[id];
            int n = snprintf(g_env[id], sizeof g_env[id], "OOT2_");
            for (const char *k = r->key; *k && n + 1 < (int)sizeof g_env[id]; k++)
                g_env[id][n++] = *k >= 'a' && *k <= 'z' ? (char)(*k - 'a' + 'A') : *k;
            g_env[id][n] = 0;
            r->env = g_env[id];
            if (r->type == RT_SETTING_ENUM) {
                r->min = 0;
                r->max = 0;
                for (const char *p = r->names; *p; p++) r->max += *p == '|';
            }
        }
        __atomic_store_n(&g_table_done, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&g_table_lock);
}

const RtSettingRow *rt_setting_row(RtSettingId id) {
    table_init();
    return id >= 0 && id < RT_SETTING_COUNT ? &g_rows[id] : NULL;
}

static int find_n(const char *key, size_t n) {
    for (int id = 0; id < RT_SETTING_COUNT; id++)
        if (strlen(g_rows[id].key) == n && !memcmp(g_rows[id].key, key, n)) return id;
    return -1;
}

int rt_setting_find(const char *key) { return key ? find_n(key, strlen(key)) : -1; }

const char *rt_setting_group_name(RtSettingGroup g) {
    static const char *const names[RT_GROUP_COUNT] = {"graphics", "pacing", "display", "audio",
                                                      "input", "enhancements", "debug"};
    return g >= 0 && g < RT_GROUP_COUNT ? names[g] : "?";
}

const char *rt_setting_apply_name(RtSettingApply a) {
    return a == RT_APPLY_LIVE ? "live" : a == RT_APPLY_RESTART ? "restart" : a == RT_APPLY_PREPARE ? "prepare" : "?";
}

const char *rt_setting_source_name(RtSettingSource s) {
    static const char *const names[RT_SOURCE_COUNT] = {"default", "file", "environment", "command line"};
    return s >= 0 && s < RT_SOURCE_COUNT ? names[s] : "?";
}

// ---- values
// The text s (n bytes) as key r takes it: 1 with *out, or 0. BOOL and INT: an optional '-' and decimal digits, nothing
// else (no space, no sign '+', no other base), from min to max; ENUM: one of the names, exactly.
static int parse_value(const RtSettingRow *r, const char *s, size_t n, int64_t *out) {
    if (r->type == RT_SETTING_ENUM) {
        int64_t i = 0;
        for (const char *p = r->names;; i++) {
            const char *bar = strchr(p, '|');
            size_t len = bar ? (size_t)(bar - p) : strlen(p);
            if (len == n && !memcmp(p, s, n)) {
                *out = i;
                return 1;
            }
            if (!bar) return 0;
            p = bar + 1;
        }
    }
    size_t i = n && s[0] == '-';
    if (i == n) return 0;
    uint64_t v = 0;
    for (size_t k = i; k < n; k++) {
        if (s[k] < '0' || s[k] > '9') return 0;
        unsigned d = (unsigned)(s[k] - '0');
        if (v > ((uint64_t)INT64_MAX - d) / 10) return 0;
        v = v * 10 + d;
    }
    int64_t x = i ? -(int64_t)v : (int64_t)v;
    // A default outside min..max is the key's off value and is taken as well (frame_rate_target: 0, or 20 to 120).
    if ((x < r->min || x > r->max) && x != r->def) return 0;
    *out = x;
    return 1;
}

// What key r takes, for a refusal: "0 or 1", "an integer from 0 to 9", "off, count or release".
static const char *takes(const RtSettingRow *r, char *out, size_t size) {
    if (r->type == RT_SETTING_BOOL) {
        snprintf(out, size, "0 or 1");
    } else if (r->type == RT_SETTING_INT) {
        if (r->def < r->min || r->def > r->max)   // a default outside the range is an off value, taken as well
            snprintf(out, size, "an integer from %lld to %lld, or %lld", (long long)r->min, (long long)r->max,
                     (long long)r->def);
        else snprintf(out, size, "an integer from %lld to %lld", (long long)r->min, (long long)r->max);
    } else {
        size_t k = 0;
        int64_t i = 0;
        out[0] = 0;
        for (const char *p = r->names; k + 1 < size; i++) {
            const char *bar = strchr(p, '|');
            int len = bar ? (int)(bar - p) : (int)strlen(p);
            const char *sep = i == 0 ? "" : i == r->max ? " or " : ", ";
            int w = snprintf(out + k, size - k, "%s%.*s", sep, len, p);
            if (w < 0) break;
            k += (size_t)w < size - k ? (size_t)w : size - k - 1;
            if (!bar) break;
            p = bar + 1;
        }
    }
    return out;
}

const char *rt_setting_format(RtSettingId id, int64_t value, char out[RT_SETTING_TEXT]) {
    const RtSettingRow *r = rt_setting_row(id);
    if (!r) {
        snprintf(out, RT_SETTING_TEXT, "?");
    } else if (r->type != RT_SETTING_ENUM) {
        snprintf(out, RT_SETTING_TEXT, "%lld", (long long)value);
    } else {
        const char *p = r->names;
        for (int64_t i = 0; i < value && p; i++) {
            p = strchr(p, '|');
            if (p) p++;
        }
        const char *bar = p ? strchr(p, '|') : NULL;
        if (p && value >= 0) snprintf(out, RT_SETTING_TEXT, "%.*s", bar ? (int)(bar - p) : (int)strlen(p), p);
        else snprintf(out, RT_SETTING_TEXT, "?");
    }
    return out;
}

// ---- files: whole reads, and a write that replaces the file by a rename
#ifdef _WIN32
static int wide(const char *path, wchar_t *out, int count) {
    return MultiByteToWideChar(CP_UTF8, 0, path, -1, out, count) > 0;
}
#endif

// The file whole into *b (exists 0 and no bytes when it does not exist): 1, or 0 with why.
static int read_whole(const char *path, Bytes *b, char *why, size_t wn) {
    b->p = NULL, b->n = 0, b->exists = 0;
    char *buf = NULL;
    size_t n = 0;
#ifdef _WIN32
    wchar_t w[PATH_BYTES];
    if (!wide(path, w, PATH_BYTES)) {
        snprintf(why, wn, "%s: not a UTF-8 path", path);
        return 0;
    }
    // Others may write, rename or delete the file while it is open: the watcher never blocks an editor or a save.
    HANDLE f = CreateFileW(w, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE, NULL,
                           OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
    if (f == INVALID_HANDLE_VALUE) {
        DWORD e = GetLastError();
        if (e == ERROR_FILE_NOT_FOUND || e == ERROR_PATH_NOT_FOUND) return 1;
        snprintf(why, wn, "%s: cannot be read (Windows error %lu)", path, (unsigned long)e);
        return 0;
    }
    LARGE_INTEGER len;
    int ok = GetFileSizeEx(f, &len) && len.QuadPart <= FILE_MAX;
    if (ok) {
        buf = malloc((size_t)len.QuadPart + 1);
        DWORD got = 0;
        ok = buf && ReadFile(f, buf, (DWORD)len.QuadPart, &got, NULL);
        n = got;
    }
    CloseHandle(f);
#else
    FILE *f = fopen(path, "rb");
    if (!f) {
        if (errno == ENOENT || errno == ENOTDIR) return 1;
        snprintf(why, wn, "%s: cannot be read (%s)", path, strerror(errno));
        return 0;
    }
    buf = malloc(FILE_MAX + 2);
    int ok = buf != NULL;
    if (ok) {
        n = fread(buf, 1, FILE_MAX + 1, f);
        ok = !ferror(f) && n <= FILE_MAX;
    }
    fclose(f);
#endif
    if (!ok) {
        free(buf);
        snprintf(why, wn, "%s: cannot be read, or holds more than %u bytes", path, FILE_MAX);
        return 0;
    }
    buf[n] = 0;
    b->p = buf, b->n = n, b->exists = 1;
    return 1;
}

static void remove_file(const char *path) {
#ifdef _WIN32
    wchar_t w[PATH_BYTES];
    if (wide(path, w, PATH_BYTES)) DeleteFileW(w);
#else
    remove(path);
#endif
}

// Write b into path's temporary file, close it, then rename it over path: 1, or 0 with why and path untouched.
static int write_replace(const char *path, const char *b, size_t n, char *why, size_t wn) {
    char tmp[PATH_BYTES + 8];
    snprintf(tmp, sizeof tmp, "%s.tmp", path);
#ifdef _WIN32
    wchar_t wt[PATH_BYTES + 8], wd[PATH_BYTES];
    if (!wide(tmp, wt, PATH_BYTES + 8) || !wide(path, wd, PATH_BYTES)) {
        snprintf(why, wn, "%s: not a UTF-8 path", path);
        return 0;
    }
    FILE *f = _wfopen(wt, L"wb");
#else
    FILE *f = fopen(tmp, "wb");
#endif
    if (!f) {
        snprintf(why, wn, "%s: cannot be created (%s)", tmp, strerror(errno));
        return 0;
    }
    int ok = fwrite(b, 1, n, f) == n && !fflush(f);
#ifdef _WIN32
    ok = ok && !_commit(_fileno(f));
#else
    ok = ok && !fsync(fileno(f));
#endif
    ok = !fclose(f) && ok;
    if (!ok) {
        snprintf(why, wn, "%s: cannot be written (%s)", tmp, strerror(errno));
        remove_file(tmp);
        return 0;
    }
    if (g_fail_rename) {
        ok = 0;
        snprintf(why, wn, "%s: the rename over %s failed (refused for a test)", tmp, path);
    } else {
#ifdef _WIN32
        ok = MoveFileExW(wt, wd, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
        if (!ok) snprintf(why, wn, "%s: the rename over %s failed (Windows error %lu)", tmp, path,
                          (unsigned long)GetLastError());
#else
        ok = !rename(tmp, path);
        if (!ok) snprintf(why, wn, "%s: the rename over %s failed (%s)", tmp, path, strerror(errno));
#endif
    }
    if (!ok) remove_file(tmp);
    return ok;
}

void rt_settings_fail_rename(int on) { g_fail_rename = on; }

// ---- the file's lines
enum { LINE_BLANK, LINE_SECTION, LINE_KEY, LINE_BAD };
typedef struct Line {
    size_t start, end, next;   // the line is [start, end), its line break [end, next)
    int kind;
    size_t a, a_len;           // LINE_KEY: the key; LINE_SECTION: the name
    size_t v, v_len;           // LINE_KEY: the value
} Line;

static int blank(char c) { return c == ' ' || c == '\t'; }

// Split b into lines (a UTF-8 byte order mark at the start is not part of the first line's text). The caller frees;
// NULL when out of memory. Nothing here logs or stops the run: the callers hold g_lock, and a log line reads census
// mode from this store.
static Line *scan(const char *b, size_t n, int *count) {
    int lines = 1;
    for (size_t i = 0; i < n; i++) lines += b[i] == '\n';
    *count = 0;
    Line *out = calloc((size_t)lines, sizeof *out);
    if (!out) return NULL;
    size_t at = n >= 3 && !memcmp(b, "\xEF\xBB\xBF", 3) ? 3 : 0;
    int k = 0;
    for (size_t s = 0; s <= n && k < lines; k++) {
        Line *l = &out[k];
        const char *nl = memchr(b + s, '\n', n - s);
        size_t e = nl ? (size_t)(nl - b) : n;
        l->start = s, l->next = nl ? e + 1 : n;
        l->end = e > s && b[e - 1] == '\r' && nl ? e - 1 : e;
        size_t p = s < at ? at : s, q = l->end;
        while (p < q && blank(b[p])) p++;
        while (q > p && blank(b[q - 1])) q--;
        const char *eq = memchr(b + p, '=', q - p);
        if (p == q || b[p] == ';' || b[p] == '#') {
            l->kind = LINE_BLANK;
        } else if (b[p] == '[' && b[q - 1] == ']' && q - p >= 2) {
            l->kind = LINE_SECTION;
            size_t a = p + 1, z = q - 1;
            while (a < z && blank(b[a])) a++;
            while (z > a && blank(b[z - 1])) z--;
            l->a = a, l->a_len = z - a;
        } else if (eq && (size_t)(eq - b) > p) {
            l->kind = LINE_KEY;
            size_t z = (size_t)(eq - b), v = z + 1;
            while (z > p && blank(b[z - 1])) z--;
            while (v < q && blank(b[v])) v++;
            l->a = p, l->a_len = z - p, l->v = v, l->v_len = q - v;
        } else {
            l->kind = LINE_BAD;
        }
        if (!nl) {
            k++;
            break;
        }
        s = e + 1;
    }
    *count = k;
    return out;
}

// The file layer b gives: 1 with out[], or 0 with why (the first line that refuses the file). log_unknown: log each key
// the table does not have (only once the store is published: see scan).
static int read_layer(const Bytes *b, Layer out[RT_SETTING_COUNT], int log_unknown, char *why, size_t wn) {
    int count, ok = 1, at[RT_SETTING_COUNT];
    for (int id = 0; id < RT_SETTING_COUNT; id++) out[id] = (Layer){0}, at[id] = -1;
    Line *lines = scan(b->p ? b->p : "", b->n, &count);
    if (!lines) {
        snprintf(why, wn, "%s: out of memory to read it", g_path);
        return 0;
    }
    for (int i = 0; i < count && ok; i++) {
        const Line *l = &lines[i];
        if (l->kind == LINE_BAD) {
            snprintf(why, wn, "%s line %d: not a comment, a [section] or key = value", g_path, i + 1);
            ok = 0;
        } else if (l->kind == LINE_KEY) {
            int id = find_n(b->p + l->a, l->a_len);
            if (id < 0) {
                if (log_unknown)
                    rt_log("settings: %s line %d: no setting is named %.*s; the line is kept and ignored", g_path,
                           i + 1, (int)l->a_len, b->p + l->a);
                continue;
            }
            const RtSettingRow *r = &g_rows[id];
            int64_t v;
            char t[160];
            if (at[id] >= 0) {
                snprintf(why, wn, "%s lines %d and %d both set %s", g_path, at[id] + 1, i + 1, r->key);
                ok = 0;
            } else if (!parse_value(r, b->p + l->v, l->v_len, &v)) {
                snprintf(why, wn, "%s line %d: %s = %.*s: %s takes %s", g_path, i + 1, r->key, (int)l->v_len,
                         b->p + l->v, r->key, takes(r, t, sizeof t));
                ok = 0;
            } else {
                at[id] = i;
                out[id] = (Layer){1, v, g_path};
            }
        }
    }
    free(lines);
    return ok;
}

static int read_env(Layer out[RT_SETTING_COUNT], char *why, size_t wn) {
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        const RtSettingRow *r = &g_rows[id];
        const char *e = getenv(r->env);
        int64_t v;
        char t[160];
        out[id] = (Layer){0};
        if (!e || !*e) continue;   // an empty variable sets nothing, as before the store
        if (!parse_value(r, e, strlen(e), &v)) {
            snprintf(why, wn, "%s=%s: %s takes %s", r->env, e, r->key, takes(r, t, sizeof t));
            return 0;
        }
        out[id] = (Layer){1, v, r->env};
    }
    return 1;
}

static int read_command(const RtSettingArg *args, int count, Layer out[RT_SETTING_COUNT], char *why, size_t wn) {
    for (int id = 0; id < RT_SETTING_COUNT; id++) out[id] = (Layer){0};
    for (int i = 0; i < count; i++) {
        const char *t = args[i].text, *eq = strchr(t, '=');
        if (!eq || eq == t) {
            snprintf(why, wn, "%s %s: not key=value", args[i].from, t);
            return 0;
        }
        int id = find_n(t, (size_t)(eq - t));
        if (id < 0) {
            snprintf(why, wn, "%s %s: no setting is named %.*s", args[i].from, t, (int)(eq - t), t);
            return 0;
        }
        const RtSettingRow *r = &g_rows[id];
        int64_t v;
        char k[160];
        if (!parse_value(r, eq + 1, strlen(eq + 1), &v)) {
            snprintf(why, wn, "%s %s: %s takes %s", args[i].from, t, r->key, takes(r, k, sizeof k));
            return 0;
        }
        out[id] = (Layer){1, v, args[i].from};
    }
    return 1;
}

// ---- publishing (g_lock held)
static RtSettingSource top(int id) {
    int s = RT_SOURCE_COUNT - 1;
    while (s > RT_SOURCE_DEFAULT && !g_layers[id][s].set) s--;
    return (RtSettingSource)s;
}

static void run_hooks(int id, int64_t old, int64_t v, RtSettingSource s, const char *from, int pending) {
    RtSettingChange c = {(RtSettingId)id, old, v, s, from, pending};
    for (int i = 0; i < g_nhooks; i++)
        if (g_hooks[i].id == RT_SETTING_ANY || g_hooks[i].id == id) g_hooks[i].fn(&c, g_hooks[i].ctx);
}

// Publish what the layers give now. starting (rt_settings_load, the first read): every key, no log, hook or count.
// Later, a live key whose value changed is published, counted, logged and handed to its hooks; a restart or prepare key
// keeps its value, and its change is logged once and handed to the hooks as pending.
static void settle(int starting) {
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        const RtSettingRow *r = &g_rows[id];
        RtSettingSource s = top(id);
        const Layer *l = &g_layers[id][s];
        int64_t old = g_value[id];
        char a[RT_SETTING_TEXT], b[RT_SETTING_TEXT];
        if (starting || l->v == old) {
            __atomic_store_n(&g_value[id], l->v, __ATOMIC_RELEASE);
            g_src[id] = s, g_from[id] = l->from, g_pending_set[id] = 0;
        } else if (r->apply == RT_APPLY_LIVE) {
            __atomic_store_n(&g_value[id], l->v, __ATOMIC_RELEASE);
            g_src[id] = s, g_from[id] = l->from;
            __atomic_add_fetch(&g_changes, 1, __ATOMIC_ACQ_REL);
            rt_log("settings: %s = %s (%s), applied now; it was %s", r->key, rt_setting_format(id, l->v, a), l->from,
                   rt_setting_format(id, old, b));
            run_hooks(id, old, l->v, s, l->from, 0);
        } else if (!g_pending_set[id] || g_pending[id] != l->v) {
            g_pending_set[id] = 1, g_pending[id] = l->v;
            rt_log("settings: %s = %s (%s) needs %s; this run keeps %s", r->key, rt_setting_format(id, l->v, a),
                   l->from, r->apply == RT_APPLY_RESTART ? "a restart" : "oot2 prepare", rt_setting_format(id, old, b));
            run_hooks(id, old, l->v, s, l->from, 1);
        }
    }
}

static void set_defaults(void) {
    for (int id = 0; id < RT_SETTING_COUNT; id++)
        g_layers[id][RT_SOURCE_DEFAULT] = (Layer){1, g_rows[id].def, DEFAULT_FROM};
}

// The first read before rt_settings_load: the environment and the defaults. A variable its key does not take ends the
// process with exit code 2, after the defaults alone are published, so a read made on the way out (a log line reads
// census mode) finds the store ready and never comes back here.
static void ready(void) {
    if (__atomic_load_n(&g_ready, __ATOMIC_ACQUIRE)) return;
    table_init();
    plat_mutex_lock(&g_lock);
    int ok = 1;
    char why[WHY_BYTES];
    if (!g_ready) {
        Layer env[RT_SETTING_COUNT];
        ok = read_env(env, why, sizeof why);
        set_defaults();
        for (int id = 0; id < RT_SETTING_COUNT; id++) g_layers[id][RT_SOURCE_ENV] = ok ? env[id] : (Layer){0};
        settle(1);
        __atomic_store_n(&g_ready, 1, __ATOMIC_RELEASE);
    }
    plat_mutex_unlock(&g_lock);
    if (!ok) {
        fprintf(stderr, "oot2: %s\n", why);
        plat_exit(2);
    }
}

static int same(const Bytes *a, const Bytes *b) {
    return a->exists == b->exists && a->n == b->n && (!a->n || !memcmp(a->p, b->p, a->n));
}

static void take(Bytes *to, Bytes *from) {
    free(to->p);
    *to = *from;
    *from = (Bytes){0};
}

// ---- start-up
int rt_settings_load(const RtSettingArg *args, int count, const char *path, char *why, size_t why_size) {
    table_init();
    plat_mutex_lock(&g_lock);
    Layer cmd[RT_SETTING_COUNT], env[RT_SETTING_COUNT], file[RT_SETTING_COUNT] = {{0}};
    Bytes b = {0};
    int ok = read_command(args, count, cmd, why, why_size) && read_env(env, why, why_size);
    g_path[0] = 0;
    if (ok && path && *path) {
        if (snprintf(g_path, sizeof g_path, "%s", path) >= (int)sizeof g_path) {
            snprintf(why, why_size, "the settings file's path is longer than %d bytes", PATH_BYTES - 1);
            g_path[0] = 0;
            ok = 0;
        } else {
            ok = read_whole(g_path, &b, why, why_size) && read_layer(&b, file, 0, why, why_size);
        }
    }
    if (ok) {
        set_defaults();
        for (int id = 0; id < RT_SETTING_COUNT; id++) {
            g_layers[id][RT_SOURCE_FILE] = g_disk[id] = file[id];
            g_layers[id][RT_SOURCE_ENV] = env[id];
            g_layers[id][RT_SOURCE_COMMAND] = cmd[id];
            g_dirty[id] = 0;
        }
        settle(1);
        take(&g_seen, &b);
        free(g_next.p);
        g_next = (Bytes){0};
        g_has_next = g_read_failed = 0;
        __atomic_store_n(&g_ready, 1, __ATOMIC_RELEASE);
        // Logged only now that every value is published: a log line reads census mode, and the first read fixes it.
        read_layer(&g_seen, file, 1, why, why_size);
    }
    free(b.p);
    plat_mutex_unlock(&g_lock);
    return ok;
}

const char *rt_settings_path(void) { return g_path; }

// ---- reading
int64_t rt_setting(RtSettingId id) {
    if (id < 0 || id >= RT_SETTING_COUNT) rt_fatal("settings: no setting has the id %d", (int)id);
    ready();
    return __atomic_load_n(&g_value[id], __ATOMIC_ACQUIRE);
}

RtSettingSource rt_setting_source(RtSettingId id, const char **from) {
    if (id < 0 || id >= RT_SETTING_COUNT) rt_fatal("settings: no setting has the id %d", (int)id);
    ready();
    plat_mutex_lock(&g_lock);
    RtSettingSource s = g_src[id];
    if (from) *from = g_from[id];
    plat_mutex_unlock(&g_lock);
    return s;
}

int rt_setting_pending(RtSettingId id, int64_t *value) {
    if (id < 0 || id >= RT_SETTING_COUNT) return 0;
    ready();
    plat_mutex_lock(&g_lock);
    int64_t v = g_layers[id][top(id)].v;
    int pending = g_rows[id].apply != RT_APPLY_LIVE && v != g_value[id];
    plat_mutex_unlock(&g_lock);
    if (pending && value) *value = v;
    return pending;
}

uint64_t rt_settings_changes(void) { return __atomic_load_n(&g_changes, __ATOMIC_ACQUIRE); }

int rt_settings_nondefault(RtSettingNow *out, int max) {
    ready();
    plat_mutex_lock(&g_lock);
    int k = 0;
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        if (g_value[id] == g_rows[id].def) continue;
        if (k < max) {
            RtSettingNow *o = &out[k];
            o->id = (RtSettingId)id, o->key = g_rows[id].key, o->source = g_src[id], o->from = g_from[id];
            rt_setting_format(id, g_value[id], o->value);
        }
        k++;
    }
    plat_mutex_unlock(&g_lock);
    return k;
}

// ---- changing
int rt_settings_set(RtSettingId id, const char *text, char *why, size_t why_size) {
    const RtSettingRow *r = rt_setting_row(id);
    int64_t v;
    char t[160];
    if (!r) {
        snprintf(why, why_size, "no setting has the id %d", (int)id);
        return 0;
    }
    if (!text || !parse_value(r, text, strlen(text), &v)) {
        snprintf(why, why_size, "%s = %s: %s takes %s", r->key, text ? text : "(nothing)", r->key, takes(r, t, sizeof t));
        return 0;
    }
    ready();
    plat_mutex_lock(&g_lock);
    g_layers[id][RT_SOURCE_FILE] = (Layer){1, v, g_path[0] ? g_path : "the settings file"};
    g_dirty[id] = 1;
    settle(0);
    plat_mutex_unlock(&g_lock);
    return 1;
}

// A growable byte buffer for the save.
typedef struct Out {
    char *p;
    size_t n, cap;
    int failed;   // out of memory: the save is refused (no fatal under g_lock, see scan)
} Out;

static void put(Out *o, const char *s, size_t n) {
    if (o->failed) return;
    if (o->n + n + 1 > o->cap) {
        size_t cap = o->cap ? o->cap : 256;
        while (o->n + n + 1 > cap) cap *= 2;
        char *p = realloc(o->p, cap);
        if (!p) {
            o->failed = 1;
            return;
        }
        o->p = p, o->cap = cap;
    }
    memcpy(o->p + o->n, s, n);
    o->n += n;
    o->p[o->n] = 0;
}

static void put_s(Out *o, const char *s) { put(o, s, strlen(s)); }

static void put_key(Out *o, int id, const char *nl) {
    char v[RT_SETTING_TEXT];
    put_s(o, g_rows[id].key);
    put_s(o, " = ");
    put_s(o, rt_setting_format(id, g_layers[id][RT_SOURCE_FILE].v, v));
    put_s(o, nl);
}

// The file's new bytes: disk's, with each changed key's value written in its own line, and each changed key the file
// lacks added at the end of its group's section, or in a new section at the end. 1, or 0 with why.
static int rewrite(const Bytes *disk, Out *o, char *why, size_t wn) {
    int count, line_of[RT_SETTING_COUNT];
    const char *b = disk->p ? disk->p : "";
    Line *lines = scan(b, disk->n, &count);
    if (!lines) {
        snprintf(why, wn, "%s: out of memory to read it", g_path);
        return 0;
    }
    const char *nl = strstr(b, "\r\n") ? "\r\n" : "\n";
    // Each key's line, whatever its value: a line the store cannot read is not the save's to judge, and is kept as it
    // is. A key named twice makes "its line" ambiguous, so the save is refused as a load would refuse the file.
    for (int id = 0; id < RT_SETTING_COUNT; id++) line_of[id] = -1;
    for (int i = 0; i < count; i++) {
        int id = lines[i].kind == LINE_KEY ? find_n(b + lines[i].a, lines[i].a_len) : -1;
        if (id < 0) continue;
        if (line_of[id] >= 0) {
            snprintf(why, wn, "%s lines %d and %d both set %s", g_path, line_of[id] + 1, i + 1, g_rows[id].key);
            free(lines);
            return 0;
        }
        line_of[id] = i;
    }
    // after[g]: the line a key of group g that the file lacks goes after: the last line that is not blank or a comment
    // in the first section named after the group (its header when it has none); -1 when no section has that name.
    int after[RT_GROUP_COUNT], in = RT_GROUP_COUNT;
    for (int g = 0; g < RT_GROUP_COUNT; g++) after[g] = -1;
    for (int i = 0; i < count; i++) {
        const Line *l = &lines[i];
        if (l->kind == LINE_SECTION) {
            in = RT_GROUP_COUNT;
            for (int g = 0; g < RT_GROUP_COUNT; g++) {
                const char *name = rt_setting_group_name((RtSettingGroup)g);
                if (strlen(name) == l->a_len && !memcmp(name, b + l->a, l->a_len) && after[g] < 0) in = g;
            }
            if (in < RT_GROUP_COUNT) after[in] = i;
        } else if (l->kind != LINE_BLANK && in < RT_GROUP_COUNT) {
            after[in] = i;
        }
    }
    for (int i = 0; i < count; i++) {
        const Line *l = &lines[i];
        int id = -1;
        for (int k = 0; k < RT_SETTING_COUNT && l->kind == LINE_KEY; k++)
            if (g_dirty[k] && line_of[k] == i) id = k;
        if (id < 0) {
            put(o, b + l->start, l->end - l->start);
        } else {
            char v[RT_SETTING_TEXT];
            put(o, b + l->start, l->v - l->start);
            put_s(o, rt_setting_format(id, g_layers[id][RT_SOURCE_FILE].v, v));
            put(o, b + l->v + l->v_len, l->end - (l->v + l->v_len));
        }
        put(o, b + l->end, l->next - l->end);
        int added = 0;
        for (int k = 0; k < RT_SETTING_COUNT; k++) {
            if (!g_dirty[k] || line_of[k] >= 0 || after[g_rows[k].group] != i) continue;
            if (!added && l->next == l->end) put_s(o, nl);   // the last line had no line break
            put_key(o, k, nl);
            added = 1;
        }
    }
    for (int g = 0; g < RT_GROUP_COUNT; g++) {
        int head = 0;
        for (int k = 0; k < RT_SETTING_COUNT; k++) {
            if (!g_dirty[k] || line_of[k] >= 0 || (int)g_rows[k].group != g || after[g] >= 0) continue;
            if (!head) {
                if (o->n && o->p[o->n - 1] != '\n') put_s(o, nl);
                if (o->n) put_s(o, nl);
                put_s(o, "[");
                put_s(o, rt_setting_group_name((RtSettingGroup)g));
                put_s(o, "]");
                put_s(o, nl);
                head = 1;
            }
            put_key(o, k, nl);
        }
    }
    free(lines);
    if (o->failed) snprintf(why, wn, "%s: out of memory for its new text", g_path);
    return !o->failed;
}

int rt_settings_save(char *why, size_t why_size) {
    ready();
    plat_mutex_lock(&g_lock);
    Bytes disk = {0};
    Out o = {0};
    int ok = 1, dirty = 0;
    for (int id = 0; id < RT_SETTING_COUNT; id++) dirty |= g_dirty[id];
    if (!g_path[0]) {
        snprintf(why, why_size, "there is no settings file (the game folder's oot2.ini, --config or OOT2_CONFIG)");
        ok = 0;
    } else if (dirty) {
        ok = read_whole(g_path, &disk, why, why_size) && rewrite(&disk, &o, why, why_size);
        if (ok && !(disk.exists && o.n == disk.n && (!o.n || !memcmp(o.p, disk.p, o.n))))
            ok = write_replace(g_path, o.p ? o.p : "", o.n, why, why_size);
        if (ok) {
            for (int id = 0; id < RT_SETTING_COUNT; id++)
                if (g_dirty[id]) g_disk[id] = g_layers[id][RT_SOURCE_FILE], g_dirty[id] = 0;
            Bytes now = {o.p, o.n, 1};
            o.p = NULL;
            take(&g_seen, &now);
            g_has_next = 0;
        }
    }
    free(disk.p);
    free(o.p);
    plat_mutex_unlock(&g_lock);
    return ok;
}

// ---- the file watcher
// A new file applied (g_lock held): each key whose value in the file changed takes it, over an edit rt_settings_set
// made and a save did not write yet; every other key keeps what it had.
static void apply_file(Bytes *b) {
    Layer file[RT_SETTING_COUNT];
    char why[WHY_BYTES];
    if (read_layer(b, file, 1, why, sizeof why)) {
        for (int id = 0; id < RT_SETTING_COUNT; id++) {
            const Layer *d = &g_disk[id];
            if (d->set != file[id].set || (d->set && d->v != file[id].v)) {
                g_layers[id][RT_SOURCE_FILE] = file[id];
                g_dirty[id] = 0;
            }
            g_disk[id] = file[id];
        }
        settle(0);
    } else {
        rt_log("settings: %s; the file is not applied, and every setting keeps its value", why);
    }
    take(&g_seen, b);
}

int rt_settings_poll(void) {
    ready();
    plat_mutex_lock(&g_lock);
    int r = 0;
    Bytes b;
    char why[WHY_BYTES];
    if (!g_path[0]) {
        r = 0;
    } else if (!read_whole(g_path, &b, why, sizeof why)) {
        if (!g_read_failed) rt_log("settings: %s; the watcher tries again", why);
        g_read_failed = 1;
    } else {
        g_read_failed = 0;
        if (same(&b, &g_seen)) {
            g_has_next = 0;
            free(b.p);
        } else if (!g_has_next || !same(&b, &g_next)) {
            take(&g_next, &b);   // seen once: applied when the next read agrees, so a half-written save is not
            g_has_next = 1;
            r = 1;
        } else {
            free(b.p);
            g_has_next = 0;
            apply_file(&g_next);
            r = 2;
        }
    }
    plat_mutex_unlock(&g_lock);
    return r;
}

static void watch_main(void *arg) {
    (void)arg;
    for (;;) {
        plat_sleep_ns(RT_SETTINGS_POLL_NS);
        rt_settings_poll();
    }
}

int rt_settings_watch(void) {
    ready();
    if (!g_path[0]) return 0;
    if (__atomic_exchange_n(&g_watching, 1, __ATOMIC_ACQ_REL)) return 1;
    if (plat_thread_start(watch_main, NULL, WATCH_STACK)) return 1;
    __atomic_store_n(&g_watching, 0, __ATOMIC_RELEASE);
    return 0;
}

// ---- hooks
void rt_settings_hook(int id, RtSettingHook fn, void *ctx) {
    if (!fn || (id != RT_SETTING_ANY && (id < 0 || id >= RT_SETTING_COUNT)))
        rt_fatal("settings: a hook for the id %d", id);
    plat_mutex_lock(&g_lock);
    if (g_nhooks == MAX_HOOKS) {
        plat_mutex_unlock(&g_lock);
        rt_fatal("settings: more than %d hooks", MAX_HOOKS);
    }
    g_hooks[g_nhooks].id = id, g_hooks[g_nhooks].fn = fn, g_hooks[g_nhooks].ctx = ctx;
    g_nhooks++;
    plat_mutex_unlock(&g_lock);
}
