// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Contract tests of the settings store (runtime/core/settings.c, oot2/settings.h), driven by
// tests/test_runtime_settings.py: `settings_test <case> [arguments]`, each case in its own process, with the environment
// and the settings files the driver wrote (synthetic text, nothing of the game). Linked with runtime/core and
// runtime/nvn, so the hold case reaches experiment E2's hold through its hook. Exit code 2 is the store's own refusal,
// as at oot2's start-up; 1 is a failed check.
//
//   table                         every row of the table, one tab-separated line each
//   show [--set K=V]... [--config P]   load, then every key: key=value<TAB>source<TAB>from
//   lazy                          no load: every key as a program that never loads the store sees it
//   nondefault [as show]          load, then the non-default listing and its count
//   save P K=V...                 load P, set each K=V, save
//   savefail P K=V...             the same with the rename refused: the save must fail
//   reload P NEW [--set K=V]...   load P, then write NEW's bytes over P and poll twice; hooks, values, the counter
//   counter P                     the change counter across sets, unchanged writes and a reformatted file
//   hold [--set K=V]... P         load, bind experiment E2's hook, then change hold_ticks by a set and by the file
//   watch P NEW                   load P, start the watcher thread, write NEW over P, wait for the change
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/nvn_pace.h"
#include "oot2/runtime.h"
#include "oot2/settings.h"
#include "plat.h"

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "check failed at line %d: %s\n", __LINE__, #cond);                                       \
            exit(1);                                                                                                   \
        }                                                                                                              \
    } while (0)

#define MAX_ARGS 32

static RtSettingArg g_args[MAX_ARGS];
static int g_nargs;
static const char *g_config;

// --set K=V and --config P from argv[from..]; any other word is a file path left in *rest (at most one).
static void parse(int argc, char **argv, int from, const char **rest) {
    for (int i = from; i < argc; i++) {
        if (!strcmp(argv[i], "--set") && i + 1 < argc) {
            CHECK(g_nargs < MAX_ARGS);
            g_args[g_nargs++] = (RtSettingArg){argv[++i], "--set"};
        } else if (!strcmp(argv[i], "--config") && i + 1 < argc) {
            g_config = argv[++i];
        } else {
            CHECK(rest && !*rest);
            *rest = argv[i];
        }
    }
}

static void load(void) {
    char why[600];
    if (!rt_settings_load(g_args, g_nargs, g_config, why, sizeof why)) {
        fprintf(stderr, "oot2: %s\n", why);
        exit(2);
    }
}

static void print_keys(void) {
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        const char *from;
        char v[RT_SETTING_TEXT];
        RtSettingSource s = rt_setting_source((RtSettingId)id, &from);
        printf("%s=%s\t%s\t%s\n", rt_setting_row((RtSettingId)id)->key,
               rt_setting_format((RtSettingId)id, rt_setting((RtSettingId)id), v), rt_setting_source_name(s), from);
    }
}

static void write_file(const char *path, const char *from) {
    uint64_t n = 0;
    const void *p = plat_map_file(from, &n);   // NULL for an empty file
    FILE *f = fopen(path, "wb");
    CHECK(f);
    CHECK(!n || fwrite(p, 1, (size_t)n, f) == n);
    CHECK(!fclose(f));
    if (p) plat_unmap_file(p, n);
}

static void hook(const RtSettingChange *c, void *ctx) {
    (void)ctx;
    char a[RT_SETTING_TEXT], b[RT_SETTING_TEXT];
    printf("hook %s %s->%s %s %s\n", rt_setting_row(c->id)->key, rt_setting_format(c->id, c->old_value, a),
           rt_setting_format(c->id, c->value, b), c->pending ? "pending" : "applied", rt_setting_source_name(c->source));
}

static void case_table(void) {
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        const RtSettingRow *r = rt_setting_row((RtSettingId)id);
        const char *types[] = {"bool", "int", "enum"};
        printf("%s\t%s\t%s\t%s\t%lld\t%lld\t%lld\t%s\t%s\t%s\n", r->key, rt_setting_group_name(r->group),
               types[r->type], rt_setting_apply_name(r->apply), (long long)r->def, (long long)r->min,
               (long long)r->max, r->names ? r->names : "", r->env, r->what);
        CHECK(rt_setting_find(r->key) == id);
    }
    CHECK(rt_setting_find("no_such_key") == -1);
}

static void case_nondefault(void) {
    load();
    RtSettingNow now[RT_SETTING_COUNT];
    int n = rt_settings_nondefault(now, RT_SETTING_COUNT);
    for (int i = 0; i < n; i++)
        printf("%s=%s\t%s\t%s\n", now[i].key, now[i].value, rt_setting_source_name(now[i].source), now[i].from);
    printf("count %d\n", n);
    CHECK(rt_settings_nondefault(now, 0) == n);   // the count does not depend on room for entries
}

// K=V words after the path: each set through rt_settings_set, as the menu will.
static void set_all(int argc, char **argv, int from) {
    for (int i = from; i < argc; i++) {
        char key[64], why[600];
        const char *eq = strchr(argv[i], '=');
        CHECK(eq && (size_t)(eq - argv[i]) < sizeof key);
        memcpy(key, argv[i], (size_t)(eq - argv[i]));
        key[eq - argv[i]] = 0;
        int id = rt_setting_find(key);
        CHECK(id >= 0);
        if (!rt_settings_set((RtSettingId)id, eq + 1, why, sizeof why)) {
            printf("set refused: %s\n", why);
            exit(1);
        }
    }
}

static void case_save(int argc, char **argv, int fail) {
    CHECK(argc >= 3);
    g_config = argv[2];
    load();
    set_all(argc, argv, 3);
    char why[600];
    rt_settings_fail_rename(fail);
    int ok = rt_settings_save(why, sizeof why);
    if (fail) {
        CHECK(!ok);
        printf("save refused: %s\n", why);
    } else {
        if (!ok) printf("save failed: %s\n", why);
        CHECK(ok);
        printf("saved\n");
    }
}

static void case_reload(int argc, char **argv) {
    const char *next = NULL;
    CHECK(argc >= 4);
    g_config = argv[2];
    parse(argc, argv, 3, &next);
    CHECK(next);
    load();
    rt_settings_hook(RT_SETTING_ANY, hook, NULL);
    uint64_t before = rt_settings_changes();
    write_file(g_config, next);
    int first = rt_settings_poll(), second = rt_settings_poll(), third = rt_settings_poll();
    printf("polls %d %d %d\n", first, second, third);
    print_keys();
    for (int id = 0; id < RT_SETTING_COUNT; id++) {
        int64_t v;
        char t[RT_SETTING_TEXT];
        if (rt_setting_pending((RtSettingId)id, &v))
            printf("pending %s=%s\n", rt_setting_row((RtSettingId)id)->key, rt_setting_format((RtSettingId)id, v, t));
    }
    printf("changes %llu\n", (unsigned long long)(rt_settings_changes() - before));
}

static void case_counter(int argc, char **argv) {
    CHECK(argc == 3);
    g_config = argv[2];
    load();
    char why[600];
    printf("start %llu\n", (unsigned long long)rt_settings_changes());
    CHECK(rt_settings_set(RT_SET_hold_ticks, "5", why, sizeof why));
    printf("set 5: %llu\n", (unsigned long long)rt_settings_changes());
    CHECK(rt_settings_set(RT_SET_hold_ticks, "5", why, sizeof why));
    printf("set 5 again: %llu\n", (unsigned long long)rt_settings_changes());
    CHECK(rt_settings_save(why, sizeof why));
    // The same value in other words: spaces, a comment, the line moved. Not a change.
    FILE *f = fopen(g_config, "wb");
    CHECK(f);
    fputs("; rewritten by hand\n[pacing]\nhold_ticks=5\n", f);
    CHECK(!fclose(f));
    int a = rt_settings_poll(), b = rt_settings_poll();
    printf("rewrite polls %d %d: %llu\n", a, b, (unsigned long long)rt_settings_changes());
    CHECK(rt_settings_set(RT_SET_hold_ticks, "6", why, sizeof why));
    printf("set 6: %llu\n", (unsigned long long)rt_settings_changes());
    CHECK(rt_settings_set(RT_SET_census, "1", why, sizeof why));   // restart: pending, never published
    printf("census 1: %llu\n", (unsigned long long)rt_settings_changes());
    CHECK(!rt_settings_set(RT_SET_hold_ticks, "-1", why, sizeof why));
    printf("refused: %s\n", why);
    printf("after a refusal: %llu, hold_ticks %lld\n", (unsigned long long)rt_settings_changes(),
           (long long)rt_setting(RT_SET_hold_ticks));
}

static void case_hold(int argc, char **argv) {
    const char *path = NULL;
    parse(argc, argv, 2, &path);
    CHECK(path);
    g_config = path;
    load();
    nvn_settings_bind();
    printf("bound: hold %d\n", nvn_present_hold_ticks());
    char why[600];
    CHECK(rt_settings_set(RT_SET_hold_ticks, "2", why, sizeof why));
    printf("set 2: hold %d\n", nvn_present_hold_ticks());
    FILE *f = fopen(path, "wb");
    CHECK(f);
    fputs("[pacing]\nhold_ticks = 7\n", f);
    CHECK(!fclose(f));
    CHECK(rt_settings_poll() == 1 && rt_settings_poll() == 2);
    printf("file 7: hold %d\n", nvn_present_hold_ticks());
}

static void case_watch(int argc, char **argv) {
    CHECK(argc == 4);
    g_config = argv[2];
    load();
    int64_t start = rt_setting(RT_SET_hold_ticks);
    CHECK(rt_settings_watch());
    uint64_t t0 = plat_now_ns();
    write_file(g_config, argv[3]);
    while (rt_setting(RT_SET_hold_ticks) == start && plat_now_ns() - t0 < 10000000000ull) plat_sleep_ns(10000000);
    printf("watch: hold_ticks %lld -> %lld after %llu ms\n", (long long)start, (long long)rt_setting(RT_SET_hold_ticks),
           (unsigned long long)((plat_now_ns() - t0) / 1000000));
}

int main(int argc, char **argv) {
    setvbuf(stdout, NULL, _IONBF, 0);
    if (argc < 2) {
        fprintf(stderr, "usage: settings_test <case> [arguments]\n");
        return 2;
    }
    const char *cs = argv[1];
    if (!strcmp(cs, "table")) {
        case_table();
    } else if (!strcmp(cs, "show")) {
        parse(argc, argv, 2, NULL);
        load();
        print_keys();
    } else if (!strcmp(cs, "lazy")) {
        print_keys();
    } else if (!strcmp(cs, "nondefault")) {
        parse(argc, argv, 2, NULL);
        case_nondefault();
    } else if (!strcmp(cs, "save")) {
        case_save(argc, argv, 0);
    } else if (!strcmp(cs, "savefail")) {
        case_save(argc, argv, 1);
    } else if (!strcmp(cs, "reload")) {
        case_reload(argc, argv);
    } else if (!strcmp(cs, "counter")) {
        case_counter(argc, argv);
    } else if (!strcmp(cs, "hold")) {
        case_hold(argc, argv);
    } else if (!strcmp(cs, "watch")) {
        case_watch(argc, argv);
    } else {
        fprintf(stderr, "settings_test: no case %s\n", cs);
        return 3;
    }
    printf("ok %s\n", cs);
    return 0;
}
