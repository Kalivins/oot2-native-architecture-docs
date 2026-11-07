// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Calls runtime/ui through its C header (oot2/ui.h), the way the rest of the runtime will: the C++ build check's proof that C
// code reaches the C++ stand-in and that the C++ standard library is linked behind it. One case per process, named on
// the command line (tests/test_runtime_ui.py): ui_test CASE [ARGS]. Exit 0 passes, 1 fails with a line on stderr.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/ui.h"

#define CHECK(cond)                                                                                                    \
    do {                                                                                                               \
        if (!(cond)) {                                                                                                 \
            fprintf(stderr, "ui_test: %s:%d: %s\n", __FILE__, __LINE__, #cond);                                        \
            return 1;                                                                                                  \
        }                                                                                                              \
    } while (0)

// Empty entries and repeats go, the rest is sorted and joined.
static int case_list(void) {
    char out[64];
    size_t n = ui_standin_list("high,low,,medium,low,high,", out, sizeof out);
    CHECK(n == strlen("high | low | medium"));
    CHECK(!strcmp(out, "high | low | medium"));
    return 0;
}

static int case_sorted(void) {
    char out[64];
    size_t n = ui_standin_list("pear,apple,fig,Banana", out, sizeof out);
    CHECK(!strcmp(out, "Banana | apple | fig | pear"));   // byte order: a capital letter sorts first
    CHECK(n == strlen(out));
    return 0;
}

// A buffer too small cuts the text, terminates it, and the return value says how long the whole text is.
static int case_cut(void) {
    char out[8];
    memset(out, 'x', sizeof out);
    size_t n = ui_standin_list("alpha,beta", out, sizeof out);
    CHECK(n == strlen("alpha | beta"));
    CHECK(!strcmp(out, "alpha |"));
    char one[1] = {'x'};
    CHECK(ui_standin_list("alpha", one, 1) == 5);
    CHECK(one[0] == '\0');
    return 0;
}

static int case_empty(void) {
    char out[16] = "unchanged";
    CHECK(ui_standin_list("", out, sizeof out) == 0);
    CHECK(out[0] == '\0');
    CHECK(ui_standin_list(",,,", out, sizeof out) == 0);
    CHECK(ui_standin_list(NULL, out, sizeof out) == 0);
    CHECK(ui_standin_list("a,b", NULL, 0) == strlen("a | b"));   // the length query: no buffer
    return 0;
}

// Ten thousand entries in a scrambled order, each twice: the container grows through the heap and the sort has work.
static int case_many(void) {
    enum { COUNT = 10000 };
    char *text = malloc((size_t)COUNT * 2 * 7 + 1);
    CHECK(text != NULL);
    char *p = text;
    for (int i = 0; i < COUNT; i++) {
        int k = (int)(((long long)i * 7919) % COUNT);   // 7919 is prime, so coprime with COUNT: a permutation
        p += sprintf(p, "e%04d,e%04d,", k, k);
    }
    size_t need = ui_standin_list(text, NULL, 0);
    CHECK(need == (size_t)COUNT * 5 + (size_t)(COUNT - 1) * 3);   // "e0000" is 5 bytes, " | " is 3
    char *out = malloc(need + 1);
    CHECK(out != NULL);
    CHECK(ui_standin_list(text, out, need + 1) == need);
    CHECK(!strncmp(out, "e0000 | e0001 | e0002", 21));
    CHECK(!strcmp(out + need - 5, "e9999"));
    free(out);
    free(text);
    return 0;
}

// ui_test flags DIALECT EXCEPTIONS RTTI: how the C++ side was compiled.
static int case_flags(int argc, char **argv) {
    CHECK(argc == 5);
    UiBuild b = ui_standin_build();
    CHECK(b.dialect == atol(argv[2]));
    CHECK(b.exceptions == atoi(argv[3]));
    CHECK(b.rtti == atoi(argv[4]));
    return 0;
}

// ui_test imgui VERSION|none: the Dear ImGui the runtime was built with, and whether its core agrees with the headers
// the stand-in saw (the ImGui fetch study). "none" is a build without the library.
static int case_imgui(int argc, char **argv) {
    CHECK(argc == 3);
    UiBuild b = ui_standin_build();
    if (!strcmp(argv[2], "none")) {
        CHECK(b.imgui_version == NULL);
        CHECK(b.imgui_agrees == 0);
        return 0;
    }
    CHECK(b.imgui_version != NULL);
    CHECK(!strcmp(b.imgui_version, argv[2]));
    CHECK(b.imgui_agrees == 1);
    return 0;
}

int main(int argc, char **argv) {
    const char *name = argc > 1 ? argv[1] : "";
    if (!strcmp(name, "list")) return case_list();
    if (!strcmp(name, "sorted")) return case_sorted();
    if (!strcmp(name, "cut")) return case_cut();
    if (!strcmp(name, "empty")) return case_empty();
    if (!strcmp(name, "many")) return case_many();
    if (!strcmp(name, "flags")) return case_flags(argc, argv);
    if (!strcmp(name, "imgui")) return case_imgui(argc, argv);
    fprintf(stderr, "usage: ui_test list|sorted|cut|empty|many|flags|imgui [ARGS]\n");
    return 2;
}
