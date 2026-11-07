// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The reader of a module's resource description (oot2/shader_desc.h) and the padding rule of a vertex module.
// A description is the translator's JSON, 5 to 40 KiB, and only a handful of members matter to the builder, so this is a
// scanner that walks the text once, reads the members it knows and skips the rest: no tree, no allocation, and a
// refusal at the first malformed byte, with the offset.
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "oot2/shader_desc.h"

typedef struct J {
    const char *begin, *p, *end;
    const char *err;
} J;

static int fail(J *j, const char *e) {
    if (!j->err) j->err = e;
    return 0;
}

static void ws(J *j) {
    while (j->p < j->end && (*j->p == ' ' || *j->p == '\n' || *j->p == '\r' || *j->p == '\t')) j->p++;
}

static int eat(J *j, char c) {
    ws(j);
    if (j->p < j->end && *j->p == c) {
        j->p++;
        return 1;
    }
    return 0;
}

// A string; at most cap - 1 bytes are kept (out may be NULL). Escapes become themselves; \u becomes '?'.
static int jstring(J *j, char *out, size_t cap) {
    ws(j);
    if (j->p >= j->end || *j->p != '"') return fail(j, "a string was expected");
    j->p++;
    size_t n = 0;
    while (j->p < j->end && *j->p != '"') {
        char c = *j->p++;
        if (c == '\\') {
            if (j->p >= j->end) break;
            c = *j->p++;
            if (c == 'u') {
                for (int i = 0; i < 4 && j->p < j->end; i++) j->p++;
                c = '?';
            } else if (c == 'n') {
                c = '\n';
            } else if (c == 't') {
                c = '\t';
            }
        }
        if (out && n + 1 < cap) out[n++] = c;
    }
    if (j->p >= j->end) return fail(j, "a string is not terminated");
    j->p++;
    if (out && cap) out[n] = 0;
    return 1;
}

// A number, kept as the integer part (a fraction or an exponent is read and dropped).
static int jint(J *j, long long *v) {
    ws(j);
    int neg = 0;
    if (j->p < j->end && *j->p == '-') neg = 1, j->p++;
    if (j->p >= j->end || *j->p < '0' || *j->p > '9') return fail(j, "a number was expected");
    long long x = 0;
    while (j->p < j->end && *j->p >= '0' && *j->p <= '9') {
        if (x < (1LL << 52)) x = x * 10 + (*j->p - '0');
        j->p++;
    }
    while (j->p < j->end && (*j->p == '.' || *j->p == 'e' || *j->p == 'E' || *j->p == '+' || *j->p == '-' ||
                             (*j->p >= '0' && *j->p <= '9')))
        j->p++;
    *v = neg ? -x : x;
    return 1;
}

static int jword(J *j, const char *w) {
    size_t n = strlen(w);
    if ((size_t)(j->end - j->p) < n || memcmp(j->p, w, n)) return 0;
    j->p += n;
    return 1;
}

static int jbool(J *j, int *v) {
    ws(j);
    if (jword(j, "true")) return *v = 1, 1;
    if (jword(j, "false")) return *v = 0, 1;
    return fail(j, "true or false was expected");
}

#define MAX_DEPTH 24

static int jskip(J *j, int depth);

// The members of an object: member(j, key, ctx) reads the value of key and returns 0 on a failure.
typedef int (*MemberFn)(J *j, const char *key, void *ctx);

static int jobject(J *j, MemberFn member, void *ctx, int depth) {
    if (depth > MAX_DEPTH) return fail(j, "nested too deeply");
    if (!eat(j, '{')) return fail(j, "an object was expected");
    if (eat(j, '}')) return 1;
    for (;;) {
        char key[48];
        if (!jstring(j, key, sizeof key)) return 0;
        if (!eat(j, ':')) return fail(j, "a colon was expected");
        if (!member(j, key, ctx)) return 0;
        if (eat(j, ',')) continue;
        if (eat(j, '}')) return 1;
        return fail(j, "a comma or the end of the object was expected");
    }
}

typedef int (*ElementFn)(J *j, void *ctx);

static int jarray(J *j, ElementFn element, void *ctx, int depth) {
    if (depth > MAX_DEPTH) return fail(j, "nested too deeply");
    if (!eat(j, '[')) return fail(j, "an array was expected");
    if (eat(j, ']')) return 1;
    for (;;) {
        if (!element(j, ctx)) return 0;
        if (eat(j, ',')) continue;
        if (eat(j, ']')) return 1;
        return fail(j, "a comma or the end of the array was expected");
    }
}

static int skip_member(J *j, const char *key, void *ctx) {
    (void)key;
    return jskip(j, *(int *)ctx + 1) ;
}

static int skip_element(J *j, void *ctx) { return jskip(j, *(int *)ctx + 1); }

static int jskip(J *j, int depth) {
    ws(j);
    if (j->p >= j->end) return fail(j, "the text ends where a value was expected");
    switch (*j->p) {
    case '{': return jobject(j, skip_member, &depth, depth);
    case '[': return jarray(j, skip_element, &depth, depth);
    case '"': return jstring(j, NULL, 0);
    case 't': case 'f': {
        int v;
        return jbool(j, &v);
    }
    case 'n':
        if (jword(j, "null")) return 1;
        return fail(j, "an unknown word");
    default: {
        long long v;
        return jint(j, &v);
    }
    }
}

// ---- the members that matter

typedef struct Ctx {
    ShaderDesc *d;
    char stage[8];
    int have_stage;
    ShdIo cur;
    ShdIo *list;
    uint32_t *count;
    int overflow;
} Ctx;

static int io_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    long long v;
    if (!strcmp(key, "location")) {
        if (!jint(j, &v)) return 0;
        x->cur.location = (int32_t)v;
    } else if (!strcmp(key, "component")) {
        if (!jint(j, &v)) return 0;
        x->cur.component = (uint8_t)(v < 0 || v > 3 ? 255 : v);
    } else if (!strcmp(key, "components")) {
        if (!jint(j, &v)) return 0;
        x->cur.components = (uint8_t)(v < 1 || v > 4 ? 255 : v);
    } else if (!strcmp(key, "array_length")) {
        if (!jint(j, &v)) return 0;
        x->cur.array_length = (int32_t)(v < 1 ? 1 : v > 1024 ? 1024 : v);
    } else if (!strcmp(key, "used")) {
        int b;
        if (!jbool(j, &b)) return 0;
        x->cur.used = (uint8_t)b;
    } else if (!strcmp(key, "type")) {
        char t[16];
        if (!jstring(j, t, sizeof t)) return 0;
        int bits = 0;
        if (!strncmp(t, "float", 5)) x->cur.type = SHD_FLOAT, bits = atoi(t + 5);
        else if (!strncmp(t, "sint", 4)) x->cur.type = SHD_SINT, bits = atoi(t + 4);
        else if (!strncmp(t, "uint", 4)) x->cur.type = SHD_UINT, bits = atoi(t + 4);
        else return fail(j, "an interface type that is not floatN, sintN or uintN");
        x->cur.bits = (uint8_t)bits;
    } else {
        return jskip(j, 4);
    }
    return 1;
}

static int io_element(J *j, void *c) {
    Ctx *x = c;
    memset(&x->cur, 0, sizeof x->cur);
    x->cur.array_length = 1;
    if (!jobject(j, io_member, x, 3)) return 0;
    if (*x->count >= SHD_MAX_IO) {
        x->overflow = 1;
        return fail(j, "more interface entries than the builder reads");
    }
    x->list[(*x->count)++] = x->cur;
    return 1;
}

static int iface_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    if (!strcmp(key, "inputs")) {
        x->list = x->d->inputs, x->count = &x->d->ninputs;
        return jarray(j, io_element, x, 2);
    }
    if (!strcmp(key, "outputs")) {
        x->list = x->d->outputs, x->count = &x->d->noutputs;
        return jarray(j, io_element, x, 2);
    }
    return jskip(j, 2);
}

static int layout_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    if (!strcmp(key, "fits")) return jbool(j, &x->d->fits);
    return jskip(j, 2);
}

static int raw_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    if (!strcmp(key, "physical_pointers")) return jbool(j, &x->d->raw_pointers);
    return jskip(j, 2);
}

static int top_member(J *j, const char *key, void *c) {
    Ctx *x = c;
    long long v;
    if (!strcmp(key, "schema")) {
        if (!jint(j, &v)) return 0;
        x->d->schema = (uint32_t)v;
        return 1;
    }
    if (!strcmp(key, "stage")) {
        x->have_stage = 1;
        return jstring(j, x->stage, sizeof x->stage);
    }
    if (!strcmp(key, "layout")) return jobject(j, layout_member, x, 1);
    if (!strcmp(key, "interface")) return jobject(j, iface_member, x, 1);
    if (!strcmp(key, "raw_access")) return jobject(j, raw_member, x, 1);
    return jskip(j, 1);
}

const char *shd_kind_name(int kind) {
    static const char *const names[SHD_KINDS] = {"vs", "ps", "cs", "ts", "ms"};
    return kind >= 0 && kind < SHD_KINDS ? names[kind] : "?";
}

int shd_parse(const char *json, size_t n, uint32_t schema, ShaderDesc *out, char *why, size_t cap) {
    memset(out, 0, sizeof *out);
    out->kind = -1;
    Ctx x = {.d = out};
    J j = {.begin = json, .p = json, .end = json + n};
    if (!jobject(&j, top_member, &x, 0)) {
        if (why && cap)
            snprintf(why, cap, "the description is not readable: %s (at byte %lu)", j.err ? j.err : "malformed",
                     (unsigned long)(j.p - j.begin));
        return 0;
    }
    ws(&j);
    if (j.p != j.end) {
        if (why && cap)
            snprintf(why, cap, "the description has text after its object (at byte %lu)", (unsigned long)(j.p - j.begin));
        return 0;
    }
    if (out->schema != schema) {
        if (why && cap) snprintf(why, cap, "the description is schema %u, the runtime reads schema %u", out->schema, schema);
        return 0;
    }
    if (!x.have_stage) {
        if (why && cap) snprintf(why, cap, "the description names no stage");
        return 0;
    }
    for (int k = 0; k < SHD_KINDS; k++)
        if (!strcmp(x.stage, shd_kind_name(k))) out->kind = k;
    if (out->kind < 0) {
        if (why && cap)
            snprintf(why, cap, "the description names the stage \"%s\", which is not one of vs, ps, cs, ts, ms", x.stage);
        return 0;
    }
    return 1;
}

// ---- padding

int shd_pad_set(const ShaderDesc *vs, const ShaderDesc *ps, ShdPad *out, int max, char *why, size_t cap) {
    uint8_t have[SHD_MAX_LOCATIONS] = {0}, need[SHD_MAX_LOCATIONS] = {0};
    uint8_t type[SHD_MAX_LOCATIONS][4];
    memset(type, 0, sizeof type);
#define REFUSE(...)                                                                                                    \
    do {                                                                                                               \
        if (why && cap) snprintf(why, cap, __VA_ARGS__);                                                               \
        return -1;                                                                                                     \
    } while (0)
    for (uint32_t i = 0; i < vs->noutputs; i++) {
        const ShdIo *o = &vs->outputs[i];
        if (o->component > 3 || o->components < 1 || o->component + o->components > 4)
            REFUSE("vertex output %d has components %u..%u", o->location, o->component, o->component + o->components);
        for (int k = 0; k < o->array_length; k++) {
            int l = o->location + k;
            if (l < 0 || l >= SHD_MAX_LOCATIONS) REFUSE("vertex output location %d is past the %d the builder reads", l, SHD_MAX_LOCATIONS);
            have[l] |= (uint8_t)(((1u << o->components) - 1) << o->component);
        }
    }
    for (uint32_t i = 0; i < ps->ninputs; i++) {
        const ShdIo *in = &ps->inputs[i];
        if (in->component > 3 || in->components < 1 || in->component + in->components > 4)
            REFUSE("pixel input %d has components %u..%u", in->location, in->component, in->component + in->components);
        for (int k = 0; k < in->array_length; k++) {
            int l = in->location + k;
            if (l < 0 || l >= SHD_MAX_LOCATIONS) REFUSE("pixel input location %d is past the %d the builder reads", l, SHD_MAX_LOCATIONS);
            uint8_t words = (uint8_t)(((1u << in->components) - 1) << in->component);
            if ((words & ~have[l]) && in->bits != 32)
                REFUSE("pixel input %d is %u bits wide: only 32-bit variables are padded", l, in->bits);
            for (int c = 0; c < 4; c++)
                if (words >> c & 1) type[l][c] = in->type;
            need[l] |= words;
        }
    }
    int n = 0;
    for (int l = 0; l < SHD_MAX_LOCATIONS; l++) {
        uint8_t miss = (uint8_t)(need[l] & ~have[l]);
        for (int c = 0; c < 4;) {
            if (!(miss >> c & 1)) {
                c++;
                continue;
            }
            int e = c + 1;
            while (e < 4 && (miss >> e & 1) && type[l][e] == type[l][c]) e++;
            if (n >= max) REFUSE("the pixel module reads more than %d vertex outputs the vertex module does not write", max);
            out[n++] = (ShdPad){(uint16_t)l, (uint8_t)c, (uint8_t)(e - c), type[l][c]};
            c = e;
        }
    }
#undef REFUSE
    return n;
}

void shd_pad_text(const ShdPad *pad, int n, char *out, size_t cap) {
    size_t at = 0;
    if (cap) out[0] = 0;
    for (int i = 0; i < n; i++) {
        int w = snprintf(out + at, at < cap ? cap - at : 0, "%u:%u:%u:%u;", pad[i].location, pad[i].component,
                         pad[i].components, pad[i].type);
        if (w < 0) break;
        at += (size_t)w;
    }
}
