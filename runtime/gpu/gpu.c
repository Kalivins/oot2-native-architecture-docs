// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The graphics back-end choice and the lookup of the runtime's own shaders (oot2/gpu.h). This file also includes Volk
// and the list of required Vulkan device items with its query (oot2/vk_required.h), so the runtime library is always
// compiled against the Vulkan headers the back end uses and the query compiles in the runtime as in the check's host
// probe. The Vulkan back end opens its device (gpu_vk_open) before its table is installed, so a machine that cannot
// run it stops the start by name at once.
#include <string.h>

#include "volk.h"

#include "oot2/gfx.h"
#include "oot2/gpu.h"
#include "oot2/gpu_vk.h"
#include "oot2/runtime.h"
#include "oot2/vk_required.h"

// Every back end a start can name, with its table and what must run before the table is installed.
static const struct {
    const char *name;
    const GfxBackend *table;
    void (*open)(void);
} g_backends[] = {
    {"null", &gfx_null_backend, NULL},
    {"vulkan", &gfx_vulkan_backend, gpu_vk_open},   // architecture section 6
};

#define BACKEND_COUNT ((int)(sizeof g_backends / sizeof g_backends[0]))

int gfx_backend_known(const char *name) {
    for (int i = 0; i < BACKEND_COUNT; i++)
        if (!strcmp(name, g_backends[i].name)) return 1;
    return 0;
}

void gfx_select(const char *name, const char *source) {
    for (int i = 0; i < BACKEND_COUNT; i++) {
        if (strcmp(name, g_backends[i].name)) continue;
        rt_log("gfx: the %s back end (%s)", name, source);
        if (g_backends[i].open) g_backends[i].open();
        gfx_set_backend(g_backends[i].table);
        return;
    }
    rt_fatal("gfx: no back end is named \"%s\" (asked for by %s); the names are null and vulkan", name, source);
}

const GpuShader *gpu_shader(const char *name) {
    for (int i = 0; i < gpu_shader_count; i++)
        if (!strcmp(gpu_shaders[i].name, name)) return &gpu_shaders[i];
    return NULL;
}
