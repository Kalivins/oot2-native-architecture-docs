// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Which Opus decoder this build has (opus.h): a table a test installed, else libopus's in a build that has it.
#include "opus.h"

static const OpusApi *g_installed;

void audio_opus_install(const OpusApi *api) { __atomic_store_n(&g_installed, api, __ATOMIC_RELEASE); }

const OpusApi *audio_opus(void) {
    const OpusApi *a = __atomic_load_n(&g_installed, __ATOMIC_ACQUIRE);
    if (a) return a;
#ifdef OOT2_OPUS
    return audio_opus_libopus();
#else
    return 0;
#endif
}
