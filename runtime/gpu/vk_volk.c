// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Volk, the Vulkan meta-loader (https://github.com/zeux/volk, MIT, Copyright (C) 2018-2026 Arseny Kapoulkine), which
// the back end loads Vulkan through: no import library is linked, so oot2.exe starts on a machine without a Vulkan
// loader, and the null back end never needs one. volk.c is not in this repository. It comes with the Vulkan SDK, or
// from tools/fetch_vulkan.py at the SDK's own pin, and tools/build_game.py puts its folder on the runtime's include
// path; its licence travels with it (THIRD_PARTY_NOTICES.md). Its own warnings are not ours to fix.
#if defined(__clang__)
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Weverything"
#endif
#include "volk.c"
#if defined(__clang__)
#pragma clang diagnostic pop
#endif
