// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The SPIR-V capabilities and extensions the device's required list gives (oot2/vk_required.h, the pipeline builder): a module
// may declare a capability only when some item of the list enables the feature behind it, because a device that has
// every item is the only device the renderer opens, and a module that needs one more feature would fail validation or
// worse on a player's device that happens to lack it. The list is derived from the items, so adding an item adds its
// capabilities; this file holds the table that says which feature each capability is. The check over a corpus runs on
// the player's own modules and is never a list of them (tests/runtime/vk_pipeline_test.c, case `caps`).
#pragma once
#include <stddef.h>
#include <stdint.h>

#include "volk.h"

#include "oot2/spirv.h"
#include "oot2/vk_required.h"

#define VKC_MAX_CAPS 128
#define VKC_MAX_EXTS 32

typedef struct VkcList {
    uint32_t caps[VKC_MAX_CAPS];
    size_t ncaps;
    const char *exts[VKC_MAX_EXTS];   // static strings
    size_t nexts;
} VkcList;

// The capabilities and extensions of a device that has every item of the required list, whose facts are f (the
// properties the subgroup items judge by are taken from the items themselves, not from the device).
void vkc_list(const Oot2VkFacts *f, VkcList *out);

// 1 when capability cap is in the list.
int vkc_has_cap(const VkcList *l, uint32_t cap);

// 0 when every capability and extension d declares is in the list; else 1 with one line in why naming the first that is
// not ("capability Int64 (11) is not in the device's required list", "extension SPV_KHR_foo is not in it").
int vkc_check(const VkcList *l, const SpvDecl *d, char *why, size_t cap);
