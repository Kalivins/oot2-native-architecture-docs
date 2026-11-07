// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// What menu.cpp (the rows, the presets, the open key: no Dear ImGui) shares with menu_draw.cpp (the frame). Internal to
// runtime/ui: nothing outside this directory includes it.
#pragma once
#include <string>
#include <vector>

#include "oot2/settings.h"
#include "oot2/ui.h"

namespace oot2ui {

// One row as it stands now: what the C side sees, and what the drawing needs to pick a control for it.
struct Row {
    UiRow info;
    RtSettingGroup group = RT_GROUP_DEBUG;   // for the order of the menu
    int id = -1;                          // the store's id, or -1 when the row has no key in the store
    RtSettingType type = RT_SETTING_BOOL;
    int64_t value = 0, min = 0, max = 0;  // the value shown (the next start's, for a pending restart key), its range
    const char *names = nullptr;          // an ENUM key's "a|b|c"
    const char *quick = nullptr;          // "value:label|value|..." buttons beside the control, or null
};

std::vector<Row> rows();   // every row, in menu order, read from the store at this call

// The one place a row writes: through the store (rt_settings_set), never to a copy. Refuses a greyed or held row.
bool commit(const Row &row, const char *text, char *why, size_t why_size);

const char *group_label(RtSettingGroup g);
void set_status(const char *text);   // the line the menu shows under its buttons (a refused value, a failed save)
const std::string &status();
UiCounterFn counters();
// The counters the menu owns itself (the store's), index 0, 1, ... until it returns false.
bool own_counter(int index, char *name, size_t name_size, char *value, size_t value_size);

}   // namespace oot2ui
