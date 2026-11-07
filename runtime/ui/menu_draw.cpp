// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The menu's frame (oot2/ui.h, ui_menu_frame): Dear ImGui builds the menu, the overlay and the counters list into draw
// lists and stops there. No window, no device and no back end is needed: the context says it has a renderer that takes
// textures (ImGuiBackendFlags_RendererHasTextures), and after each frame this file answers the font atlas's requests
// the way a back end's texture update does, with no pixels sent anywhere, so the glyphs are laid out and the draw lists
// complete. The Vulkan present path (a later ticket) renders the same draw lists with Dear ImGui's own back end.
//
// Every control is bound to the store through the rows (menu.cpp): it shows the value the row was just built with and
// changes it only through commit(), so no control holds a value of its own. Without Dear ImGui in the build
// (OOT2_WITH_IMGUI), the frame draws nothing and says so.
#include <algorithm>
#include <array>
#include <cfloat>
#include <cstdio>
#include <cstring>

#include "menu_internal.h"

#ifdef OOT2_WITH_IMGUI
#include "imgui.h"

using namespace oot2ui;

namespace {

ImGuiContext *g_ctx = nullptr;
bool g_advanced = false;                  // the menu's Advanced switch: the menu's own state, no setting
float g_pointer_x = -FLT_MAX, g_pointer_y = -FLT_MAX;
int g_pointer_down = 0;
float g_sent_x = -FLT_MAX, g_sent_y = -FLT_MAX;
int g_sent_down = 0;
float g_frame_ms = 0.0f;                  // the overlay's smoothed frame time
std::vector<std::array<float, 4>> g_rects;
bool g_close = false;

void ensure_context() {
    if (g_ctx) return;
    IMGUI_CHECKVERSION();
    g_ctx = ImGui::CreateContext();
    ImGuiIO &io = ImGui::GetIO();
    io.IniFilename = nullptr;   // the menu keeps nothing in imgui.ini: its settings are the store's
    io.LogFilename = nullptr;
    io.BackendPlatformName = "oot2 menu (no window)";
    io.BackendRendererName = "oot2 menu (draw lists only)";
    io.BackendFlags |= ImGuiBackendFlags_RendererHasTextures;
    io.ConfigFlags |= ImGuiConfigFlags_NoMouseCursorChange;
    ImGui::StyleColorsDark();
}

// What a back end does with the font atlas's requests, minus the upload.
void answer_texture_requests() {
    for (ImTextureData *tex : ImGui::GetPlatformIO().Textures) {
        if (tex->Status == ImTextureStatus_WantCreate) {
            tex->SetTexID((ImTextureID)1);
            tex->SetStatus(ImTextureStatus_OK);
        } else if (tex->Status == ImTextureStatus_WantUpdates) {
            tex->SetStatus(ImTextureStatus_OK);
        } else if (tex->Status == ImTextureStatus_WantDestroy) {
            tex->SetTexID((ImTextureID)0);
            tex->SetStatus(ImTextureStatus_Destroyed);
        }
    }
}

void pick(const Row &row, const char *text) {
    char why[300];
    set_status(commit(row, text, why, sizeof why) ? "" : why);
}

// "value:label|value|..." to its pieces; a piece with no label shows its value.
void quick_buttons(const Row &row) {
    for (const char *p = row.quick; p && *p;) {
        const char *end = std::strchr(p, '|');
        const size_t n = end ? (size_t)(end - p) : std::strlen(p);
        char piece[48], value[24], label[24];
        std::snprintf(piece, sizeof piece, "%.*s", (int)n, p);
        const char *colon = std::strchr(piece, ':');
        std::snprintf(value, sizeof value, "%.*s", colon ? (int)(colon - piece) : (int)std::strlen(piece), piece);
        std::snprintf(label, sizeof label, "%s", colon ? colon + 1 : value);
        ImGui::SameLine();
        if (ImGui::Button(label)) pick(row, value);
        p = end ? end + 1 : p + n;
    }
}

void draw_row(size_t index, const Row &row, UiDraw &d) {
    ImGui::PushID((int)index);
    const bool greyed = row.info.state == UI_ROW_HELD || row.info.state == UI_ROW_GREYED;
    ImGui::AlignTextToFramePadding();
    ImGui::TextUnformatted(row.info.label);
    ImGui::SameLine(300.0f);
    if (greyed) ImGui::BeginDisabled();
    if (row.id < 0) {
        ImGui::TextUnformatted("-");
    } else if (row.type == RT_SETTING_BOOL) {
        bool on = row.value != 0;
        if (ImGui::Checkbox("##v", &on)) pick(row, on ? "1" : "0");
    } else if (row.type == RT_SETTING_ENUM) {
        if (ImGui::BeginCombo("##v", row.info.value)) {
            int at = 0;
            for (const char *p = row.names; p && *p; at++) {
                const char *end = std::strchr(p, '|');
                const size_t n = end ? (size_t)(end - p) : std::strlen(p);
                char name[32];
                std::snprintf(name, sizeof name, "%.*s", (int)n, p);
                if (ImGui::Selectable(name, at == row.value)) pick(row, name);
                p = end ? end + 1 : p + n;
            }
            ImGui::EndCombo();
        }
    } else {
        int v = (int)row.value;
        ImGui::SetNextItemWidth(120.0f);
        if (ImGui::InputInt("##v", &v, 1, 100, ImGuiInputTextFlags_EnterReturnsTrue)) {
            char text[RT_SETTING_TEXT];
            std::snprintf(text, sizeof text, "%d", v);
            pick(row, text);
        }
    }
    const ImVec2 lo = ImGui::GetItemRectMin(), hi = ImGui::GetItemRectMax();
    if (index < g_rects.size()) g_rects[index] = {lo.x, lo.y, hi.x - lo.x, hi.y - lo.y};
    if (!greyed) quick_buttons(row);
    if (row.info.help[0] && ImGui::IsItemHovered(ImGuiHoveredFlags_AllowWhenDisabled))
        ImGui::SetTooltip("%s", row.info.help);
    if (greyed) ImGui::EndDisabled();
    if (row.info.note[0]) {
        ImGui::SameLine();
        ImGui::TextDisabled("%s", row.info.note);
    }
    ImGui::PopID();
    d.rows++;
}

void draw_menu(UiDraw &d, int width, int height) {
    ImGui::SetNextWindowPos(ImVec2(16.0f, 16.0f), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(std::min(820.0f, width - 32.0f), std::min(900.0f, height - 32.0f)),
                             ImGuiCond_FirstUseEver);
    bool open = true;
    g_close = false;
    if (ImGui::Begin("Settings", &open, ImGuiWindowFlags_NoCollapse)) {
        for (int p = 0; p < ui_menu_preset_count(); p++) {
            if (p) ImGui::SameLine();
            if (ImGui::Button(ui_menu_preset_name(p))) {
                char note[200];
                const int held = ui_menu_apply_preset(p, note, sizeof note);
                char line[300];
                if (held) std::snprintf(line, sizeof line, "%s: %d setting(s) kept their value (%s)",
                                        ui_menu_preset_name(p), held, note);
                else line[0] = '\0';
                set_status(line);
            }
        }
        ImGui::SameLine();
        if (ImGui::Button("Save")) {
            char why[300];
            set_status(ui_menu_save(why, sizeof why) ? "" : why);
        }
        ImGui::SameLine();
        if (ImGui::Button("Close")) g_close = true;
        ImGui::SameLine();
        ImGui::Checkbox("Advanced", &g_advanced);
        if (!status().empty()) ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", status().c_str());
        ImGui::Separator();

        const std::vector<Row> all = rows();
        g_rects.assign(all.size(), {0.0f, 0.0f, 0.0f, 0.0f});
        const char *group = nullptr;
        bool group_open = false;
        for (size_t i = 0; i < all.size(); i++) {
            const Row &row = all[i];
            if (row.info.advanced && !g_advanced) continue;
            if (!group || std::strcmp(group, row.info.group) != 0) {
                group = row.info.group;
                group_open = ImGui::CollapsingHeader(group, ImGuiTreeNodeFlags_DefaultOpen);
            }
            if (group_open) draw_row(i, row, d);
        }
    }
    ImGui::End();
    d.windows++;
    if (!open) g_close = true;
}

void draw_overlay(double dt, int width) {
    const float ms = dt > 0.0 ? (float)(dt * 1000.0) : 0.0f;
    g_frame_ms = g_frame_ms == 0.0f ? ms : g_frame_ms + (ms - g_frame_ms) * 0.1f;
    ImGui::SetNextWindowPos(ImVec2((float)width - 8.0f, 8.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.6f);
    ImGui::Begin("##overlay", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
    ImGui::Text("%.2f ms  %.0f fps", g_frame_ms, g_frame_ms > 0.0f ? 1000.0f / g_frame_ms : 0.0f);
    ImGui::End();
}

void draw_counters(int width) {
    ImGui::SetNextWindowPos(ImVec2((float)width - 8.0f, 44.0f), ImGuiCond_Always, ImVec2(1.0f, 0.0f));
    ImGui::SetNextWindowBgAlpha(0.6f);
    ImGui::Begin("##counters", nullptr,
                 ImGuiWindowFlags_NoDecoration | ImGuiWindowFlags_AlwaysAutoResize | ImGuiWindowFlags_NoSavedSettings |
                     ImGuiWindowFlags_NoFocusOnAppearing | ImGuiWindowFlags_NoNav | ImGuiWindowFlags_NoInputs);
    char name[64], value[64];
    for (int i = 0; own_counter(i, name, sizeof name, value, sizeof value); i++) ImGui::Text("%s: %s", name, value);
    if (UiCounterFn fn = counters())
        for (int i = 0; fn(i, name, sizeof name, value, sizeof value); i++) ImGui::Text("%s: %s", name, value);
    ImGui::End();
}

}   // namespace

extern "C" void ui_menu_pointer(float x, float y, int down) {
    g_pointer_x = x, g_pointer_y = y, g_pointer_down = down ? 1 : 0;
}

extern "C" int ui_menu_frame(const UiFrame *frame, UiDraw *out) {
    if (out) *out = UiDraw{};
    const bool menu = ui_menu_is_open() != 0;
    const bool overlay = rt_setting(RT_SET_overlay) != 0;
    const bool counters_on = rt_setting(RT_SET_counters) != 0;
    if (!menu && !overlay && !counters_on) return 0;
    ensure_context();
    const int width = frame && frame->width > 0 ? frame->width : 1280;
    const int height = frame && frame->height > 0 ? frame->height : 720;
    const double dt = frame ? frame->dt : 0.0;
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)width, (float)height);
    io.DeltaTime = dt > 0.0 ? (float)dt : 1.0f / 60.0f;
    if (g_pointer_x != g_sent_x || g_pointer_y != g_sent_y) io.AddMousePosEvent(g_pointer_x, g_pointer_y);
    if (g_pointer_down != g_sent_down) io.AddMouseButtonEvent(0, g_pointer_down != 0);
    g_sent_x = g_pointer_x, g_sent_y = g_pointer_y, g_sent_down = g_pointer_down;

    ImGui::NewFrame();
    UiDraw d = {};
    if (menu) draw_menu(d, width, height);
    if (overlay) {
        draw_overlay(dt, width);
        d.windows++;
    }
    if (counters_on) {
        draw_counters(width);
        d.windows++;
    }
    ImGui::Render();
    const ImDrawData *dd = ImGui::GetDrawData();
    d.vertices = dd->TotalVtxCount;
    d.indices = dd->TotalIdxCount;
    for (int i = 0; i < dd->CmdListsCount; i++) d.commands += dd->CmdLists[i]->CmdBuffer.Size;
    answer_texture_requests();
    if (out) *out = d;
    if (g_close) {   // the Close button, or the window's own: after the frame, as the open key's close is
        g_close = false;
        ui_menu_open(0);
    }
    return 1;
}

extern "C" int ui_menu_row_rect(int index, float rect[4]) {
    if (index < 0 || index >= (int)g_rects.size() || g_rects[(size_t)index][2] <= 0.0f) return 0;
    std::memcpy(rect, g_rects[(size_t)index].data(), 4 * sizeof(float));
    return 1;
}

#else   // built without Dear ImGui: the model works, nothing is drawn

extern "C" void ui_menu_pointer(float, float, int) {}
extern "C" int ui_menu_frame(const UiFrame *, UiDraw *out) {
    if (out) *out = UiDraw{};
    return 0;
}
extern "C" int ui_menu_row_rect(int, float[4]) { return 0; }

#endif
