// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The menu's frame (oot2/ui.h, ui_menu_frame): Dear ImGui builds the menu, the overlay and the counters list into draw
// lists and stops there. No window, no device and no back end is needed: the context says it has a renderer that takes
// textures (ImGuiBackendFlags_RendererHasTextures), and after each frame this file answers the font atlas's requests
// the way a back end's texture update does, with no pixels sent anywhere, so the glyphs are laid out and the draw lists
// complete. The Vulkan present path (runtime/gpu/vk_overlay.c) renders the same draw lists: it takes the atlas through
// the texture ops (ui_menu_set_texture_ops) and reads the lists through ui_menu_draw_data, flattened for it.
//
// Every control is bound to the store through the rows (menu.cpp): it shows the value the row was just built with and
// changes it only through commit(), so no control holds a value of its own. Without Dear ImGui in the build
// (OOT2_WITH_IMGUI), the frame draws nothing and says so.
#include <algorithm>
#include <atomic>
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
std::atomic<float> g_pointer_x{-FLT_MAX}, g_pointer_y{-FLT_MAX};   // written by the window thread, read by the frame's
std::atomic<int> g_pointer_down{0};
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

UiTextureOps g_ops = {};
bool g_have_ops = false;

// What a back end does with the font atlas's requests: with a renderer's texture ops (ui_menu_set_texture_ops) the
// request goes to it, and without one the answer is the upload's absence, so the glyphs are laid out and the draw lists
// complete with no pixels sent anywhere.
void answer_texture_requests() {
    for (ImTextureData *tex : ImGui::GetPlatformIO().Textures) {
        const bool alpha = tex->Format == ImTextureFormat_Alpha8;
        if (tex->Status == ImTextureStatus_WantCreate) {
            uint64_t id = 1;
            if (g_have_ops) id = g_ops.create(tex->Width, tex->Height, alpha ? 1 : 0, (const uint8_t *)tex->GetPixels());
            if (!id) continue;   // the renderer could not: Dear ImGui asks again next frame
            tex->SetTexID((ImTextureID)id);
            tex->SetStatus(ImTextureStatus_OK);
        } else if (tex->Status == ImTextureStatus_WantUpdates) {
            if (g_have_ops) {
                const ImTextureRect &r = tex->UpdateRect;
                g_ops.update((uint64_t)tex->TexID, r.x, r.y, r.w, r.h, alpha ? 1 : 0,
                             (const uint8_t *)tex->GetPixelsAt(r.x, r.y), tex->GetPitch());
            }
            tex->SetStatus(ImTextureStatus_OK);
        } else if (tex->Status == ImTextureStatus_WantDestroy && tex->UnusedFrames >= 3) {
            if (g_have_ops) g_ops.destroy((uint64_t)tex->TexID);
            tex->SetTexID((ImTextureID)0);
            tex->SetStatus(ImTextureStatus_Destroyed);
        }
    }
}

std::vector<UiVertex> g_vertices;
std::vector<uint32_t> g_indices;
std::vector<UiCmd> g_cmds;
bool g_built = false;   // the last ui_menu_frame built lists (it returned 1)
static_assert(sizeof(ImDrawVert) == sizeof(UiVertex), "UiVertex is Dear ImGui's vertex");

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
        const std::string line = status();
        if (!line.empty()) ImGui::TextColored(ImVec4(1.0f, 0.45f, 0.4f, 1.0f), "%s", line.c_str());
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
    g_pointer_x.store(x, std::memory_order_relaxed), g_pointer_y.store(y, std::memory_order_relaxed);
    g_pointer_down.store(down ? 1 : 0, std::memory_order_relaxed);
}

extern "C" void ui_menu_set_texture_ops(const UiTextureOps *ops) {
    g_have_ops = ops && ops->create && ops->update && ops->destroy;
    g_ops = g_have_ops ? *ops : UiTextureOps{};
}

extern "C" int ui_menu_prepare(int width, int height) {
    ensure_context();
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)(width > 0 ? width : 1280), (float)(height > 0 ? height : 720));
    io.DeltaTime = 1.0f / 60.0f;
    ImGui::NewFrame();
    ImGui::Render();
    answer_texture_requests();
    return 1;
}

// The last frame's draw lists, flattened (oot2/ui.h): the vertices of every list one after the other, and each command's
// indices copied out with its vertex offset and its list's place in that array added, so the renderer needs neither.
extern "C" void ui_menu_draw_data(UiDrawData *out) {
    *out = UiDrawData{};
    g_vertices.clear(), g_indices.clear(), g_cmds.clear();
    if (!g_ctx || !g_built) return;
    const ImDrawData *dd = ImGui::GetDrawData();
    if (!dd || !dd->Valid) return;
    for (int n = 0; n < dd->CmdListsCount; n++) {
        const ImDrawList *list = dd->CmdLists[n];
        const uint32_t base = (uint32_t)g_vertices.size();
        const UiVertex *v = reinterpret_cast<const UiVertex *>(list->VtxBuffer.Data);
        g_vertices.insert(g_vertices.end(), v, v + list->VtxBuffer.Size);
        for (const ImDrawCmd &cmd : list->CmdBuffer) {
            if (cmd.UserCallback || cmd.ElemCount == 0) continue;
            UiCmd c = {};
            c.clip[0] = cmd.ClipRect.x, c.clip[1] = cmd.ClipRect.y, c.clip[2] = cmd.ClipRect.z, c.clip[3] = cmd.ClipRect.w;
            c.texture = (uint64_t)cmd.GetTexID();
            c.first_index = (uint32_t)g_indices.size();
            c.index_count = cmd.ElemCount;
            for (unsigned i = 0; i < cmd.ElemCount; i++)
                g_indices.push_back(base + cmd.VtxOffset + list->IdxBuffer.Data[cmd.IdxOffset + i]);
            g_cmds.push_back(c);
        }
    }
    out->vertices = g_vertices.data(), out->vertex_count = (uint32_t)g_vertices.size();
    out->indices = g_indices.data(), out->index_count = (uint32_t)g_indices.size();
    out->cmds = g_cmds.data(), out->cmd_count = (uint32_t)g_cmds.size();
}

extern "C" int ui_menu_frame(const UiFrame *frame, UiDraw *out) {
    if (out) *out = UiDraw{};
    const bool menu = ui_menu_is_open() != 0;
    const bool overlay = rt_setting(RT_SET_overlay) != 0;
    const bool counters_on = rt_setting(RT_SET_counters) != 0;
    g_built = false;
    if (!menu && !overlay && !counters_on) return 0;
    ensure_context();
    const int width = frame && frame->width > 0 ? frame->width : 1280;
    const int height = frame && frame->height > 0 ? frame->height : 720;
    const double dt = frame ? frame->dt : 0.0;
    ImGuiIO &io = ImGui::GetIO();
    io.DisplaySize = ImVec2((float)width, (float)height);
    io.DeltaTime = dt > 0.0 ? (float)dt : 1.0f / 60.0f;
    const float px = g_pointer_x.load(std::memory_order_relaxed), py = g_pointer_y.load(std::memory_order_relaxed);
    const int pdown = g_pointer_down.load(std::memory_order_relaxed);
    if (px != g_sent_x || py != g_sent_y) io.AddMousePosEvent(px, py);
    if (pdown != g_sent_down) io.AddMouseButtonEvent(0, pdown != 0);
    g_sent_x = px, g_sent_y = py, g_sent_down = pdown;

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
    g_built = true;
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
extern "C" void ui_menu_set_texture_ops(const UiTextureOps *) {}
extern "C" int ui_menu_prepare(int, int) { return 0; }
extern "C" void ui_menu_draw_data(UiDrawData *out) { *out = UiDrawData{}; }
extern "C" int ui_menu_frame(const UiFrame *, UiDraw *out) {
    if (out) *out = UiDraw{};
    return 0;
}
extern "C" int ui_menu_row_rect(int, float[4]) { return 0; }

#endif
