// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The C face of runtime/ui, the runtime's one C++ directory (architecture section 2: the runtime is C11, and the
// libraries a menu can be built on are C++). C++ stays behind this header: every function here is `extern "C"`, takes
// and returns C types, and the C++ side is built without exceptions and without run-time type information
// (tools/build_game.py, CXX_FLAGS), because the runtime does not unwind C++ exceptions (the C++ exceptions study). No C file
// includes a C++ header, and no C++ object may let an exception leave a function declared here.
//
// The directory holds the configuration menu (menu.cpp, menu_draw.cpp: the functions after the stand-in's below) and
// a stand-in that proves the C++ build: one file that uses the C++ standard library (a container, a string, an
// algorithm) so that building and linking it proves the path (the C++ build check), and that reports which Dear ImGui, if
// any, the runtime was built with (the ImGui fetch study). The tests of the build keep using the stand-in.
#pragma once
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// How the C++ side was compiled: its language level (the value of __cplusplus, 201703 for C++17) and whether
// exceptions and run-time type information were on (both 0 by the build's flags). A test reads it, so a change of
// the flags that does not reach the compiler shows.
//
// imgui_version is Dear ImGui's version string when the runtime was built with the fetched library
// (tools/fetch_imgui.py, tools/build_game.py --imgui), else NULL. imgui_agrees is 1 when the compiled core and the headers this file saw agree
// on the version and on the layout of the structures a caller shares with it (Dear ImGui's own check), else 0; it is
// 0 with no library. A core built from other headers than the caller's stops in Dear ImGui's assert instead of
// returning 0, in a build that has asserts.
typedef struct UiBuild {
    long dialect;
    int exceptions;
    int rtti;
    const char *imgui_version;
    int imgui_agrees;
} UiBuild;

UiBuild ui_standin_build(void);

// Split text at commas, drop the empty entries and the repeats, sort the rest and write them to out joined with
// " | ", terminated within cap bytes (out may be NULL when cap is 0). Returns the length of the whole joined text
// without its terminator: a value of cap or more means the text was cut.
size_t ui_standin_list(const char *text, char *out, size_t cap);

// ---- the configuration menu (the menu survey; the menu answers)
//
// The menu is rows over the settings store (oot2/settings.h) and never over a variable of its own: a row reads its
// value from the store each time it is asked or drawn, and changes it only through rt_settings_set, which applies a
// live key at once and keeps a restart key's value for the next start. Rows are described by a table in
// runtime/ui/menu.cpp, in the groups of the store (graphics, frame rate and pacing, display, audio, input,
// enhancements, debug); a row whose key the store does not have yet (the window's, the movie hold's) is greyed with its
// reason and becomes a live row on its own the day the key is registered, and a store key that has no row of its own
// is listed in its group by name, so no setting is out of the menu's reach. A row is LIVE (a change reaches the running
// game), RESTART or PREPARE (the next start, or the next oot2 prepare, uses it), HELD (the command line or the
// environment holds the key and wins: greyed with its source) or GREYED (no consumer yet: greyed with its reason).
//
// Two presets, "Reference" (every setting at its default, which is the state the gates and the captures use) and
// "Test" (the same with the overlay and the counters on), each set through the store like a row.
//
// Changes are written to the settings file when the menu closes, or when it is asked to, never on a frame that changes
// nothing; the file keeps every byte the changed lines do not own (rt_settings_save).
//
// The menu is drawn with Dear ImGui into draw lists and no more: the Vulkan present path (a later ticket) renders them,
// so a build without a window, a device or the null back end's own, runs the same code, which is how the tests drive
// it. All calls but ui_menu_input and ui_menu_pointer come from the one thread that builds the frame; those two may come from
// the thread that pumps window events: the pointer is stored in atomics, and the open key's close saves the settings file on
// the calling thread (never on a frame), with the status line it leaves under a lock.
typedef enum UiRowState {
    UI_ROW_LIVE,      // a change reaches the running game
    UI_ROW_RESTART,   // the next start uses it
    UI_ROW_PREPARE,   // the next oot2 prepare uses it
    UI_ROW_HELD,      // the command line or the environment holds the key: greyed, with its source in note
    UI_ROW_GREYED     // nothing consumes it yet: greyed, with its reason in note
} UiRowState;

typedef struct UiRow {
    const char *group;   // "Frame rate and pacing"
    const char *label;
    const char *key;     // the store's key, or NULL when the row has none (yet)
    const char *help;    // what the row does, from the store when it has a key; may be empty
    UiRowState state;
    int advanced;        // shown only with the menu's Advanced switch on
    char value[24];      // what the row shows: the store's value as text (the next start's, for a restart key); "" with no key
    char note[192];      // why the row is greyed or held, or what an editable row asks of the player; may be empty
} UiRow;

int ui_menu_row_count(void);
int ui_menu_row(int index, UiRow *out);   // the row now, read from the store; 0 for an index out of range

// What a click on a row does: set the row's key to text through the store; 1, or 0 with why (a greyed or held row, a
// value the key does not take). The change is saved when the menu closes (or ui_menu_save is called).
int ui_menu_set(int index, const char *text, char *why, size_t why_size);

int ui_menu_preset_count(void);
const char *ui_menu_preset_name(int index);   // "Reference", "Test"
// Set every key of the store to the preset's value (a key already there is left as it is). Returns how many keys keep
// another value because the command line or the environment holds them (0 when the preset is whole), or -1 for an
// index out of range; note (may be NULL when note_size is 0) names the first held key.
int ui_menu_apply_preset(int index, char *note, size_t note_size);

// Write the changes made through the menu to the settings file: 1, or 0 with why (rt_settings_save).
int ui_menu_save(char *why, size_t why_size);

// The open key. The platform layer reports each press and release of the buttons below as it reads them; the menu
// keeps its own state. F1 and the pad's Guide button toggle it; Escape closes it (and is not consumed while it is
// closed, so the game still sees it); Minus and Plus held together toggle it once. Returns 1 when the menu consumed the
// event (the platform layer does not pass it on), else 0. A scripted run (ui_menu_set_scripted) ignores every button.
// The first button of the chord is not held back from the game by this call: that is the input layer's to do.
typedef enum UiButton { UI_KEY_F1, UI_KEY_ESCAPE, UI_PAD_GUIDE, UI_PAD_MINUS, UI_PAD_PLUS } UiButton;
int ui_menu_input(UiButton button, int down);
void ui_menu_set_scripted(int on);
int ui_menu_is_open(void);
void ui_menu_open(int open);   // closing saves a changed setting, as the key does

// The pointer, in window pixels, with the left button down or not (0 or 1).
void ui_menu_pointer(float x, float y, int down);

// Counters for the counters list, from the modules that own them: index 0, 1, ... until it returns 0.
typedef int (*UiCounterFn)(int index, char *name, size_t name_size, char *value, size_t value_size);
void ui_menu_set_counters(UiCounterFn fn);

// One frame of the menu, the overlay and the counters, for a window of width by height pixels and the time since the
// last frame. Builds the draw lists (Dear ImGui's, drawn by the present path) and reports what it built; returns 0 and
// builds nothing when the menu is closed and the overlay and the counters are off, so a closed menu costs nothing on
// the frame, and when the runtime was built without Dear ImGui. The first call creates the context and its font; it
// needs no window, no device and no back end.
typedef struct UiFrame {
    int width, height;
    double dt;   // seconds since the last frame
} UiFrame;
typedef struct UiDraw {
    int windows;    // the menu, the overlay and the counters list that were drawn
    int rows;       // rows drawn (the menu's, not the counters')
    int vertices;
    int indices;
    int commands;
} UiDraw;
int ui_menu_frame(const UiFrame *frame, UiDraw *out);
// Where the last frame drew a row's own control (x, y, width, height), for the pointer; 0 when it was not drawn.
int ui_menu_row_rect(int index, float rect[4]);

// ---- drawing the frame (the Vulkan present path, runtime/gpu/vk_overlay.c)
//
// The last frame's draw lists, flattened for a renderer that knows no Dear ImGui type: one vertex array, one array of 32-bit
// indices that already hold the vertex offsets, and the commands in drawing order. Valid until the next ui_menu_frame, and
// to be read on the thread that made it. An empty set (the menu closed and the overlay off, or no Dear ImGui) says so with
// zero commands. A vertex is Dear ImGui's own: position and texture coordinates in window pixels, and a colour as
// 0xAABBGGRR (red in the low byte). A command draws index_count indices from first_index with texture `texture` (the id
// the renderer's create returned), inside clip = x0, y0, x1, y1 in window pixels.
typedef struct UiVertex {
    float x, y, u, v;
    uint32_t color;
} UiVertex;
typedef struct UiCmd {
    float clip[4];
    uint64_t texture;
    uint32_t first_index, index_count;
} UiCmd;
typedef struct UiDrawData {
    const UiVertex *vertices;
    uint32_t vertex_count;
    const uint32_t *indices;
    uint32_t index_count;
    const UiCmd *cmds;
    uint32_t cmd_count;
} UiDrawData;
void ui_menu_draw_data(UiDrawData *out);

// The renderer's textures. Dear ImGui builds its font atlas as a texture and asks its renderer to create, update and
// destroy it; with these set, ui_menu_frame passes every such request on instead of answering it itself (the way the
// frame runs with no renderer). create returns the texture's id (not 0) or 0 when it cannot; pixels are width x height texels,
// 4 bytes each (red, green, blue, alpha) or 1 byte each when alpha_only. update gives the changed rectangle, whose first
// texel is at pixels and whose rows are pitch bytes apart. destroy comes once no frame that drew the texture can still be
// in flight. All three run inside ui_menu_frame, on the thread that builds the frame; a NULL set restores the default.
typedef struct UiTextureOps {
    uint64_t (*create)(int width, int height, int alpha_only, const uint8_t *pixels);
    void (*update)(uint64_t id, int x, int y, int width, int height, int alpha_only, const uint8_t *pixels, int pitch);
    void (*destroy)(uint64_t id);
} UiTextureOps;
void ui_menu_set_texture_ops(const UiTextureOps *ops);
// Make the context and the font atlas now, for a window of width by height pixels, and pass the atlas to the texture ops,
// so that the first frame that shows something creates nothing. 1 when it did, 0 without Dear ImGui.
int ui_menu_prepare(int width, int height);

#ifdef __cplusplus
}
#endif
