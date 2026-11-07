// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The overlay's vertex stage (runtime/gpu/vk_overlay.c): the menu's vertices come in window pixels and leave in clip space
// through one scale and one translation, pushed with the draw. The colour is Dear ImGui's, 8 bits a channel, as it is.
#version 450

layout(location = 0) in vec2 in_pos;
layout(location = 1) in vec2 in_uv;
layout(location = 2) in vec4 in_color;

layout(push_constant) uniform Push {
    vec2 scale;
    vec2 translate;
} pc;

layout(location = 0) out vec4 out_color;
layout(location = 1) out vec2 out_uv;

void main() {
    out_color = in_color;
    out_uv = in_uv;
    gl_Position = vec4(in_pos * pc.scale + pc.translate, 0.0, 1.0);
}
