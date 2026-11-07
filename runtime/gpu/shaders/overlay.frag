// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// The overlay's fragment stage (runtime/gpu/vk_overlay.c): the vertex colour times the texel of the font atlas (or of
// another texture the menu draws), blended over the picture by the pipeline.
#version 450

layout(location = 0) in vec4 in_color;
layout(location = 1) in vec2 in_uv;

layout(set = 0, binding = 0) uniform sampler2D tex;

layout(location = 0) out vec4 out_color;

void main() {
    out_color = in_color * texture(tex, in_uv);
}
