// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
// oot2-native: added to the vendored tree (xlate/third_party/README.md)

namespace Ryujinx.Graphics.Shader
{
    /// <summary>
    /// oot2-native: textures read through push constant slots (the layout `uni` of oot2-native's stage bindings work). A texture whose
    /// <see cref="TextureDefinition.Set"/> is <see cref="SlotSet"/> is no descriptor of its own: its binding number is a slot of
    /// 32-bit words in the push constants, from byte <see cref="SlotsOffset"/>, holding an NVN texture handle (texture ID in
    /// bits 0..19, sampler ID in bits 20..31). The shader loads the image at that texture ID from the global image array and
    /// the sampler at that sampler ID from the global sampler array, and combines them. An array of textures takes one slot:
    /// element i reads the image at texture ID + i and the sampler at sampler ID + i, so the host lays an array's elements
    /// out in consecutive entries of both arrays.
    /// </summary>
    public sealed class PushTextureLayout
    {
        /// <summary>The set number that marks a texture definition as a push constant slot; never written to the module.</summary>
        public int SlotSet { get; init; }

        public int ImagesSet { get; init; }
        public int ImagesBinding { get; init; }
        public int ImagesLength { get; init; }
        public int SamplersSet { get; init; }
        public int SamplersBinding { get; init; }
        public int SamplersLength { get; init; }

        /// <summary>Byte offset of slot 0 in the push constants; the bytes before it belong to the host.</summary>
        public int SlotsOffset { get; init; }

        /// <summary>Slots in the push constant block; the block is <see cref="SlotsOffset"/> + 4 * SlotCount bytes.</summary>
        public int SlotCount { get; init; }
    }
}
