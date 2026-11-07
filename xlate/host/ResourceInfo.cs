// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Ryujinx.Graphics.Shader;
using System;
using System.Collections.Generic;

namespace Oot2.Xlate;

/// <summary>
/// The texture type vocabulary of the job contract
/// ("1d|1darray|2d|2darray|2dms|2dmsarray|3d|cube|cubearray|buffer") and the
/// translator's SamplerType flags.
/// </summary>
internal static class TypeNames
{
    // Adapted from nxnative dm/gfx/tools/dmsh_xlate/Program.cs (MIT): the mapping of the answer strings.
    private static readonly Dictionary<string, SamplerType> ByName = new(StringComparer.Ordinal)
    {
        ["1d"] = SamplerType.Texture1D,
        ["1darray"] = SamplerType.Texture1D | SamplerType.Array,
        ["buffer"] = SamplerType.TextureBuffer,
        ["2d"] = SamplerType.Texture2D,
        ["2darray"] = SamplerType.Texture2D | SamplerType.Array,
        ["2dms"] = SamplerType.Texture2D | SamplerType.Multisample,
        ["2dmsarray"] = SamplerType.Texture2D | SamplerType.Multisample | SamplerType.Array,
        ["3d"] = SamplerType.Texture3D,
        ["cube"] = SamplerType.TextureCube,
        ["cubearray"] = SamplerType.TextureCube | SamplerType.Array,
    };

    public static bool TryParse(string name, out SamplerType type)
    {
        return ByName.TryGetValue(name, out type);
    }

    /// <summary>The contract name of a type; "sampler" for a bare sampler, which has no dimension.</summary>
    public static string Describe(SamplerType type)
    {
        string dim = (type & SamplerType.Mask) switch
        {
            SamplerType.None => "sampler",
            SamplerType.Texture1D => "1d",
            SamplerType.TextureBuffer => "buffer",
            SamplerType.Texture2D => "2d",
            SamplerType.Texture3D => "3d",
            SamplerType.TextureCube => "cube",
            _ => "unknown",
        };

        if (dim is "sampler" or "buffer" or "unknown")
        {
            return dim;
        }

        if ((type & SamplerType.Multisample) != 0)
        {
            dim += "ms";
        }

        if ((type & SamplerType.Array) != 0)
        {
            dim += "array";
        }

        return dim;
    }
}

/// <summary>
/// Decoding of the texture handles the translator hands to the accessor into
/// the numbering the job answers use (the slots of NVN's handle tables in c[2]).
/// </summary>
internal static class HandleSlots
{
    // NVN's handle tables in c[2], byte offsets, 8 bytes per slot: combined
    // textures 0x20 + 8 r, images 0x120 + 8 r (8 slots), separate textures
    // 0x168 + 8 r (numbered 128 + r), separate samplers 0x568 + 8 r.
    private const int TexBase = 0x20;
    private const int ImgBase = 0x120;
    private const int ImgSlots = 8;
    private const int SepTexBase = 0x168;
    private const int SepTexSlot0 = 128;
    private const int SampBase = 0x568;
    private const int SampSlots = 128;

    /// <summary>
    /// A handle's texture or image slot, or -1 when neither of its two c[2]
    /// words is in the table. The handle ORs two words in no fixed order, so
    /// both are tried; only the first for a combined handle or a constant sampler.
    /// Adapted from nxnative dm/gfx/tools/dmsh_xlate/Program.cs, Slot() (MIT).
    /// </summary>
    public static int Slot(int handle, int cbufSlot, bool image)
    {
        if (cbufSlot >= 0 && (cbufSlot & 0xffff) != 2)
        {
            return -1;
        }

        (int w0, int w1, TextureHandleType type) = TextureHandle.UnpackOffsets(handle);
        bool firstOnly = type is TextureHandleType.CombinedSampler or TextureHandleType.SeparateConstantSamplerHandle;

        for (int i = 0; i < (firstOnly ? 1 : 2); i++)
        {
            int b = (i == 0 ? w0 : w1) * 4;

            if (b % 8 != 0)
            {
                continue;
            }

            if (image && b >= ImgBase && b < ImgBase + 8 * ImgSlots)
            {
                return (b - ImgBase) / 8;
            }

            if (!image && b >= TexBase && b < SepTexBase)
            {
                return (b - TexBase) / 8;
            }

            if (!image && b >= SepTexBase && b < SampBase)
            {
                return SepTexSlot0 + (b - SepTexBase) / 8;
            }
        }

        return -1;
    }

    /// <summary>The slot of a sampler word (separate samplers), or -1; numbered from 0.</summary>
    public static int SamplerSlot(int handle)
    {
        (int w0, int w1, TextureHandleType type) = TextureHandle.UnpackOffsets(handle);

        foreach (int w in type is TextureHandleType.CombinedSampler or TextureHandleType.SeparateConstantSamplerHandle
                     ? new[] { w0 }
                     : new[] { w0, w1 })
        {
            int b = w * 4;

            if (b % 8 == 0 && b >= SampBase && b < SampBase + 8 * SampSlots)
            {
                return (b - SampBase) / 8;
            }
        }

        return -1;
    }
}
