// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Ryujinx.Graphics.Shader;
using System;
using System.Collections.Generic;

namespace Oot2.Xlate;

/// <summary>
/// The descriptor layout `uni` (docs/architecture.md D13; spike S3, ticket 20; numbering by ticket 23). Every module of a
/// stage numbers its resources the same way, whatever program it is paired with, so that one pipeline layout serves every
/// graphics pipeline and a vertex or pixel module is built once:
///
///   set 0, global and update-after-bind, bound once per command buffer: binding 0 the game's texture pool, sampled images
///          (every image type a view of the one binding), binding 1 its sampler pool.
///   set 1, push descriptors (32 at most on the reference GPU): binding 0 the support buffer, shared by every stage, then
///          fixed slots per stage and kind, a module's n-th constant bank in its stage's n-th uniform slot and likewise for
///          storage buffers and storage images:
///            vertex or mesh   uniform 1..8, storage 9..13
///            pixel            uniform 14..24, storage 25..29, storage image 30..31
///          A compute pipeline has a layout of its own with the same sets: uniform 1..8, storage 9..24, storage image 25..31.
///   set 2, task only (mesh pipelines): uniform 0..7, storage 8..15, a set the host writes per draw, for want of push slots.
///   push constants, 128 bytes: 16 for the page-table address and span (D15), then 28 texture slots of 4 bytes, each an NVN
///          texture handle read by the module (Ryujinx.Graphics.Shader.PushTextureLayout): vertex 0..4, mesh 0..1, pixel
///          5..27, compute 0..27. Bound textures and texture arrays take one slot each; pool textures take none.
///          Bytes 24..35 (the vertex stage's slots 2..4, free in a mesh pipeline, which has no vertex stage) are kept for
///          the draw's first task, read by the task stage or, without one, the mesh stage; the mesh bindings ticket settles
///          what they hold (ticket 59). A module reads them as words 6..8 of the block the slots are declared in.
///
/// The slot counts are the corpus maxima of spike S3's census and of tickets 23 and 59 for compute, task and mesh (ticket
/// 59 counted mesh programs at 2 textures at most, in the update's archives and in the base-only ones). A module that
/// needs more than its stage has still translates: the resources past the range get bindings from 32 (slots from 28), and
/// the description says it does not fit. Texel buffers and arrays of storage images have no slot and are handled the same.
/// </summary>
internal sealed class UniLayout
{
    public const string Name = "uni";
    public const int GlobalSet = 0;
    public const int PushSet = 1;
    public const int TaskSet = 2;
    public const int ImagesBinding = 0;
    public const int SamplersBinding = 1;
    public const int SupportBinding = 0;
    public const int PushConstantBytes = 128;
    public const int SlotsOffset = 16;
    public const int SlotCount = 28;
    public const int PushDescriptorLimit = 32;

    // The first task of a mesh pipeline's draw: the bytes of the vertex stage's texture slots 2..4.
    public const int FirstTaskOffset = 24;
    public const int FirstTaskBytes = 12;

    // A set number no module declares: Ryujinx's texture definitions in it are push constant slots, the binding being the slot.
    public const int SlotSet = 15;

    private sealed record Range(int First, int Count);

    private sealed record Role(string Name, int BufferSet, Range Uniform, Range Storage, Range Image, Range Slots, Range FirstTask);

    private static readonly Range None = new(0, 0);
    private static readonly Range FirstTaskRange = new(FirstTaskOffset, FirstTaskBytes);

    private static readonly Role PreRasterization = new("pre_rasterization", PushSet, new(1, 8), new(9, 5), None, new(0, 5), None);
    private static readonly Role Mesh = new("mesh", PushSet, new(1, 8), new(9, 5), None, new(0, 2), FirstTaskRange);
    private static readonly Role Pixel = new("pixel", PushSet, new(14, 11), new(25, 5), new(30, 2), new(5, 23), None);
    private static readonly Role Compute = new("compute", PushSet, new(1, 8), new(9, 16), new(25, 7), new(0, 28), None);
    private static readonly Role Task = new("task", TaskSet, new(0, 8), new(8, 8), None, None, FirstTaskRange);

    private readonly Role _role;
    private int _uniforms, _storage, _images, _slots, _overflowBindings, _overflowSlots;

    public UniLayout(string stage)
    {
        _role = stage switch
        {
            "vs" => PreRasterization,
            "ms" => Mesh,
            "ps" => Pixel,
            "ts" => Task,
            _ => Compute,
        };
    }

    public string RoleName => _role.Name;

    public int BufferSet => _role.BufferSet;

    public (int First, int Count) Uniform => (_role.Uniform.First, _role.Uniform.Count);

    public (int First, int Count) Storage => (_role.Storage.First, _role.Storage.Count);

    public (int First, int Count) Image => (_role.Image.First, _role.Image.Count);

    public (int First, int Count) Slots => (_role.Slots.First, _role.Slots.Count);

    /// <summary>The push constant bytes kept for the draw's first task (offset, length); (0, 0) for a stage that has none.</summary>
    public (int First, int Count) FirstTask => (_role.FirstTask.First, _role.FirstTask.Count);

    /// <summary>What did not fit the stage's ranges, one line per resource; empty when the module fits.</summary>
    public List<string> Overflow { get; } = new();

    public SetBindingPair NextUniform(int bank) => Next(_role.Uniform, ref _uniforms, $"constant bank c{bank}", "uniform");

    public SetBindingPair NextStorage(int ordinal) => Next(_role.Storage, ref _storage, $"storage buffer {ordinal}", "storage");

    public SetBindingPair NextImage(int count, bool isBuffer)
    {
        if (count != 1 || isBuffer)
        {
            return OverflowBinding(isBuffer ? "storage texel buffer" : $"array of {count} storage images");
        }

        return Next(_role.Image, ref _images, "storage image", "storage image");
    }

    /// <summary>A uniform texel buffer: no slot of the layout holds one.</summary>
    public SetBindingPair TexelBuffer() => OverflowBinding("uniform texel buffer");

    public SetBindingPair NextTextureSlot(int count)
    {
        string what = count == 1 ? "texture" : $"array of {count} textures";

        if (_slots < _role.Slots.Count)
        {
            return new(SlotSet, _role.Slots.First + _slots++);
        }

        _slots++;
        Overflow.Add($"{what}: no texture slot left for the {_role.Name} stage ({_role.Slots.Count})");

        return new(SlotSet, SlotCount + _overflowSlots++);
    }

    /// <summary>How the translator reads textures given a push constant slot.</summary>
    public static PushTextureLayout PushTextures(int heapImages, int heapSamplers)
    {
        return new PushTextureLayout
        {
            SlotSet = SlotSet,
            ImagesSet = GlobalSet,
            ImagesBinding = ImagesBinding,
            // The pool's own arrays have these lengths (BindlessElimination.GenerateBindlessAccess), so the views are shared
            ImagesLength = Math.Max(2, heapImages),
            SamplersSet = GlobalSet,
            SamplersBinding = SamplersBinding,
            SamplersLength = Math.Max(2, heapSamplers),
            SlotsOffset = SlotsOffset,
            SlotCount = SlotCount,
        };
    }

    private SetBindingPair Next(Range range, ref int used, string what, string kind)
    {
        if (used < range.Count)
        {
            return new(_role.BufferSet, range.First + used++);
        }

        used++;

        return OverflowBinding($"{what}: no {kind} slot left for the {_role.Name} stage ({range.Count})");
    }

    private SetBindingPair OverflowBinding(string what)
    {
        Overflow.Add(what);

        return new(_role.BufferSet, PushDescriptorLimit + _overflowBindings++);
    }
}
