// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using System;
using System.Collections.Generic;

namespace Ryujinx.Graphics.Shader
{
    public enum MeshStageKind
    {
        Task,
        Mesh,
    }

    /// <summary>One component a task or mesh program writes: Location is -1 for the position, 0 to 31 for a generic attribute, Component 0 to 3.</summary>
    public readonly record struct MeshComponent(int Location, int Component);

    /// <summary>
    /// What the translator needs to turn a Switch 2 task or mesh program into a VK_EXT_mesh_shader stage: the sizes the
    /// control section gives and the components the program header names. The stage memory the program talks to
    /// (ISBERD, ISBEWR) becomes four regions: the task payload, the attribute lines of the mesh outputs, the map bytes
    /// and the primitive count. The model is Lichii's (GPL-3.0-or-later; emit_spirv_mesh.cpp, DefineMesh).
    /// </summary>
    public sealed class MeshStageInfo
    {
        /// <summary>Attribute lines are groups of this many vertices or primitives, one 32-word line per component.</summary>
        public const int Block = 32;

        public MeshStageKind Kind { get; init; }

        /// <summary>The workgroup's invocations (control +0x58).</summary>
        public int Threads { get; init; }

        /// <summary>Words of the payload region: a task's output, which a mesh stage with a task reads as its input. At least 2 for a task (the group count is word 1).</summary>
        public int PayloadWords { get; init; }

        /// <summary>A mesh stage with a task reads the payload; without one it has no payload variable.</summary>
        public bool HasTask { get; init; }

        public int MaxVertices { get; init; }

        public int MaxPrimitives { get; init; }

        /// <summary>Per-vertex components in the order of the attribute lines (position first, then generics ascending).</summary>
        public IReadOnlyList<MeshComponent> VertexLines { get; init; } = Array.Empty<MeshComponent>();

        /// <summary>Per-primitive components in line order, generic attributes only.</summary>
        public IReadOnlyList<MeshComponent> PrimitiveLines { get; init; } = Array.Empty<MeshComponent>();

        public bool ReadsPayload => Kind == MeshStageKind.Task || HasTask;

        /// <summary>Word of the first primitive line in the attribute region: vertex groups come first.</summary>
        public int PrimitiveBase => Groups(MaxVertices) * VertexLines.Count * Block;

        /// <summary>Words of the attribute region (a mesh program's ISBEWR 0xf8 target), at least 1.</summary>
        public int AttributeWords => Math.Max(PrimitiveBase + Groups(MaxPrimitives) * PrimitiveLines.Count * Block, 1);

        /// <summary>Words of the map region: one word per map byte, the count apart (byte 3), index bytes from byte 4.</summary>
        public int MapWords => 4 + 3 * MaxPrimitives;

        private static int Groups(int count) => (count + Block - 1) / Block;
    }
}
