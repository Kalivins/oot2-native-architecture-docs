// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Spv.Generator;
using static Spv.Specification;

namespace Ryujinx.Graphics.Shader.CodeGen.Spirv
{
    partial class CodeGenContext
    {
        /// <summary>The variables of a task or mesh stage's declarations (MeshEmission); null for any other module.</summary>
        internal MeshEmission.State MeshState { get; set; }

        // SPIR-V 1.4 drops the BufferBlock decoration (the Uniform storage class with it): a storage buffer of a task or mesh module is a StorageBuffer variable of a Block structure
        public StorageClass StorageBufferClass => Definitions.MeshStage != null ? StorageClass.StorageBuffer : StorageClass.Uniform;

        public Decoration StorageBufferBlock => Definitions.MeshStage != null ? Decoration.Block : Decoration.BufferBlock;
    }
}
