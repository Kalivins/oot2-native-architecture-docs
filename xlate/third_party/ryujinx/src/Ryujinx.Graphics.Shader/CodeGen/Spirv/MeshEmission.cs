// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors
//
// Task and mesh stages (SPV_EXT_mesh_shader) for Switch 2 programs. A C# port of the model of Lichii's
// src/shader_recompiler/backend/spirv/emit_spirv_mesh.cpp and EmitContext::DefineMesh (SPDX-FileCopyrightText:
// Copyright 2026 Eden Emulator Project, GPL-3.0-or-later), written over this generator.
//
// The programs talk to the hardware through the stage buffer (ISBERD, ISBEWR). A task program writes the number of
// mesh workgroups at byte 4 of its output region and its payload after it; each mesh workgroup reads that region (its
// own number at byte 0) and writes its vertices and primitives to an attribute region in groups of 32, one 32-word line
// per component, and the primitive count and 8-bit vertex indices to a map region. Here a task's output region is the
// task payload, the mesh output regions are workgroup memories, and one epilogue (reached by every invocation, since
// every exit of the program branches to one return) hands them to Vulkan after a barrier.

using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.StructuredIr;
using System;
using System.Collections.Generic;
using static Spv.Specification;
using SpvInstruction = Spv.Generator.Instruction;
using SpvLiteralInteger = Spv.Generator.LiteralInteger;

namespace Ryujinx.Graphics.Shader.CodeGen.Spirv
{
    static class MeshEmission
    {
        // The memories the frontend addresses by id (ResourceManager.SetMeshMemories) are found by these names
        public const string PayloadName = "isbe_payload";
        public const string AttributesName = "isbe_attributes";
        public const string MapName = "isbe_map";
        public const string CountName = "isbe_primitive_count";

        // SPV_EXT_mesh_shader needs SPIR-V 1.4
        public const uint SpirvVersion = (1u << 16) | (4u << 8);

        /// <summary>An output variable and, for a vec4 (the position), the component this line writes; -1 for a scalar array.</summary>
        internal readonly record struct Target(SpvInstruction Variable, int Element);

        internal sealed class State
        {
            public SpvInstruction Payload;
            public SpvInstruction Attributes;
            public SpvInstruction Map;
            public SpvInstruction Count;
            public SpvInstruction PrimitiveIndices;
            public readonly List<Target> VertexTargets = [];
            public readonly List<Target> PrimitiveTargets = [];
        }

        private static SpvInstruction FindMemory(CodeGenContext context, string name)
        {
            foreach ((int id, MemoryDefinition memory) in context.Properties.SharedMemories)
            {
                if (memory.Name == name)
                {
                    return context.SharedMemories[id];
                }
            }

            return null;
        }

        private static SpvInstruction Declare(CodeGenContext context, SpvInstruction arrayType, string name)
        {
            SpvInstruction pointerType = context.TypePointer(StorageClass.Output, arrayType);
            SpvInstruction variable = context.Variable(pointerType, StorageClass.Output);

            context.Name(variable, name);
            context.AddGlobalVariable(variable);

            return variable;
        }

        /// <summary>
        /// The outputs of a mesh stage, from the job's map and nothing else: the position as an array of vec4 (a built-in has to
        /// be whole), every other written component as an array of float at its location and component, per-primitive ones
        /// decorated PerPrimitiveEXT, and the triangle indices. A component the map does not name is never declared (the
        /// worst program is at 86 percent of the output memory limit, the mesh shader study).
        /// </summary>
        public static void DeclareOutputs(CodeGenContext context)
        {
            MeshStageInfo mesh = context.Definitions.MeshStage;
            State state = new() { Payload = FindMemory(context, PayloadName) };

            context.MeshState = state;

            if (mesh.Kind == MeshStageKind.Task)
            {
                if (state.Payload == null)
                {
                    throw new InvalidOperationException("a task program has no payload memory");
                }

                return;
            }

            state.Attributes = FindMemory(context, AttributesName);
            state.Map = FindMemory(context, MapName);
            state.Count = FindMemory(context, CountName);

            if (state.Attributes == null || state.Map == null || state.Count == null)
            {
                throw new InvalidOperationException("a mesh program lacks one of its stage memories");
            }

            SpvInstruction u32 = context.TypeU32();
            SpvInstruction f32 = context.TypeFP32();
            SpvInstruction vertexCount = context.Constant(u32, mesh.MaxVertices);
            SpvInstruction primitiveCount = context.Constant(u32, mesh.MaxPrimitives);

            state.PrimitiveIndices = Declare(context, context.TypeArray(context.TypeVector(u32, 3), primitiveCount), "isbe_triangle_indices");
            context.Decorate(state.PrimitiveIndices, Decoration.BuiltIn, (SpvLiteralInteger)BuiltIn.PrimitiveTriangleIndicesEXT);

            SpvInstruction position = null;

            foreach (MeshComponent line in mesh.VertexLines)
            {
                if (line.Location < 0)
                {
                    if (position == null)
                    {
                        position = Declare(context, context.TypeArray(context.TypeVector(f32, 4), vertexCount), "isbe_position");
                        context.Decorate(position, Decoration.BuiltIn, (SpvLiteralInteger)BuiltIn.Position);
                    }

                    state.VertexTargets.Add(new Target(position, line.Component));
                }
                else
                {
                    state.VertexTargets.Add(new Target(DeclareComponent(context, line, vertexCount, perPrimitive: false), -1));
                }
            }

            foreach (MeshComponent line in mesh.PrimitiveLines)
            {
                state.PrimitiveTargets.Add(new Target(DeclareComponent(context, line, primitiveCount, perPrimitive: true), -1));
            }
        }

        private static SpvInstruction DeclareComponent(CodeGenContext context, MeshComponent line, SpvInstruction count, bool perPrimitive)
        {
            SpvInstruction variable = Declare(context, context.TypeArray(context.TypeFP32(), count), $"isbe_{(perPrimitive ? "prim" : "vert")}_l{line.Location}_c{line.Component}");

            context.Decorate(variable, Decoration.Location, (SpvLiteralInteger)line.Location);
            context.Decorate(variable, Decoration.Component, (SpvLiteralInteger)line.Component);

            if (perPrimitive)
            {
                context.Decorate(variable, Decoration.PerPrimitiveNV);
            }

            return variable;
        }

        /// <summary>The entry point and its execution modes. Declared after every function is generated: from SPIR-V 1.4 the interface lists every global variable.</summary>
        public static void AddEntryPoint(CodeGenContext context, SpvInstruction main)
        {
            MeshStageInfo mesh = context.Definitions.MeshStage;
            bool task = mesh.Kind == MeshStageKind.Task;

            context.AddEntryPoint(task ? ExecutionModel.TaskEXT : ExecutionModel.MeshEXT, main, "main", context.GetMainInterface());
            context.AddExecutionMode(
                main,
                ExecutionMode.LocalSize,
                (SpvLiteralInteger)context.Definitions.ComputeLocalSizeX,
                (SpvLiteralInteger)context.Definitions.ComputeLocalSizeY,
                (SpvLiteralInteger)context.Definitions.ComputeLocalSizeZ);

            if (!task)
            {
                // OutputPrimitivesNV and OutputTrianglesNV are the numbers of OutputPrimitivesEXT and OutputTrianglesEXT
                context.AddExecutionMode(main, ExecutionMode.OutputVertices, (SpvLiteralInteger)mesh.MaxVertices);
                context.AddExecutionMode(main, ExecutionMode.OutputPrimitivesNV, (SpvLiteralInteger)mesh.MaxPrimitives);
                context.AddExecutionMode(main, ExecutionMode.OutputTrianglesNV);
            }
        }

        /// <summary>Every invocation of a mesh workgroup stores the zero count, then one barrier, before the program's first instruction.</summary>
        public static void Prologue(CodeGenContext context)
        {
            if (context.Definitions.MeshStage.Kind != MeshStageKind.Mesh)
            {
                return;
            }

            context.Store(CountPointer(context), context.Constant(context.TypeU32(), 0));
            Barrier(context);
        }

        /// <summary>
        /// What the main function's one return does. A task: a barrier, the group count from payload word 1, and the
        /// terminator OpEmitMeshTasksEXT with the payload. A mesh stage: a barrier, OpSetMeshOutputsEXT with the declared
        /// maximum vertex count and the clamped primitive count, then the copy of every vertex and primitive line to the outputs.
        /// </summary>
        public static void Epilogue(CodeGenContext context)
        {
            if (context.CurrentBlock.Type != AstBlockType.Main)
            {
                throw new InvalidOperationException("the return of a task or mesh program is not in its main block: its exits did not converge");
            }

            MeshStageInfo mesh = context.Definitions.MeshStage;
            State state = context.MeshState;
            SpvInstruction u32 = context.TypeU32();

            Barrier(context);

            if (mesh.Kind == MeshStageKind.Task)
            {
                SpvInstruction one = context.Constant(u32, 1);
                SpvInstruction word = context.AccessChain(context.TypePointer(StorageClass.TaskPayloadWorkgroupEXT, u32), state.Payload, one);

                context.EmitMeshTasks(context.Load(u32, word), one, one, state.Payload);

                return;
            }

            SpvInstruction count = Min(context, context.Load(u32, CountPointer(context)), context.Constant(u32, mesh.MaxPrimitives));
            SpvInstruction maxVertices = context.Constant(u32, mesh.MaxVertices);

            context.SetMeshOutputs(maxVertices, count);

            SpvInstruction invocation = LocalInvocation(context);
            int threads = mesh.Threads;
            int vertexLines = mesh.VertexLines.Count;
            int primitiveLines = mesh.PrimitiveLines.Count;

            for (int first = 0; first < mesh.MaxVertices; first += threads)
            {
                SpvInstruction vertex = context.IAdd(u32, invocation, context.Constant(u32, first));

                if (first + threads > mesh.MaxVertices)
                {
                    If(context, context.ULessThan(context.TypeBool(), vertex, maxVertices), () => CopyLines(context, state, vertex, 0, vertexLines, state.VertexTargets));
                }
                else
                {
                    CopyLines(context, state, vertex, 0, vertexLines, state.VertexTargets);
                }
            }

            SpvInstruction mapPointer = context.TypePointer(StorageClass.Workgroup, u32);
            SpvInstruction indicesPointer = context.TypePointer(StorageClass.Output, context.TypeVector(u32, 3));

            for (int first = 0; first < mesh.MaxPrimitives; first += threads)
            {
                SpvInstruction primitive = context.IAdd(u32, invocation, context.Constant(u32, first));

                If(context, context.ULessThan(context.TypeBool(), primitive, count), () =>
                {
                    // Vertex indices: map bytes 4 + 3 * primitive, one word a byte
                    SpvInstruction byteOffset = context.IAdd(u32, context.IMul(u32, primitive, context.Constant(u32, 3)), context.Constant(u32, 4));
                    SpvInstruction[] indices = new SpvInstruction[3];

                    for (int element = 0; element < 3; element++)
                    {
                        SpvInstruction address = context.IAdd(u32, byteOffset, context.Constant(u32, element));

                        indices[element] = context.Load(u32, context.AccessChain(mapPointer, state.Map, address));
                    }

                    context.Store(
                        context.AccessChain(indicesPointer, state.PrimitiveIndices, primitive),
                        context.CompositeConstruct(context.TypeVector(u32, 3), indices));
                    CopyLines(context, state, primitive, mesh.PrimitiveBase, primitiveLines, state.PrimitiveTargets);
                });
            }

            context.Return();
        }

        private static SpvInstruction CountPointer(CodeGenContext context)
        {
            SpvInstruction u32 = context.TypeU32();

            return context.AccessChain(context.TypePointer(StorageClass.Workgroup, u32), context.MeshState.Count, context.Constant(u32, 0));
        }

        private static SpvInstruction LocalInvocation(CodeGenContext context)
        {
            SpvInstruction u32 = context.TypeU32();
            SpvInstruction id = context.Inputs[new IoDefinition(StorageKind.Input, IoVariable.ThreadId)];

            return context.Load(u32, context.AccessChain(context.TypePointer(StorageClass.Input, u32), id, context.Constant(u32, 0)));
        }

        private static SpvInstruction Min(CodeGenContext context, SpvInstruction a, SpvInstruction b)
        {
            return context.Select(context.TypeU32(), context.ULessThan(context.TypeBool(), a, b), a, b);
        }

        private static void Barrier(CodeGenContext context)
        {
            SpvInstruction u32 = context.TypeU32();

            context.ControlBarrier(
                context.Constant(u32, Scope.Workgroup),
                context.Constant(u32, Scope.Workgroup),
                context.Constant(u32, MemorySemanticsMask.WorkgroupMemory | MemorySemanticsMask.AcquireRelease));
        }

        // Runs body under if (condition)
        private static void If(CodeGenContext context, SpvInstruction condition, Action body)
        {
            SpvInstruction thenLabel = context.Label();
            SpvInstruction mergeLabel = context.Label();

            context.SelectionMerge(mergeLabel, SelectionControlMask.MaskNone);
            context.BranchConditional(condition, thenLabel, mergeLabel);
            context.AddLabel(thenLabel);
            body();
            context.Branch(mergeLabel);
            context.AddLabel(mergeLabel);
        }

        // The lines of element `index` (a vertex or a primitive) from the attribute region: the word of line 0 is
        // base + (index / 32) * lines * 32 + index % 32, each next line 32 words on.
        private static void CopyLines(CodeGenContext context, State state, SpvInstruction index, int baseWord, int lines, List<Target> targets)
        {
            SpvInstruction u32 = context.TypeU32();
            SpvInstruction f32 = context.TypeFP32();
            SpvInstruction group = context.ShiftRightLogical(u32, index, context.Constant(u32, 5));
            SpvInstruction slot = context.BitwiseAnd(u32, index, context.Constant(u32, 31));
            SpvInstruction first = context.IAdd(
                u32,
                context.IMul(u32, group, context.Constant(u32, lines * MeshStageInfo.Block)),
                context.IAdd(u32, slot, context.Constant(u32, baseWord)));
            SpvInstruction sharedPointer = context.TypePointer(StorageClass.Workgroup, u32);
            SpvInstruction outputPointer = context.TypePointer(StorageClass.Output, f32);

            for (int line = 0; line < lines; line++)
            {
                Target target = targets[line];
                SpvInstruction address = context.IAdd(u32, first, context.Constant(u32, line * MeshStageInfo.Block));
                SpvInstruction word = context.Load(u32, context.AccessChain(sharedPointer, state.Attributes, address));
                SpvInstruction pointer = target.Element < 0
                    ? context.AccessChain(outputPointer, target.Variable, index)
                    : context.AccessChain(outputPointer, target.Variable, index, context.Constant(u32, target.Element));

                context.Store(pointer, context.Bitcast(f32, word));
            }
        }
    }
}
