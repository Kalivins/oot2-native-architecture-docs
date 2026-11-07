// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Ryujinx.Graphics.Shader;
using System;
using System.Collections.Generic;

namespace Oot2.Xlate;

/// <summary>
/// The VK_EXT_mesh_shader limits a task or mesh program is checked against. The translator has no device: prepare fills
/// the "mesh_limits" job option from the device, and a job without it takes the RTX 3060 Ti's, as read on the reference PC
/// (driver 617.14; research 13, finding 1). The names are the Vulkan properties' with the "maxMesh" or "maxTask" prefix
/// in snake case.
/// </summary>
internal sealed record MeshLimits
{
    public int MaxTaskInvocations { get; init; } = 128;
    public int MaxMeshInvocations { get; init; } = 128;
    public int MaxOutputVertices { get; init; } = 256;
    public int MaxOutputPrimitives { get; init; } = 256;
    public int MaxOutputComponents { get; init; } = 128;
    public int MaxOutputMemory { get; init; } = 32768;
    public int MaxSharedMemory { get; init; } = 28672;
    public int MaxPayloadAndSharedMemory { get; init; } = 28672;
    public int MaxPayloadAndOutputMemory { get; init; } = 48128;
    public int MaxTaskPayload { get; init; } = 16384;
    public int MaxTaskPayloadAndSharedMemory { get; init; } = 32768;

    public static readonly MeshLimits Default = new();

    /// <summary>The option names, in the order the description lists them.</summary>
    public static readonly string[] Names =
    [
        "max_task_invocations", "max_mesh_invocations", "max_mesh_output_vertices", "max_mesh_output_primitives",
        "max_mesh_output_components", "max_mesh_output_memory", "max_mesh_shared_memory", "max_mesh_payload_and_shared_memory",
        "max_mesh_payload_and_output_memory", "max_task_payload", "max_task_payload_and_shared_memory",
    ];

    public int Get(string name) => name switch
    {
        "max_task_invocations" => MaxTaskInvocations,
        "max_mesh_invocations" => MaxMeshInvocations,
        "max_mesh_output_vertices" => MaxOutputVertices,
        "max_mesh_output_primitives" => MaxOutputPrimitives,
        "max_mesh_output_components" => MaxOutputComponents,
        "max_mesh_output_memory" => MaxOutputMemory,
        "max_mesh_shared_memory" => MaxSharedMemory,
        "max_mesh_payload_and_shared_memory" => MaxPayloadAndSharedMemory,
        "max_mesh_payload_and_output_memory" => MaxPayloadAndOutputMemory,
        "max_task_payload" => MaxTaskPayload,
        "max_task_payload_and_shared_memory" => MaxTaskPayloadAndSharedMemory,
        _ => throw new ArgumentException(name),
    };

    public MeshLimits With(string name, int value) => name switch
    {
        "max_task_invocations" => this with { MaxTaskInvocations = value },
        "max_mesh_invocations" => this with { MaxMeshInvocations = value },
        "max_mesh_output_vertices" => this with { MaxOutputVertices = value },
        "max_mesh_output_primitives" => this with { MaxOutputPrimitives = value },
        "max_mesh_output_components" => this with { MaxOutputComponents = value },
        "max_mesh_output_memory" => this with { MaxOutputMemory = value },
        "max_mesh_shared_memory" => this with { MaxSharedMemory = value },
        "max_mesh_payload_and_shared_memory" => this with { MaxPayloadAndSharedMemory = value },
        "max_mesh_payload_and_output_memory" => this with { MaxPayloadAndOutputMemory = value },
        "max_task_payload" => this with { MaxTaskPayload = value },
        "max_task_payload_and_shared_memory" => this with { MaxTaskPayloadAndSharedMemory = value },
        _ => throw new ArgumentException(name),
    };
}

/// <summary>What a task or mesh program costs against the limits, by Vulkan's formulas (research 13, finding 3).</summary>
internal readonly record struct MeshSizes(int PayloadBytes, int OutputBytes, int VertexLocations, int PrimitiveLocations, int SharedBytes);

internal static class MeshChecks
{
    private const int Block = MeshStageInfo.Block;

    public static MeshSizes Measure(MeshStageInfo mesh)
    {
        int payload = mesh.ReadsPayload ? mesh.PayloadWords * 4 : 0;

        if (mesh.Kind == MeshStageKind.Task)
        {
            return new MeshSizes(payload, 0, 0, 0, 0);
        }

        int vertexLocations = Locations(mesh.VertexLines);
        int primitiveLocations = Locations(mesh.PrimitiveLines);

        // Vulkan counts every output location as four scalars of 4 bytes, for the maximum vertex and primitive counts
        // rounded up to the granularity of 32: the worst program of the dump (160 vertices, 11 locations) is 28,160 bytes
        int output = 16 * (vertexLocations * Align(mesh.MaxVertices) + primitiveLocations * Align(mesh.MaxPrimitives));

        // The staging of Lichii's model: the attribute lines, the map and the count, a word each
        int shared = (mesh.AttributeWords + mesh.MapWords + 1) * 4;

        return new MeshSizes(payload, output, vertexLocations, primitiveLocations, shared);
    }

    /// <summary>
    /// Refuse a program that cannot be a task or mesh stage on the device the limits describe, by the name of the first
    /// limit it is past, or that has shared memory of its own (the stage memory would lie over it, as nxnative refuses too).
    /// </summary>
    public static void Enforce(Job job)
    {
        MeshStageInfo mesh = job.MeshStage;

        if (mesh == null)
        {
            return;
        }

        if (job.MeshOwnShared)
        {
            throw new JobException(Kinds.UnsupportedStage, "a task or mesh program with shared memory of its own (LDS, STS or ATOMS) is not translated: its stage memory would lie over it");
        }

        MeshLimits limits = job.Options.MeshLimits;
        MeshSizes sizes = Measure(mesh);
        bool task = mesh.Kind == MeshStageKind.Task;

        Check(mesh.Threads, task ? limits.MaxTaskInvocations : limits.MaxMeshInvocations, task ? "maxTaskWorkGroupInvocations" : "maxMeshWorkGroupInvocations");

        if (task)
        {
            Check(sizes.PayloadBytes, limits.MaxTaskPayload, "maxTaskPayloadSize");
            Check(sizes.PayloadBytes, limits.MaxTaskPayloadAndSharedMemory, "maxTaskPayloadAndSharedMemorySize");

            return;
        }

        Check(mesh.MaxVertices, limits.MaxOutputVertices, "maxMeshOutputVertices");
        Check(mesh.MaxPrimitives, limits.MaxOutputPrimitives, "maxMeshOutputPrimitives");
        Check(Math.Max(sizes.VertexLocations, sizes.PrimitiveLocations) * 4, limits.MaxOutputComponents, "maxMeshOutputComponents");
        Check(sizes.OutputBytes, limits.MaxOutputMemory, "maxMeshOutputMemorySize");
        Check(sizes.SharedBytes, limits.MaxSharedMemory, "maxMeshSharedMemorySize");
        Check(sizes.PayloadBytes + sizes.SharedBytes, limits.MaxPayloadAndSharedMemory, "maxMeshPayloadAndSharedMemorySize");
        Check(sizes.PayloadBytes + sizes.OutputBytes, limits.MaxPayloadAndOutputMemory, "maxMeshPayloadAndOutputMemorySize");
    }

    private static void Check(int value, int limit, string name)
    {
        if (value > limit)
        {
            throw new JobException(Kinds.OverLimit, $"{name}: the program needs {value}, the device allows {limit}");
        }
    }

    private static int Align(int count) => (count + Block - 1) / Block * Block;

    private static int Locations(IReadOnlyList<MeshComponent> lines)
    {
        HashSet<int> seen = new();

        foreach (MeshComponent line in lines)
        {
            seen.Add(line.Location);
        }

        return seen.Count;
    }
}
