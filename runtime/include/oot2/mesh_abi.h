/* SPDX-License-Identifier: GPL-3.0-or-later */
/* Copyright (C) 2026 The oot2-native contributors */
#ifndef OOT2_MESH_ABI_H
#define OOT2_MESH_ABI_H

/*
 * How a task or mesh draw tells its stages the draw's first task (architecture D16), as the runtime writes it.
 * The translator's copy is xlate/third_party/ryujinx/src/Ryujinx.Graphics.Shader/MeshAbi.cs, which documents
 * it; tests/test_mesh_abi.py fails when the two differ.
 *
 * Push constants, bytes 24 to 35 (words 6 to 8, the vertex stage's slots 2 to 4, which a mesh pipeline does not
 * use): words 6 and 7 are the device address (low, high), 4-byte aligned, of one 32-bit word that holds the draw's
 * first task; word 8 is kept and written as 0. A task stage adds that word to the workgroup number it reads, and
 * so does a mesh stage that has no task stage; a mesh stage under a task adds nothing.
 *
 * A direct draw (vkCmdDrawMeshTasksEXT) points the address at a word it wrote: one word of zero that lives as
 * long as the device when the first task is 0, a word of a ring the frame owns otherwise. An indirect draw
 * (vkCmdDrawMeshTasksIndirectEXT) points it at the word of its record in the conversion pass's output, an array
 * of records of MESH_INDIRECT_RECORD_BYTES: the three group counts of VkDrawMeshTasksIndirectCommandEXT (count,
 * 1, 1), then the first task. The shader reads that word, so the barrier that follows the conversion pass makes
 * it visible to the task and mesh stages (shader read) as well as to the indirect command read.
 */

#define MESH_FIRST_TASK_PUSH_WORD 6
#define MESH_FIRST_TASK_PUSH_OFFSET 24
#define MESH_FIRST_TASK_ADDRESS_BYTES 8
#define MESH_FIRST_TASK_WORD_BYTES 4

#define MESH_INDIRECT_COMMAND_BYTES 12
#define MESH_INDIRECT_FIRST_OFFSET 12
#define MESH_INDIRECT_RECORD_BYTES 16

#endif
