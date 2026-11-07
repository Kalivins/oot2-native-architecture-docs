// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

namespace Ryujinx.Graphics.Shader
{
    /// <summary>
    /// How a task or mesh draw tells its stages the draw's first task (architecture D16), and the record an indirect draw's conversion pass writes. The runtime's
    /// copy is runtime/include/oot2/mesh_abi.h; tests/test_mesh_abi.py fails when the two differ, so a number changed here is changed there.
    ///
    /// Push constants, bytes 24 to 35 (words 6 to 8, the vertex stage's slots 2 to 4, which no stage of a mesh pipeline uses; UniLayout.FirstTaskOffset and FirstTaskBytes):
    ///   words 6, 7: the device address (low, high), 4-byte aligned, of one 32-bit word that holds the draw's first task
    ///   word 8:     kept, read by nothing; the host writes 0
    /// A task stage adds that word to the workgroup number it reads (ISBERD mode 0x78, or 0x38 at address 0), and so does a mesh stage that has no task stage. A mesh
    /// stage under a task is numbered within its task's launch and adds nothing. A module that reads the word carries the capabilities of physical addressing
    /// (description: mesh_stage.first_task.read).
    ///
    /// A direct draw (vkCmdDrawMeshTasksEXT, group count in the call) points the address at a word it wrote: a word of zero that lives as long as the device when the first task is
    /// 0, a word of a ring the frame owns otherwise. An indirect draw (vkCmdDrawMeshTasksIndirectEXT) points it at the word of its record in the conversion pass's output, which
    /// is an array of records of 16 bytes: the three group counts the extension reads (count, 1, 1), then the first task. The word is read by the shader, so a barrier makes the
    /// conversion pass's write visible to the task and mesh stages as well as to the indirect command read.
    /// </summary>
    public static class MeshAbi
    {
        public const int FirstTaskPushWord = 6;
        public const int FirstTaskPushOffset = 24;
        public const int FirstTaskAddressBytes = 8;
        public const int FirstTaskWordBytes = 4;

        public const int IndirectCommandBytes = 12;
        public const int IndirectFirstOffset = 12;
        public const int IndirectRecordBytes = 16;
    }
}
