// oot2-native: the two instructions of SPV_EXT_mesh_shader, which the vendored grammar (Core 1.x, the NV
// names only) does not have, and a read-only view of the module's global variables (the entry point of a
// SPIR-V 1.4 module lists all of them). MIT, like the rest of Spv.Generator.

using System.Collections.Generic;
using static Spv.Specification;

namespace Spv.Generator
{
    public partial class Module
    {
        /// <summary>Every global variable declared so far, in declaration order.</summary>
        public IReadOnlyList<Instruction> GlobalVariables => _globals;

        /// <summary>OpSetMeshOutputsEXT: the vertex and primitive counts of this workgroup's output.</summary>
        public Instruction SetMeshOutputs(Instruction vertexCount, Instruction primitiveCount)
        {
            Instruction result = NewInstruction(Op.OpSetMeshOutputsEXT);

            result.AddOperand(vertexCount);
            result.AddOperand(primitiveCount);
            AddToFunctionDefinitions(result);

            return result;
        }

        /// <summary>OpEmitMeshTasksEXT: a block terminator that launches groupCountX * Y * Z mesh workgroups with the payload.</summary>
        public Instruction EmitMeshTasks(Instruction groupCountX, Instruction groupCountY, Instruction groupCountZ, Instruction payload)
        {
            Instruction result = NewInstruction(Op.OpEmitMeshTasksEXT);

            result.AddOperand(groupCountX);
            result.AddOperand(groupCountY);
            result.AddOperand(groupCountZ);
            result.AddOperand(payload);
            AddToFunctionDefinitions(result);

            return result;
        }
    }
}
