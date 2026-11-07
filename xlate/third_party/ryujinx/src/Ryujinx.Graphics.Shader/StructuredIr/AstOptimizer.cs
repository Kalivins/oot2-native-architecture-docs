using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using System;
using System.Collections.Generic;
using System.Linq;

using static Ryujinx.Graphics.Shader.StructuredIr.AstHelper;

namespace Ryujinx.Graphics.Shader.StructuredIr
{
    static class AstOptimizer
    {
        public static void Optimize(StructuredProgramContext context)
        {
            AstBlock mainBlock = context.CurrentFunction.MainBlock;

            // When debug mode is enabled, we disable expression propagation
            // (this makes comparison with the disassembly easier).
            if (!context.DebugMode)
            {
                AstBlockVisitor visitor = new(mainBlock);

                foreach (IAstNode node in visitor.Visit())
                {
                    if (node is AstAssignment assignment && assignment.Destination is AstOperand propVar)
                    {
                        // oot2-native: what the expression does decides whether it may move or go. One that is pinned (an atomic, a call, an image write, a store
                        // to memory, a barrier) stays a statement where it is, even when nothing reads its result: moving it reorders the effect, dropping it
                        // loses it. One that reads memory the shader can write, or needs the lanes it had (a shuffle, a vote, a derivative), moves only down its
                        // own block to the next statement that uses it, with no write between (see CanMove).
                        Effects effects = Classify(assignment.Source);

                        bool isWorthPropagating = propVar.Uses.Count == 1 || IsWorthPropagating(assignment.Source);

                        if (propVar.Defs.Count == 1 && isWorthPropagating && CanMove(assignment, propVar, effects))
                        {
                            PropagateExpression(propVar, assignment.Source);
                        }

                        if (propVar.Type == OperandType.LocalVariable && propVar.Uses.Count == 0 && (effects & Effects.Pinned) == 0)
                        {
                            visitor.Block.Remove(assignment);

                            // A pinned assignment to the same variable stays, and so does the variable it writes.
                            if (!HasPinnedDefinition(propVar))
                            {
                                context.CurrentFunction.Locals.Remove(propVar);
                            }
                        }
                    }
                }
            }

            RemoveEmptyBlocks(mainBlock);
        }

        // oot2-native: what an expression does besides computing a value.
        [Flags]
        private enum Effects
        {
            None = 0,
            Pinned = 1,         // changes memory or the program's state, or is ordered against it: an atomic, a call, a store, a barrier, an image write, a discard
            Reads = 2,          // reads memory a store can change: shared, local, storage, global or device memory, an image
            Convergent = 4,     // needs the lanes it was written for: a shuffle, a vote, a ballot, a derivative, a texture sample
        }

        // How many statements a read or a convergent expression may be carried over.
        private const int MoveWindow = 64;

        private static Effects Classify(IAstNode node)
        {
            if (node is not AstOperation op)
            {
                return Effects.None;
            }

            Effects effects = ClassifyOperation(op);

            for (int index = 0; index < op.SourcesCount; index++)
            {
                effects |= Classify(op.GetSource(index));
            }

            return effects;
        }

        private static Effects ClassifyOperation(AstOperation op)
        {
            Instruction inst = op.Inst & Instruction.Mask;

            if (inst.IsAtomic())
            {
                return Effects.Pinned;
            }

            switch (inst)
            {
                case Instruction.Call:
                case Instruction.ImageAtomic:
                case Instruction.ImageStore:
                case Instruction.Barrier:
                case Instruction.GroupMemoryBarrier:
                case Instruction.MemoryBarrier:
                case Instruction.SubgroupBarrier:
                case Instruction.EmitVertex:
                case Instruction.EndPrimitive:
                case Instruction.Discard:
                case Instruction.FSIBegin:
                case Instruction.FSIEnd:
                case Instruction.MatrixMultiplyAdd:
                    return Effects.Pinned;

                case Instruction.Store:
                    // A shader output is never read back through a plain read here: the reads of outputs below are pinned.
                    return op.StorageKind.IsOutput() ? Effects.None : Effects.Pinned;

                case Instruction.ImageLoad:
                    return Effects.Reads;

                case Instruction.Load:
                    // A constant buffer, an input and the push constants are read-only for the whole draw.
                    if (op.StorageKind is StorageKind.None or StorageKind.Input or StorageKind.InputPerPatch or StorageKind.ConstantBuffer or StorageKind.PushConstant)
                    {
                        return Effects.None;
                    }

                    return op.StorageKind.IsOutput() ? Effects.Pinned : Effects.Reads;

                case Instruction.Ballot:
                case Instruction.VoteAll:
                case Instruction.VoteAllEqual:
                case Instruction.VoteAny:
                case Instruction.Shuffle:
                case Instruction.ShuffleDown:
                case Instruction.ShuffleUp:
                case Instruction.ShuffleXor:
                case Instruction.SwizzleAdd:
                case Instruction.Ddx:
                case Instruction.Ddy:
                case Instruction.Lod:
                case Instruction.TextureSample:
                    return Effects.Convergent;
            }

            return Effects.None;
        }

        // Whether `assignment`'s expression may be carried to its only use. A pure one may, as in Ryujinx. A pinned one may not. One that reads or is
        // convergent may when the next statement of its own block that mentions the variable is reached over plain statements only (no nested block, so
        // the lanes and the loop it runs in stay the same), none of which is pinned (a write would change what a read sees, a discard the lanes a
        // derivative needs), and when that statement's operands are not pinned either (they are evaluated first, and the statement's own effect comes
        // after them). The use may be a plain statement or the condition of an `if`, never a loop's condition, which is evaluated again each time round.
        private static bool CanMove(AstAssignment assignment, AstOperand propVar, Effects effects)
        {
            // A call passes its arguments by pointer to a variable (the SPIR-V generator casts each to an operand), so an expression is never folded into one.
            foreach (IAstNode use in propVar.Uses)
            {
                if (use is AstOperation { Inst: var inst } && (inst & Instruction.Mask) == Instruction.Call)
                {
                    return false;
                }
            }

            if (effects == Effects.None)
            {
                return true;
            }

            if ((effects & Effects.Pinned) != 0 || propVar.Uses.Count != 1)
            {
                return false;
            }

            IAstNode node = Next(assignment);

            for (int steps = 0; node != null && steps < MoveWindow; steps++, node = Next(node))
            {
                if (node is AstBlock block)
                {
                    return block.Type == AstBlockType.If && Mentions(block.Condition, propVar) == 1 && (Classify(block.Condition) & Effects.Pinned) == 0;
                }

                if (node is AstComment)
                {
                    continue;
                }

                IAstNode expression = node switch
                {
                    AstAssignment statement => statement.Source,
                    AstOperation statement => statement,
                    _ => null,
                };

                if (expression == null)
                {
                    return false;
                }

                int mentions = Mentions(expression, propVar);

                if (mentions == 0)
                {
                    if ((Classify(expression) & Effects.Pinned) != 0)
                    {
                        return false;
                    }

                    continue;
                }

                // The statement that uses it, once: a variable named twice would make the expression run twice. Its operands, not its own effect, run
                // before the carried expression.
                if (mentions != 1)
                {
                    return false;
                }

                if (expression is not AstOperation use)
                {
                    return true;
                }

                for (int index = 0; index < use.SourcesCount; index++)
                {
                    if ((Classify(use.GetSource(index)) & Effects.Pinned) != 0)
                    {
                        return false;
                    }
                }

                return true;
            }

            return false;
        }

        private static bool HasPinnedDefinition(AstOperand variable)
        {
            foreach (IAstNode def in variable.Defs)
            {
                if (def is AstAssignment assignment && (Classify(assignment.Source) & Effects.Pinned) != 0)
                {
                    return true;
                }
            }

            return false;
        }

        // How many times the operand appears in the expression (a use list names the statement once however many times it is read).
        private static int Mentions(IAstNode node, AstOperand operand)
        {
            if (ReferenceEquals(node, operand))
            {
                return 1;
            }

            int count = 0;

            if (node is AstOperation op)
            {
                for (int index = 0; index < op.SourcesCount; index++)
                {
                    count += Mentions(op.GetSource(index), operand);
                }
            }

            return count;
        }

        private static bool IsWorthPropagating(IAstNode source)
        {
            if (source is not AstOperation srcOp)
            {
                return false;
            }

            if (!InstructionInfo.IsUnary(srcOp.Inst))
            {
                return false;
            }

            return srcOp.GetSource(0) is AstOperand || srcOp.Inst == Instruction.Copy;
        }

        private static void PropagateExpression(AstOperand propVar, IAstNode source)
        {
            IAstNode[] uses = propVar.Uses.ToArray();

            foreach (IAstNode useNode in uses)
            {
                if (useNode is AstBlock useBlock)
                {
                    useBlock.Condition = source;
                }
                else if (useNode is AstOperation useOperation)
                {
                    for (int srcIndex = 0; srcIndex < useOperation.SourcesCount; srcIndex++)
                    {
                        if (useOperation.GetSource(srcIndex) == propVar)
                        {
                            useOperation.SetSource(srcIndex, source);
                        }
                    }
                }
                else if (useNode is AstAssignment useAssignment)
                {
                    useAssignment.Source = source;
                }
            }
        }

        private static void RemoveEmptyBlocks(AstBlock mainBlock)
        {
            Queue<AstBlock> pending = new();

            pending.Enqueue(mainBlock);

            while (pending.TryDequeue(out AstBlock block))
            {
                foreach (IAstNode node in block)
                {
                    if (node is AstBlock childBlock)
                    {
                        pending.Enqueue(childBlock);
                    }
                }

                AstBlock parent = block.Parent;

                if (parent == null)
                {
                    continue;
                }

                AstBlock nextBlock = Next(block) as AstBlock;

                bool hasElse = nextBlock != null && nextBlock.Type == AstBlockType.Else;

                bool isIf = block.Type == AstBlockType.If;

                if (block.Count == 0)
                {
                    if (isIf)
                    {
                        if (hasElse)
                        {
                            nextBlock.TurnIntoIf(InverseCond(block.Condition));
                        }

                        parent.Remove(block);
                    }
                    else if (block.Type == AstBlockType.Else)
                    {
                        parent.Remove(block);
                    }
                }
                else if (isIf && parent.Type == AstBlockType.Else && parent.Count == (hasElse ? 2 : 1))
                {
                    AstBlock parentOfParent = parent.Parent;

                    parent.Remove(block);

                    parentOfParent.AddAfter(parent, block);

                    if (hasElse)
                    {
                        parent.Remove(nextBlock);

                        parentOfParent.AddAfter(block, nextBlock);
                    }

                    parentOfParent.Remove(parent);

                    block.TurnIntoElseIf();
                }
            }
        }
    }
}
