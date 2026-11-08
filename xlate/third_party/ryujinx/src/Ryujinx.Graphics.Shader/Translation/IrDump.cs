using Ryujinx.Graphics.Shader.Decoders;
using Ryujinx.Graphics.Shader.IntermediateRepresentation;
using Ryujinx.Graphics.Shader.StructuredIr;
using System;
using System.Collections.Generic;
using System.IO;
using System.Text;

namespace Ryujinx.Graphics.Shader.Translation
{
    // oot2-native: a text dump of the translator's IR for one program, off unless the host turns it on (Oot2.Xlate.Translation, the environment variable
    // OOT2_XLATE_IR_DUMP names the folder). One file per stage and function: after the emitters (emit), after the SSA rename (ssa), after each step of the
    // optimizer and each iteration of its cleanup loop, after each transform pass, the final IR before the structured form (final), the same with the phis
    // lowered to copies (phi-removed), and the structured tree before and after the AST optimizer (ast-pre-opt, ast-final). counts.tsv has, for each stage, how
    // many operations of the kinds a shared memory or tensor loss shows in. Nothing here changes a translation.
    public static class IrDump
    {
        private static string _folder;
        private static int _pass;
        private static int _seq;
        private static int _function;
        private static Dictionary<Operand, int> _ids;
        private static Dictionary<AstOperand, int> _astIds;

        public static bool Enabled => _folder != null;

        // One translation pass of a job (the host translates a job twice when it recovers texture types): files of a pass start with p<pass>.
        public static void Begin(string folder, int pass)
        {
            Directory.CreateDirectory(folder);

            _folder = folder;
            _pass = pass;
            _seq = 0;
            _function = 0;
            _ids = new Dictionary<Operand, int>();
            _astIds = new Dictionary<AstOperand, int>();

            File.WriteAllText(Path.Combine(folder, $"p{pass}.counts.tsv"), "seq\tstage\tfunc\tops\tphis\tshared_load\tshared_store\tbarrier\tshuffle\tmma\tcall\tatomic\n");
        }

        public static void End()
        {
            _folder = null;
            _ids = null;
            _astIds = null;
        }

        internal static void SetFunction(int index)
        {
            _function = index;
        }

        internal static void Write(string stage, BasicBlock[] blocks)
        {
            Write(stage, _function, blocks);
        }

        internal static void Write(string stage, int function, BasicBlock[] blocks)
        {
            if (_folder == null)
            {
                return;
            }

            StringBuilder sb = new();
            Counts counts = new();

            foreach (BasicBlock block in blocks)
            {
                sb.Append($"B{block.Index}:");

                if (block.Predecessors.Count != 0)
                {
                    sb.Append("  preds");

                    foreach (BasicBlock pred in block.Predecessors)
                    {
                        sb.Append($" B{pred.Index}");
                    }
                }

                sb.Append('\n');

                foreach (INode node in block.Operations)
                {
                    sb.Append("  ");
                    Format(sb, node);
                    sb.Append('\n');

                    counts.Add(node);
                }

                if (block.Next != null || block.Branch != null)
                {
                    sb.Append($"  -> next {(block.Next != null ? "B" + block.Next.Index : "-")}, branch {(block.Branch != null ? "B" + block.Branch.Index : "-")}\n");
                }
            }

            int seq = _seq++;

            File.WriteAllText(Path.Combine(_folder, $"p{_pass}.{seq:00}.{stage}.f{function}.txt"), sb.ToString());
            File.AppendAllText(Path.Combine(_folder, $"p{_pass}.counts.tsv"), counts.Row(seq, stage, function));
        }

        internal static void WriteAst(string stage, string function, AstBlock main)
        {
            if (_folder == null)
            {
                return;
            }

            StringBuilder sb = new();
            Counts counts = new();

            _astIds.Clear();
            FormatAst(sb, main, 0, counts);

            int seq = _seq++;

            File.WriteAllText(Path.Combine(_folder, $"p{_pass}.{seq:00}.{stage}.{function}.txt"), sb.ToString());
            File.AppendAllText(Path.Combine(_folder, $"p{_pass}.counts.tsv"), counts.Row(seq, stage, function));
        }

        private sealed class Counts
        {
            private int _ops, _phis, _sharedLoad, _sharedStore, _barrier, _shuffle, _mma, _call, _atomic;

            public void Add(INode node)
            {
                if (node is PhiNode)
                {
                    _phis++;
                }
                else if (node is Operation operation)
                {
                    AddOperation(operation.Inst, operation.StorageKind);
                }
            }

            public void AddOperation(Instruction instruction, StorageKind kind)
            {
                Instruction inst = instruction & Instruction.Mask;

                _ops++;

                bool shared = kind is StorageKind.SharedMemory or StorageKind.SharedMemory8 or StorageKind.SharedMemory16;

                switch (inst)
                {
                    case Instruction.Load when shared:
                        _sharedLoad++;
                        break;
                    case Instruction.Store when shared:
                        _sharedStore++;
                        break;
                    case Instruction.Barrier:
                        _barrier++;
                        break;
                    case Instruction.Shuffle:
                    case Instruction.ShuffleDown:
                    case Instruction.ShuffleUp:
                    case Instruction.ShuffleXor:
                        _shuffle++;
                        break;
                    case Instruction.MatrixMultiplyAdd:
                        _mma++;
                        break;
                    case Instruction.Call:
                        _call++;
                        break;
                    case Instruction.AtomicAdd:
                    case Instruction.AtomicAnd:
                    case Instruction.AtomicCompareAndSwap:
                    case Instruction.AtomicMaxS32:
                    case Instruction.AtomicMaxU32:
                    case Instruction.AtomicMinS32:
                    case Instruction.AtomicMinU32:
                    case Instruction.AtomicOr:
                    case Instruction.AtomicSwap:
                    case Instruction.AtomicXor:
                        _atomic++;
                        break;
                }
            }

            public string Row(int seq, string stage, object function)
            {
                return $"{seq}\t{stage}\t{function}\t{_ops}\t{_phis}\t{_sharedLoad}\t{_sharedStore}\t{_barrier}\t{_shuffle}\t{_mma}\t{_call}\t{_atomic}\n";
            }
        }

        private static string Name(Operand operand)
        {
            if (operand == null)
            {
                return "_";
            }

            switch (operand.Type)
            {
                case OperandType.Constant:
                    return $"0x{operand.Value:x}";
                case OperandType.ConstantBuffer:
                    return $"cb{operand.GetCbufSlot()}[0x{operand.GetCbufOffset():x}]";
                case OperandType.LocalVariable:
                    if (!_ids.TryGetValue(operand, out int id))
                    {
                        id = _ids.Count;
                        _ids.Add(operand, id);
                    }

                    return $"v{id}";
                case OperandType.Register:
                    Register reg = operand.GetRegister();

                    return $"{reg.Type}{reg.Index}";
                case OperandType.Argument:
                    return $"arg{operand.Value}";
                case OperandType.Undefined:
                    return "undef";
                default:
                    return $"{operand.Type}{operand.Value}";
            }
        }

        private static void Format(StringBuilder sb, INode node)
        {
            if (node is PhiNode phi)
            {
                sb.Append($"{Name(phi.Dest)} = phi");

                for (int index = 0; index < phi.SourcesCount; index++)
                {
                    sb.Append($" [B{phi.GetBlock(index).Index}: {Name(phi.GetSource(index))}]");
                }

                return;
            }

            if (node is CommentNode comment)
            {
                sb.Append($"; {comment.Comment}");

                return;
            }

            if (node is not Operation op)
            {
                sb.Append(node.GetType().Name);

                return;
            }

            for (int index = 0; index < op.DestsCount; index++)
            {
                sb.Append(index == 0 ? "" : ", ").Append(Name(op.GetDest(index)));
            }

            if (op.DestsCount != 0)
            {
                sb.Append(" = ");
            }

            sb.Append(op.Inst.ToString().Replace(", ", "|"));

            if (op.StorageKind != StorageKind.None)
            {
                sb.Append('.').Append(op.StorageKind);
            }

            if (op is TextureOperation tex)
            {
                sb.Append($" [{tex.Type} {tex.Flags} set{tex.Set}.{tex.Binding}]");
            }

            if (op.Index != 0)
            {
                sb.Append($" #{op.Index}");
            }

            for (int index = 0; index < op.SourcesCount; index++)
            {
                sb.Append(index == 0 ? " " : ", ").Append(Name(op.GetSource(index)));
            }
        }

        private static string AstName(AstOperand operand)
        {
            switch (operand.Type)
            {
                case OperandType.Constant:
                    return $"0x{operand.Value:x}";
                case OperandType.ConstantBuffer:
                    return $"cb{(operand.Value >> 27) & 31}[0x{operand.Value & 0x7ffffff:x}]";
                case OperandType.LocalVariable:
                    if (!_astIds.TryGetValue(operand, out int id))
                    {
                        id = _astIds.Count;
                        _astIds.Add(operand, id);
                    }

                    return $"v{id}";
                default:
                    return $"{operand.Type}{operand.Value}";
            }
        }

        private static void FormatAstExpr(StringBuilder sb, IAstNode node, Counts counts)
        {
            switch (node)
            {
                case null:
                    sb.Append("null");
                    break;
                case AstOperand operand:
                    sb.Append(AstName(operand));
                    break;
                case AstOperation operation:
                    counts.AddOperation(operation.Inst, operation.StorageKind);

                    sb.Append(operation.Inst.ToString().Replace(", ", "|"));

                    if (operation.StorageKind != StorageKind.None)
                    {
                        sb.Append('.').Append(operation.StorageKind);
                    }

                    sb.Append('(');

                    for (int index = 0; index < operation.SourcesCount; index++)
                    {
                        sb.Append(index == 0 ? "" : ", ");
                        FormatAstExpr(sb, operation.GetSource(index), counts);
                    }

                    sb.Append(')');
                    break;
                default:
                    sb.Append(node.GetType().Name);
                    break;
            }
        }

        private static void FormatAst(StringBuilder sb, AstBlock block, int indent, Counts counts)
        {
            string pad = new(' ', indent * 2);

            foreach (IAstNode node in block)
            {
                switch (node)
                {
                    case AstBlock inner:
                        sb.Append(pad).Append(inner.Type);

                        if (inner.Condition != null)
                        {
                            sb.Append(" (");
                            FormatAstExpr(sb, inner.Condition, counts);
                            sb.Append(')');
                        }

                        sb.Append(" {\n");
                        FormatAst(sb, inner, indent + 1, counts);
                        sb.Append(pad).Append("}\n");
                        break;
                    case AstAssignment assignment:
                        sb.Append(pad);
                        FormatAstExpr(sb, assignment.Destination, counts);
                        sb.Append(" = ");
                        FormatAstExpr(sb, assignment.Source, counts);
                        sb.Append('\n');
                        break;
                    case AstOperation:
                        sb.Append(pad);
                        FormatAstExpr(sb, node, counts);
                        sb.Append('\n');
                        break;
                    case AstComment comment:
                        sb.Append(pad).Append("; ").Append(comment.Comment).Append('\n');
                        break;
                    default:
                        sb.Append(pad).Append(node.GetType().Name).Append('\n');
                        break;
                }
            }
        }
    }
}
