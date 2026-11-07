// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Text;

namespace Oot2.Xlate;

/// <summary>What the post-pass found in a module and what it changed.</summary>
internal sealed class FixupResult
{
    public byte[] Module { get; init; }
    public List<uint> Capabilities { get; init; }
    public List<uint> Added { get; init; }
    public List<string> Extensions { get; init; }
    public List<string> ExtensionsAdded { get; init; }
    public uint ExecutionModel { get; init; }
    public List<DescriptorAccess> Access { get; init; }
    public List<string> Decorated { get; init; }
}

/// <summary>
/// How a module uses one storage buffer or storage image variable: whether its
/// instructions read or write the memory behind it, and whether the module (after
/// the post-pass) declares it NonWritable or NonReadable.
/// </summary>
internal sealed class DescriptorAccess
{
    public uint Id { get; init; }
    public bool Image { get; init; }
    public int Set { get; init; } = -1;
    public int Binding { get; init; } = -1;
    public bool Reads { get; set; }
    public bool Writes { get; set; }
    public bool NonWritable { get; set; }
    public bool NonReadable { get; set; }
}

/// <summary>
/// The post-pass. The vendored generator declares the capabilities its own output
/// needs for OpenGL-style use and misses some that Vulkan's validation rules ask
/// for. Rather than patch the vendored tree, the host scans the emitted module and
/// inserts the missing OpCapability instructions.
///
/// It declares StorageImageArrayNonUniformIndexing, which the generator no longer
/// does, where an array of storage images is indexed with a non-uniform index.
///
/// It also marks storage buffers and storage images by what the module does with
/// them: NonWritable when no instruction writes the memory behind the variable,
/// NonReadable when none reads it. The generator declares every one writable, and
/// Vulkan accepts a writable storage buffer or image in a vertex shader only with
/// vertexPipelineStoresAndAtomics, and in a fragment shader only with
/// fragmentStoresAndAtomics (VUID-RuntimeSpirv-NonWritable-06341 and -06340);
/// spike S2 met that on its first draw.
///
/// Every enumerant and opcode number below is the value in the Vulkan SDK's
/// spirv.h (include/spirv/unified1); the line of the header is given with each.
/// </summary>
internal static class SpirvFixup
{
    // Instruction opcodes (spirv.h SpvOp...).
    private const uint OpExtension = 10;           // spirv.h:1773
    private const uint OpEntryPoint = 15;          // spirv.h:1777
    private const uint OpCapability = 17;          // spirv.h:1779
    private const uint OpTypeImage = 25;           // spirv.h:1786
    private const uint OpLoad = 61;                // spirv.h:1818
    private const uint OpDecorate = 71;            // spirv.h:1828
    private const uint OpMemberDecorate = 72;      // spirv.h:1829
    private const uint OpImageRead = 98;           // spirv.h:1853
    private const uint OpImageWrite = 99;          // spirv.h:1854
    private const uint OpImageSparseRead = 320;    // spirv.h:2057
    private const uint OpGroupNonUniformShuffleUp = 347;     // spirv.h:2084
    private const uint OpGroupNonUniformShuffleDown = 348;   // spirv.h:2085
    private const uint OpLine = 8;                 // spirv.h:1772
    private const uint OpExtInst = 12;             // spirv.h:1775
    private const uint OpSampledImage = 86;        // spirv.h:1841
    private const uint OpImageDrefGather = 97;     // spirv.h:1852
    private const uint OpImage = 100;              // spirv.h:1855
    private const uint OpImageSparseSampleImplicitLod = 305;  // spirv.h:2042
    private const uint OpVectorShuffle = 79;       // spirv.h:1835
    private const uint OpCompositeExtract = 81;    // spirv.h:1837
    private const uint OpCompositeInsert = 82;     // spirv.h:1838
    private const uint OpLoopMerge = 246;          // spirv.h:1988
    private const uint OpSelectionMerge = 247;     // spirv.h:1989
    private const uint OpBranchConditional = 250;  // spirv.h:1992
    private const uint OpSwitch = 251;             // spirv.h:1993
    private const uint OpImageSparseDrefGather = 315;  // spirv.h:2052
    private const uint OpTypeArray = 28;           // spirv.h:1789
    private const uint OpTypeRuntimeArray = 29;    // spirv.h:1790
    private const uint OpTypePointer = 32;         // spirv.h:1793
    private const uint OpFunction = 54;            // spirv.h:1812
    private const uint OpVariable = 59;            // spirv.h:1816
    private const uint OpImageTexelPointer = 60;   // spirv.h:1817
    private const uint OpStore = 62;               // spirv.h:1819
    private const uint OpCopyMemory = 63;          // spirv.h:1820
    private const uint OpCopyMemorySized = 64;     // spirv.h:1821
    private const uint OpAccessChain = 65;         // spirv.h:1822
    private const uint OpInBoundsAccessChain = 66; // spirv.h:1823
    private const uint OpPtrAccessChain = 67;      // spirv.h:1824
    private const uint OpArrayLength = 68;         // spirv.h:1825
    private const uint OpInBoundsPtrAccessChain = 70;  // spirv.h:1827
    private const uint OpDecorationGroup = 73;     // spirv.h:1830
    private const uint OpGroupMemberDecorate = 75; // spirv.h:1832
    private const uint OpCopyObject = 83;          // spirv.h:1839
    private const uint OpImageQueryFormat = 101;   // spirv.h:1856
    private const uint OpImageQuerySamples = 107;  // spirv.h:1862
    private const uint OpAtomicLoad = 227;         // spirv.h:1971
    private const uint OpAtomicStore = 228;        // spirv.h:1972
    private const uint OpAtomicExchange = 229;     // spirv.h:1973
    private const uint OpAtomicXor = 242;          // spirv.h:1986
    private const uint OpAtomicFlagTestAndSet = 318;   // spirv.h:2055
    private const uint OpAtomicFlagClear = 319;    // spirv.h:2056
    private const uint OpDecorateId = 332;         // spirv.h:2069
    private const uint OpDecorateString = 5632;    // spirv.h:2399
    private const uint OpMemberDecorateString = 5633;  // spirv.h:2401
    private const uint OpAtomicFMinEXT = 5614;     // spirv.h:2395
    private const uint OpAtomicFMaxEXT = 5615;     // spirv.h:2396
    private const uint OpAtomicFAddEXT = 6035;     // spirv.h:2636

    // Capabilities (spirv.h SpvCapability...).
    public const uint CapStorageImageMultisample = 27;           // spirv.h:1104
    public const uint CapClipDistance = 32;                      // spirv.h:1109
    public const uint CapCullDistance = 33;                      // spirv.h:1110
    public const uint CapImageCubeArray = 34;                    // spirv.h:1111
    public const uint CapSampled1D = 43;                         // spirv.h:1120
    public const uint CapImage1D = 44;                           // spirv.h:1121
    public const uint CapSampledCubeArray = 45;                  // spirv.h:1122
    public const uint CapSampledBuffer = 46;                     // spirv.h:1123
    public const uint CapImageBuffer = 47;                       // spirv.h:1124
    public const uint CapImageMSArray = 48;                      // spirv.h:1125
    public const uint CapStorageImageReadWithoutFormat = 55;     // spirv.h:1132
    public const uint CapStorageImageWriteWithoutFormat = 56;    // spirv.h:1133
    public const uint CapGroupNonUniformShuffleRelative = 66;    // spirv.h:1143
    public const uint CapStorageImageArrayNonUniformIndexing = 5309;  // spirv.h:1260
    public const uint CapPhysicalStorageBufferAddresses = 5347;  // spirv.h:1275

    // Operand values.
    private const uint DecorationBlock = 2;        // spirv.h:509
    private const uint DecorationBufferBlock = 3;  // spirv.h:510
    public const uint DecorationNonWritable = 24;  // spirv.h:530
    public const uint DecorationNonReadable = 25;  // spirv.h:531
    private const uint DecorationBinding = 33;     // spirv.h:539
    private const uint DecorationDescriptorSet = 34;   // spirv.h:540
    private const uint StorageClassUniformConstant = 0;  // spirv.h:225
    private const uint StorageClassUniform = 2;    // spirv.h:227
    private const uint StorageClassStorageBuffer = 12;   // spirv.h:237
    private const uint DecorationBuiltIn = 11;     // spirv.h:518
    private const uint DecorationNonUniform = 5300;    // spirv.h:582
    private const uint BuiltInClipDistance = 3;    // spirv.h:712
    private const uint BuiltInCullDistance = 4;    // spirv.h:713
    private const uint Dim1D = 0;                  // spirv.h:267
    private const uint DimCube = 3;                // spirv.h:270
    private const uint DimBuffer = 5;              // spirv.h:272
    private const uint DimSubpassData = 6;
    private const uint ImageFormatUnknown = 0;     // spirv.h:294

    private const uint Magic = 0x07230203;         // spirv.h:45
    private const int HeaderWords = 5;

    public static string CapabilityName(uint capability)
    {
        string name = Enum.GetName((Spv.Specification.Capability)capability);

        return name ?? $"Capability{capability}";
    }

    public static string ExecutionModelName(uint model)
    {
        return model switch
        {
            0 => "Vertex",
            1 => "TessellationControl",
            2 => "TessellationEvaluation",
            3 => "Geometry",
            4 => "Fragment",
            5 => "GLCompute",
            6 => "Kernel",
            5364 => "TaskEXT",
            5365 => "MeshEXT",
            _ => $"ExecutionModel{model}",
        };
    }

    /// <summary>
    /// Scan <paramref name="spirv"/>; with <paramref name="fix"/> also return a
    /// module with the missing capabilities declared. A malformed module throws.
    /// </summary>
    public static FixupResult Apply(byte[] spirv, bool fix)
    {
        if (spirv == null || spirv.Length < HeaderWords * 4 || spirv.Length % 4 != 0)
        {
            throw new InvalidOperationException("the module is not a whole number of words or has no header");
        }

        ReadOnlySpan<uint> w = MemoryMarshal.Cast<byte, uint>(spirv);

        if (w[0] != Magic)
        {
            throw new InvalidOperationException("the module does not start with the SPIR-V magic number");
        }

        SortedSet<uint> declared = new();
        SortedSet<uint> needed = new();
        List<string> extensions = new();
        Dictionary<uint, uint> imageSampled = new();   // OpTypeImage id -> sampled operand when the format is Unknown
        Dictionary<uint, uint> loadType = new();       // OpLoad result id -> its type id
        StorageImageIndexing indexing = new();
        int capabilityEnd = HeaderWords;
        uint model = uint.MaxValue;

        int i = HeaderWords;

        while (i < w.Length)
        {
            int count = (int)(w[i] >> 16);
            uint op = w[i] & 0xffff;

            if (count == 0 || i + count > w.Length)
            {
                throw new InvalidOperationException($"instruction at word {i} has a bad length {count}");
            }

            indexing.Note(op, w.Slice(i, count));

            switch (op)
            {
                case OpCapability when count == 2:
                    declared.Add(w[i + 1]);
                    capabilityEnd = i + count;
                    break;
                case OpExtension:
                    extensions.Add(ReadString(w.Slice(i + 1, count - 1)));
                    break;
                case OpEntryPoint when count >= 2 && model == uint.MaxValue:
                    model = w[i + 1];
                    break;
                case OpTypeImage when count >= 9:
                    NoteImageType(w.Slice(i, count), needed, imageSampled);
                    break;
                case OpLoad when count >= 4:
                    loadType[w[i + 2]] = w[i + 1];
                    break;
                case OpImageRead when count >= 5:
                    if (ImageIsUnformatted(w[i + 3], loadType, imageSampled))
                    {
                        needed.Add(CapStorageImageReadWithoutFormat);
                    }

                    break;
                case OpImageSparseRead when count >= 5:
                    if (ImageIsUnformatted(w[i + 3], loadType, imageSampled))
                    {
                        needed.Add(CapStorageImageReadWithoutFormat);
                    }

                    break;
                case OpImageWrite when count >= 4:
                    if (ImageIsUnformatted(w[i + 1], loadType, imageSampled))
                    {
                        needed.Add(CapStorageImageWriteWithoutFormat);
                    }

                    break;
                case OpDecorate when count >= 4 && w[i + 2] == DecorationBuiltIn:
                    NoteBuiltIn(w[i + 3], needed);
                    break;
                case OpMemberDecorate when count >= 5 && w[i + 3] == DecorationBuiltIn:
                    NoteBuiltIn(w[i + 4], needed);
                    break;
                case OpGroupNonUniformShuffleUp or OpGroupNonUniformShuffleDown:
                    needed.Add(CapGroupNonUniformShuffleRelative);
                    break;
            }

            i += count;
        }

        if (indexing.Found)
        {
            needed.Add(CapStorageImageArrayNonUniformIndexing);
        }

        List<uint> added = new();

        foreach (uint cap in needed)
        {
            if (!declared.Contains(cap))
            {
                added.Add(cap);
            }
        }

        AccessScan scan = ScanAccess(w);
        List<(uint Id, uint Decoration)> decorations = new();
        List<string> decorated = new();

        foreach (DescriptorAccess a in scan.Access)
        {
            if (!a.Writes && !a.NonWritable)
            {
                decorations.Add((a.Id, DecorationNonWritable));
            }

            if (!a.Reads && !a.NonReadable)
            {
                decorations.Add((a.Id, DecorationNonReadable));
            }
        }

        byte[] module = spirv;
        SortedSet<uint> finalCaps = new(declared);

        if (fix && (added.Count > 0 || decorations.Count > 0))
        {
            // Capabilities go after the last OpCapability, decorations after the
            // last annotation (or before the first type when there is none); the
            // annotation section always follows the capabilities.
            int decorateAt = Math.Max(scan.DecorateAt, capabilityEnd);
            uint[] words = new uint[w.Length + added.Count * 2 + decorations.Count * 3];
            w.Slice(0, capabilityEnd).CopyTo(words);

            int o = capabilityEnd;

            foreach (uint cap in added)
            {
                words[o++] = (2u << 16) | OpCapability;
                words[o++] = cap;
                finalCaps.Add(cap);
            }

            w.Slice(capabilityEnd, decorateAt - capabilityEnd).CopyTo(words.AsSpan(o));
            o += decorateAt - capabilityEnd;

            foreach ((uint id, uint decoration) in decorations)
            {
                words[o++] = (3u << 16) | OpDecorate;
                words[o++] = id;
                words[o++] = decoration;
            }

            w.Slice(decorateAt).CopyTo(words.AsSpan(o));
            module = new byte[words.Length * 4];
            Buffer.BlockCopy(words, 0, module, 0, module.Length);

            foreach (DescriptorAccess a in scan.Access)
            {
                if (!a.Writes && !a.NonWritable)
                {
                    decorated.Add($"NonWritable@{a.Set}.{a.Binding}");
                    a.NonWritable = true;
                }

                if (!a.Reads && !a.NonReadable)
                {
                    decorated.Add($"NonReadable@{a.Set}.{a.Binding}");
                    a.NonReadable = true;
                }
            }
        }
        else
        {
            added.Clear();
        }

        return new FixupResult
        {
            Module = module,
            Capabilities = new List<uint>(finalCaps),
            Added = added,
            Extensions = extensions,
            ExtensionsAdded = new List<string>(),
            ExecutionModel = model,
            Access = scan.Access,
            Decorated = decorated,
        };
    }

    /// <summary>
    /// Whether an array of storage images is indexed with a non-uniform index: an access chain whose base is a
    /// variable of an array of images with Sampled 2 (a runtime array included; an array of arrays of images is not
    /// a valid UniformConstant variable) and whose result or one of whose index operands is decorated NonUniform,
    /// or a load of such a chain's result that is decorated NonUniform. The vendored generator marks every indexed
    /// access NonUniform, since it cannot know that an index is uniform, so an indexed access is what the rule
    /// finds. Arrays of sampled images and of
    /// storage buffers need other capabilities (SampledImageArrayNonUniformIndexing, which the generator declares)
    /// and do not count. Annotations come before types and types before the code, so one pass in the order of the
    /// module sees each definition before its use.
    /// </summary>
    private sealed class StorageImageIndexing
    {
        private readonly HashSet<uint> _nonUniform = new();
        private readonly HashSet<uint> _storageImages = new();       // OpTypeImage ids with Sampled 2
        private readonly HashSet<uint> _arrays = new();              // array types whose element is one
        private readonly HashSet<uint> _arrayPointers = new();       // pointer types to those arrays
        private readonly HashSet<uint> _arrayVariables = new();
        private readonly HashSet<uint> _chains = new();              // access chain results that index such a variable

        public bool Found { get; private set; }

        public void Note(uint op, ReadOnlySpan<uint> inst)
        {
            switch (op)
            {
                case OpDecorate when inst.Length >= 3 && inst[2] == DecorationNonUniform:
                    _nonUniform.Add(inst[1]);
                    break;
                case OpTypeImage when inst.Length >= 9:
                    if (inst[7] == 2 && inst[3] != DimSubpassData)
                    {
                        _storageImages.Add(inst[1]);
                    }

                    break;
                case OpTypeArray when inst.Length >= 4:
                case OpTypeRuntimeArray when inst.Length >= 3:
                    if (_storageImages.Contains(inst[2]))
                    {
                        _arrays.Add(inst[1]);
                    }

                    break;
                case OpTypePointer when inst.Length >= 4:
                    if (_arrays.Contains(inst[3]))
                    {
                        _arrayPointers.Add(inst[1]);
                    }

                    break;
                case OpVariable when inst.Length >= 4:
                    if (_arrayPointers.Contains(inst[1]))
                    {
                        _arrayVariables.Add(inst[2]);
                    }

                    break;
                case OpAccessChain or OpInBoundsAccessChain or OpPtrAccessChain or OpInBoundsPtrAccessChain when inst.Length >= 5:
                    if (_arrayVariables.Contains(inst[3]))
                    {
                        _chains.Add(inst[2]);

                        bool nonUniform = _nonUniform.Contains(inst[2]);

                        for (int k = 4; k < inst.Length; k++)
                        {
                            nonUniform |= _nonUniform.Contains(inst[k]);
                        }

                        Found |= nonUniform;
                    }

                    break;
                case OpLoad when inst.Length >= 4:
                    if (_chains.Contains(inst[3]) && _nonUniform.Contains(inst[2]))
                    {
                        Found = true;
                    }

                    break;
            }
        }
    }

    private sealed class AccessScan
    {
        public List<DescriptorAccess> Access { get; } = new();
        public int DecorateAt { get; set; }
    }

    /// <summary>
    /// Which storage buffers and storage images the module reads and writes.
    /// A storage buffer is a Uniform variable of a BufferBlock structure (what the
    /// vendored generator emits) or a StorageBuffer variable of a Block structure;
    /// a storage image is a UniformConstant variable of an image with Sampled 2;
    /// either may be an array of them. Pointers derived from a variable by access
    /// chains and copies, and images loaded from it, count as the variable. Loads,
    /// stores, copies, atomics and image reads and writes are classified; any other
    /// instruction that takes such a pointer or image counts as both a read and a
    /// write, so that nothing is ever marked on a guess.
    /// </summary>
    private static AccessScan ScanAccess(ReadOnlySpan<uint> w)
    {
        AccessScan result = new();
        Dictionary<uint, (uint StorageClass, uint Pointee)> pointers = new();
        Dictionary<uint, uint> arrayElement = new();
        Dictionary<uint, uint> imageSampled = new();       // OpTypeImage id -> Sampled operand (Dim SubpassData left out)
        HashSet<uint> blockStructs = new();
        HashSet<uint> bufferBlockStructs = new();
        Dictionary<uint, int> sets = new();
        Dictionary<uint, int> bindings = new();
        HashSet<uint> nonWritable = new();
        HashSet<uint> nonReadable = new();
        List<(uint Id, uint Type)> globals = new();
        int annotationEnd = -1;
        int firstType = -1;
        int functions = w.Length;

        int i = HeaderWords;

        while (i < w.Length)
        {
            int count = (int)(w[i] >> 16);
            uint op = w[i] & 0xffff;

            if (op == OpFunction)
            {
                functions = i;
                break;
            }

            switch (op)
            {
                case OpDecorate when count >= 3:
                    uint target = w[i + 1];

                    switch (w[i + 2])
                    {
                        case DecorationBlock: blockStructs.Add(target); break;
                        case DecorationBufferBlock: bufferBlockStructs.Add(target); break;
                        case DecorationNonWritable: nonWritable.Add(target); break;
                        case DecorationNonReadable: nonReadable.Add(target); break;
                        case DecorationDescriptorSet when count >= 4: sets[target] = (int)w[i + 3]; break;
                        case DecorationBinding when count >= 4: bindings[target] = (int)w[i + 3]; break;
                    }

                    break;
                case OpTypePointer when count >= 4:
                    pointers[w[i + 1]] = (w[i + 2], w[i + 3]);
                    break;
                case OpTypeArray when count >= 4:
                case OpTypeRuntimeArray when count >= 3:
                    arrayElement[w[i + 1]] = w[i + 2];
                    break;
                case OpTypeImage when count >= 9:
                    if (w[i + 3] != DimSubpassData)
                    {
                        imageSampled[w[i + 1]] = w[i + 7];
                    }

                    break;
                case OpVariable when count >= 4:
                    globals.Add((w[i + 2], w[i + 1]));
                    break;
            }

            if (op is OpDecorate or OpMemberDecorate or (>= OpDecorationGroup and <= OpGroupMemberDecorate)
                or OpDecorateId or OpDecorateString or OpMemberDecorateString)
            {
                annotationEnd = i + count;
            }
            else if (firstType < 0 && op is >= 19 and <= 39)   // OpTypeVoid .. OpTypeForwardPointer (spirv.h:1780-1800)
            {
                firstType = i;
            }

            i += count;
        }

        result.DecorateAt = annotationEnd >= 0 ? annotationEnd : firstType >= 0 ? firstType : functions;

        // The variables, and what each derived id stands for.
        Dictionary<uint, DescriptorAccess> roots = new();      // pointer id -> its variable
        Dictionary<uint, DescriptorAccess> images = new();     // loaded image id -> its variable

        foreach ((uint id, uint type) in globals)
        {
            if (!pointers.TryGetValue(type, out var p))
            {
                continue;
            }

            uint t = p.Pointee;

            while (arrayElement.TryGetValue(t, out uint element))
            {
                t = element;
            }

            bool buffer = (p.StorageClass == StorageClassUniform && bufferBlockStructs.Contains(t))
                || (p.StorageClass == StorageClassStorageBuffer && blockStructs.Contains(t));
            bool image = p.StorageClass == StorageClassUniformConstant && imageSampled.TryGetValue(t, out uint sampled) && sampled == 2;

            if (!buffer && !image)
            {
                continue;
            }

            DescriptorAccess a = new()
            {
                Id = id,
                Image = image,
                Set = sets.TryGetValue(id, out int set) ? set : -1,
                Binding = bindings.TryGetValue(id, out int binding) ? binding : -1,
                NonWritable = nonWritable.Contains(id),
                NonReadable = nonReadable.Contains(id),
            };

            result.Access.Add(a);
            roots[id] = a;
        }

        if (roots.Count == 0)
        {
            return result;
        }

        // Derivations: definitions come before their uses in a function's block
        // order, so one pass finds every access chain, copy and image load.
        for (i = functions; i < w.Length; i += (int)(w[i] >> 16))
        {
            int count = (int)(w[i] >> 16);
            uint op = w[i] & 0xffff;

            switch (op)
            {
                case OpAccessChain or OpInBoundsAccessChain or OpPtrAccessChain or OpInBoundsPtrAccessChain when count >= 4:
                    if (roots.TryGetValue(w[i + 3], out DescriptorAccess chained))
                    {
                        roots[w[i + 2]] = chained;
                    }

                    break;
                case OpCopyObject when count >= 4:
                    if (roots.TryGetValue(w[i + 3], out DescriptorAccess copied))
                    {
                        roots[w[i + 2]] = copied;
                    }
                    else if (images.TryGetValue(w[i + 3], out DescriptorAccess copiedImage))
                    {
                        images[w[i + 2]] = copiedImage;
                    }

                    break;
                case OpLoad when count >= 4:
                    if (roots.TryGetValue(w[i + 3], out DescriptorAccess loaded) && loaded.Image)
                    {
                        images[w[i + 2]] = loaded;
                    }

                    break;
            }
        }

        // Uses.
        for (i = functions; i < w.Length; i += (int)(w[i] >> 16))
        {
            int count = (int)(w[i] >> 16);
            uint op = w[i] & 0xffff;

            switch (op)
            {
                case OpAccessChain or OpInBoundsAccessChain or OpPtrAccessChain or OpInBoundsPtrAccessChain or OpCopyObject:
                case OpArrayLength:
                case >= OpImageQueryFormat and <= OpImageQuerySamples:
                    break;   // derivations and queries: no access to the memory
                case OpLoad when count >= 4:
                    if (roots.TryGetValue(w[i + 3], out DescriptorAccess read) && !read.Image)
                    {
                        read.Reads = true;
                    }

                    break;
                case OpStore when count >= 3:
                    Mark(roots, w[i + 1], reads: false, writes: true);
                    Escape(roots, images, w[i + 2]);
                    break;
                case OpCopyMemory or OpCopyMemorySized when count >= 3:
                    Mark(roots, w[i + 1], reads: false, writes: true);
                    Mark(roots, w[i + 2], reads: true, writes: false);
                    break;
                case OpImageTexelPointer when count >= 4:
                    Mark(roots, w[i + 3], reads: true, writes: true);
                    break;
                case OpImageRead or OpImageSparseRead when count >= 4:
                    Mark(images, w[i + 3], reads: true, writes: false);
                    break;
                case OpImageWrite when count >= 2:
                    Mark(images, w[i + 1], reads: false, writes: true);
                    break;
                case OpAtomicLoad when count >= 4:
                    Mark(roots, w[i + 3], reads: true, writes: false);
                    break;
                case OpAtomicStore or OpAtomicFlagClear when count >= 2:
                    Mark(roots, w[i + 1], reads: false, writes: true);
                    break;
                case (>= OpAtomicExchange and <= OpAtomicXor) or OpAtomicFlagTestAndSet
                    or OpAtomicFMinEXT or OpAtomicFMaxEXT or OpAtomicFAddEXT when count >= 4:
                    Mark(roots, w[i + 3], reads: true, writes: true);
                    break;
                default:
                    // Only id operands: a literal (an extended instruction number, a component index) can
                    // equal a variable's id by chance.
                    (int first, int end) = IdOperands(op, count);

                    for (int k = first; k < end; k++)
                    {
                        Escape(roots, images, w[i + k]);
                    }

                    break;
            }
        }

        // Variables at one set and binding alias the same memory (the generator's float view of a storage
        // buffer for OpAtomicFAddEXT is one): a write through any of them is a write through all.
        foreach (DescriptorAccess a in result.Access)
        {
            foreach (DescriptorAccess b in result.Access)
            {
                if (a != b && a.Set >= 0 && a.Binding >= 0 && a.Set == b.Set && a.Binding == b.Binding && a.Image == b.Image)
                {
                    a.Reads |= b.Reads;
                    a.Writes |= b.Writes;
                }
            }
        }

        return result;
    }

    // The words of an instruction (from 1) that may hold ids. Instructions with literal operands keep their
    // ids only; image sampling takes sampled images, never a storage buffer or storage image, and debug lines
    // and function variables take none of them.
    private static (int First, int End) IdOperands(uint op, int count)
    {
        return op switch
        {
            OpLine or OpVariable => (0, 0),
            OpExtInst => (5, count),                            // type, result, set, instruction, operands
            OpCompositeExtract => (3, Math.Min(4, count)),      // type, result, composite, indexes
            OpCompositeInsert or OpVectorShuffle => (3, Math.Min(5, count)),
            OpSwitch => (1, Math.Min(3, count)),                // selector, default, literal and label pairs
            OpBranchConditional => (1, Math.Min(4, count)),     // condition, two labels, weights
            OpLoopMerge => (1, Math.Min(3, count)),             // two labels, loop control and its parameters
            OpSelectionMerge => (1, Math.Min(2, count)),
            (>= OpSampledImage and <= OpImageDrefGather) or OpImage or (>= OpImageSparseSampleImplicitLod and <= OpImageSparseDrefGather) => (0, 0),
            _ => (1, count),
        };
    }

    private static void Mark(Dictionary<uint, DescriptorAccess> ids, uint id, bool reads, bool writes)
    {
        if (ids.TryGetValue(id, out DescriptorAccess a))
        {
            a.Reads |= reads;
            a.Writes |= writes;
        }
    }

    // A pointer or image used where the scan cannot follow it: assume both.
    private static void Escape(Dictionary<uint, DescriptorAccess> roots, Dictionary<uint, DescriptorAccess> images, uint id)
    {
        Mark(roots, id, reads: true, writes: true);
        Mark(images, id, reads: true, writes: true);
    }

    // OpTypeImage: result, sampled type, Dim, Depth, Arrayed, MS, Sampled, Format.
    private static void NoteImageType(ReadOnlySpan<uint> inst, SortedSet<uint> needed, Dictionary<uint, uint> unformatted)
    {
        uint id = inst[1], dim = inst[3], arrayed = inst[5], ms = inst[6], sampled = inst[7], format = inst[8];

        if (sampled == 2 && format == ImageFormatUnknown && dim != DimSubpassData)
        {
            unformatted[id] = sampled;
        }

        if (dim == Dim1D)
        {
            if (sampled == 1)
            {
                needed.Add(CapSampled1D);
            }
            else if (sampled == 2)
            {
                needed.Add(CapImage1D);
            }
        }
        else if (dim == DimCube && arrayed == 1)
        {
            if (sampled == 1)
            {
                needed.Add(CapSampledCubeArray);
            }
            else if (sampled == 2)
            {
                needed.Add(CapImageCubeArray);
            }
        }
        else if (dim == DimBuffer)
        {
            if (sampled == 1)
            {
                needed.Add(CapSampledBuffer);
            }
            else if (sampled == 2)
            {
                needed.Add(CapImageBuffer);
            }
        }

        if (ms == 1 && sampled == 2)
        {
            needed.Add(CapStorageImageMultisample);

            if (arrayed == 1)
            {
                needed.Add(CapImageMSArray);
            }
        }
    }

    private static void NoteBuiltIn(uint builtIn, SortedSet<uint> needed)
    {
        if (builtIn == BuiltInClipDistance)
        {
            needed.Add(CapClipDistance);
        }
        else if (builtIn == BuiltInCullDistance)
        {
            needed.Add(CapCullDistance);
        }
    }

    private static bool ImageIsUnformatted(uint imageId, Dictionary<uint, uint> loadType, Dictionary<uint, uint> unformatted)
    {
        return loadType.TryGetValue(imageId, out uint type) && unformatted.ContainsKey(type);
    }

    private static string ReadString(ReadOnlySpan<uint> words)
    {
        byte[] bytes = new byte[words.Length * 4];
        MemoryMarshal.Cast<uint, byte>(words).CopyTo(bytes);
        int n = Array.IndexOf(bytes, (byte)0);

        return Encoding.UTF8.GetString(bytes, 0, n < 0 ? bytes.Length : n);
    }
}
