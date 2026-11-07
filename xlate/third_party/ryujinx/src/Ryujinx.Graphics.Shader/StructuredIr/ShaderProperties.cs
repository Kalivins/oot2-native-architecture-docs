using System;
using System.Collections.Generic;

namespace Ryujinx.Graphics.Shader.StructuredIr
{
    class ShaderProperties
    {
        private readonly Dictionary<int, BufferDefinition> _constantBuffers;
        private readonly Dictionary<int, BufferDefinition> _storageBuffers;
        private readonly Dictionary<SetBindingPair, TextureDefinition> _textures;
        private readonly Dictionary<(int Set, int Binding, SamplerType Type), TextureDefinition> _textureVariants;
        private readonly Dictionary<SetBindingPair, TextureDefinition> _images;
        private readonly Dictionary<int, MemoryDefinition> _localMemories;
        private readonly Dictionary<int, MemoryDefinition> _sharedMemories;

        public IReadOnlyDictionary<int, BufferDefinition> ConstantBuffers => _constantBuffers;
        public IReadOnlyDictionary<int, BufferDefinition> StorageBuffers => _storageBuffers;
        public IReadOnlyDictionary<SetBindingPair, TextureDefinition> Textures => _textures;

        // oot2-native: every type an array of textures is used with, by set, binding and type. ResourceManager keys arrays by type, but a host
        // that binds one texture pool for all types (this project's set 0, binding 0) gives every type the same set and binding, and _textures
        // keeps only the last; the SPIR-V generator declares one variable per type here instead (Declarations.DeclareSamplers)
        public IReadOnlyDictionary<(int Set, int Binding, SamplerType Type), TextureDefinition> TextureVariants => _textureVariants;
        public IReadOnlyDictionary<SetBindingPair, TextureDefinition> Images => _images;
        public IReadOnlyDictionary<int, MemoryDefinition> LocalMemories => _localMemories;
        public IReadOnlyDictionary<int, MemoryDefinition> SharedMemories => _sharedMemories;

        // nxnative: a global memory access traced to no storage buffer goes through a physical storage buffer pointer (GlobalToStorage.TryGenerateRawStorageOp)
        public bool UsesPhysicalPointers { get; set; }

        // oot2-native: the module walks the page table of RawTableAbi, read from the first push constant words, and sets store bits when RawStores
        public bool UsesRawTable { get; set; }
        public bool RawStores { get; set; }

        // oot2-native: the module adds the draw's first task to the workgroup number it reads, from the word at the device address in push constant words 6 and 7 (MeshAbi)
        public bool UsesFirstTask { get; set; }

        // oot2-native: textures read through push constant slots (IGpuAccessor.QueryPushTextureLayout), or null
        public PushTextureLayout PushTextures { get; set; }

        public ShaderProperties()
        {
            _constantBuffers = new Dictionary<int, BufferDefinition>();
            _storageBuffers = new Dictionary<int, BufferDefinition>();
            _textures = new Dictionary<SetBindingPair, TextureDefinition>();
            _textureVariants = new Dictionary<(int Set, int Binding, SamplerType Type), TextureDefinition>();
            _images = new Dictionary<SetBindingPair, TextureDefinition>();
            _localMemories = new Dictionary<int, MemoryDefinition>();
            _sharedMemories = new Dictionary<int, MemoryDefinition>();
        }

        public void AddOrUpdateConstantBuffer(BufferDefinition definition)
        {
            _constantBuffers[definition.Binding] = definition;
        }

        public void AddOrUpdateStorageBuffer(BufferDefinition definition)
        {
            _storageBuffers[definition.Binding] = definition;
        }

        public void AddOrUpdateTexture(TextureDefinition definition)
        {
            _textures[new(definition.Set, definition.Binding)] = definition;

            if (definition.ArrayLength != 1)
            {
                _textureVariants[(definition.Set, definition.Binding, definition.Type)] = definition;
            }
        }

        public void AddOrUpdateImage(TextureDefinition definition)
        {
            _images[new(definition.Set, definition.Binding)] = definition;
        }

        public int AddLocalMemory(MemoryDefinition definition)
        {
            int id = _localMemories.Count;
            _localMemories.Add(id, definition);

            return id;
        }

        public int AddSharedMemory(MemoryDefinition definition)
        {
            int id = _sharedMemories.Count;
            _sharedMemories.Add(id, definition);

            return id;
        }

        public static TextureFormat GetTextureFormat(IGpuAccessor gpuAccessor, int handle, int cbufSlot = -1)
        {
            // When the formatted load extension is supported, we don't need to
            // specify a format, we can just declare it without a format and the GPU will handle it.
            if (gpuAccessor.QueryHostSupportsImageLoadFormatted())
            {
                return TextureFormat.Unknown;
            }

            TextureFormat format = gpuAccessor.QueryTextureFormat(handle, cbufSlot);

            if (format == TextureFormat.Unknown)
            {
                gpuAccessor.Log($"Unknown format for texture {handle}.");

                format = TextureFormat.R8G8B8A8Unorm;
            }

            return format;
        }

        private static bool FormatSupportsAtomic(TextureFormat format)
        {
            return format is TextureFormat.R32Sint or TextureFormat.R32Uint;
        }

        public static TextureFormat GetTextureFormatAtomic(IGpuAccessor gpuAccessor, int handle, int cbufSlot = -1, bool strict = false)
        {
            // Atomic image instructions do not support GL_EXT_shader_image_load_formatted,
            // and must have a type specified. Default to R32Sint if not available.

            TextureFormat format = gpuAccessor.QueryTextureFormat(handle, cbufSlot);

            if (!FormatSupportsAtomic(format))
            {
                if (strict)
                {
                    throw new NotSupportedException($"Metal image atomics require an R32 integer descriptor format, got {format}.");
                }

                gpuAccessor.Log($"Unsupported format for texture {handle}: {format}.");

                format = TextureFormat.R32Sint;
            }

            return format;
        }
    }
}
