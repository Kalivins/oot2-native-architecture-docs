using Spv.Generator;

namespace Ryujinx.Graphics.Shader.CodeGen.Spirv
{
    readonly struct SamplerDeclaration
    {
        public readonly Instruction ImageType;
        public readonly Instruction SampledImageType;
        public readonly Instruction SampledImagePointerType;
        public readonly Instruction Image;
        public readonly bool IsIndexed;

        // oot2-native: a texture read through a push constant slot (PushTextureLayout): the slot, else -1. Image is then the global
        // image array (a view of this type) or, for a sampler, the global sampler array, and SampledImagePointerType points at one
        // element of it
        public readonly int PushSlot;

        public SamplerDeclaration(
            Instruction imageType,
            Instruction sampledImageType,
            Instruction sampledImagePointerType,
            Instruction image,
            bool isIndexed,
            int pushSlot = -1)
        {
            ImageType = imageType;
            SampledImageType = sampledImageType;
            SampledImagePointerType = sampledImagePointerType;
            Image = image;
            IsIndexed = isIndexed;
            PushSlot = pushSlot;
        }
    }
}
