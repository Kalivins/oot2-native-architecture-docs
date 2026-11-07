using Ryujinx.Graphics.Shader.Translation;

namespace Ryujinx.Graphics.Shader.StructuredIr
{
    readonly struct MemoryDefinition
    {
        public string Name { get; }
        public AggregateType Type { get; }
        public int ArrayLength { get; }

        // oot2-native: declared in the TaskPayloadWorkgroupEXT storage class (a task's payload)
        public bool TaskPayload { get; }

        public MemoryDefinition(string name, AggregateType type, int arrayLength = 1, bool taskPayload = false)
        {
            Name = name;
            Type = type;
            ArrayLength = arrayLength;
            TaskPayload = taskPayload;
        }
    }
}
