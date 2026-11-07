// SPDX-License-Identifier: GPL-3.0-or-later
// Copyright (C) 2026 The oot2-native contributors

using Ryujinx.Graphics.Shader;
using System;
using System.Diagnostics;
using System.IO;
using System.Text;
using System.Text.Json;

namespace Oot2.Xlate;

/// <summary>
/// The request/reply layer of "--serve": one JSON request per line in, one text
/// reply per line out (job contract v1, tools/shaders/pool.py).
///
///   ok &lt;key&gt; &lt;micros&gt; &lt;spv_bytes&gt;
///   fail &lt;key&gt; &lt;micros&gt; &lt;kind&gt;: &lt;message&gt;
///
/// The structure of the loop follows the --serve mode of the reference
/// translators (DKB tools/xlate, MIT), with the replies of this project's contract.
/// </summary>
internal sealed class Host
{
    public const int Protocol = 1;
    private const int MaxMessage = 400;

    private readonly string _translatorId;
    private bool _cold = true;

    public Host(string translatorId)
    {
        _translatorId = translatorId;
    }

    /// <summary>True when the request asks the loop to end.</summary>
    public bool Quit { get; private set; }

    /// <summary>The reply line for one request line, without its line break. Null for a blank line.</summary>
    public string Handle(string line)
    {
        if (string.IsNullOrWhiteSpace(line))
        {
            return null;
        }

        string key = "-";
        TranslatorTiming translatorTiming = new();

        try
        {
            using JsonDocument doc = JsonDocument.Parse(line);
            JsonElement root = doc.RootElement;

            if (root.ValueKind != JsonValueKind.Object)
            {
                throw new JobException(Kinds.BadJob, "request is not a JSON object");
            }

            string op = "translate";
            JsonElement jobElement = default;
            string outDir = null;
            bool hasJob = false;

            foreach (JsonProperty p in root.EnumerateObject())
            {
                switch (p.Name)
                {
                    case "op":
                        if (p.Value.ValueKind != JsonValueKind.String)
                        {
                            throw new JobException(Kinds.BadJob, "op is not a string");
                        }

                        op = p.Value.GetString();
                        break;
                    case "job":
                        jobElement = p.Value;
                        hasJob = true;
                        break;
                    case "out":
                        if (p.Value.ValueKind != JsonValueKind.String || p.Value.GetString().Length == 0)
                        {
                            throw new JobException(Kinds.BadJob, "out is not a folder path");
                        }

                        outDir = p.Value.GetString();
                        break;
                    default:
                        throw new JobException(Kinds.BadJob, $"unknown request field '{Clip(p.Name)}'");
                }
            }

            switch (op)
            {
                case "hello":
                    return $"ok hello 0 {HelloJson()}";
                case "quit":
                    Quit = true;

                    return "ok quit 0 bye";
                case "translate":
                    break;
                default:
                    throw new JobException(Kinds.BadJob, $"unknown op '{Clip(op)}'");
            }

            if (!hasJob || outDir == null)
            {
                throw new JobException(Kinds.BadJob, "a translate request needs 'job' and 'out'");
            }

            key = Job.PeekKey(jobElement) ?? "-";

            return Translate(jobElement, outDir, ref key, translatorTiming);
        }
        catch (JsonException e)
        {
            return Fail(key, translatorTiming, Kinds.BadJob, "request is not valid JSON: " + FirstLine(e.Message));
        }
        catch (JobException e)
        {
            return Fail(key, translatorTiming, e.Kind, e.Message);
        }
        catch (IOException e)
        {
            return Fail(key, translatorTiming, Kinds.Io, $"{e.GetType().Name}: {e.Message}");
        }
        catch (UnauthorizedAccessException e)
        {
            return Fail(key, translatorTiming, Kinds.Io, e.Message);
        }
        catch (Exception e)
        {
            // Anything the translation phases did not classify themselves.
            return Fail(key, translatorTiming, Kinds.Translate, $"{e.GetType().Name}: {e.Message}");
        }
    }

    private string Translate(JsonElement jobElement, string outDir, ref string key, TranslatorTiming translatorTiming)
    {
        long wall0 = Stopwatch.GetTimestamp();
        long alloc0 = GC.GetAllocatedBytesForCurrentThread();
        int gc0 = GC.CollectionCount(0), gc1 = GC.CollectionCount(1), gc2 = GC.CollectionCount(2);
        bool cold = _cold;
        _cold = false;

        Job job = Job.Parse(jobElement);
        key = job.Key;

        // A task or mesh program past the device's limits, or with shared memory of its own, is refused by name before any work
        MeshChecks.Enforce(job);

        byte[] code = ReadCode(job);
        TranslationOutcome outcome = Translation.Run(job, code, translatorTiming);
        long p0 = Stopwatch.GetTimestamp();
        FixupResult fix;

        try
        {
            fix = SpirvFixup.Apply(outcome.Spirv, job.Options.CapabilityFixup);
        }
        catch (Exception e)
        {
            throw new JobException(Kinds.Validate, $"capability post-pass: {e.Message}");
        }

        byte[] spirv = fix.Module;
        long postTicks = Stopwatch.GetTimestamp() - p0;
        NoteAccessDisagreements(outcome, fix);

        JobTiming timing = new()
        {
            TranslatorUs = translatorTiming.Micros,
            DecodeUs = TranslatorTiming.ToMicros(translatorTiming.DecodeTicks),
            TranslateUs = TranslatorTiming.ToMicros(translatorTiming.TranslateTicks),
            PostUs = TranslatorTiming.ToMicros(postTicks),
            TotalUs = TranslatorTiming.ToMicros(Stopwatch.GetTimestamp() - wall0),
            AllocBytes = GC.GetAllocatedBytesForCurrentThread() - alloc0,
            Gc = new[] { GC.CollectionCount(0) - gc0, GC.CollectionCount(1) - gc1, GC.CollectionCount(2) - gc2 },
            Cold = cold,
        };

        byte[] resJson = ResultWriter.BuildResJson(job, outcome, fix, spirv, _translatorId, timing);
        ResultWriter.WriteFiles(outDir, job.Key, spirv, resJson);

        return $"ok {job.Key} {timing.TranslatorUs} {spirv.Length}";
    }

    // The post-pass decides NonWritable and NonReadable from the module's own
    // instructions; the translator's usage flags say which buffers and images it
    // meant to write. A disagreement is recorded, never silently resolved.
    private static void NoteAccessDisagreements(TranslationOutcome outcome, FixupResult fix)
    {
        foreach (BufferDescriptor b in outcome.Info.SBuffers)
        {
            NoteAccess(outcome.Accessor, "storage buffer", ResultWriter.Find(fix, b.Set, b.Binding, image: false),
                b.Set, b.Binding, (b.Flags & BufferUsageFlags.Write) != 0);
        }

        foreach (TextureDescriptor t in outcome.Info.Images)
        {
            NoteAccess(outcome.Accessor, "storage image", ResultWriter.Find(fix, t.Set, t.Binding, image: true),
                t.Set, t.Binding, (t.Flags & TextureUsageFlags.ImageStore) != 0);
        }
    }

    private static void NoteAccess(Accessor accessor, string what, DescriptorAccess access, int set, int binding, bool flaggedWrite)
    {
        if (access == null)
        {
            accessor.AddNote($"{what} at set {set} binding {binding} is not declared in the module");
        }
        else if (access.Writes != flaggedWrite)
        {
            accessor.AddNote($"{what} at set {set} binding {binding}: the module {(access.Writes ? "writes" : "never writes")} it, "
                + $"the translator's usage flags say {(flaggedWrite ? "written" : "read only")}");
        }
    }

    private static byte[] ReadCode(Job job)
    {
        if (!File.Exists(job.CodePath))
        {
            throw new JobException(Kinds.Io, $"code file not found: {Clip(job.CodePath)}");
        }

        long fileLength = new FileInfo(job.CodePath).Length;
        long length = job.CodeLength ?? fileLength - job.CodeOffset;

        if (length <= 0 || length % 16 != 0)
        {
            throw new JobException(Kinds.BadJob, $"the code slice is {length} bytes, not a positive multiple of 16");
        }

        if (job.CodeOffset + length > fileLength)
        {
            throw new JobException(Kinds.Io, $"the code slice {job.CodeOffset}+{length} is past the end of the file ({fileLength} bytes)");
        }

        byte[] code = new byte[length];

        using FileStream fs = new(job.CodePath, FileMode.Open, FileAccess.Read, FileShare.Read);
        fs.Seek(job.CodeOffset, SeekOrigin.Begin);
        fs.ReadExactly(code);

        if ((job.Stage is "vs" or "ps") && code.Length < 0x90)
        {
            throw new JobException(Kinds.BadJob, $"a graphics program of {code.Length} bytes has no room for its 0x80-byte header and an instruction");
        }

        return code;
    }

    private static string Fail(string key, TranslatorTiming timing, string kind, string message)
    {
        string text = Sanitize($"{kind}: {message}", MaxMessage);

        return $"fail {key} {timing.Micros} {text}";
    }

    /// <summary>The hello payload: one line of JSON.</summary>
    public string HelloJson()
    {
        using MemoryStream ms = new();

        using (Utf8JsonWriter w = new(ms))
        {
            JobOptions d = new();

            w.WriteStartObject();
            w.WriteNumber("proto", Protocol);
            w.WriteString("translator_id", _translatorId);
            w.WriteString("spirv", "1.3");
            w.WriteString("spirv_task_mesh", "1.4");
            w.WriteString("target", "vulkan1.3");
            w.WriteStartObject("defaults");
            w.WriteBoolean("debug", d.Debug);
            w.WriteBoolean("reduced", d.Reduced);
            w.WriteBoolean("float64", d.Float64);
            w.WriteBoolean("nonuniform_indexing", d.NonUniformIndexing);
            w.WriteBoolean("capability_fixup", d.CapabilityFixup);
            w.WriteString("variant", d.Variant);
            w.WriteNumber("heap_images", Job.DefaultHeapImages);
            w.WriteNumber("heap_samplers", Job.DefaultHeapSamplers);
            w.WriteEndObject();
            w.WriteEndObject();
        }

        return Encoding.UTF8.GetString(ms.ToArray());
    }

    /// <summary>One line, no control characters, at most <paramref name="max"/> characters.</summary>
    public static string Sanitize(string text, int max)
    {
        StringBuilder sb = new(Math.Min(text.Length, max));

        foreach (char c in text)
        {
            if (sb.Length >= max)
            {
                break;
            }

            sb.Append(c < ' ' || c == 0x7f ? ' ' : c);
        }

        return sb.ToString();
    }

    private static string FirstLine(string text)
    {
        int n = text.IndexOf('\n');

        return n < 0 ? text : text.Substring(0, n);
    }

    private static string Clip(string s)
    {
        return s.Length <= 60 ? s : s.Substring(0, 60) + "...";
    }
}
