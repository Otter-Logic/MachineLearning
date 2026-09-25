using System.Buffers.Binary;
using System.Text;

namespace OtterLogic.MachineLearning.Inference.Export;

/// <summary>
/// Writes protocol-buffer wire format: the encoding an <c>.onnx</c> file is, and
/// nothing more of it than a model needs.
/// <para>
/// Hand-written rather than taken from Google.Protobuf for the reason this stack
/// carries no Accord and no MathNet: a second copy of a widely used assembly at a
/// different version is an assembly conflict waiting for a user who runs another
/// plug-in that loads it. The wire format is four cases — varint, fixed 32,
/// fixed 64, length-delimited — and every ONNX message a model needs is made of
/// those, so this is a few dozen lines and a table of field numbers, not a
/// dependency.
/// </para>
/// <para>
/// Only writing. Reading a model is ONNX Runtime's job and it does it well; this
/// exists because it has no writer.
/// </para>
/// </summary>
internal sealed class ProtobufWriter
{
    private const int VarintWire = 0;
    private const int Fixed64Wire = 1;
    private const int LengthDelimitedWire = 2;
    private const int Fixed32Wire = 5;

    private readonly MemoryStream _bytes = new();

    /// <summary>An integer field: int32, int64, an enum, a bool.</summary>
    public void Varint(int field, long value)
    {
        Tag(field, VarintWire);
        // Two's complement as an unsigned varint, ten bytes for a negative — the
        // encoding for int64 as protobuf defines it (not zig-zag, which is sint64).
        WriteVarint(unchecked((ulong)value));
    }

    /// <summary>A float field, four bytes little-endian.</summary>
    public void Float(int field, float value)
    {
        Tag(field, Fixed32Wire);
        Span<byte> buffer = stackalloc byte[4];
        BinaryPrimitives.WriteSingleLittleEndian(buffer, value);
        _bytes.Write(buffer);
    }

    /// <summary>A double field, eight bytes little-endian.</summary>
    public void Double(int field, double value)
    {
        Tag(field, Fixed64Wire);
        Span<byte> buffer = stackalloc byte[8];
        BinaryPrimitives.WriteDoubleLittleEndian(buffer, value);
        _bytes.Write(buffer);
    }

    /// <summary>A bytes field.</summary>
    public void Bytes(int field, ReadOnlySpan<byte> data)
    {
        Tag(field, LengthDelimitedWire);
        WriteVarint((ulong)data.Length);
        _bytes.Write(data);
    }

    /// <summary>A string field, UTF-8.</summary>
    public void String(int field, string text) => Bytes(field, Encoding.UTF8.GetBytes(text));

    /// <summary>A nested message field.</summary>
    public void Message(int field, ProtobufWriter nested) => Bytes(field, nested.ToArray());

    /// <summary>The bytes written so far.</summary>
    public byte[] ToArray() => _bytes.ToArray();

    /// <summary>Length in bytes so far.</summary>
    public long Length => _bytes.Length;

    private void Tag(int field, int wireType) => WriteVarint((ulong)((field << 3) | wireType));

    private void WriteVarint(ulong value)
    {
        while (value >= 0x80)
        {
            _bytes.WriteByte((byte)(value | 0x80));
            value >>= 7;
        }

        _bytes.WriteByte((byte)value);
    }
}
