namespace OtterLogic.MachineLearning.Inference.Export;

/// <summary>
/// One attribute on an ONNX node — an operator's setting, as opposed to its inputs.
/// Made through the factories so the wire type is always right for the value.
/// </summary>
public sealed class OnnxAttribute
{
    // AttributeProto.AttributeType, as onnx.proto numbers them.
    private const int FloatType = 1;
    private const int IntType = 2;
    private const int StringType = 3;
    private const int FloatsType = 6;
    private const int IntsType = 7;
    private const int StringsType = 8;

    private readonly int _type;
    private readonly float[]? _floats;
    private readonly long[]? _ints;
    private readonly string[]? _strings;

    private OnnxAttribute(string name, int type, float[]? floats, long[]? ints, string[]? strings)
    {
        Name = name;
        _type = type;
        _floats = floats;
        _ints = ints;
        _strings = strings;
    }

    /// <summary>The attribute's name, as the operator's specification spells it.</summary>
    public string Name { get; }

    public static OnnxAttribute Int(string name, long value) => new(name, IntType, null, new[] { value }, null);

    public static OnnxAttribute Float(string name, float value) => new(name, FloatType, new[] { value }, null, null);

    public static OnnxAttribute String(string name, string value) => new(name, StringType, null, null, new[] { value });

    public static OnnxAttribute Ints(string name, IEnumerable<long> values) => new(name, IntsType, null, values.ToArray(), null);

    public static OnnxAttribute Floats(string name, IEnumerable<float> values) => new(name, FloatsType, values.ToArray(), null, null);

    public static OnnxAttribute Strings(string name, IEnumerable<string> values) => new(name, StringsType, null, null, values.ToArray());

    internal void WriteTo(ProtobufWriter attribute)
    {
        attribute.String(1, Name);
        attribute.Varint(20, _type);

        switch (_type)
        {
            case FloatType:
                attribute.Float(2, _floats![0]);
                break;
            case IntType:
                attribute.Varint(3, _ints![0]);
                break;
            case StringType:
                attribute.String(4, _strings![0]);
                break;
            case FloatsType:
                // One tag per element rather than a packed run: onnx.proto is
                // proto2 syntax without [packed], and every parser reads this form.
                foreach (float f in _floats!)
                    attribute.Float(7, f);
                break;
            case IntsType:
                foreach (long i in _ints!)
                    attribute.Varint(8, i);
                break;
            case StringsType:
                foreach (string s in _strings!)
                    attribute.String(9, s);
                break;
        }
    }
}
