using System.Buffers.Binary;

namespace OtterLogic.MachineLearning.Inference.Export;

/// <summary>
/// A model graph being written: one float input called <c>features</c>, the
/// operators between, and the outputs <see cref="OnnxModel"/> reads.
/// <para>
/// This is the writing half of the ONNX seam. The reading half never changes for a
/// model made here rather than by scikit-learn or PyTorch: the graph interface is
/// the one fixed in <see cref="OnnxModel"/> — <c>features</c> in; <c>value</c>, or
/// <c>label</c> and <c>probabilities</c>, out — and the metadata record travels
/// inside the file under the same key. A learner exports by wiring a handful of
/// operators between the input and one of the two heads; the arithmetic it needs
/// is small (an affine step, a tree ensemble, a softmax) and each is one method here.
/// </para>
/// <para>
/// Every constant goes in as float32, because that is what the graph computes in
/// and what ONNX Runtime's CPU kernels take. A caller comparing a model's own
/// double answers with the graph's must expect float32 differences; see
/// <see cref="OnnxExport"/>.
/// </para>
/// </summary>
public sealed class OnnxGraph
{
    /// <summary>The default operator domain, <c>ai.onnx</c>, and the version of it written.</summary>
    public const long OpsetVersion = 17;

    /// <summary>The domain the tree ensemble lives in.</summary>
    public const string MlDomain = "ai.onnx.ml";

    /// <summary>The version of <see cref="MlDomain"/> written. Version 3 is the last with <c>TreeEnsembleRegressor</c> as its own operator, and every runtime since 2020 reads it.</summary>
    public const long MlOpsetVersion = 3;

    /// <summary>IR version 8 is what onnx 1.14 and ONNX Runtime 1.15 onward read, and none of the operators here need newer.</summary>
    private const long IrVersion = 8;

    private const int FloatElement = 1;
    private const int Int64Element = 7;

    private readonly List<Operator> _nodes = new();
    private readonly List<Tensor> _initializers = new();
    private readonly List<ValueInfo> _outputs = new();
    private readonly HashSet<string> _names = new(StringComparer.Ordinal);
    private bool _usesMl;

    /// <summary>Starts a graph taking <paramref name="featureCount"/> float features per row.</summary>
    public OnnxGraph(int featureCount)
    {
        if (featureCount < 1)
            throw new ArgumentOutOfRangeException(nameof(featureCount), featureCount, "A model takes at least one feature.");

        FeatureCount = featureCount;
        _names.Add(OnnxModel.InputName);
    }

    /// <summary>Columns the input takes.</summary>
    public int FeatureCount { get; }

    /// <summary>The name of the graph's input tensor: float32, rows by <see cref="FeatureCount"/>.</summary>
    public string Input => OnnxModel.InputName;

    /// <summary>A float constant, with the given shape.</summary>
    public string Constant(string name, IReadOnlyList<double> values, params int[] dims)
    {
        RequireCount(values.Count, dims, name);
        var raw = new byte[values.Count * 4];
        for (int i = 0; i < values.Count; i++)
            BinaryPrimitives.WriteSingleLittleEndian(raw.AsSpan(i * 4), (float)values[i]);

        string unique = Unique(name);
        _initializers.Add(new Tensor(unique, FloatElement, dims, raw));
        return unique;
    }

    /// <summary>A float matrix constant, <c>[rows, columns]</c> as the array is.</summary>
    public string Constant(string name, double[,] matrix)
    {
        int rows = matrix.GetLength(0);
        int columns = matrix.GetLength(1);
        var flat = new double[rows * columns];
        for (int i = 0; i < rows; i++)
            for (int j = 0; j < columns; j++)
                flat[i * columns + j] = matrix[i, j];

        return Constant(name, flat, rows, columns);
    }

    /// <summary>An int64 constant, with the given shape.</summary>
    public string Constant(string name, IReadOnlyList<long> values, params int[] dims)
    {
        RequireCount(values.Count, dims, name);
        var raw = new byte[values.Count * 8];
        for (int i = 0; i < values.Count; i++)
            BinaryPrimitives.WriteInt64LittleEndian(raw.AsSpan(i * 8), values[i]);

        string unique = Unique(name);
        _initializers.Add(new Tensor(unique, Int64Element, dims, raw));
        return unique;
    }

    /// <summary>
    /// Adds an operator with one output and returns that output's name.
    /// </summary>
    /// <param name="opType">The operator, e.g. <c>MatMul</c>.</param>
    /// <param name="inputs">Tensor names, in the order the operator takes them.</param>
    /// <param name="attributes">Its settings.</param>
    public string Node(string opType, IReadOnlyList<string> inputs, params OnnxAttribute[] attributes)
        => Node(opType, string.Empty, inputs, attributes);

    /// <summary>Adds an operator from a named domain with one output.</summary>
    public string Node(string opType, string domain, IReadOnlyList<string> inputs, params OnnxAttribute[] attributes)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        foreach (string input in inputs)
            if (!_names.Contains(input))
                throw new ArgumentException($"'{input}' is not a tensor in this graph.", nameof(inputs));

        if (domain == MlDomain)
            _usesMl = true;

        string stem = opType.ToLowerInvariant();
        string output = Unique(stem);
        _nodes.Add(new Operator(opType, domain, inputs.ToArray(), new[] { output }, attributes, Unique(stem + "_node")));
        return output;
    }

    // ---- the affine and activation steps every non-tree learner is made of ----

    /// <summary>
    /// Subtracts <paramref name="centre"/> and divides by <paramref name="scale"/>,
    /// column by column: the standardisation a model was fitted with, baked in so the
    /// reader never repeats it and cannot repeat it wrongly. Both arrays are per
    /// input column; a column the model does not use should carry centre 0 and
    /// scale 1, never a scale of zero.
    /// </summary>
    public string Standardise(string input, IReadOnlyList<double> centre, IReadOnlyList<double> scale)
    {
        if (centre.Count != FeatureCount || scale.Count != FeatureCount)
            throw new ArgumentException($"Centre and scale need one value per feature ({FeatureCount}).");
        for (int j = 0; j < FeatureCount; j++)
            if (!(Math.Abs(scale[j]) > 0.0) || !double.IsFinite(scale[j]) || !double.IsFinite(centre[j]))
                throw new ArgumentException($"Column {j} has centre {centre[j]} and scale {scale[j]}; a graph dividing by that answers NaN.");

        string centred = Node("Sub", new[] { input, Constant("centre", centre, FeatureCount) });
        return Node("Div", new[] { centred, Constant("scale", scale, FeatureCount) });
    }

    /// <summary>
    /// <c>x · W + b</c>. <paramref name="weights"/> is <c>[inputs, outputs]</c>, so a
    /// row of it is one input's contribution to every output; that is the layout
    /// MatMul wants for a batch of rows on the left.
    /// </summary>
    public string Dense(string input, double[,] weights, IReadOnlyList<double> bias)
    {
        if (bias.Count != weights.GetLength(1))
            throw new ArgumentException($"The weights give {weights.GetLength(1)} outputs and there are {bias.Count} bias values.");

        string product = Node("MatMul", new[] { input, Constant("weights", weights) });
        return Node("Add", new[] { product, Constant("bias", bias, bias.Count) });
    }

    public string Relu(string input) => Node("Relu", new[] { input });

    /// <summary>Softmax across the columns of each row.</summary>
    public string Softmax(string input) => Node("Softmax", new[] { input }, OnnxAttribute.Int("axis", 1));

    /// <summary>
    /// <c>x · scale + offset</c> on every element — the un-standardisation of a
    /// regression target the model was fitted to in standard units.
    /// </summary>
    public string Rescale(string input, double scale, double offset)
    {
        string scaled = Node("Mul", new[] { input, Constant("target_scale", new[] { scale }, 1) });
        return Node("Add", new[] { scaled, Constant("target_offset", new[] { offset }, 1) });
    }

    /// <summary>
    /// One score per row into a two-column probability: <c>[1 − σ(s), σ(s)]</c>. The
    /// head for a binary classifier fitted as a single logistic score, which is how
    /// boosting fits two classes.
    /// </summary>
    public string SigmoidPair(string score)
    {
        string p = Node("Sigmoid", new[] { score });
        string q = Node("Sub", new[] { Constant("one", new[] { 1.0 }, 1), p });
        return Node("Concat", new[] { q, p }, OnnxAttribute.Int("axis", 1));
    }

    /// <summary>The trees summed or averaged into <see cref="Export.TreeEnsemble.TargetCount"/> columns; see <see cref="Export.TreeEnsemble"/>.</summary>
    public string TreeEnsemble(string input, TreeEnsemble ensemble)
    {
        ArgumentNullException.ThrowIfNull(ensemble);
        ensemble.Validate(FeatureCount);
        return Node("TreeEnsembleRegressor", MlDomain, new[] { input }, ensemble.Attributes().ToArray());
    }

    // ---- the two heads OnnxModel reads ----

    /// <summary>Declares <paramref name="tensor"/>, float <c>[rows, 1]</c>, as the regressor's <c>value</c>.</summary>
    public void ValueOutput(string tensor)
    {
        Require(tensor);
        RequireNoOutputs();
        _outputs.Add(new ValueInfo(OnnxModel.ValueOutput, FloatElement, new long[] { -1, 1 }));
        Rename(tensor, OnnxModel.ValueOutput);
    }

    /// <summary>
    /// Declares <paramref name="probabilities"/>, float <c>[rows, classes]</c>, as the
    /// classifier's <c>probabilities</c>, and derives its <c>label</c> — the index of
    /// the largest, ties to the lowest index as ArgMax breaks them — as int64 <c>[rows]</c>.
    /// </summary>
    public void ClassificationOutputs(string probabilities, int classCount)
    {
        Require(probabilities);
        RequireNoOutputs();
        if (classCount < 2)
            throw new ArgumentOutOfRangeException(nameof(classCount), classCount, "A classifier has at least two classes.");

        string label = Node("ArgMax", new[] { probabilities },
            OnnxAttribute.Int("axis", 1), OnnxAttribute.Int("keepdims", 0), OnnxAttribute.Int("select_last_index", 0));

        _outputs.Add(new ValueInfo(OnnxModel.LabelOutput, Int64Element, new long[] { -1 }));
        _outputs.Add(new ValueInfo(OnnxModel.ProbabilitiesOutput, FloatElement, new long[] { -1, classCount }));
        Rename(label, OnnxModel.LabelOutput);
        Rename(probabilities, OnnxModel.ProbabilitiesOutput);
    }

    /// <summary>
    /// The finished model as the bytes of an <c>.onnx</c> file.
    /// </summary>
    /// <param name="metadata">What the model is, written into the file under <see cref="ModelMetadata.Key"/>. Its feature count and task must match the graph's.</param>
    /// <param name="producer">Who made it, for the file's <c>producer_name</c>.</param>
    public byte[] ToBytes(ModelMetadata metadata, string producer = "otterlogic")
    {
        ArgumentNullException.ThrowIfNull(metadata);
        metadata.Validate();
        if (_outputs.Count == 0)
            throw new InvalidOperationException("The graph has no outputs: call ValueOutput or ClassificationOutputs first.");
        if (metadata.Features.Count != FeatureCount)
            throw new ArgumentException(
                $"The metadata names {metadata.Features.Count} features and the graph takes {FeatureCount}.", nameof(metadata));

        bool classification = _outputs.Any(o => o.Name == OnnxModel.LabelOutput);
        if (classification != (metadata.Task == ModelTask.Classification))
            throw new ArgumentException("The metadata's task does not match the head the graph was given.", nameof(metadata));

        var graph = new ProtobufWriter();
        foreach (var node in _nodes)
            graph.Message(1, node.Write());
        graph.String(2, "otterlogic");
        foreach (var tensor in _initializers)
            graph.Message(5, tensor.Write());
        graph.Message(11, new ValueInfo(Input, FloatElement, new long[] { -1, FeatureCount }).Write());
        foreach (var output in _outputs)
            graph.Message(12, output.Write());

        var model = new ProtobufWriter();
        model.Varint(1, IrVersion);
        model.String(2, producer);
        model.String(6, "An OtterLogic model. Its metadata_props['otterlogic'] says what it predicts and what it takes.");
        model.Message(7, graph);

        var opset = new ProtobufWriter();
        opset.String(1, string.Empty);
        opset.Varint(2, OpsetVersion);
        model.Message(8, opset);
        if (_usesMl)
        {
            var ml = new ProtobufWriter();
            ml.String(1, MlDomain);
            ml.Varint(2, MlOpsetVersion);
            model.Message(8, ml);
        }

        var entry = new ProtobufWriter();
        entry.String(1, ModelMetadata.Key);
        entry.String(2, metadata.ToJson());
        model.Message(14, entry);

        return model.ToArray();
    }

    // ---- plumbing ----

    private void Require(string tensor)
    {
        if (!_names.Contains(tensor))
            throw new ArgumentException($"'{tensor}' is not a tensor in this graph.");
    }

    private void RequireNoOutputs()
    {
        if (_outputs.Count > 0)
            throw new InvalidOperationException("The graph already has its outputs.");
    }

    /// <summary>
    /// Gives an existing tensor the name a head needs, wherever it is produced or
    /// consumed. Renaming rather than adding an Identity node keeps the graph as
    /// small as the arithmetic.
    /// </summary>
    private void Rename(string from, string to)
    {
        if (_names.Contains(to))
            throw new InvalidOperationException($"'{to}' is already a tensor in this graph.");

        if (from == Input)
            throw new InvalidOperationException("An output has to be produced by an operator, not be the input itself.");

        foreach (var node in _nodes)
        {
            for (int i = 0; i < node.Inputs.Length; i++)
                if (node.Inputs[i] == from) node.Inputs[i] = to;
            for (int i = 0; i < node.Outputs.Length; i++)
                if (node.Outputs[i] == from) node.Outputs[i] = to;
        }

        _names.Remove(from);
        _names.Add(to);
    }

    private string Unique(string stem)
    {
        string name = stem;
        int n = 1;
        while (!_names.Add(name))
            name = $"{stem}_{++n}";
        return name;
    }

    private static void RequireCount(int count, int[] dims, string name)
    {
        long expected = 1;
        foreach (int dim in dims)
        {
            if (dim < 1) throw new ArgumentException($"Constant '{name}' has a dimension of {dim}.");
            expected *= dim;
        }

        if (expected != count)
            throw new ArgumentException($"Constant '{name}' holds {count} values for a shape of {expected}.");
    }

    private sealed class Operator
    {
        public Operator(string opType, string domain, string[] inputs, string[] outputs, OnnxAttribute[] attributes, string name)
        {
            OpType = opType;
            Domain = domain;
            Inputs = inputs;
            Outputs = outputs;
            Attributes = attributes;
            Name = name;
        }

        public string OpType { get; }
        public string Domain { get; }
        public string[] Inputs { get; }
        public string[] Outputs { get; }
        public OnnxAttribute[] Attributes { get; }
        public string Name { get; }

        public ProtobufWriter Write()
        {
            var node = new ProtobufWriter();
            foreach (string input in Inputs) node.String(1, input);
            foreach (string output in Outputs) node.String(2, output);
            node.String(3, Name);
            node.String(4, OpType);
            foreach (var attribute in Attributes)
            {
                var proto = new ProtobufWriter();
                attribute.WriteTo(proto);
                node.Message(5, proto);
            }
            if (Domain.Length > 0) node.String(7, Domain);
            return node;
        }
    }

    private sealed record Tensor(string Name, int ElementType, int[] Dims, byte[] Raw)
    {
        public ProtobufWriter Write()
        {
            var tensor = new ProtobufWriter();
            foreach (int dim in Dims) tensor.Varint(1, dim);
            tensor.Varint(2, ElementType);
            tensor.String(8, Name);
            tensor.Bytes(9, Raw);
            return tensor;
        }
    }

    /// <summary>A graph input or output: name, element type, shape with -1 for the row dimension.</summary>
    private sealed record ValueInfo(string Name, int ElementType, long[] Dims)
    {
        public ProtobufWriter Write()
        {
            var shape = new ProtobufWriter();
            foreach (long dim in Dims)
            {
                var dimension = new ProtobufWriter();
                if (dim < 0) dimension.String(2, "N");
                else dimension.Varint(1, dim);
                shape.Message(1, dimension);
            }

            var tensorType = new ProtobufWriter();
            tensorType.Varint(1, ElementType);
            tensorType.Message(2, shape);

            var type = new ProtobufWriter();
            type.Message(1, tensorType);

            var info = new ProtobufWriter();
            info.String(1, Name);
            info.Message(2, type);
            return info;
        }
    }
}
