using Microsoft.ML.OnnxRuntime;
using Microsoft.ML.OnnxRuntime.Tensors;

namespace OtterLogic.MachineLearning.Inference;

/// <summary>
/// A trained model opened from an <c>.onnx</c> file, ready to answer for rows.
/// <para>
/// Nothing here knows or cares what trained the model. scikit-learn, PyTorch or a
/// hand-built graph all arrive as the same thing: a frozen computation with one
/// float input of shape <c>[rows, features]</c> and the outputs named below, and a
/// <see cref="ModelMetadata"/> record in the file saying what the numbers mean. That
/// is the whole boundary, and it is what lets the trainer be replaced without a
/// change on this side.
/// </para>
/// <para>
/// Opening a file builds an ONNX Runtime session, which is expensive; running one is
/// not. Hold the instance and reuse it — a Grasshopper component keeps one per file
/// and rebuilds only when <see cref="LastWriteTimeUtc"/> changes.
/// </para>
/// </summary>
public sealed class OnnxModel : IDisposable
{
    /// <summary>The graph's input: float32, <c>[rows, features]</c>.</summary>
    public const string InputName = "features";

    /// <summary>A regressor's output: float32, <c>[rows, 1]</c>.</summary>
    public const string ValueOutput = "value";

    /// <summary>A classifier's chosen class: int64, <c>[rows]</c>, a position in <see cref="ModelMetadata.Classes"/>.</summary>
    public const string LabelOutput = "label";

    /// <summary>A classifier's probability per class: float32, <c>[rows, classes]</c>.</summary>
    public const string ProbabilitiesOutput = "probabilities";

    private readonly InferenceSession _session;
    private readonly string _inputName;
    private readonly string[] _outputNames;
    private bool _disposed;

    private OnnxModel(
        string path, DateTime lastWriteTimeUtc, InferenceSession session, ModelMetadata? metadata,
        string inputName, string[] outputNames, int featureCount, ModelTask task)
    {
        Path = path;
        LastWriteTimeUtc = lastWriteTimeUtc;
        _session = session;
        Metadata = metadata;
        _inputName = inputName;
        _outputNames = outputNames;
        FeatureCount = featureCount;
        Task = task;
    }

    /// <summary>The file this was opened from.</summary>
    public string Path { get; }

    /// <summary>When the file was last written, at the moment it was opened. Compare to know whether to reopen.</summary>
    public DateTime LastWriteTimeUtc { get; }

    /// <summary>
    /// What the file says about itself, or null for an <c>.onnx</c> from elsewhere
    /// that carries no OtterLogic metadata. Such a model still runs — the graph is
    /// enough for that — but its features are unnamed and its classes are numbered.
    /// </summary>
    public ModelMetadata? Metadata { get; }

    /// <summary>True when the file carries <see cref="ModelMetadata"/>.</summary>
    public bool HasMetadata => Metadata is not null;

    /// <summary>Columns each row must hold, in order.</summary>
    public int FeatureCount { get; }

    /// <summary>Whether it answers with a class or a number.</summary>
    public ModelTask Task { get; }

    /// <summary>The feature names in input order, or an empty list for a model without metadata.</summary>
    public IReadOnlyList<string> FeatureNames => Metadata?.Features ?? Array.Empty<string>();

    /// <summary>
    /// Opens a model file.
    /// </summary>
    /// <exception cref="FileNotFoundException">No file there.</exception>
    /// <exception cref="InvalidDataException">
    /// ONNX Runtime cannot read it, its metadata is malformed, or its graph does not
    /// have the shape a model needs — one float input of two dimensions.
    /// </exception>
    public static OnnxModel Load(string path)
    {
        if (string.IsNullOrWhiteSpace(path))
            throw new ArgumentException("No model file was given.", nameof(path));

        if (!File.Exists(path))
            throw new FileNotFoundException($"There is no model file at '{path}'.", path);

        DateTime written = File.GetLastWriteTimeUtc(path);

        InferenceSession session;
        try
        {
            session = new InferenceSession(path);
        }
        catch (OnnxRuntimeException ex)
        {
            throw new InvalidDataException($"'{path}' is not a model ONNX Runtime can open: {ex.Message}", ex);
        }

        try
        {
            ModelMetadata? metadata = null;
            if (session.ModelMetadata.CustomMetadataMap.TryGetValue(ModelMetadata.Key, out string? json))
                metadata = ModelMetadata.FromJson(json);

            // With metadata the input is by name; without, whatever the graph has
            // first is the best guess there is.
            string inputName;
            if (metadata is not null)
            {
                if (!session.InputMetadata.ContainsKey(InputName))
                    throw new InvalidDataException(
                        $"The model carries OtterLogic metadata but has no input called '{InputName}'. Its inputs are: "
                        + string.Join(", ", session.InputMetadata.Keys) + ".");
                inputName = InputName;
            }
            else
            {
                inputName = session.InputMetadata.Keys.FirstOrDefault()
                    ?? throw new InvalidDataException("The model has no inputs.");
            }

            var input = session.InputMetadata[inputName];
            if (input.ElementType != typeof(float))
                throw new InvalidDataException(
                    $"The model's input '{inputName}' takes {input.ElementType.Name}, and a model here takes float32.");

            if (input.Dimensions.Length != 2)
                throw new InvalidDataException(
                    $"The model's input '{inputName}' has {input.Dimensions.Length} dimensions. A model here takes "
                    + "rows by features.");

            int graphFeatures = input.Dimensions[1];
            int featureCount;
            if (metadata is not null)
            {
                if (graphFeatures > 0 && graphFeatures != metadata.Features.Count)
                    throw new InvalidDataException(
                        $"The model's metadata names {metadata.Features.Count} features and its graph takes "
                        + $"{graphFeatures}. The file is inconsistent; re-export it.");
                featureCount = metadata.Features.Count;
            }
            else
            {
                if (graphFeatures <= 0)
                    throw new InvalidDataException(
                        "The model carries no OtterLogic metadata and its graph does not fix how many features it "
                        + "takes, so there is no way to know what to feed it.");
                featureCount = graphFeatures;
            }

            var outputs = session.OutputMetadata;
            ModelTask task;
            string[] outputNames;

            if (metadata is not null)
            {
                task = metadata.Task;
                outputNames = task == ModelTask.Classification
                    ? RequireOutputs(outputs, LabelOutput, ProbabilitiesOutput)
                    : RequireOutputs(outputs, ValueOutput);
            }
            else if (outputs.ContainsKey(LabelOutput))
            {
                task = ModelTask.Classification;
                outputNames = outputs.ContainsKey(ProbabilitiesOutput)
                    ? new[] { LabelOutput, ProbabilitiesOutput }
                    : new[] { LabelOutput };
            }
            else
            {
                string first = outputs.Keys.FirstOrDefault()
                    ?? throw new InvalidDataException("The model has no outputs.");
                task = outputs[first].ElementType == typeof(long) || outputs[first].ElementType == typeof(int)
                    ? ModelTask.Classification
                    : ModelTask.Regression;
                outputNames = new[] { first };
            }

            return new OnnxModel(path, written, session, metadata, inputName, outputNames, featureCount, task);
        }
        catch
        {
            session.Dispose();
            throw;
        }
    }

    private static string[] RequireOutputs(IReadOnlyDictionary<string, NodeMetadata> outputs, params string[] names)
    {
        foreach (var name in names)
        {
            if (!outputs.ContainsKey(name))
                throw new InvalidDataException(
                    $"The model carries OtterLogic metadata but has no output called '{name}'. Its outputs are: "
                    + string.Join(", ", outputs.Keys) + ".");
        }

        return names;
    }

    /// <summary>
    /// Answers for a block of rows.
    /// </summary>
    /// <param name="inputs">Rows by features, in the order <see cref="FeatureNames"/> gives.</param>
    /// <exception cref="ArgumentException">The wrong number of columns, or a value that is not finite.</exception>
    /// <exception cref="InvalidDataException">The graph answered with a shape or type a model here does not produce.</exception>
    public ModelPrediction Predict(double[,] inputs)
    {
        ArgumentNullException.ThrowIfNull(inputs);
        ObjectDisposedException.ThrowIf(_disposed, this);

        int rows = inputs.GetLength(0);
        int columns = inputs.GetLength(1);

        if (columns != FeatureCount)
        {
            string expected = FeatureNames.Count > 0
                ? $"The model expects {FeatureCount}: {string.Join(", ", FeatureNames)}."
                : $"The model expects {FeatureCount}.";
            throw new ArgumentException($"Each row holds {columns} values. {expected}", nameof(inputs));
        }

        if (rows == 0)
            return Task == ModelTask.Classification
                ? new ModelPrediction(Array.Empty<string>(), Array.Empty<int>(), Array.Empty<double>(), new double[0, 0])
                : new ModelPrediction(Array.Empty<double>());

        var tensor = new DenseTensor<float>(new[] { rows, columns });
        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                double value = inputs[i, j];
                if (!double.IsFinite(value))
                    throw new ArgumentException($"Row {i} holds {value} at position {j}. A model needs finite numbers.", nameof(inputs));
                tensor[i, j] = (float)value;
            }
        }

        var feed = new[] { NamedOnnxValue.CreateFromTensor(_inputName, tensor) };

        IDisposableReadOnlyCollection<DisposableNamedOnnxValue> results;
        try
        {
            results = _session.Run(feed, _outputNames);
        }
        catch (OnnxRuntimeException ex)
        {
            throw new InvalidDataException($"The model failed to run: {ex.Message}", ex);
        }

        using (results)
        {
            return Task == ModelTask.Classification
                ? ReadClassification(results, rows)
                : ReadRegression(results, rows);
        }
    }

    private ModelPrediction ReadRegression(IReadOnlyList<DisposableNamedOnnxValue> results, int rows)
    {
        var values = ReadFloats(results[0], rows, "value");
        return new ModelPrediction(values);
    }

    private ModelPrediction ReadClassification(IReadOnlyList<DisposableNamedOnnxValue> results, int rows)
    {
        var indices = ReadLabels(results[0], rows);

        double[,] probabilities = new double[0, 0];
        double[] confidence = new double[rows];

        if (results.Count > 1 && results[1].Value is Tensor<float> tensor)
        {
            if (tensor.Dimensions.Length != 2 || tensor.Dimensions[0] != rows)
                throw new InvalidDataException(
                    $"The model's probabilities came back as [{string.Join(", ", tensor.Dimensions.ToArray())}] "
                    + $"for {rows} rows.");

            int classes = tensor.Dimensions[1];
            probabilities = new double[rows, classes];
            for (int i = 0; i < rows; i++)
            {
                double best = double.NegativeInfinity;
                for (int c = 0; c < classes; c++)
                {
                    double p = tensor[i, c];
                    probabilities[i, c] = p;
                    if (p > best) best = p;
                }

                // The confidence is the probability of the class the graph chose,
                // which is the maximum unless the graph's own tie-break differs
                // from ours; reading it by index keeps the two consistent.
                int chosen = indices[i];
                confidence[i] = chosen >= 0 && chosen < classes ? probabilities[i, chosen] : best;
            }
        }
        else
        {
            // A graph with a label and no probabilities is certain of everything,
            // and says so.
            Array.Fill(confidence, 1.0);
        }

        var classNames = Metadata?.Classes;
        var labels = new string[rows];
        for (int i = 0; i < rows; i++)
        {
            int index = indices[i];
            if (classNames is not null)
            {
                if (index < 0 || index >= classNames.Count)
                    throw new InvalidDataException(
                        $"The model answered class {index} and its metadata lists {classNames.Count} classes.");
                labels[i] = classNames[index];
            }
            else
            {
                labels[i] = index.ToString(System.Globalization.CultureInfo.InvariantCulture);
            }
        }

        return new ModelPrediction(labels, indices, confidence, probabilities);
    }

    private static int[] ReadLabels(DisposableNamedOnnxValue output, int rows)
    {
        var indices = new int[rows];
        switch (output.Value)
        {
            case Tensor<long> longs:
                RequireCount(longs, rows, "label");
                for (int i = 0; i < rows; i++) indices[i] = checked((int)longs.GetValue(i));
                return indices;
            case Tensor<int> ints:
                RequireCount(ints, rows, "label");
                for (int i = 0; i < rows; i++) indices[i] = ints.GetValue(i);
                return indices;
            case Tensor<string> strings:
                // An exported classifier whose labels were text. Not what the
                // trainer writes, but a model from elsewhere may; the name is the
                // class and its index is whatever position it is found at.
                RequireCount(strings, rows, "label");
                for (int i = 0; i < rows; i++)
                {
                    string text = strings.GetValue(i);
                    indices[i] = int.TryParse(text, System.Globalization.NumberStyles.Integer,
                        System.Globalization.CultureInfo.InvariantCulture, out int parsed) ? parsed : -1;
                }
                return indices;
            default:
                throw new InvalidDataException(
                    $"The model's label output is {output.Value?.GetType().Name ?? "empty"}, not an integer tensor.");
        }
    }

    private static double[] ReadFloats(DisposableNamedOnnxValue output, int rows, string what)
    {
        if (output.Value is not Tensor<float> tensor)
            throw new InvalidDataException(
                $"The model's {what} output is {output.Value?.GetType().Name ?? "empty"}, not a float tensor.");

        RequireCount(tensor, rows, what);

        var values = new double[rows];
        for (int i = 0; i < rows; i++)
            values[i] = tensor.GetValue(i);
        return values;
    }

    private static void RequireCount<T>(Tensor<T> tensor, int rows, string what)
    {
        if (tensor.Length != rows)
            throw new InvalidDataException(
                $"The model's {what} output holds {tensor.Length} values for {rows} rows.");
    }

    /// <summary>What the file says about itself, as lines, or one line saying it says nothing.</summary>
    public IReadOnlyList<string> Describe()
        => Metadata?.Describe() ?? new[]
        {
            $"No OtterLogic metadata: this is an .onnx from elsewhere. It takes {FeatureCount} unnamed "
            + $"features and answers with a {(Task == ModelTask.Classification ? "class index" : "number")}.",
        };

    public void Dispose()
    {
        if (_disposed) return;
        _disposed = true;
        _session.Dispose();
    }
}
