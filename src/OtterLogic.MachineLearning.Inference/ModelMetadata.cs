using System.Globalization;
using System.Text;
using System.Text.Json;
using System.Text.Json.Serialization;

namespace OtterLogic.MachineLearning.Inference;

/// <summary>
/// What a trained model knows about itself, carried inside the <c>.onnx</c> file
/// rather than beside it.
/// <para>
/// An ONNX graph holds the arithmetic and nothing else: not which column goes in
/// which input, not what the output means, not what the classes are called. All of
/// that is here, written into the file's <c>metadata_props</c> under
/// <see cref="Key"/> by whatever trained it and read back by <see cref="OnnxModel"/>.
/// Inside the file rather than in a sidecar so that a model is <em>one</em> file:
/// a file that has to travel with a partner is a file that arrives alone.
/// </para>
/// <para>
/// This record and its Python twin in <c>python/trainer/otterlogic_trainer/metadata.py</c>
/// are the one thing the two languages have to agree on. The inference parity
/// fixture is what keeps them agreeing.
/// </para>
/// </summary>
public sealed record ModelMetadata
{
    /// <summary>The <c>metadata_props</c> key the JSON is stored under.</summary>
    public const string Key = "otterlogic";

    /// <summary>The only format there has been.</summary>
    public const int CurrentFormat = 1;

    /// <summary>The layout of this record, so a later format can be refused rather than misread.</summary>
    public int Format { get; init; } = CurrentFormat;

    /// <summary>Whether the model answers with a class or a number.</summary>
    public ModelTask Task { get; init; }

    /// <summary>
    /// The feature columns, in the order the model's input takes them. Order is all
    /// the graph knows about its inputs, so the names are the only thing that lets a
    /// person check they wired the right columns.
    /// </summary>
    public IReadOnlyList<string> Features { get; init; } = Array.Empty<string>();

    /// <summary>The column the model predicts.</summary>
    public string Target { get; init; } = string.Empty;

    /// <summary>
    /// For a classifier, the class names in the order the model indexes them, so the
    /// graph's <c>label</c> output is a position in this list. Empty for a regressor.
    /// </summary>
    public IReadOnlyList<string> Classes { get; init; } = Array.Empty<string>();

    /// <summary>
    /// The dataset schema's extractor version, copied through, so a model trained on
    /// one version of the features can refuse rows from another the way a dataset
    /// folder does.
    /// </summary>
    public string? ExtractorVersion { get; init; }

    /// <summary>What produced the model.</summary>
    public TrainerInfo? Trainer { get; init; }

    /// <summary>When it was trained.</summary>
    public DateTimeOffset? Trained { get; init; }

    /// <summary>What it was trained on.</summary>
    public DataInfo? Data { get; init; }

    /// <summary>
    /// The score on the held-out groups, by name. A flat map rather than a typed record
    /// so a trainer can report what suits the task without this type changing:
    /// <c>accuracy</c>, <c>noInformationRate</c>, <c>balancedAccuracy</c> and
    /// <c>macroF1</c> for a classifier; <c>rSquared</c>, <c>meanAbsoluteError</c> and
    /// <c>rootMeanSquaredError</c> for a regressor — the same names the Evaluate
    /// components report, so the two can be read side by side.
    /// </summary>
    public IReadOnlyDictionary<string, double> Score { get; init; } = new Dictionary<string, double>();

    /// <summary>Checks the record describes a model that can be run.</summary>
    /// <exception cref="InvalidDataException">It does not; the message says what is wrong.</exception>
    public void Validate()
    {
        if (Format != CurrentFormat)
            throw new InvalidDataException(
                $"The model's metadata is format {Format} and this version of OtterLogic reads format {CurrentFormat}.");

        if (Features.Count == 0)
            throw new InvalidDataException("The model's metadata lists no features.");

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var feature in Features)
        {
            if (string.IsNullOrWhiteSpace(feature))
                throw new InvalidDataException("The model's metadata has a feature with no name.");
            if (!seen.Add(feature))
                throw new InvalidDataException($"The model's metadata lists the feature '{feature}' twice.");
        }

        if (string.IsNullOrWhiteSpace(Target))
            throw new InvalidDataException("The model's metadata does not say what it predicts.");

        if (Task == ModelTask.Classification)
        {
            if (Classes.Count < 2)
                throw new InvalidDataException(
                    $"A classifier needs at least two classes and this one lists {Classes.Count}.");

            if (Classes.Any(string.IsNullOrWhiteSpace))
                throw new InvalidDataException("The model's metadata has a class with no name.");

            if (Classes.Distinct(StringComparer.Ordinal).Count() != Classes.Count)
                throw new InvalidDataException("The model's metadata lists the same class twice.");
        }
        else if (Classes.Count > 0)
        {
            throw new InvalidDataException("A regressor predicts a number, but the metadata lists classes.");
        }

        if (Data is { } data && (data.Rows <= 0 || data.Groups <= 0))
            throw new InvalidDataException("The model's metadata reports training on no rows or no groups.");
    }

    /// <summary>
    /// The record as lines a person can read — what the model predicts, what it
    /// wants, where it came from and how it did. This is what a Predict component
    /// reports with nothing wired but the file, and is the reason there is no
    /// separate readable sidecar.
    /// </summary>
    public IReadOnlyList<string> Describe()
    {
        var lines = new List<string>();

        lines.Add(Task == ModelTask.Classification
            ? $"Predicts {Target}: one of {string.Join(", ", Classes)}."
            : $"Predicts {Target}: a number.");

        lines.Add($"{Features.Count} feature{(Features.Count == 1 ? string.Empty : "s")}, in this order: "
            + string.Join(", ", Features) + ".");

        var origin = new StringBuilder("Trained");
        if (Trained is { } when)
            origin.Append(' ').Append(when.ToString("yyyy-MM-dd", CultureInfo.InvariantCulture));
        if (Trainer is { } trainer)
        {
            origin.Append(" with ").Append(trainer.Name).Append(' ').Append(trainer.Version);
            if (!string.IsNullOrWhiteSpace(trainer.ModelType))
                origin.Append(" (").Append(Humanise(trainer.ModelType)).Append(')');
        }
        if (Data is { } data)
            origin.Append($" on {data.Rows} rows from {data.Groups} group{(data.Groups == 1 ? string.Empty : "s")}");
        origin.Append('.');
        if (origin.Length > "Trained.".Length)
            lines.Add(origin.ToString());

        if (Data is { HoldoutGroups.Count: > 0 } held)
            lines.Add("Scored on held-out groups: " + string.Join(", ", held.HoldoutGroups) + ".");

        if (Score.Count > 0)
            lines.Add("Score: " + string.Join(", ",
                Score.Select(s => $"{Humanise(s.Key)} {s.Value.ToString("0.###", CultureInfo.InvariantCulture)}")) + ".");

        if (!string.IsNullOrWhiteSpace(ExtractorVersion))
            lines.Add($"Extractor version {ExtractorVersion}.");

        return lines;
    }

    /// <summary><c>noInformationRate</c> as "no information rate", for the report.</summary>
    private static string Humanise(string camel)
    {
        var text = new StringBuilder(camel.Length + 4);
        for (int i = 0; i < camel.Length; i++)
        {
            char c = camel[i];
            if (char.IsUpper(c) && i > 0 && !char.IsUpper(camel[i - 1]))
                text.Append(' ');
            text.Append(char.ToLowerInvariant(c));
        }

        return text.ToString();
    }

    private static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = false,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    /// <summary>The record as the JSON stored in the model file. Enums go out as words, as the Python side writes them.</summary>
    public string ToJson() => JsonSerializer.Serialize(this, Json);

    /// <summary>Reads the JSON stored in a model file, and validates what it finds.</summary>
    /// <exception cref="InvalidDataException">Not this record, or not a model that can be run.</exception>
    public static ModelMetadata FromJson(string json)
    {
        ModelMetadata? metadata;
        try
        {
            metadata = JsonSerializer.Deserialize<ModelMetadata>(json, Json);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"The model's metadata could not be read: {ex.Message}", ex);
        }

        if (metadata is null)
            throw new InvalidDataException("The model's metadata is empty.");

        metadata.Validate();
        return metadata;
    }
}

/// <summary>What produced a model.</summary>
/// <param name="Name">The trainer, e.g. <c>otterlogic-trainer</c>.</param>
/// <param name="Version">Its version.</param>
/// <param name="ModelType">The method it fitted, in the trainer's own words, e.g. <c>boostedTrees</c>.</param>
public sealed record TrainerInfo(string Name, string Version, string ModelType);

/// <summary>What a model was trained on.</summary>
/// <param name="Rows">Rows in the dataset, held-out ones included.</param>
/// <param name="Groups">Groups in the dataset.</param>
/// <param name="HoldoutGroups">The groups the score was measured on, which the fit never saw.</param>
public sealed record DataInfo(int Rows, int Groups, IReadOnlyList<string> HoldoutGroups);
