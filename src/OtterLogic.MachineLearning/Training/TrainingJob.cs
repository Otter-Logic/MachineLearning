using System.Text.Json;
using System.Text.Json.Serialization;

namespace OtterLogic.MachineLearning.Training;

/// <summary>
/// Everything the trainer needs to produce a model, as the <c>job.json</c> it is
/// started with.
/// <para>
/// A file rather than arguments so that the whole request is in one place a person
/// can read after the fact, and so that adding a setting later does not change how
/// the process is launched. The Python side reads exactly these names.
/// </para>
/// </summary>
public sealed record TrainingJob
{
    /// <summary>The dataset folder — <c>schema.json</c> and one CSV per model under <c>models/</c>.</summary>
    public required string DatasetFolder { get; init; }

    /// <summary>
    /// The target column to predict. Null means the schema's only target; a schema
    /// with several needs one named. Whether it is a class or a number comes from the
    /// schema, which is why there is one job and not two.
    /// </summary>
    public string? Target { get; init; }

    /// <summary>The method to fit and its settings. Boosted trees unless told otherwise.</summary>
    public Learner Learner { get; init; } = new BoostedTreesLearner();

    /// <summary>Whether whole groups or single rows are held back to score on. Groups unless the samples came with none.</summary>
    public HoldoutBy HoldoutBy { get; init; } = HoldoutBy.Group;

    /// <summary>
    /// Share of the groups — or of the rows, by <see cref="HoldoutBy"/> — held out to
    /// score on. The same default as <see cref="Data.GroupSplit.Holdout"/>, for the same reason.
    /// </summary>
    public double HoldoutFraction { get; init; } = 0.25;

    /// <summary>Fixes the split and the fit, so the same job gives the same model.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Where the <c>.onnx</c> is written. Written beside and renamed over, so a half-written file is never left there.</summary>
    public required string OutputPath { get; init; }

    /// <summary>Complains about anything the trainer would refuse, before a process is started for it.</summary>
    /// <exception cref="ArgumentException">Something is wrong; the message says what.</exception>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(DatasetFolder))
            throw new ArgumentException("No dataset folder was given.");

        if (!Directory.Exists(DatasetFolder))
            throw new ArgumentException($"There is no folder at '{DatasetFolder}'.");

        if (!File.Exists(Path.Combine(DatasetFolder, "schema.json")))
            throw new ArgumentException(
                $"'{DatasetFolder}' is not a dataset folder: it has no schema.json. Write Dataset creates one.");

        if (Target is not null && string.IsNullOrWhiteSpace(Target))
            throw new ArgumentException("The target is blank. Leave it out to use the schema's only target.");

        if (Learner is null)
            throw new ArgumentException("No learner was given.");

        try
        {
            Learner.Validate();
        }
        catch (ArgumentOutOfRangeException ex)
        {
            throw new ArgumentException($"{Learner.Name}: {ex.Message}", ex);
        }

        if (!Enum.IsDefined(HoldoutBy))
            throw new ArgumentException($"{(int)HoldoutBy} is not a way to hold out.");

        if (!(HoldoutFraction > 0.0 && HoldoutFraction < 1.0))
            throw new ArgumentException("The holdout must be between 0 and 1, exclusive.");

        if (string.IsNullOrWhiteSpace(OutputPath))
            throw new ArgumentException("No output path was given for the model.");

        if (!string.Equals(Path.GetExtension(OutputPath), ".onnx", StringComparison.OrdinalIgnoreCase))
            throw new ArgumentException($"The model file must end in .onnx; '{OutputPath}' does not.");

        string? folder = Path.GetDirectoryName(Path.GetFullPath(OutputPath));
        if (folder is not null && !Directory.Exists(folder))
            throw new ArgumentException($"The folder for the model, '{folder}', does not exist.");
    }

    private static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    /// <summary>The job as <c>job.json</c>. The learner goes out as an object with a <c>type</c>, because the Python side reads it.</summary>
    public string ToJson() => JsonSerializer.Serialize(this, Json);

    /// <summary>Reads a <c>job.json</c> back.</summary>
    public static TrainingJob FromJson(string json)
        => JsonSerializer.Deserialize<TrainingJob>(json, Json)
            ?? throw new InvalidDataException("job.json is empty.");
}
