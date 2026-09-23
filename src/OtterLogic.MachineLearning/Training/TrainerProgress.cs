using System.Text.Json;
using System.Text.Json.Serialization;

namespace OtterLogic.MachineLearning.Training;

/// <summary>
/// One line of the trainer's <c>progress.jsonl</c>.
/// <para>
/// The trainer appends a line as each stage begins and a last one when it is done
/// or has failed; the launcher reads whatever has landed since it last looked. A
/// file of lines rather than a pipe so that nothing blocks on either side, the
/// history is there to read if a job goes wrong, and a trainer written in any
/// language can produce it.
/// </para>
/// </summary>
public sealed record TrainerProgress
{
    /// <summary>What the trainer is doing: <c>reading</c>, <c>fitting</c>, <c>scoring</c>, <c>exporting</c>, <c>checking</c>.</summary>
    public string? Stage { get; init; }

    /// <summary>A line for a person, e.g. "7 models, 8412 rows".</summary>
    public string? Message { get; init; }

    /// <summary>True on the last line of a job that produced a model.</summary>
    public bool Done { get; init; }

    /// <summary>Set on the last line of a job that did not.</summary>
    public string? Error { get; init; }

    /// <summary>The model file written, on the last line.</summary>
    public string? Model { get; init; }

    /// <summary>The held-out score, by name, on the last line. The same map the model's metadata carries.</summary>
    public IReadOnlyDictionary<string, double>? Score { get; init; }

    /// <summary>Lines for a person on the last line — the confusion matrix, the residual summary, timings.</summary>
    public IReadOnlyList<string>? Report { get; init; }

    /// <summary>True once the job has ended, either way.</summary>
    [JsonIgnore]
    public bool Finished => Done || Error is not null;

    private static readonly JsonSerializerOptions Json = new()
    {
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        PropertyNameCaseInsensitive = true,
    };

    /// <summary>Reads one line. A line that is not this record is an error line, so a broken trainer is reported rather than ignored.</summary>
    public static TrainerProgress FromJsonLine(string line)
    {
        try
        {
            return JsonSerializer.Deserialize<TrainerProgress>(line, Json)
                ?? new TrainerProgress { Error = "The trainer wrote an empty progress line." };
        }
        catch (JsonException ex)
        {
            return new TrainerProgress { Error = $"The trainer wrote a progress line that could not be read: {ex.Message}" };
        }
    }

    /// <summary>The record as a line, for tests and for a trainer written in C#.</summary>
    public string ToJsonLine() => JsonSerializer.Serialize(this, Json);
}
