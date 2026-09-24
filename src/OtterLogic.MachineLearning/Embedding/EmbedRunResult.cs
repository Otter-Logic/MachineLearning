using System.Globalization;
using OtterLogic.Core;

namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// A map of samples, with the readings a person needs to trust it.
/// </summary>
public sealed class EmbedRunResult
{
    internal EmbedRunResult(EmbeddingOutcome outcome, bool standardised, int[] keptColumns, IReadOnlyList<Note> notes)
    {
        Outcome = outcome;
        Standardised = standardised;
        KeptColumns = keptColumns;
        Notes = notes;
    }

    /// <summary>What the method itself returned, with its axes over every input column.</summary>
    public EmbeddingOutcome Outcome { get; }

    /// <summary>The method's name, or for an automatic choice the name of the one chosen.</summary>
    public string Method => Outcome.Method;

    /// <summary>n x k: each sample's place on the map.</summary>
    public double[,] Coordinates => Outcome.Coordinates;

    /// <summary>k x d: each axis as a direction through the input columns; null when the axes mean nothing of their own.</summary>
    public double[,]? Axes => Outcome.Axes;

    /// <summary>Share of the samples' structure the map carries, 0 to 1; NaN when the method cannot say.</summary>
    public double Retained => Outcome.Retained;

    /// <summary>Per sample, how badly the map places it; null when the method cannot measure that.</summary>
    public double[]? Distortion => Outcome.Distortion;

    public int SampleCount => Outcome.SampleCount;

    public int Dimensions => Outcome.Dimensions;

    /// <summary>Whether the columns were brought to a common scale before mapping.</summary>
    public bool Standardised { get; }

    /// <summary>The columns the map used, in order. Every column when nothing was dropped; empty when only a graph was wired.</summary>
    public int[] KeptColumns { get; }

    /// <summary>Everything a user should hear about this run, in the order it came up.</summary>
    public IReadOnlyList<Note> Notes { get; }

    /// <summary>
    /// The run in words, one line per fact, for a report a person reads without
    /// wiring anything else: what ran and why, how much of the structure the map
    /// keeps, and what each axis is made of when it is made of anything.
    /// </summary>
    /// <param name="featureNames">One per input column, for the axes; unnamed columns are called Feature 0, Feature 1, ...</param>
    public IReadOnlyList<string> Report(IReadOnlyList<string>? featureNames = null)
    {
        var invariant = CultureInfo.InvariantCulture;
        var lines = new List<string>();

        string headline = $"{Method}: {SampleCount} samples on {Dimensions} {(Dimensions == 1 ? "axis" : "axes")}";
        if (Outcome.Rationale is { } rationale)
            headline += " — chosen because " + rationale;
        lines.Add(headline + (headline.EndsWith('.') ? "" : "."));

        if (!double.IsNaN(Retained))
            lines.Add($"The map carries {(100.0 * Retained).ToString("0", invariant)}% of the samples' structure.");

        lines.AddRange(Outcome.Details);

        if (KeptColumns.Length > 0)
            lines.Add(Standardised
                ? "Columns were brought to a common scale before mapping."
                : "Columns were used as they arrived; the largest units carry the most weight.");

        if (Axes is { } axes)
        {
            string NameOf(int j) => featureNames is not null && j < featureNames.Count ? featureNames[j] : $"Feature {j}";
            int d = axes.GetLength(1);
            for (int c = 0; c < axes.GetLength(0); c++)
            {
                // The few columns that make the axis, largest loading first, so a
                // person can name the axis without reading every number.
                var loadings = Enumerable.Range(0, d)
                    .Where(j => Math.Abs(axes[c, j]) > 1e-9)
                    .OrderByDescending(j => Math.Abs(axes[c, j]))
                    .Take(3)
                    .Select(j => $"{axes[c, j].ToString("+0.00;-0.00", invariant)} {NameOf(j)}");
                lines.Add($"Axis {c + 1}: {(loadings.Any() ? string.Join(", ", loadings) : "nothing")}.");
            }
        }

        foreach (var note in Notes)
            lines.Add(note.Level == NoteLevel.Warning ? "Warning: " + note.Text : note.Text);

        return lines;
    }
}
