using System.Globalization;
using OtterLogic.Core;
using OtterLogic.MachineLearning.Decomposition;

namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// Multidimensional scaling as a method on a wire: the samples placed so that the
/// distances between the points match the distances between the samples as
/// closely as a few dimensions allow, with a stress that says how closely.
/// </summary>
public sealed record MultidimensionalScalingMethod : EmbeddingMethod
{
    /// <summary>
    /// Improve the closed-form map against the distances themselves. On by
    /// default; it never makes the map worse, and off is only for speed on
    /// thousands of samples.
    /// </summary>
    public bool Refine { get; init; } = true;

    public override string Name => "Multidimensional Scaling";

    public override string Describe() => Refine ? Name : $"{Name}, unrefined";

    public override void Validate(int sampleCount, int dimensions)
        => new MultidimensionalScalingOptions { Dimensions = dimensions }.Validate(sampleCount);

    public override EmbeddingOutcome Fit(EmbeddingQuery query, int dimensions)
    {
        ArgumentNullException.ThrowIfNull(query);
        var data = query.RequireData(Name);

        var scaling = MultidimensionalScaling.FromFeatures(data, new MultidimensionalScalingOptions
        {
            Dimensions = dimensions,
            Refine = Refine,
        });

        var invariant = CultureInfo.InvariantCulture;
        string reading = scaling.Stress < 0.05 ? "the map is faithful"
            : scaling.Stress < 0.1 ? "the map is good"
            : scaling.Stress < 0.2 ? "the map is rough; distances on it are indicative"
            : "the samples need more dimensions than this to lay out; trust neighbours, not distances";

        return new EmbeddingOutcome(Name, scaling.Coordinates, axes: null, scaling.Retained, scaling.Distortion,
            notes: scaling.Notes.Select(Note.Remark).ToArray(),
            details: new[]
            {
                $"Stress {scaling.Stress.ToString("0.00", invariant)}: {reading}. Zero would keep every distance exactly. "
                + "Distortion says, per sample, which points not to trust.",
                "The axes have no meaning of their own: only the distances between points do. Turn or mirror the map "
                + "and it says the same thing.",
            });
    }
}
