using System.Globalization;
using OtterLogic.Core;
using OtterLogic.MachineLearning.Decomposition;

namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// Principal components as a method on a wire: the samples projected onto the
/// directions they spread furthest along. The one method whose axes mean
/// something — each is a direction through the input columns, and the report
/// says which columns make it.
/// </summary>
public sealed record PrincipalComponentsMethod : EmbeddingMethod
{
    /// <summary>
    /// Scale every axis to the same spread. Off by default: the point of the map
    /// is usually that the first axis is the big one, and whitening hides that.
    /// On, the map is the one a mixture model would be fitted in.
    /// </summary>
    public bool Whiten { get; init; }

    public override string Name => "Principal Components";

    public override string Describe() => Whiten ? $"{Name}, whitened" : Name;

    public override void Validate(int sampleCount, int dimensions)
    {
        if (dimensions < 1)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "A map needs at least one dimension.");
    }

    public override EmbeddingOutcome Fit(EmbeddingQuery query, int dimensions)
    {
        ArgumentNullException.ThrowIfNull(query);
        var data = query.RequireData(Name);
        int n = data.GetLength(0);
        int d = data.GetLength(1);

        var pca = PrincipalComponents.FitCount(data, dimensions, Whiten);
        var projected = pca.Transform(data);
        var notes = new List<Note>();

        // FitCount clamps to the directions the data has, so a map asked for in more
        // dimensions than the samples vary in gets zero on the spare axes and says so.
        var coordinates = new double[n, dimensions];
        var axes = new double[dimensions, d];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < pca.Count; c++)
                coordinates[i, c] = projected[i, c];
        for (int c = 0; c < pca.Count; c++)
            for (int j = 0; j < d; j++)
                axes[c, j] = pca.Components[c, j];

        if (pca.Count < dimensions)
            notes.Add(Note.Remark(
                $"The samples vary in only {Count(pca.Count, "direction")}, so axis "
                + $"{string.Join(", ", Enumerable.Range(pca.Count + 1, dimensions - pca.Count))} of {dimensions} carries nothing."));

        double kept = pca.ExplainedVariance.Sum();
        var shares = pca.ExplainedVariance.Select(v => kept > 0.0 ? pca.ExplainedVarianceRatio * v / kept : 0.0).ToArray();
        var invariant = CultureInfo.InvariantCulture;

        return new EmbeddingOutcome(Name, coordinates, axes, pca.ExplainedVarianceRatio, distortion: null, notes,
            details: new[]
            {
                "Each axis is a direction through the input columns, largest spread first: "
                + string.Join(", ", shares.Select((s, c) => $"axis {c + 1} carries {(100.0 * s).ToString("0", invariant)}%"))
                + " of the variation.",
            });
    }
}
