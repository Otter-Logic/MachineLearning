using System.Globalization;
using OtterLogic.Core;
using OtterLogic.Graphs;
using OtterLogic.MachineLearning.Decomposition;
using OtterLogic.MachineLearning.Distances;

namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// A spectral embedding as a method on a wire: samples laid out by what they
/// connect to rather than by how far apart their values are. Given a graph it
/// lays out the graph; given values alone it links each sample to its nearest
/// few and lays out that.
/// </summary>
public sealed record SpectralEmbeddingMethod : EmbeddingMethod
{
    /// <summary>
    /// How many of its nearest samples each sample is linked to when the graph has
    /// to be built from values. Ten by default, scikit-learn's; ignored when a
    /// graph is wired.
    /// </summary>
    public int Neighbours { get; init; } = 10;

    /// <summary>Fixed, so the same samples give the same map every solve.</summary>
    public int Seed { get; init; } = 1;

    public override string Name => "Spectral Embedding";

    public override string Describe() => $"{Name}, {Neighbours} neighbours when built from values";

    public override void Validate(int sampleCount, int dimensions)
    {
        if (Neighbours < 1)
            throw new ArgumentOutOfRangeException(nameof(Neighbours), Neighbours, "Need at least one neighbour.");
        if (dimensions < 1)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "A map needs at least one dimension.");
        if (dimensions >= sampleCount)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions,
                $"{sampleCount} samples span at most {sampleCount - 1} dimension(s).");
    }

    public override EmbeddingOutcome Fit(EmbeddingQuery query, int dimensions)
    {
        ArgumentNullException.ThrowIfNull(query);
        int n = query.SampleCount;
        var details = new List<string>();

        // A wired graph is read as which nodes are joined and nothing more. Its
        // weights are whatever the builder gave them — lengths from Graph From Lines,
        // costs from a route — and a length read as a strength would pull the
        // furthest-apart nodes closest together. A graph built here from values has
        // the mutual-neighbour weights the clustering uses, which are strengths.
        WeightedGraph graph;
        if (query.Graph is { } wired)
        {
            graph = wired.Reweight((_, _, _) => 1.0);
            details.Add("Connections were read as present or absent; their weights played no part.");
        }
        else
        {
            int k = Math.Min(Neighbours, n - 1);
            graph = NeighbourGraph.Of(query.Data!, k);
            details.Add($"Each sample was linked to its nearest {k} by its values, and the links laid out.");
        }

        var spectral = SpectralEmbedding.Of(graph, dimensions, Seed);
        var notes = new List<Note>();
        var invariant = CultureInfo.InvariantCulture;

        int unplaced = n - spectral.Placed.Length;
        if (unplaced > 0)
            notes.Add(Note.Warning(
                $"{Count(unplaced, "sample")} connect to nothing and sit at the origin; they belong nowhere on this map."));

        if (spectral.GraphComponents > 1)
            notes.Add(Note.Remark(
                $"The graph falls into {spectral.GraphComponents} separate pieces. Where each piece sits relative to the "
                + "others on the map means nothing; only positions within a piece do."));

        if (!spectral.Converged)
            notes.Add(Note.Warning("The eigensolver did not settle; the map may be rough."));

        details.Add("Laplacian eigenvalues: "
            + string.Join(", ", spectral.Eigenvalues.Select(v => v.ToString("0.###", invariant)))
            + (double.IsNaN(spectral.EigenGap)
                ? "."
                : $". A large gap after the {Ordinal(dimensions)} says the graph has about {dimensions} regions."));

        return new EmbeddingOutcome(Name, spectral.Coordinates, axes: null, retained: double.NaN, distortion: null, notes, details);
    }

    private static string Ordinal(int n) => n switch
    {
        1 => "first",
        2 => "second",
        3 => "third",
        _ => $"{n}th",
    };
}
