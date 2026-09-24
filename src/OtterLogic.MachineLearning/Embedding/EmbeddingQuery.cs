using OtterLogic.Graphs;

namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// Everything an embedding method is handed: the samples' values, the graph
/// between them, or both.
/// <para>
/// The data side of one wire. A method reads what it needs and says in words
/// what is missing when it is not there: principal components and
/// multidimensional scaling need values, a spectral embedding takes a graph and
/// builds one from values when none is wired. One type for every method, so a
/// method's signature does not force a user to know which it will want.
/// </para>
/// </summary>
public sealed class EmbeddingQuery
{
    /// <param name="data">n x d, one row per sample, or null when only a graph is given.</param>
    /// <param name="graph">One node per sample, or null when only values are given.</param>
    /// <exception cref="ArgumentException">Neither was given, the values are not finite, or the two disagree about how many samples there are.</exception>
    public EmbeddingQuery(double[,]? data, WeightedGraph? graph)
    {
        if (data is null && graph is null)
            throw new ArgumentException("Wire Data, a Graph, or both: there is nothing to lay out.");

        if (data is not null)
        {
            int n = data.GetLength(0);
            int d = data.GetLength(1);
            if (n < 2)
                throw new ArgumentException($"Need at least two samples to map; got {n}.", nameof(data));
            if (d < 1)
                throw new ArgumentException("The samples have no values.", nameof(data));

            for (int i = 0; i < n; i++)
                for (int j = 0; j < d; j++)
                    if (!double.IsFinite(data[i, j]))
                        throw new ArgumentException($"Sample {i} holds {data[i, j]} at position {j}; values must be finite.", nameof(data));

            if (graph is not null && graph.NodeCount != n)
                throw new ArgumentException(
                    $"Data has {n} samples but the Graph has {graph.NodeCount} nodes. One node per sample, in the same order.", nameof(graph));
        }
        else if (graph!.NodeCount < 2)
        {
            throw new ArgumentException($"Need at least two nodes to map; got {graph.NodeCount}.", nameof(graph));
        }

        Data = data;
        Graph = graph;
    }

    /// <summary>One row per sample, or null when only a graph was given.</summary>
    public double[,]? Data { get; }

    /// <summary>One node per sample, or null when only values were given.</summary>
    public WeightedGraph? Graph { get; }

    public bool HasData => Data is not null;

    public bool HasGraph => Graph is not null;

    /// <summary>How many samples there are to place.</summary>
    public int SampleCount => Data?.GetLength(0) ?? Graph!.NodeCount;

    /// <summary>The same graph over different values — what standardising the columns produces.</summary>
    public EmbeddingQuery WithData(double[,] data) => new(data, Graph);

    /// <summary>The values, or a sentence naming what to wire when there are none.</summary>
    public double[,] RequireData(string method)
        => Data ?? throw new ArgumentException(
            $"{method} places samples by their values and needs Data: one branch per sample. "
            + "With only a Graph wired, use Spectral Embedding, which lays nodes out by what they connect to.");
}
