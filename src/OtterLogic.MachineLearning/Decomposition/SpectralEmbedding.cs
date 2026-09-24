using OtterLogic.Graphs;

namespace OtterLogic.MachineLearning.Decomposition;

/// <summary>
/// A graph laid out by its leading Laplacian eigenvectors, and what the spectrum says
/// about it.
/// </summary>
public sealed class SpectralEmbeddingResult
{
    internal SpectralEmbeddingResult(
        double[,] coordinates, int[] placed, double[] eigenvalues, int graphComponents, int iterations, bool converged)
    {
        Coordinates = coordinates;
        Placed = placed;
        Eigenvalues = eigenvalues;
        GraphComponents = graphComponents;
        Iterations = iterations;
        Converged = converged;
    }

    /// <summary>
    /// n x k: one row per node, in which nodes of one well-connected region sit
    /// close together whatever shape the region has anywhere else. Rows of nodes
    /// with no connections are zero.
    /// <para>
    /// These are the random-walk Laplacian's eigenvectors, as scikit-learn uses:
    /// the symmetric Laplacian's divided by the square root of degree. Without that
    /// a low-degree node on a region's fringe sits nearer the origin than its region
    /// does, and anything reading the map files it with the wrong one.
    /// </para>
    /// </summary>
    public double[,] Coordinates { get; }

    /// <summary>The nodes that have connections and so a place on the map, ascending.</summary>
    public int[] Placed { get; }

    /// <summary>Number of nodes, placed or not.</summary>
    public int NodeCount => Coordinates.GetLength(0);

    /// <summary>Number of coordinates per node.</summary>
    public int Dimensions => Coordinates.GetLength(1);

    /// <summary>
    /// Smallest eigenvalues of the normalised graph Laplacian, ascending — k of them,
    /// and one more when there are nodes to spare. Between 0 and 2. Zero repeats once
    /// per connected piece; a large gap after the k-th says the graph has k regions.
    /// </summary>
    public double[] Eigenvalues { get; }

    /// <summary>The gap between the k-th eigenvalue and the next, or NaN when there was no next.</summary>
    public double EigenGap => Eigenvalues.Length > Dimensions ? Eigenvalues[Dimensions] - Eigenvalues[Dimensions - 1] : double.NaN;

    /// <summary>Connected pieces among the placed nodes.</summary>
    public int GraphComponents { get; }

    /// <summary>Eigensolver iterations.</summary>
    public int Iterations { get; }

    /// <summary>Whether the eigensolver settled before its iteration cap.</summary>
    public bool Converged { get; }
}

/// <summary>
/// Lays a graph out by the leading eigenvectors of its normalised Laplacian: nodes
/// joined by a path of strong connections land close together, whatever the
/// distances between them anywhere else.
/// <para>
/// The embedding half of spectral clustering, on its own. Unsupervised's
/// <c>SpectralClustering</c> partitions this map with k-means; a component that
/// only wants to <em>see</em> the map, a trained graph model that wants it as an
/// input, and a toolkit laying out a network all want the same coordinates with
/// no partition, and each computing its own would be three copies of the one
/// numerical step that is easy to get subtly wrong. It sits here, one layer down
/// from the clustering, because it is a decomposition and learns nothing.
/// </para>
/// <para>
/// Nodes with no connections are set aside before the eigenproblem rather than
/// after. Each contributes an empty row, and so an eigenvector of its own at an
/// eigenvalue unrelated to any region; leaving them in only gives the solver more
/// directions to wade through. They come back as zero rows, and
/// <see cref="SpectralEmbeddingResult.Placed"/> says which rows are real.
/// </para>
/// </summary>
public static class SpectralEmbedding
{
    /// <summary>
    /// Embeds <paramref name="affinity"/> in <paramref name="dimensions"/> coordinates.
    /// </summary>
    /// <param name="affinity">Which nodes are related, and how strongly. Weights are similarities: larger is closer.</param>
    /// <param name="dimensions">Coordinates per node, at least one and no more than the nodes with connections.</param>
    /// <param name="seed">Seed for the eigensolver's start. Fixed, so the same graph gives the same map.</param>
    /// <param name="tolerance">Residual tolerance handed to the eigensolver.</param>
    /// <exception cref="ArgumentException">Too few nodes have connections for that many dimensions.</exception>
    public static SpectralEmbeddingResult Of(WeightedGraph affinity, int dimensions, int seed = 1, double tolerance = 1e-8)
    {
        ArgumentNullException.ThrowIfNull(affinity);
        if (dimensions < 1)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "An embedding needs at least one dimension.");

        int n = affinity.NodeCount;
        int k = dimensions;

        var placed = Enumerable.Range(0, n).Where(i => affinity.Degree(i) > 0.0).ToArray();
        int m = placed.Length;

        if (m < k)
            throw new ArgumentException(
                $"Only {m} of {n} nodes have any connections, too few for a {k}-dimensional embedding.", nameof(affinity));

        var graph = m == n ? affinity : Subgraph(affinity, placed);
        graph.ConnectedComponents(out int components);

        // The normalised adjacency has eigenvalues in [-1, 1]; the ones wanted are
        // the largest. Shifting by the identity and halving maps that onto [0, 1]
        // without moving any eigenvector, and makes the operator positive
        // semi-definite — so largest in value and largest in magnitude agree, and
        // a strongly bipartite piece of graph, with eigenvalues near -1, cannot
        // crowd out the ones near +1.
        double[,] Multiply(double[,] block)
        {
            var y = graph.Propagate(block, selfWeight: 0.0);
            for (int i = 0; i < y.GetLength(0); i++)
                for (int c = 0; c < y.GetLength(1); c++)
                    y[i, c] = 0.5 * (y[i, c] + block[i, c]);

            return y;
        }

        int wanted = Math.Min(k + 1, m);
        var (values, vectors, iterations, converged) = LeadingEigen.Solve(m, Multiply, wanted, seed, tolerance);

        // Back from the shifted adjacency to the Laplacian: L = I - A_norm, and
        // the shift made each value (1 + a) / 2, so the Laplacian's is 2 - 2v.
        var eigenvalues = values.Select(v => Math.Clamp(2.0 - 2.0 * v, 0.0, 2.0)).ToArray();

        var coordinates = new double[n, k];
        for (int i = 0; i < m; i++)
        {
            double scale = 1.0 / Math.Sqrt(graph.Degree(i));
            for (int c = 0; c < k; c++)
                coordinates[placed[i], c] = vectors[i, c] * scale;
        }

        return new SpectralEmbeddingResult(coordinates, placed, eigenvalues, components, iterations, converged);
    }

    /// <summary>The graph restricted to <paramref name="nodes"/>, renumbered 0..m-1 in the same order.</summary>
    private static WeightedGraph Subgraph(WeightedGraph graph, int[] nodes)
    {
        var index = new Dictionary<int, int>(nodes.Length);
        for (int i = 0; i < nodes.Length; i++)
            index[nodes[i]] = i;

        var edges = graph.Edges()
            .Where(e => index.ContainsKey(e.A) && index.ContainsKey(e.B))
            .Select(e => (index[e.A], index[e.B], e.Weight));

        return WeightedGraph.FromEdges(nodes.Length, edges);
    }
}
