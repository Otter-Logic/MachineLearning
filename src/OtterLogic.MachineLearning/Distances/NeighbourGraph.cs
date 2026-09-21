using OtterLogic.Graphs;

namespace OtterLogic.MachineLearning.Distances;

/// <summary>
/// The k-nearest-neighbour graph of a sample matrix — the way into the graph
/// methods for a caller that has only a point cloud.
/// <para>
/// It was <c>WeightedGraph.NearestNeighbours</c> until the graph contract moved
/// down into Graphs. It stayed behind because it is the one constructor that
/// measures a distance between samples, and Graphs references nothing: a graph
/// there is nodes and edges, and what a sample is, or how far apart two of them
/// are, is this layer's business. A second nearest-neighbour search written down
/// there to avoid the reference would be the seventh copy
/// <see cref="Euclidean"/> exists to prevent.
/// </para>
/// </summary>
public static class NeighbourGraph
{
    /// <summary>
    /// The k-nearest-neighbour graph of a point cloud, by Euclidean distance.
    /// <para>
    /// Nearest-neighbour relations are not symmetric — a point on the edge of a
    /// cluster may count a core point among its nearest while the core point does
    /// not return the favour. The two directions are averaged: an edge both ends
    /// chose weighs one, an edge only one end chose weighs a half. That is
    /// scikit-learn's symmetrisation, and it keeps the information a plain union
    /// throws away, that mutual neighbours are the stronger evidence.
    /// </para>
    /// <para>
    /// The neighbours come from <see cref="Euclidean.Nearest"/>: O(n^2 d) time and
    /// O(n k) memory, ties to the lower index, so the graph is a function of the
    /// data alone.
    /// </para>
    /// <para>
    /// The halves are summed here and each edge handed over once.
    /// <see cref="WeightedGraph.FromEdges(int, IEnumerable{ValueTuple{int, int, double}})"/>
    /// reads a repeated edge as one relationship reported twice and keeps the
    /// larger weight, which would turn every mutual pair back into a half.
    /// </para>
    /// </summary>
    /// <param name="x">n x d data, rows are samples.</param>
    /// <param name="neighbours">Neighbours per point, not counting itself. Between 1 and n - 1.</param>
    public static WeightedGraph Of(double[,] x, int neighbours)
    {
        ArgumentNullException.ThrowIfNull(x);

        int n = x.GetLength(0);

        if (neighbours < 1 || neighbours > n - 1)
            throw new ArgumentOutOfRangeException(nameof(neighbours), neighbours,
                $"Need between 1 and {n - 1} neighbours for {n} samples.");

        int k = neighbours;
        var chosen = Euclidean.Nearest(x, k).Index;

        var unique = new Dictionary<long, double>(n * k);
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                int j = chosen[i, c];
                long key = (long)Math.Min(i, j) * n + Math.Max(i, j);
                unique[key] = unique.TryGetValue(key, out double half) ? half + 0.5 : 0.5;
            }
        }

        return WeightedGraph.FromEdges(n,
            unique.Select(edge => ((int)(edge.Key / n), (int)(edge.Key % n), edge.Value)));
    }

    /// <summary>
    /// The same neighbours, with each edge weighing the distance it spans — a graph
    /// to travel over rather than one to cluster on.
    /// <para>
    /// <see cref="Of"/> weighs an edge by how mutual the choice was, which is a
    /// similarity: right for spectral clustering, and exactly backwards for a route,
    /// where a high weight has to mean far. This one is for the caller about to
    /// hand the graph to a shortest-path search. An edge is kept if either end
    /// chose the other, since a road does not stop being a road because only one
    /// end thinks of it as near.
    /// </para>
    /// <para>
    /// What may block an edge is the caller's to say and is asked once per pair,
    /// lower index first. This layer measures distances and has no idea what an
    /// obstacle is; Graphs' planar obstacles are the usual answer, passed in as a
    /// predicate so that neither layer has to know the other's types. A blocked
    /// edge still used up one of its ends' <paramref name="neighbours"/>, so among
    /// dense obstacles ask for a few more than would otherwise be needed.
    /// </para>
    /// <para>
    /// Two rows in the same place are zero apart, and a zero-weight edge is dropped
    /// by <see cref="WeightedGraph"/>, so coincident points come out unconnected to
    /// each other. They are one place; the caller should pass it once.
    /// </para>
    /// </summary>
    /// <param name="x">n x d positions, rows are nodes.</param>
    /// <param name="neighbours">Neighbours each node proposes, not counting itself. Between 1 and n - 1.</param>
    /// <param name="maximumDistance">Edges longer than this are left out. Infinity keeps every one.</param>
    /// <param name="blocked">Given two node indices, lower first, whether the edge between them is unusable. Null blocks nothing.</param>
    public static WeightedGraph ByDistance(
        double[,] x, int neighbours,
        double maximumDistance = double.PositiveInfinity, Func<int, int, bool>? blocked = null)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (double.IsNaN(maximumDistance) || maximumDistance <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(maximumDistance), maximumDistance,
                "The maximum distance must be above zero.");

        int n = x.GetLength(0);
        var (index, distance) = Euclidean.Nearest(x, neighbours);

        var unique = new Dictionary<long, double>(n * neighbours);
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < neighbours; c++)
            {
                int j = index[i, c];
                if (distance[i, c] > maximumDistance)
                    break;   // nearest first, so every later one is further still

                int a = Math.Min(i, j), b = Math.Max(i, j);
                long key = (long)a * n + b;
                if (unique.ContainsKey(key))
                    continue;

                // NaN marks a pair already found blocked, so the caller is asked once.
                unique[key] = blocked is not null && blocked(a, b) ? double.NaN : distance[i, c];
            }
        }

        return WeightedGraph.FromEdges(n, unique
            .Where(edge => !double.IsNaN(edge.Value))
            .Select(edge => ((int)(edge.Key / n), (int)(edge.Key % n), edge.Value)));
    }
}
