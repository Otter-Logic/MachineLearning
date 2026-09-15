namespace OtterLogic.MachineLearning.Graphs;

/// <summary>
/// Least-cost routes from a set of source nodes to every node of a
/// <see cref="WeightedGraph"/>: the cost of the cheapest route, which source it
/// starts from, and the node before this one on it.
/// </summary>
/// <param name="Cost">Cost of the cheapest route to each node; positive infinity where no route exists.</param>
/// <param name="Source">The source the cheapest route to each node starts from; -1 where no route exists.</param>
/// <param name="Previous">The node before each node on its cheapest route; -1 at a source and where no route exists.</param>
public sealed record ShortestPathsResult(double[] Cost, int[] Source, int[] Previous)
{
    /// <summary>Whether any source reaches <paramref name="node"/>.</summary>
    public bool Reaches(int node) => Source[node] >= 0;

    /// <summary>
    /// The cheapest route to <paramref name="node"/>, from its source to the node
    /// itself. Empty where no route exists.
    /// </summary>
    public int[] RouteTo(int node)
    {
        if (!Reaches(node))
            return Array.Empty<int>();

        var route = new List<int>();
        for (int at = node; at >= 0; at = Previous[at])
            route.Add(at);

        route.Reverse();
        return route.ToArray();
    }
}

/// <summary>
/// Multi-source Dijkstra over a <see cref="WeightedGraph"/>.
/// <para>
/// The graph's weights say how strongly two nodes are related; a route's cost is
/// a different question, so the caller supplies it per edge rather than having it
/// read off the weight. One graph then serves both: a clustering reads its
/// similarities, and a route search reads whatever cost the caller's domain
/// gives the same edge — length, a penalty for climbing, anything non-negative.
/// </para>
/// <para>
/// Ties go to the lower-numbered source, then the lower-numbered previous node,
/// so the routes are a property of the graph and the costs alone, never of the
/// order the queue happened to pop equal costs in. That matters for the same
/// reason every other graph algorithm here sorts its neighbours: an adaptor
/// re-solves on every upstream change, and a route that flips between two equal
/// choices from one solve to the next is unusable.
/// </para>
/// </summary>
public static class ShortestPaths
{
    /// <summary>
    /// Cheapest routes from every node in <paramref name="sources"/> to every node.
    /// </summary>
    /// <param name="graph">The graph to route over. Edges are traversable both ways.</param>
    /// <param name="sources">Nodes a route may start from, each at cost zero. Duplicates are ignored.</param>
    /// <param name="cost">
    /// Cost of moving from <c>from</c> to <c>to</c> along an edge of weight <c>weight</c>:
    /// finite and not negative, or positive infinity for an edge that may not be used in
    /// that direction. Null for the weight itself.
    /// </param>
    public static ShortestPathsResult From(
        WeightedGraph graph, IEnumerable<int> sources, Func<int, int, double, double>? cost = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(sources);

        int n = graph.NodeCount;
        var best = new double[n];
        var source = new int[n];
        var previous = new int[n];
        Array.Fill(best, double.PositiveInfinity);
        Array.Fill(source, -1);
        Array.Fill(previous, -1);

        var queue = new PriorityQueue<int, (double Cost, int Source, int Node)>();

        foreach (int s in sources.Distinct().OrderBy(s => s))
        {
            if (s < 0 || s >= n)
                throw new ArgumentOutOfRangeException(nameof(sources), s, $"Source {s} is outside 0..{n - 1}.");

            best[s] = 0.0;
            source[s] = s;
            queue.Enqueue(s, (0.0, s, s));
        }

        var settled = new bool[n];
        while (queue.TryDequeue(out int node, out var key))
        {
            if (settled[node] || key.Cost > best[node] || key.Source != source[node])
                continue;

            settled[node] = true;
            var neighbours = graph.Neighbours(node);
            var weights = graph.EdgeWeights(node);

            for (int e = 0; e < neighbours.Length; e++)
            {
                int next = neighbours[e];
                if (settled[next])
                    continue;

                double step = cost is null ? weights[e] : cost(node, next, weights[e]);
                if (double.IsNaN(step) || step < 0.0 || double.IsNegativeInfinity(step))
                    throw new InvalidOperationException(
                        $"The cost of edge ({node}, {next}) is {step}; costs must be non-negative, or positive infinity to forbid it.");
                if (double.IsPositiveInfinity(step))
                    continue;

                double through = best[node] + step;
                bool better = through < best[next]
                    || (through == best[next] && (source[node] < source[next]
                        || (source[node] == source[next] && node < previous[next])));

                if (!better)
                    continue;

                best[next] = through;
                source[next] = source[node];
                previous[next] = node;
                queue.Enqueue(next, (through, source[node], next));
            }
        }

        return new ShortestPathsResult(best, source, previous);
    }
}
