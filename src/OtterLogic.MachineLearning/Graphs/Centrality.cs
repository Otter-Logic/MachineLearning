namespace OtterLogic.MachineLearning.Graphs;

/// <summary>
/// How much of a <see cref="WeightedGraph"/>'s traffic passes through each node.
/// <para>
/// Here beside <see cref="ShortestPaths"/> for the same reason: it is a reading of
/// a graph, not a method that learns anything, and a clustering that wants it as a
/// feature and a trained graph network that wants it as an input feature must
/// compute it the same way.
/// </para>
/// </summary>
public static class Centrality
{
    /// <summary>
    /// Betweenness centrality by hop count, normalised to 0..1: the share of
    /// shortest routes between every other pair of nodes that pass through each
    /// node.
    /// <para>
    /// Brandes (2001), counting hops rather than reading weights. A weight here
    /// says how strongly two nodes are related, which is the opposite of a route
    /// cost, and turning one into the other is a judgement this layer has no basis
    /// for. Hops are the reading every caller can agree on.
    /// </para>
    /// <para>
    /// Exact costs one breadth-first search per node, O(n e). Past
    /// <paramref name="maximumSources"/> nodes it starts from that many sources
    /// spread evenly through the node order and scales the counts up — the
    /// estimate Brandes and Pich (2007) describe — because a few thousand searches
    /// is where a Grasshopper re-solve stops feeling live. Evenly spread rather
    /// than random, so the same graph gives the same answer on every solve.
    /// </para>
    /// </summary>
    /// <param name="graph">The graph. Weights are ignored.</param>
    /// <param name="maximumSources">Searches to run at most; every node when the graph is no larger.</param>
    public static double[] Betweenness(WeightedGraph graph, int maximumSources = 2000)
    {
        ArgumentNullException.ThrowIfNull(graph);
        if (maximumSources < 1)
            throw new ArgumentOutOfRangeException(nameof(maximumSources), maximumSources, "Need at least one source.");

        int n = graph.NodeCount;
        var centrality = new double[n];
        if (n < 3)
            return centrality;

        int sources = Math.Min(n, maximumSources);
        var distance = new int[n];
        var paths = new double[n];
        var dependency = new double[n];
        var order = new int[n];
        var queue = new int[n];

        for (int s = 0; s < sources; s++)
        {
            int source = sources == n ? s : (int)((long)s * n / sources);

            Array.Fill(distance, -1);
            Array.Clear(paths);
            Array.Clear(dependency);

            distance[source] = 0;
            paths[source] = 1.0;
            int head = 0, tail = 0, settled = 0;
            queue[tail++] = source;

            while (head < tail)
            {
                int node = queue[head++];
                order[settled++] = node;

                foreach (int next in graph.Neighbours(node))
                {
                    if (distance[next] < 0)
                    {
                        distance[next] = distance[node] + 1;
                        queue[tail++] = next;
                    }

                    if (distance[next] == distance[node] + 1)
                        paths[next] += paths[node];
                }
            }

            // Back from the furthest node: each hands its share on to the nodes
            // one hop nearer the source, in proportion to the routes through them.
            for (int k = settled - 1; k > 0; k--)
            {
                int node = order[k];
                foreach (int previous in graph.Neighbours(node))
                    if (distance[previous] == distance[node] - 1)
                        dependency[previous] += paths[previous] / paths[node] * (1.0 + dependency[node]);

                centrality[node] += dependency[node];
            }
        }

        // Every unordered pair is counted from both ends when every node is a
        // source, hence the half; a sample scales up to what all n would give.
        double scale = 0.5 * n / sources / ((n - 1) * (n - 2) / 2.0);
        for (int i = 0; i < n; i++)
            centrality[i] *= scale;

        return centrality;
    }
}
