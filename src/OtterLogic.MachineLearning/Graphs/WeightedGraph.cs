namespace OtterLogic.MachineLearning.Graphs;

/// <summary>
/// An undirected graph over n samples, with a non-negative weight on every edge.
/// <para>
/// It lives here rather than in a paradigm repo because it is a data contract, not
/// an algorithm. Spectral clustering, connectivity-constrained hierarchical
/// clustering and message passing in Unsupervised all consume it, and a trained
/// graph network in DeepLearning will want exactly the same thing on its input
/// side — the same edges, the same normalisation. One type below all of them is
/// what stops four slightly different adjacency lists appearing.
/// </para>
/// <para>
/// Where the edges come from is the caller's business and is deliberately not
/// modelled. A toolkit that knows which of its elements touch builds the graph
/// from that; a caller with only a point cloud uses
/// <see cref="NearestNeighbours"/>. The algorithms above cannot tell the
/// difference, which is the point.
/// </para>
/// <para>
/// Stored as compressed sparse rows with every edge held in both directions, and
/// neighbours sorted by index. The sort is not tidiness: it fixes the order every
/// sum over a neighbourhood is taken in, so two graphs built from the same edges
/// in a different order produce bit-identical results downstream.
/// </para>
/// </summary>
public sealed class WeightedGraph
{
    private readonly int[] _offsets;
    private readonly int[] _targets;
    private readonly double[] _weights;

    private WeightedGraph(int nodeCount, int[] offsets, int[] targets, double[] weights)
    {
        NodeCount = nodeCount;
        _offsets = offsets;
        _targets = targets;
        _weights = weights;
    }

    /// <summary>Number of nodes — one per sample.</summary>
    public int NodeCount { get; }

    /// <summary>Number of undirected edges, each counted once.</summary>
    public int EdgeCount => _targets.Length / 2;

    /// <summary>Builds a graph from unweighted edges; every edge gets weight one.</summary>
    /// <param name="nodeCount">Number of nodes. Nodes with no edge are allowed and stay isolated.</param>
    /// <param name="edges">Pairs of node indices, in either orientation.</param>
    public static WeightedGraph FromEdges(int nodeCount, IEnumerable<(int A, int B)> edges)
    {
        ArgumentNullException.ThrowIfNull(edges);
        return FromEdges(nodeCount, edges.Select(e => (e.A, e.B, 1.0)));
    }

    /// <summary>
    /// Builds a graph from weighted edges.
    /// <para>
    /// Three things are normalised away rather than rejected, because each is a
    /// routine way for real input to arrive. An edge listed twice — once from each
    /// end is the usual cause — keeps the larger weight rather than summing, since
    /// it is one relationship reported twice, not two. A self-loop is dropped: the
    /// algorithms that want one add their own at a weight they choose, and in a
    /// Laplacian it contributes nothing but degree. A zero weight is dropped, as
    /// it would otherwise count as a neighbour that carries nothing.
    /// </para>
    /// </summary>
    /// <param name="nodeCount">Number of nodes. Nodes with no edge are allowed and stay isolated.</param>
    /// <param name="edges">Node index pairs with a finite, non-negative weight each.</param>
    public static WeightedGraph FromEdges(int nodeCount, IEnumerable<(int A, int B, double Weight)> edges)
    {
        ArgumentNullException.ThrowIfNull(edges);
        if (nodeCount < 1)
            throw new ArgumentOutOfRangeException(nameof(nodeCount), nodeCount, "A graph needs at least one node.");

        var unique = new Dictionary<long, double>();

        foreach (var (a, b, weight) in edges)
        {
            if (a < 0 || a >= nodeCount || b < 0 || b >= nodeCount)
                throw new ArgumentOutOfRangeException(nameof(edges),
                    $"Edge ({a}, {b}) refers to a node outside 0..{nodeCount - 1}.");
            if (double.IsNaN(weight) || double.IsInfinity(weight) || weight < 0.0)
                throw new ArgumentOutOfRangeException(nameof(edges),
                    $"Edge ({a}, {b}) has weight {weight}; weights must be finite and non-negative.");

            if (a == b || weight == 0.0)
                continue;

            long key = (long)Math.Min(a, b) * nodeCount + Math.Max(a, b);
            unique[key] = unique.TryGetValue(key, out double existing) ? Math.Max(existing, weight) : weight;
        }

        return Build(nodeCount, unique);
    }

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
    /// O(n^2 d) time and O(n k) memory, with no spatial index. At a few thousand
    /// samples in a handful of columns that is well under a second, and it avoids
    /// holding an n x n distance matrix. Ties are broken towards the lower index so
    /// the graph is a function of the data alone.
    /// </para>
    /// </summary>
    /// <param name="x">n x d data, rows are samples.</param>
    /// <param name="neighbours">Neighbours per point, not counting itself. Between 1 and n - 1.</param>
    public static WeightedGraph NearestNeighbours(double[,] x, int neighbours)
    {
        ArgumentNullException.ThrowIfNull(x);

        int n = x.GetLength(0);
        int d = x.GetLength(1);

        if (neighbours < 1 || neighbours > n - 1)
            throw new ArgumentOutOfRangeException(nameof(neighbours), neighbours,
                $"Need between 1 and {n - 1} neighbours for {n} samples.");

        int k = neighbours;
        var chosen = new int[n, k];
        var windowDistance = new double[k];
        var windowIndex = new int[k];

        for (int i = 0; i < n; i++)
        {
            Array.Fill(windowDistance, double.MaxValue);
            Array.Fill(windowIndex, -1);

            for (int j = 0; j < n; j++)
            {
                if (j == i)
                    continue;

                double distance = 0.0;
                for (int c = 0; c < d; c++)
                {
                    double delta = x[i, c] - x[j, c];
                    distance += delta * delta;
                }

                // Strictly less, so an equal distance never displaces the lower
                // index already held — j arrives in ascending order.
                if (distance >= windowDistance[k - 1])
                    continue;

                int position = k - 1;
                while (position > 0 && windowDistance[position - 1] > distance)
                {
                    windowDistance[position] = windowDistance[position - 1];
                    windowIndex[position] = windowIndex[position - 1];
                    position--;
                }

                windowDistance[position] = distance;
                windowIndex[position] = j;
            }

            for (int c = 0; c < k; c++)
                chosen[i, c] = windowIndex[c];
        }

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

        return Build(n, unique);
    }

    /// <summary>Indices of the nodes adjacent to <paramref name="node"/>, ascending.</summary>
    public ReadOnlySpan<int> Neighbours(int node)
        => _targets.AsSpan(_offsets[node], _offsets[node + 1] - _offsets[node]);

    /// <summary>Weights of the edges to <see cref="Neighbours"/>, in the same order.</summary>
    public ReadOnlySpan<double> EdgeWeights(int node)
        => _weights.AsSpan(_offsets[node], _offsets[node + 1] - _offsets[node]);

    /// <summary>Weighted degree: the sum of the weights of every edge at <paramref name="node"/>.</summary>
    public double Degree(int node)
    {
        double sum = 0.0;
        foreach (double weight in EdgeWeights(node))
            sum += weight;

        return sum;
    }

    /// <summary>Every edge once, lower index first, in ascending order.</summary>
    public IEnumerable<(int A, int B, double Weight)> Edges()
    {
        for (int a = 0; a < NodeCount; a++)
            for (int e = _offsets[a]; e < _offsets[a + 1]; e++)
                if (_targets[e] > a)
                    yield return (a, _targets[e], _weights[e]);
    }

    /// <summary>
    /// Connected component of every node.
    /// <para>
    /// Worth checking before anything spectral or propagating runs, because both
    /// behave differently across a disconnection: no signal crosses it at all.
    /// Components are numbered in order of their lowest node, so the numbering is
    /// a property of the graph rather than of the traversal.
    /// </para>
    /// </summary>
    /// <param name="count">Number of components, isolated nodes each counting as one.</param>
    public int[] ConnectedComponents(out int count)
    {
        var component = new int[NodeCount];
        Array.Fill(component, -1);

        count = 0;
        var stack = new Stack<int>();

        for (int start = 0; start < NodeCount; start++)
        {
            if (component[start] >= 0)
                continue;

            component[start] = count;
            stack.Push(start);

            while (stack.Count > 0)
            {
                int node = stack.Pop();
                foreach (int next in Neighbours(node))
                {
                    if (component[next] >= 0)
                        continue;

                    component[next] = count;
                    stack.Push(next);
                }
            }

            count++;
        }

        return component;
    }

    /// <summary>
    /// A copy with every edge's weight replaced.
    /// <para>
    /// The seam between connectivity and similarity: a caller supplies which
    /// nodes are related, and this lets something else say how strongly. An edge
    /// whose new weight is zero or less is dropped.
    /// </para>
    /// </summary>
    /// <param name="weight">Given the two ends (lower index first) and the current weight, returns the new weight.</param>
    public WeightedGraph Reweight(Func<int, int, double, double> weight)
    {
        ArgumentNullException.ThrowIfNull(weight);

        var unique = new Dictionary<long, double>(EdgeCount);
        foreach (var (a, b, current) in Edges())
        {
            double updated = weight(a, b, current);
            if (double.IsNaN(updated) || double.IsInfinity(updated))
                throw new InvalidOperationException(
                    $"Reweighting edge ({a}, {b}) produced {updated}; weights must be finite.");

            if (updated > 0.0)
                unique[(long)a * NodeCount + b] = updated;
        }

        return Build(NodeCount, unique);
    }

    /// <summary>
    /// One step of symmetric normalised propagation: returns
    /// <c>D^-1/2 (A + sI) D^-1/2 X</c>, where <c>D</c> is the degree including the
    /// self-loop.
    /// <para>
    /// This is the one operator the graph algorithms above share. With
    /// <paramref name="selfWeight"/> zero it is the normalised adjacency whose
    /// leading eigenvectors spectral clustering embeds by; with it at one it is
    /// the renormalised adjacency a graph convolutional network propagates with.
    /// The symmetric normalisation, rather than dividing by degree on one side,
    /// is what keeps the operator symmetric and its eigenvalues in [-1, 1] — and
    /// so what stops a hub with many neighbours from dominating every sum it
    /// takes part in.
    /// </para>
    /// <para>
    /// A node with no edges and no self-loop has zero degree and gets a zero row:
    /// nothing reaches it and it reaches nothing.
    /// </para>
    /// </summary>
    /// <param name="x">n x p, one row per node. Not modified.</param>
    /// <param name="selfWeight">Weight of the self-loop added to every node, zero or more.</param>
    public double[,] Propagate(double[,] x, double selfWeight)
    {
        ArgumentNullException.ThrowIfNull(x);
        if (x.GetLength(0) != NodeCount)
            throw new ArgumentException($"Expected {NodeCount} rows, got {x.GetLength(0)}.", nameof(x));
        if (double.IsNaN(selfWeight) || selfWeight < 0.0)
            throw new ArgumentOutOfRangeException(nameof(selfWeight), selfWeight, "Self weight cannot be negative.");

        int n = NodeCount;
        int p = x.GetLength(1);

        var scale = new double[n];
        for (int i = 0; i < n; i++)
        {
            double degree = Degree(i) + selfWeight;
            scale[i] = degree > 0.0 ? 1.0 / Math.Sqrt(degree) : 0.0;
        }

        var y = new double[n, p];
        for (int i = 0; i < n; i++)
        {
            if (scale[i] == 0.0)
                continue;

            double own = selfWeight * scale[i] * scale[i];
            for (int c = 0; c < p; c++)
                y[i, c] = own * x[i, c];

            for (int e = _offsets[i]; e < _offsets[i + 1]; e++)
            {
                int j = _targets[e];
                double factor = _weights[e] * scale[i] * scale[j];
                for (int c = 0; c < p; c++)
                    y[i, c] += factor * x[j, c];
            }
        }

        return y;
    }

    /// <summary>
    /// Lays undirected edges out as sorted compressed rows, each edge stored from
    /// both ends. Keys are <c>lower * n + higher</c>.
    /// </summary>
    private static WeightedGraph Build(int n, Dictionary<long, double> unique)
    {
        var counts = new int[n];
        foreach (long key in unique.Keys)
        {
            counts[(int)(key / n)]++;
            counts[(int)(key % n)]++;
        }

        var offsets = new int[n + 1];
        for (int i = 0; i < n; i++)
            offsets[i + 1] = offsets[i] + counts[i];

        var targets = new int[offsets[n]];
        var weights = new double[offsets[n]];
        var cursor = (int[])offsets.Clone();

        // Ascending keys leave every row sorted without a second pass: row b
        // receives its lower neighbours a from keys a*n + b, all of which are
        // below b*n, and only then its upper neighbours from keys b*n + c.
        foreach (var (key, weight) in unique.OrderBy(pair => pair.Key))
        {
            int a = (int)(key / n);
            int b = (int)(key % n);

            targets[cursor[a]] = b;
            weights[cursor[a]++] = weight;
            targets[cursor[b]] = a;
            weights[cursor[b]++] = weight;
        }

        return new WeightedGraph(n, offsets, targets, weights);
    }
}
