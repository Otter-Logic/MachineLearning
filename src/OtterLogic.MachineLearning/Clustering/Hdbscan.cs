namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// Hierarchical density-based clustering — HDBSCAN, by mutual reachability,
/// a minimum spanning tree, and cluster extraction by excess of mass.
/// <para>
/// The reason it earns its place beside <see cref="KMeans"/> and
/// <see cref="GaussianMixture"/> is that it answers a question neither can. Both
/// of those must put every point somewhere and must be told how many groups to
/// look for. This is told neither: it finds however many dense regions the data
/// supports, of whatever shape, and labels everything else <c>-1</c>. On
/// structural demand data that matters, because a handful of members really are
/// one-offs, and a model that forces them into the nearest family produces a
/// family nobody can detail.
/// </para>
/// <para>
/// The cost is that density is a weaker signal than a fitted model. Where the
/// groups genuinely are round and well separated, k-means will find them more
/// cleanly and a mixture will tell you more about the overlap. This one is for
/// when they are not.
/// </para>
/// <para>
/// Implementation follows Campello, Moulavi and Sander (2013) and the reference
/// <c>hdbscan</c> library: core distances, a mutual reachability MST, a
/// condensed cluster tree, and excess-of-mass selection. The MST is built by
/// Prim's in O(n^2) with no distance matrix held in memory — at a few thousand
/// members in three principal components that is milliseconds and a few
/// kilobytes, and it avoids the spatial index the reference needs at scale.
/// </para>
/// </summary>
public static class Hdbscan
{
    /// <summary>
    /// Stands in for 1/0 when two points coincide.
    /// <para>
    /// Infinity would be the true value and it poisons the stability sums, which
    /// subtract one lambda from another and would produce NaN. A large finite
    /// value keeps the arithmetic well defined and preserves the ordering, which
    /// is all the extraction needs. Duplicate rows are not a contrived input
    /// here — a structural model repeats identical members constantly.
    /// </para>
    /// </summary>
    private const double MaximumLambda = 1e12;

    /// <summary>
    /// Clusters by density. No cluster count is asked for or returned as a
    /// setting — the number found is an output.
    /// </summary>
    /// <param name="x">n x d data, rows are samples. Expected to be standardised already.</param>
    /// <param name="options">Fit settings.</param>
    public static HdbscanResult Fit(double[,] x, HdbscanOptions options)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(options);

        int n = x.GetLength(0);
        if (n < 2)
            throw new ArgumentException("Need at least two points to cluster.", nameof(x));

        options.Validate(n);

        var core = CoreDistances(x, options.EffectiveMinimumSamples);
        var mst = MinimumSpanningTree(x, core);
        var hierarchy = SingleLinkage(mst, n);
        var condensed = Condense(hierarchy, n, options.MinimumClusterSize);

        return Extract(condensed, n);
    }

    /// <summary>
    /// Distance from each point to its k-th nearest neighbour, counting itself.
    /// <para>
    /// This is the density estimate the whole algorithm rests on: a small core
    /// distance means the point sits somewhere crowded. Kept as a k-element
    /// insertion window rather than a sort, because k is small and n is not.
    /// </para>
    /// </summary>
    private static double[] CoreDistances(double[,] x, int minSamples)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);
        int k = Math.Min(minSamples, n);

        var core = new double[n];
        var window = new double[k];

        for (int i = 0; i < n; i++)
        {
            Array.Fill(window, double.MaxValue);

            for (int j = 0; j < n; j++)
            {
                double distance = Math.Sqrt(KMeans.SquaredDistance(x, i, x, j, d));
                if (distance >= window[k - 1])
                    continue;

                int position = k - 1;
                while (position > 0 && window[position - 1] > distance)
                {
                    window[position] = window[position - 1];
                    position--;
                }

                window[position] = distance;
            }

            core[i] = window[k - 1];
        }

        return core;
    }

    /// <summary>
    /// Prim's algorithm over the mutual reachability graph.
    /// <para>
    /// Mutual reachability is <c>max(core(a), core(b), d(a, b))</c> — it pushes
    /// sparse points away from everything while leaving dense neighbourhoods at
    /// their true distances, which is what stops a chain of outliers linking two
    /// real clusters together.
    /// </para>
    /// <para>
    /// Edge weights are computed on demand rather than held as an n x n matrix.
    /// At two thousand points that matrix would be thirty megabytes to save
    /// arithmetic that costs milliseconds.
    /// </para>
    /// </summary>
    private static (int A, int B, double Weight)[] MinimumSpanningTree(double[,] x, double[] core)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        var inTree = new bool[n];
        var cheapest = new double[n];
        var parent = new int[n];

        Array.Fill(cheapest, double.MaxValue);
        Array.Fill(parent, -1);
        cheapest[0] = 0.0;

        var edges = new List<(int, int, double)>(n - 1);

        for (int step = 0; step < n; step++)
        {
            int next = -1;
            double best = double.MaxValue;

            for (int v = 0; v < n; v++)
            {
                if (!inTree[v] && cheapest[v] < best)
                {
                    best = cheapest[v];
                    next = v;
                }
            }

            if (next < 0)
                break;

            inTree[next] = true;
            if (parent[next] >= 0)
                edges.Add((parent[next], next, cheapest[next]));

            for (int v = 0; v < n; v++)
            {
                if (inTree[v])
                    continue;

                double distance = Math.Sqrt(KMeans.SquaredDistance(x, next, x, v, d));
                double reachability = Math.Max(distance, Math.Max(core[next], core[v]));

                if (reachability < cheapest[v])
                {
                    cheapest[v] = reachability;
                    parent[v] = next;
                }
            }
        }

        return edges.ToArray();
    }

    /// <summary>One merge in the single-linkage hierarchy.</summary>
    private readonly record struct HierarchyNode(int Left, int Right, double Distance, int Size);

    /// <summary>
    /// Turns the MST into a single-linkage dendrogram by adding edges shortest
    /// first, in the layout SciPy's <c>linkage</c> uses: points are nodes
    /// <c>0..n-1</c> and each merge creates the next node above <c>n</c>.
    /// </summary>
    private static HierarchyNode[] SingleLinkage((int A, int B, double Weight)[] mst, int n)
    {
        var ordered = mst.OrderBy(e => e.Weight).ToArray();

        var parent = new int[2 * n - 1];
        var size = new int[2 * n - 1];
        Array.Fill(parent, -1);
        for (int i = 0; i < n; i++)
            size[i] = 1;

        var hierarchy = new HierarchyNode[n - 1];
        int nextNode = n;

        foreach (var (a, b, weight) in ordered)
        {
            int rootA = Find(parent, a);
            int rootB = Find(parent, b);
            int merged = size[rootA] + size[rootB];

            hierarchy[nextNode - n] = new HierarchyNode(rootA, rootB, weight, merged);

            parent[rootA] = nextNode;
            parent[rootB] = nextNode;
            size[nextNode] = merged;
            nextNode++;
        }

        return hierarchy;
    }

    private static int Find(int[] parent, int node)
    {
        while (parent[node] != -1)
            node = parent[node];

        return node;
    }

    /// <summary>One edge of the condensed cluster tree.</summary>
    /// <param name="Parent">The cluster the child left.</param>
    /// <param name="Child">A point index below n, or a child cluster label at or above n.</param>
    /// <param name="Lambda">Inverse of the distance at which it left, so larger is denser.</param>
    /// <param name="ChildSize">Points carried by the child.</param>
    private readonly record struct CondensedEdge(int Parent, int Child, double Lambda, int ChildSize);

    /// <summary>
    /// Condenses the dendrogram: every merge in the hierarchy becomes either a
    /// genuine split into two clusters, or a handful of points falling out of one.
    /// <para>
    /// This is where <see cref="HdbscanOptions.MinimumClusterSize"/> does its
    /// work, and it is the step that makes the algorithm robust. A single-linkage
    /// dendrogram has n-1 splits and almost all of them are one point leaving;
    /// treating those as noise events rather than as new clusters is what stops
    /// the result being a chain.
    /// </para>
    /// </summary>
    private static List<CondensedEdge> Condense(HierarchyNode[] hierarchy, int n, int minimumClusterSize)
    {
        int root = 2 * n - 2;
        var condensed = new List<CondensedEdge>();

        var relabel = new int[2 * n - 1];
        var ignore = new bool[2 * n - 1];
        relabel[root] = n;
        int nextLabel = n + 1;

        foreach (int node in Subtree(hierarchy, n, root))
        {
            if (ignore[node] || node < n)
                continue;

            var entry = hierarchy[node - n];
            double lambda = entry.Distance > 0.0
                ? Math.Min(1.0 / entry.Distance, MaximumLambda)
                : MaximumLambda;

            int left = entry.Left;
            int right = entry.Right;
            int leftCount = left >= n ? hierarchy[left - n].Size : 1;
            int rightCount = right >= n ? hierarchy[right - n].Size : 1;

            bool leftSurvives = leftCount >= minimumClusterSize;
            bool rightSurvives = rightCount >= minimumClusterSize;

            if (leftSurvives && rightSurvives)
            {
                // A real split: both halves are big enough to be clusters.
                relabel[left] = nextLabel++;
                condensed.Add(new CondensedEdge(relabel[node], relabel[left], lambda, leftCount));

                relabel[right] = nextLabel++;
                condensed.Add(new CondensedEdge(relabel[node], relabel[right], lambda, rightCount));
            }
            else if (!leftSurvives && !rightSurvives)
            {
                // The cluster dissolves here; everything below is noise from now on.
                FallOut(hierarchy, n, left, relabel[node], lambda, condensed, ignore);
                FallOut(hierarchy, n, right, relabel[node], lambda, condensed, ignore);
            }
            else if (!leftSurvives)
            {
                // The big half carries on as the same cluster, the small half leaves.
                relabel[right] = relabel[node];
                FallOut(hierarchy, n, left, relabel[node], lambda, condensed, ignore);
            }
            else
            {
                relabel[left] = relabel[node];
                FallOut(hierarchy, n, right, relabel[node], lambda, condensed, ignore);
            }
        }

        return condensed;
    }

    /// <summary>Records every point under <paramref name="subtree"/> as leaving its parent cluster.</summary>
    private static void FallOut(
        HierarchyNode[] hierarchy, int n, int subtree, int parentLabel, double lambda,
        List<CondensedEdge> condensed, bool[] ignore)
    {
        foreach (int node in Subtree(hierarchy, n, subtree))
        {
            if (node < n)
                condensed.Add(new CondensedEdge(parentLabel, node, lambda, 1));

            ignore[node] = true;
        }
    }

    /// <summary>Every node at or below <paramref name="start"/>, breadth first.</summary>
    private static IEnumerable<int> Subtree(HierarchyNode[] hierarchy, int n, int start)
    {
        var queue = new Queue<int>();
        queue.Enqueue(start);

        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            yield return node;

            if (node < n)
                continue;

            var entry = hierarchy[node - n];
            queue.Enqueue(entry.Left);
            queue.Enqueue(entry.Right);
        }
    }

    /// <summary>
    /// Selects clusters by excess of mass and turns the selection into labels.
    /// <para>
    /// Stability of a cluster is the total density range over which its points
    /// stayed in it. The selection walks the condensed tree bottom up and keeps a
    /// cluster whenever it is more stable than its children put together —
    /// which is what lets the result mix a broad sparse group with a tight dense
    /// one, something a single density threshold cannot do.
    /// </para>
    /// <para>
    /// The root is never selectable. Data with no split above the minimum
    /// cluster size therefore comes back entirely as noise rather than as one
    /// cluster containing everything, which is the reference library's default
    /// and the honest answer: "there is one blob here" is not a clustering.
    /// </para>
    /// </summary>
    private static HdbscanResult Extract(List<CondensedEdge> condensed, int n)
    {
        var labels = new int[n];
        var probabilities = new double[n];
        Array.Fill(labels, -1);

        if (condensed.Count == 0)
            return new HdbscanResult(labels, probabilities, Array.Empty<double>());

        // Where each point left the tree, and where each cluster hangs from.
        var pointLambda = new double[n];
        var pointParent = new int[n];
        Array.Fill(pointParent, -1);

        var clusterParent = new Dictionary<int, int>();
        var births = new Dictionary<int, double>();

        foreach (var edge in condensed)
        {
            if (edge.Child < n)
            {
                pointLambda[edge.Child] = edge.Lambda;
                pointParent[edge.Child] = edge.Parent;
            }
            else
            {
                clusterParent[edge.Child] = edge.Parent;
                births[edge.Child] = edge.Lambda;
            }
        }

        var stability = new Dictionary<int, double>();
        foreach (var edge in condensed)
        {
            double birth = births.TryGetValue(edge.Parent, out double b) ? b : 0.0;
            stability.TryGetValue(edge.Parent, out double running);
            stability[edge.Parent] = running + (edge.Lambda - birth) * edge.ChildSize;
        }

        var children = new Dictionary<int, List<int>>();
        foreach (var edge in condensed)
        {
            if (edge.Child < n)
                continue;

            if (!children.TryGetValue(edge.Parent, out var list))
                children[edge.Parent] = list = new List<int>();

            list.Add(edge.Child);
        }

        int root = n;

        // A child cluster is always labelled after its parent, so descending
        // order visits children first — which is what excess of mass needs.
        var order = stability.Keys.Where(c => c != root).OrderByDescending(c => c).ToList();
        var selected = new HashSet<int>(order);
        var mass = new Dictionary<int, double>(stability);

        foreach (int cluster in order)
        {
            double below = children.TryGetValue(cluster, out var kids)
                ? kids.Sum(kid => mass.TryGetValue(kid, out double m) ? m : 0.0)
                : 0.0;

            if (below > mass[cluster])
            {
                // The children explain the data better than the parent does.
                selected.Remove(cluster);
                mass[cluster] = below;
            }
            else
            {
                foreach (int descendant in Descendants(children, cluster))
                    selected.Remove(descendant);
            }
        }

        if (selected.Count == 0)
            return new HdbscanResult(labels, probabilities, Array.Empty<double>());

        // Which selected cluster, if any, each point ended up inside.
        var owner = new int[n];
        Array.Fill(owner, -1);

        for (int i = 0; i < n; i++)
        {
            int node = pointParent[i];
            while (node >= 0)
            {
                if (selected.Contains(node))
                {
                    owner[i] = node;
                    break;
                }

                node = clusterParent.TryGetValue(node, out int up) ? up : -1;
            }
        }

        // Number the clusters largest first, so a small change upstream does not
        // permute them and shuffle every colour downstream.
        var ordered = selected
            .Select(c => (Cluster: c, Count: owner.Count(o => o == c)))
            .Where(entry => entry.Count > 0)
            .OrderByDescending(entry => entry.Count)
            .ThenBy(entry => entry.Cluster)
            .ToList();

        var index = new Dictionary<int, int>();
        for (int c = 0; c < ordered.Count; c++)
            index[ordered[c].Cluster] = c;

        var peak = new double[ordered.Count];
        for (int i = 0; i < n; i++)
        {
            if (owner[i] < 0 || !index.TryGetValue(owner[i], out int c))
                continue;

            labels[i] = c;
            peak[c] = Math.Max(peak[c], pointLambda[i]);
        }

        for (int i = 0; i < n; i++)
        {
            int c = labels[i];
            if (c < 0)
                continue;

            probabilities[i] = peak[c] > 0.0 ? Math.Min(1.0, pointLambda[i] / peak[c]) : 1.0;
        }

        var stabilities = ordered.Select(entry => stability[entry.Cluster]).ToArray();
        return new HdbscanResult(labels, probabilities, stabilities);
    }

    private static IEnumerable<int> Descendants(Dictionary<int, List<int>> children, int cluster)
    {
        if (!children.TryGetValue(cluster, out var direct))
            yield break;

        var queue = new Queue<int>(direct);
        while (queue.Count > 0)
        {
            int node = queue.Dequeue();
            yield return node;

            if (children.TryGetValue(node, out var next))
                foreach (int child in next)
                    queue.Enqueue(child);
        }
    }
}
