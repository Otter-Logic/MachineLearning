namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// A fitted k-means partition: where every sample went, where the centres
/// landed, and how tight the result is.
/// </summary>
public sealed class KMeansResult
{
    internal KMeansResult(
        int[] labels, double[,] centroids, double inertia, int iterations, bool converged)
    {
        Labels = labels;
        Centroids = centroids;
        Inertia = inertia;
        Iterations = iterations;
        Converged = converged;
    }

    /// <summary>Cluster index per sample, 0-based.</summary>
    public int[] Labels { get; }

    /// <summary>Cluster centres, k x d, in the space the fit was run in.</summary>
    public double[,] Centroids { get; }

    /// <summary>
    /// Total squared distance from every sample to its own centre — scikit-learn's
    /// <c>inertia_</c>.
    /// <para>
    /// Lower is tighter, but it falls monotonically as k rises and so cannot
    /// choose k on its own. Use it to compare restarts at one k, or to find an
    /// elbow across several; use a silhouette to compare partitions.
    /// </para>
    /// </summary>
    public double Inertia { get; }

    /// <summary>Lloyd iterations taken by the winning restart.</summary>
    public int Iterations { get; }

    /// <summary>Whether the winning restart stopped moving before the iteration cap.</summary>
    public bool Converged { get; }

    /// <summary>Number of samples fitted.</summary>
    public int SampleCount => Labels.Length;

    /// <summary>Number of clusters.</summary>
    public int ClusterCount => Centroids.GetLength(0);

    /// <summary>Sample indices bucketed by cluster, ready to drive geometry downstream.</summary>
    public int[][] Clusters()
    {
        int k = ClusterCount;
        var buckets = new List<int>[k];
        for (int c = 0; c < k; c++)
            buckets[c] = new List<int>();

        for (int i = 0; i < Labels.Length; i++)
            buckets[Labels[i]].Add(i);

        return buckets.Select(b => b.ToArray()).ToArray();
    }
}
