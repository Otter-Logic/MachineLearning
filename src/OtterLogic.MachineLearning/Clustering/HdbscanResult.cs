namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// A fitted HDBSCAN: which points formed clusters, which were left as noise, and
/// how firmly each clustered point belongs where it was put.
/// </summary>
public sealed class HdbscanResult
{
    internal HdbscanResult(
        int[] labels, double[] probabilities, double[] clusterStabilities)
    {
        Labels = labels;
        Probabilities = probabilities;
        ClusterStabilities = clusterStabilities;
    }

    /// <summary>
    /// Cluster index per point, or <c>-1</c> for noise.
    /// <para>
    /// The <c>-1</c> is the output that separates this from k-means and a
    /// mixture, both of which must place every point somewhere. A point labelled
    /// noise is one the data does not support putting in any family, and saying
    /// so is more useful than forcing it into the nearest one.
    /// </para>
    /// </summary>
    public int[] Labels { get; }

    /// <summary>
    /// Membership strength per point, between 0 and 1. Noise points are 0.
    /// <para>
    /// One means the point survived to its cluster's densest level; low values
    /// sit on the cluster's fringe and were nearly noise.
    /// </para>
    /// </summary>
    public double[] Probabilities { get; }

    /// <summary>
    /// Stability of each selected cluster — the excess of mass the extraction
    /// maximised. Larger means the cluster persisted over a wider range of
    /// density thresholds, so it is a more believable group.
    /// </summary>
    public double[] ClusterStabilities { get; }

    /// <summary>Number of clusters found, not counting noise.</summary>
    public int ClusterCount => ClusterStabilities.Length;

    /// <summary>Number of points left as noise.</summary>
    public int NoiseCount => Labels.Count(l => l < 0);

    /// <summary>Number of points fitted.</summary>
    public int SampleCount => Labels.Length;

    /// <summary>
    /// Fraction of points left as noise, between 0 and 1.
    /// <para>
    /// The headline diagnostic. A little noise is the algorithm doing its job; a
    /// lot means either the data has no density structure at these settings, or
    /// the minimum cluster size is asking for families larger than the data
    /// contains.
    /// </para>
    /// </summary>
    public double NoiseFraction => SampleCount == 0 ? 0.0 : (double)NoiseCount / SampleCount;

    /// <summary>
    /// Point indices bucketed by cluster, noise excluded. Index c holds the
    /// members of cluster c.
    /// </summary>
    public int[][] Clusters()
    {
        var buckets = new List<int>[ClusterCount];
        for (int c = 0; c < ClusterCount; c++)
            buckets[c] = new List<int>();

        for (int i = 0; i < Labels.Length; i++)
            if (Labels[i] >= 0)
                buckets[Labels[i]].Add(i);

        return buckets.Select(b => b.ToArray()).ToArray();
    }

    /// <summary>Indices of the points left as noise.</summary>
    public int[] Noise()
        => Enumerable.Range(0, Labels.Length).Where(i => Labels[i] < 0).ToArray();
}
