namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// Internal validation measures — how good a partition looks using only the data
/// and the labels, with no ground truth to check against.
/// <para>
/// Both measures here reward the same thing: compact clusters that sit far
/// apart. That makes them useful and it makes them biased, and the bias matters
/// whenever they are used to choose between algorithms rather than between
/// settings. k-means optimises almost exactly what a silhouette measures, so a
/// contest judged on silhouette alone is one k-means is close to guaranteed to
/// win — not because it found the truth, but because the referee and the player
/// agree on the definition of good. Use these to answer "are these clusters
/// clean", which they do well, and something else to answer "do they overlap" or
/// "is this data messy".
/// </para>
/// </summary>
public static class ClusterQuality
{
    /// <summary>
    /// Mean silhouette over every labelled point, between -1 and 1.
    /// <para>
    /// A point's silhouette compares the mean distance to its own cluster
    /// against the mean distance to the nearest other cluster. Near 1 is
    /// comfortably inside a well-separated cluster; near 0 is on a boundary;
    /// negative means it is closer to another cluster than to its own.
    /// </para>
    /// <para>
    /// Points labelled <c>-1</c> are excluded rather than treated as a cluster.
    /// Noise is not a group and has no centre; scoring it as one would punish
    /// <see cref="Hdbscan"/> for the very thing it is there to do. Read the
    /// silhouette alongside the noise fraction, never instead of it.
    /// </para>
    /// </summary>
    /// <param name="x">n x d data, rows are samples.</param>
    /// <param name="labels">Cluster index per sample; negative means noise.</param>
    /// <returns>The mean silhouette, or 0 when fewer than two clusters carry points.</returns>
    public static double Silhouette(double[,] x, int[] labels)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(labels);

        int d = x.GetLength(1);
        var members = Members(labels);
        int clusterCount = members.Length;

        if (members.Count(m => m.Count > 0) < 2)
            return 0.0;

        double total = 0.0;
        int counted = 0;

        for (int i = 0; i < labels.Length; i++)
        {
            int own = labels[i];
            if (own < 0)
                continue;

            counted++;

            // A cluster of one has no within-cluster distance to speak of.
            // Convention, and scikit-learn's, is to score it zero.
            if (members[own].Count < 2)
                continue;

            double inside = 0.0;
            foreach (int j in members[own])
                if (j != i)
                    inside += Math.Sqrt(KMeans.SquaredDistance(x, i, x, j, d));

            inside /= members[own].Count - 1;

            double nearest = double.MaxValue;
            for (int c = 0; c < clusterCount; c++)
            {
                if (c == own || members[c].Count == 0)
                    continue;

                double outside = 0.0;
                foreach (int j in members[c])
                    outside += Math.Sqrt(KMeans.SquaredDistance(x, i, x, j, d));

                outside /= members[c].Count;
                if (outside < nearest)
                    nearest = outside;
            }

            if (nearest == double.MaxValue)
                continue;

            double divisor = Math.Max(inside, nearest);
            if (divisor > 0.0)
                total += (nearest - inside) / divisor;
        }

        return counted == 0 ? 0.0 : total / counted;
    }

    /// <summary>
    /// Davies-Bouldin index over every labelled point. Lower is better, and zero
    /// is the unreachable ideal.
    /// <para>
    /// For each cluster it finds the worst other cluster — the one whose spread
    /// plus its own, divided by the gap between their centres, is largest — and
    /// averages that worst case. So it is dominated by the single most confusable
    /// pair, where a silhouette averages over everything. The two disagreeing is
    /// worth reading as "one pair of clusters is doing all the damage".
    /// </para>
    /// <para>
    /// Noise is excluded, as in <see cref="Silhouette"/>.
    /// </para>
    /// </summary>
    /// <returns>The index, or <see cref="double.NaN"/> when fewer than two clusters carry points.</returns>
    public static double DaviesBouldin(double[,] x, int[] labels)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(labels);

        int d = x.GetLength(1);
        var members = Members(labels);

        var populated = Enumerable.Range(0, members.Length).Where(c => members[c].Count > 0).ToArray();
        if (populated.Length < 2)
            return double.NaN;

        var centroids = new double[members.Length, d];
        var spread = new double[members.Length];

        foreach (int c in populated)
        {
            foreach (int i in members[c])
                for (int j = 0; j < d; j++)
                    centroids[c, j] += x[i, j];

            for (int j = 0; j < d; j++)
                centroids[c, j] /= members[c].Count;
        }

        foreach (int c in populated)
        {
            double sum = 0.0;
            foreach (int i in members[c])
                sum += Math.Sqrt(KMeans.SquaredDistance(x, i, centroids, c, d));

            spread[c] = sum / members[c].Count;
        }

        double total = 0.0;
        foreach (int a in populated)
        {
            double worst = 0.0;

            foreach (int b in populated)
            {
                if (a == b)
                    continue;

                double gap = Math.Sqrt(KMeans.SquaredDistance(centroids, a, centroids, b, d));
                if (gap <= 0.0)
                    continue;

                worst = Math.Max(worst, (spread[a] + spread[b]) / gap);
            }

            total += worst;
        }

        return total / populated.Length;
    }

    /// <summary>
    /// Point indices per cluster, ignoring negative labels. The array is indexed
    /// by label, so it can contain empty entries where a label went unused.
    /// </summary>
    private static List<int>[] Members(int[] labels)
    {
        int width = 0;
        foreach (int label in labels)
            if (label + 1 > width)
                width = label + 1;

        var members = new List<int>[width];
        for (int c = 0; c < width; c++)
            members[c] = new List<int>();

        for (int i = 0; i < labels.Length; i++)
            if (labels[i] >= 0)
                members[labels[i]].Add(i);

        return members;
    }
}
