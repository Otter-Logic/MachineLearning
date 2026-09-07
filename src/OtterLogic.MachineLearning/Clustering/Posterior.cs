namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// Reductions over a responsibility matrix.
/// <para>
/// Shared because the pipeline reorders components before returning them, so it
/// needs these over its own reordered matrix rather than the one the fit
/// produced — and two copies of an argmax is two places for the tie-breaking to
/// drift apart.
/// </para>
/// </summary>
internal static class Posterior
{
    /// <summary>Index of the largest responsibility in each row, first one winning a tie.</summary>
    internal static int[] ArgMax(double[,] responsibilities)
    {
        int n = responsibilities.GetLength(0);
        int k = responsibilities.GetLength(1);
        var labels = new int[n];

        for (int i = 0; i < n; i++)
        {
            int best = 0;
            for (int c = 1; c < k; c++)
                if (responsibilities[i, c] > responsibilities[i, best])
                    best = c;

            labels[i] = best;
        }

        return labels;
    }

    /// <summary>The largest responsibility in each row — how sure the model is.</summary>
    internal static double[] RowMax(double[,] responsibilities)
    {
        int n = responsibilities.GetLength(0);
        int k = responsibilities.GetLength(1);
        var maxima = new double[n];

        for (int i = 0; i < n; i++)
        {
            double best = responsibilities[i, 0];
            for (int c = 1; c < k; c++)
                if (responsibilities[i, c] > best)
                    best = responsibilities[i, c];

            maxima[i] = best;
        }

        return maxima;
    }
}
