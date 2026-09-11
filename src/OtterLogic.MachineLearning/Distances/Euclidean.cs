namespace OtterLogic.MachineLearning.Distances;

/// <summary>
/// Euclidean distance between rows of sample matrices.
/// <para>
/// Here because every paradigm measures it — clustering to place a sample, a
/// nearest-neighbour regressor to find its neighbours, a graph to decide which
/// samples to join — and it had been written out six times across the stack, one
/// of them hidden inside k-means, so that density clustering and quality scores
/// depended on k-means for no reason but a distance. One copy, summed in one
/// order, is also what keeps results bit-identical wherever it is used.
/// </para>
/// </summary>
public static class Euclidean
{
    /// <summary>
    /// Squared distance between row <paramref name="rowA"/> of <paramref name="a"/>
    /// and row <paramref name="rowB"/> of <paramref name="b"/>, over the columns of
    /// <paramref name="a"/>.
    /// <para>
    /// Squared because most callers only compare distances, and the square root is
    /// the one expensive step in an inner loop that runs n² times.
    /// </para>
    /// </summary>
    public static double Squared(double[,] a, int rowA, double[,] b, int rowB)
    {
        int d = a.GetLength(1);
        double sum = 0.0;
        for (int j = 0; j < d; j++)
        {
            double delta = a[rowA, j] - b[rowB, j];
            sum += delta * delta;
        }

        return sum;
    }

    /// <summary>Distance between two rows — the square root of <see cref="Squared"/>.</summary>
    public static double Between(double[,] a, int rowA, double[,] b, int rowB)
        => Math.Sqrt(Squared(a, rowA, b, rowB));

    /// <summary>
    /// The <paramref name="count"/> nearest other rows to every row of
    /// <paramref name="x"/>, nearest first, ties to the lower index.
    /// <para>
    /// Brute force: O(n² d) time and O(n k) memory, never an n x n matrix. At a
    /// few thousand samples in a handful of columns that is well under a second,
    /// and it is exact, which a spatial index in many dimensions is not without
    /// care. Each row keeps a k-long insertion window rather than sorting all n,
    /// because k is small and n is not.
    /// </para>
    /// <para>
    /// A row is never its own neighbour. A caller that counts the row itself — a
    /// density estimate that asks for the k-th point including this one — wants
    /// entry k − 2, or zero for k of one.
    /// </para>
    /// </summary>
    /// <param name="x">n x d data, rows are samples.</param>
    /// <param name="count">Neighbours per row, between 1 and n − 1.</param>
    /// <returns>
    /// <c>Index[i, c]</c> is the c-th nearest row to row i, and
    /// <c>Distance[i, c]</c> its distance.
    /// </returns>
    public static (int[,] Index, double[,] Distance) Nearest(double[,] x, int count)
    {
        ArgumentNullException.ThrowIfNull(x);

        int n = x.GetLength(0);
        if (count < 1 || count > n - 1)
            throw new ArgumentOutOfRangeException(nameof(count), count,
                $"Need between 1 and {n - 1} neighbours for {n} samples.");

        int k = count;
        var index = new int[n, k];
        var distance = new double[n, k];
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

                double squared = Squared(x, i, x, j);

                // Strictly less, so an equal distance never displaces the lower
                // index already held — j arrives in ascending order.
                if (squared >= windowDistance[k - 1])
                    continue;

                int position = k - 1;
                while (position > 0 && windowDistance[position - 1] > squared)
                {
                    windowDistance[position] = windowDistance[position - 1];
                    windowIndex[position] = windowIndex[position - 1];
                    position--;
                }

                windowDistance[position] = squared;
                windowIndex[position] = j;
            }

            for (int c = 0; c < k; c++)
            {
                index[i, c] = windowIndex[c];
                distance[i, c] = Math.Sqrt(windowDistance[c]);
            }
        }

        return (index, distance);
    }
}
