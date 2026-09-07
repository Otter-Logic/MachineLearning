namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// Eigendecomposition of a real symmetric matrix by cyclic Jacobi rotation.
/// <para>
/// Sixty lines, no dependency, and — the part that matters in a Grasshopper
/// component — completely deterministic. The obvious alternative, randomised
/// SVD, exists to approximate the leading components of matrices too large to
/// decompose exactly. A 6x6 covariance is not that, and randomisation would put
/// a seed and a run-to-run wobble into a result users expect to be stable.
/// </para>
/// <para>
/// Jacobi is also the numerically kindest option for small symmetric matrices:
/// it computes small eigenvalues to high relative accuracy, which QR-based
/// methods do not guarantee. That matters here because the smallest eigenvalues
/// are exactly the ones the variance threshold has to judge.
/// </para>
/// </summary>
internal static class SymmetricEigen
{
    /// <summary>
    /// Decomposes <paramref name="matrix"/> into eigenvalues and eigenvectors,
    /// sorted by descending eigenvalue.
    /// </summary>
    /// <param name="matrix">A symmetric d x d matrix. Not modified.</param>
    /// <returns>
    /// Eigenvalues descending, and eigenvectors as the <em>columns</em> of the
    /// returned matrix — <c>vectors[row, i]</c> is component <c>row</c> of the
    /// eigenvector for <c>values[i]</c>.
    /// </returns>
    internal static (double[] Values, double[,] Vectors) Decompose(double[,] matrix)
    {
        int d = matrix.GetLength(0);
        if (matrix.GetLength(1) != d)
            throw new ArgumentException("Matrix must be square.", nameof(matrix));

        // Work on a copy: a is driven toward diagonal, v accumulates the rotations.
        var a = (double[,])matrix.Clone();
        var v = Identity(d);

        // Fifty sweeps is far beyond what convergence needs — Jacobi is
        // quadratically convergent and d = 6 settles in three or four. The cap
        // is a guard against a non-symmetric matrix arriving by mistake.
        for (int sweep = 0; sweep < 50; sweep++)
        {
            double off = 0.0;
            for (int p = 0; p < d - 1; p++)
                for (int q = p + 1; q < d; q++)
                    off += a[p, q] * a[p, q];

            if (off <= 1e-30)
                break;

            for (int p = 0; p < d - 1; p++)
            {
                for (int q = p + 1; q < d; q++)
                {
                    if (Math.Abs(a[p, q]) < 1e-300)
                        continue;

                    // The rotation that zeroes a[p, q]. Taking the smaller root
                    // of t keeps the rotation angle under 45 degrees, which is
                    // what stops rounding error accumulating across sweeps.
                    double theta = (a[q, q] - a[p, p]) / (2.0 * a[p, q]);
                    double t = Math.Sign(theta) / (Math.Abs(theta) + Math.Sqrt(theta * theta + 1.0));
                    if (theta == 0.0)
                        t = 1.0;

                    double c = 1.0 / Math.Sqrt(t * t + 1.0);
                    double s = t * c;

                    Rotate(a, v, d, p, q, c, s);
                }
            }
        }

        var values = new double[d];
        for (int i = 0; i < d; i++)
            values[i] = a[i, i];

        return SortDescending(values, v, d);
    }

    private static void Rotate(double[,] a, double[,] v, int d, int p, int q, double c, double s)
    {
        double app = a[p, p];
        double aqq = a[q, q];
        double apq = a[p, q];

        a[p, p] = c * c * app - 2.0 * s * c * apq + s * s * aqq;
        a[q, q] = s * s * app + 2.0 * s * c * apq + c * c * aqq;
        a[p, q] = 0.0;
        a[q, p] = 0.0;

        for (int i = 0; i < d; i++)
        {
            if (i != p && i != q)
            {
                double aip = a[i, p];
                double aiq = a[i, q];
                a[i, p] = c * aip - s * aiq;
                a[p, i] = a[i, p];
                a[i, q] = s * aip + c * aiq;
                a[q, i] = a[i, q];
            }

            double vip = v[i, p];
            double viq = v[i, q];
            v[i, p] = c * vip - s * viq;
            v[i, q] = s * vip + c * viq;
        }
    }

    /// <summary>
    /// Sorts by descending eigenvalue and fixes each eigenvector's sign so the
    /// largest-magnitude entry is positive.
    /// <para>
    /// The sign fix is not cosmetic. An eigenvector and its negation are equally
    /// valid, so without a convention the sign is whatever the arithmetic
    /// happened to produce — and a principal component that flips sign between
    /// two nearly identical models makes every downstream group index jump.
    /// It is also what lets these results be compared against numpy's, which
    /// applies no such convention of its own.
    /// </para>
    /// </summary>
    private static (double[] Values, double[,] Vectors) SortDescending(double[] values, double[,] vectors, int d)
    {
        var order = Enumerable.Range(0, d).OrderByDescending(i => values[i]).ToArray();

        var sortedValues = new double[d];
        var sortedVectors = new double[d, d];

        for (int i = 0; i < d; i++)
        {
            int from = order[i];
            sortedValues[i] = values[from];

            int largest = 0;
            for (int r = 1; r < d; r++)
                if (Math.Abs(vectors[r, from]) > Math.Abs(vectors[largest, from]))
                    largest = r;

            double sign = vectors[largest, from] < 0.0 ? -1.0 : 1.0;

            for (int r = 0; r < d; r++)
                sortedVectors[r, i] = sign * vectors[r, from];
        }

        return (sortedValues, sortedVectors);
    }

    private static double[,] Identity(int d)
    {
        var m = new double[d, d];
        for (int i = 0; i < d; i++)
            m[i, i] = 1.0;
        return m;
    }
}
