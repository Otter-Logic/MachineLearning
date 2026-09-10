namespace OtterLogic.MachineLearning.Decomposition;

/// <summary>
/// Principal component analysis by exact eigendecomposition of the sample
/// covariance, with optional whitening.
/// <para>
/// At a handful of columns the job PCA is doing here is not dimensionality
/// reduction — it is <em>decorrelation and whitening</em>, plus dropping the
/// directions the data genuinely has no variance in. A flat case of a
/// three-dimensional quantity has columns that are identically zero, and a column
/// of zeros produces a singular covariance that stops EM dead. Retaining by
/// cumulative variance rather than a fixed count is what makes that case handle
/// itself.
/// </para>
/// <para>
/// It is also where per-column weighting becomes real. A mixture model with
/// free covariance is invariant to scaling a column: the fitted variance scales
/// with it and the factor cancels out of the responsibilities. An
/// eigendecomposition is not invariant — a column with more variance pulls the
/// principal axes toward itself. Weighting applied before this step changes the
/// rotation, and therefore changes the grouping.
/// </para>
/// <para>
/// Conventions follow scikit-learn's <c>PCA</c> so results can be compared
/// directly: sample covariance with denominator n-1, components as rows, and
/// whitening that divides by the square root of the explained variance.
/// </para>
/// </summary>
public sealed class PrincipalComponents
{
    /// <summary>Variance across <em>all</em> components, including those dropped.</summary>
    private readonly double _totalVariance;

    /// <summary>Whether <see cref="Transform"/> scales each component to unit variance.</summary>
    private readonly bool _whiten;

    private PrincipalComponents(
        double[] mean, double[,] components, double[] explainedVariance,
        double totalVariance, bool whiten)
    {
        Mean = mean;
        Components = components;
        ExplainedVariance = explainedVariance;
        _totalVariance = totalVariance;
        _whiten = whiten;
    }

    /// <summary>Column means of the training data, subtracted before projection.</summary>
    public double[] Mean { get; }

    /// <summary>
    /// The retained components, one per row: <c>Components[c, j]</c> is the
    /// loading of input column <c>j</c> on component <c>c</c>.
    /// </summary>
    public double[,] Components { get; }

    /// <summary>Variance along each retained component, descending.</summary>
    public double[] ExplainedVariance { get; }

    /// <summary>Number of components retained.</summary>
    public int Count => ExplainedVariance.Length;

    /// <summary>Fraction of the total variance the retained components carry.</summary>
    public double ExplainedVarianceRatio =>
        _totalVariance <= 0.0 ? 1.0 : ExplainedVariance.Sum() / _totalVariance;

    /// <summary>
    /// Fits the decomposition and decides how many components to keep.
    /// </summary>
    /// <param name="x">n x d data, rows are samples.</param>
    /// <param name="varianceThreshold">
    /// Keep the fewest leading components whose cumulative variance reaches this
    /// fraction of the total. 0.99 is a sensible default; 1.0 keeps everything
    /// that is not numerically degenerate.
    /// </param>
    /// <param name="whiten">
    /// Divide each component by its standard deviation, giving unit variance on
    /// every axis. This is what makes a diagonal covariance mixture reasonable.
    /// Note that whitening amplifies whatever is left in low-variance
    /// directions, which is precisely why the threshold must drop them first.
    /// </param>
    public static PrincipalComponents Fit(double[,] x, double varianceThreshold = 0.99, bool whiten = true)
    {
        var (mean, eigenvalues, eigenvectors, total) = Decompose(x);
        int keep = ChooseCount(eigenvalues, total, varianceThreshold);
        return Build(mean, eigenvalues, eigenvectors, total, keep, whiten);
    }

    /// <summary>
    /// Fits and keeps a fixed number of components, rather than however many a
    /// variance threshold asks for.
    /// <para>
    /// This is the overload to use when the downstream step needs a known
    /// dimensionality — a fixed projection every solve, so results stay
    /// comparable when the data changes slightly and a threshold would have
    /// silently kept a different number.
    /// </para>
    /// <para>
    /// The count is clamped to the columns available. Asking for three from data
    /// whose constant columns have already been dropped, leaving fewer than three,
    /// returns what there is rather than failing — check <see cref="Count"/>
    /// against what you asked for.
    /// </para>
    /// </summary>
    /// <param name="x">n x d data, rows are samples.</param>
    /// <param name="componentCount">How many components to keep, at least one.</param>
    /// <param name="whiten">Scale the components to unit variance.</param>
    public static PrincipalComponents FitCount(double[,] x, int componentCount, bool whiten = true)
    {
        if (componentCount < 1)
            throw new ArgumentOutOfRangeException(nameof(componentCount), componentCount,
                "Need at least one component.");

        var (mean, eigenvalues, eigenvectors, total) = Decompose(x);
        int keep = Math.Min(componentCount, eigenvalues.Length);
        return Build(mean, eigenvalues, eigenvectors, total, keep, whiten);
    }

    /// <summary>
    /// The eigendecomposition both overloads share — everything up to the point
    /// where they differ on how many components to keep.
    /// </summary>
    private static (double[] Mean, double[] Eigenvalues, double[,] Eigenvectors, double Total) Decompose(
        double[,] x)
    {
        ArgumentNullException.ThrowIfNull(x);

        int n = x.GetLength(0);
        int d = x.GetLength(1);
        if (n < 2)
            throw new ArgumentException("Need at least two samples to estimate a covariance.", nameof(x));

        var mean = new double[d];
        for (int j = 0; j < d; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
                sum += x[i, j];
            mean[j] = sum / n;
        }

        // Sample covariance, denominator n-1, matching numpy's default and
        // scikit-learn's explained_variance_.
        var cov = new double[d, d];
        for (int j = 0; j < d; j++)
        {
            for (int k = j; k < d; k++)
            {
                double sum = 0.0;
                for (int i = 0; i < n; i++)
                    sum += (x[i, j] - mean[j]) * (x[i, k] - mean[k]);

                double c = sum / (n - 1);
                cov[j, k] = c;
                cov[k, j] = c;
            }
        }

        var (eigenvalues, eigenvectors) = SymmetricEigen.Decompose(cov);

        // Rounding can push a genuinely zero eigenvalue slightly negative.
        for (int i = 0; i < d; i++)
            if (eigenvalues[i] < 0.0)
                eigenvalues[i] = 0.0;

        return (mean, eigenvalues, eigenvectors, eigenvalues.Sum());
    }

    private static PrincipalComponents Build(
        double[] mean, double[] eigenvalues, double[,] eigenvectors, double total, int keep, bool whiten)
    {
        int d = mean.Length;

        var components = new double[keep, d];
        var explained = new double[keep];
        for (int c = 0; c < keep; c++)
        {
            explained[c] = eigenvalues[c];
            for (int j = 0; j < d; j++)
                components[c, j] = eigenvectors[j, c];
        }

        return new PrincipalComponents(mean, components, explained, total, whiten);
    }

    /// <summary>
    /// How many leading components to retain. Anything carrying essentially no
    /// variance is dropped whatever the threshold asks for — whitening such a
    /// component divides by a number close to zero and turns rounding noise into
    /// a unit-variance axis the mixture then tries to explain.
    /// </summary>
    private static int ChooseCount(double[] eigenvalues, double total, double threshold)
    {
        if (total <= 0.0)
            return 1;

        double floor = total * 1e-10;
        int usable = eigenvalues.Count(v => v > floor);
        if (usable == 0)
            return 1;

        if (threshold >= 1.0)
            return usable;

        double cumulative = 0.0;
        for (int c = 0; c < usable; c++)
        {
            cumulative += eigenvalues[c];
            if (cumulative / total >= threshold)
                return c + 1;
        }

        return usable;
    }

    /// <summary>Projects n x d data onto the retained components, giving n x <see cref="Count"/>.</summary>
    public double[,] Transform(double[,] x)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);
        if (d != Mean.Length)
            throw new ArgumentException($"Expected {Mean.Length} columns, got {d}.", nameof(x));

        var z = new double[n, Count];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < Count; c++)
            {
                double sum = 0.0;
                for (int j = 0; j < d; j++)
                    sum += (x[i, j] - Mean[j]) * Components[c, j];

                z[i, c] = _whiten ? sum / Math.Sqrt(ExplainedVariance[c]) : sum;
            }
        }

        return z;
    }

    /// <summary>
    /// Maps points in component space back to input space.
    /// <para>
    /// This is what turns a cluster centre into something an engineer can read.
    /// It is exact only when every component was retained; anything the
    /// threshold dropped is gone, so a centre comes back as its projection onto
    /// the retained subspace. That is the honest answer rather than a lossless
    /// one, and at a 0.99 threshold the difference is not visible.
    /// </para>
    /// </summary>
    public double[,] InverseTransform(double[,] z)
    {
        int n = z.GetLength(0);
        if (z.GetLength(1) != Count)
            throw new ArgumentException($"Expected {Count} components, got {z.GetLength(1)}.", nameof(z));

        int d = Mean.Length;
        var x = new double[n, d];

        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < d; j++)
            {
                double sum = 0.0;
                for (int c = 0; c < Count; c++)
                {
                    double coefficient = _whiten
                        ? z[i, c] * Math.Sqrt(ExplainedVariance[c])
                        : z[i, c];
                    sum += coefficient * Components[c, j];
                }

                x[i, j] = sum + Mean[j];
            }
        }

        return x;
    }
}
