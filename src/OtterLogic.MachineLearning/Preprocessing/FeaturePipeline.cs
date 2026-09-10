namespace OtterLogic.MachineLearning.Preprocessing;

/// <summary>
/// The preprocessing that runs before PCA and the mixture: row normalisation,
/// log transform, standardisation, and per-column weights.
/// <para>
/// This is where most of the quality of a grouping is decided, and it is the
/// part no library hands you. A mixture fitted to raw measurements will give an
/// answer; it will often be a poor one, because real inputs break two assumptions
/// at once — magnitudes are heavily right-skewed where the model expects
/// something Gaussian, and columns in different units sit side by side with no
/// shared scale.
/// </para>
/// <para>
/// <em>Which</em> of these steps suits what data is deliberately left to the
/// caller, so every switch defaults off. Whether a column is skewed enough to log,
/// or whether size or only shape should count, is a claim about what the columns
/// mean — and that belongs to the toolkit that knows.
/// </para>
/// </summary>
public sealed class FeaturePipeline
{
    private readonly bool _normaliseRows;
    private readonly bool _logTransform;
    private readonly double[] _centre;
    private readonly double[] _scale;
    private readonly double[] _weights;

    private FeaturePipeline(
        bool normaliseRows, bool logTransform,
        int[] keptColumns, double[] centre, double[] scale, double[] weights)
    {
        _normaliseRows = normaliseRows;
        _logTransform = logTransform;
        _centre = centre;
        _scale = scale;
        _weights = weights;
        KeptColumns = keptColumns;
    }

    /// <summary>
    /// Indices of the input columns that survived. A column with no variance
    /// carries no information and produces a singular covariance, so it is
    /// dropped rather than regularised into pretending otherwise. Any flat case
    /// of a quantity measured in three dimensions hits this every time, in the
    /// columns for the dimension it lacks.
    /// </summary>
    public int[] KeptColumns { get; }

    /// <summary>
    /// Fits the transform.
    /// </summary>
    /// <param name="x">n x d raw data.</param>
    /// <param name="logTransform">
    /// Apply log(1 + x). Worth having on for right-skewed magnitudes: a handful
    /// of very large values and a long tail of small ones is not a shape a
    /// Gaussian describes, and without this one component swallows the tail while
    /// the rest split hairs among the small values. Off by default.
    /// </param>
    /// <param name="normaliseRows">
    /// Scale each row to unit length first, discarding overall magnitude and
    /// keeping only the proportion between columns. This is the difference
    /// between grouping samples that are alike in size and grouping samples that
    /// are alike in <em>shape</em> whatever their size. It changes the answer more
    /// than any other switch here.
    /// </param>
    /// <param name="weights">
    /// Per-column multipliers applied after standardisation, so a weight of one
    /// means "count this the same as the others". Null means all ones.
    /// <para>
    /// These do their work through the eigendecomposition that follows, not
    /// through the mixture. A mixture with free covariance is invariant to
    /// scaling a column — the fitted variance absorbs it and the responsibilities
    /// are unchanged. An eigendecomposition is not: more variance on a column
    /// pulls the principal axes toward it. So weighting here is real, and
    /// weighting a mixture with no PCA in front of it would not be.
    /// </para>
    /// </param>
    public static FeaturePipeline Fit(
        double[,] x, bool logTransform = false, bool normaliseRows = false, double[]? weights = null)
    {
        ArgumentNullException.ThrowIfNull(x);

        int n = x.GetLength(0);
        int d = x.GetLength(1);

        if (weights is not null && weights.Length != d)
            throw new ArgumentException($"Expected {d} weights, got {weights.Length}.", nameof(weights));

        var staged = ApplyRowAndLog(x, logTransform, normaliseRows);

        var centre = new double[d];
        var scale = new double[d];
        for (int j = 0; j < d; j++)
        {
            double sum = 0.0;
            for (int i = 0; i < n; i++)
                sum += staged[i, j];
            centre[j] = sum / n;

            double squares = 0.0;
            for (int i = 0; i < n; i++)
            {
                double delta = staged[i, j] - centre[j];
                squares += delta * delta;
            }

            // Population standard deviation, matching scikit-learn's StandardScaler.
            scale[j] = Math.Sqrt(squares / n);
        }

        var effectiveWeights = new double[d];
        for (int j = 0; j < d; j++)
            effectiveWeights[j] = weights?[j] ?? 1.0;

        // A column is dropped when it is constant, or when its weight asks for
        // it to be. The scale threshold is relative to the column's own centre
        // so it means the same thing whatever units the column is in.
        var kept = new List<int>(d);
        for (int j = 0; j < d; j++)
        {
            double reference = Math.Max(1.0, Math.Abs(centre[j]));
            bool constant = scale[j] <= 1e-12 * reference;
            bool excluded = effectiveWeights[j] == 0.0;

            if (!constant && !excluded)
                kept.Add(j);
        }

        if (kept.Count == 0)
            throw new InvalidOperationException(
                "Every column is constant or zero-weighted, so there is nothing to cluster on.");

        return new FeaturePipeline(
            normaliseRows, logTransform, kept.ToArray(), centre, scale, effectiveWeights);
    }

    /// <summary>Applies the fitted transform, returning n x <see cref="KeptColumns"/>.Length.</summary>
    public double[,] Transform(double[,] x)
    {
        int n = x.GetLength(0);
        var staged = ApplyRowAndLog(x, _logTransform, _normaliseRows);

        var z = new double[n, KeptColumns.Length];
        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < KeptColumns.Length; c++)
            {
                int j = KeptColumns[c];
                z[i, c] = (staged[i, j] - _centre[j]) / _scale[j] * _weights[j];
            }
        }

        return z;
    }

    /// <summary>
    /// Maps transformed points back to the original units, so a cluster centre
    /// can be read in the terms the data arrived in.
    /// <para>
    /// Dropped columns come back as their constant value. Row normalisation
    /// cannot be undone — the magnitude it divided out is gone — so with that
    /// switch on, a centre returns as a unit-length direction describing the
    /// proportion between columns rather than their size. That is still the right
    /// answer to the question that switch asks.
    /// </para>
    /// </summary>
    public double[,] InverseTransform(double[,] z)
    {
        int n = z.GetLength(0);
        int d = _centre.Length;

        if (z.GetLength(1) != KeptColumns.Length)
            throw new ArgumentException(
                $"Expected {KeptColumns.Length} columns, got {z.GetLength(1)}.", nameof(z));

        var x = new double[n, d];

        for (int i = 0; i < n; i++)
        {
            // A dropped column was constant, so its centre is that constant.
            for (int j = 0; j < d; j++)
                x[i, j] = _logTransform ? Math.Exp(_centre[j]) - 1.0 : _centre[j];

            for (int c = 0; c < KeptColumns.Length; c++)
            {
                int j = KeptColumns[c];
                double value = z[i, c] / _weights[j] * _scale[j] + _centre[j];
                x[i, j] = _logTransform ? Math.Exp(value) - 1.0 : value;
            }
        }

        return x;
    }

    private static double[,] ApplyRowAndLog(double[,] x, bool logTransform, bool normaliseRows)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);
        var staged = new double[n, d];

        for (int i = 0; i < n; i++)
        {
            double factor = 1.0;

            if (normaliseRows)
            {
                double squares = 0.0;
                for (int j = 0; j < d; j++)
                    squares += x[i, j] * x[i, j];

                double norm = Math.Sqrt(squares);
                // An all-zero row has no direction. Leaving it at the origin is
                // the honest answer; it will land in whichever component covers
                // the origin and its confidence will say so.
                factor = norm > 0.0 ? 1.0 / norm : 1.0;
            }

            for (int j = 0; j < d; j++)
            {
                double value = x[i, j] * factor;

                if (logTransform)
                {
                    if (value < 0.0)
                        throw new ArgumentException(
                            "The log transform expects non-negative values. If these are signed, "
                            + "take magnitudes first or turn the log transform off.", nameof(x));

                    value = Math.Log(1.0 + value);
                }

                staged[i, j] = value;
            }
        }

        return staged;
    }
}
