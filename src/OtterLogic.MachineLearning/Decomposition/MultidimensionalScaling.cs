using OtterLogic.MachineLearning.Distances;

namespace OtterLogic.MachineLearning.Decomposition;

/// <summary>
/// Places samples as points so that the distances between the points match the
/// distances between the samples as closely as a few dimensions allow —
/// multidimensional scaling.
/// <para>
/// Two stages. <b>Classical scaling</b> (Torgerson, 1952) squares the distances,
/// double-centres them into the dot products the samples would have if they were
/// real points about their centroid, and takes the leading eigenvectors of that
/// matrix, each scaled by the root of its eigenvalue, as the coordinates. It is
/// exact and closed-form when the distances are those of real points. When they
/// are not — route distances on a graph, a disagreement between clusterings — the
/// dot products it fits are a proxy, so <b>stress majorisation</b> (SMACOF, de
/// Leeuw, 1977) then refines the map against the distances themselves, every step
/// guaranteed not to raise the stress.
/// </para>
/// <para>
/// Given features, the classical map <em>is</em> the projection onto the leading
/// principal components without whitening, so that is how it is computed: an
/// eigendecomposition of the d x d covariance rather than of an n x n matrix, which
/// is instant where the n x n route took twenty seconds at two thousand samples.
/// What this class adds over <see cref="PrincipalComponents"/> is that it also takes
/// any distance at all, and reports how far the map had to bend them to fit.
/// </para>
/// <para>
/// Given distances, the centred matrix B can have negative eigenvalues, and
/// <see cref="LeadingEigen"/> needs an operator with none. Shifting B by a bound on
/// the most negative is the usual repair, but a safe bound is loose, and it crowds
/// the leading eigenvalues together relative to the shifted spectrum — measured on a
/// thousand samples, 26 iterations and five seconds against 2 and 83 ms unshifted.
/// B² has no negative eigenvalues at all and <em>widens</em> the gaps instead, so the
/// solve runs on it and each eigenvalue's sign is read back from B afterwards. Memory
/// and refinement time are still quadratic in the samples.
/// </para>
/// </summary>
public static class MultidimensionalScaling
{
    /// <summary>Residual tolerance for the eigensolver: these vectors are coordinates someone looks at.</summary>
    private const double EigenTolerance = 1e-10;

    /// <summary>
    /// An eigenvalue below this share of the largest is read as zero: solver
    /// residue, not structure. Far below any real axis and far above the solver's
    /// own accuracy.
    /// </summary>
    private const double NegligibleShare = 1e-6;

    /// <summary>
    /// Eigenpairs sought beyond the axes wanted, so that a few large negative
    /// eigenvalues — which B² ranks alongside the positive ones — do not crowd a
    /// wanted axis out. Doubled until enough positive ones are found.
    /// </summary>
    private const int Headroom = 4;

    /// <summary>
    /// Maps samples by the straight-line distance between their feature rows.
    /// </summary>
    /// <param name="features">n x d, one row per sample, already prepared — columns in different units should be scaled first.</param>
    /// <param name="options">Settings; null for the defaults.</param>
    public static MultidimensionalScalingResult FromFeatures(double[,] features, MultidimensionalScalingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(features);
        options ??= new MultidimensionalScalingOptions();

        int n = features.GetLength(0);
        int d = features.GetLength(1);

        if (d < 1)
            throw new ArgumentException("The features have no columns.", nameof(features));
        if (n < 2)
            throw new ArgumentException($"Need at least two samples to map; got {n}.", nameof(features));

        for (int i = 0; i < n; i++)
            for (int j = 0; j < d; j++)
                if (!double.IsFinite(features[i, j]))
                    throw new ArgumentException($"features[{i}, {j}] is {features[i, j]}; features must be finite.", nameof(features));

        options.Validate(n);
        int k = options.Dimensions;
        var notes = new List<string>();

        if (d <= k)
            notes.Add($"The samples have only {d} column(s), so a {k}-dimensional map shows nothing the columns do "
                + "not already — it only rotates them. The map earns its place when there are more columns than axes.");

        // B's nonzero eigenvalues are the covariance's times n − 1, with the same
        // projections as coordinates, so every one is available here and the
        // retained share needs no Frobenius norm.
        var pca = PrincipalComponents.FitCount(features, d, whiten: false);
        var projected = pca.Transform(features);
        var all = pca.ExplainedVariance.Select(v => v * (n - 1)).ToArray();

        var eigenvalues = Enumerable.Range(0, k).Select(a => a < all.Length ? all[a] : 0.0).ToArray();
        var coordinates = new double[n, k];
        for (int i = 0; i < n; i++)
            for (int a = 0; a < Math.Min(k, projected.GetLength(1)); a++)
                coordinates[i, a] = projected[i, a];

        double totalSquares = all.Sum(v => v * v);
        double keptSquares = ZeroEmptyAxes(coordinates, eigenvalues, notes);

        var delta = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                delta[i, j] = delta[j, i] = Euclidean.Between(features, i, features, j);

        return Finish(delta, coordinates, eigenvalues, Share(keptSquares, totalSquares), double.NaN, options, notes);
    }

    /// <summary>
    /// Maps samples by a distance the caller has already worked out.
    /// </summary>
    /// <param name="distances">
    /// n x n: zero on the diagonal, the same both ways round, never negative. Any
    /// distance at all — a route length, a disagreement, a mixed-type dissimilarity.
    /// </param>
    /// <param name="options">Settings; null for the defaults.</param>
    public static MultidimensionalScalingResult FromDistances(double[,] distances, MultidimensionalScalingOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(distances);
        options ??= new MultidimensionalScalingOptions();

        var delta = Checked(distances);
        int n = delta.GetLength(0);

        if (n < 2)
            throw new ArgumentException($"Need at least two samples to map; got {n}.", nameof(distances));

        options.Validate(n);
        int k = options.Dimensions;
        var notes = new List<string>();

        var b = DoubleCentre(delta);

        double totalSquares = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                totalSquares += b[i, j] * b[i, j];

        if (totalSquares == 0.0)
        {
            notes.Add("Every distance is zero, so every sample sits on the same spot.");
            return new MultidimensionalScalingResult(
                new double[n, k], new double[k], 0.0, 0.0, 0.0, 0.0, new double[n], 0, true, notes);
        }

        var (positive, mostNegative) = Leading(b, k, options.Seed);

        var eigenvalues = positive.Select(pair => pair.Value).ToArray();
        var coordinates = new double[n, k];
        for (int a = 0; a < positive.Count; a++)
        {
            double scale = Math.Sqrt(Math.Max(positive[a].Value, 0.0));
            for (int i = 0; i < n; i++)
                coordinates[i, a] = positive[a].Vector[i] * scale;
        }

        double keptSquares = ZeroEmptyAxes(coordinates, eigenvalues, notes);
        double leading = Math.Max(eigenvalues[0], 0.0);

        if (mostNegative < -NegligibleShare * leading)
            notes.Add($"These distances are not those of points in any flat space (an eigenvalue of "
                + $"{mostNegative:G3}, against a largest of {leading:G3}), so no number of axes draws them exactly. "
                + "Stress says how far this map had to bend them.");

        return Finish(delta, coordinates, eigenvalues, Share(keptSquares, totalSquares), Math.Min(mostNegative, 0.0), options, notes);
    }

    /// <summary>
    /// The k most positive eigenpairs of B, and the most negative eigenvalue among
    /// those as large in size as any of them.
    /// <para>
    /// Solved on B², whose eigenvectors are B's and whose eigenvalues are their
    /// squares, so it ranks eigenpairs by size whatever their sign; each one's sign
    /// is its Rayleigh quotient on B. A negative eigenvalue this never meets is
    /// smaller in size than the weakest axis on the map.
    /// </para>
    /// </summary>
    private static (List<(double Value, double[] Vector)> Positive, double MostNegative) Leading(double[,] b, int k, int seed)
    {
        int n = b.GetLength(0);
        int count = Math.Min(n, k + Headroom);

        while (true)
        {
            var (_, vectors, _, _) = LeadingEigen.Solve(
                n, block => Product(b, Product(b, block)), count, seed, EigenTolerance);

            var positive = new List<(double Value, double[] Vector)>();
            double mostNegative = 0.0;

            for (int c = 0; c < count; c++)
            {
                var vector = new double[n];
                for (int i = 0; i < n; i++)
                    vector[i] = vectors[i, c];

                double value = RayleighQuotient(b, vector);
                if (value < 0.0)
                    mostNegative = Math.Min(mostNegative, value);
                else
                    positive.Add((value, vector));
            }

            if (positive.Count >= k || count == n)
            {
                positive = positive.OrderByDescending(pair => pair.Value).Take(k).ToList();
                while (positive.Count < k)
                    positive.Add((0.0, new double[n]));

                return (positive, mostNegative);
            }

            count = Math.Min(n, 2 * count);
        }
    }

    /// <summary>
    /// Zeroes the coordinates of every axis with nothing on it, says so, and returns
    /// the summed squares of the eigenvalues kept.
    /// </summary>
    private static double ZeroEmptyAxes(double[,] coordinates, double[] eigenvalues, List<string> notes)
    {
        int n = coordinates.GetLength(0);
        int k = eigenvalues.Length;
        double leading = Math.Max(eigenvalues[0], 0.0);
        double kept = 0.0;
        var empty = new List<int>();

        for (int a = 0; a < k; a++)
        {
            if (eigenvalues[a] > NegligibleShare * leading)
            {
                kept += eigenvalues[a] * eigenvalues[a];
                continue;
            }

            eigenvalues[a] = 0.0;
            empty.Add(a);
            for (int i = 0; i < n; i++)
                coordinates[i, a] = 0.0;
        }

        if (empty.Count > 0)
            notes.Add($"Axis {string.Join(", ", empty.Select(a => a + 1))} of {k} carries no structure: the samples "
                + $"already lie flat in {k - empty.Count} dimension(s), and a map with fewer axes shows everything.");

        return kept;
    }

    /// <summary>Stress of the classical map, refinement if asked for, and the result.</summary>
    private static MultidimensionalScalingResult Finish(
        double[,] delta,
        double[,] coordinates,
        double[] eigenvalues,
        double retained,
        double mostNegative,
        MultidimensionalScalingOptions options,
        List<string> notes)
    {
        double classicalStress = Stress(delta, coordinates, out _);
        double stress = classicalStress;
        int iterations = 0;
        bool converged = true;

        if (options.Refine && classicalStress > 0.0)
            (coordinates, stress, iterations, converged) = Majorise(delta, coordinates, options);

        if (!converged)
            notes.Add($"Refinement stopped at its cap of {options.MaximumIterations} steps while still improving. "
                + "The map is usable and no worse than the classical one, but not settled.");

        Stress(delta, coordinates, out var distortion);

        return new MultidimensionalScalingResult(
            coordinates, eigenvalues, retained, mostNegative, classicalStress, stress, distortion, iterations, converged, notes);
    }

    /// <summary>A distance table validated and made exactly symmetric.</summary>
    private static double[,] Checked(double[,] distances)
    {
        int n = distances.GetLength(0);
        if (distances.GetLength(1) != n)
            throw new ArgumentException(
                $"A distance table is square — one row and one column per sample — but this is {n} x {distances.GetLength(1)}.",
                nameof(distances));

        double largest = 0.0;
        for (int i = 0; i < n; i++)
        {
            for (int j = 0; j < n; j++)
            {
                double value = distances[i, j];
                if (!double.IsFinite(value) || value < 0.0)
                    throw new ArgumentException(
                        $"distances[{i}, {j}] is {value}; a distance must be finite and not negative. "
                        + "An unreachable pair on a graph needs a finite stand-in before it can be mapped.",
                        nameof(distances));
                largest = Math.Max(largest, value);
            }
        }

        // Relative to the largest distance so the check means the same thing in
        // millimetres and in metres; loose enough to forgive a table assembled on a
        // canvas from two passes that rounded differently.
        double slack = 1e-9 * Math.Max(1.0, largest);
        var symmetric = new double[n, n];

        for (int i = 0; i < n; i++)
        {
            if (distances[i, i] > slack)
                throw new ArgumentException(
                    $"Sample {i}'s distance to itself is {distances[i, i]}; it must be zero.", nameof(distances));

            for (int j = i + 1; j < n; j++)
            {
                if (Math.Abs(distances[i, j] - distances[j, i]) > slack)
                    throw new ArgumentException(
                        $"The distance from {i} to {j} is {distances[i, j]} but from {j} to {i} is {distances[j, i]}. "
                        + "A distance is the same both ways round.", nameof(distances));

                symmetric[i, j] = symmetric[j, i] = 0.5 * (distances[i, j] + distances[j, i]);
            }
        }

        return symmetric;
    }

    /// <summary>
    /// −½ J D² J: the squared distances with every row mean, every column mean
    /// added back and the grand mean removed. For distances between real points
    /// this is exactly their matrix of dot products about the centroid.
    /// </summary>
    private static double[,] DoubleCentre(double[,] delta)
    {
        int n = delta.GetLength(0);
        var rowMean = new double[n];
        double grand = 0.0;

        for (int i = 0; i < n; i++)
        {
            double sum = 0.0;
            for (int j = 0; j < n; j++)
                sum += delta[i, j] * delta[i, j];
            rowMean[i] = sum / n;
            grand += rowMean[i];
        }

        grand /= n;

        var b = new double[n, n];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < n; j++)
                b[i, j] = -0.5 * (delta[i, j] * delta[i, j] - rowMean[i] - rowMean[j] + grand);

        return b;
    }

    private static double[,] Product(double[,] b, double[,] block)
    {
        int n = b.GetLength(0);
        int p = block.GetLength(1);
        var image = new double[n, p];

        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < p; c++)
            {
                double sum = 0.0;
                for (int j = 0; j < n; j++)
                    sum += b[i, j] * block[j, c];
                image[i, c] = sum;
            }
        }

        return image;
    }

    private static double RayleighQuotient(double[,] b, double[] v)
    {
        int n = v.Length;
        double numerator = 0.0;
        double denominator = 0.0;

        for (int i = 0; i < n; i++)
        {
            double row = 0.0;
            for (int j = 0; j < n; j++)
                row += b[i, j] * v[j];
            numerator += v[i] * row;
            denominator += v[i] * v[i];
        }

        return denominator > 0.0 ? numerator / denominator : 0.0;
    }

    private static double Share(double kept, double total) => total > 0.0 ? Math.Clamp(kept / total, 0.0, 1.0) : 0.0;

    /// <summary>
    /// SMACOF: repeated Guttman transforms, each moving every point to the
    /// average of where each other point says it should be — along the line
    /// between them, at the true distance. Stress never rises from one step to
    /// the next, which is what makes starting from the classical map safe.
    /// <para>
    /// One pass over the pairs per step, each pair visited once: the distances it
    /// reads give both the current map's stress and the next map's positions.
    /// </para>
    /// </summary>
    private static (double[,] Coordinates, double Stress, int Iterations, bool Converged) Majorise(
        double[,] delta, double[,] start, MultidimensionalScalingOptions options)
    {
        int n = delta.GetLength(0);
        int k = start.GetLength(1);
        var x = (double[,])start.Clone();
        var next = new double[n, k];

        double scale = 0.0;
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                scale += delta[i, j] * delta[i, j];

        double previous = 0.0;

        for (int step = 0; step <= options.MaximumIterations; step++)
        {
            Array.Clear(next);
            double misfit = 0.0;

            for (int i = 0; i < n; i++)
            {
                for (int j = i + 1; j < n; j++)
                {
                    double d = Distance(x, i, j);
                    double error = delta[i, j] - d;
                    misfit += error * error;

                    if (d <= 0.0)
                        continue;

                    double ratio = delta[i, j] / d;
                    for (int a = 0; a < k; a++)
                    {
                        double pull = ratio * (x[i, a] - x[j, a]);
                        next[i, a] += pull;
                        next[j, a] -= pull;
                    }
                }
            }

            // misfit is the stress of x, the map this step started from. Stop on x
            // once the step into it bought too little; next is then discarded.
            if (step > 0 && previous - misfit <= options.Tolerance * previous)
                return (x, Math.Sqrt(misfit / scale), step, true);

            if (step == options.MaximumIterations)
                return (x, Math.Sqrt(misfit / scale), step, false);

            previous = misfit;

            for (int i = 0; i < n; i++)
                for (int a = 0; a < k; a++)
                    next[i, a] /= n;

            (x, next) = (next, x);
        }

        throw new InvalidOperationException("Unreachable: the loop returns on its last step.");
    }

    /// <summary>Kruskal's stress-1 of a map, and the same measure per sample.</summary>
    private static double Stress(double[,] delta, double[,] x, out double[] perSample)
    {
        int n = delta.GetLength(0);
        var misfit = new double[n];
        var scale = new double[n];

        for (int i = 0; i < n; i++)
        {
            for (int j = i + 1; j < n; j++)
            {
                double error = delta[i, j] - Distance(x, i, j);
                double squared = delta[i, j] * delta[i, j];

                misfit[i] += error * error;
                misfit[j] += error * error;
                scale[i] += squared;
                scale[j] += squared;
            }
        }

        perSample = new double[n];
        for (int i = 0; i < n; i++)
            perSample[i] = scale[i] > 0.0 ? Math.Sqrt(misfit[i] / scale[i]) : 0.0;

        // Every pair was counted from both ends, which cancels in the ratio.
        double totalMisfit = misfit.Sum();
        double totalScale = scale.Sum();
        return totalScale > 0.0 ? Math.Sqrt(totalMisfit / totalScale) : 0.0;
    }

    private static double Distance(double[,] x, int i, int j)
    {
        double sum = 0.0;
        for (int a = 0; a < x.GetLength(1); a++)
        {
            double difference = x[i, a] - x[j, a];
            sum += difference * difference;
        }

        return Math.Sqrt(sum);
    }
}
