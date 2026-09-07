namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// Gaussian mixture model fitted by expectation-maximisation.
/// <para>
/// Nothing here is trained ahead of time and nothing is shipped. Unlike a
/// learned model, a mixture computes its own parameters from the data in front
/// of it, on every solve — so this is an algorithm, not a set of weights, and it
/// belongs in C# rather than behind an ONNX boundary.
/// </para>
/// <para>
/// Two implementation choices are load-bearing rather than stylistic. All
/// arithmetic is in log space with a log-sum-exp normaliser: at six dimensions
/// with well-separated groups, computing densities and then dividing underflows
/// to zero and yields NaN responsibilities. And every covariance estimate gets a
/// floor added to its diagonal, without which a component collapsing onto
/// duplicate rows drives the likelihood to infinity.
/// </para>
/// <para>
/// Conventions follow scikit-learn's <c>GaussianMixture</c> — parameter
/// counting, the regularisation floor, and convergence on the change in mean
/// log-likelihood — so the two can be compared directly. See
/// <c>python/fixtures</c>.
/// </para>
/// </summary>
public static class GaussianMixture
{
    private const double Log2Pi = 1.8378770664093454835606594728112;

    /// <summary>
    /// Fits a mixture, restarting from several random initialisations and
    /// keeping whichever reached the highest log-likelihood.
    /// </summary>
    /// <param name="x">n x d data, rows are samples. Expected to be whitened already.</param>
    /// <param name="options">Fit settings. <see cref="GaussianMixtureOptions.Seed"/> makes this deterministic.</param>
    public static GaussianMixtureResult Fit(double[,] x, GaussianMixtureOptions options)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(options);

        int n = x.GetLength(0);
        options.Validate(n);

        var seeds = new Random(options.Seed);
        GaussianMixtureResult? best = null;

        for (int restart = 0; restart < options.Restarts; restart++)
        {
            var rng = new Random(seeds.Next());
            var candidate = FitOnce(x, options, InitialiseByKMeans(x, options, rng));

            // Strictly greater, so the earliest restart wins a tie. With a fixed
            // seed that makes the whole thing reproducible down to which start
            // produced the answer.
            if (best is null || candidate.LogLikelihood > best.LogLikelihood)
                best = candidate;
        }

        return best!;
    }

    /// <summary>
    /// Fits from explicitly supplied starting parameters, with no restarts.
    /// <para>
    /// This exists to be testable. Comparing against scikit-learn from a random
    /// start conflates two different questions — whether the EM arithmetic
    /// agrees, and whether the initialisation found the same local optimum.
    /// Pinning the start separates them: from identical parameters the two
    /// implementations should agree to machine precision, and any disagreement
    /// is a bug rather than a different hill.
    /// </para>
    /// </summary>
    public static GaussianMixtureResult FitFrom(
        double[,] x,
        GaussianMixtureOptions options,
        double[] mixingWeights,
        double[,] means,
        double[][,] covariances)
    {
        ArgumentNullException.ThrowIfNull(x);
        ArgumentNullException.ThrowIfNull(options);

        options.Validate(x.GetLength(0));

        var model = Model.FromCovariances(options.Covariance, mixingWeights, means, covariances);
        return FitOnce(x, options, model);
    }

    private static GaussianMixtureResult FitOnce(double[,] x, GaussianMixtureOptions options, Model model)
    {
        int n = x.GetLength(0);
        int k = options.Components;

        var responsibilities = new double[n, k];
        double previous = double.NegativeInfinity;
        double meanLogLikelihood = double.NegativeInfinity;
        bool converged = false;
        int iteration = 0;

        // E-step, M-step, then test — in that order, and the order matters. Test
        // before the M-step and the fit stops one update earlier than
        // scikit-learn's does, which shows up as a small parameter difference
        // that looks like an arithmetic bug and is not one.
        for (iteration = 1; iteration <= options.MaxIterations; iteration++)
        {
            meanLogLikelihood = model.EStep(x, responsibilities);
            model.MStep(x, responsibilities, options.RegularisationFloor);

            // First time round the previous bound is negative infinity, so the
            // change is infinite and the test cannot fire.
            double change = meanLogLikelihood - previous;
            previous = meanLogLikelihood;

            if (Math.Abs(change) < options.Tolerance)
            {
                converged = true;
                break;
            }
        }

        if (!converged)
            iteration = options.MaxIterations;

        // One last E-step so the responsibilities returned belong to the final
        // parameters rather than to the state one M-step behind them.
        meanLogLikelihood = model.EStep(x, responsibilities);

        return new GaussianMixtureResult(
            model.Weights,
            model.Means,
            model.CovarianceMatrices(),
            responsibilities,
            meanLogLikelihood * n,
            iteration,
            converged,
            ParameterCount(k, x.GetLength(1), options.Covariance));
    }

    /// <summary>
    /// Free parameters in the model: means, mixing weights (one is determined by
    /// the rest), and covariance entries. Counted exactly as scikit-learn counts
    /// them so BIC values are comparable.
    /// </summary>
    internal static int ParameterCount(int k, int d, CovarianceType covariance)
    {
        int covarianceParameters = covariance switch
        {
            CovarianceType.Spherical => k,
            CovarianceType.Diagonal => k * d,
            CovarianceType.Full => k * d * (d + 1) / 2,
            _ => throw new ArgumentOutOfRangeException(nameof(covariance)),
        };

        return covarianceParameters + k * d + (k - 1);
    }

    /// <summary>
    /// Seeds EM with a k-means++ start followed by Lloyd's algorithm, then turns
    /// the hard assignment into parameters via one M-step — the same route
    /// scikit-learn's default <c>init_params="kmeans"</c> takes.
    /// <para>
    /// k-means++ rather than uniform sampling is worth the twenty lines: picking
    /// seeds with probability proportional to squared distance from the nearest
    /// existing seed avoids the common failure where two seeds land in the same
    /// dense group and a real group is left with none.
    /// </para>
    /// </summary>
    private static Model InitialiseByKMeans(double[,] x, GaussianMixtureOptions options, Random rng)
    {
        int n = x.GetLength(0);
        int k = options.Components;

        var labels = KMeans(x, k, rng);

        var responsibilities = new double[n, k];
        for (int i = 0; i < n; i++)
            responsibilities[i, labels[i]] = 1.0;

        var model = Model.Empty(options.Covariance, k, x.GetLength(1));
        model.MStep(x, responsibilities, options.RegularisationFloor);
        return model;
    }

    private static int[] KMeans(double[,] x, int k, Random rng, int maxIterations = 100)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        var centres = KMeansPlusPlusSeeds(x, k, rng);
        var labels = new int[n];

        for (int iteration = 0; iteration < maxIterations; iteration++)
        {
            bool moved = false;

            for (int i = 0; i < n; i++)
            {
                int best = 0;
                double bestDistance = double.MaxValue;

                for (int c = 0; c < k; c++)
                {
                    double distance = 0.0;
                    for (int j = 0; j < d; j++)
                    {
                        double delta = x[i, j] - centres[c, j];
                        distance += delta * delta;
                    }

                    if (distance < bestDistance)
                    {
                        bestDistance = distance;
                        best = c;
                    }
                }

                if (labels[i] != best)
                {
                    labels[i] = best;
                    moved = true;
                }
            }

            if (iteration > 0 && !moved)
                break;

            RecomputeCentres(x, labels, centres, k);
        }

        return labels;
    }

    /// <summary>
    /// Greedy k-means++ seeding.
    /// <para>
    /// Plain k-means++ draws one candidate per seed, with probability
    /// proportional to its squared distance from the nearest seed already
    /// chosen. The greedy variant draws several, evaluates what each would do to
    /// the total squared distance, and keeps the best — which costs a few
    /// hundred extra operations and measurably improves the optimum EM
    /// subsequently climbs to.
    /// </para>
    /// <para>
    /// This is not a micro-optimisation. Fitting five components to data with
    /// four real families, plain sampling landed on a local optimum roughly two
    /// per cent worse by BIC than scikit-learn reached, consistently. The number
    /// of trials below is scikit-learn's, and it closes that gap.
    /// </para>
    /// </summary>
    private static double[,] KMeansPlusPlusSeeds(double[,] x, int k, Random rng)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        int trials = 2 + (int)Math.Log(k);

        var centres = new double[k, d];
        var nearest = new double[n];

        int first = rng.Next(n);
        for (int j = 0; j < d; j++)
            centres[0, j] = x[first, j];

        double potential = 0.0;
        for (int i = 0; i < n; i++)
        {
            nearest[i] = SquaredDistance(x, i, centres, 0, d);
            potential += nearest[i];
        }

        for (int c = 1; c < k; c++)
        {
            int best = -1;
            double bestPotential = double.MaxValue;

            for (int trial = 0; trial < trials; trial++)
            {
                int index = SampleProportionally(nearest, potential, rng, n);

                double trialPotential = 0.0;
                for (int i = 0; i < n; i++)
                    trialPotential += Math.Min(nearest[i], SquaredDistance(x, i, x, index, d));

                if (trialPotential < bestPotential)
                {
                    bestPotential = trialPotential;
                    best = index;
                }
            }

            for (int j = 0; j < d; j++)
                centres[c, j] = x[best, j];

            potential = 0.0;
            for (int i = 0; i < n; i++)
            {
                nearest[i] = Math.Min(nearest[i], SquaredDistance(x, i, x, best, d));
                potential += nearest[i];
            }
        }

        return centres;
    }

    /// <summary>
    /// Draws an index with probability proportional to <paramref name="weights"/>.
    /// Falls back to uniform when every weight is zero, which happens when every
    /// point coincides with a seed already chosen — rare in general, and routine
    /// when a model repeats the same member hundreds of times.
    /// </summary>
    private static int SampleProportionally(double[] weights, double total, Random rng, int n)
    {
        if (total <= 0.0 || double.IsNaN(total))
            return rng.Next(n);

        double target = rng.NextDouble() * total;
        double cumulative = 0.0;

        for (int i = 0; i < n; i++)
        {
            cumulative += weights[i];
            if (cumulative >= target)
                return i;
        }

        return n - 1;
    }

    private static double SquaredDistance(double[,] a, int rowA, double[,] b, int rowB, int d)
    {
        double sum = 0.0;
        for (int j = 0; j < d; j++)
        {
            double delta = a[rowA, j] - b[rowB, j];
            sum += delta * delta;
        }

        return sum;
    }

    private static void RecomputeCentres(double[,] x, int[] labels, double[,] centres, int k)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        var counts = new int[k];
        var sums = new double[k, d];

        for (int i = 0; i < n; i++)
        {
            counts[labels[i]]++;
            for (int j = 0; j < d; j++)
                sums[labels[i], j] += x[i, j];
        }

        for (int c = 0; c < k; c++)
        {
            if (counts[c] == 0)
            {
                // An empty cluster leaves a component with no data and a
                // degenerate covariance. Move it onto whichever point is worst
                // served by its current centre — the same repair k-means
                // implementations generally make, and it keeps k honest.
                MoveToWorstServedPoint(x, labels, centres, c);
                continue;
            }

            for (int j = 0; j < d; j++)
                centres[c, j] = sums[c, j] / counts[c];
        }
    }

    private static void MoveToWorstServedPoint(double[,] x, int[] labels, double[,] centres, int empty)
    {
        int n = x.GetLength(0);
        int d = x.GetLength(1);

        int worst = 0;
        double worstDistance = -1.0;

        for (int i = 0; i < n; i++)
        {
            int owner = labels[i];
            double distance = 0.0;
            for (int j = 0; j < d; j++)
            {
                double delta = x[i, j] - centres[owner, j];
                distance += delta * delta;
            }

            if (distance > worstDistance)
            {
                worstDistance = distance;
                worst = i;
            }
        }

        for (int j = 0; j < d; j++)
            centres[empty, j] = x[worst, j];

        labels[worst] = empty;
    }

    /// <summary>
    /// Mixture parameters plus the derived quantities the E-step needs.
    /// <para>
    /// Diagonal and spherical covariances take a separate code path from full
    /// ones, and not for tidiness: the diagonal log-density costs d operations
    /// per sample per component where the full one costs d squared. At the sizes
    /// this runs at — thousands of members, ten restarts, a couple of hundred
    /// iterations — that is the difference between a component that keeps up
    /// with a slider and one that does not.
    /// </para>
    /// </summary>
    private sealed class Model
    {
        private readonly CovarianceType _type;
        private readonly int _k;
        private readonly int _d;

        // Diagonal and spherical: variance per component per dimension.
        private readonly double[,] _variances;

        // Full: the covariance, plus the upper-triangular Cholesky factor of its
        // inverse. Storing the precision factor rather than the covariance is
        // what turns the Mahalanobis distance into a matrix-vector product.
        private readonly double[][,] _covariances;
        private readonly double[][,] _precisionChol;
        private readonly double[] _logDetPrecisionChol;

        public double[] Weights { get; }

        public double[,] Means { get; }

        private Model(CovarianceType type, int k, int d)
        {
            _type = type;
            _k = k;
            _d = d;

            Weights = new double[k];
            Means = new double[k, d];

            _variances = new double[k, d];
            _covariances = new double[k][,];
            _precisionChol = new double[k][,];
            _logDetPrecisionChol = new double[k];

            for (int c = 0; c < k; c++)
            {
                _covariances[c] = new double[d, d];
                _precisionChol[c] = new double[d, d];
            }
        }

        public static Model Empty(CovarianceType type, int k, int d) => new(type, k, d);

        public static Model FromCovariances(
            CovarianceType type, double[] weights, double[,] means, double[][,] covariances)
        {
            int k = weights.Length;
            int d = means.GetLength(1);
            var model = new Model(type, k, d);

            Array.Copy(weights, model.Weights, k);
            Array.Copy(means, model.Means, means.Length);

            for (int c = 0; c < k; c++)
            {
                if (type == CovarianceType.Full)
                {
                    Array.Copy(covariances[c], model._covariances[c], covariances[c].Length);
                }
                else
                {
                    for (int j = 0; j < d; j++)
                        model._variances[c, j] = covariances[c][j, j];
                }
            }

            model.RefreshDerived();
            return model;
        }

        /// <summary>Covariances as full matrices, whatever the fitted shape.</summary>
        public double[][,] CovarianceMatrices()
        {
            if (_type == CovarianceType.Full)
                return _covariances;

            var matrices = new double[_k][,];
            for (int c = 0; c < _k; c++)
            {
                matrices[c] = new double[_d, _d];
                for (int j = 0; j < _d; j++)
                    matrices[c][j, j] = _variances[c, j];
            }

            return matrices;
        }

        /// <summary>
        /// Posterior responsibility of every component for every sample, and the
        /// mean log-likelihood per sample as the by-product.
        /// </summary>
        public double EStep(double[,] x, double[,] responsibilities)
        {
            int n = x.GetLength(0);
            var logWeights = new double[_k];
            for (int c = 0; c < _k; c++)
                logWeights[c] = Math.Log(Weights[c]);

            var logProb = new double[_k];
            double totalLogLikelihood = 0.0;

            for (int i = 0; i < n; i++)
            {
                double max = double.NegativeInfinity;
                for (int c = 0; c < _k; c++)
                {
                    logProb[c] = logWeights[c] + LogDensity(x, i, c);
                    if (logProb[c] > max)
                        max = logProb[c];
                }

                // log-sum-exp, shifted by the maximum. Without the shift, six
                // dimensions of well-separated data underflow every term to zero
                // and the normaliser below divides zero by zero.
                double sum = 0.0;
                for (int c = 0; c < _k; c++)
                    sum += Math.Exp(logProb[c] - max);

                double logNormaliser = max + Math.Log(sum);
                totalLogLikelihood += logNormaliser;

                for (int c = 0; c < _k; c++)
                    responsibilities[i, c] = Math.Exp(logProb[c] - logNormaliser);
            }

            return totalLogLikelihood / n;
        }

        private double LogDensity(double[,] x, int i, int c)
        {
            if (_type == CovarianceType.Full)
            {
                double quadratic = 0.0;
                var u = _precisionChol[c];

                for (int col = 0; col < _d; col++)
                {
                    double y = 0.0;
                    // u is upper triangular, so only rows up to col contribute.
                    for (int row = 0; row <= col; row++)
                        y += (x[i, row] - Means[c, row]) * u[row, col];

                    quadratic += y * y;
                }

                return -0.5 * (_d * Log2Pi + quadratic) + _logDetPrecisionChol[c];
            }

            double sum = 0.0;
            for (int j = 0; j < _d; j++)
            {
                double delta = x[i, j] - Means[c, j];
                sum += delta * delta / _variances[c, j];
            }

            return -0.5 * (_d * Log2Pi + sum) + _logDetPrecisionChol[c];
        }

        /// <summary>
        /// Re-estimates weights, means and covariances from the current
        /// responsibilities. The regularisation floor is added to every
        /// covariance diagonal here, which is the only place it is applied.
        /// </summary>
        public void MStep(double[,] x, double[,] responsibilities, double regularisation)
        {
            int n = x.GetLength(0);

            // The nudge matches scikit-learn's, and stops a component that has
            // lost all its data from dividing by zero rather than shrinking.
            const double Nudge = 10.0 * 2.220446049250313e-16;

            var counts = new double[_k];
            for (int c = 0; c < _k; c++)
            {
                double sum = 0.0;
                for (int i = 0; i < n; i++)
                    sum += responsibilities[i, c];
                counts[c] = sum + Nudge;
            }

            for (int c = 0; c < _k; c++)
                Weights[c] = counts[c] / n;

            for (int c = 0; c < _k; c++)
            {
                for (int j = 0; j < _d; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < n; i++)
                        sum += responsibilities[i, c] * x[i, j];
                    Means[c, j] = sum / counts[c];
                }
            }

            if (_type == CovarianceType.Full)
                UpdateFullCovariances(x, responsibilities, counts, regularisation);
            else
                UpdateDiagonalVariances(x, responsibilities, counts, regularisation);

            RefreshDerived();
        }

        private void UpdateDiagonalVariances(
            double[,] x, double[,] responsibilities, double[] counts, double regularisation)
        {
            int n = x.GetLength(0);

            for (int c = 0; c < _k; c++)
            {
                for (int j = 0; j < _d; j++)
                {
                    double sum = 0.0;
                    for (int i = 0; i < n; i++)
                    {
                        double delta = x[i, j] - Means[c, j];
                        sum += responsibilities[i, c] * delta * delta;
                    }

                    _variances[c, j] = sum / counts[c] + regularisation;
                }

                // Spherical is the diagonal fit averaged across dimensions —
                // one radius rather than one per axis.
                if (_type == CovarianceType.Spherical)
                {
                    double mean = 0.0;
                    for (int j = 0; j < _d; j++)
                        mean += _variances[c, j];
                    mean /= _d;

                    for (int j = 0; j < _d; j++)
                        _variances[c, j] = mean;
                }
            }
        }

        private void UpdateFullCovariances(
            double[,] x, double[,] responsibilities, double[] counts, double regularisation)
        {
            int n = x.GetLength(0);

            for (int c = 0; c < _k; c++)
            {
                var cov = _covariances[c];
                Array.Clear(cov);

                for (int i = 0; i < n; i++)
                {
                    double r = responsibilities[i, c];
                    if (r == 0.0)
                        continue;

                    for (int j = 0; j < _d; j++)
                    {
                        double dj = x[i, j] - Means[c, j];
                        for (int m = j; m < _d; m++)
                            cov[j, m] += r * dj * (x[i, m] - Means[c, m]);
                    }
                }

                for (int j = 0; j < _d; j++)
                {
                    for (int m = j; m < _d; m++)
                    {
                        double value = cov[j, m] / counts[c];
                        cov[j, m] = value;
                        cov[m, j] = value;
                    }

                    cov[j, j] += regularisation;
                }
            }
        }

        /// <summary>
        /// Recomputes whatever the log-density needs from the covariances: the
        /// precision Cholesky factor for full fits, and the log-determinant term
        /// in every case.
        /// </summary>
        private void RefreshDerived()
        {
            if (_type != CovarianceType.Full)
            {
                for (int c = 0; c < _k; c++)
                {
                    double logDet = 0.0;
                    for (int j = 0; j < _d; j++)
                        logDet += Math.Log(_variances[c, j]);

                    // log |precision|^(1/2) = -0.5 log |covariance|
                    _logDetPrecisionChol[c] = -0.5 * logDet;
                }

                return;
            }

            for (int c = 0; c < _k; c++)
            {
                var lower = Cholesky(_covariances[c], _d);
                var inverse = InvertLowerTriangular(lower, _d);

                // Store (L^-1)^T, so that precision = U U^T with U upper
                // triangular and the quadratic form is a plain dot product.
                var u = _precisionChol[c];
                Array.Clear(u);
                for (int row = 0; row < _d; row++)
                    for (int col = row; col < _d; col++)
                        u[row, col] = inverse[col, row];

                double logDet = 0.0;
                for (int j = 0; j < _d; j++)
                    logDet += Math.Log(u[j, j]);

                _logDetPrecisionChol[c] = logDet;
            }
        }

        private static double[,] Cholesky(double[,] a, int d)
        {
            var l = new double[d, d];

            for (int i = 0; i < d; i++)
            {
                for (int j = 0; j <= i; j++)
                {
                    double sum = a[i, j];
                    for (int m = 0; m < j; m++)
                        sum -= l[i, m] * l[j, m];

                    if (i == j)
                    {
                        // The regularisation floor should have made this
                        // impossible. If it still happens the covariance is
                        // degenerate beyond what the floor covers, and failing
                        // loudly beats returning NaN responsibilities that look
                        // like a clustering.
                        if (sum <= 0.0)
                            throw new InvalidOperationException(
                                "Covariance is not positive definite. Raise the regularisation floor, "
                                + "reduce the component count, or drop near-constant columns.");

                        l[i, j] = Math.Sqrt(sum);
                    }
                    else
                    {
                        l[i, j] = sum / l[j, j];
                    }
                }
            }

            return l;
        }

        private static double[,] InvertLowerTriangular(double[,] l, int d)
        {
            var inverse = new double[d, d];

            for (int col = 0; col < d; col++)
            {
                inverse[col, col] = 1.0 / l[col, col];
                for (int row = col + 1; row < d; row++)
                {
                    double sum = 0.0;
                    for (int m = col; m < row; m++)
                        sum += l[row, m] * inverse[m, col];

                    inverse[row, col] = -sum / l[row, row];
                }
            }

            return inverse;
        }
    }
}
