namespace OtterLogic.MachineLearning.Decomposition;

/// <summary>
/// The few largest eigenpairs of a large symmetric positive semi-definite
/// operator, by Chebyshev-filtered subspace iteration with a Rayleigh-Ritz step
/// (Zhou and Saad, 2007).
/// <para>
/// <see cref="SymmetricEigen"/> decomposes a matrix completely, which is right
/// for a covariance with a handful of columns and hopeless for an n x n graph
/// operator at a few thousand samples — Jacobi is cubic, and the matrix would not
/// even be held densely. Spectral methods want only the leading handful of
/// eigenvectors, so this finds exactly those and never forms the matrix: the
/// operator arrives as a product, which for a sparse graph costs one pass over its
/// edges.
/// </para>
/// <para>
/// A block method rather than Lanczos, because a block finds a repeated
/// eigenvalue as many times as it is repeated. A graph in three disconnected
/// pieces has the eigenvalue one three times over, and single-vector Lanczos
/// would find it once — a real input here, not an edge case. It carries more
/// vectors than asked for, since the rate is set by the gap between the last
/// wanted eigenvalue and the first one <em>outside</em> the block, and when the
/// block is as wide as the operator the Rayleigh-Ritz step is an exact dense
/// decomposition, so small problems finish in one iteration with no separate path.
/// </para>
/// <para>
/// The Chebyshev filter is what makes it fast enough to sit behind a slider.
/// Plain subspace iteration multiplies each unwanted direction down by roughly
/// the ratio of its eigenvalue to the wanted ones — and a normalised graph
/// operator's leading eigenvalues sit close together near one, so that ratio is
/// close to one and the iteration crawls. A polynomial that stays small on the
/// unwanted part of the spectrum and grows steeply above it does many iterations'
/// worth of separation per step. Measured on a spectral clustering of 3,000
/// samples in five clusters: 1,116 plain iterations and eleven seconds, against 13
/// filtered iterations and under half a second for the whole clustering.
/// </para>
/// <para>
/// Deterministic for a given seed. The start is pseudo-random because a
/// structured start can be orthogonal to an eigenvector it is meant to find, but
/// the seed is fixed, so repeated calls on the same operator agree to the bit.
/// </para>
/// </summary>
public static class LeadingEigen
{
    /// <summary>
    /// Degree of the Chebyshev filter — operator products per iteration.
    /// </summary>
    private const int FilterDegree = 10;

    /// <summary>
    /// Finds the <paramref name="count"/> largest eigenvalues and their
    /// eigenvectors.
    /// </summary>
    /// <param name="size">Dimension n of the operator.</param>
    /// <param name="multiply">
    /// Applies the operator to an n x p block of column vectors, returning n x p.
    /// It must be symmetric and positive semi-definite: "largest" here is largest
    /// in magnitude, and those coincide only when nothing is negative. Shift the
    /// operator first if it is not.
    /// </param>
    /// <param name="count">How many eigenpairs, between 1 and <paramref name="size"/>.</param>
    /// <param name="seed">Seed for the starting block.</param>
    /// <param name="tolerance">
    /// Converged when every wanted Ritz pair's residual <c>|Av - θv|</c> is below this
    /// fraction of the largest eigenvalue.
    /// </param>
    /// <param name="maxIterations">Cap on iterations. Hitting it returns the best estimate and says so.</param>
    /// <returns>
    /// Eigenvalues descending, eigenvectors as unit-length <em>columns</em> with the
    /// largest-magnitude entry of each made positive — the same conventions as
    /// <see cref="SymmetricEigen"/>.
    /// </returns>
    public static (double[] Values, double[,] Vectors, int Iterations, bool Converged) Solve(
        int size,
        Func<double[,], double[,]> multiply,
        int count,
        int seed = 1,
        double tolerance = 1e-8,
        int maxIterations = 3000)
    {
        ArgumentNullException.ThrowIfNull(multiply);
        if (size < 1)
            throw new ArgumentOutOfRangeException(nameof(size), size, "The operator needs at least one dimension.");
        if (count < 1 || count > size)
            throw new ArgumentOutOfRangeException(nameof(count), count, $"Can find between 1 and {size} eigenpairs.");
        if (tolerance <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(tolerance), tolerance, "Tolerance must be positive.");
        if (maxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(maxIterations), maxIterations, "Need at least one iteration.");

        int n = size;

        // Oversample by at least as many again, and never by fewer than sixteen.
        // Error in the wanted pairs shrinks each iteration by the ratio of the
        // first eigenvalue outside the block to the last one inside it, so every
        // extra column pushes that ratio further down the spectrum — and at these
        // widths the block's own arithmetic is trivial beside one operator pass.
        int width = Math.Min(n, Math.Max(2 * count, count + 16));

        var rng = new Random(seed);
        var basis = new double[n, width];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < width; c++)
                basis[i, c] = rng.NextDouble() - 0.5;

        Orthonormalise(basis, rng);

        double[,] Apply(double[,] block)
        {
            var image = multiply(block);
            if (image.GetLength(0) != n || image.GetLength(1) != width)
                throw new InvalidOperationException(
                    $"The operator returned {image.GetLength(0)} x {image.GetLength(1)}; expected {n} x {width}.");

            return image;
        }

        double[] values;
        double[,] vectors;
        bool converged = false;
        int iteration = 0;

        while (true)
        {
            iteration++;
            var image = Apply(basis);

            // Rayleigh-Ritz: the best approximations to eigenpairs that the
            // current subspace can express, from a width x width problem.
            var projected = TransposeTimes(basis, image);
            for (int a = 0; a < width; a++)
                for (int b = a + 1; b < width; b++)
                    projected[a, b] = projected[b, a] = 0.5 * (projected[a, b] + projected[b, a]);

            var (ritzValues, ritzVectors) = SymmetricEigen.Decompose(projected);

            vectors = Times(basis, ritzVectors);
            var imageOfVectors = Times(image, ritzVectors);
            values = ritzValues;

            double scale = Math.Max(Math.Abs(values[0]), double.Epsilon);
            double worst = 0.0;
            for (int c = 0; c < count; c++)
            {
                double squares = 0.0;
                for (int i = 0; i < n; i++)
                {
                    double r = imageOfVectors[i, c] - values[c] * vectors[i, c];
                    squares += r * r;
                }

                worst = Math.Max(worst, Math.Sqrt(squares));
            }

            if (worst <= tolerance * scale)
            {
                converged = true;
                break;
            }

            if (iteration >= maxIterations)
                break;

            // Next subspace: the current Ritz vectors, filtered. Everything between
            // zero — the floor of a positive semi-definite spectrum — and the
            // smallest Ritz value still in the block is unwanted, and the filter
            // damps exactly that interval while amplifying everything above it.
            double floor = 0.0;
            double ceiling = values[width - 1];

            basis = ceiling - floor > 1e-12 * scale
                ? ChebyshevFilter(Apply, vectors, imageOfVectors, floor, ceiling)
                : imageOfVectors;

            Orthonormalise(basis, rng);
        }

        var resultValues = new double[count];
        var resultVectors = new double[n, count];
        for (int c = 0; c < count; c++)
        {
            resultValues[c] = values[c];

            int largest = 0;
            for (int i = 1; i < n; i++)
                if (Math.Abs(vectors[i, c]) > Math.Abs(vectors[largest, c]))
                    largest = i;

            // An eigenvector and its negation are equally valid; without a
            // convention, the sign is whatever the iteration happened to produce.
            double sign = vectors[largest, c] < 0.0 ? -1.0 : 1.0;
            double norm = 0.0;
            for (int i = 0; i < n; i++)
                norm += vectors[i, c] * vectors[i, c];
            norm = Math.Sqrt(norm);

            for (int i = 0; i < n; i++)
                resultVectors[i, c] = norm > 0.0 ? sign * vectors[i, c] / norm : 0.0;
        }

        return (resultValues, resultVectors, iteration, converged);
    }

    /// <summary>
    /// Applies the degree-<see cref="FilterDegree"/> Chebyshev polynomial that is
    /// bounded by one on <c>[floor, ceiling]</c> and grows fast above it, by the
    /// three-term recurrence. The operator's image of the block is already known
    /// from the Rayleigh-Ritz step, which saves one product.
    /// </summary>
    private static double[,] ChebyshevFilter(
        Func<double[,], double[,]> apply, double[,] block, double[,] imageOfBlock, double floor, double ceiling)
    {
        int n = block.GetLength(0);
        int width = block.GetLength(1);

        double half = 0.5 * (ceiling - floor);
        double centre = 0.5 * (ceiling + floor);

        var previous = block;
        var current = new double[n, width];
        for (int i = 0; i < n; i++)
            for (int c = 0; c < width; c++)
                current[i, c] = (imageOfBlock[i, c] - centre * block[i, c]) / half;

        for (int degree = 2; degree <= FilterDegree; degree++)
        {
            var image = apply(current);
            var next = new double[n, width];
            for (int i = 0; i < n; i++)
                for (int c = 0; c < width; c++)
                    next[i, c] = 2.0 * (image[i, c] - centre * current[i, c]) / half - previous[i, c];

            previous = current;
            current = next;
        }

        return current;
    }

    /// <summary>
    /// Modified Gram-Schmidt, run twice. One pass loses orthogonality in exactly
    /// the case this solver exists for — nearly parallel columns as they converge
    /// onto the same dominant direction — and a second pass restores it.
    /// <para>
    /// A column that vanishes has collapsed into the span of the ones before it,
    /// which happens routinely once the block holds an exact invariant subspace. It
    /// is replaced with a fresh random direction rather than left at zero, which
    /// would otherwise sit in the block contributing nothing for the rest of the
    /// run.
    /// </para>
    /// </summary>
    private static void Orthonormalise(double[,] q, Random rng)
    {
        int n = q.GetLength(0);
        int width = q.GetLength(1);

        for (int c = 0; c < width; c++)
        {
            for (int attempt = 0; attempt < 3; attempt++)
            {
                double before = ColumnNorm(q, c);

                for (int pass = 0; pass < 2; pass++)
                {
                    for (int prior = 0; prior < c; prior++)
                    {
                        double dot = 0.0;
                        for (int i = 0; i < n; i++)
                            dot += q[i, prior] * q[i, c];
                        for (int i = 0; i < n; i++)
                            q[i, c] -= dot * q[i, prior];
                    }
                }

                double after = ColumnNorm(q, c);
                if (after > 1e-10 * Math.Max(before, 1e-300))
                {
                    for (int i = 0; i < n; i++)
                        q[i, c] /= after;
                    break;
                }

                for (int i = 0; i < n; i++)
                    q[i, c] = rng.NextDouble() - 0.5;
            }
        }
    }

    private static double ColumnNorm(double[,] q, int c)
    {
        double sum = 0.0;
        for (int i = 0; i < q.GetLength(0); i++)
            sum += q[i, c] * q[i, c];

        return Math.Sqrt(sum);
    }

    /// <summary><c>A^T B</c> for two n x p blocks.</summary>
    private static double[,] TransposeTimes(double[,] a, double[,] b)
    {
        int n = a.GetLength(0);
        int p = a.GetLength(1);
        var result = new double[p, p];

        for (int i = 0; i < n; i++)
            for (int r = 0; r < p; r++)
            {
                double air = a[i, r];
                for (int c = 0; c < p; c++)
                    result[r, c] += air * b[i, c];
            }

        return result;
    }

    /// <summary><c>A S</c> for an n x p block and a p x p matrix.</summary>
    private static double[,] Times(double[,] a, double[,] s)
    {
        int n = a.GetLength(0);
        int p = a.GetLength(1);
        var result = new double[n, p];

        for (int i = 0; i < n; i++)
            for (int m = 0; m < p; m++)
            {
                double aim = a[i, m];
                if (aim == 0.0)
                    continue;

                for (int c = 0; c < p; c++)
                    result[i, c] += aim * s[m, c];
            }

        return result;
    }
}
