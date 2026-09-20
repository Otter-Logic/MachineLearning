using OtterLogic.MachineLearning.Decomposition;

namespace OtterLogic.MachineLearning.Shapes;

/// <summary>
/// Describes every outline in a population by the same row of numbers, learned
/// from the population rather than chosen in advance.
/// <para>
/// The usual way to compare shapes is to pick a handful of measurements — length,
/// width, area, how far off-centre — and compare those. It works until it does
/// not: two outlines can agree on every one of those and still be a rectangle
/// notched at its corners and a rectangle notched in its middle. What is missing
/// is never the arithmetic, it is that somebody had to decide in advance what
/// mattered.
/// </para>
/// <para>
/// This decides instead. Every outline is read at the same count of points, the
/// population's own average shape is found, and the directions it actually varies
/// in fall out of a decomposition. A shape is then <em>where it sits</em> among
/// the others, which catches a notch, a curve or a raked corner without any of
/// them having been anticipated.
/// </para>
/// <para>
/// Three steps, in order:
/// </para>
/// <list type="number">
/// <item><b>Read</b> — resample each outline at a fixed count of points, evenly
/// along its length, so a four-corner shape and a forty-corner one are the same
/// size of thing to compare.</item>
/// <item><b>Align</b> — take each off its own centre, and find which point of one
/// answers to which point of another. A closed outline has no natural first
/// point, so where the reading starts is searched for, and both ways round, since
/// which way an outline was drawn says nothing about its shape. Scale, turn and
/// mirror are only taken out if asked for. Each outline is aligned to the running
/// average and the average recomputed, until it stops moving.</item>
/// <item><b>Decompose</b> — principal components over the aligned coordinates.
/// The components are themselves shapes, which is what lets one be <em>looked
/// at</em> rather than only reported: see
/// <see cref="ShapeSignatureResult.Variation"/>.</item>
/// </list>
/// <para>
/// The scores come out scaled so that the distance between two of them is the
/// root-mean-square distance between the two outlines. A tolerance on this
/// signature is therefore a tolerance in the units the outlines came in, which is
/// what makes it safe to cut a clustering of them at a number somebody can
/// justify.
/// </para>
/// <para>
/// This knows nothing about what the outlines are of. Which of them count as the
/// same thing, and what to call them, is a judgement for whoever is asking.
/// </para>
/// </summary>
public static class ShapeSignature
{
    /// <summary>
    /// How finely the starting point is hunted down once the best whole step has
    /// been found. Sixty halvings of a step take it below anything the outlines
    /// were measured to; see <see cref="Refine"/> for why it cannot simply be read
    /// off.
    /// </summary>
    private const int Refinements = 60;

    /// <summary>
    /// Describes a population of outlines.
    /// </summary>
    /// <param name="outlines">Each outline as k x d points in order, k varying, d the same throughout and 2 or 3.</param>
    /// <param name="options">How finely to read them, what to take out before comparing, and how many numbers to describe one with.</param>
    public static ShapeSignatureResult Fit(IReadOnlyList<double[,]> outlines, ShapeSignatureOptions options)
    {
        ArgumentNullException.ThrowIfNull(outlines);
        ArgumentNullException.ThrowIfNull(options);

        if (outlines.Count == 0)
            throw new ArgumentException("Need at least one outline to describe.", nameof(outlines));

        int dimensions = Dimensions(outlines);
        options.Validate(dimensions);

        int points = options.Points, width = points * dimensions, n = outlines.Count;
        var notes = new List<string>();

        // 1. Read every outline as a path that can be sampled anywhere along its
        //    length, and take a first reading from each to measure it by.
        var paths = new Path[n];
        var read = new double[n][];
        var size = new double[n];
        int detailed = 0;

        for (int i = 0; i < n; i++)
        {
            if (outlines[i].GetLength(0) > points)
                detailed++;

            paths[i] = Path.Of(outlines[i], options.Closed, dimensions, i);
            var shape = paths[i].At(points, 0.0);
            Centre(shape, points, dimensions);
            size[i] = Radius(shape, points);
            read[i] = shape;
        }

        if (detailed > 0)
            notes.Add($"{detailed} outline(s) were drawn with more corners than they were read at. Raise Points to "
                + "keep detail this has smoothed over.");

        // 2. Align. Outlines that read exactly alike are one thing to align, not
        //    many — a population of real outlines is mostly repeats, and hunting for
        //    where two outlines answer to each other is the expensive step.
        var of = new List<int>();
        var repeats = new List<int>();
        var which = new int[n];
        var seen = new Dictionary<double[], int>(SameShape.Default);

        for (int i = 0; i < n; i++)
        {
            if (!seen.TryGetValue(read[i], out int at))
            {
                at = of.Count;
                seen[read[i]] = at;
                of.Add(i);
                repeats.Add(0);
            }

            which[i] = at;
            repeats[at]++;
        }

        var mean = Settle(read[of[0]], size[of[0]], points, options);
        var aligned = new double[of.Count][];

        for (int round = 0; round < options.Rounds; round++)
        {
            for (int u = 0; u < of.Count; u++)
                aligned[u] = Align(paths[of[u]], mean, points, dimensions, options);

            var next = Average(aligned, repeats, width);
            if (options.NormaliseScale)
                Rescale(next, points);

            // Root-mean-square, so the test is in the units the outlines came in
            // however many points they were read at.
            double moved = Math.Sqrt(Apart(next, mean) / points);
            mean = next;

            if (moved <= options.Tolerance)
                break;
        }

        // One last pass, so what is decomposed is aligned to the mean that is
        // reported rather than to the one before it.
        for (int u = 0; u < of.Count; u++)
            aligned[u] = Align(paths[of[u]], mean, points, dimensions, options);

        // 3. Decompose. Dividing by the square root of the point count is what puts
        //    the scores in root-mean-square units — see ShapeSignatureResult.Scores —
        //    and whitening would undo it, so it is never on.
        double spread = Math.Sqrt(points);
        var matrix = new double[n, width];
        for (int i = 0; i < n; i++)
        {
            var row = aligned[which[i]];
            for (int j = 0; j < width; j++)
                matrix[i, j] = row[j] / spread;
        }

        // A covariance needs two rows. One outline has no variation to describe
        // either way, so it is decomposed against a copy of itself: no variance, no
        // components, and the note below says as much.
        var fitted = n > 1 ? matrix : Twice(matrix, width);

        var components = options.Components > 0
            ? PrincipalComponents.FitCount(fitted, options.Components, whiten: false)
            : PrincipalComponents.Fit(fitted, options.Variance, whiten: false);

        var scores = components.Transform(matrix);
        var residual = new double[n];

        // A component with no spread is not a way the population varies, it is the
        // decomposition having nothing to report. Say so rather than handing back a
        // column of zeros and letting a caller read meaning into it.
        if (!components.ExplainedVariance.Any(variance => variance > 0.0))
            notes.Add("Every outline is the same shape, so there is nothing to describe them apart by.");

        if (components.Count > 0)
        {
            var rebuilt = components.InverseTransform(scores);
            for (int i = 0; i < n; i++)
            {
                double sum = 0.0;
                for (int j = 0; j < width; j++)
                {
                    double off = matrix[i, j] - rebuilt[i, j];
                    sum += off * off;
                }

                residual[i] = Math.Sqrt(sum);
            }
        }

        return new ShapeSignatureResult(components, scores, size, residual, points, dimensions, notes);
    }

    /// <summary>The first reading, made ready to be the mean the others align to.</summary>
    private static double[] Settle(double[] shape, double size, int points, ShapeSignatureOptions options)
    {
        var mean = (double[])shape.Clone();
        if (!options.NormaliseScale || size <= options.Tolerance)
            return mean;

        for (int j = 0; j < mean.Length; j++)
            mean[j] /= size;

        return mean;
    }

    private static double[,] Twice(double[,] matrix, int width)
    {
        var doubled = new double[2, width];
        for (int j = 0; j < width; j++)
            doubled[0, j] = doubled[1, j] = matrix[0, j];

        return doubled;
    }

    /// <summary>How many coordinates the outlines have, and that they all agree.</summary>
    private static int Dimensions(IReadOnlyList<double[,]> outlines)
    {
        if (outlines[0] is null)
            throw new ArgumentException("Outline 0 is missing.", nameof(outlines));

        int dimensions = outlines[0].GetLength(1);
        if (dimensions is not (2 or 3))
            throw new ArgumentException($"Outlines have {dimensions} coordinate(s) per point; they need 2 or 3.",
                nameof(outlines));

        for (int i = 0; i < outlines.Count; i++)
        {
            if (outlines[i] is null)
                throw new ArgumentException($"Outline {i} is missing.", nameof(outlines));
            if (outlines[i].GetLength(1) != dimensions)
                throw new ArgumentException(
                    $"Outline {i} has {outlines[i].GetLength(1)} coordinate(s) per point where outline 0 has {dimensions}.",
                    nameof(outlines));
        }

        return dimensions;
    }

    /// <summary>
    /// The best this outline can be made to answer to another: where its reading
    /// starts, which way round it runs, whether it is the mirror that matches, and
    /// how it is turned.
    /// </summary>
    private static double[] Align(
        Path path, double[] mean, int points, int dimensions, ShapeSignatureOptions options)
    {
        double[]? best = null;
        double closest = double.PositiveInfinity;

        foreach (var variant in Variants(path, options))
        {
            double[] Draw(double phase)
            {
                var shape = variant.At(points, phase);
                Centre(shape, points, dimensions);

                if (options.NormaliseScale)
                {
                    double radius = Radius(shape, points);
                    if (radius > options.Tolerance)
                        for (int j = 0; j < shape.Length; j++)
                            shape[j] /= radius;
                }

                if (options.NormaliseRotation)
                    Turn(shape, mean, points);

                return shape;
            }

            if (!options.Closed)
            {
                Keep(Draw(0.0));
                continue;
            }

            // Whole steps first, which are only a reordering of one reading and so
            // cost nothing to try.
            var start = Draw(0.0);
            double step = 1.0 / points;
            int at = 0;
            double here = double.PositiveInfinity;

            for (int k = 0; k < points; k++)
            {
                var shifted = From(start, k, points, dimensions);
                if (options.NormaliseRotation)
                    Turn(shifted, mean, points);

                double apart = Apart(shifted, mean);
                if (apart >= here)
                    continue;

                here = apart;
                at = k;
            }

            Keep(Refine(Draw, shape => Apart(shape, mean), at * step, 0.5 * step));
        }

        return best!;

        void Keep(double[] shape)
        {
            double apart = Apart(shape, mean);
            if (apart >= closest)
                return;

            closest = apart;
            best = shape;
        }
    }

    /// <summary>
    /// Hunts for the starting point between two whole steps.
    /// <para>
    /// This is not fussiness. Readings are spaced evenly <em>along the outline's
    /// length</em>, so where they fall depends on which corner the outline happened
    /// to be listed from: a corner a third of a step round puts every reading a
    /// third of a step out. On a 12 m outline read at thirty-two points that is 125
    /// millimetres between one shape and the very same shape listed from a different
    /// corner, which would make the signature a description of how somebody drew a
    /// thing rather than of what it is. Whole steps cannot fix it — only looking
    /// between them can.
    /// </para>
    /// <para>
    /// Golden section, because near a match the distance falls off straight rather
    /// than smoothly: a reading out by a little is out by proportionately that much,
    /// so there is no curvature to solve for and the interval simply has to be
    /// closed. Sixty closings take a step below anything the outlines were measured
    /// to.
    /// </para>
    /// </summary>
    private static double[] Refine(
        Func<double, double[]> draw, Func<double[], double> apart, double around, double reach)
    {
        const double Golden = 0.618033988749895;
        double low = around - reach, high = around + reach;

        double a = high - Golden * (high - low), b = low + Golden * (high - low);
        var drawnA = draw(a);
        var drawnB = draw(b);
        double atA = apart(drawnA), atB = apart(drawnB);

        for (int i = 0; i < Refinements; i++)
        {
            if (atA < atB)
            {
                (high, b, drawnB, atB) = (b, a, drawnA, atA);
                a = high - Golden * (high - low);
                drawnA = draw(a);
                atA = apart(drawnA);
            }
            else
            {
                (low, a, drawnA, atA) = (a, b, drawnB, atB);
                b = low + Golden * (high - low);
                drawnB = draw(b);
                atB = apart(drawnB);
            }
        }

        return atA < atB ? drawnA : drawnB;
    }

    /// <summary>The ways this outline may be read, all of which are the same shape.</summary>
    private static IEnumerable<Path> Variants(Path path, ShapeSignatureOptions options)
    {
        foreach (var way in Ways(path, options))
        {
            yield return way;

            for (int turn = 1; turn < options.Turns; turn++)
                yield return way.Turned(2.0 * Math.PI * turn / options.Turns);
        }
    }

    /// <summary>
    /// The outline read forwards and backwards — which way round it was drawn says
    /// nothing about its shape — and its mirror the same two ways where a mirror
    /// counts as the same shape.
    /// </summary>
    private static IEnumerable<Path> Ways(Path path, ShapeSignatureOptions options)
    {
        yield return path;
        yield return path.Backwards();

        if (!options.AllowReflection)
            yield break;

        var mirrored = path.Mirrored();
        yield return mirrored;
        yield return mirrored.Backwards();
    }

    /// <summary>The same readings, started from a different one of them.</summary>
    private static double[] From(double[] shape, int start, int points, int dimensions)
    {
        if (start == 0)
            return (double[])shape.Clone();

        var started = new double[shape.Length];
        for (int p = 0; p < points; p++)
        {
            int at = (p + start) % points;
            for (int a = 0; a < dimensions; a++)
                started[p * dimensions + a] = shape[at * dimensions + a];
        }

        return started;
    }

    /// <summary>
    /// Turns an outline to sit as squarely as it can on another. Two dimensions
    /// only: the turn is one angle there and has a closed form, which is why
    /// <see cref="ShapeSignatureOptions.Validate"/> refuses it in three.
    /// </summary>
    private static void Turn(double[] shape, double[] mean, int points)
    {
        double across = 0.0, along = 0.0;
        for (int p = 0; p < points; p++)
        {
            double x = shape[2 * p], y = shape[2 * p + 1];
            double mx = mean[2 * p], my = mean[2 * p + 1];
            across += x * my - y * mx;
            along += x * mx + y * my;
        }

        if (across == 0.0 && along == 0.0)
            return;

        double angle = Math.Atan2(across, along);
        double cos = Math.Cos(angle), sin = Math.Sin(angle);

        for (int p = 0; p < points; p++)
        {
            double x = shape[2 * p], y = shape[2 * p + 1];
            shape[2 * p] = x * cos - y * sin;
            shape[2 * p + 1] = x * sin + y * cos;
        }
    }

    private static double[] Average(double[][] shapes, List<int> repeats, int width)
    {
        var mean = new double[width];
        int total = 0;

        for (int u = 0; u < shapes.Length; u++)
        {
            for (int j = 0; j < width; j++)
                mean[j] += repeats[u] * shapes[u][j];
            total += repeats[u];
        }

        for (int j = 0; j < width; j++)
            mean[j] /= total;

        return mean;
    }

    private static void Centre(double[] shape, int points, int dimensions)
    {
        for (int a = 0; a < dimensions; a++)
        {
            double sum = 0.0;
            for (int p = 0; p < points; p++)
                sum += shape[p * dimensions + a];

            double centre = sum / points;
            for (int p = 0; p < points; p++)
                shape[p * dimensions + a] -= centre;
        }
    }

    /// <summary>An outline's size: how far its readings sit from its centre, root mean square.</summary>
    private static double Radius(double[] shape, int points)
    {
        double sum = 0.0;
        foreach (double coordinate in shape)
            sum += coordinate * coordinate;

        return Math.Sqrt(sum / points);
    }

    /// <summary>
    /// Holds the mean at unit size. Without this, every outline being free to
    /// shrink lets the whole population walk toward nothing, which fits perfectly
    /// and says nothing.
    /// </summary>
    private static void Rescale(double[] shape, int points)
    {
        double radius = Radius(shape, points);
        if (radius <= 0.0)
            return;

        for (int j = 0; j < shape.Length; j++)
            shape[j] /= radius;
    }

    private static double Apart(double[] a, double[] b)
    {
        double sum = 0.0;
        for (int j = 0; j < a.Length; j++)
        {
            double off = a[j] - b[j];
            sum += off * off;
        }

        return sum;
    }

    /// <summary>
    /// An outline kept as a path, so it can be read from anywhere along its length
    /// rather than only from the corner it happened to be listed from.
    /// </summary>
    private sealed class Path
    {
        private readonly double[][] _corners;
        private readonly double[] _along;
        private readonly bool _closed;
        private readonly int _dimensions;

        private Path(double[][] corners, double[] along, bool closed, int dimensions)
        {
            _corners = corners;
            _along = along;
            _closed = closed;
            _dimensions = dimensions;
        }

        private double Total => _along[^1];

        public static Path Of(double[,] outline, bool closed, int dimensions, int index)
        {
            var kept = new List<double[]>(outline.GetLength(0));
            for (int i = 0; i < outline.GetLength(0); i++)
            {
                var corner = new double[dimensions];
                for (int a = 0; a < dimensions; a++)
                {
                    corner[a] = outline[i, a];
                    if (!double.IsFinite(corner[a]))
                        throw new ArgumentException($"Outline {index} has a point that is not finite.", nameof(outline));
                }

                if (kept.Count == 0 || Between(kept[^1], corner) > 0.0)
                    kept.Add(corner);
            }

            // An outline given closed repeats its first corner at the end.
            if (closed && kept.Count > 1 && Between(kept[0], kept[^1]) <= 0.0)
                kept.RemoveAt(kept.Count - 1);

            int fewest = closed ? 3 : 2;
            if (kept.Count < fewest)
                throw new ArgumentException(
                    $"Outline {index} has {kept.Count} distinct point(s); it needs at least {fewest}.", nameof(outline));

            int segments = closed ? kept.Count : kept.Count - 1;
            var along = new double[segments + 1];
            for (int s = 0; s < segments; s++)
                along[s + 1] = along[s] + Between(kept[s], kept[(s + 1) % kept.Count]);

            if (along[segments] <= 0.0)
                throw new ArgumentException($"Outline {index} has no length.", nameof(outline));

            return new Path(kept.ToArray(), along, closed, dimensions);
        }

        /// <summary>The same path read the other way round.</summary>
        public Path Backwards()
        {
            var corners = new double[_corners.Length][];
            for (int i = 0; i < corners.Length; i++)
                corners[i] = _corners[_corners.Length - 1 - i];

            return Rebuilt(corners);
        }

        /// <summary>The same path turned about its origin. Two dimensions only.</summary>
        public Path Turned(double angle)
        {
            double cos = Math.Cos(angle), sin = Math.Sin(angle);
            var corners = new double[_corners.Length][];

            for (int i = 0; i < corners.Length; i++)
            {
                double x = _corners[i][0], y = _corners[i][1];
                corners[i] = new[] { x * cos - y * sin, x * sin + y * cos };
            }

            // Turning changes nothing about how long anything is, so the lengths
            // already worked out still stand.
            return new Path(corners, _along, _closed, _dimensions);
        }

        /// <summary>The same path flipped across its last axis.</summary>
        public Path Mirrored()
        {
            var corners = new double[_corners.Length][];
            for (int i = 0; i < corners.Length; i++)
            {
                corners[i] = (double[])_corners[i].Clone();
                corners[i][_dimensions - 1] = -corners[i][_dimensions - 1];
            }

            return Rebuilt(corners);
        }

        private Path Rebuilt(double[][] corners)
        {
            int segments = _closed ? corners.Length : corners.Length - 1;
            var along = new double[segments + 1];
            for (int s = 0; s < segments; s++)
                along[s + 1] = along[s] + Between(corners[s], corners[(s + 1) % corners.Length]);

            return new Path(corners, along, _closed, _dimensions);
        }

        /// <summary>
        /// The path read at a fixed count of points, evenly along its length, with
        /// the first reading <paramref name="phase"/> of the way round from the
        /// corner it was listed from. Phase means nothing on an open path, which is
        /// always read end to end.
        /// </summary>
        public double[] At(int points, double phase)
        {
            var shape = new double[points * _dimensions];
            int segments = _along.Length - 1;

            for (int p = 0; p < points; p++)
            {
                double part = _closed
                    ? (phase + (double)p / points) % 1.0
                    : (double)p / (points - 1);

                if (part < 0.0)
                    part += 1.0;

                double reach = part * Total;
                int at = Segment(reach, segments);
                double span = _along[at + 1] - _along[at];
                double into = span > 0.0 ? (reach - _along[at]) / span : 0.0;

                var a = _corners[at];
                var b = _corners[(at + 1) % _corners.Length];
                for (int c = 0; c < _dimensions; c++)
                    shape[p * _dimensions + c] = a[c] + into * (b[c] - a[c]);
            }

            return shape;
        }

        /// <summary>Which segment a length along the path falls in, by bisection.</summary>
        private int Segment(double reach, int segments)
        {
            int low = 0, high = segments - 1;
            while (low < high)
            {
                int middle = (low + high + 1) / 2;
                if (_along[middle] <= reach)
                    low = middle;
                else
                    high = middle - 1;
            }

            return low;
        }

        private static double Between(double[] a, double[] b)
        {
            double sum = 0.0;
            for (int i = 0; i < a.Length; i++)
            {
                double off = a[i] - b[i];
                sum += off * off;
            }

            return Math.Sqrt(sum);
        }
    }

    /// <summary>Two outlines read exactly alike, to the bit.</summary>
    private sealed class SameShape : IEqualityComparer<double[]>
    {
        public static readonly SameShape Default = new();

        public bool Equals(double[]? a, double[]? b) => a is not null && b is not null && a.AsSpan().SequenceEqual(b);

        public int GetHashCode(double[] shape)
        {
            var hash = new HashCode();
            foreach (double coordinate in shape)
                hash.Add(coordinate);
            return hash.ToHashCode();
        }
    }
}
