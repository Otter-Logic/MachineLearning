using OtterLogic.MachineLearning.Shapes;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class ShapeSignatureTests
{
    private static double[,] Outline(params (double X, double Y)[] corners)
    {
        var rows = new double[corners.Length, 2];
        for (int i = 0; i < corners.Length; i++)
            (rows[i, 0], rows[i, 1]) = (corners[i].X, corners[i].Y);
        return rows;
    }

    private static double[,] Rectangle(double length, double width, double atX = 0.0, double atY = 0.0)
        => Outline((atX, atY), (atX + length, atY), (atX + length, atY + width), (atX, atY + width));

    /// <summary>A rectangle with a square bitten out of each top corner.</summary>
    private static double[,] Corners()
        => Outline((0, 0), (4000, 0), (4000, 1500), (3500, 1500), (3500, 2000), (500, 2000), (500, 1500), (0, 1500));

    /// <summary>
    /// A rectangle with one square bitten out of the middle of its top edge — the
    /// same area removed, from the same band, symmetrically. Deliberately the twin
    /// of <see cref="Corners"/> under every measurement anyone would think to take.
    /// </summary>
    private static double[,] Middle()
        => Outline((0, 0), (4000, 0), (4000, 2000), (2500, 2000), (2500, 1500), (1500, 1500), (1500, 2000), (0, 2000));

    /// <summary>An L, for a shape with a handedness to it.</summary>
    private static double[,] Ell()
        => Outline((0, 0), (4000, 0), (4000, 1000), (1000, 1000), (1000, 2000), (0, 2000));

    private static double[,] Turned(double[,] outline, double angle)
    {
        double cos = Math.Cos(angle), sin = Math.Sin(angle);
        var turned = new double[outline.GetLength(0), 2];
        for (int i = 0; i < outline.GetLength(0); i++)
        {
            turned[i, 0] = outline[i, 0] * cos - outline[i, 1] * sin;
            turned[i, 1] = outline[i, 0] * sin + outline[i, 1] * cos;
        }

        return turned;
    }

    private static double[,] Scaled(double[,] outline, double by)
    {
        var scaled = new double[outline.GetLength(0), 2];
        for (int i = 0; i < outline.GetLength(0); i++)
            for (int a = 0; a < 2; a++)
                scaled[i, a] = outline[i, a] * by;

        return scaled;
    }

    /// <summary>Flipped across the x axis, which reverses the way it winds.</summary>
    private static double[,] Mirrored(double[,] outline)
    {
        var mirrored = new double[outline.GetLength(0), 2];
        for (int i = 0; i < outline.GetLength(0); i++)
        {
            mirrored[i, 0] = outline[i, 0];
            mirrored[i, 1] = -outline[i, 1];
        }

        return mirrored;
    }

    /// <summary>The same outline listed from a different one of its corners.</summary>
    private static double[,] From(double[,] outline, int corner)
    {
        int k = outline.GetLength(0);
        var rolled = new double[k, 2];
        for (int i = 0; i < k; i++)
            for (int a = 0; a < 2; a++)
                rolled[i, a] = outline[(i + corner) % k, a];

        return rolled;
    }

    /// <summary>How far apart two outlines are in the signature: the promise the scores carry.</summary>
    private static double Apart(ShapeSignatureResult result, int a, int b)
    {
        double sum = 0.0;
        for (int c = 0; c < result.Count; c++)
        {
            double off = result.Scores[a, c] - result.Scores[b, c];
            sum += off * off;
        }

        return Math.Sqrt(sum);
    }

    private static ShapeSignatureOptions Reading(
        int points = 32, bool scale = false, bool rotation = false, bool reflection = false, double variance = 1.0)
        => new()
        {
            Points = points,
            NormaliseScale = scale,
            NormaliseRotation = rotation,
            AllowReflection = reflection,
            Variance = variance,
        };

    [Fact]
    public void TheSameShapeInTwoPlacesHasTheSameSignature()
    {
        var result = ShapeSignature.Fit(
            new[] { Rectangle(4000, 2000), Rectangle(4000, 2000, 50000, -9000) }, Reading());

        Assert.Equal(0.0, Apart(result, 0, 1), 6);
    }

    /// <summary>
    /// A closed outline has no natural first corner, so which one it was listed
    /// from cannot change what it is.
    /// </summary>
    [Fact]
    public void WhereAnOutlineStartsDoesNotMatter()
    {
        var ell = Ell();
        var result = ShapeSignature.Fit(new[] { ell, From(ell, 3) }, Reading());

        Assert.Equal(0.0, Apart(result, 0, 1), 6);
    }

    /// <summary>And nor does which way round it was drawn.</summary>
    [Fact]
    public void WhichWayRoundAnOutlineWasDrawnDoesNotMatter()
    {
        var ell = Ell();
        int k = ell.GetLength(0);
        var backwards = new double[k, 2];
        for (int i = 0; i < k; i++)
            for (int a = 0; a < 2; a++)
                backwards[i, a] = ell[k - 1 - i, a];

        var result = ShapeSignature.Fit(new[] { ell, backwards }, Reading());

        Assert.Equal(0.0, Apart(result, 0, 1), 6);
    }

    /// <summary>
    /// Corners that are not corners change nothing: the same rectangle drawn with
    /// points along its edges reads the same, because outlines are read by length
    /// rather than by corner.
    /// </summary>
    [Fact]
    public void ExtraPointsAlongAStraightEdgeChangeNothing()
    {
        var plain = Rectangle(4000, 2000);
        var padded = Outline(
            (0, 0), (1000, 0), (2500, 0), (4000, 0), (4000, 900), (4000, 2000),
            (2000, 2000), (0, 2000), (0, 1200));

        var result = ShapeSignature.Fit(new[] { plain, padded }, Reading());

        Assert.Equal(0.0, Apart(result, 0, 1), 6);
    }

    /// <summary>
    /// The reason for the whole thing. These two outlines agree on every
    /// measurement anybody would think to take by hand — same bounding box, same
    /// area, same centre of area — and they are plainly not the same panel. A
    /// hand-picked list of measurements calls them one type. This does not.
    /// </summary>
    [Fact]
    public void TwoShapesNoHandPickedMeasurementCanTellApartAreToldApart()
    {
        var corners = Corners();
        var middle = Middle();

        // First, that they really do agree on all of it.
        var (lengthA, widthA, areaA, centreA) = Measured(corners);
        var (lengthB, widthB, areaB, centreB) = Measured(middle);

        Assert.Equal(lengthA, lengthB, 6);
        Assert.Equal(widthA, widthB, 6);
        Assert.Equal(areaA, areaB, 6);
        Assert.Equal(centreA.X, centreB.X, 6);
        Assert.Equal(centreA.Y, centreB.Y, 6);

        // And then that the signature is not fooled.
        var result = ShapeSignature.Fit(new[] { corners, middle }, Reading());
        Assert.True(Apart(result, 0, 1) > 100.0,
            $"The two shapes came out only {Apart(result, 0, 1):G4} apart.");
    }

    /// <summary>Bounding box, area and centre of area — the measurements a person picks.</summary>
    private static (double Length, double Width, double Area, (double X, double Y) Centre) Measured(double[,] outline)
    {
        int k = outline.GetLength(0);
        double lowX = double.MaxValue, highX = double.MinValue, lowY = double.MaxValue, highY = double.MinValue;
        double twice = 0.0, x = 0.0, y = 0.0;

        for (int i = 0; i < k; i++)
        {
            lowX = Math.Min(lowX, outline[i, 0]);
            highX = Math.Max(highX, outline[i, 0]);
            lowY = Math.Min(lowY, outline[i, 1]);
            highY = Math.Max(highY, outline[i, 1]);

            int j = (i + 1) % k;
            double cross = outline[i, 0] * outline[j, 1] - outline[j, 0] * outline[i, 1];
            twice += cross;
            x += (outline[i, 0] + outline[j, 0]) * cross;
            y += (outline[i, 1] + outline[j, 1]) * cross;
        }

        return (highX - lowX, highY - lowY, Math.Abs(twice) / 2.0, (x / (3.0 * twice), y / (3.0 * twice)));
    }

    /// <summary>
    /// The promise on the scores: how far apart two signatures are is how far apart
    /// the two outlines are, point for corresponding point. That is what lets a
    /// clustering of them be cut at a tolerance somebody can justify.
    /// </summary>
    [Fact]
    public void DistanceBetweenSignaturesIsDistanceBetweenOutlines()
    {
        var result = ShapeSignature.Fit(
            new[] { Rectangle(4000, 2000), Corners(), Middle(), Ell() }, Reading());

        Assert.All(result.Residual, r => Assert.Equal(0.0, r, 6));

        for (int a = 0; a < 4; a++)
            for (int b = a + 1; b < 4; b++)
            {
                var one = result.ShapeOf(a);
                var two = result.ShapeOf(b);

                double sum = 0.0;
                for (int p = 0; p < result.Points; p++)
                    for (int c = 0; c < result.Dimensions; c++)
                    {
                        double off = one[p, c] - two[p, c];
                        sum += off * off;
                    }

                double rootMeanSquare = Math.Sqrt(sum / result.Points);
                Assert.Equal(rootMeanSquare, Apart(result, a, b), 6);
            }
    }

    [Fact]
    public void AMirrorIsADifferentShapeUnlessItIsAllowed()
    {
        var pair = new[] { Ell(), Mirrored(Ell()) };

        Assert.True(Apart(ShapeSignature.Fit(pair, Reading()), 0, 1) > 100.0,
            "A shape and its mirror should not be the same by default.");

        Assert.Equal(0.0, Apart(ShapeSignature.Fit(pair, Reading(reflection: true)), 0, 1), 6);
    }

    [Fact]
    public void TurningAnOutlineChangesItUnlessItIsNormalised()
    {
        var pair = new[] { Ell(), Turned(Ell(), 0.6) };

        Assert.True(Apart(ShapeSignature.Fit(pair, Reading()), 0, 1) > 100.0,
            "A turned shape should not be the same by default.");

        Assert.Equal(0.0, Apart(ShapeSignature.Fit(pair, Reading(rotation: true)), 0, 1), 6);
    }

    /// <summary>
    /// A flat thing with no up and no down is the same thing turned half way round,
    /// and that is not the same as letting it be turned to any angle at all.
    /// </summary>
    [Fact]
    public void AHalfTurnIsTheSameShapeWhenTurnsSaysSo()
    {
        var half = new[] { Ell(), Turned(Ell(), Math.PI) };
        var quarter = new[] { Ell(), Turned(Ell(), 0.5 * Math.PI) };

        Assert.True(Apart(ShapeSignature.Fit(half, Reading()), 0, 1) > 100.0,
            "A half turn should not be the same by default.");

        var turning = new ShapeSignatureOptions { Points = 32, Variance = 1.0, Turns = 2 };
        Assert.Equal(0.0, Apart(ShapeSignature.Fit(half, turning), 0, 1), 6);

        Assert.True(Apart(ShapeSignature.Fit(quarter, turning), 0, 1) > 100.0,
            "Half turns should not quietly allow a quarter turn as well.");

        var quartering = new ShapeSignatureOptions { Points = 32, Variance = 1.0, Turns = 4 };
        Assert.Equal(0.0, Apart(ShapeSignature.Fit(quarter, quartering), 0, 1), 6);
    }

    [Fact]
    public void TurningInThreeDimensionsIsRefusedToo()
    {
        var flat = new double[4, 3];
        for (int i = 0; i < 4; i++)
            (flat[i, 0], flat[i, 1], flat[i, 2]) = (i % 3, i / 2, 0.0);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => ShapeSignature.Fit(new[] { flat }, new ShapeSignatureOptions { Turns = 2 }));
    }

    [Fact]
    public void SizeIsSetAsideOnlyWhenItIsAskedFor()
    {
        var pair = new[] { Ell(), Scaled(Ell(), 3.0) };

        var kept = ShapeSignature.Fit(pair, Reading());
        Assert.True(Apart(kept, 0, 1) > 100.0, "A bigger copy should not be the same by default.");
        Assert.Equal(3.0, kept.Size[1] / kept.Size[0], 6);

        var setAside = ShapeSignature.Fit(pair, Reading(scale: true));
        Assert.Equal(0.0, Apart(setAside, 0, 1), 6);
        Assert.Equal(3.0, setAside.Size[1] / setAside.Size[0], 6);
    }

    /// <summary>
    /// What a component means is not in any number — it is in the shape it draws.
    /// Walking the mean along a component has to give outlines, and standing still
    /// has to give the mean.
    /// </summary>
    [Fact]
    public void AComponentCanBeLookedAt()
    {
        var result = ShapeSignature.Fit(
            new[] { Rectangle(4000, 2000), Corners(), Middle(), Ell(), Rectangle(3000, 2500) }, Reading());

        Assert.True(result.Count >= 2);

        var still = result.Variation(0, 0.0);
        var mean = result.MeanShape;
        Assert.Equal(result.Points, still.GetLength(0));
        Assert.Equal(result.Dimensions, still.GetLength(1));

        for (int p = 0; p < result.Points; p++)
            for (int c = 0; c < result.Dimensions; c++)
                Assert.Equal(mean[p, c], still[p, c], 6);

        // A spread along the first component is the spread it reported.
        var one = result.Variation(0, 1.0);
        double sum = 0.0;
        for (int p = 0; p < result.Points; p++)
            for (int c = 0; c < result.Dimensions; c++)
            {
                double off = one[p, c] - mean[p, c];
                sum += off * off;
            }

        Assert.Equal(result.Spread[0], Math.Sqrt(sum / result.Points), 6);
    }

    [Fact]
    public void KeepingFewerComponentsLeavesAResidual()
    {
        var outlines = new[] { Rectangle(4000, 2000), Corners(), Middle(), Ell(), Rectangle(3000, 2500) };

        var all = ShapeSignature.Fit(outlines, Reading());
        var one = ShapeSignature.Fit(outlines, new ShapeSignatureOptions { Components = 1 });

        Assert.Equal(1, one.Count);
        Assert.True(all.Count > 1);
        Assert.All(all.Residual, r => Assert.Equal(0.0, r, 6));
        Assert.True(one.Residual.Max() > 0.0, "Keeping one component of several should leave something behind.");
        Assert.True(one.Carried < all.Carried);
    }

    [Fact]
    public void OneOutlineOnItsOwnHasNothingToVaryAgainst()
    {
        var result = ShapeSignature.Fit(new[] { Ell() }, Reading());

        Assert.All(result.Spread, spread => Assert.Equal(0.0, spread, 9));
        Assert.Contains(result.Notes, n => n.Contains("same shape"));
    }

    [Fact]
    public void OutlinesInThreeDimensionsAreRead()
    {
        var flat = new double[4, 3];
        var corners = new[] { (0.0, 0.0), (4000.0, 0.0), (4000.0, 2000.0), (0.0, 2000.0) };
        for (int i = 0; i < 4; i++)
            (flat[i, 0], flat[i, 1], flat[i, 2]) = (corners[i].Item1, 7000.0, corners[i].Item2);

        var result = ShapeSignature.Fit(new[] { flat, flat }, Reading());

        Assert.Equal(3, result.Dimensions);
        Assert.Equal(0.0, Apart(result, 0, 1), 6);
    }

    [Fact]
    public void ReadingTooCoarselyIsSaidSo()
    {
        var result = ShapeSignature.Fit(new[] { Corners(), Middle() }, Reading(points: 4));

        Assert.Contains(result.Notes, n => n.Contains("more corners than they were read at"));
    }

    [Theory]
    [InlineData(2)]
    [InlineData(0)]
    [InlineData(ShapeSignatureOptions.MostPoints + 1)]
    public void AnImpossibleNumberOfPointsIsRefused(int points)
        => Assert.Throws<ArgumentOutOfRangeException>(
            () => ShapeSignature.Fit(new[] { Ell() }, new ShapeSignatureOptions { Points = points }));

    [Fact]
    public void TurningInThreeDimensionsIsRefused()
    {
        var flat = new double[4, 3];
        for (int i = 0; i < 4; i++)
            (flat[i, 0], flat[i, 1], flat[i, 2]) = (i % 3, i / 2, 0.0);

        Assert.Throws<ArgumentOutOfRangeException>(
            () => ShapeSignature.Fit(new[] { flat }, new ShapeSignatureOptions { NormaliseRotation = true }));
    }

    [Fact]
    public void NoOutlinesIsRefused()
        => Assert.Throws<ArgumentException>(() => ShapeSignature.Fit(Array.Empty<double[,]>(), Reading()));

    [Fact]
    public void OutlinesThatDisagreeOnDimensionsAreRefused()
        => Assert.Throws<ArgumentException>(
            () => ShapeSignature.Fit(new[] { Ell(), new double[4, 3] }, Reading()));

    [Fact]
    public void AnOutlineWithNoLengthIsRefused()
        => Assert.Throws<ArgumentException>(
            () => ShapeSignature.Fit(new[] { Outline((5, 5), (5, 5), (5, 5)) }, Reading()));

    [Fact]
    public void AnOutlineWithAPointThatIsNotFiniteIsRefused()
    {
        var broken = Ell();
        broken[2, 1] = double.NaN;
        Assert.Throws<ArgumentException>(() => ShapeSignature.Fit(new[] { broken }, Reading()));
    }
}
