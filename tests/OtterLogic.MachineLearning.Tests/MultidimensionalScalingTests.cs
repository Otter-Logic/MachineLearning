using OtterLogic.MachineLearning.Decomposition;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public sealed class MultidimensionalScalingTests
{
    private static double MapDistance(double[,] x, int i, int j)
    {
        double sum = 0.0;
        for (int a = 0; a < x.GetLength(1); a++)
            sum += (x[i, a] - x[j, a]) * (x[i, a] - x[j, a]);
        return Math.Sqrt(sum);
    }

    /// <summary>
    /// The corners of a 3 x 4 rectangle, given only their distances, come back as
    /// a rectangle: every distance on the map is the one asked for.
    /// </summary>
    [Fact]
    public void RecoversARectangleFromItsDistances()
    {
        var distances = new double[,]
        {
            { 0, 3, 4, 5 },
            { 3, 0, 5, 4 },
            { 4, 5, 0, 3 },
            { 5, 4, 3, 0 },
        };

        var map = MultidimensionalScaling.FromDistances(distances);

        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                Numeric.Close(distances[i, j], MapDistance(map.Coordinates, i, j), 1e-8, $"distance {i}-{j}");

        Assert.True(map.Stress < 1e-8, $"stress {map.Stress}");
        Numeric.Close(1.0, map.Retained, 1e-8, "retained");
        Assert.Empty(map.Notes);
    }

    /// <summary>
    /// On features, classical scaling is principal components without whitening —
    /// the same projection, axis for axis, up to the sign of each. PCA is itself
    /// checked against scikit-learn, so this pins the classical stage to it.
    /// </summary>
    [Fact]
    public void ClassicalMapOfFeaturesIsThePrincipalProjection()
    {
        var x = Fixture.Load("pca_plain").Matrix("x");

        var map = MultidimensionalScaling.FromFeatures(x, new MultidimensionalScalingOptions { Refine = false });
        var projected = PrincipalComponents.FitCount(x, 2, whiten: false).Transform(x);

        for (int a = 0; a < 2; a++)
        {
            double sign = Math.Sign(map.Coordinates[0, a]) == Math.Sign(projected[0, a]) ? 1.0 : -1.0;
            for (int i = 0; i < x.GetLength(0); i++)
                Numeric.Close(projected[i, a], sign * map.Coordinates[i, a], 1e-7, $"sample {i}, axis {a}");
        }
    }

    /// <summary>Refinement starts from the classical map and every step lowers stress or keeps it.</summary>
    [Fact]
    public void RefinementNeverRaisesStress()
    {
        var rng = new Random(5);
        var x = new double[40, 6];
        for (int i = 0; i < 40; i++)
            for (int j = 0; j < 6; j++)
                x[i, j] = rng.NextDouble();

        var map = MultidimensionalScaling.FromFeatures(x);

        Assert.True(map.Iterations > 0);
        Assert.True(map.Stress <= map.ClassicalStress + 1e-12,
            $"refined {map.Stress} against classical {map.ClassicalStress}");
    }

    /// <summary>
    /// A star of three leaves: every leaf one hop from the centre and two from each
    /// other. A triangle with sides of 2 has a circumradius of 1.15, not 1, so no
    /// flat drawing honours every distance — the centred matrix has a negative
    /// eigenvalue, and the map says so rather than pretending.
    /// </summary>
    [Fact]
    public void RouteDistancesThatNoFlatSpaceHoldsAreReported()
    {
        var star = new double[,]
        {
            { 0, 1, 1, 1 },
            { 1, 0, 2, 2 },
            { 1, 2, 0, 2 },
            { 1, 2, 2, 0 },
        };

        var map = MultidimensionalScaling.FromDistances(star);

        Assert.True(map.MostNegativeEigenvalue < -1e-3, $"most negative {map.MostNegativeEigenvalue}");
        Assert.True(map.Stress > 1e-3, $"stress {map.Stress}");
        Assert.Contains(map.Notes, note => note.Contains("not those of points in any flat space"));
    }

    [Fact]
    public void FeaturesAreNeverCheckedForFlatness()
    {
        var map = MultidimensionalScaling.FromFeatures(new double[,] { { 0, 0, 1 }, { 1, 0, 2 }, { 0, 1, 0 }, { 3, 2, 1 } });
        Assert.True(double.IsNaN(map.MostNegativeEigenvalue));
    }

    /// <summary>Points on a line, asked for a 2D map: the second axis is empty, and the note says why.</summary>
    [Fact]
    public void AnAxisWithNothingOnItIsZeroAndNamed()
    {
        var line = new double[6, 3];
        for (int i = 0; i < 6; i++)
        {
            line[i, 0] = i;
            line[i, 1] = 2 * i;
            line[i, 2] = -i;
        }

        var map = MultidimensionalScaling.FromFeatures(line);

        for (int i = 0; i < 6; i++)
            Assert.Equal(0.0, map.Coordinates[i, 1]);
        Assert.Contains(map.Notes, note => note.StartsWith("Axis 2 of 2 carries no structure"));
    }

    [Fact]
    public void TooFewColumnsToNeedAMapIsNoted()
    {
        var map = MultidimensionalScaling.FromFeatures(new double[,] { { 0, 0 }, { 1, 0 }, { 0, 2 }, { 3, 1 } });
        Assert.Contains(map.Notes, note => note.Contains("only rotates them"));
    }

    /// <summary>Distances in millimetres give the map in millimetres, and the same stress.</summary>
    [Fact]
    public void StressIsTheSameInAnyUnits()
    {
        var star = new double[,] { { 0, 1, 1, 1 }, { 1, 0, 2, 2 }, { 1, 2, 0, 2 }, { 1, 2, 2, 0 } };
        var scaled = new double[4, 4];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                scaled[i, j] = 1000.0 * star[i, j];

        var metres = MultidimensionalScaling.FromDistances(star);
        var millimetres = MultidimensionalScaling.FromDistances(scaled);

        Numeric.Close(metres.Stress, millimetres.Stress, 1e-6, "stress");
        Numeric.Close(1000.0 * MapDistance(metres.Coordinates, 1, 2), MapDistance(millimetres.Coordinates, 1, 2), 1e-6, "distance");
    }

    [Fact]
    public void TheSameInputGivesTheSameMap()
    {
        var x = Fixture.Load("pca_plain").Matrix("x");

        var first = MultidimensionalScaling.FromFeatures(x);
        var second = MultidimensionalScaling.FromFeatures(x);

        Assert.Equal(first.Coordinates, second.Coordinates);
    }

    [Fact]
    public void AllZeroDistancesPutEverySampleOnOneSpot()
    {
        var map = MultidimensionalScaling.FromDistances(new double[3, 3]);

        Assert.Equal(0.0, map.Stress);
        Assert.All(map.Coordinates.Cast<double>(), value => Assert.Equal(0.0, value));
    }

    [Fact]
    public void AnAsymmetricTableIsRefused()
    {
        Assert.Throws<ArgumentException>(() =>
            MultidimensionalScaling.FromDistances(new double[,] { { 0, 1, 2 }, { 1, 0, 1 }, { 3, 1, 0 } }));
    }

    [Fact]
    public void ANonZeroDiagonalIsRefused()
    {
        Assert.Throws<ArgumentException>(() =>
            MultidimensionalScaling.FromDistances(new double[,] { { 1, 1, 2 }, { 1, 0, 1 }, { 2, 1, 0 } }));
    }

    [Fact]
    public void ANegativeDistanceIsRefused()
    {
        Assert.Throws<ArgumentException>(() =>
            MultidimensionalScaling.FromDistances(new double[,] { { 0, -1, 2 }, { -1, 0, 1 }, { 2, 1, 0 } }));
    }

    [Fact]
    public void ANonSquareTableIsRefused()
    {
        Assert.Throws<ArgumentException>(() => MultidimensionalScaling.FromDistances(new double[3, 4]));
    }

    [Fact]
    public void MoreDimensionsThanTheSamplesSpanAreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() =>
            MultidimensionalScaling.FromFeatures(new double[,] { { 0, 1 }, { 1, 0 }, { 2, 2 } },
                new MultidimensionalScalingOptions { Dimensions = 3 }));
    }
}
