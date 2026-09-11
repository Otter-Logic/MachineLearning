using OtterLogic.MachineLearning.Distances;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The one distance every paradigm measures with, and the nearest-neighbour
/// search built on it. The graph tests hold the search against scikit-learn;
/// these pin what the graph cannot see — the order, the ties, and the distances.
/// </summary>
public sealed class EuclideanTests
{
    [Fact]
    public void MeasuresBetweenRowsOfTwoMatrices()
    {
        var a = new double[,] { { 0.0, 0.0 }, { 1.0, 1.0 } };
        var b = new double[,] { { 3.0, 4.0 } };

        Assert.Equal(25.0, Euclidean.Squared(a, 0, b, 0));
        Assert.Equal(5.0, Euclidean.Between(a, 0, b, 0));
        Assert.Equal(0.0, Euclidean.Between(a, 1, a, 1));
    }

    [Fact]
    public void NearestComesBackNearestFirstAndNeverItself()
    {
        var x = new double[,] { { 0.0 }, { 10.0 }, { 1.0 }, { 3.0 }, { 6.0 } };

        var (index, distance) = Euclidean.Nearest(x, 3);

        Assert.Equal(new[] { 2, 3, 4 }, Row(index, 0));
        Assert.Equal(new[] { 1.0, 3.0, 6.0 }, Row(distance, 0));
        Assert.Equal(new[] { 4, 3, 2 }, Row(index, 1));

        for (int i = 0; i < x.GetLength(0); i++)
            Assert.DoesNotContain(i, Row(index, i));
    }

    [Fact]
    public void NearestBreaksTiesTowardsTheLowerIndex()
    {
        // Samples 1, 2 and 3 are all exactly one away from sample 0.
        var x = new double[,] { { 0.0, 0.0 }, { 1.0, 0.0 }, { 0.0, 1.0 }, { -1.0, 0.0 } };

        Assert.Equal(new[] { 1, 2 }, Row(Euclidean.Nearest(x, 2).Index, 0));
    }

    [Theory]
    [InlineData(0)]
    [InlineData(4)]
    public void NearestRefusesACountOutsideOneToNMinusOne(int count)
    {
        var x = new double[4, 1];
        Assert.Throws<ArgumentOutOfRangeException>(() => Euclidean.Nearest(x, count));
    }

    private static T[] Row<T>(T[,] m, int i) => Enumerable.Range(0, m.GetLength(1)).Select(c => m[i, c]).ToArray();
}
