using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>Comparison helpers, and the one clustering metric the tests need.</summary>
internal static class Numeric
{
    internal static void Close(double expected, double actual, double tolerance, string what)
    {
        double difference = Math.Abs(expected - actual);
        double allowed = tolerance * Math.Max(1.0, Math.Abs(expected));

        Assert.True(difference <= allowed,
            $"{what}: expected {expected:R}, got {actual:R} (difference {difference:R}, allowed {allowed:R})");
    }

    internal static void Close(double[] expected, double[] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int i = 0; i < expected.Length; i++)
            Close(expected[i], actual[i], tolerance, $"{what}[{i}]");
    }

    internal static void Close(double[,] expected, double[,] actual, double tolerance, string what)
    {
        Assert.Equal(expected.GetLength(0), actual.GetLength(0));
        Assert.Equal(expected.GetLength(1), actual.GetLength(1));

        for (int i = 0; i < expected.GetLength(0); i++)
            for (int j = 0; j < expected.GetLength(1); j++)
                Close(expected[i, j], actual[i, j], tolerance, $"{what}[{i},{j}]");
    }

    internal static void Close(double[][,] expected, double[][,] actual, double tolerance, string what)
    {
        Assert.Equal(expected.Length, actual.Length);
        for (int c = 0; c < expected.Length; c++)
            Close(expected[c], actual[c], tolerance, $"{what}[{c}]");
    }

    /// <summary>
    /// Adjusted Rand index between two labellings: 1.0 is identical grouping,
    /// 0.0 is what random agreement would give.
    /// <para>
    /// The right measure for comparing against scikit-learn, because cluster
    /// numbering is arbitrary. Two implementations can produce exactly the same
    /// partition and disagree about every single label, so counting matching
    /// labels would report nonsense; this counts pairs of members that both
    /// implementations put together or apart, and is invariant to renumbering.
    /// </para>
    /// </summary>
    internal static double AdjustedRandIndex(int[] a, int[] b)
    {
        Assert.Equal(a.Length, b.Length);
        int n = a.Length;

        int rows = a.Max() + 1;
        int columns = b.Max() + 1;

        var contingency = new int[rows, columns];
        for (int i = 0; i < n; i++)
            contingency[a[i], b[i]]++;

        double index = 0.0;
        var rowSums = new int[rows];
        var columnSums = new int[columns];

        for (int i = 0; i < rows; i++)
        {
            for (int j = 0; j < columns; j++)
            {
                index += Choose2(contingency[i, j]);
                rowSums[i] += contingency[i, j];
                columnSums[j] += contingency[i, j];
            }
        }

        double rowTotal = rowSums.Sum(Choose2);
        double columnTotal = columnSums.Sum(Choose2);
        double expected = rowTotal * columnTotal / Choose2(n);
        double maximum = 0.5 * (rowTotal + columnTotal);

        return maximum - expected == 0.0 ? 1.0 : (index - expected) / (maximum - expected);
    }

    private static double Choose2(int count) => count * (count - 1) / 2.0;
}
