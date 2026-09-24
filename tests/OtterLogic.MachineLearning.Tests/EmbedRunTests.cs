using OtterLogic.Graphs;
using OtterLogic.MachineLearning.Decomposition;
using OtterLogic.MachineLearning.Embedding;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// Each embedding method on the wire, held to the decomposition it wraps, and the
/// run around them: what is chosen when nothing is wired, and what a dropped
/// column does to the axes.
/// </summary>
public sealed class EmbedRunTests
{
    /// <summary>Forty samples on a tilted line in three columns, with a little spread off it.</summary>
    private static double[,] Ribbon()
    {
        var x = new double[40, 3];
        for (int i = 0; i < 40; i++)
        {
            double t = i / 39.0;
            x[i, 0] = 10.0 * t;
            x[i, 1] = 4.0 * t + 0.3 * Math.Sin(7.0 * i);
            x[i, 2] = 0.1 * Math.Cos(5.0 * i);
        }

        return x;
    }

    [Fact]
    public void PrincipalComponentsMatchTheDecompositionAndNameTheirAxes()
    {
        var x = Ribbon();
        var result = EmbedRun.Fit(new EmbeddingQuery(x, null), new PrincipalComponentsMethod(),
            new EmbedRunOptions { Standardise = false, Dimensions = 2 });

        var pca = PrincipalComponents.FitCount(x, 2, whiten: false);
        Numeric.Close(pca.Transform(x), result.Coordinates, 1e-12, "coordinates");
        Numeric.Close(pca.Components, result.Axes!, 1e-12, "axes");
        Assert.Equal(pca.ExplainedVarianceRatio, result.Retained, 12);

        var report = result.Report(new[] { "Length", "Height", "Wobble" });
        Assert.Contains(report, line => line.StartsWith("Axis 1:") && line.Contains("Length"));
    }

    /// <summary>
    /// A constant column is dropped before the fit and put back as a zero loading,
    /// so axis loadings still line up with the columns the user named.
    /// </summary>
    [Fact]
    public void ADroppedColumnComesBackAsAZeroLoading()
    {
        var x = new double[40, 3];
        var ribbon = Ribbon();
        for (int i = 0; i < 40; i++)
        {
            x[i, 0] = ribbon[i, 0];
            x[i, 1] = 7.0;
            x[i, 2] = ribbon[i, 1];
        }

        var result = EmbedRun.Fit(new EmbeddingQuery(x, null), new PrincipalComponentsMethod());

        Assert.Equal(new[] { 0, 2 }, result.KeptColumns);
        Assert.Equal(3, result.Axes!.GetLength(1));
        Assert.Equal(0.0, result.Axes[0, 1]);
        Assert.NotEqual(0.0, result.Axes[0, 0]);
        Assert.Contains(result.Notes, note => note.Text.Contains("left out"));
    }

    [Fact]
    public void ScalingKeepsTheDistancesOfPointsThatAlreadyLieFlat()
    {
        var x = new double[30, 2];
        var rng = new Random(3);
        for (int i = 0; i < 30; i++)
        {
            x[i, 0] = rng.NextDouble() * 10.0;
            x[i, 1] = rng.NextDouble() * 10.0;
        }

        var result = EmbedRun.Fit(new EmbeddingQuery(x, null), new MultidimensionalScalingMethod(),
            new EmbedRunOptions { Standardise = false });

        Assert.Null(result.Axes);
        Assert.Equal(1.0, result.Retained, 6);
        Assert.All(result.Distortion!, d => Assert.True(d < 1e-6, $"distortion {d}"));

        for (int i = 0; i < 30; i++)
            for (int j = i + 1; j < 30; j++)
            {
                double truth = Math.Sqrt(Math.Pow(x[i, 0] - x[j, 0], 2) + Math.Pow(x[i, 1] - x[j, 1], 2));
                double mapped = Math.Sqrt(Math.Pow(result.Coordinates[i, 0] - result.Coordinates[j, 0], 2)
                    + Math.Pow(result.Coordinates[i, 1] - result.Coordinates[j, 1], 2));
                Numeric.Close(truth, mapped, 1e-6, $"distance {i}-{j}");
            }
    }

    /// <summary>
    /// Two cliques joined by one edge: the spectral map puts each clique's nodes on
    /// top of each other and the two cliques apart, and reads the wired weights as
    /// present or absent.
    /// </summary>
    [Fact]
    public void SpectralEmbeddingLaysAGraphOutByItsConnections()
    {
        var edges = new List<(int, int, double)>();
        for (int a = 0; a < 5; a++)
            for (int b = a + 1; b < 5; b++)
            {
                edges.Add((a, b, 100.0));
                edges.Add((a + 5, b + 5, 1.0));
            }
        edges.Add((4, 5, 50.0));
        var graph = WeightedGraph.FromEdges(10, edges);

        var result = EmbedRun.Fit(new EmbeddingQuery(null, graph), new SpectralEmbeddingMethod(),
            new EmbedRunOptions { Dimensions = 2 });

        double Gap(int i, int j) => Math.Abs(result.Coordinates[i, 1] - result.Coordinates[j, 1]);
        Assert.True(Gap(0, 1) < 1e-6 * Math.Max(1.0, Gap(0, 9)), "nodes of one clique should coincide");
        Assert.True(Gap(0, 9) > 0.1, "the two cliques should sit apart on the second axis");
        Assert.True(double.IsNaN(result.Retained));
        Assert.Contains(result.Outcome.Details, d => d.Contains("present or absent"));
    }

    [Fact]
    public void WithNothingWiredTheChoiceFollowsWhatWasGiven()
    {
        var x = Ribbon();
        var fromValues = EmbedRun.Fit(new EmbeddingQuery(x, null));
        Assert.Equal("Multidimensional Scaling", fromValues.Method);
        Assert.Contains("chosen because", fromValues.Report()[0]);

        var graph = WeightedGraph.FromEdges(40, Enumerable.Range(0, 39).Select(i => (i, i + 1)));
        var fromGraph = EmbedRun.Fit(new EmbeddingQuery(x, graph));
        Assert.Equal("Spectral Embedding", fromGraph.Method);
    }

    [Fact]
    public void SaysWhatToWireWhenAMethodLacksIt()
    {
        var graph = WeightedGraph.FromEdges(4, new[] { (0, 1), (1, 2), (2, 3) });
        var ex = Assert.Throws<ArgumentException>(
            () => EmbedRun.Fit(new EmbeddingQuery(null, graph), new PrincipalComponentsMethod()));
        Assert.Contains("Data", ex.Message);
    }

    [Fact]
    public void RefusesMoreDimensionsThanTheSamplesSpan()
    {
        var x = new double[3, 2] { { 0, 0 }, { 1, 0 }, { 0, 1 } };
        Assert.Throws<ArgumentOutOfRangeException>(
            () => EmbedRun.Fit(new EmbeddingQuery(x, null), null, new EmbedRunOptions { Dimensions = 3 }));
    }
}
