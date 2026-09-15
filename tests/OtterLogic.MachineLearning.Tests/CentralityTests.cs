using OtterLogic.MachineLearning.Graphs;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class CentralityTests
{
    /// <summary>
    /// A star: every route between two leaves passes through the hub and nothing
    /// else, so the hub scores one and every leaf zero.
    /// </summary>
    [Fact]
    public void TheHubOfAStarCarriesEveryRoute()
    {
        var graph = WeightedGraph.FromEdges(5, new[] { (0, 1), (0, 2), (0, 3), (0, 4) });
        var centrality = Centrality.Betweenness(graph);

        Assert.Equal(1.0, centrality[0], 12);
        for (int leaf = 1; leaf < 5; leaf++)
            Assert.Equal(0.0, centrality[leaf], 12);
    }

    /// <summary>
    /// A path of five: networkx's normalised betweenness is 0, 1/2, 2/3, 1/2, 0.
    /// </summary>
    [Fact]
    public void MatchesTheTextbookValuesOnAPath()
    {
        var graph = WeightedGraph.FromEdges(5, new[] { (0, 1), (1, 2), (2, 3), (3, 4) });
        var centrality = Centrality.Betweenness(graph);

        var expected = new[] { 0.0, 0.5, 2.0 / 3.0, 0.5, 0.0 };
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], centrality[i], 12);
    }

    /// <summary>
    /// Two equal routes split the credit: in a square each corner carries half of
    /// the one pair it sits between.
    /// </summary>
    [Fact]
    public void EqualRoutesShareTheCredit()
    {
        var graph = WeightedGraph.FromEdges(4, new[] { (0, 1), (1, 2), (2, 3), (3, 0) });
        var centrality = Centrality.Betweenness(graph);

        Assert.All(centrality, c => Assert.Equal(1.0 / 6.0, c, 12));
    }

    /// <summary>
    /// Sampled sources estimate the exact answer: on a long path, where node i
    /// carries i(n-1-i) of the pairs, a fifth of the sources lands the middle
    /// within a few per cent and still leaves the ends at zero.
    /// </summary>
    [Fact]
    public void SampledSourcesEstimateTheExactValues()
    {
        const int n = 101;
        var graph = WeightedGraph.FromEdges(n, Enumerable.Range(0, n - 1).Select(i => (i, i + 1)));

        var exact = Centrality.Betweenness(graph);
        var sampled = Centrality.Betweenness(graph, maximumSources: 20);

        Assert.Equal(50.0 * 50.0 / ((n - 1) * (n - 2) / 2.0), exact[50], 12);
        Assert.InRange(sampled[50], 0.9 * exact[50], 1.1 * exact[50]);
        Assert.Equal(0.0, sampled[0], 12);
        Assert.Equal(0.0, sampled[n - 1], 12);
    }
}
