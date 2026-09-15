using OtterLogic.MachineLearning.Graphs;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class ShortestPathsTests
{
    /// <summary>
    /// A square with a cheap diagonal: the diagonal wins over the two sides, and the
    /// route is read back from the previous nodes.
    /// </summary>
    [Fact]
    public void FindsTheCheapestRouteAndReadsItBack()
    {
        var graph = WeightedGraph.FromEdges(4, new[] { (0, 1, 1.0), (1, 2, 1.0), (2, 3, 1.0), (3, 0, 1.0), (0, 2, 1.5) });
        var result = ShortestPaths.From(graph, new[] { 0 });

        Assert.Equal(new[] { 0.0, 1.0, 1.5, 1.0 }, result.Cost);
        Assert.Equal(new[] { 0, 2 }, result.RouteTo(2));
        Assert.All(result.Source, s => Assert.Equal(0, s));
    }

    /// <summary>Several sources: every node is reached from the nearest one.</summary>
    [Fact]
    public void EveryNodeIsReachedFromItsNearestSource()
    {
        var graph = WeightedGraph.FromEdges(5, new[] { (0, 1), (1, 2), (2, 3), (3, 4) });
        var result = ShortestPaths.From(graph, new[] { 0, 4 });

        Assert.Equal(new[] { 0, 0, 0, 4, 4 }, result.Source);
        Assert.Equal(new[] { 0.0, 1.0, 2.0, 1.0, 0.0 }, result.Cost);
    }

    /// <summary>
    /// Equal-cost choices go to the lower-numbered source, so the answer does not
    /// depend on the order the sources were listed in.
    /// </summary>
    [Fact]
    public void TiesGoToTheLowerSourceWhateverTheOrderGiven()
    {
        var graph = WeightedGraph.FromEdges(3, new[] { (0, 1), (1, 2) });

        var forwards = ShortestPaths.From(graph, new[] { 0, 2 });
        var backwards = ShortestPaths.From(graph, new[] { 2, 0 });

        Assert.Equal(0, forwards.Source[1]);
        Assert.Equal(forwards.Source, backwards.Source);
        Assert.Equal(forwards.Previous, backwards.Previous);
    }

    /// <summary>
    /// The cost is the caller's, per direction: an edge priced at infinity one way is
    /// a one-way edge, and a node behind it is unreached.
    /// </summary>
    [Fact]
    public void ACallerCostCanForbidADirection()
    {
        var graph = WeightedGraph.FromEdges(3, new[] { (0, 1), (1, 2) });
        var result = ShortestPaths.From(graph, new[] { 0 }, (from, to, _) => to > from && to == 2 ? double.PositiveInfinity : 1.0);

        Assert.True(result.Reaches(1));
        Assert.False(result.Reaches(2));
        Assert.Equal(double.PositiveInfinity, result.Cost[2]);
        Assert.Empty(result.RouteTo(2));
    }

    [Fact]
    public void RejectsANegativeCostAndASourceOutsideTheGraph()
    {
        var graph = WeightedGraph.FromEdges(2, new[] { (0, 1) });

        Assert.Throws<InvalidOperationException>(() => ShortestPaths.From(graph, new[] { 0 }, (_, _, _) => -1.0));
        Assert.Throws<ArgumentOutOfRangeException>(() => ShortestPaths.From(graph, new[] { 2 }));
    }
}
