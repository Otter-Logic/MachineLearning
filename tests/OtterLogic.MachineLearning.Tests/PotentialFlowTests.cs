using OtterLogic.MachineLearning.Graphs;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class PotentialFlowTests
{
    /// <summary>
    /// A chain grounded at one end with a unit injected at the other: the same unit
    /// flows along every edge, and the potential climbs by one over the conductance each step.
    /// </summary>
    [Fact]
    public void AChainCarriesWhatIsInjectedAllTheWayToGround()
    {
        var graph = WeightedGraph.FromEdges(4, new[] { (0, 1, 2.0), (1, 2, 1.0), (2, 3, 0.5) });
        var result = PotentialFlow.Solve(graph, new[] { 0.0, 0.0, 0.0, 1.0 }, new[] { 0 });

        Assert.True(result.Converged);
        Assert.Equal(0.0, result.Potential[0]);
        Assert.Equal(0.5, result.Potential[1], 8);
        Assert.Equal(1.5, result.Potential[2], 8);
        Assert.Equal(3.5, result.Potential[3], 8);
        Assert.Equal(1.0, result.Flow(3, 2, 0.5), 8);
        Assert.Equal(1.0, result.Flow(1, 0, 2.0), 8);
    }

    /// <summary>
    /// Two equal routes to ground share the flow equally — the answer a route search
    /// cannot give, because it has to pick one.
    /// </summary>
    [Fact]
    public void EqualRoutesShareTheFlowEqually()
    {
        var graph = WeightedGraph.FromEdges(4, new[] { (0, 1), (0, 2), (1, 3), (2, 3) });
        var result = PotentialFlow.Solve(graph, new[] { 1.0, 0.0, 0.0, 0.0 }, new[] { 3 });

        Assert.Equal(result.Potential[1], result.Potential[2], 10);
        Assert.Equal(0.5, result.Flow(0, 1, 1.0), 8);
        Assert.Equal(0.5, result.Flow(0, 2, 1.0), 8);
    }

    /// <summary>What goes in comes out: the flow into the grounded nodes is the total injected.</summary>
    [Fact]
    public void EverythingInjectedDrainsAtTheGroundedNodes()
    {
        var graph = WeightedGraph.FromEdges(6, new[] { (0, 1, 1.0), (1, 2, 3.0), (2, 3, 1.0), (3, 4, 2.0), (4, 5, 1.0), (1, 4, 0.7) });
        var injection = new[] { 0.0, 1.0, 2.0, 0.5, 1.5, 0.0 };
        var result = PotentialFlow.Solve(graph, injection, new[] { 0, 5 });

        double drained = result.Flow(1, 0, 1.0) + result.Flow(4, 5, 1.0);
        Assert.Equal(5.0, drained, 7);
    }

    /// <summary>A piece with no grounded node has nowhere to drain, and says so rather than failing.</summary>
    [Fact]
    public void APieceWithNoGroundIsUnreachedNotAnError()
    {
        var graph = WeightedGraph.FromEdges(4, new[] { (0, 1), (2, 3) });
        var result = PotentialFlow.Solve(graph, new[] { 0.0, 1.0, 1.0, 1.0 }, new[] { 0 });

        Assert.True(result.Reached[1]);
        Assert.False(result.Reached[2]);
        Assert.True(double.IsNaN(result.Potential[3]));
        Assert.Equal(0.0, result.Flow(2, 3, 1.0));
    }
}
