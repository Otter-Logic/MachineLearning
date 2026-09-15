using OtterLogic.MachineLearning.Graphs;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class CutVerticesTests
{
    /// <summary>
    /// Two triangles joined through one node: removing it strands the smaller
    /// side, and no other node is a cut at all.
    /// </summary>
    [Fact]
    public void TheJoiningNodeStrandsTheSmallerSide()
    {
        // 0-1-2 triangle, 2-3 bridge, 3-4-5-6 square with a diagonal.
        var graph = WeightedGraph.FromEdges(7, new[]
        {
            (0, 1), (1, 2), (2, 0),
            (2, 3),
            (3, 4), (4, 5), (5, 6), (6, 3), (3, 5),
        });

        var stranded = CutVertices.Stranded(graph);

        // Removing 2 leaves {0,1} and {3,4,5,6}: two stranded.
        Assert.Equal(2, stranded[2]);
        // Removing 3 leaves {0,1,2} and {4,5,6}: three stranded, ties either way.
        Assert.Equal(3, stranded[3]);
        foreach (int node in new[] { 0, 1, 4, 5, 6 })
            Assert.Equal(0, stranded[node]);
    }

    /// <summary>
    /// A chain: every interior node is a cut, and how much it strands grows toward
    /// the middle — which is what separates a spur from a real weak point.
    /// </summary>
    [Fact]
    public void AChainStrandsMostAtItsMiddle()
    {
        var graph = WeightedGraph.FromEdges(5, new[] { (0, 1), (1, 2), (2, 3), (3, 4) });
        Assert.Equal(new[] { 0, 1, 2, 1, 0 }, CutVertices.Stranded(graph));
    }

    /// <summary>A cycle has no cut vertex, and the root of the search is not special.</summary>
    [Fact]
    public void ACycleHasNone()
    {
        var graph = WeightedGraph.FromEdges(6, Enumerable.Range(0, 6).Select(i => (i, (i + 1) % 6)));
        Assert.All(CutVertices.Stranded(graph), s => Assert.Equal(0, s));
    }

    /// <summary>
    /// The search root is a cut exactly when it has two separable children — a
    /// star's hub strands everything but one leaf.
    /// </summary>
    [Fact]
    public void AStarsHubStrandsAllButOneLeaf()
    {
        var graph = WeightedGraph.FromEdges(5, new[] { (0, 1), (0, 2), (0, 3), (0, 4) });
        Assert.Equal(new[] { 3, 0, 0, 0, 0 }, CutVertices.Stranded(graph));
    }

    /// <summary>Separate components are read separately, and an isolated node strands nothing.</summary>
    [Fact]
    public void ComponentsAreReadOnTheirOwn()
    {
        var graph = WeightedGraph.FromEdges(6, new[] { (0, 1), (1, 2), (3, 4) });
        Assert.Equal(new[] { 0, 1, 0, 0, 0, 0 }, CutVertices.Stranded(graph));
    }
}
