using OtterLogic.MachineLearning.Graphs;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The graph contract every graph algorithm above this layer consumes: how edges
/// are normalised on the way in, the nearest-neighbour graph against
/// scikit-learn's, and the propagation operator against dense arithmetic.
/// </summary>
public sealed class WeightedGraphTests
{
    [Fact]
    public void NearestNeighboursMatchesScikitLearn()
    {
        var fixture = Fixture.Load("graph");
        var x = fixture.Matrix("x");
        var graph = WeightedGraph.NearestNeighbours(x, fixture.Int("neighbours"));

        var expected = fixture.Section("expected").Matrix("edges");
        var actual = graph.Edges().ToArray();

        Assert.Equal(expected.GetLength(0), actual.Length);
        for (int e = 0; e < actual.Length; e++)
        {
            Assert.Equal((int)expected[e, 0], actual[e].A);
            Assert.Equal((int)expected[e, 1], actual[e].B);
            Assert.Equal(expected[e, 2], actual[e].Weight);
        }
    }

    [Theory]
    [InlineData(0.0, "propagate_plain")]
    [InlineData(1.0, "propagate_self")]
    public void PropagationMatchesTheDenseFormula(double selfWeight, string key)
    {
        var fixture = Fixture.Load("graph");
        var x = fixture.Matrix("x");
        var graph = WeightedGraph.NearestNeighbours(x, fixture.Int("neighbours"));

        Numeric.Close(fixture.Section("expected").Matrix(key), graph.Propagate(x, selfWeight), 1e-12, key);
    }

    /// <summary>
    /// Real edge lists arrive untidy — each edge reported from both ends, the odd
    /// self-reference, a zero where a relationship was switched off. Each is
    /// normalised on the way in, so an algorithm above never meets one.
    /// </summary>
    [Fact]
    public void NormalisesUntidyEdgeLists()
    {
        var graph = WeightedGraph.FromEdges(5, new[]
        {
            (0, 1, 0.5),
            (1, 0, 2.0),   // the same edge from the other end: the larger weight wins
            (2, 2, 9.0),   // self-loop: dropped
            (3, 4, 0.0),   // zero weight: dropped
            (4, 1, 1.0),
        });

        Assert.Equal(2, graph.EdgeCount);
        Assert.Equal(new[] { (0, 1, 2.0), (1, 4, 1.0) }, graph.Edges().ToArray());
        Assert.Equal(new[] { 0, 4 }, graph.Neighbours(1).ToArray());
        Assert.Equal(3.0, graph.Degree(1));
        Assert.Equal(0.0, graph.Degree(2));
    }

    [Fact]
    public void RejectsEdgesItCannotRepresent()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => WeightedGraph.FromEdges(3, new[] { (0, 3) }));
        Assert.Throws<ArgumentOutOfRangeException>(() => WeightedGraph.FromEdges(3, new[] { (0, 1, -1.0) }));
        Assert.Throws<ArgumentOutOfRangeException>(() => WeightedGraph.FromEdges(3, new[] { (0, 1, double.NaN) }));
    }

    /// <summary>
    /// Components are numbered by their lowest node rather than by traversal
    /// order, so the numbering is a property of the graph.
    /// </summary>
    [Fact]
    public void NumbersComponentsByTheirLowestNode()
    {
        var graph = WeightedGraph.FromEdges(6, new[] { (4, 5), (1, 3), (3, 0) });
        var components = graph.ConnectedComponents(out int count);

        Assert.Equal(3, count);
        Assert.Equal(new[] { 0, 0, 1, 0, 2, 2 }, components);
    }

    /// <summary>
    /// A node with no edge and no self-loop has nothing to average over. It must
    /// come back as zeros rather than as the NaN that dividing by its zero degree
    /// would produce.
    /// </summary>
    [Fact]
    public void IsolatedNodesPropagateToZeroNotNaN()
    {
        var graph = WeightedGraph.FromEdges(3, new[] { (0, 1) });
        var y = graph.Propagate(new double[,] { { 1.0 }, { 2.0 }, { 3.0 } }, selfWeight: 0.0);

        Assert.Equal(0.0, y[2, 0]);
        Assert.Equal(2.0, y[0, 0]);
        Assert.Equal(1.0, y[1, 0]);
    }

    [Fact]
    public void ReweightingKeepsTopologyAndDropsZeroes()
    {
        var graph = WeightedGraph.FromEdges(4, new[] { (0, 1), (1, 2), (2, 3) });
        var reweighted = graph.Reweight((a, b, w) => a == 1 ? 0.0 : w * (a + b));

        Assert.Equal(new[] { (0, 1, 1.0), (2, 3, 5.0) }, reweighted.Edges().ToArray());
    }
}
