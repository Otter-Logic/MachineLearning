using OtterLogic.Graphs;
using OtterLogic.MachineLearning.Distances;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The nearest-neighbour graph against scikit-learn's, and the propagation
/// operator run over it against dense arithmetic.
/// <para>
/// The second is a check on Graphs' <see cref="WeightedGraph.Propagate"/> and sits
/// up here all the same: its reference graph is scikit-learn's, and the fixture
/// and the code that builds that graph from samples both live in this repo. The
/// rest of the graph contract is tested in Graphs.
/// </para>
/// </summary>
public sealed class NeighbourGraphTests
{
    [Fact]
    public void NearestNeighboursMatchesScikitLearn()
    {
        var fixture = Fixture.Load("graph");
        var x = fixture.Matrix("x");
        var graph = NeighbourGraph.Of(x, fixture.Int("neighbours"));

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
        var graph = NeighbourGraph.Of(x, fixture.Int("neighbours"));

        Numeric.Close(fixture.Section("expected").Matrix(key), graph.Propagate(x, selfWeight), 1e-12, key);
    }

    /// <summary>
    /// For routing, an edge weighs the distance it spans, and is kept if either end
    /// chose it. Four points on a line, unevenly spaced: with one neighbour each, 0 and
    /// 1 choose each other, 2 chooses 1, and 3 chooses 2.
    /// </summary>
    [Fact]
    public void ByDistanceWeighsEdgesByTheirLengthAndKeepsOneSidedChoices()
    {
        var x = new double[,] { { 0.0, 0.0 }, { 1.0, 0.0 }, { 3.0, 0.0 }, { 7.0, 0.0 } };
        var graph = NeighbourGraph.ByDistance(x, 1);

        Assert.Equal(new[] { (0, 1, 1.0), (1, 2, 2.0), (2, 3, 4.0) }, graph.Edges().ToArray());
    }

    [Fact]
    public void ByDistanceLeavesOutEdgesLongerThanTheMaximum()
    {
        var x = new double[,] { { 0.0, 0.0 }, { 1.0, 0.0 }, { 3.0, 0.0 }, { 7.0, 0.0 } };
        var graph = NeighbourGraph.ByDistance(x, 3, maximumDistance: 2.5);

        Assert.Equal(new[] { (0, 1, 1.0), (1, 2, 2.0) }, graph.Edges().ToArray());
        Assert.Equal(0, graph.Neighbours(3).Length);
    }

    /// <summary>
    /// What blocks an edge is the caller's to say. It is asked once per pair, lower
    /// index first, however many times the pair turns up among the neighbours.
    /// </summary>
    [Fact]
    public void ByDistanceAsksTheCallerOncePerPairWhatIsBlocked()
    {
        var x = new double[,] { { 0.0, 0.0 }, { 1.0, 0.0 }, { 3.0, 0.0 }, { 7.0, 0.0 } };
        var asked = new List<(int, int)>();

        var graph = NeighbourGraph.ByDistance(x, 3, blocked: (a, b) =>
        {
            asked.Add((a, b));
            return a == 1 && b == 2;
        });

        Assert.Equal(6, asked.Count);
        Assert.Equal(asked.Distinct().Count(), asked.Count);
        Assert.All(asked, pair => Assert.True(pair.Item1 < pair.Item2));
        Assert.DoesNotContain(2, graph.Neighbours(1).ToArray());
        Assert.Contains(2, graph.Neighbours(0).ToArray());
    }

    /// <summary>
    /// The whole preparation a routing caller makes, end to end: a grid of points, an
    /// obstacle across the middle, and a route that has to go over the top of it. On a
    /// unit grid with diagonals the detour is known exactly - six diagonal steps and two
    /// straight ones - and it clips both top corners of the obstacle, which touching
    /// allows and passing through would not.
    /// </summary>
    [Fact]
    public void AGridRoutesRoundAnObstacleByTheExactDetour()
    {
        const int side = 9;
        var x = new double[side * side, 2];
        for (int r = 0; r < side; r++)
        {
            for (int c = 0; c < side; c++)
            {
                x[r * side + c, 0] = c;
                x[r * side + c, 1] = r;
            }
        }

        // A wall from below the grid up to y = 6.5, between x = 3.5 and x = 5.5.
        var wall = new double[,] { { 3.5, -1.0 }, { 5.5, -1.0 }, { 5.5, 6.5 }, { 3.5, 6.5 } };
        var obstacles = new OtterLogic.Graphs.Planar.PlanarObstacles(new[] { wall }, 1e-9);

        var enclosed = Enumerable.Range(0, side * side).Select(i => obstacles.Contains(x[i, 0], x[i, 1])).ToArray();
        var graph = NeighbourGraph.ByDistance(x, 8, maximumDistance: 1.5,
            blocked: (a, b) => enclosed[a] || enclosed[b] || obstacles.Blocks(x[a, 0], x[a, 1], x[b, 0], x[b, 1]));

        int start = 4 * side + 0, end = 4 * side + 8;
        var routes = Dijkstra.From(graph, new[] { start }, targets: new[] { end });

        Assert.Equal(2.0 + 6.0 * Math.Sqrt(2.0), routes.Cost[end], 9);

        var route = routes.RouteTo(end);
        for (int k = 1; k < route.Length; k++)
            Assert.False(obstacles.Blocks(x[route[k - 1], 0], x[route[k - 1], 1], x[route[k], 0], x[route[k], 1]));

        // Fourteen grid points lie inside the wall; each keeps its index and connects to nothing.
        Assert.Equal(14, enclosed.Count(e => e));
        Assert.All(Enumerable.Range(0, side * side).Where(i => enclosed[i]), i => Assert.Equal(0, graph.Neighbours(i).Length));
    }
}
