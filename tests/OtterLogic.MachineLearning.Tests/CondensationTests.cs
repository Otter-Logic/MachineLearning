using OtterLogic.MachineLearning.Graphs;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class CondensationTests
{
    /// <summary>A chain of dependence: each node one higher than what it depends on.</summary>
    [Fact]
    public void AChainRanksEachNodeAboveWhatItDependsOn()
    {
        var result = Condensation.Of(4, new[] { (3, 2), (2, 1), (1, 0) });

        Assert.Equal(4, result.ComponentCount);
        Assert.Equal(new[] { 0, 1, 2, 3 }, Enumerable.Range(0, 4).Select(result.HeightOf).ToArray());
    }

    /// <summary>Nodes that depend on each other share a rank, and what hangs off the loop sits above it.</summary>
    [Fact]
    public void ACycleIsOneComponentAtOneHeight()
    {
        var result = Condensation.Of(5, new[] { (1, 2), (2, 3), (3, 1), (1, 0), (4, 2) });

        Assert.Equal(3, result.ComponentCount);
        Assert.Equal(result.Component[1], result.Component[2]);
        Assert.Equal(result.Component[2], result.Component[3]);
        Assert.Equal(0, result.HeightOf(0));
        Assert.Equal(1, result.HeightOf(2));
        Assert.Equal(2, result.HeightOf(4));
    }

    /// <summary>Height is the longest chain below, not the shortest.</summary>
    [Fact]
    public void HeightFollowsTheLongestChain()
    {
        var result = Condensation.Of(4, new[] { (3, 0), (3, 2), (2, 1), (1, 0) });
        Assert.Equal(3, result.HeightOf(3));
    }

    /// <summary>A chain far longer than any call stack would take.</summary>
    [Fact]
    public void ALongChainDoesNotOverflow()
    {
        const int n = 200_000;
        var result = Condensation.Of(n, Enumerable.Range(1, n - 1).Select(i => (i, i - 1)));
        Assert.Equal(n - 1, result.HeightOf(n - 1));
    }
}
