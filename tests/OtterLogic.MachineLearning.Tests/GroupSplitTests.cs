using OtterLogic.MachineLearning.Data;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The split exists to keep a group on one side. Everything else about it — how
/// many, which ones — is secondary to that, so that is what is pinned hardest.
/// </summary>
public sealed class GroupSplitTests
{
    private static string[] Groups(params (string Name, int Rows)[] groups)
        => groups.SelectMany(g => Enumerable.Repeat(g.Name, g.Rows)).ToArray();

    [Fact]
    public void NoGroupEverStraddlesTheSplit()
    {
        var groups = Groups(("a", 5), ("b", 50), ("c", 1), ("d", 12), ("e", 8), ("f", 3), ("g", 20), ("h", 9));

        for (int seed = 0; seed < 25; seed++)
        {
            var split = GroupSplit.Holdout(groups, 0.25, seed);

            Assert.Empty(split.TrainGroups.Intersect(split.TestGroups));
            Assert.All(split.TrainRows, i => Assert.Contains(groups[i], split.TrainGroups));
            Assert.All(split.TestRows, i => Assert.Contains(groups[i], split.TestGroups));
            Assert.Equal(groups.Length, split.TrainRows.Length + split.TestRows.Length);
            Assert.Equal(2, split.TestGroups.Length);
        }
    }

    [Fact]
    public void DoesNotDependOnTheOrderTheRowsArrivedIn()
    {
        var groups = Groups(("a", 3), ("b", 4), ("c", 5), ("d", 6));
        var reversed = Enumerable.Reverse(groups).ToArray();

        Assert.Equal(GroupSplit.Holdout(groups).TestGroups, GroupSplit.Holdout(reversed).TestGroups);
    }

    [Fact]
    public void ReturnsTheSameSplitEveryTime()
    {
        var groups = Groups(("a", 3), ("b", 4), ("c", 5), ("d", 6), ("e", 7));

        Assert.Equal(GroupSplit.Holdout(groups, seed: 7).TestRows, GroupSplit.Holdout(groups, seed: 7).TestRows);
    }

    [Theory]
    [InlineData(0.01)]
    [InlineData(0.99)]
    public void AlwaysLeavesSomethingOnBothSides(double fraction)
    {
        var split = GroupSplit.Holdout(Groups(("a", 3), ("b", 4), ("c", 5)), fraction);

        Assert.NotEmpty(split.TrainRows);
        Assert.NotEmpty(split.TestRows);
    }

    [Fact]
    public void RefusesASingleGroupAndSaysWhy()
    {
        var ex = Assert.Throws<ArgumentException>(() => GroupSplit.Holdout(Groups(("only", 40))));

        Assert.Contains("two groups", ex.Message);
    }
}
