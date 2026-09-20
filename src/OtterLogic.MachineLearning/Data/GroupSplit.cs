namespace OtterLogic.MachineLearning.Data;

/// <summary>
/// Holds back whole groups for testing, never individual rows.
/// <para>
/// There is deliberately no split by row. Rows from one group are not independent
/// samples — everything exported from one project shares its author and its
/// habits — so a row held out at random has near-copies of itself in the training
/// set, and the score measures how well the model recognises a project it has
/// already seen. The question a user is actually asking is how it will do on the
/// next project, and only holding out whole ones answers that. How large the gap is
/// depends on the data and has not been measured here yet — when it has, the number
/// belongs in this comment.
/// </para>
/// <para>
/// The same holds where the group is not a project. Any rows produced together — one
/// sweep, one batch, one machine — go on one side or the other.
/// </para>
/// </summary>
public static class GroupSplit
{
    /// <summary>
    /// Chooses the groups to hold out.
    /// </summary>
    /// <param name="groups">The group of every row.</param>
    /// <param name="testFraction">
    /// Share of the <em>groups</em> to hold out, rounded to the nearest whole group
    /// and kept between one group and all but one. A quarter by default: enough that
    /// the score is not one project's luck, without starving the fit when there are
    /// only a handful. Groups rather than rows, so the choice does not depend on
    /// which projects happen to be large — the row share that results is reported on
    /// <see cref="GroupSplitResult.TestRowFraction"/>.
    /// </param>
    /// <param name="seed">
    /// Fixed by default, for the usual reason — Grasshopper re-solves constantly, and
    /// a score that changes with no input changing is unusable. Change it to see how
    /// much the score owes to which projects were held out; with few groups, a lot.
    /// </param>
    public static GroupSplitResult Holdout(IReadOnlyList<string> groups, double testFraction = 0.25, int seed = 1)
    {
        ArgumentNullException.ThrowIfNull(groups);

        if (!(testFraction > 0.0 && testFraction < 1.0))
            throw new ArgumentOutOfRangeException(nameof(testFraction), testFraction,
                "The test fraction must be between 0 and 1, exclusive.");

        // Sorted before shuffling so the choice depends on which groups exist and
        // on the seed, not on the order the rows happened to arrive in.
        var distinct = groups.Distinct(StringComparer.Ordinal).OrderBy(g => g, StringComparer.Ordinal).ToArray();

        if (distinct.Length < 2)
            throw new ArgumentException(
                $"Need at least two groups to hold one out, and there {(distinct.Length == 1 ? "is one" : "are none")}. "
                + "A score from rows of the same group the model trained on says nothing about the next one — "
                + "add another model to the dataset first.", nameof(groups));

        var rng = new Random(seed);
        for (int i = distinct.Length - 1; i > 0; i--)
        {
            int j = rng.Next(i + 1);
            (distinct[i], distinct[j]) = (distinct[j], distinct[i]);
        }

        int held = (int)Math.Round(testFraction * distinct.Length, MidpointRounding.AwayFromZero);
        held = Math.Clamp(held, 1, distinct.Length - 1);

        var test = new HashSet<string>(distinct.Take(held), StringComparer.Ordinal);
        var trainRows = new List<int>();
        var testRows = new List<int>();

        for (int i = 0; i < groups.Count; i++)
            (test.Contains(groups[i]) ? testRows : trainRows).Add(i);

        return new GroupSplitResult(
            trainRows.ToArray(), testRows.ToArray(),
            distinct.Skip(held).OrderBy(g => g, StringComparer.Ordinal).ToArray(),
            distinct.Take(held).OrderBy(g => g, StringComparer.Ordinal).ToArray());
    }
}

/// <summary>Which rows, and which groups, landed on each side of a <see cref="GroupSplit"/>.</summary>
public sealed class GroupSplitResult
{
    internal GroupSplitResult(int[] trainRows, int[] testRows, string[] trainGroups, string[] testGroups)
    {
        TrainRows = trainRows;
        TestRows = testRows;
        TrainGroups = trainGroups;
        TestGroups = testGroups;
    }

    /// <summary>Row indices to fit on, ascending.</summary>
    public int[] TrainRows { get; }

    /// <summary>Row indices held out, ascending.</summary>
    public int[] TestRows { get; }

    /// <summary>The groups fitted on.</summary>
    public string[] TrainGroups { get; }

    /// <summary>The groups held out. Name them when reporting a score: it is a score on these.</summary>
    public string[] TestGroups { get; }

    /// <summary>Share of the rows that were held out, which follows from how large the chosen groups are.</summary>
    public double TestRowFraction => (double)TestRows.Length / (TrainRows.Length + TestRows.Length);
}
