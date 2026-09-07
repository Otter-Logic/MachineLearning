using System.Globalization;
using System.Text;

namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// The outcome of a grouping: which member went where, how sure the model is,
/// and what each group looks like in the units it came in as.
/// </summary>
public sealed class DesignGroupingResult
{
    internal DesignGroupingResult(
        int[] labels,
        double[,] responsibilities,
        double[] confidence,
        double[,] centres,
        double[] groupShares,
        GaussianMixtureResult mixture,
        PrincipalComponents? principalComponents,
        int[] keptColumns,
        int inputColumnCount)
    {
        Labels = labels;
        Responsibilities = responsibilities;
        Confidence = confidence;
        Centres = centres;
        GroupShares = groupShares;
        Mixture = mixture;
        PrincipalComponents = principalComponents;
        KeptColumns = keptColumns;
        InputColumnCount = inputColumnCount;
    }

    /// <summary>Group index per member, 0-based.</summary>
    public int[] Labels { get; }

    /// <summary>Soft membership, n x k. Row sums to one.</summary>
    public double[,] Responsibilities { get; }

    /// <summary>
    /// Highest responsibility per member. Anything much below 0.6 is a member
    /// sitting between two groups, and is worth a look rather than a rubber
    /// stamp.
    /// </summary>
    public double[] Confidence { get; }

    /// <summary>
    /// Group centres, k x d, mapped back through whitening, PCA,
    /// standardisation and the log into the original units.
    /// <para>
    /// The output that decides whether the tool is useful. A grouping nobody can
    /// name is a grouping nobody will act on, and this is what lets somebody say
    /// "group three is the high-moment family".
    /// </para>
    /// </summary>
    public double[,] Centres { get; }

    /// <summary>Fraction of members in each group, by mixing weight.</summary>
    public double[] GroupShares { get; }

    /// <summary>The underlying fit, for BIC, log-likelihood and convergence.</summary>
    public GaussianMixtureResult Mixture { get; }

    /// <summary>The fitted decomposition, or null if PCA was skipped.</summary>
    public PrincipalComponents? PrincipalComponents { get; }

    /// <summary>Input columns that survived the constant-column check.</summary>
    public int[] KeptColumns { get; }

    /// <summary>Number of columns supplied.</summary>
    public int InputColumnCount { get; }

    /// <summary>Member indices bucketed by group, ready to drive geometry downstream.</summary>
    public int[][] Groups()
    {
        int k = GroupShares.Length;
        var buckets = new List<int>[k];
        for (int c = 0; c < k; c++)
            buckets[c] = new List<int>();

        for (int i = 0; i < Labels.Length; i++)
            buckets[Labels[i]].Add(i);

        return buckets.Select(b => b.ToArray()).ToArray();
    }

    /// <summary>
    /// A short diagnostic block, meant to be wired straight to a panel. Reports
    /// the things that decide whether to believe the answer: whether EM
    /// converged, what the model cost in parameters, how much variance survived
    /// PCA, and how confident the assignments are on average.
    /// </summary>
    public string Report()
    {
        var text = new StringBuilder();
        var invariant = CultureInfo.InvariantCulture;

        text.AppendLine($"Members      {Labels.Length}");
        text.AppendLine($"Groups       {GroupShares.Length}");
        text.AppendLine(
            $"Columns      {KeptColumns.Length} of {InputColumnCount} kept"
            + (KeptColumns.Length < InputColumnCount
                ? $" (dropped: {string.Join(", ", Enumerable.Range(0, InputColumnCount).Except(KeptColumns))})"
                : string.Empty));

        if (PrincipalComponents is { } pca)
            text.AppendLine(
                $"Components   {pca.Count} retained, "
                + $"{(pca.ExplainedVarianceRatio * 100.0).ToString("0.00", invariant)}% of variance");
        else
            text.AppendLine("Components   PCA skipped");

        text.AppendLine(
            $"EM           {(Mixture.Converged ? "converged" : "hit the iteration cap")} "
            + $"after {Mixture.Iterations} iterations");
        text.AppendLine($"Parameters   {Mixture.ParameterCount}");
        text.AppendLine($"Log-lik      {Mixture.LogLikelihood.ToString("0.000", invariant)} "
            + $"({Mixture.MeanLogLikelihood.ToString("0.0000", invariant)} per member)");
        text.AppendLine($"BIC          {Mixture.Bic.ToString("0.000", invariant)}");
        text.AppendLine($"AIC          {Mixture.Aic.ToString("0.000", invariant)}");
        text.AppendLine($"Confidence   mean {Confidence.Average().ToString("0.000", invariant)}, "
            + $"{Confidence.Count(c => c < 0.6)} member(s) below 0.6");
        text.Append($"Shares       {string.Join(", ", GroupShares.Select(w => w.ToString("0.000", invariant)))}");

        return text.ToString();
    }
}
