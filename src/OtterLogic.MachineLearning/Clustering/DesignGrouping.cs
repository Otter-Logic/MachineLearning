namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// The whole grouping pipeline, end to end: preprocess, decompose, fit, and map
/// the answer back into the units it arrived in.
/// <para>
/// This is the entry point a Grasshopper component calls. The adaptor's only job
/// is to unpack a data tree into an array and pack the results back out again —
/// every decision about what happens in between lives here, so the same call
/// works identically from a test, from another toolkit, or from a Rhino command
/// if one is ever wanted.
/// </para>
/// </summary>
public static class DesignGrouping
{
    /// <summary>
    /// Groups n members described by d non-negative values each.
    /// </summary>
    /// <param name="data">
    /// n x d. For the design grouping case these are six-degree-of-freedom
    /// magnitudes per member — three forces and three moments, unsigned.
    /// </param>
    /// <param name="options">Pipeline settings.</param>
    public static DesignGroupingResult Group(double[,] data, DesignGroupingOptions options)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(options);

        int n = data.GetLength(0);
        int d = data.GetLength(1);

        if (n < 2)
            throw new ArgumentException("Need at least two members to group.", nameof(data));
        if (options.Groups > n)
            throw new ArgumentException(
                $"Cannot ask for {options.Groups} groups from {n} members.", nameof(options));

        var pipeline = FeaturePipeline.Fit(data, options.LogTransform, options.NormaliseRows, options.Weights);
        var prepared = pipeline.Transform(data);

        PrincipalComponents? pca = null;
        var forFitting = prepared;

        if (options.PcaVariance > 0.0)
        {
            pca = PrincipalComponents.Fit(prepared, options.PcaVariance, options.Whiten);
            forFitting = pca.Transform(prepared);
        }

        var mixture = GaussianMixture.Fit(forFitting, options.ToMixtureOptions());
        var order = StableOrder(mixture);

        var responsibilities = Reorder(mixture.Responsibilities, order);
        var labels = Posterior.ArgMax(responsibilities);
        var confidence = Posterior.RowMax(responsibilities);
        var shares = order.Select(c => mixture.MixingWeights[c]).ToArray();

        var centres = BackTransform(mixture.Means, order, pca, pipeline);

        return new DesignGroupingResult(
            labels, responsibilities, confidence, centres, shares,
            mixture, pca, pipeline.KeptColumns, d);
    }

    /// <summary>
    /// Sweeps k over a range and reports the fit criteria for each, so the count
    /// can be chosen by looking rather than guessing.
    /// <para>
    /// Deliberately does not pick a winner. The lowest BIC is a suggestion, not
    /// an answer: the useful k is usually the one that is both near the elbow and
    /// means something to whoever has to detail the result, and no criterion
    /// knows about the second half of that.
    /// </para>
    /// </summary>
    /// <param name="data">n x d, as for <see cref="Group"/>.</param>
    /// <param name="options">Pipeline settings; <see cref="DesignGroupingOptions.Groups"/> is ignored.</param>
    /// <param name="minimumGroups">Lowest k to try, at least one.</param>
    /// <param name="maximumGroups">Highest k to try.</param>
    public static IReadOnlyList<GroupCountCandidate> ChooseGroupCount(
        double[,] data, DesignGroupingOptions options, int minimumGroups = 2, int maximumGroups = 10)
    {
        ArgumentNullException.ThrowIfNull(data);
        ArgumentNullException.ThrowIfNull(options);

        if (minimumGroups < 1)
            throw new ArgumentOutOfRangeException(nameof(minimumGroups), minimumGroups, "k starts at one.");
        if (maximumGroups < minimumGroups)
            throw new ArgumentOutOfRangeException(nameof(maximumGroups), maximumGroups,
                "Upper bound is below the lower bound.");

        int limit = Math.Min(maximumGroups, data.GetLength(0));
        var candidates = new List<GroupCountCandidate>(limit - minimumGroups + 1);

        for (int k = minimumGroups; k <= limit; k++)
        {
            var result = Group(data, options with { Groups = k });
            candidates.Add(new GroupCountCandidate(
                k,
                result.Mixture.Bic,
                result.Mixture.Aic,
                result.Confidence.Average()));
        }

        return candidates;
    }

    /// <summary>
    /// Orders components by descending mixing weight.
    /// <para>
    /// Not cosmetic. EM labels its components in whatever order initialisation
    /// happened to produce, so a small change upstream can permute them — group
    /// zero becomes group two — and every colour and geometry assignment
    /// downstream jumps for no reason a user can see. Sorting on a property of
    /// the fit rather than on its history is what stops that.
    /// </para>
    /// </summary>
    private static int[] StableOrder(GaussianMixtureResult mixture)
    {
        int k = mixture.ComponentCount;
        return Enumerable.Range(0, k)
            .OrderByDescending(c => mixture.MixingWeights[c])
            .ThenBy(c => c)
            .ToArray();
    }

    private static double[,] Reorder(double[,] responsibilities, int[] order)
    {
        int n = responsibilities.GetLength(0);
        int k = order.Length;
        var reordered = new double[n, k];

        for (int i = 0; i < n; i++)
            for (int c = 0; c < k; c++)
                reordered[i, c] = responsibilities[i, order[c]];

        return reordered;
    }

    private static double[,] BackTransform(
        double[,] means, int[] order, PrincipalComponents? pca, FeaturePipeline pipeline)
    {
        int k = order.Length;
        int width = means.GetLength(1);

        var ordered = new double[k, width];
        for (int c = 0; c < k; c++)
            for (int j = 0; j < width; j++)
                ordered[c, j] = means[order[c], j];

        var inPreparedSpace = pca is null ? ordered : pca.InverseTransform(ordered);
        return pipeline.InverseTransform(inPreparedSpace);
    }
}

/// <summary>One row of a group-count sweep.</summary>
/// <param name="Groups">The k that was tried.</param>
/// <param name="Bic">Bayesian information criterion, lower is better.</param>
/// <param name="Aic">Akaike information criterion, lower is better.</param>
/// <param name="MeanConfidence">
/// Average highest responsibility. A model whose assignments all sit near 1/k
/// has not found structure, whatever its BIC says.
/// </param>
public readonly record struct GroupCountCandidate(
    int Groups,
    double Bic,
    double Aic,
    double MeanConfidence);
