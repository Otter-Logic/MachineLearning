namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// The knobs on a k-means fit. Defaults suit a few hundred to a few thousand
/// samples in a handful of standardised dimensions.
/// </summary>
public sealed record KMeansOptions
{
    /// <summary>Number of clusters, k.</summary>
    public int Clusters { get; init; } = 4;

    /// <summary>Hard cap on Lloyd iterations per restart.</summary>
    public int MaxIterations { get; init; } = 100;

    /// <summary>
    /// How many times to refit from a different k-means++ start, keeping the
    /// lowest inertia.
    /// <para>
    /// Lloyd's algorithm descends to a local optimum and stops, so restarts are
    /// the cheapest accuracy available. Ten is scikit-learn's default and costs
    /// milliseconds at these sizes.
    /// </para>
    /// </summary>
    public int Restarts { get; init; } = 10;

    /// <summary>
    /// Seed for the initialisation. Fixed by default, and it must stay that way:
    /// Grasshopper re-solves constantly, and a component that returns different
    /// clusters from identical inputs is unusable.
    /// </summary>
    public int Seed { get; init; } = 1;

    internal void Validate(int sampleCount)
    {
        if (Clusters < 1)
            throw new ArgumentOutOfRangeException(nameof(Clusters), Clusters, "Need at least one cluster.");
        if (Clusters > sampleCount)
            throw new ArgumentOutOfRangeException(nameof(Clusters), Clusters,
                $"Cannot fit {Clusters} clusters to {sampleCount} samples.");
        if (MaxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(MaxIterations), MaxIterations, "Need at least one iteration.");
        if (Restarts < 1)
            throw new ArgumentOutOfRangeException(nameof(Restarts), Restarts, "Need at least one restart.");
    }
}
