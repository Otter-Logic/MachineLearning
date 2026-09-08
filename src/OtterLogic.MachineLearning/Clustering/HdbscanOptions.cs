namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// The two knobs on an HDBSCAN fit, and only two — the point of the algorithm is
/// that it does not need to be told how many clusters to look for.
/// </summary>
public sealed record HdbscanOptions
{
    /// <summary>
    /// The smallest group of points that counts as a cluster rather than as
    /// noise.
    /// <para>
    /// The one setting that matters, and the one that reads in the units of the
    /// problem: "fewer than this many members is not a family worth detailing".
    /// Raise it and small groups dissolve into noise; lower it and the fit starts
    /// naming coincidences.
    /// </para>
    /// </summary>
    public int MinimumClusterSize { get; init; } = 5;

    /// <summary>
    /// How many neighbours the density estimate looks at. Null follows
    /// <see cref="MinimumClusterSize"/>, which is the usual choice.
    /// <para>
    /// This is the conservativeness dial, held separately because it does
    /// something different: it sets the core distance, so raising it declares
    /// more of the sparse ground to be noise without changing what counts as a
    /// cluster once found.
    /// </para>
    /// </summary>
    public int? MinimumSamples { get; init; }

    /// <summary>
    /// A default <see cref="MinimumClusterSize"/> for <paramref name="sampleCount"/>
    /// samples, for callers with no opinion.
    /// <para>
    /// Two per cent of the data, floored at five and capped at twenty-five. The
    /// floor is where a group stops being distinguishable from a coincidence;
    /// the cap stops a large model demanding implausibly large families before it
    /// will admit any.
    /// </para>
    /// </summary>
    public static int DefaultMinimumClusterSize(int sampleCount)
        => Math.Clamp((int)Math.Round(sampleCount * 0.02), 5, 25);

    internal void Validate(int sampleCount)
    {
        if (MinimumClusterSize < 2)
            throw new ArgumentOutOfRangeException(nameof(MinimumClusterSize), MinimumClusterSize,
                "A cluster needs at least two points.");
        if (MinimumClusterSize > sampleCount)
            throw new ArgumentOutOfRangeException(nameof(MinimumClusterSize), MinimumClusterSize,
                $"Minimum cluster size {MinimumClusterSize} exceeds the {sampleCount} samples given.");
        if (MinimumSamples is { } samples)
        {
            if (samples < 1)
                throw new ArgumentOutOfRangeException(nameof(MinimumSamples), samples,
                    "Need at least one neighbour.");
            if (samples > sampleCount)
                throw new ArgumentOutOfRangeException(nameof(MinimumSamples), samples,
                    $"Minimum samples {samples} exceeds the {sampleCount} samples given.");
        }
    }

    internal int EffectiveMinimumSamples => MinimumSamples ?? MinimumClusterSize;
}
