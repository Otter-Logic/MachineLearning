namespace OtterLogic.MachineLearning.Decomposition;

/// <summary>
/// Settings for <see cref="MultidimensionalScaling"/>. Every one has a default.
/// </summary>
public sealed record MultidimensionalScalingOptions
{
    /// <summary>
    /// How many coordinates each sample gets. Two or three to look at; more is a
    /// legitimate request here — an embedding for a later step — even though a
    /// front-end that draws the result will want to refuse it.
    /// </summary>
    public int Dimensions { get; init; } = 2;

    /// <summary>
    /// Improve the classical map by stress majorisation (SMACOF), starting from it.
    /// <para>
    /// On by default. Classical scaling fits the dot products the distances imply,
    /// which is exact when the distances are those of real points and a compromise
    /// weighted toward the largest distances when they are not; refinement fits the
    /// distances themselves. It never makes the map worse — every step of the
    /// majorisation lowers the stress or leaves it — and starting from the classical
    /// map rather than a random one keeps the answer the same on every solve.
    /// </para>
    /// </summary>
    public bool Refine { get; init; } = true;

    /// <summary>Most refinement steps. Each costs one pass over every pair of samples.</summary>
    public int MaximumIterations { get; init; } = 300;

    /// <summary>
    /// Refinement stops once a step lowers the squared stress by less than this
    /// share of it.
    /// <para>
    /// Majorisation takes most of its gain in the first few dozen steps and then
    /// crawls: at a millionth, maps of 250 to 2,000 samples all ran to the 300-step
    /// cap for a stress that had long stopped visibly changing. A ten-thousandth
    /// stops in the flat tail rather than chasing it.
    /// </para>
    /// </summary>
    public double Tolerance { get; init; } = 1e-4;

    /// <summary>Seed for the eigensolver's starting block. Fixed, so a re-solve gives the same map.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Checks these settings against the number of samples they will map.</summary>
    public void Validate(int sampleCount)
    {
        if (Dimensions < 1)
            throw new ArgumentOutOfRangeException(nameof(Dimensions), Dimensions, "A map needs at least one dimension.");
        if (Dimensions >= sampleCount)
            throw new ArgumentOutOfRangeException(nameof(Dimensions), Dimensions,
                $"{sampleCount} samples span at most {sampleCount - 1} dimension(s), so a {Dimensions}-dimensional "
                + "map has axes with nothing to show. Map more samples, or ask for fewer dimensions.");
        if (MaximumIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(MaximumIterations), MaximumIterations, "Need at least one iteration.");
        if (!(Tolerance > 0.0))
            throw new ArgumentOutOfRangeException(nameof(Tolerance), Tolerance, "Tolerance must be positive.");
    }
}
