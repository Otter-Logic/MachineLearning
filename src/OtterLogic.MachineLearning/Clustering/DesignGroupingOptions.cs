namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// Every control on the grouping pipeline, in the order the pipeline applies
/// them: row normalisation, log transform, standardisation, weights, PCA, then
/// the mixture.
/// </summary>
public sealed record DesignGroupingOptions
{
    /// <summary>Number of groups, k.</summary>
    public int Groups { get; init; } = 4;

    /// <summary>Apply log(1 + x) first. On by default, because magnitudes are right-skewed.</summary>
    public bool LogTransform { get; init; } = true;

    /// <summary>
    /// Scale each member to unit length first, keeping only the proportion
    /// between degrees of freedom. Off by default.
    /// <para>
    /// Turning this on usually means turning <see cref="LogTransform"/> off:
    /// once every row is unit length the values are bounded proportions, and the
    /// skew the log exists to tame has already gone.
    /// </para>
    /// </summary>
    public bool NormaliseRows { get; init; }

    /// <summary>Per-column multipliers, or null for equal weighting. Zero drops a column.</summary>
    public double[]? Weights { get; init; }

    /// <summary>
    /// Fraction of variance the retained principal components must cover.
    /// Set to zero to skip PCA entirely — which also makes
    /// <see cref="Weights"/> largely inert for anything but a spherical fit.
    /// </summary>
    public double PcaVariance { get; init; } = 0.99;

    /// <summary>Scale the principal components to unit variance. Normally yes.</summary>
    public bool Whiten { get; init; } = true;

    /// <summary>Shape of each component's covariance.</summary>
    public CovarianceType Covariance { get; init; } = CovarianceType.Diagonal;

    /// <summary>Restarts of the EM fit; the best log-likelihood wins.</summary>
    public int Restarts { get; init; } = 10;

    /// <summary>Iteration cap per restart.</summary>
    public int MaxIterations { get; init; } = 200;

    /// <summary>Convergence tolerance on the mean log-likelihood.</summary>
    public double Tolerance { get; init; } = 1e-4;

    /// <summary>Seed. Fixed, so a Grasshopper re-solve returns the same groups.</summary>
    public int Seed { get; init; } = 1;

    /// <summary>Floor added to every covariance diagonal.</summary>
    public double RegularisationFloor { get; init; } = 1e-6;

    internal GaussianMixtureOptions ToMixtureOptions() => new()
    {
        Components = Groups,
        Covariance = Covariance,
        Restarts = Restarts,
        MaxIterations = MaxIterations,
        Tolerance = Tolerance,
        Seed = Seed,
        RegularisationFloor = RegularisationFloor,
    };
}
