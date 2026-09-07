namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// The knobs on the EM fit. Defaults are chosen for a few hundred to a few
/// thousand members in six whitened dimensions.
/// </summary>
public sealed record GaussianMixtureOptions
{
    /// <summary>Number of mixture components, k.</summary>
    public int Components { get; init; } = 4;

    /// <summary>Shape each component's covariance may take.</summary>
    public CovarianceType Covariance { get; init; } = CovarianceType.Diagonal;

    /// <summary>Hard cap on EM iterations per restart.</summary>
    public int MaxIterations { get; init; } = 200;

    /// <summary>
    /// Convergence threshold on the change in mean log-likelihood per sample,
    /// matching scikit-learn's criterion.
    /// </summary>
    public double Tolerance { get; init; } = 1e-4;

    /// <summary>
    /// How many times to refit from a different random start, keeping the best
    /// by log-likelihood.
    /// <para>
    /// EM is not convex — it climbs to a local optimum and stops. Restarts are
    /// the cheapest accuracy available here by a wide margin: at these data
    /// sizes ten of them cost milliseconds and routinely find a noticeably
    /// better fit than one.
    /// </para>
    /// </summary>
    public int Restarts { get; init; } = 10;

    /// <summary>
    /// Seed for initialisation. Fixed by default, and it must stay that way:
    /// Grasshopper re-solves constantly, and a component that returns different
    /// groups from identical inputs is unusable.
    /// </summary>
    public int Seed { get; init; } = 1;

    /// <summary>
    /// Added to the diagonal of every covariance estimate.
    /// <para>
    /// Not a nicety. Without a floor a component can collapse onto a single
    /// point, its variance heads to zero and the likelihood to infinity, and the
    /// fit ends in NaN. That failure needs duplicate or near-duplicate rows to
    /// trigger it, and a structural model is full of identical members.
    /// </para>
    /// </summary>
    public double RegularisationFloor { get; init; } = 1e-6;

    internal void Validate(int sampleCount)
    {
        if (Components < 1)
            throw new ArgumentOutOfRangeException(nameof(Components), Components, "Need at least one component.");
        if (Components > sampleCount)
            throw new ArgumentOutOfRangeException(nameof(Components), Components,
                $"Cannot fit {Components} components to {sampleCount} samples.");
        if (MaxIterations < 1)
            throw new ArgumentOutOfRangeException(nameof(MaxIterations), MaxIterations, "Need at least one iteration.");
        if (Restarts < 1)
            throw new ArgumentOutOfRangeException(nameof(Restarts), Restarts, "Need at least one restart.");
        if (Tolerance < 0.0)
            throw new ArgumentOutOfRangeException(nameof(Tolerance), Tolerance, "Tolerance cannot be negative.");
        if (RegularisationFloor < 0.0)
            throw new ArgumentOutOfRangeException(nameof(RegularisationFloor), RegularisationFloor,
                "Regularisation floor cannot be negative.");
    }
}
