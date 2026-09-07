namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// The shape each mixture component's covariance is allowed to take.
/// <para>
/// This is the bias-variance dial. Parameters per component at d dimensions:
/// spherical 1, diagonal d, full d(d+1)/2. At d = 6 that is 1, 6 and 21 — so a
/// full model with k = 8 fits about 220 parameters, which wants a good deal
/// more than the few hundred members a frame typically has.
/// </para>
/// <para>
/// <see cref="Diagonal"/> is the default because PCA whitening has already
/// decorrelated the axes, which is most of what full covariance buys.
/// </para>
/// </summary>
public enum CovarianceType
{
    /// <summary>
    /// One variance per component, shared across every dimension. Components are
    /// round balls of differing size. The only shape whose result depends on
    /// per-column weighting even without PCA in front of it.
    /// </summary>
    Spherical,

    /// <summary>
    /// One variance per dimension per component. Axis-aligned ellipsoids.
    /// The default, and the right partner for whitened principal components.
    /// </summary>
    Diagonal,

    /// <summary>
    /// A full covariance matrix per component. Ellipsoids at any orientation.
    /// Worth it only when there is genuinely correlated structure left after
    /// whitening, and only when n/k comfortably exceeds d(d+1)/2.
    /// </summary>
    Full,
}
