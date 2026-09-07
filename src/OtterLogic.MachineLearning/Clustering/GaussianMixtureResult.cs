namespace OtterLogic.MachineLearning.Clustering;

/// <summary>
/// A fitted mixture: the parameters, the soft assignments, and enough
/// diagnostics to judge whether the fit is worth believing.
/// </summary>
public sealed class GaussianMixtureResult
{
    internal GaussianMixtureResult(
        double[] mixingWeights,
        double[,] means,
        double[][,] covariances,
        double[,] responsibilities,
        double logLikelihood,
        int iterations,
        bool converged,
        int parameterCount)
    {
        MixingWeights = mixingWeights;
        Means = means;
        Covariances = covariances;
        Responsibilities = responsibilities;
        LogLikelihood = logLikelihood;
        Iterations = iterations;
        Converged = converged;
        ParameterCount = parameterCount;
    }

    /// <summary>Mixing proportion of each component, summing to one.</summary>
    public double[] MixingWeights { get; }

    /// <summary>Component means, k x d, in whatever space the fit was run in.</summary>
    public double[,] Means { get; }

    /// <summary>
    /// Component covariances, k matrices of d x d. Diagonal and spherical fits
    /// return diagonal matrices rather than a compressed form, so callers do not
    /// have to branch on <see cref="CovarianceType"/>.
    /// </summary>
    public double[][,] Covariances { get; }

    /// <summary>
    /// Soft assignments, n x k. Row i sums to one and holds the posterior
    /// probability that sample i came from each component. This is the output
    /// that makes a mixture worth more than k-means.
    /// </summary>
    public double[,] Responsibilities { get; }

    /// <summary>Total log-likelihood of the data under the fitted model.</summary>
    public double LogLikelihood { get; }

    /// <summary>Mean log-likelihood per sample — scikit-learn's <c>score</c>.</summary>
    public double MeanLogLikelihood => LogLikelihood / SampleCount;

    /// <summary>EM iterations taken by the winning restart.</summary>
    public int Iterations { get; }

    /// <summary>Whether the winning restart met the tolerance before the iteration cap.</summary>
    public bool Converged { get; }

    /// <summary>Free parameters in the model, used by <see cref="Bic"/> and <see cref="Aic"/>.</summary>
    public int ParameterCount { get; }

    /// <summary>Number of samples fitted.</summary>
    public int SampleCount => Responsibilities.GetLength(0);

    /// <summary>Number of components.</summary>
    public int ComponentCount => MixingWeights.Length;

    /// <summary>
    /// Bayesian information criterion, lower is better. Penalises parameters by
    /// ln(n), so it prefers fewer components than AIC does — which is usually
    /// what you want when the components are meant to become real connection
    /// families somebody has to detail.
    /// </summary>
    public double Bic => -2.0 * LogLikelihood + ParameterCount * Math.Log(SampleCount);

    /// <summary>Akaike information criterion, lower is better.</summary>
    public double Aic => -2.0 * LogLikelihood + 2.0 * ParameterCount;

    /// <summary>Hard assignment of each sample to its most probable component.</summary>
    public int[] Labels() => Posterior.ArgMax(Responsibilities);

    /// <summary>
    /// The highest responsibility for each sample, between 1/k and 1.
    /// <para>
    /// Read this as confidence. A member sitting at 0.95 belongs where it was
    /// put; one at 0.4 is on a boundary between two families and is exactly the
    /// case an engineer should look at rather than take on trust.
    /// </para>
    /// </summary>
    public double[] Confidence() => Posterior.RowMax(Responsibilities);
}
