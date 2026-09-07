using OtterLogic.MachineLearning.Clustering;
using Xunit;
using Xunit.Abstractions;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// EM against scikit-learn, from an identical starting point.
/// <para>
/// Pinning the initialisation is what makes this a test of arithmetic. Left to
/// initialise themselves the two implementations would climb different hills and
/// a mismatch would prove nothing; started from the same parameters they should
/// agree to something near machine precision, and any drift is a real
/// difference in how the update is computed.
/// </para>
/// </summary>
public sealed class GaussianMixtureTests
{
    private readonly ITestOutputHelper _output;

    public GaussianMixtureTests(ITestOutputHelper output) => _output = output;

    [Theory]
    [InlineData("em_diag", CovarianceType.Diagonal)]
    [InlineData("em_full", CovarianceType.Full)]
    [InlineData("em_spherical", CovarianceType.Spherical)]
    public void MatchesScikitLearnFromTheSameStart(string name, CovarianceType covariance)
    {
        var fixture = Fixture.Load(name);
        var x = fixture.Matrix("x");
        var init = fixture.Section("init");
        var expected = fixture.Section("expected");

        var options = new GaussianMixtureOptions
        {
            Components = fixture.Int("components"),
            Covariance = covariance,
            RegularisationFloor = fixture.Scalar("reg_covar"),
            Tolerance = fixture.Scalar("tolerance"),
            MaxIterations = fixture.Int("max_iterations"),
        };

        var result = GaussianMixture.FitFrom(
            x, options, init.Vector("weights"), init.Matrix("means"), init.Cube("covariances"));

        _output.WriteLine(
            $"{name}: {result.Iterations} iterations (sklearn {expected.Int("iterations")}), "
            + $"BIC {result.Bic:F6} (sklearn {expected.Scalar("bic"):F6})");

        Assert.Equal(expected.Int("parameter_count"), result.ParameterCount);
        Assert.Equal(expected.Flag("converged"), result.Converged);
        Assert.Equal(expected.Int("iterations"), result.Iterations);

        Numeric.Close(expected.Vector("weights"), result.MixingWeights, 1e-9, "mixing weights");
        Numeric.Close(expected.Matrix("means"), result.Means, 1e-8, "means");
        Numeric.Close(expected.Cube("covariances"), result.Covariances, 1e-8, "covariances");
        Numeric.Close(expected.Matrix("responsibilities"), result.Responsibilities, 1e-8, "responsibilities");

        Numeric.Close(expected.Scalar("mean_log_likelihood"), result.MeanLogLikelihood, 1e-10, "mean log-likelihood");
        Numeric.Close(expected.Scalar("bic"), result.Bic, 1e-9, "BIC");
        Numeric.Close(expected.Scalar("aic"), result.Aic, 1e-9, "AIC");
    }

    /// <summary>
    /// Grasshopper re-solves on any upstream change. A component that returns
    /// different groups from identical inputs is unusable, so the seed has to be
    /// the only source of randomness.
    /// </summary>
    [Fact]
    public void IsDeterministicAcrossRepeatedFits()
    {
        var x = Fixture.Load("em_diag").Matrix("x");
        var options = new GaussianMixtureOptions { Components = 4, Seed = 42, Restarts = 5 };

        var first = GaussianMixture.Fit(x, options);
        var second = GaussianMixture.Fit(x, options);

        Numeric.Close(first.LogLikelihood, second.LogLikelihood, 0.0, "log-likelihood");
        Numeric.Close(first.Means, second.Means, 0.0, "means");
        Assert.Equal(first.Labels(), second.Labels());
    }

    /// <summary>
    /// Restarts are the cheapest accuracy on offer, and this is the evidence.
    /// One start is one local optimum; ten keep the best of ten.
    /// </summary>
    [Fact]
    public void RestartsDoNotWorsenTheFit()
    {
        var x = Fixture.Load("em_diag").Matrix("x");

        var once = GaussianMixture.Fit(x, new GaussianMixtureOptions { Components = 6, Restarts = 1, Seed = 3 });
        var often = GaussianMixture.Fit(x, new GaussianMixtureOptions { Components = 6, Restarts = 20, Seed = 3 });

        _output.WriteLine($"1 restart: {once.LogLikelihood:F4}   20 restarts: {often.LogLikelihood:F4}");
        Assert.True(often.LogLikelihood >= once.LogLikelihood - 1e-9,
            $"20 restarts reached {often.LogLikelihood:R}, worse than 1 restart at {once.LogLikelihood:R}");
    }

    /// <summary>
    /// Responsibilities are a posterior over components, so every row is a
    /// distribution. If log-sum-exp were wrong this is what would break first.
    /// </summary>
    [Fact]
    public void ResponsibilitiesFormADistribution()
    {
        var x = Fixture.Load("em_diag").Matrix("x");
        var result = GaussianMixture.Fit(x, new GaussianMixtureOptions { Components = 5, Seed = 11 });

        for (int i = 0; i < result.SampleCount; i++)
        {
            double sum = 0.0;
            for (int c = 0; c < result.ComponentCount; c++)
            {
                Assert.InRange(result.Responsibilities[i, c], 0.0, 1.0);
                sum += result.Responsibilities[i, c];
            }

            Numeric.Close(1.0, sum, 1e-12, $"row {i} sums to one");
        }

        Assert.All(result.Confidence(), c => Assert.InRange(c, 1.0 / result.ComponentCount - 1e-12, 1.0));
    }

    /// <summary>
    /// Duplicate rows are the classic way to blow a mixture up: a component
    /// collapses onto them, its variance heads to zero and the likelihood to
    /// infinity. A structural model is full of identical members, so this is a
    /// realistic input rather than a contrived one — and the regularisation
    /// floor is the only thing standing between it and NaN.
    /// </summary>
    [Fact]
    public void SurvivesHeavilyDuplicatedRows()
    {
        var x = new double[120, 3];
        for (int i = 0; i < 120; i++)
        {
            // Three exactly repeated points, forty copies each.
            int group = i % 3;
            x[i, 0] = group;
            x[i, 1] = group * 2.0;
            x[i, 2] = group == 1 ? 1.0 : 0.0;
        }

        var result = GaussianMixture.Fit(x, new GaussianMixtureOptions { Components = 3, Seed = 5 });

        Assert.False(double.IsNaN(result.LogLikelihood));
        Assert.False(double.IsInfinity(result.LogLikelihood));
        Assert.All(result.Confidence(), c => Assert.False(double.IsNaN(c)));
        Assert.Equal(3, result.Labels().Distinct().Count());
    }
}
