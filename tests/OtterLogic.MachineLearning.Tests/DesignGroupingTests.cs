using OtterLogic.MachineLearning.Clustering;
using Xunit;
using Xunit.Abstractions;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The whole pipeline, against scikit-learn doing the same job its own way.
/// <para>
/// Unlike <see cref="GaussianMixtureTests"/> this does not pin the
/// initialisation, so the two are not expected to match parameter for
/// parameter — they are two non-convex optimisers each keeping the best of ten
/// starts. What is expected is that they reach an equally good fit, measured by
/// BIC, and agree about which member belongs with which, measured by the
/// adjusted Rand index.
/// </para>
/// </summary>
public sealed class DesignGroupingTests
{
    private readonly ITestOutputHelper _output;

    public DesignGroupingTests(ITestOutputHelper output) => _output = output;

    private static DesignGroupingOptions Matching(int groups) => new()
    {
        Groups = groups,
        LogTransform = true,
        NormaliseRows = false,
        PcaVariance = 0.99,
        Whiten = true,
        Covariance = CovarianceType.Diagonal,
        Restarts = 10,
        MaxIterations = 200,
        Tolerance = 1e-4,
        RegularisationFloor = 1e-6,
        Seed = 1,
    };

    [Fact]
    public void ReachesTheSameQualityOfFitAsScikitLearn()
    {
        var fixture = Fixture.Load("quality");
        var raw = fixture.Matrix("raw");
        var truth = fixture.Integers("true_family");
        var expected = fixture.Section("expected");

        var result = DesignGrouping.Group(raw, Matching(fixture.Int("components")));

        double theirBic = expected.Scalar("bic");
        double ourBic = result.Mixture.Bic;
        double agreement = Numeric.AdjustedRandIndex(expected.Integers("labels"), result.Labels);
        double theirTruth = Numeric.AdjustedRandIndex(truth, expected.Integers("labels"));
        double ourTruth = Numeric.AdjustedRandIndex(truth, result.Labels);

        _output.WriteLine("                       sklearn        OtterLogic");
        _output.WriteLine($"BIC                 {theirBic,12:F3}  {ourBic,16:F3}");
        _output.WriteLine($"mean log-likelihood {expected.Scalar("mean_log_likelihood"),12:F4}  "
                          + $"{result.Mixture.MeanLogLikelihood,16:F4}");
        _output.WriteLine($"mean confidence     {expected.Scalar("mean_confidence"),12:F4}  "
                          + $"{result.Confidence.Average(),16:F4}");
        _output.WriteLine($"iterations          {expected.Int("iterations"),12}  {result.Mixture.Iterations,16}");
        _output.WriteLine($"ARI vs true family  {theirTruth,12:F4}  {ourTruth,16:F4}");
        _output.WriteLine($"\nARI between the two implementations: {agreement:F4}");
        _output.WriteLine($"PCA retained {result.PrincipalComponents!.Count} components "
                          + $"(sklearn {expected.Int("retained_components")})");

        Assert.Equal(expected.Int("retained_components"), result.PrincipalComponents.Count);

        // Same criterion, same data, same number of restarts. A meaningfully
        // worse BIC would mean the optimiser is losing ground, not that it found
        // a different equally good answer.
        Assert.True(ourBic <= theirBic * 1.02,
            $"BIC {ourBic:F3} is more than 2% worse than scikit-learn's {theirBic:F3}");

        // Two optimisers on the same non-convex surface will not always agree on
        // boundary members. Broad agreement on the partition is the real test.
        Assert.True(agreement > 0.90,
            $"Adjusted Rand index against scikit-learn is only {agreement:F4}");
    }

    /// <summary>
    /// Components are ordered by descending mixing weight before they leave the
    /// pipeline. Without that, a small upstream change can permute cluster
    /// indices and every downstream colour and geometry assignment jumps.
    /// </summary>
    [Fact]
    public void OrdersGroupsByShareSoLabelsAreStable()
    {
        var raw = Fixture.Load("quality").Matrix("raw");
        var result = DesignGrouping.Group(raw, Matching(4));

        for (int c = 1; c < result.GroupShares.Length; c++)
            Assert.True(result.GroupShares[c] <= result.GroupShares[c - 1] + 1e-12,
                $"group {c} has a larger share than group {c - 1}");

        Numeric.Close(1.0, result.GroupShares.Sum(), 1e-12, "shares sum to one");
    }

    /// <summary>
    /// The centres output is what makes a grouping readable, so it has to come
    /// back in the units the data arrived in — not in whitened principal
    /// component space where the numbers mean nothing to anyone.
    /// </summary>
    [Fact]
    public void ReportsCentresInTheOriginalUnits()
    {
        var fixture = Fixture.Load("quality");
        var raw = fixture.Matrix("raw");
        var result = DesignGrouping.Group(raw, Matching(4));

        int d = raw.GetLength(1);
        var columnMax = new double[d];
        for (int j = 0; j < d; j++)
            for (int i = 0; i < raw.GetLength(0); i++)
                columnMax[j] = Math.Max(columnMax[j], raw[i, j]);

        _output.WriteLine("group   share        " + string.Join("      ", fixture.Section("expected") is null
            ? Array.Empty<string>()
            : new[] { "Fx", "Fy", "Fz", "Mx", "My", "Mz" }));

        for (int c = 0; c < result.Centres.GetLength(0); c++)
        {
            var row = Enumerable.Range(0, d).Select(j => result.Centres[c, j]).ToArray();
            _output.WriteLine($"{c,5}   {result.GroupShares[c]:F3}   "
                              + string.Join("  ", row.Select(v => $"{v,8:F2}")));

            for (int j = 0; j < d; j++)
            {
                Assert.True(row[j] >= 0.0, $"centre[{c},{j}] is negative: {row[j]}");
                Assert.True(row[j] <= columnMax[j],
                    $"centre[{c},{j}] = {row[j]:F3} exceeds the largest observed value {columnMax[j]:F3}");
            }
        }
    }

    /// <summary>
    /// A planar frame has identically zero out-of-plane demand. The pipeline has
    /// to drop those columns rather than hand a singular covariance to EM.
    /// </summary>
    [Fact]
    public void DropsColumnsWithNoVariance()
    {
        var source = Fixture.Load("quality").Matrix("raw");
        int n = source.GetLength(0);

        var planar = new double[n, 6];
        for (int i = 0; i < n; i++)
        {
            planar[i, 0] = source[i, 0];
            planar[i, 1] = source[i, 1];
            planar[i, 2] = 0.0;   // out of plane
            planar[i, 3] = 0.0;   // out of plane
            planar[i, 4] = source[i, 4];
            planar[i, 5] = 0.0;   // out of plane
        }

        var result = DesignGrouping.Group(planar, Matching(3));

        Assert.Equal(new[] { 0, 1, 4 }, result.KeptColumns);
        Assert.Equal(6, result.InputColumnCount);
        Assert.False(double.IsNaN(result.Mixture.LogLikelihood));
        _output.WriteLine(result.Report());
    }

    /// <summary>
    /// A weight of zero excludes a column outright, and unequal weights change
    /// the rotation PCA finds — which is the whole reason weighting is applied
    /// before the decomposition rather than inside the mixture.
    /// </summary>
    [Fact]
    public void WeightsChangeTheGrouping()
    {
        var raw = Fixture.Load("quality").Matrix("raw");

        var even = DesignGrouping.Group(raw, Matching(4));
        var momentLed = DesignGrouping.Group(raw, Matching(4) with
        {
            Weights = new[] { 0.25, 0.25, 0.25, 3.0, 3.0, 3.0 },
        });
        var forcesOnly = DesignGrouping.Group(raw, Matching(4) with
        {
            Weights = new[] { 1.0, 1.0, 1.0, 0.0, 0.0, 0.0 },
        });

        Assert.Equal(new[] { 0, 1, 2 }, forcesOnly.KeptColumns);

        double momentShift = Numeric.AdjustedRandIndex(even.Labels, momentLed.Labels);
        double forceShift = Numeric.AdjustedRandIndex(even.Labels, forcesOnly.Labels);

        _output.WriteLine($"ARI, even vs moment-weighted: {momentShift:F4}");
        _output.WriteLine($"ARI, even vs forces only:     {forceShift:F4}");

        Assert.True(momentShift < 0.999, "moment weighting left the grouping untouched");
        Assert.True(forceShift < 0.999, "dropping the moment columns left the grouping untouched");
    }

    /// <summary>
    /// Row normalisation discards magnitude and keeps only the proportion
    /// between degrees of freedom. It should therefore stop separating members
    /// that differ only in how heavily loaded they are.
    /// </summary>
    [Fact]
    public void RowNormalisationGroupsByShapeRatherThanSize()
    {
        var raw = Fixture.Load("quality").Matrix("raw");

        var bySize = DesignGrouping.Group(raw, Matching(4));
        var byShape = DesignGrouping.Group(raw, Matching(4) with
        {
            NormaliseRows = true,
            LogTransform = false,
        });

        double agreement = Numeric.AdjustedRandIndex(bySize.Labels, byShape.Labels);
        _output.WriteLine($"ARI, size-sensitive vs shape-only: {agreement:F4}");

        Assert.True(agreement < 0.999, "row normalisation left the grouping untouched");
        Assert.All(byShape.Confidence, c => Assert.False(double.IsNaN(c)));
    }

    [Fact]
    public void SweepsGroupCountAndTracksScikitLearn()
    {
        var fixture = Fixture.Load("sweep");
        var raw = fixture.Matrix("raw");

        var ours = DesignGrouping.ChooseGroupCount(
            raw, Matching(4), fixture.Int("minimum"), fixture.Int("maximum"));
        var theirs = fixture.Rows("expected").ToArray();

        Assert.Equal(theirs.Length, ours.Count);

        _output.WriteLine("  k        sklearn BIC     OtterLogic BIC   mean confidence");
        for (int i = 0; i < ours.Count; i++)
        {
            double theirBic = theirs[i].Scalar("bic");
            _output.WriteLine($"{ours[i].Groups,3}  {theirBic,17:F3}  {ours[i].Bic,17:F3}"
                              + $"  {ours[i].MeanConfidence,16:F4}");

            Assert.Equal(theirs[i].Int("groups"), ours[i].Groups);
            Assert.True(ours[i].Bic <= theirBic * 1.02,
                $"at k={ours[i].Groups} our BIC {ours[i].Bic:F3} is more than 2% worse than {theirBic:F3}");
        }

        int ourBest = ours.OrderBy(c => c.Bic).First().Groups;
        int theirBest = theirs.OrderBy(c => c.Scalar("bic")).First().Int("groups");
        _output.WriteLine($"\nlowest BIC at k={ourBest} (sklearn k={theirBest})");
    }
}
