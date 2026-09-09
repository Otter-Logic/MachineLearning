using OtterLogic.MachineLearning.Decomposition;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The Jacobi eigendecomposition against scikit-learn's <c>PCA</c>, which uses a
/// full SVD of the centred data. Different algorithms, same answer — which is
/// the point: the decomposition is determined by the data, not by the route
/// taken to it.
/// </summary>
public sealed class PrincipalComponentsTests
{
    [Theory]
    [InlineData("pca_whiten")]
    [InlineData("pca_plain")]
    public void MatchesScikitLearn(string name)
    {
        var fixture = Fixture.Load(name);
        var x = fixture.Matrix("x");
        bool whiten = fixture.Flag("whiten");
        var expected = fixture.Section("expected");

        // Threshold of 1.0 keeps every non-degenerate component, matching
        // PCA(n_components=None).
        var pca = PrincipalComponents.Fit(x, varianceThreshold: 1.0, whiten: whiten);

        Numeric.Close(expected.Vector("mean"), pca.Mean, 1e-10, "mean");
        Numeric.Close(expected.Vector("explained_variance"), pca.ExplainedVariance, 1e-9, "explained variance");
        Numeric.Close(expected.Matrix("components"), pca.Components, 1e-8, "components");
        Numeric.Close(expected.Matrix("transformed"), pca.Transform(x), 1e-8, "transformed");
    }

    [Fact]
    public void InverseTransformRecoversTheInput()
    {
        var fixture = Fixture.Load("pca_whiten");
        var x = fixture.Matrix("x");

        var pca = PrincipalComponents.Fit(x, varianceThreshold: 1.0, whiten: true);
        Numeric.Close(x, pca.InverseTransform(pca.Transform(x)), 1e-9, "round trip");
    }

    /// <summary>
    /// A column of zeros is what a planar frame's out-of-plane degrees of
    /// freedom look like. Retaining it would leave an eigenvalue of zero for
    /// whitening to divide by.
    /// </summary>
    [Fact]
    public void DropsDegenerateDirections()
    {
        var x = new double[40, 3];
        for (int i = 0; i < 40; i++)
        {
            x[i, 0] = i;
            x[i, 1] = 2.0 * i;   // perfectly correlated with column 0
            x[i, 2] = 5.0;       // constant
        }

        var pca = PrincipalComponents.Fit(x, varianceThreshold: 1.0, whiten: true);

        Assert.Equal(1, pca.Count);
        Assert.All(Enumerable.Range(0, pca.Count), c => Assert.True(pca.ExplainedVariance[c] > 0.0));
    }

    /// <summary>
    /// The variance threshold is what lets the retained count follow the data
    /// rather than a number somebody typed.
    /// </summary>
    [Fact]
    public void RetainsFewerComponentsAsTheThresholdDrops()
    {
        var fixture = Fixture.Load("pca_whiten");
        var x = fixture.Matrix("x");

        int all = PrincipalComponents.Fit(x, 1.0, whiten: true).Count;
        int most = PrincipalComponents.Fit(x, 0.99, whiten: true).Count;
        int some = PrincipalComponents.Fit(x, 0.80, whiten: true).Count;

        Assert.True(all >= most, $"1.0 kept {all}, 0.99 kept {most}");
        Assert.True(most >= some, $"0.99 kept {most}, 0.80 kept {some}");
        Assert.True(some >= 1);
    }
}
