using OtterLogic.MachineLearning.Decomposition;
using OtterLogic.MachineLearning.Graphs;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The leading-eigenvector solver against numpy's dense <c>eigh</c>.
/// <para>
/// Values are compared directly. Vectors are checked by residual and
/// orthonormality instead, because the fixture graph has three components — an
/// eigenvalue of one repeated three times — and inside a repeated eigenvalue any
/// rotation of the eigenvectors is as correct as any other. Comparing them entry
/// by entry would fail on a difference that means nothing.
/// </para>
/// </summary>
public sealed class LeadingEigenTests
{
    private static Func<double[,], double[,]> ShiftedNormalisedAdjacency(WeightedGraph graph) => block =>
    {
        var y = graph.Propagate(block, selfWeight: 0.0);
        for (int i = 0; i < y.GetLength(0); i++)
            for (int c = 0; c < y.GetLength(1); c++)
                y[i, c] = 0.5 * (y[i, c] + block[i, c]);

        return y;
    };

    [Fact]
    public void MatchesADenseDecompositionOfAGraphOperator()
    {
        var fixture = Fixture.Load("graph");
        var graph = WeightedGraph.NearestNeighbours(fixture.Matrix("x"), fixture.Int("neighbours"));
        var expected = fixture.Section("expected").Vector("leading_values");
        var multiply = ShiftedNormalisedAdjacency(graph);

        var (values, vectors, iterations, converged) =
            LeadingEigen.Solve(graph.NodeCount, multiply, expected.Length, tolerance: 1e-10);

        Assert.True(converged, $"did not converge in {iterations} iterations");
        Numeric.Close(expected, values, 1e-9, "eigenvalues");

        var image = multiply(vectors);
        for (int c = 0; c < values.Length; c++)
        {
            double residual = 0.0;
            for (int i = 0; i < graph.NodeCount; i++)
            {
                double r = image[i, c] - values[c] * vectors[i, c];
                residual += r * r;
            }

            Assert.True(Math.Sqrt(residual) < 1e-8, $"eigenpair {c} has residual {Math.Sqrt(residual):E2}");

            for (int other = 0; other <= c; other++)
            {
                double dot = 0.0;
                for (int i = 0; i < graph.NodeCount; i++)
                    dot += vectors[i, c] * vectors[i, other];

                Numeric.Close(other == c ? 1.0 : 0.0, dot, 1e-10, $"vector {c} . vector {other}");
            }
        }
    }

    /// <summary>
    /// When the block is as wide as the operator, the Rayleigh-Ritz step is an
    /// exact dense decomposition — so a small problem finishes in one iteration.
    /// The matrix is built as <c>Q diag(λ) Q^T</c> from a Householder reflection,
    /// so its eigenpairs are known exactly rather than taken from another solver.
    /// </summary>
    [Fact]
    public void SmallOperatorsAreSolvedExactlyInOneIteration()
    {
        double[] lambda = { 4.0, 3.0, 2.0, 1.0 };
        double[] v = { 1.0, 2.0, 3.0, 4.0 };
        double vv = v.Sum(e => e * e);

        var q = new double[4, 4];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                q[i, j] = (i == j ? 1.0 : 0.0) - 2.0 * v[i] * v[j] / vv;

        var matrix = new double[4, 4];
        for (int i = 0; i < 4; i++)
            for (int j = 0; j < 4; j++)
                for (int c = 0; c < 4; c++)
                    matrix[i, j] += q[i, c] * lambda[c] * q[j, c];

        double[,] Multiply(double[,] block)
        {
            int p = block.GetLength(1);
            var y = new double[4, p];
            for (int i = 0; i < 4; i++)
                for (int j = 0; j < 4; j++)
                    for (int c = 0; c < p; c++)
                        y[i, c] += matrix[i, j] * block[j, c];

            return y;
        }

        var (values, vectors, iterations, converged) = LeadingEigen.Solve(4, Multiply, 2);

        Assert.True(converged);
        Assert.Equal(1, iterations);

        for (int c = 0; c < 2; c++)
        {
            Numeric.Close(lambda[c], values[c], 1e-12, $"value {c}");

            // Column c of Q, under the same sign convention the solver applies.
            int largest = Enumerable.Range(0, 4).OrderByDescending(i => Math.Abs(q[i, c])).First();
            double sign = q[largest, c] < 0.0 ? -1.0 : 1.0;
            for (int i = 0; i < 4; i++)
                Numeric.Close(sign * q[i, c], vectors[i, c], 1e-10, $"vector[{i},{c}]");
        }
    }

    [Fact]
    public void IsDeterministicForAGivenSeed()
    {
        var fixture = Fixture.Load("graph");
        var graph = WeightedGraph.NearestNeighbours(fixture.Matrix("x"), fixture.Int("neighbours"));
        var multiply = ShiftedNormalisedAdjacency(graph);

        var first = LeadingEigen.Solve(graph.NodeCount, multiply, 4, seed: 5);
        var second = LeadingEigen.Solve(graph.NodeCount, multiply, 4, seed: 5);

        Numeric.Close(first.Values, second.Values, 0.0, "values");
        Numeric.Close(first.Vectors, second.Vectors, 0.0, "vectors");
    }
}
