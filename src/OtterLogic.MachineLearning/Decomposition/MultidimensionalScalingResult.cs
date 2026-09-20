namespace OtterLogic.MachineLearning.Decomposition;

/// <summary>
/// A map of samples, and how far it can be trusted.
/// <para>
/// Only distances between points mean anything. The axes have no units of their
/// own and no name — rotate or mirror the map and it says exactly the same thing —
/// which is the difference between this and <see cref="PrincipalComponents"/>,
/// whose axes are made of the input columns.
/// </para>
/// </summary>
public sealed class MultidimensionalScalingResult
{
    internal MultidimensionalScalingResult(
        double[,] coordinates,
        double[] eigenvalues,
        double retained,
        double mostNegativeEigenvalue,
        double classicalStress,
        double stress,
        double[] distortion,
        int iterations,
        bool converged,
        IReadOnlyList<string> notes)
    {
        Coordinates = coordinates;
        Eigenvalues = eigenvalues;
        Retained = retained;
        MostNegativeEigenvalue = mostNegativeEigenvalue;
        ClassicalStress = classicalStress;
        Stress = stress;
        Distortion = distortion;
        Iterations = iterations;
        Converged = converged;
        Notes = notes;
    }

    /// <summary>n x dimensions: each sample's place on the map, in the units of the distances.</summary>
    public double[,] Coordinates { get; }

    /// <summary>Number of samples mapped.</summary>
    public int SampleCount => Coordinates.GetLength(0);

    /// <summary>Number of coordinates per sample.</summary>
    public int Dimensions => Coordinates.GetLength(1);

    /// <summary>
    /// The classical map's eigenvalue for each axis, largest first — how much of
    /// the distances' structure that axis carries. Zero or below means the axis has
    /// nothing to show and its coordinates are zero.
    /// </summary>
    public double[] Eigenvalues { get; }

    /// <summary>
    /// Share of the distances' structure the map's axes carry, 0 to 1: the squared
    /// eigenvalues of the kept axes over the squares of all of them — Mardia's
    /// criterion.
    /// <para>
    /// Squares rather than the eigenvalues themselves because distances that no
    /// flat space can hold produce negative eigenvalues, and a share of a total
    /// that negative terms have reduced can exceed one. The sum of squares is the
    /// Frobenius norm of the centred matrix, exact without finding every
    /// eigenvalue. It weighs strong axes more heavily than the variance share
    /// <see cref="PrincipalComponents.ExplainedVarianceRatio"/> reports, so the two
    /// are not the same number on the same data.
    /// </para>
    /// </summary>
    public double Retained { get; }

    /// <summary>
    /// The smallest eigenvalue of the centred matrix, when the input was a
    /// distance table; NaN when it was features, whose distances are those of real
    /// points by construction. Clearly negative means the distances cannot be
    /// drawn exactly in any number of flat dimensions — route distances on a graph
    /// are the usual cause — and the map is a compromise however many axes it has.
    /// </summary>
    public double MostNegativeEigenvalue { get; }

    /// <summary>Stress of the classical map, before any refinement.</summary>
    public double ClassicalStress { get; }

    /// <summary>
    /// Kruskal's stress-1 of the final map: the root of the summed squared
    /// disagreement between map distances and true ones, over the summed squared
    /// true distances. Zero is a perfect map. It is a share, so it reads the same
    /// whatever units the distances are in.
    /// </summary>
    public double Stress { get; }

    /// <summary>
    /// Per sample, the same measure over that sample's own distances only. High
    /// values are the points whose neighbours on the map are not really their
    /// neighbours — the places not to trust the picture.
    /// </summary>
    public double[] Distortion { get; }

    /// <summary>Refinement steps taken; zero when refinement was off.</summary>
    public int Iterations { get; }

    /// <summary>Whether refinement settled before the iteration cap. True when it was off.</summary>
    public bool Converged { get; }

    /// <summary>Anything a reader of the map should know, in plain words.</summary>
    public IReadOnlyList<string> Notes { get; }
}
