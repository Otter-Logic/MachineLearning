using OtterLogic.MachineLearning.Decomposition;

namespace OtterLogic.MachineLearning.Shapes;

/// <summary>
/// What a population of outlines turned out to be: a row of numbers describing
/// each one, the average shape they vary around, and the ways they vary.
/// </summary>
public sealed class ShapeSignatureResult
{
    private readonly PrincipalComponents _components;
    private readonly int _points;

    internal ShapeSignatureResult(
        PrincipalComponents components, double[,] scores, double[] size, double[] residual,
        int points, int dimensions, IReadOnlyList<string> notes)
    {
        _components = components;
        _points = points;
        Scores = scores;
        Size = size;
        Residual = residual;
        Dimensions = dimensions;
        Notes = notes;
    }

    /// <summary>
    /// One row per outline: where it sits in the shape space, as many numbers as
    /// <see cref="Count"/>.
    /// <para>
    /// These are in the units the outlines came in, and scaled so that the
    /// straight-line distance between two rows is the <b>root-mean-square
    /// distance between the two outlines, point for corresponding point</b>. That
    /// is what makes a tolerance on this signature a tolerance somebody can
    /// justify: cut a clustering of these at 5 and no two members are more than 5
    /// apart, averaged around their outlines. <see cref="Residual"/> says how much
    /// of that distance the retained components did not carry.
    /// </para>
    /// </summary>
    public double[,] Scores { get; }

    /// <summary>Per outline, its size — the root-mean-square distance of its points from its own centre.</summary>
    public double[] Size { get; }

    /// <summary>
    /// Per outline, how far it sits from its own description: the root-mean-square
    /// distance between it and the shape <see cref="Reconstruct"/> builds from its
    /// row. Zero when every component was kept.
    /// </summary>
    public double[] Residual { get; }

    /// <summary>How many coordinates each point has.</summary>
    public int Dimensions { get; }

    /// <summary>Anything that changed what was found, in plain words.</summary>
    public IReadOnlyList<string> Notes { get; }

    /// <summary>How many numbers describe one shape.</summary>
    public int Count => _components.Count;

    /// <summary>How much of the variation between the outlines the retained components carry, 0 to 1.</summary>
    public double Carried => _components.ExplainedVarianceRatio;

    /// <summary>
    /// How much of the variation sits along each retained component, largest
    /// first, as a distance in the units the outlines came in. The first is what
    /// the population mostly differs by.
    /// </summary>
    public double[] Spread => _components.ExplainedVariance.Select(Math.Sqrt).ToArray();

    /// <summary>
    /// The average outline the population varies around, <see cref="Points"/> x
    /// <see cref="Dimensions"/>, centred on the origin.
    /// </summary>
    public double[,] MeanShape => Reshape(_components.Mean);

    /// <summary>How many points each outline was read at.</summary>
    public int Points => _points;

    /// <summary>
    /// What one component <em>looks like</em>: the mean shape moved along it by
    /// <paramref name="amount"/> of its own spread, as an outline you can draw.
    /// <para>
    /// This is how a component gets named. Drawing the mean at -2, 0 and +2 shows
    /// what the first component is — one population's is "how raked", another's is
    /// "how deep the notch" — and nothing else in the result will tell you that.
    /// </para>
    /// </summary>
    /// <param name="component">Which component, from zero.</param>
    /// <param name="amount">How many of its spreads to move by; negative goes the other way.</param>
    public double[,] Variation(int component, double amount)
    {
        if (component < 0 || component >= Count)
            throw new ArgumentOutOfRangeException(nameof(component), component,
                $"There are {Count} component(s) to look along.");

        var scores = new double[Count];
        scores[component] = amount * Spread[component];
        return Reconstruct(scores);
    }

    /// <summary>
    /// The outline a row of numbers describes, <see cref="Points"/> x
    /// <see cref="Dimensions"/>, centred on the origin. The way back out of the
    /// shape space, and what makes it a space to search in rather than only a
    /// description.
    /// </summary>
    public double[,] Reconstruct(double[] scores)
    {
        ArgumentNullException.ThrowIfNull(scores);
        if (scores.Length != Count)
            throw new ArgumentException($"Need {Count} number(s) to describe a shape, not {scores.Length}.", nameof(scores));

        var row = new double[1, Count];
        for (int c = 0; c < Count; c++)
            row[0, c] = scores[c];

        var flat = _components.InverseTransform(row);
        var one = new double[flat.GetLength(1)];
        for (int j = 0; j < one.Length; j++)
            one[j] = flat[0, j];

        return Reshape(one);
    }

    /// <summary>The outline one sample was described as, which is <see cref="Reconstruct"/> of its own row.</summary>
    public double[,] ShapeOf(int sample)
    {
        if (sample < 0 || sample >= Scores.GetLength(0))
            throw new ArgumentOutOfRangeException(nameof(sample), sample, "No such outline.");

        var scores = new double[Count];
        for (int c = 0; c < Count; c++)
            scores[c] = Scores[sample, c];

        return Reconstruct(scores);
    }

    /// <summary>
    /// A flat vector of coordinates back into points. The stored vector is divided
    /// by the square root of the point count, which is what puts
    /// <see cref="Scores"/> in root-mean-square units, so it is multiplied back
    /// here.
    /// </summary>
    private double[,] Reshape(double[] flat)
    {
        double back = Math.Sqrt(_points);
        var shape = new double[_points, Dimensions];
        for (int p = 0; p < _points; p++)
            for (int a = 0; a < Dimensions; a++)
                shape[p, a] = flat[p * Dimensions + a] * back;

        return shape;
    }
}
