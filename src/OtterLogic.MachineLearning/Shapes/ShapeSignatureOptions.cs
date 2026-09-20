namespace OtterLogic.MachineLearning.Shapes;

/// <summary>
/// Settings for <see cref="ShapeSignature"/>: how finely each outline is read,
/// what counts as the same shape, and how many numbers to describe one with.
/// </summary>
public sealed record ShapeSignatureOptions
{
    /// <summary>
    /// How many points each outline is resampled to. Every outline is read at the
    /// same count whatever it was drawn with, which is what makes a four-corner
    /// shape and a forty-corner one comparable at all.
    /// <para>
    /// The cost is quadratic in this — the correspondence search compares every
    /// starting point against every other — so it is capped. Thirty-two carries a
    /// notch or a curve; raising it past a hundred buys detail nothing downstream
    /// can use.
    /// </para>
    /// </summary>
    public int Points { get; init; } = 32;

    /// <summary>
    /// Whether the outlines close back on themselves. Closed outlines have no
    /// natural first point, so one is searched for; open ones keep the order they
    /// were given, and are only ever considered end to end.
    /// </summary>
    public bool Closed { get; init; } = true;

    /// <summary>
    /// Divide each outline by its own size before comparing, so that two shapes of
    /// the same proportions are the same shape.
    /// <para>
    /// Off by default, because the common case is that size is part of what
    /// distinguishes one thing from another. Turn it on to ask the narrower
    /// question — what proportions are there — and read
    /// <see cref="ShapeSignatureResult.Size"/> for the sizes it set aside.
    /// </para>
    /// </summary>
    public bool NormaliseScale { get; init; }

    /// <summary>
    /// Turn each outline to sit as squarely as it can on the others before
    /// comparing, so that a shape and the same shape turned are the same shape.
    /// <para>
    /// Off by default: outlines usually arrive already in a frame that means
    /// something, and turning them throws that away. Only available in two
    /// dimensions, because the turn is a single angle there and an orientation
    /// search in three needs machinery this has no use for yet.
    /// </para>
    /// </summary>
    public bool NormaliseRotation { get; init; }

    /// <summary>
    /// How many equally spaced turns of an outline count as the same shape.
    /// <para>
    /// One, the default, means a turned shape is a different shape. <b>Two</b> is the
    /// useful one: a shape and the same shape turned half way round are the same,
    /// which is what a flat thing with no up and no down is — turn it round and it
    /// is the part you already had. Four adds the quarter turns, for something that
    /// can be used on either axis.
    /// </para>
    /// <para>
    /// Unlike <see cref="NormaliseRotation"/> this does not let a shape be turned to
    /// <em>any</em> angle, so an outline stays squarely in the frame it arrived in.
    /// Two dimensions only, for the same reason.
    /// </para>
    /// </summary>
    public int Turns { get; init; } = 1;

    /// <summary>
    /// Whether an outline and its mirror image count as the same shape.
    /// <para>
    /// Off by default, and deliberately: a thing and its mirror are usually two
    /// things to make, not one. Turn it on when only the geometry matters and
    /// which way round it faces does not.
    /// </para>
    /// </summary>
    public bool AllowReflection { get; init; }

    /// <summary>
    /// How many numbers to describe a shape with. Zero, the default, lets
    /// <see cref="Variance"/> decide — which is usually what you want, since how
    /// many a population needs is itself a finding.
    /// </summary>
    public int Components { get; init; }

    /// <summary>
    /// When <see cref="Components"/> is zero, keep the fewest directions carrying
    /// this much of the variation between the shapes.
    /// </summary>
    public double Variance { get; init; } = 0.99;

    /// <summary>
    /// How many times to align every outline to the running mean and recompute it.
    /// The mean usually stops moving after two or three; the cap is there so a
    /// population that will not settle still returns.
    /// </summary>
    public int Rounds { get; init; } = 8;

    /// <summary>
    /// When the mean shape moves less than this between rounds, it has settled.
    /// In the same units as the outlines.
    /// </summary>
    public double Tolerance { get; init; } = 1e-9;

    /// <summary>The most points an outline may be read at, and why is on <see cref="Points"/>.</summary>
    public const int MostPoints = 128;

    /// <summary>Checks these settings.</summary>
    /// <param name="dimensions">How many coordinates each point has, for the checks that depend on it.</param>
    public void Validate(int dimensions = 2)
    {
        int fewest = Closed ? 3 : 2;
        if (Points < fewest || Points > MostPoints)
            throw new ArgumentOutOfRangeException(nameof(Points), Points,
                $"Read each outline at between {fewest} and {MostPoints} points.");

        if (NormaliseRotation && dimensions != 2)
            throw new ArgumentOutOfRangeException(nameof(NormaliseRotation), NormaliseRotation,
                $"Turning an outline square is only available in two dimensions, not {dimensions}.");

        if (Turns < 1)
            throw new ArgumentOutOfRangeException(nameof(Turns), Turns,
                "A shape counts as itself at least once; one turn means a turned shape is a different shape.");

        if (Turns > 1 && dimensions != 2)
            throw new ArgumentOutOfRangeException(nameof(Turns), Turns,
                $"Turning an outline is only available in two dimensions, not {dimensions}.");

        if (Components < 0)
            throw new ArgumentOutOfRangeException(nameof(Components), Components,
                "Describe a shape with no fewer than no numbers; zero lets the variance decide.");

        if (!double.IsFinite(Variance) || Variance <= 0.0 || Variance > 1.0)
            throw new ArgumentOutOfRangeException(nameof(Variance), Variance,
                "The variance to keep runs above zero and up to one.");

        if (Rounds < 1)
            throw new ArgumentOutOfRangeException(nameof(Rounds), Rounds, "Align at least once.");

        if (!double.IsFinite(Tolerance) || Tolerance <= 0.0)
            throw new ArgumentOutOfRangeException(nameof(Tolerance), Tolerance, "The tolerance must be above zero.");
    }
}
