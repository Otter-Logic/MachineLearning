namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// The method chosen when none is wired, from what was.
/// <para>
/// A graph wired is a user saying the connections are what matter, so the map
/// follows them. Values alone get a scaling: it keeps the distances between the
/// samples as they are, reports how far it had to bend them, and is what a person
/// means by "lay these out so I can see them". Principal components is never
/// chosen, because its axes are the point of it, and a user who wants named axes
/// will wire it.
/// </para>
/// </summary>
public sealed record AutoEmbeddingMethod : EmbeddingMethod
{
    public override string Name => "Auto";

    public override string Describe() => "Chosen from what is wired";

    public override void Validate(int sampleCount, int dimensions)
    {
        if (dimensions < 1)
            throw new ArgumentOutOfRangeException(nameof(dimensions), dimensions, "A map needs at least one dimension.");
    }

    public override EmbeddingOutcome Fit(EmbeddingQuery query, int dimensions)
    {
        ArgumentNullException.ThrowIfNull(query);

        if (query.HasGraph)
        {
            var spectral = new SpectralEmbeddingMethod();
            spectral.Validate(query.SampleCount, dimensions);
            return spectral.Fit(query, dimensions)
                .WithRationale("a Graph is wired, and the map should follow its connections");
        }

        var scaling = new MultidimensionalScalingMethod();
        scaling.Validate(query.SampleCount, dimensions);
        return scaling.Fit(query, dimensions)
            .WithRationale("the samples are values with no graph, and a map that keeps the distances between them is the one to look at first");
    }
}
