namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// What <see cref="EmbedRun"/> does around the method: the preparation before it
/// and the size of the map it asks for.
/// </summary>
public sealed record EmbedRunOptions
{
    /// <summary>
    /// Bring every column to the same scale before mapping. On by default, for
    /// the reason OtterCluster gives: every method here measures distances, and a
    /// length in millimetres beside an angle in radians is a map that has only
    /// looked at the length. Off is for columns that already share a scale that
    /// means something. Ignored when only a graph was wired.
    /// </summary>
    public bool Standardise { get; init; } = true;

    /// <summary>
    /// Coordinates per sample. Two by default, to look at; three to look at in
    /// space; more is a legitimate request for a later step to read.
    /// </summary>
    public int Dimensions { get; init; } = 2;

    public void Validate()
    {
        if (Dimensions < 1)
            throw new ArgumentOutOfRangeException(nameof(Dimensions), Dimensions, "A map needs at least one dimension.");
    }
}
