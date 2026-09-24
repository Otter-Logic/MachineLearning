using OtterLogic.Core;

namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// What every embedding method hands back, whatever it is: where each sample
/// landed, and what the method can honestly say about the map.
/// <para>
/// The common ground between three results that are otherwise unalike. Principal
/// components has axes made of the input columns and a share of variance; scaling
/// has a stress and a distortion per sample; a spectral embedding has a spectrum
/// and nodes it could not place. Each reduces to coordinates, axes where the axes
/// mean something, a share retained where one can be said, a distortion per
/// sample where one can be measured, and notes. Anything else goes into
/// <see cref="Details"/> as lines for the report.
/// </para>
/// </summary>
public sealed class EmbeddingOutcome
{
    /// <param name="method">The method's name, or for an automatic choice the name of the one chosen.</param>
    /// <param name="coordinates">n x k, one row per sample.</param>
    /// <param name="axes">
    /// k x d, each axis as a direction through the input columns — or null when the
    /// axes are not made of the columns and have no meaning of their own, as a
    /// scaling's or a spectral embedding's do not.
    /// </param>
    /// <param name="retained">Share of the samples' structure the map carries, 0 to 1, or NaN when the method has no honest number.</param>
    /// <param name="distortion">Per sample, how badly the map places it, or null when the method cannot measure that.</param>
    /// <param name="notes">What a user should hear about this map that the coordinates alone do not say.</param>
    /// <param name="details">Lines for a report. Informational only.</param>
    /// <param name="rationale">Why this method, when it was chosen rather than asked for. Null otherwise.</param>
    public EmbeddingOutcome(
        string method, double[,] coordinates, double[,]? axes, double retained, double[]? distortion,
        IReadOnlyList<Note>? notes = null, IReadOnlyList<string>? details = null, string? rationale = null)
    {
        ArgumentNullException.ThrowIfNull(coordinates);
        if (string.IsNullOrWhiteSpace(method))
            throw new ArgumentException("The method needs a name.", nameof(method));
        if (axes is not null && axes.GetLength(0) != coordinates.GetLength(1))
            throw new ArgumentException(
                $"{axes.GetLength(0)} axes for {coordinates.GetLength(1)} coordinates; give one per coordinate or none.", nameof(axes));
        if (distortion is not null && distortion.Length != coordinates.GetLength(0))
            throw new ArgumentException(
                $"{distortion.Length} distortions for {coordinates.GetLength(0)} samples; give one per sample or none.", nameof(distortion));

        Method = method;
        Coordinates = coordinates;
        Axes = axes;
        Retained = retained;
        Distortion = distortion;
        Notes = notes ?? Array.Empty<Note>();
        Details = details ?? Array.Empty<string>();
        Rationale = rationale;
    }

    /// <summary>The method's name, or for an automatic choice the name of the one chosen.</summary>
    public string Method { get; }

    /// <summary>n x k: each sample's place on the map.</summary>
    public double[,] Coordinates { get; }

    /// <summary>k x d: each axis as a direction through the input columns; null when the axes mean nothing of their own.</summary>
    public double[,]? Axes { get; }

    /// <summary>Share of the samples' structure the map carries, 0 to 1; NaN when the method cannot say.</summary>
    public double Retained { get; }

    /// <summary>Per sample, how badly the map places it; null when the method cannot measure that.</summary>
    public double[]? Distortion { get; }

    /// <summary>What a user should hear about this map that the coordinates alone do not say.</summary>
    public IReadOnlyList<Note> Notes { get; }

    /// <summary>Lines for a report. Informational only.</summary>
    public IReadOnlyList<string> Details { get; }

    /// <summary>Why this method, when it was chosen rather than asked for.</summary>
    public string? Rationale { get; }

    public int SampleCount => Coordinates.GetLength(0);

    public int Dimensions => Coordinates.GetLength(1);

    /// <summary>The same outcome with the reason it was chosen attached.</summary>
    public EmbeddingOutcome WithRationale(string rationale)
        => new(Method, Coordinates, Axes, Retained, Distortion, Notes, Details, rationale);

    /// <summary>The same outcome with its axes replaced — how dropped columns are put back.</summary>
    public EmbeddingOutcome WithAxes(double[,]? axes)
        => new(Method, Coordinates, axes, Retained, Distortion, Notes, Details, Rationale);
}
