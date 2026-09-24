namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// A way of laying samples out as points, with its settings chosen, ready to be
/// handed samples or a graph.
/// <para>
/// The mechanism behind one wire, cut the way <c>ClusteringMethod</c> and
/// <c>GraphMethod</c> are: a Grasshopper user picks a method component — Principal
/// Components, Spectral Embedding — sets its one setting if it has one, and wires
/// the result into the one component that holds the data. The method knows nothing
/// about where the samples come from; the data component knows nothing about how
/// any method works. Adding a method is one record here and one small component
/// there, and the data component never changes.
/// </para>
/// <para>
/// Here rather than in a paradigm repo because every method on this wire is a
/// decomposition, and a decomposition learns nothing: it lives beside
/// <c>PrincipalComponents</c> and <c>MultidimensionalScaling</c>, which the
/// paradigms above already share.
/// </para>
/// </summary>
public abstract record EmbeddingMethod
{
    /// <summary>The method's name as a user knows it: "Principal Components".</summary>
    public abstract string Name { get; }

    /// <summary>The name and the settings that matter, in one line for a report.</summary>
    public abstract string Describe();

    /// <summary>
    /// Complains about anything the fit would refuse, before any data is prepared for it.
    /// </summary>
    /// <exception cref="ArgumentOutOfRangeException">A setting is outside what the method accepts.</exception>
    /// <exception cref="ArgumentException">The method cannot run on this many samples in that many dimensions.</exception>
    public abstract void Validate(int sampleCount, int dimensions);

    /// <summary>
    /// Lays the samples out in <paramref name="dimensions"/> coordinates and says
    /// what the method can honestly say about the map.
    /// </summary>
    /// <param name="query">The values, the graph, or both. Values are expected on a common scale already; <see cref="EmbedRun"/> standardises before calling this.</param>
    /// <param name="dimensions">Coordinates per sample.</param>
    /// <exception cref="ArgumentException">The query lacks what this method needs; the message says what to wire.</exception>
    public abstract EmbeddingOutcome Fit(EmbeddingQuery query, int dimensions);

    public sealed override string ToString() => Describe();

    /// <summary>"3 samples" or "1 sample".</summary>
    protected static string Count(int n, string noun) => $"{n} {noun}{(n == 1 ? "" : "s")}";
}
