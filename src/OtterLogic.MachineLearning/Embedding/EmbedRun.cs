using OtterLogic.Core;
using OtterLogic.MachineLearning.Preprocessing;

namespace OtterLogic.MachineLearning.Embedding;

/// <summary>
/// Samples and a method in, a map a person can look at out: standardise, lay
/// out, then read the result back over the columns the samples arrived with.
/// <para>
/// The one entry point the data component calls, so every decision between a
/// tree of numbers and a scatter of points lives here rather than in an adaptor:
/// which method runs when none was chosen, which columns were dropped, how the
/// axes relate to the columns a user named. Any method goes through the same
/// call, and a toolkit that wants exactly what the component does calls this and
/// gets it.
/// </para>
/// </summary>
public static class EmbedRun
{
    /// <summary>
    /// Lays out <paramref name="query"/> with <paramref name="method"/>, or chooses a
    /// method from what the query carries when none is given.
    /// </summary>
    /// <param name="query">The values, the graph, or both, in whatever units the values have.</param>
    /// <param name="method">The method, or null to choose one from what is wired, with the reason in the outcome.</param>
    /// <param name="options">What to do around the method; null for the defaults.</param>
    /// <exception cref="ArgumentException">The query or the method cannot be used as given; the message says why.</exception>
    public static EmbedRunResult Fit(EmbeddingQuery query, EmbeddingMethod? method = null, EmbedRunOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(query);
        method ??= new AutoEmbeddingMethod();
        options ??= new EmbedRunOptions();
        options.Validate();

        int n = query.SampleCount;
        if (options.Dimensions >= n)
            throw new ArgumentOutOfRangeException(nameof(options), options.Dimensions,
                $"{n} samples span at most {n - 1} dimension(s), so a {options.Dimensions}-dimensional map has axes with "
                + "nothing to show. Map more samples, or ask for fewer dimensions.");

        // Before any data is prepared, so a bad setting costs nothing but the message.
        method.Validate(n, options.Dimensions);

        var notes = new List<Note>();
        var fitted = query;
        int[] kept = query.Data is { } data ? Enumerable.Range(0, data.GetLength(1)).ToArray() : Array.Empty<int>();
        int width = kept.Length;

        if (query.Data is { } raw && options.Standardise)
        {
            FeaturePipeline pipeline;
            try
            {
                pipeline = FeaturePipeline.Fit(raw);
            }
            catch (InvalidOperationException)
            {
                throw new ArgumentException(
                    "Every column has the same value in every sample, so there is nothing to lay out.", nameof(query));
            }

            kept = pipeline.KeptColumns;
            if (kept.Length < width)
            {
                var dropped = Enumerable.Range(0, width).Except(kept).Select(j => j.ToString());
                notes.Add(Note.Remark(
                    $"Column(s) {string.Join(", ", dropped)} never change across the samples and were left out."));
            }

            fitted = query.WithData(pipeline.Transform(raw));
        }

        var outcome = method.Fit(fitted, options.Dimensions);
        notes.AddRange(outcome.Notes);

        // Axes come back over the columns the method saw; a dropped column gets a
        // zero loading so axis j still lines up with the user's column j.
        if (outcome.Axes is { } axes && kept.Length < width)
        {
            var full = new double[axes.GetLength(0), width];
            for (int c = 0; c < axes.GetLength(0); c++)
                for (int k = 0; k < kept.Length; k++)
                    full[c, kept[k]] = axes[c, k];
            outcome = outcome.WithAxes(full);
        }

        return new EmbedRunResult(outcome, options.Standardise && query.HasData, kept, notes);
    }
}
