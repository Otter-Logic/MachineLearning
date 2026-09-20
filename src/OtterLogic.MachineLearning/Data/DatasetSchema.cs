using System.Text.Json;
using System.Text.Json.Serialization;

namespace OtterLogic.MachineLearning.Data;

/// <summary>
/// The contract a dataset folder holds every file in it to: which columns, in which
/// order, for what, and which version of the code produced them.
/// <para>
/// It is written once as <c>schema.json</c> beside the data and checked on every
/// write after that. A dataset is gathered over months, one model at a time, from
/// definitions that get edited in between — and the failure this exists to stop is
/// the twentieth model landing with its columns in a different order from the first
/// nineteen. That trains without any error at all, on rows where "length" is
/// sometimes an angle.
/// </para>
/// </summary>
public sealed record DatasetSchema
{
    /// <summary>The layout of the file itself, so a later format can tell an old folder from a broken one.</summary>
    public int FormatVersion { get; init; } = CurrentFormatVersion;

    /// <summary>
    /// Whatever the caller uses to version the code that produced the features.
    /// <para>
    /// Compared for equality and nothing else. Rows from two versions of an extractor
    /// are not the same measurement even when every column name matches — a
    /// connectivity count that started including supports, say — so they are refused
    /// into one folder, and a trained model's sidecar carries this so inference can
    /// refuse a mismatch the same way.
    /// </para>
    /// </summary>
    public string ExtractorVersion { get; init; } = "1";

    /// <summary>Every column, in file order.</summary>
    public IReadOnlyList<DatasetColumn> Columns { get; init; } = Array.Empty<DatasetColumn>();

    /// <summary>The only format there has been.</summary>
    public const int CurrentFormatVersion = 1;

    /// <summary>The feature columns, in order — the order a model's inputs are in.</summary>
    [JsonIgnore]
    public IReadOnlyList<DatasetColumn> Features => Columns.Where(c => c.Role == ColumnRole.Feature).ToArray();

    /// <summary>The target columns, in order.</summary>
    [JsonIgnore]
    public IReadOnlyList<DatasetColumn> Targets => Columns.Where(c => c.Role == ColumnRole.Target).ToArray();

    /// <summary>The identifier column, if there is one.</summary>
    [JsonIgnore]
    public DatasetColumn? Id => Columns.FirstOrDefault(c => c.Role == ColumnRole.Id);

    /// <summary>The named column, or a complaint listing the ones there are.</summary>
    public DatasetColumn Column(string name)
        => Columns.FirstOrDefault(c => string.Equals(c.Name, name, StringComparison.OrdinalIgnoreCase))
            ?? throw new ArgumentException(
                $"No column named '{name}'. The columns are: {string.Join(", ", Columns.Select(c => c.Name))}.",
                nameof(name));

    /// <summary>Checks the schema makes sense on its own, before any rows are held against it.</summary>
    public void Validate()
    {
        if (string.IsNullOrWhiteSpace(ExtractorVersion))
            throw new ArgumentException("The extractor version is blank. Any text will do, but it has to be there to be compared.");

        var seen = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var column in Columns)
        {
            if (string.IsNullOrWhiteSpace(column.Name))
                throw new ArgumentException("A column has no name.");

            // Trimmed names only. "length " and "length" are one column to a person
            // and two to a header comparison.
            if (column.Name != column.Name.Trim())
                throw new ArgumentException($"Column '{column.Name}' has a space at the start or end of its name.");

            if (!seen.Add(column.Name))
                throw new ArgumentException(
                    $"Two columns are both called '{column.Name}'. Names are compared without regard to case.");

            if (column.Role == ColumnRole.Feature && column.Kind != ColumnKind.Number)
                throw new ArgumentException(
                    $"Feature '{column.Name}' is a category. Features are numbers: turn a category into one "
                    + "column per class holding 0 or 1, which is what any model would do with it anyway.");

            if (column.Role == ColumnRole.Id && column.Kind != ColumnKind.Category)
                throw new ArgumentException($"Identifier '{column.Name}' must be a category column.");

            if (column.Classes.Count > 0 && column.Kind != ColumnKind.Category)
                throw new ArgumentException($"Column '{column.Name}' is a number but lists classes.");

            if (column.Classes.Distinct(StringComparer.Ordinal).Count() != column.Classes.Count)
                throw new ArgumentException($"Column '{column.Name}' lists the same class twice.");
        }

        if (!Columns.Any(c => c.Role == ColumnRole.Feature))
            throw new ArgumentException("A dataset needs at least one feature column.");

        if (Columns.Count(c => c.Role == ColumnRole.Id) > 1)
            throw new ArgumentException("A dataset has at most one identifier column.");
    }

    /// <summary>
    /// Holds <paramref name="incoming"/> against this schema and returns the schema
    /// the folder should carry afterwards.
    /// <para>
    /// Names, roles, kinds, order and extractor version must all match; the one thing
    /// allowed to differ is the classes, which are merged with this schema's first so
    /// that no existing class index moves.
    /// </para>
    /// </summary>
    /// <exception cref="InvalidOperationException">The two do not describe the same table.</exception>
    public DatasetSchema Merge(DatasetSchema incoming)
    {
        ArgumentNullException.ThrowIfNull(incoming);

        if (!string.Equals(ExtractorVersion, incoming.ExtractorVersion, StringComparison.Ordinal))
            throw new InvalidOperationException(
                $"This dataset was written by extractor version '{ExtractorVersion}' and these rows come from "
                + $"'{incoming.ExtractorVersion}'. Rows from two versions are not the same measurement, so they "
                + "do not share a folder: start a new one, or re-export the earlier models with the new version.");

        if (Columns.Count != incoming.Columns.Count)
            throw new InvalidOperationException(
                $"This dataset has {Columns.Count} columns and these rows have {incoming.Columns.Count}. "
                + $"Expected: {string.Join(", ", Columns.Select(c => c.Name))}.");

        var merged = new DatasetColumn[Columns.Count];
        for (int j = 0; j < Columns.Count; j++)
        {
            var mine = Columns[j];
            var theirs = incoming.Columns[j];

            if (!string.Equals(mine.Name, theirs.Name, StringComparison.OrdinalIgnoreCase))
                throw new InvalidOperationException(
                    $"Column {j} of this dataset is '{mine.Name}' and these rows have '{theirs.Name}' there. "
                    + "Column order is part of the contract — a model is trained on positions, not names.");

            if (mine.Role != theirs.Role || mine.Kind != theirs.Kind)
                throw new InvalidOperationException(
                    $"Column '{mine.Name}' is a {mine.Kind} {mine.Role} in this dataset "
                    + $"and a {theirs.Kind} {theirs.Role} in these rows.");

            merged[j] = mine with
            {
                Classes = mine.Classes.Concat(theirs.Classes.Except(mine.Classes, StringComparer.Ordinal)).ToArray(),
            };
        }

        return this with { Columns = merged };
    }

    /// <summary>The classes in <paramref name="merged"/> that this schema does not have yet, by column.</summary>
    public IReadOnlyList<(string Column, string Class)> ClassesAddedBy(DatasetSchema merged)
    {
        var added = new List<(string, string)>();
        for (int j = 0; j < Columns.Count && j < merged.Columns.Count; j++)
            foreach (var name in merged.Columns[j].Classes.Except(Columns[j].Classes, StringComparer.Ordinal))
                added.Add((Columns[j].Name, name));

        return added;
    }

    private static readonly JsonSerializerOptions Json = new()
    {
        WriteIndented = true,
        PropertyNamingPolicy = JsonNamingPolicy.CamelCase,
        DefaultIgnoreCondition = JsonIgnoreCondition.WhenWritingNull,
        Converters = { new JsonStringEnumConverter(JsonNamingPolicy.CamelCase) },
    };

    /// <summary>
    /// The schema as the text of <c>schema.json</c>. Enums go out as words, because
    /// the Python trainer reads this file too and a bare 1 means nothing there.
    /// </summary>
    public string ToJson() => JsonSerializer.Serialize(this, Json);

    /// <summary>Reads <c>schema.json</c> back, and validates what it finds.</summary>
    public static DatasetSchema FromJson(string json)
    {
        DatasetSchema? schema;
        try
        {
            schema = JsonSerializer.Deserialize<DatasetSchema>(json, Json);
        }
        catch (JsonException ex)
        {
            throw new InvalidDataException($"schema.json could not be read: {ex.Message}", ex);
        }

        if (schema is null)
            throw new InvalidDataException("schema.json is empty.");

        if (schema.FormatVersion != CurrentFormatVersion)
            throw new InvalidDataException(
                $"schema.json is format {schema.FormatVersion} and this version of OtterLogic reads format "
                + $"{CurrentFormatVersion}.");

        try
        {
            schema.Validate();
        }
        catch (ArgumentException ex)
        {
            throw new InvalidDataException($"schema.json is not a usable schema: {ex.Message}", ex);
        }

        return schema;
    }
}
