namespace OtterLogic.MachineLearning.Data;

/// <summary>One column of a dataset: what it is called, what it is for, and what it holds.</summary>
public sealed record DatasetColumn
{
    /// <summary>
    /// The header in every CSV, and what a trained model's sidecar asserts on. Column
    /// order is baked into a model, so the name is the only thing that can catch two
    /// columns wired the wrong way round.
    /// </summary>
    public required string Name { get; init; }

    /// <summary>What the column is for.</summary>
    public ColumnRole Role { get; init; } = ColumnRole.Feature;

    /// <summary>Whether it holds quantities or names. Features are always numbers.</summary>
    public ColumnKind Kind { get; init; } = ColumnKind.Number;

    /// <summary>Free text, recorded so whoever opens the folder next year can read it. Never checked.</summary>
    public string? Unit { get; init; }

    /// <summary>
    /// For a category column, every class seen so far, in the order first seen.
    /// <para>
    /// The position in this list is the class index a model is trained on, so it is
    /// append-only: a class met for the first time in the tenth project goes on the
    /// end, and the indices the first nine were written with still mean what they
    /// meant. Sorting it would be tidier and would silently relabel every model
    /// already trained.
    /// </para>
    /// </summary>
    public IReadOnlyList<string> Classes { get; init; } = Array.Empty<string>();

    /// <summary>A number feature.</summary>
    public static DatasetColumn Feature(string name, string? unit = null)
        => new() { Name = name, Role = ColumnRole.Feature, Kind = ColumnKind.Number, Unit = unit };

    /// <summary>A number target — a regression.</summary>
    public static DatasetColumn NumberTarget(string name, string? unit = null)
        => new() { Name = name, Role = ColumnRole.Target, Kind = ColumnKind.Number, Unit = unit };

    /// <summary>A class target — a classification. Classes are collected as rows are added.</summary>
    public static DatasetColumn ClassTarget(string name)
        => new() { Name = name, Role = ColumnRole.Target, Kind = ColumnKind.Category };

    /// <summary>The row identifier.</summary>
    public static DatasetColumn Id(string name)
        => new() { Name = name, Role = ColumnRole.Id, Kind = ColumnKind.Category };
}
