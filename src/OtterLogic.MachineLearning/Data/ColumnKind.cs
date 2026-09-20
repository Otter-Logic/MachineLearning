namespace OtterLogic.MachineLearning.Data;

/// <summary>Whether a column holds quantities or names.</summary>
public enum ColumnKind
{
    /// <summary>A finite number. Predicting one is a regression.</summary>
    Number,

    /// <summary>
    /// One of a closed set of names. Predicting one is a classification.
    /// <para>
    /// Declared rather than inferred, because the commonest classes of all are
    /// written as numbers: a column of 0 and 1 parses perfectly well as a quantity,
    /// and a regression fitted to it returns 0.37 with no complaint.
    /// </para>
    /// </summary>
    Category,
}
