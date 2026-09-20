namespace OtterLogic.MachineLearning.Data;

/// <summary>What a dataset column is for.</summary>
public enum ColumnRole
{
    /// <summary>An input a model learns from. Always a number.</summary>
    Feature,

    /// <summary>What a model is asked to predict — a number or a class.</summary>
    Target,

    /// <summary>
    /// Names the row, so a prediction can be traced back to the thing it was made
    /// about. Never learned from: an identifier that leaks into the features is a
    /// model memorising which row is which.
    /// </summary>
    Id,
}
