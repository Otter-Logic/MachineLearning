namespace OtterLogic.MachineLearning.Inference;

/// <summary>
/// What a model answered for a block of rows, in the row order they were given.
/// <para>
/// One type for both tasks, because the caller usually does not know which it has
/// until the file is opened: a classifier fills <see cref="Labels"/>,
/// <see cref="Confidence"/> and <see cref="Probabilities"/>; a regressor fills
/// <see cref="Values"/>. <see cref="Task"/> says which.
/// </para>
/// </summary>
public sealed class ModelPrediction
{
    internal ModelPrediction(string[] labels, int[] labelIndices, double[] confidence, double[,] probabilities)
    {
        Task = ModelTask.Classification;
        Labels = labels;
        LabelIndices = labelIndices;
        Confidence = confidence;
        Probabilities = probabilities;
        Values = Array.Empty<double>();
    }

    internal ModelPrediction(double[] values)
    {
        Task = ModelTask.Regression;
        Values = values;
        Labels = Array.Empty<string>();
        LabelIndices = Array.Empty<int>();
        Confidence = Array.Empty<double>();
        Probabilities = new double[0, 0];
    }

    /// <summary>Which kind of answer this is.</summary>
    public ModelTask Task { get; }

    /// <summary>Rows answered.</summary>
    public int SampleCount => Task == ModelTask.Classification ? Labels.Length : Values.Length;

    /// <summary>The predicted class name per row. Empty for a regressor.</summary>
    public string[] Labels { get; }

    /// <summary>The predicted class as its position in the model's class list. Empty for a regressor.</summary>
    public int[] LabelIndices { get; }

    /// <summary>
    /// The probability of the predicted class, per row. Empty for a regressor: a
    /// boosted tree has no honest number to give here, and a made-up one would be
    /// read as if it were honest.
    /// </summary>
    public double[] Confidence { get; }

    /// <summary>Rows by classes, the probability of each class. Empty for a regressor, or for a classifier that reports none.</summary>
    public double[,] Probabilities { get; }

    /// <summary>The predicted number per row. Empty for a classifier.</summary>
    public double[] Values { get; }
}
