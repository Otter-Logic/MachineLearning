namespace OtterLogic.MachineLearning.Inference;

/// <summary>What a trained model answers with.</summary>
public enum ModelTask
{
    /// <summary>A class, from the list in the model's metadata, with a probability per class.</summary>
    Classification,

    /// <summary>A number.</summary>
    Regression,
}
