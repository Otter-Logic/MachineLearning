namespace OtterLogic.MachineLearning.Training;

/// <summary>
/// What the trainer holds back to score on.
/// </summary>
public enum HoldoutBy
{
    /// <summary>
    /// Whole groups — in a dataset folder, whole models. The honest score: rows from
    /// one model are near-copies of each other, so a model tested on rows whose
    /// neighbours it trained on reports an accuracy it will never reach on a new
    /// project.
    /// </summary>
    Group = 0,

    /// <summary>
    /// Rows at random, for samples that came with no groups. The score may be
    /// optimistic for the reason above, and the report says so — but a score is
    /// better than none, and a user who has groups can wire them.
    /// </summary>
    Row = 1,
}
