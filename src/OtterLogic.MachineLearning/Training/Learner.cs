using System.Text.Json.Serialization;

namespace OtterLogic.MachineLearning.Training;

/// <summary>
/// A method the trainer can fit, with its settings chosen: the learning half of
/// a training job, as one value on one wire.
/// <para>
/// The trainer runs in Python and reads this from <c>job.json</c>, so every
/// learner is a JSON object with a <c>type</c> naming the method and its
/// settings beside it in camelCase. The Python side (<c>models.py</c>) reads
/// exactly these names; a learner is offered only once its ONNX export has
/// passed a round-trip parity test, because a method that exports badly is a
/// model that runs and is quietly wrong.
/// </para>
/// <para>
/// Each learner carries only what a person who has looked the method up would
/// expect to set. Everything else — learning rates, tolerances, early stopping —
/// is fixed in the trainer, because a knob nobody can read is a knob set wrong.
/// </para>
/// </summary>
[JsonPolymorphic(TypeDiscriminatorPropertyName = "type")]
[JsonDerivedType(typeof(BoostedTreesLearner), BoostedTreesLearner.Type)]
[JsonDerivedType(typeof(NeuralNetworkLearner), NeuralNetworkLearner.Type)]
[JsonDerivedType(typeof(LinearLearner), LinearLearner.Type)]
[JsonDerivedType(typeof(NearestNeighboursLearner), NearestNeighboursLearner.Type)]
public abstract record Learner
{
    /// <summary>The method's name as a user knows it: "Boosted Trees".</summary>
    [JsonIgnore]
    public abstract string Name { get; }

    /// <summary>The <c>type</c> written into the job, and into the model's metadata as what trained it.</summary>
    [JsonIgnore]
    public abstract string TypeName { get; }

    /// <summary>The name and the settings that matter, in one line for a report.</summary>
    public abstract string Describe();

    /// <summary>Complains about a setting the trainer would refuse, before a process is started for it.</summary>
    /// <exception cref="ArgumentOutOfRangeException">A setting is outside what the method accepts.</exception>
    public abstract void Validate();

    public sealed override string ToString() => Describe();
}

/// <summary>
/// Gradient-boosted decision trees. The default: on tables of a few thousand
/// rows it is usually the most accurate thing available and fits in seconds,
/// scale does not matter to it, and it is unbothered by a feature that is useless.
/// </summary>
public sealed record BoostedTreesLearner : Learner
{
    public const string Type = "boostedTrees";

    /// <summary>How many trees are added, one after another. More fits closer, and past a few hundred mostly slower.</summary>
    public int Trees { get; init; } = 100;

    /// <summary>How deep each tree may grow, or 0 for no limit. A limit makes the fit smoother and less prone to memorising rows.</summary>
    public int Depth { get; init; }

    public override string Name => "Boosted Trees";

    public override string TypeName => Type;

    public override string Describe()
        => $"{Name}, {Trees} trees" + (Depth > 0 ? $" of depth {Depth}" : "");

    public override void Validate()
    {
        if (Trees < 1)
            throw new ArgumentOutOfRangeException(nameof(Trees), Trees, "Need at least one tree.");
        if (Depth < 0)
            throw new ArgumentOutOfRangeException(nameof(Depth), Depth, "Depth is a count, or 0 for no limit.");
    }
}

/// <summary>
/// A small fully-connected network. The comparison a user will ask for, and the
/// shape deep-learning models grow from. Slower to fit than trees, and it needs the
/// standardisation the trainer bakes into the graph.
/// </summary>
public sealed record NeuralNetworkLearner : Learner
{
    public const string Type = "neuralNetwork";

    /// <summary>Neurons in each hidden layer, input to output. One layer of 64 by default.</summary>
    public int[] HiddenLayers { get; init; } = { 64 };

    /// <summary>Passes over the training rows. Five hundred by default; the fit stops early once it settles.</summary>
    public int Iterations { get; init; } = 500;

    public override string Name => "Neural Network";

    public override string TypeName => Type;

    public override string Describe()
        => $"{Name}, hidden layers of {string.Join(", ", HiddenLayers)}, up to {Iterations} iterations";

    public override void Validate()
    {
        if (HiddenLayers is null || HiddenLayers.Length == 0)
            throw new ArgumentOutOfRangeException(nameof(HiddenLayers), "Need at least one hidden layer.");
        foreach (int width in HiddenLayers)
            if (width < 1)
                throw new ArgumentOutOfRangeException(nameof(HiddenLayers), width, "Every hidden layer needs at least one neuron.");
        if (Iterations < 1)
            throw new ArgumentOutOfRangeException(nameof(Iterations), Iterations, "Need at least one iteration.");
    }

    // An array property breaks a record's value equality — two learners with the
    // same layers would compare unequal — so equality is spelled out over the
    // contents, which is what a value means here.
    public bool Equals(NeuralNetworkLearner? other)
        => other is not null && Iterations == other.Iterations && HiddenLayers.AsSpan().SequenceEqual(other.HiddenLayers);

    public override int GetHashCode()
    {
        var hash = new HashCode();
        hash.Add(Iterations);
        foreach (int width in HiddenLayers)
            hash.Add(width);
        return hash.ToHashCode();
    }
}

/// <summary>
/// A linear model: ridge regression for a number, logistic regression for a class.
/// The simplest honest model, and the one to compare the others against — when it
/// does as well as boosted trees, the relationship is a straight line and the
/// trees are not earning their keep.
/// </summary>
public sealed record LinearLearner : Learner
{
    public const string Type = "linear";

    /// <summary>
    /// How strongly the coefficients are pulled toward zero. One by default. More
    /// makes the fit smoother and more cautious; less lets it follow the rows more
    /// closely. Ridge's alpha, and one over logistic regression's C.
    /// </summary>
    public double Regularisation { get; init; } = 1.0;

    public override string Name => "Linear Model";

    public override string TypeName => Type;

    public override string Describe() => $"{Name}, regularisation {Regularisation:0.###}";

    public override void Validate()
    {
        if (!(Regularisation > 0.0) || !double.IsFinite(Regularisation))
            throw new ArgumentOutOfRangeException(nameof(Regularisation), Regularisation, "Regularisation must be a positive number.");
    }
}

/// <summary>
/// Answers with the known answers of the nearest training rows: their commonest
/// class, or the average of their values. No fitting to speak of, and no
/// assumption about the shape of the relationship.
/// </summary>
public sealed record NearestNeighboursLearner : Learner
{
    public const string Type = "nearestNeighbours";

    /// <summary>How many nearest training rows have a say. Five by default; more smooths, fewer follows local detail.</summary>
    public int Neighbours { get; init; } = 5;

    public override string Name => "Nearest Neighbours";

    public override string TypeName => Type;

    public override string Describe() => $"{Name}, {Neighbours} neighbours";

    public override void Validate()
    {
        if (Neighbours < 1)
            throw new ArgumentOutOfRangeException(nameof(Neighbours), Neighbours, "Need at least one neighbour.");
    }
}
