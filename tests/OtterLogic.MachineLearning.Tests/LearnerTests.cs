using OtterLogic.MachineLearning.Training;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The learner records: what they write into <c>job.json</c>, that it reads back
/// as the same value, and what they refuse. The names asserted here are the ones
/// <c>models.py</c> reads, so a rename on this side fails here before it fails in
/// a user's job.
/// </summary>
public class LearnerTests
{
    public static IEnumerable<object[]> Every() => new[]
    {
        new object[] { new BoostedTreesLearner { Trees = 250, Depth = 3 }, "boostedTrees" },
        new object[] { new NeuralNetworkLearner { HiddenLayers = new[] { 32, 16 }, Iterations = 300 }, "neuralNetwork" },
        new object[] { new LinearLearner { Regularisation = 0.5 }, "linear" },
        new object[] { new NearestNeighboursLearner { Neighbours = 9 }, "nearestNeighbours" },
    };

    private static TrainingJob JobWith(Learner learner) => new()
    {
        DatasetFolder = "C:/datasets/x",
        OutputPath = "C:/models/x.onnx",
        Learner = learner,
    };

    [Theory]
    [MemberData(nameof(Every))]
    public void RoundTripsThroughAJobWithItsTypeNamed(Learner learner, string type)
    {
        string json = JobWith(learner).ToJson();

        Assert.Equal(type, learner.TypeName);
        Assert.Contains($"\"type\": \"{type}\"", json);

        var back = TrainingJob.FromJson(json).Learner;
        Assert.IsType(learner.GetType(), back);
        Assert.Equal(learner, back);
    }

    [Fact]
    public void WritesTheSettingNamesThePythonSideReads()
    {
        Assert.Contains("\"trees\": 250", JobWith(new BoostedTreesLearner { Trees = 250 }).ToJson());
        Assert.Contains("\"depth\": 3", JobWith(new BoostedTreesLearner { Depth = 3 }).ToJson());
        Assert.Contains("\"hiddenLayers\"", JobWith(new NeuralNetworkLearner()).ToJson());
        Assert.Contains("\"iterations\": 500", JobWith(new NeuralNetworkLearner()).ToJson());
        Assert.Contains("\"regularisation\": 1", JobWith(new LinearLearner()).ToJson());
        Assert.Contains("\"neighbours\": 5", JobWith(new NearestNeighboursLearner()).ToJson());
    }

    [Fact]
    public void ATypeNobodyKnowsIsRefusedOnRead()
    {
        const string json = """{"datasetFolder": "x", "outputPath": "x.onnx", "learner": {"type": "randomForest"}}""";
        Assert.ThrowsAny<System.Text.Json.JsonException>(() => TrainingJob.FromJson(json));
    }

    [Theory]
    [MemberData(nameof(Every))]
    public void DescribesItselfByNameAndSetting(Learner learner, string type)
    {
        string text = learner.Describe();
        Assert.StartsWith(learner.Name, text);
        Assert.Equal(text, learner.ToString());
        Assert.NotEqual(type, text);
    }

    [Fact]
    public void DescribesDepthOnlyWhenLimited()
    {
        Assert.Equal("Boosted Trees, 100 trees", new BoostedTreesLearner().Describe());
        Assert.Equal("Boosted Trees, 40 trees of depth 6", new BoostedTreesLearner { Trees = 40, Depth = 6 }.Describe());
        Assert.Contains("32, 16", new NeuralNetworkLearner { HiddenLayers = new[] { 32, 16 } }.Describe());
    }

    [Theory]
    [MemberData(nameof(Every))]
    public void DefaultsAndTheTestValuesAreValid(Learner learner, string type)
    {
        Assert.NotNull(type);
        learner.Validate();
        ((Learner)Activator.CreateInstance(learner.GetType())!).Validate();
    }

    public static IEnumerable<object[]> Bad() => new[]
    {
        new object[] { new BoostedTreesLearner { Trees = 0 } },
        new object[] { new BoostedTreesLearner { Depth = -1 } },
        new object[] { new NeuralNetworkLearner { HiddenLayers = Array.Empty<int>() } },
        new object[] { new NeuralNetworkLearner { HiddenLayers = new[] { 64, 0 } } },
        new object[] { new NeuralNetworkLearner { Iterations = 0 } },
        new object[] { new LinearLearner { Regularisation = 0.0 } },
        new object[] { new LinearLearner { Regularisation = -1.0 } },
        new object[] { new LinearLearner { Regularisation = double.NaN } },
        new object[] { new LinearLearner { Regularisation = double.PositiveInfinity } },
        new object[] { new NearestNeighboursLearner { Neighbours = 0 } },
    };

    [Theory]
    [MemberData(nameof(Bad))]
    public void RefusesASettingTheTrainerWouldRefuse(Learner learner)
    {
        Assert.Throws<ArgumentOutOfRangeException>(learner.Validate);
    }

    [Fact]
    public void NeuralNetworksCompareByTheirLayers()
    {
        var a = new NeuralNetworkLearner { HiddenLayers = new[] { 32, 16 }, Iterations = 200 };
        var b = new NeuralNetworkLearner { HiddenLayers = new[] { 32, 16 }, Iterations = 200 };
        var wider = new NeuralNetworkLearner { HiddenLayers = new[] { 64, 16 }, Iterations = 200 };
        var deeper = new NeuralNetworkLearner { HiddenLayers = new[] { 32, 16, 8 }, Iterations = 200 };
        var longer = b with { Iterations = 201 };

        Assert.Equal(a, b);
        Assert.Equal(a.GetHashCode(), b.GetHashCode());
        Assert.NotEqual(a, wider);
        Assert.NotEqual(a, deeper);
        Assert.NotEqual(a, longer);
        Assert.NotEqual<Learner>(a, new BoostedTreesLearner());
    }

    [Fact]
    public void TheOtherLearnersCompareByValue()
    {
        Assert.Equal(new BoostedTreesLearner { Trees = 7 }, new BoostedTreesLearner { Trees = 7 });
        Assert.NotEqual(new BoostedTreesLearner { Trees = 7 }, new BoostedTreesLearner { Trees = 8 });
        Assert.Equal(new LinearLearner { Regularisation = 2 }, new LinearLearner { Regularisation = 2 });
        Assert.Equal(new NearestNeighboursLearner { Neighbours = 3 }, new NearestNeighboursLearner { Neighbours = 3 });
    }
}
