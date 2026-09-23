using OtterLogic.MachineLearning.Training;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class TrainingJobTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "otterlogic-job-" + Guid.NewGuid().ToString("N"));

    public TrainingJobTests()
    {
        Directory.CreateDirectory(Path.Combine(_root, "dataset"));
        File.WriteAllText(Path.Combine(_root, "dataset", "schema.json"), "{}");
    }

    public void Dispose() => Directory.Delete(_root, recursive: true);

    private TrainingJob Valid() => new()
    {
        DatasetFolder = Path.Combine(_root, "dataset"),
        OutputPath = Path.Combine(_root, "model.onnx"),
    };

    [Fact]
    public void WritesWhatThePythonSideReads()
    {
        var job = Valid() with
        {
            Target = "stiff",
            Learner = new NeuralNetworkLearner { HiddenLayers = new[] { 32, 16 }, Iterations = 300 },
            HoldoutBy = HoldoutBy.Row,
            HoldoutFraction = 0.3,
            Seed = 7,
        };
        string json = job.ToJson();

        Assert.Contains("\"datasetFolder\"", json);
        Assert.Contains("\"target\": \"stiff\"", json);
        Assert.Contains("\"learner\": {", json);
        Assert.Contains("\"type\": \"neuralNetwork\"", json);
        Assert.Contains("\"hiddenLayers\"", json);
        Assert.Contains("\"iterations\": 300", json);
        Assert.Contains("\"holdoutBy\": \"row\"", json);
        Assert.Contains("\"holdoutFraction\": 0.3", json);
        Assert.Contains("\"seed\": 7", json);
        Assert.Contains("\"outputPath\"", json);

        // The old flat key is gone; the Python side reads the object.
        Assert.DoesNotContain("modelType", json);
    }

    [Fact]
    public void DefaultsToBoostedTreesHeldOutByGroup()
    {
        var job = Valid();
        string json = job.ToJson();

        Assert.IsType<BoostedTreesLearner>(job.Learner);
        Assert.Equal(HoldoutBy.Group, job.HoldoutBy);
        Assert.Contains("\"type\": \"boostedTrees\"", json);
        Assert.Contains("\"trees\": 100", json);
        Assert.Contains("\"depth\": 0", json);
        Assert.Contains("\"holdoutBy\": \"group\"", json);
    }

    [Fact]
    public void OmitsAMissingTarget()
    {
        Assert.DoesNotContain("target", Valid().ToJson());
    }

    [Fact]
    public void RoundTrips()
    {
        var job = Valid() with
        {
            Target = "t",
            Learner = new NeuralNetworkLearner { HiddenLayers = new[] { 8, 4 } },
            HoldoutBy = HoldoutBy.Row,
        };
        var back = TrainingJob.FromJson(job.ToJson());
        Assert.Equal(job, back);
    }

    [Fact]
    public void ReadsAJobWrittenByHand()
    {
        // The discriminator comes first, as the serialiser writes it and as the
        // reader requires it.
        const string json = """
            {
              "datasetFolder": "C:/datasets/x",
              "learner": { "type": "boostedTrees", "trees": 250, "depth": 3 },
              "holdoutBy": "row",
              "outputPath": "C:/models/x.onnx"
            }
            """;

        var job = TrainingJob.FromJson(json);

        var trees = Assert.IsType<BoostedTreesLearner>(job.Learner);
        Assert.Equal(250, trees.Trees);
        Assert.Equal(3, trees.Depth);
        Assert.Equal(HoldoutBy.Row, job.HoldoutBy);
        Assert.Null(job.Target);
    }

    [Fact]
    public void ValidAsBuilt()
    {
        Valid().Validate();
    }

    [Fact]
    public void RefusesAFolderWithoutASchema()
    {
        Directory.CreateDirectory(Path.Combine(_root, "empty"));
        var job = Valid() with { DatasetFolder = Path.Combine(_root, "empty") };
        var ex = Assert.Throws<ArgumentException>(job.Validate);
        Assert.Contains("schema.json", ex.Message);
    }

    [Fact]
    public void RefusesAnOutputThatIsNotOnnx()
    {
        var job = Valid() with { OutputPath = Path.Combine(_root, "model.txt") };
        Assert.Throws<ArgumentException>(job.Validate);
    }

    [Fact]
    public void RefusesAnOutputFolderThatDoesNotExist()
    {
        var job = Valid() with { OutputPath = Path.Combine(_root, "nowhere", "model.onnx") };
        Assert.Throws<ArgumentException>(job.Validate);
    }

    [Theory]
    [InlineData(0.0)]
    [InlineData(1.0)]
    public void RefusesAHoldoutThatLeavesNothing(double fraction)
    {
        var job = Valid() with { HoldoutFraction = fraction };
        Assert.Throws<ArgumentException>(job.Validate);
    }

    [Fact]
    public void RefusesALearnerWithABadSettingAndNamesIt()
    {
        var job = Valid() with { Learner = new BoostedTreesLearner { Trees = 0 } };
        var ex = Assert.Throws<ArgumentException>(job.Validate);
        Assert.Contains("Boosted Trees", ex.Message);
        Assert.Contains("tree", ex.Message);
    }

    [Fact]
    public void RefusesAMissingLearner()
    {
        var job = Valid() with { Learner = null! };
        Assert.Throws<ArgumentException>(job.Validate);
    }

    [Fact]
    public void RefusesAHoldoutByThatIsNotOne()
    {
        var job = Valid() with { HoldoutBy = (HoldoutBy)9 };
        Assert.Throws<ArgumentException>(job.Validate);
    }
}
