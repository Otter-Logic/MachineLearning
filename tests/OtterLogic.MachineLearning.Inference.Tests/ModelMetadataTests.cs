using Xunit;

namespace OtterLogic.MachineLearning.Inference.Tests;

public class ModelMetadataTests
{
    private static ModelMetadata Classifier() => new()
    {
        Task = ModelTask.Classification,
        Features = new[] { "span", "sag", "rest_factor" },
        Target = "stiff",
        Classes = new[] { "no", "yes" },
        ExtractorVersion = "1",
        Trainer = new TrainerInfo("otterlogic-trainer", "0.1.0", "boostedTrees"),
        Trained = new DateTimeOffset(2026, 9, 23, 14, 2, 11, TimeSpan.Zero),
        Data = new DataInfo(160, 4, new[] { "2025-003" }),
        Score = new Dictionary<string, double> { ["accuracy"] = 0.94, ["noInformationRate"] = 0.71 },
    };

    [Fact]
    public void RoundTripsThroughJson()
    {
        var original = Classifier();
        var back = ModelMetadata.FromJson(original.ToJson());

        Assert.Equal(original.Task, back.Task);
        Assert.Equal(original.Features, back.Features);
        Assert.Equal(original.Target, back.Target);
        Assert.Equal(original.Classes, back.Classes);
        Assert.Equal(original.ExtractorVersion, back.ExtractorVersion);
        Assert.Equal(original.Trainer, back.Trainer);
        Assert.Equal(original.Trained, back.Trained);
        Assert.Equal(original.Data!.Rows, back.Data!.Rows);
        Assert.Equal(original.Data.HoldoutGroups, back.Data.HoldoutGroups);
        Assert.Equal(0.94, back.Score["accuracy"]);
    }

    [Fact]
    public void ReadsWhatThePythonSideWrites()
    {
        // Byte-for-byte the shape metadata.py produces: camelCase keys, the task
        // and model type as words, the time as ISO-8601 UTC with a Z.
        const string json = """
            {"format":1,"task":"regression","features":["span","load"],"target":"deflection",
             "extractorVersion":"1","trainer":{"name":"otterlogic-trainer","version":"0.1.0","modelType":"boostedTrees"},
             "trained":"2026-09-23T14:02:11Z","data":{"rows":160,"groups":4,"holdoutGroups":["2025-003"]},
             "score":{"rSquared":0.91,"meanAbsoluteError":2.3,"rootMeanSquaredError":3.1}}
            """;

        var metadata = ModelMetadata.FromJson(json);

        Assert.Equal(ModelTask.Regression, metadata.Task);
        Assert.Empty(metadata.Classes);
        Assert.Equal("boostedTrees", metadata.Trainer!.ModelType);
        Assert.Equal(new DateTimeOffset(2026, 9, 23, 14, 2, 11, TimeSpan.Zero), metadata.Trained);
        Assert.Equal(2.3, metadata.Score["meanAbsoluteError"]);
    }

    [Fact]
    public void RefusesAFormatItDoesNotKnow()
    {
        var ex = Assert.Throws<InvalidDataException>(() => ModelMetadata.FromJson("""{"format":2,"task":"regression","features":["a"],"target":"t"}"""));
        Assert.Contains("format 2", ex.Message);
    }

    [Fact]
    public void RefusesAClassifierWithOneClass()
    {
        var one = Classifier() with { Classes = new[] { "yes" } };
        Assert.Throws<InvalidDataException>(one.Validate);
    }

    [Fact]
    public void RefusesARegressorWithClasses()
    {
        var wrong = Classifier() with { Task = ModelTask.Regression };
        Assert.Throws<InvalidDataException>(wrong.Validate);
    }

    [Fact]
    public void RefusesDuplicateFeatures()
    {
        var twice = Classifier() with { Features = new[] { "span", "Span" } };
        var ex = Assert.Throws<InvalidDataException>(twice.Validate);
        Assert.Contains("twice", ex.Message);
    }

    [Fact]
    public void RefusesMalformedJson()
    {
        Assert.Throws<InvalidDataException>(() => ModelMetadata.FromJson("{not json"));
    }

    [Fact]
    public void DescribesItselfForAPerson()
    {
        var lines = Classifier().Describe();

        Assert.Equal("Predicts stiff: one of no, yes.", lines[0]);
        Assert.Equal("3 features, in this order: span, sag, rest_factor.", lines[1]);
        Assert.Equal("Trained 2026-09-23 with otterlogic-trainer 0.1.0 (boosted trees) on 160 rows from 4 groups.", lines[2]);
        Assert.Equal("Scored on held-out groups: 2025-003.", lines[3]);
        Assert.Equal("Score: accuracy 0.94, no information rate 0.71.", lines[4]);
        Assert.Equal("Extractor version 1.", lines[5]);
    }

    [Fact]
    public void DescribesARegressorWithNothingButTheEssentials()
    {
        var bare = new ModelMetadata { Task = ModelTask.Regression, Features = new[] { "a" }, Target = "t" };
        var lines = bare.Describe();

        Assert.Equal(2, lines.Count);
        Assert.Equal("Predicts t: a number.", lines[0]);
        Assert.Equal("1 feature, in this order: a.", lines[1]);
    }
}
