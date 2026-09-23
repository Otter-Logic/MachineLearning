using System.Text.Json;
using Xunit;

namespace OtterLogic.MachineLearning.Inference.Tests;

/// <summary>
/// The parity test across the ONNX boundary. The fixtures were trained and
/// answered by the Python trainer; the C# side opens the same files and must give
/// the same answers, to float32 tolerance.
/// </summary>
public class OnnxModelTests
{
    private static string Fixture(string name) => Path.Combine(AppContext.BaseDirectory, "Fixtures", name);

    private sealed record Expected(
        string[] Features, double[][] Rows, Classifier Classifier, Regressor Regressor);

    private sealed record Classifier(int[] Labels, double[][] Probabilities);

    private sealed record Regressor(double[] Values);

    private static readonly Expected Answers = JsonSerializer.Deserialize<Expected>(
        File.ReadAllText(Fixture("inference.json")),
        new JsonSerializerOptions { PropertyNameCaseInsensitive = true })!;

    private static double[,] Rows()
    {
        var rows = Answers.Rows;
        var block = new double[rows.Length, rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[i].Length; j++)
                block[i, j] = rows[i][j];
        return block;
    }

    [Fact]
    public void ClassifierMatchesPython()
    {
        using var model = OnnxModel.Load(Fixture("classifier.onnx"));

        Assert.True(model.HasMetadata);
        Assert.Equal(ModelTask.Classification, model.Task);
        Assert.Equal(Answers.Features, model.FeatureNames);
        Assert.Equal(new[] { "no", "yes" }, model.Metadata!.Classes);
        Assert.Equal("fixture-1", model.Metadata.ExtractorVersion);
        Assert.Equal("boostedTrees", model.Metadata.Trainer!.ModelType);

        var prediction = model.Predict(Rows());

        Assert.Equal(Answers.Rows.Length, prediction.SampleCount);
        Assert.Equal(Answers.Classifier.Labels, prediction.LabelIndices);
        Assert.Equal(Answers.Classifier.Labels.Select(i => model.Metadata.Classes[i]), prediction.Labels);

        for (int i = 0; i < prediction.SampleCount; i++)
        {
            for (int c = 0; c < 2; c++)
                Assert.Equal(Answers.Classifier.Probabilities[i][c], prediction.Probabilities[i, c], 1e-5);

            Assert.Equal(prediction.Probabilities[i, prediction.LabelIndices[i]], prediction.Confidence[i], 1e-12);
        }
    }

    [Fact]
    public void RegressorMatchesPython()
    {
        using var model = OnnxModel.Load(Fixture("regressor.onnx"));

        Assert.Equal(ModelTask.Regression, model.Task);
        Assert.Equal("deflection", model.Metadata!.Target);
        Assert.True(model.Metadata.Score["rSquared"] > 0);

        var prediction = model.Predict(Rows());

        Assert.Empty(prediction.Labels);
        Assert.Empty(prediction.Confidence);
        for (int i = 0; i < prediction.SampleCount; i++)
            Assert.Equal(Answers.Regressor.Values[i], prediction.Values[i], 1e-3);
    }

    [Fact]
    public void RefusesTheWrongNumberOfColumns()
    {
        using var model = OnnxModel.Load(Fixture("regressor.onnx"));

        var ex = Assert.Throws<ArgumentException>(() => model.Predict(new double[2, 3]));
        Assert.Contains("expects 4", ex.Message);
        Assert.Contains("span", ex.Message);
    }

    [Fact]
    public void RefusesANonFiniteValue()
    {
        using var model = OnnxModel.Load(Fixture("regressor.onnx"));
        var rows = Rows();
        rows[1, 2] = double.NaN;

        Assert.Throws<ArgumentException>(() => model.Predict(rows));
    }

    [Fact]
    public void AnswersNothingForNoRows()
    {
        using var model = OnnxModel.Load(Fixture("classifier.onnx"));
        var prediction = model.Predict(new double[0, 4]);
        Assert.Equal(0, prediction.SampleCount);
    }

    [Fact]
    public void MissingFileIsSaidSo()
    {
        Assert.Throws<FileNotFoundException>(() => OnnxModel.Load(Fixture("nothing.onnx")));
    }

    [Fact]
    public void NotAModelIsSaidSo()
    {
        string path = Path.Combine(Path.GetTempPath(), Path.GetRandomFileName() + ".onnx");
        File.WriteAllText(path, "this is not a model");
        try
        {
            Assert.Throws<InvalidDataException>(() => OnnxModel.Load(path));
        }
        finally
        {
            File.Delete(path);
        }
    }

    [Fact]
    public void DescribesTheFile()
    {
        using var model = OnnxModel.Load(Fixture("classifier.onnx"));
        var lines = model.Describe();

        Assert.Equal("Predicts stiff: one of no, yes.", lines[0]);
        Assert.StartsWith("4 features, in this order: span, sag, rest_factor, load.", lines[1]);
    }

    [Fact]
    public void RemembersWhenTheFileWasWritten()
    {
        string path = Fixture("regressor.onnx");
        using var model = OnnxModel.Load(path);
        Assert.Equal(File.GetLastWriteTimeUtc(path), model.LastWriteTimeUtc);
        Assert.Equal(path, model.Path);
    }
}
