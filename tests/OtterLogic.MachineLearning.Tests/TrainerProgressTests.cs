using OtterLogic.MachineLearning.Training;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class TrainerProgressTests
{
    [Fact]
    public void ReadsAStageLine()
    {
        var progress = TrainerProgress.FromJsonLine("""{"stage": "fitting", "message": "Fitting boosted trees on 240 rows."}""");

        Assert.Equal("fitting", progress.Stage);
        Assert.Equal("Fitting boosted trees on 240 rows.", progress.Message);
        Assert.False(progress.Finished);
    }

    [Fact]
    public void ReadsTheDoneLineThePythonSideWrites()
    {
        var progress = TrainerProgress.FromJsonLine(
            """{"stage": "done", "message": "Done.", "done": true, "model": "C:\\models\\m.onnx", "score": {"rSquared": 0.9, "meanAbsoluteError": 1.5}, "report": ["a", "b"]}""");

        Assert.True(progress.Done);
        Assert.True(progress.Finished);
        Assert.Null(progress.Error);
        Assert.Equal(@"C:\models\m.onnx", progress.Model);
        Assert.Equal(0.9, progress.Score!["rSquared"]);
        Assert.Equal(new[] { "a", "b" }, progress.Report);
    }

    [Fact]
    public void ReadsAnErrorLine()
    {
        var progress = TrainerProgress.FromJsonLine("""{"error": "There is no dataset folder at 'x'."}""");

        Assert.True(progress.Finished);
        Assert.False(progress.Done);
        Assert.Contains("no dataset folder", progress.Error);
    }

    [Fact]
    public void ALineThatIsNotJsonIsAnError()
    {
        var progress = TrainerProgress.FromJsonLine("Traceback (most recent call last):");

        Assert.True(progress.Finished);
        Assert.Contains("could not be read", progress.Error);
    }

    [Fact]
    public void RoundTrips()
    {
        var original = new TrainerProgress
        {
            Stage = "done", Done = true, Model = "m.onnx",
            Score = new Dictionary<string, double> { ["accuracy"] = 0.8 },
            Report = new[] { "line" },
        };

        var back = TrainerProgress.FromJsonLine(original.ToJsonLine());

        Assert.Equal(original.Stage, back.Stage);
        Assert.Equal(original.Done, back.Done);
        Assert.Equal(original.Model, back.Model);
        Assert.Equal(0.8, back.Score!["accuracy"]);
        Assert.Equal(original.Report, back.Report);
    }
}
