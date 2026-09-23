using System.Globalization;
using OtterLogic.MachineLearning.Data;
using OtterLogic.MachineLearning.Training;
using Xunit;
using Xunit.Abstractions;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// Runs the real Python trainer from C#, end to end: a dataset folder written by
/// <see cref="DatasetFolder"/>, a job started by <see cref="TrainerProcess"/>, a
/// model on disk.
/// <para>
/// Needs a Python with the trainer package installed — the checkout's own
/// <c>python/.venv</c> after <c>pip install -e trainer</c>, or whatever
/// <see cref="TrainerRuntime"/> finds. Without one the tests pass with a note
/// rather than fail, so a hosted runner with no Python still goes green; the
/// Python side has its own tests for the trainer itself.
/// </para>
/// <para>
/// In one collection with <see cref="TrainerRuntimeTests"/>, which redirects the
/// install root and writes a fake <c>python.exe</c> under it; run at the same
/// time, <see cref="TrainerRuntime.Find"/> here could hand that back.
/// </para>
/// </summary>
[Collection("trainer runtime")]
public class TrainerProcessTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _root = Path.Combine(Path.GetTempPath(), "otterlogic-trainer-" + Guid.NewGuid().ToString("N"));

    public TrainerProcessTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_root);
    }

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    /// <summary>The trainer's Python, or null with a reason on the test output.</summary>
    private string? Python()
    {
        string? found = TrainerRuntime.Find();
        if (found is not null)
            return found;

        // tests/<project>/bin/<config>/<tfm>/ is five levels below the repo root.
        string repo = Path.GetFullPath(Path.Combine(AppContext.BaseDirectory, "..", "..", "..", "..", ".."));
        string venv = Path.Combine(repo, "python", ".venv", "Scripts", "python.exe");
        if (File.Exists(venv))
            return venv;

        _output.WriteLine($"Skipped: no trainer runtime. {TrainerRuntime.Describe()} No venv at '{venv}'.");
        return null;
    }

    /// <summary>Fifty rows of a made-up sweep for one model, offset a little by its index so models differ.</summary>
    private static Dataset Sweep(Random rng, int g, bool classes, string? group)
    {
        const int n = 50;
        var features = new double[n, 3];
        var numbers = new double[n];
        var labels = new string[n];
        for (int i = 0; i < n; i++)
        {
            double span = 8 + 22 * rng.NextDouble() + g;
            double sag = 0.5 + 2.5 * rng.NextDouble();
            double load = 1 + 9 * rng.NextDouble();
            features[i, 0] = span;
            features[i, 1] = sag;
            features[i, 2] = load;
            numbers[i] = 0.02 * span * span * load / sag + rng.NextDouble();
            labels[i] = numbers[i] < 60 ? "yes" : "no";
        }

        var columns = new List<DatasetColumn>
        {
            DatasetColumn.Feature("span"), DatasetColumn.Feature("sag"), DatasetColumn.Feature("load"),
            classes ? DatasetColumn.ClassTarget("stiff") : DatasetColumn.NumberTarget("deflection"),
        };
        var schema = new DatasetSchema { Columns = columns };
        string[]? groups = group is null ? null : Enumerable.Repeat(group, n).ToArray();

        return classes
            ? Dataset.Create(schema, features,
                new Dictionary<string, double[]>(),
                new Dictionary<string, string[]> { ["stiff"] = labels },
                ids: null, groups: groups)
            : Dataset.Create(schema, features,
                new Dictionary<string, double[]> { ["deflection"] = numbers },
                new Dictionary<string, string[]>(),
                ids: null, groups: groups);
    }

    private string WriteDataset(bool classes)
    {
        string folder = Path.Combine(_root, "dataset");
        var rng = new Random(3);
        for (int g = 0; g < 4; g++)
            DatasetFolder.Write(folder, "2025-00" + g.ToString(CultureInfo.InvariantCulture), Sweep(rng, g, classes, null));

        return folder;
    }

    /// <summary>Four models' worth of rows on one wire, with or without the group each came from.</summary>
    private static (Dataset dataset, string[]? groups) Samples(bool classes, bool withGroups)
    {
        var rng = new Random(3);
        var parts = Enumerable.Range(0, 4)
            .Select(g => Sweep(rng, g, classes, withGroups ? "2025-00" + g.ToString(CultureInfo.InvariantCulture) : null))
            .ToList();

        // One table from four: rows appended, in order.
        int n = parts.Sum(p => p.RowCount);
        var features = new double[n, 3];
        var numbers = new double[n];
        var labels = new string[n];
        var groups = new string[n];
        int row = 0;
        foreach (var part in parts)
        {
            for (int i = 0; i < part.RowCount; i++, row++)
            {
                for (int j = 0; j < 3; j++)
                    features[row, j] = part.Features[i, j];
                if (classes)
                    labels[row] = part.ClassTarget("stiff")[i];
                else
                    numbers[row] = part.NumberTarget("deflection")[i];
                groups[row] = part.Groups[i];
            }
        }

        var dataset = classes
            ? Dataset.Create(parts[0].Schema, features, null, new Dictionary<string, string[]> { ["stiff"] = labels })
            : Dataset.Create(parts[0].Schema, features, new Dictionary<string, double[]> { ["deflection"] = numbers }, null);

        return (dataset, withGroups ? groups : null);
    }

    private static TrainerProgress WaitForEnd(TrainerProcess run, TimeSpan timeout)
    {
        var deadline = DateTime.UtcNow + timeout;
        while (DateTime.UtcNow < deadline)
        {
            var progress = run.Poll();
            if (progress.Finished)
            {
                // The final line lands a moment before the exit code does.
                run.WaitForExit(TimeSpan.FromSeconds(30));
                return progress;
            }
            Thread.Sleep(200);
        }

        run.Cancel();
        throw new TimeoutException("The trainer did not finish in " + timeout);
    }

    [Fact]
    public void TrainsARegressorFromAFolderTheReaderWrote()
    {
        string? python = Python();
        if (python is null) return;

        var job = new TrainingJob
        {
            DatasetFolder = WriteDataset(classes: false),
            OutputPath = Path.Combine(_root, "regressor.onnx"),
            Learner = new BoostedTreesLearner { Trees = 80, Depth = 4 },
        };

        using var run = TrainerProcess.Start(job, python, Path.Combine(_root, "job"));
        var end = WaitForEnd(run, TimeSpan.FromMinutes(3));

        _output.WriteLine(run.Output);
        Assert.True(end.Done, end.Error);
        Assert.Equal(job.OutputPath, end.Model);
        Assert.True(File.Exists(job.OutputPath));
        Assert.True(end.Score!["rSquared"] > 0.5, "R² " + end.Score["rSquared"]);
        Assert.NotEmpty(end.Report!);
        Assert.StartsWith("Trained on", end.Report![0]);
        Assert.Equal(0, run.ExitCode);
        Assert.True(File.Exists(Path.Combine(_root, "job", "job.json")));
    }

    [Fact]
    public void TrainsAClassifierAndReportsAConfusion()
    {
        string? python = Python();
        if (python is null) return;

        var job = new TrainingJob
        {
            DatasetFolder = WriteDataset(classes: true),
            Target = "stiff",
            OutputPath = Path.Combine(_root, "classifier.onnx"),
        };

        using var run = TrainerProcess.Start(job, python, Path.Combine(_root, "job"));
        var end = WaitForEnd(run, TimeSpan.FromMinutes(3));

        Assert.True(end.Done, end.Error);
        Assert.True(end.Score!.ContainsKey("accuracy"));
        Assert.Contains(end.Report!, line => line.StartsWith("Confusion", StringComparison.Ordinal));
    }

    [Theory]
    [InlineData("linear")]
    [InlineData("nearestNeighbours")]
    public void TheOtherLearnersReachTheTrainerByTheirTypeName(string type)
    {
        string? python = Python();
        if (python is null) return;

        Learner learner = type == "linear" ? new LinearLearner { Regularisation = 0.5 } : new NearestNeighboursLearner { Neighbours = 3 };
        var job = new TrainingJob
        {
            DatasetFolder = WriteDataset(classes: false),
            OutputPath = Path.Combine(_root, type + ".onnx"),
            Learner = learner,
        };

        using var run = TrainerProcess.Start(job, python, Path.Combine(_root, "job"));
        var end = WaitForEnd(run, TimeSpan.FromMinutes(3));

        Assert.True(end.Done, end.Error);
        Assert.True(File.Exists(job.OutputPath));
        Assert.Equal(type, learner.TypeName);
    }

    [Fact]
    public void ReportsTheTrainersOwnErrorForABadTarget()
    {
        string? python = Python();
        if (python is null) return;

        var job = new TrainingJob
        {
            DatasetFolder = WriteDataset(classes: false),
            Target = "span",
            OutputPath = Path.Combine(_root, "never.onnx"),
        };

        using var run = TrainerProcess.Start(job, python, Path.Combine(_root, "job"));
        var end = WaitForEnd(run, TimeSpan.FromMinutes(1));

        Assert.False(end.Done);
        Assert.Contains("feature column, not a target", end.Error);
        Assert.False(File.Exists(job.OutputPath));
        Assert.Equal(1, run.ExitCode);
    }

    [Fact]
    public void AProcessThatDiesWithoutAResultIsReportedWithItsStderr()
    {
        if (!OperatingSystem.IsWindows()) return;

        // Stands in for a Python that cannot even import the trainer: it writes
        // nothing to the progress file, complains on stderr and exits non-zero.
        string fake = Path.Combine(_root, "fake-trainer.cmd");
        File.WriteAllText(fake, "@echo off\r\necho boom: no such module >&2\r\nexit /b 3\r\n");

        var job = new TrainingJob
        {
            DatasetFolder = WriteDataset(classes: false),
            OutputPath = Path.Combine(_root, "never.onnx"),
        };

        using var run = TrainerProcess.Start(job, fake, Path.Combine(_root, "job"));
        var end = WaitForEnd(run, TimeSpan.FromMinutes(1));

        Assert.False(end.Done);
        Assert.Contains("exit code 3", end.Error);
        Assert.Contains("boom", end.Error);
        Assert.False(File.Exists(job.OutputPath));
    }

    [Fact]
    public void CancelStopsIt()
    {
        string? python = Python();
        if (python is null) return;

        var job = new TrainingJob
        {
            DatasetFolder = WriteDataset(classes: false),
            OutputPath = Path.Combine(_root, "cancelled.onnx"),
            Learner = new NeuralNetworkLearner(),
        };

        using var run = TrainerProcess.Start(job, python, Path.Combine(_root, "job"));
        run.Cancel();

        Assert.False(run.IsRunning);
        Assert.True(run.Poll().Finished);
    }

    [Fact]
    public void RefusesAJobThatWouldFailInTheProcess()
    {
        var job = new TrainingJob { DatasetFolder = Path.Combine(_root, "nowhere"), OutputPath = "x.onnx" };
        Assert.Throws<ArgumentException>(() => TrainerProcess.Start(job, "python.exe"));
    }

    [Fact]
    public void RuntimeDescribesWhereItLooked()
    {
        string description = TrainerRuntime.Describe();
        Assert.Contains(TrainerRuntime.EnvironmentVariable, description);
        Assert.Contains("trainer", description);
    }

    [Fact]
    public void SamplesWithGroupsAreWrittenAsModelsAndHeldOutByGroup()
    {
        string? python = Python();
        if (python is null) return;

        var (dataset, groups) = Samples(classes: false, withGroups: true);
        string output = Path.Combine(_root, "grouped.onnx");
        string work = Path.Combine(_root, "job");

        using var run = TrainerProcess.StartOnSamples(dataset, groups, new BoostedTreesLearner { Trees = 60 }, 0.25, output, python, workFolder: work);
        var end = WaitForEnd(run, TimeSpan.FromMinutes(3));

        _output.WriteLine(run.Output);
        Assert.True(end.Done, end.Error);
        Assert.True(File.Exists(output));
        Assert.Equal(HoldoutBy.Group, run.Job.HoldoutBy);
        Assert.Equal(Path.Combine(work, "dataset"), run.Job.DatasetFolder);
        Assert.Equal(new[] { "2025-000", "2025-001", "2025-002", "2025-003" }, DatasetFolder.Models(run.Job.DatasetFolder));

        // Whole models held out, named in the report; no caveat about random rows.
        Assert.StartsWith("Trained on 150 rows from 3 models", end.Report![0]);
        Assert.Contains("held-out (2025-00", end.Report[0]);
        Assert.DoesNotContain(end.Report, line => line.Contains("at random", StringComparison.Ordinal));
    }

    [Fact]
    public void SamplesWithoutGroupsAreHeldOutByRowAndTheReportSaysSo()
    {
        string? python = Python();
        if (python is null) return;

        var (dataset, groups) = Samples(classes: true, withGroups: false);
        Assert.Null(groups);
        string output = Path.Combine(_root, "ungrouped.onnx");
        string work = Path.Combine(_root, "job");

        using var run = TrainerProcess.StartOnSamples(dataset, groups, new LinearLearner(), 0.2, output, python, workFolder: work);
        var end = WaitForEnd(run, TimeSpan.FromMinutes(3));

        _output.WriteLine(run.Output);
        Assert.True(end.Done, end.Error);
        Assert.True(File.Exists(output));
        Assert.Equal(HoldoutBy.Row, run.Job.HoldoutBy);
        Assert.Equal(new[] { SampleFolder.UngroupedModel }, DatasetFolder.Models(run.Job.DatasetFolder));

        // The caveat comes first, so it qualifies every number after it.
        Assert.Contains("held out at random", end.Report![0]);
        Assert.Contains("optimistic", end.Report[0]);
        Assert.Contains("40 rows held out at random", end.Report[1]);
        Assert.True(end.Score!.ContainsKey("accuracy"));
    }

    [Fact]
    public void OneGroupIsNoGroupsForTheHoldout()
    {
        if (!OperatingSystem.IsWindows()) return;

        // No trainer needed to see the choice: the job is written before the
        // process starts, and a stub that exits at once stands in for Python.
        string stub = Path.Combine(_root, "stub.cmd");
        File.WriteAllText(stub, "@echo off\r\nexit /b 0\r\n");

        var (dataset, _) = Samples(classes: false, withGroups: false);
        string[] oneGroup = Enumerable.Repeat("only", dataset.RowCount).ToArray();

        using var run = TrainerProcess.StartOnSamples(dataset, oneGroup, new BoostedTreesLearner(), 0.25, Path.Combine(_root, "x.onnx"), stub, workFolder: Path.Combine(_root, "job"));

        Assert.Equal(HoldoutBy.Row, run.Job.HoldoutBy);
        Assert.Equal(new[] { "only" }, DatasetFolder.Models(run.Job.DatasetFolder));
        Assert.Contains("\"holdoutBy\": \"row\"", File.ReadAllText(Path.Combine(_root, "job", "job.json")));
    }

    [Fact]
    public void StartOnSamplesRefusesABadLearnerBeforeWritingAnything()
    {
        var (dataset, groups) = Samples(classes: false, withGroups: true);
        string work = Path.Combine(_root, "job");

        Assert.Throws<ArgumentException>(() =>
            TrainerProcess.StartOnSamples(dataset, groups, new NearestNeighboursLearner { Neighbours = 0 }, 0.25, Path.Combine(_root, "x.onnx"), "python.exe", workFolder: work));

        Assert.False(File.Exists(Path.Combine(work, "job.json")));
    }
}
