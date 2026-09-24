using OtterLogic.Dataset.Data;
using System.Diagnostics;
using System.Text;

namespace OtterLogic.MachineLearning.Training;

/// <summary>
/// One training run, in a separate process, watched through its progress file.
/// <para>
/// Out of process for two reasons that are both about Rhino. A training loop on the
/// UI thread freezes the canvas for as long as it runs, and a fault in a native
/// numerical library — a bad wheel, a driver — takes the host down with it. A
/// process of its own can do neither.
/// </para>
/// <para>
/// The caller starts it, calls <see cref="Poll"/> whenever it wants the latest
/// state, and disposes it. Nothing here waits: a Grasshopper component polls from
/// its own solve and schedules the next one, so the canvas keeps responding.
/// </para>
/// </summary>
public sealed class TrainerProcess : IDisposable
{
    /// <summary>The module the runtime is asked to run.</summary>
    public const string Module = "otterlogic_trainer";

    private readonly Process _process;
    private readonly StringBuilder _stderr = new();
    private readonly object _gate = new();
    private long _offset;
    private string _partial = string.Empty;
    private bool _disposed;

    private TrainerProcess(Process process, TrainingJob job, string workFolder)
    {
        _process = process;
        Job = job;
        WorkFolder = workFolder;
        StartedUtc = DateTime.UtcNow;
    }

    /// <summary>The request this run is for.</summary>
    public TrainingJob Job { get; }

    /// <summary>Holds <c>job.json</c>, <c>progress.jsonl</c> and the trainer's stderr, for reading after the fact.</summary>
    public string WorkFolder { get; }

    /// <summary>The trainer's progress file.</summary>
    public string ProgressFile => Path.Combine(WorkFolder, "progress.jsonl");

    /// <summary>When the process was started.</summary>
    public DateTime StartedUtc { get; }

    /// <summary>The last state <see cref="Poll"/> saw.</summary>
    public TrainerProgress Latest { get; private set; } = new() { Stage = "starting", Message = "Starting the trainer." };

    /// <summary>True while the process is alive.</summary>
    public bool IsRunning => !_process.HasExited;

    /// <summary>The process's exit code, once it has one.</summary>
    public int? ExitCode => _process.HasExited ? _process.ExitCode : null;

    /// <summary>
    /// Writes the job file and starts the trainer.
    /// </summary>
    /// <param name="job">What to train. Validated first, so a bad request fails here rather than in the process.</param>
    /// <param name="python">The interpreter, from <see cref="TrainerRuntime.Find"/>.</param>
    /// <param name="workFolder">
    /// Where to put the job's files. A fresh folder under the temp directory by
    /// default; given explicitly by tests and by anyone who wants to keep them.
    /// </param>
    /// <exception cref="ArgumentException">The job or the interpreter is wrong.</exception>
    /// <exception cref="InvalidOperationException">The process could not be started.</exception>
    public static TrainerProcess Start(TrainingJob job, string python, string? workFolder = null)
    {
        ArgumentNullException.ThrowIfNull(job);
        job.Validate();

        if (string.IsNullOrWhiteSpace(python) || !File.Exists(python))
            throw new ArgumentException($"There is no Python at '{python}'.", nameof(python));

        workFolder ??= NewWorkFolder();
        Directory.CreateDirectory(workFolder);

        string jobFile = Path.Combine(workFolder, "job.json");
        File.WriteAllText(jobFile, job.ToJson(), new UTF8Encoding(false));

        string progressFile = Path.Combine(workFolder, "progress.jsonl");
        if (File.Exists(progressFile))
            File.Delete(progressFile);

        var start = new ProcessStartInfo
        {
            FileName = python,
            UseShellExecute = false,
            CreateNoWindow = true,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            WorkingDirectory = workFolder,
        };
        start.ArgumentList.Add("-m");
        start.ArgumentList.Add(Module);
        start.ArgumentList.Add(jobFile);

        // Unbuffered and UTF-8, or a trainer that dies mid-line leaves nothing in
        // the log, and a non-ASCII column name breaks the pipe on a machine set to
        // a legacy code page.
        start.Environment["PYTHONUNBUFFERED"] = "1";
        start.Environment["PYTHONIOENCODING"] = "utf-8";

        var process = new Process { StartInfo = start, EnableRaisingEvents = true };
        var run = new TrainerProcess(process, job, workFolder);

        // Both streams are drained asynchronously. A process that fills a pipe
        // nobody reads blocks on its next write and never finishes.
        process.OutputDataReceived += (_, e) => run.Append(e.Data);
        process.ErrorDataReceived += (_, e) => run.Append(e.Data);

        try
        {
            if (!process.Start())
                throw new InvalidOperationException($"'{python}' did not start.");
        }
        catch (System.ComponentModel.Win32Exception ex)
        {
            process.Dispose();
            throw new InvalidOperationException($"'{python}' could not be started: {ex.Message}", ex);
        }

        process.BeginOutputReadLine();
        process.BeginErrorReadLine();
        return run;
    }

    /// <summary>A fresh folder under the temp directory for one job's files, named so that two jobs never share one.</summary>
    public static string NewWorkFolder()
        => Path.Combine(Path.GetTempPath(), "OtterLogic", "trainer-jobs",
            DateTime.UtcNow.ToString("yyyyMMdd-HHmmss", System.Globalization.CultureInfo.InvariantCulture)
            + "-" + Guid.NewGuid().ToString("N")[..8]);

    /// <summary>
    /// Writes samples that arrived on a wire into a dataset folder under the job's
    /// work folder and starts the trainer on it.
    /// <para>
    /// The trainer only reads folders, so this is the one place samples become one.
    /// Whole groups are held out when the samples came with at least two distinct
    /// groups; rows at random otherwise, which the job records so the trainer can
    /// say in its report that the score may be optimistic.
    /// </para>
    /// </summary>
    /// <param name="dataset">The rows, with one target. <see cref="SampleTable.FromColumns"/> builds one from wires.</param>
    /// <param name="groups">One per row, or null; the model each row came from, for the holdout.</param>
    /// <param name="learner">The method to fit.</param>
    /// <param name="holdoutFraction">Share of groups, or rows, held back to score on.</param>
    /// <param name="outputPath">Where the <c>.onnx</c> is written.</param>
    /// <param name="python">The interpreter, from <see cref="TrainerRuntime.Find"/>.</param>
    /// <param name="seed">Fixes the split and the fit.</param>
    /// <param name="workFolder">Where the job's files go; a fresh temp folder by default.</param>
    /// <exception cref="ArgumentException">The samples, the learner or the interpreter is wrong.</exception>
    /// <exception cref="InvalidOperationException">The process could not be started.</exception>
    public static TrainerProcess StartOnSamples(
        SampleTable dataset, IReadOnlyList<string>? groups, Learner learner, double holdoutFraction,
        string outputPath, string python, int seed = 1, string? workFolder = null)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        ArgumentNullException.ThrowIfNull(learner);

        workFolder ??= NewWorkFolder();
        string datasetFolder = Path.Combine(workFolder, "dataset");
        SampleFolder.Write(datasetFolder, dataset, groups);

        var job = new TrainingJob
        {
            DatasetFolder = datasetFolder,
            Learner = learner,
            HoldoutBy = SampleFolder.DistinctGroups(groups ?? dataset.Groups) >= 2 ? HoldoutBy.Group : HoldoutBy.Row,
            HoldoutFraction = holdoutFraction,
            Seed = seed,
            OutputPath = outputPath,
        };

        return Start(job, python, workFolder);
    }

    private void Append(string? line)
    {
        if (line is null) return;
        lock (_gate)
        {
            if (_stderr.Length < 64 * 1024)
                _stderr.AppendLine(line);
        }
    }

    /// <summary>
    /// Reads whatever the trainer has written since the last call and returns the
    /// latest state. Once the process has exited without a final line, the state is
    /// an error carrying the tail of its stderr, which is where a Python traceback
    /// goes.
    /// </summary>
    public TrainerProgress Poll()
    {
        ObjectDisposedException.ThrowIf(_disposed, this);

        foreach (var line in ReadNewLines())
        {
            var progress = TrainerProgress.FromJsonLine(line);
            Latest = progress;
            if (progress.Finished)
                return Latest;
        }

        if (Latest.Finished || !_process.HasExited)
            return Latest;

        // Exited, and the last line was not a final one. Give the streams a moment
        // to drain — the exit event can land before the last stderr line does.
        _process.WaitForExit();
        var remaining = ReadNewLines().Select(TrainerProgress.FromJsonLine).ToList();
        if (remaining.Count > 0)
        {
            Latest = remaining[^1];
            if (Latest.Finished)
                return Latest;
        }

        string tail;
        lock (_gate)
        {
            tail = Tail(_stderr.ToString(), 12);
        }

        Latest = new TrainerProgress
        {
            Stage = Latest.Stage,
            Error = $"The trainer stopped without reporting a result (exit code {_process.ExitCode})."
                + (tail.Length > 0 ? "\n" + tail : string.Empty),
        };
        return Latest;
    }

    private IEnumerable<string> ReadNewLines()
    {
        if (!File.Exists(ProgressFile))
            yield break;

        string chunk;
        using (var stream = new FileStream(ProgressFile, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete))
        {
            if (stream.Length <= _offset)
                yield break;

            stream.Seek(_offset, SeekOrigin.Begin);
            using var reader = new StreamReader(stream, Encoding.UTF8);
            chunk = reader.ReadToEnd();
            _offset = stream.Length;
        }

        // A line is only a line once its newline has landed; a partial one is held
        // for the next call rather than parsed as broken JSON.
        string text = _partial + chunk;
        int start = 0;
        while (true)
        {
            int end = text.IndexOf('\n', start);
            if (end < 0) break;

            string line = text[start..end].TrimEnd('\r');
            start = end + 1;
            if (line.Length > 0)
                yield return line;
        }

        _partial = text[start..];
    }

    private static string Tail(string text, int lines)
    {
        var all = text.Split('\n', StringSplitOptions.RemoveEmptyEntries);
        return string.Join("\n", all.Skip(Math.Max(0, all.Length - lines)).Select(l => l.TrimEnd('\r')));
    }

    /// <summary>The trainer's stdout and stderr so far, for a log.</summary>
    public string Output
    {
        get
        {
            lock (_gate)
            {
                return _stderr.ToString();
            }
        }
    }

    /// <summary>
    /// Waits for the process to end. The final progress line lands a moment before
    /// the process exits, so a caller that wants <see cref="ExitCode"/> after
    /// <see cref="Poll"/> reports a finish should wait here first.
    /// </summary>
    /// <returns>False if it was still running when the timeout ran out.</returns>
    public bool WaitForExit(TimeSpan timeout)
    {
        ObjectDisposedException.ThrowIf(_disposed, this);
        return _process.WaitForExit((int)Math.Min(int.MaxValue, timeout.TotalMilliseconds));
    }

    /// <summary>Stops the trainer. Safe to call on one that has already finished.</summary>
    public void Cancel()
    {
        if (_disposed || _process.HasExited)
            return;

        try
        {
            _process.Kill(entireProcessTree: true);
            _process.WaitForExit(5000);
        }
        catch (InvalidOperationException)
        {
            // Exited between the check and the kill.
        }

        if (!Latest.Finished)
            Latest = new TrainerProgress { Stage = Latest.Stage, Error = "Cancelled." };
    }

    public void Dispose()
    {
        if (_disposed) return;
        Cancel();
        _disposed = true;
        _process.Dispose();
    }
}
