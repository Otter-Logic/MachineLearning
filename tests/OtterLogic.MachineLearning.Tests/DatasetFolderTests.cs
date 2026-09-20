using OtterLogic.MachineLearning.Data;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// The dataset folder is a contract between Grasshopper sessions months apart, so
/// what is pinned here is what goes wrong over months: the same model written twice,
/// a definition whose columns changed in between, a class met for the first time in
/// the tenth file, and text that CSV has to quote.
/// </summary>
public sealed class DatasetFolderTests : IDisposable
{
    private readonly string _folder = Path.Combine(Path.GetTempPath(), "otterlogic-dataset-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        if (Directory.Exists(_folder))
            Directory.Delete(_folder, recursive: true);
    }

    private static DatasetSchema Schema(string version = "1") => new()
    {
        ExtractorVersion = version,
        Columns = new[]
        {
            DatasetColumn.Id("id"),
            DatasetColumn.Feature("length", "m"),
            DatasetColumn.Feature("angle"),
            DatasetColumn.ClassTarget("kind"),
            DatasetColumn.NumberTarget("cost"),
        },
    };

    private static Dataset Rows(DatasetSchema schema, string[] ids, double[,] features, string[] kinds, double[] costs)
        => Dataset.Create(schema, features,
            new Dictionary<string, double[]> { ["cost"] = costs },
            new Dictionary<string, string[]> { ["kind"] = kinds },
            ids);

    private static Dataset First() => Rows(Schema(),
        new[] { "a", "b", "c" },
        new[,] { { 1.5, 0.1 }, { 2.25, 1.0 / 3.0 }, { 1e-9, -4.0 } },
        new[] { "fixed", "pinned", "fixed" },
        new[] { 10.0, 20.0, 30.0 });

    private static Dataset Second() => Rows(Schema(),
        new[] { "d", "e" },
        new[,] { { 3.0, 0.5 }, { 4.0, 0.25 } },
        new[] { "sliding", "fixed" },
        new[] { 40.0, 50.0 });

    [Fact]
    public void RoundTripsEveryValueExactly()
    {
        DatasetFolder.Write(_folder, "2024-001", First());

        var read = DatasetFolder.Read(_folder);

        // Exactly, not closely: a feature that changes in the last bit between
        // training and inference is a different input to the model.
        Assert.Equal(First().Features, read.Features);
        Assert.Equal(new[] { "a", "b", "c" }, read.Ids);
        Assert.Equal(new[] { "fixed", "pinned", "fixed" }, read.ClassTarget("kind"));
        Assert.Equal(new[] { 10.0, 20.0, 30.0 }, read.NumberTarget("cost"));
        Assert.All(read.Groups, g => Assert.Equal("2024-001", g));
        Assert.Equal("m", read.Schema.Column("length").Unit);
    }

    [Fact]
    public void WritingTheSameModelAgainReplacesItsRows()
    {
        DatasetFolder.Write(_folder, "2024-001", First());
        DatasetFolder.Write(_folder, "2024-001", First());
        DatasetFolder.Write(_folder, "2024-001", Second());

        var read = DatasetFolder.Read(_folder);

        Assert.Equal(2, read.RowCount);
        Assert.Equal(new[] { "d", "e" }, read.Ids);
    }

    [Fact]
    public void AnUnchangedWriteDoesNotTouchTheFile()
    {
        var first = DatasetFolder.Write(_folder, "2024-001", First());
        var stamp = new DateTime(2020, 1, 1, 0, 0, 0, DateTimeKind.Utc);
        File.SetLastWriteTimeUtc(first.Path, stamp);

        DatasetFolder.Write(_folder, "2024-001", First());

        Assert.Equal(stamp, File.GetLastWriteTimeUtc(first.Path));
    }

    [Fact]
    public void ModelsBecomeGroupsAndNewClassesGoOnTheEnd()
    {
        DatasetFolder.Write(_folder, "2024-001", First());
        var second = DatasetFolder.Write(_folder, "2025-002", Second());

        var read = DatasetFolder.Read(_folder);

        Assert.Equal(new[] { ("2024-001", 3), ("2025-002", 2) }, read.RowsPerGroup());

        // "sliding" arrived second, so it is class 2 — and "fixed" is still class
        // 0, as it was for anything trained before the second model existed.
        Assert.Equal(new[] { "fixed", "pinned", "sliding" }, read.Schema.Column("kind").Classes);
        Assert.Equal(new[] { 0, 1, 0, 2, 0 }, read.ClassIndices("kind"));
        Assert.Equal(new[] { ("kind", "sliding") }, second.ClassesAdded);
        Assert.Equal(new[] { ("fixed", 3), ("pinned", 1), ("sliding", 1) }, read.ClassCounts("kind"));
    }

    [Fact]
    public void RefusesRowsWhoseColumnsAreInADifferentOrder()
    {
        DatasetFolder.Write(_folder, "2024-001", First());

        var swapped = new DatasetSchema
        {
            Columns = new[]
            {
                DatasetColumn.Id("id"),
                DatasetColumn.Feature("angle"),
                DatasetColumn.Feature("length"),
                DatasetColumn.ClassTarget("kind"),
                DatasetColumn.NumberTarget("cost"),
            },
        };

        var ex = Assert.Throws<InvalidOperationException>(() => DatasetFolder.Write(_folder, "2025-002",
            Rows(swapped, new[] { "d" }, new[,] { { 0.5, 3.0 } }, new[] { "fixed" }, new[] { 1.0 })));

        Assert.Contains("length", ex.Message);
        Assert.Equal(new[] { "2024-001" }, DatasetFolder.Models(_folder));
    }

    [Fact]
    public void RefusesRowsFromAnotherExtractorVersion()
    {
        DatasetFolder.Write(_folder, "2024-001", First());

        var ex = Assert.Throws<InvalidOperationException>(() => DatasetFolder.Write(_folder, "2025-002",
            Rows(Schema("2"), new[] { "d" }, new[,] { { 3.0, 0.5 } }, new[] { "fixed" }, new[] { 1.0 })));

        Assert.Contains("'1'", ex.Message);
        Assert.Contains("'2'", ex.Message);
    }

    [Fact]
    public void QuotesWhatCsvCannotHoldBare()
    {
        var awkward = Rows(Schema(),
            new[] { "beam, level 2", "the \"long\" one", "two\nlines" },
            new[,] { { 1.0, 2.0 }, { 3.0, 4.0 }, { 5.0, 6.0 } },
            new[] { "fixed, both ends", "pinned", "pinned" },
            new[] { 1.0, 2.0, 3.0 });

        DatasetFolder.Write(_folder, "awkward", awkward);
        var read = DatasetFolder.Read(_folder);

        Assert.Equal(awkward.Ids, read.Ids);
        Assert.Equal(awkward.ClassTarget("kind"), read.ClassTarget("kind"));
    }

    [Fact]
    public void WritesNumbersTheSameWhateverTheMachineIsSetTo()
    {
        var culture = Thread.CurrentThread.CurrentCulture;
        try
        {
            // A decimal comma would turn 1.5 into two columns.
            Thread.CurrentThread.CurrentCulture = new System.Globalization.CultureInfo("de-DE");

            DatasetFolder.Write(_folder, "2024-001", First());
            Assert.Equal(First().Features, DatasetFolder.Read(_folder).Features);
        }
        finally
        {
            Thread.CurrentThread.CurrentCulture = culture;
        }
    }

    [Fact]
    public void NamesTheFileWhenALabelIsNotInTheSchema()
    {
        var written = DatasetFolder.Write(_folder, "2024-001", First());
        File.WriteAllText(written.Path, File.ReadAllText(written.Path).Replace("pinned", "Pinned"));

        var ex = Assert.Throws<InvalidDataException>(() => DatasetFolder.Read(_folder));

        Assert.Contains("models/2024-001.csv", ex.Message);
        Assert.Contains("'Pinned'", ex.Message);
    }

    [Theory]
    [InlineData("")]
    [InlineData("a/b")]
    [InlineData("what?")]
    [InlineData(" padded")]
    [InlineData("dotted.")]
    public void RefusesAModelIdThatCannotBeAFileName(string modelId)
    {
        Assert.Throws<ArgumentException>(() => DatasetFolder.Write(_folder, modelId, First()));
    }

    [Fact]
    public void SaysWhenAFolderIsNotADataset()
    {
        Directory.CreateDirectory(_folder);
        Assert.Throws<FileNotFoundException>(() => DatasetFolder.Read(_folder));
    }
}
