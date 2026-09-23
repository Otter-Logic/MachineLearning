using OtterLogic.MachineLearning.Data;
using OtterLogic.MachineLearning.Training;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

public class SampleFolderTests : IDisposable
{
    private readonly string _root = Path.Combine(Path.GetTempPath(), "otterlogic-samples-" + Guid.NewGuid().ToString("N"));

    public SampleFolderTests() => Directory.CreateDirectory(_root);

    public void Dispose()
    {
        try { Directory.Delete(_root, recursive: true); } catch (IOException) { }
    }

    /// <summary>Five rows, one number target, the group column as given.</summary>
    private static Dataset Samples(string[]? groups = null)
    {
        var schema = new DatasetSchema
        {
            Columns = new List<DatasetColumn> { DatasetColumn.Feature("a"), DatasetColumn.Feature("b"), DatasetColumn.NumberTarget("y") },
        };
        var features = new double[5, 2];
        var y = new double[5];
        for (int i = 0; i < 5; i++)
        {
            features[i, 0] = i;
            features[i, 1] = 10 - i;
            y[i] = 2 * i;
        }

        return Dataset.Create(schema, features, new Dictionary<string, double[]> { ["y"] = y }, null, ids: null, groups: groups);
    }

    [Fact]
    public void WritesOneModelPerGroupInTheOrderFirstMet()
    {
        string folder = Path.Combine(_root, "dataset");
        var groups = new[] { "B", "A", "B", "C", "A" };

        var written = SampleFolder.Write(folder, Samples(), groups);

        Assert.Equal(new[] { "B", "A", "C" }, written);
        Assert.Equal(new[] { "A", "B", "C" }, DatasetFolder.Models(folder));

        var back = DatasetFolder.Read(folder);
        Assert.Equal(5, back.RowCount);

        // Every row is in the model its group named, with its own values.
        for (int i = 0; i < back.RowCount; i++)
        {
            int original = (int)back.Features[i, 0];
            Assert.Equal(groups[original], back.Groups[i]);
            Assert.Equal(2.0 * original, back.NumberTarget("y")[i]);
        }
    }

    [Fact]
    public void BlankGroupsGoIntoOneModelCalledSamples()
    {
        string folder = Path.Combine(_root, "dataset");

        var written = SampleFolder.Write(folder, Samples());

        Assert.Equal(new[] { SampleFolder.UngroupedModel }, written);
        Assert.Equal(new[] { "samples" }, DatasetFolder.Models(folder));
        Assert.Equal(5, DatasetFolder.Read(folder).RowCount);
    }

    [Fact]
    public void BlankAndNamedGroupsMix()
    {
        string folder = Path.Combine(_root, "dataset");

        var written = SampleFolder.Write(folder, Samples(), new[] { "A", "", "A", "  ", "B" });

        Assert.Equal(new[] { "A", "samples", "B" }, written);
        Assert.Equal(3, DatasetFolder.Read(folder).RowsPerGroup().Count);
    }

    [Fact]
    public void UsesTheDatasetsOwnGroupsWhenNoneAreGiven()
    {
        string folder = Path.Combine(_root, "dataset");

        var written = SampleFolder.Write(folder, Samples(new[] { "X", "X", "Y", "Y", "Y" }));

        Assert.Equal(new[] { "X", "Y" }, written);
    }

    [Theory]
    [InlineData("2025/003: office?", "2025_003_ office_")]
    [InlineData("a<b>c|d\"e*f\\g", "a_b_c_d_e_f_g")]
    [InlineData("tab\there", "tab_here")]
    [InlineData("  padded  ", "padded")]
    [InlineData("trailing dots...", "trailing dots")]
    [InlineData("2025-003", "2025-003")]
    public void ReplacesWhatCannotBeAFileName(string group, string expected)
    {
        Assert.Equal(expected, SampleFolder.ModelId(group));
    }

    [Theory]
    [InlineData(null)]
    [InlineData("")]
    [InlineData("   ")]
    [InlineData("...")]
    public void ANameWithNothingLeftIsSamples(string? group)
    {
        Assert.Equal(SampleFolder.UngroupedModel, SampleFolder.ModelId(group));
    }

    [Fact]
    public void AReplacedNameStillWrites()
    {
        string folder = Path.Combine(_root, "dataset");

        var written = SampleFolder.Write(folder, Samples(), new[] { "p/1", "p/1", "p/1", "p:2", "p:2" });

        Assert.Equal(new[] { "p_1", "p_2" }, written);
        Assert.Equal(new[] { "p_1", "p_2" }, DatasetFolder.Models(folder));
    }

    [Fact]
    public void CountsDistinctGroupsWithBlanksAsOne()
    {
        Assert.Equal(0, SampleFolder.DistinctGroups(null));
        Assert.Equal(0, SampleFolder.DistinctGroups(Array.Empty<string>()));
        Assert.Equal(1, SampleFolder.DistinctGroups(new[] { "", " ", "" }));
        Assert.Equal(1, SampleFolder.DistinctGroups(new[] { "A", "A" }));
        Assert.Equal(3, SampleFolder.DistinctGroups(new[] { "A", "", "B", "A" }));

        // Two names that become the same file are one group, because they are
        // one model on disk.
        Assert.Equal(1, SampleFolder.DistinctGroups(new[] { "p/1", "p:1" }));
    }

    [Fact]
    public void RefusesTheWrongNumberOfGroups()
    {
        var ex = Assert.Throws<ArgumentException>(() => SampleFolder.Write(Path.Combine(_root, "d"), Samples(), new[] { "A", "B" }));
        Assert.Contains("5 rows and 2 groups", ex.Message);
    }

    [Fact]
    public void RefusesNoFolderAndNoRows()
    {
        Assert.Throws<ArgumentException>(() => SampleFolder.Write("", Samples()));
        Assert.Throws<ArgumentNullException>(() => SampleFolder.Write(Path.Combine(_root, "d"), null!));

        var empty = Samples().Select(Array.Empty<int>());
        Assert.Throws<ArgumentException>(() => SampleFolder.Write(Path.Combine(_root, "d"), empty));
    }
}
