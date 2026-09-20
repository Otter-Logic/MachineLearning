using OtterLogic.MachineLearning.Data;
using Xunit;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>The table and its schema, before any of it reaches a disk.</summary>
public sealed class DatasetTests
{
    private static readonly DatasetSchema Schema = new()
    {
        Columns = new[]
        {
            DatasetColumn.Feature("a"),
            DatasetColumn.Feature("b"),
            DatasetColumn.ClassTarget("kind"),
        },
    };

    private static Dataset Table(string[] kinds, double[,]? features = null) => Dataset.Create(
        Schema, features ?? new[,] { { 1.0, 7.0 }, { 2.0, 7.0 }, { 3.0, 7.0 } },
        classTargets: new Dictionary<string, string[]> { ["kind"] = kinds });

    [Fact]
    public void ClassesAreNumberedInTheOrderFirstSeen()
    {
        var table = Table(new[] { "pinned", "fixed", "pinned" });

        Assert.Equal(new[] { "pinned", "fixed" }, table.Schema.Column("kind").Classes);
        Assert.Equal(new[] { 0, 1, 0 }, table.ClassIndices("kind"));
    }

    [Fact]
    public void ASubsetKeepsTheNumberingOfTheWhole()
    {
        var subset = Table(new[] { "pinned", "fixed", "pinned" }).Select(new[] { 1 });

        // Row 1 is the only "fixed", and it is still class 1 — not class 0 of a
        // table that happens to contain nothing else.
        Assert.Equal(new[] { 1 }, subset.ClassIndices("kind"));
        Assert.Equal(new[,] { { 2.0, 7.0 } }, subset.Features);
    }

    [Fact]
    public void ReportsFeaturesThatNeverChange()
    {
        Assert.Equal(new[] { "b" }, Table(new[] { "x", "y", "x" }).ConstantFeatures());
    }

    [Fact]
    public void RefusesANumberThatIsNotFinite()
    {
        var ex = Assert.Throws<ArgumentException>(() =>
            Table(new[] { "x", "y" }, new[,] { { 1.0, 2.0 }, { double.NaN, 2.0 } }));

        Assert.Contains("'a'", ex.Message);
    }

    [Fact]
    public void RefusesARowWithNoLabel()
    {
        Assert.Throws<ArgumentException>(() => Table(new[] { "x", " ", "y" }));
    }

    [Fact]
    public void RefusesACategoryFeature()
    {
        var schema = new DatasetSchema
        {
            Columns = new[] { new DatasetColumn { Name = "a", Role = ColumnRole.Feature, Kind = ColumnKind.Category } },
        };

        Assert.Throws<ArgumentException>(schema.Validate);
    }

    [Fact]
    public void RefusesTwoColumnsWithOneName()
    {
        var schema = new DatasetSchema
        {
            Columns = new[] { DatasetColumn.Feature("Length"), DatasetColumn.Feature("length") },
        };

        Assert.Throws<ArgumentException>(schema.Validate);
    }

    [Fact]
    public void BuildsFromNamesAndTextTheWayAFrontEndHoldsThem()
    {
        var table = Dataset.FromColumns(
            new[] { "a", "b" }, new[,] { { 1.0, 2.0 }, { 3.0, 4.0 } },
            new[] { "released", "cost" }, new[,] { { "1", "10.5" }, { "0", "2e1" } },
            new[] { false, true },
            ids: new[] { "m1", "m2" });

        // "1" and "0" stay classes because they were declared so, not numbers
        // because they look like them.
        Assert.Equal(new[] { "1", "0" }, table.ClassTarget("released"));
        Assert.Equal(new[] { 10.5, 20.0 }, table.NumberTarget("cost"));
        Assert.Equal(new[,] { { "1", "10.5" }, { "0", "20" } }, table.TargetsAsText());
        Assert.Equal("id", table.Schema.Id!.Name);
    }

    [Fact]
    public void ANumberTargetHoldingTextIsNamed()
    {
        var ex = Assert.Throws<ArgumentException>(() => Dataset.FromColumns(
            new[] { "a" }, new[,] { { 1.0 }, { 2.0 } },
            new[] { "kind" }, new[,] { { "3" }, { "fixed" } }, new[] { true }));

        Assert.Contains("'kind'", ex.Message);
        Assert.Contains("'fixed'", ex.Message);
    }

    [Fact]
    public void TheDescriptionWarnsWhenOneClassDominates()
    {
        var kinds = Enumerable.Repeat("fixed", 9).Append("pinned").ToArray();
        var features = new double[10, 2];
        for (int i = 0; i < 10; i++)
            (features[i, 0], features[i, 1]) = (i, 7.0);

        var lines = Table(kinds, features).Describe();

        Assert.Contains(lines, l => l.Contains("fixed 9 (90%)"));
        Assert.Contains(lines, l => l.Contains("balanced accuracy"));
        Assert.Contains(lines, l => l.Contains("Never changes") && l.Contains("b"));
    }

    [Fact]
    public void TheSchemaSurvivesJson()
    {
        var schema = Table(new[] { "pinned", "fixed", "pinned" }).Schema with { ExtractorVersion = "2.1" };

        var back = DatasetSchema.FromJson(schema.ToJson());

        Assert.Equal("2.1", back.ExtractorVersion);
        Assert.Equal(schema.Columns.Select(c => (c.Name, c.Role, c.Kind)), back.Columns.Select(c => (c.Name, c.Role, c.Kind)));
        Assert.Equal(new[] { "pinned", "fixed" }, back.Column("kind").Classes);

        // Words, not integers: the Python trainer reads this file too.
        Assert.Contains("\"role\": \"target\"", schema.ToJson());
        Assert.Contains("\"kind\": \"category\"", schema.ToJson());
    }
}
