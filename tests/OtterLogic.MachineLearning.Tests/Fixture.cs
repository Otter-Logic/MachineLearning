using System.Text.Json;

namespace OtterLogic.MachineLearning.Tests;

/// <summary>
/// Reads the JSON written by <c>python/fixtures/make_fixtures.py</c>.
/// <para>
/// Deliberately untyped. The fixtures exist so that scikit-learn's answers can
/// be checked against, and a set of mirror record types would be one more thing
/// to keep in step with the generator for no benefit — a missing key throws here
/// just as loudly as a deserialisation failure would.
/// </para>
/// </summary>
internal sealed class Fixture
{
    private readonly JsonElement _root;

    private Fixture(JsonElement root) => _root = root;

    internal static Fixture Load(string name)
    {
        var path = Path.Combine(AppContext.BaseDirectory, "Fixtures", $"{name}.json");
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"Fixture '{name}' is missing. Run: python python/fixtures/make_fixtures.py", path);

        return new Fixture(JsonDocument.Parse(File.ReadAllText(path)).RootElement.Clone());
    }

    internal Fixture Section(string name) => new(_root.GetProperty(name));

    internal double Scalar(string name) => _root.GetProperty(name).GetDouble();

    internal int Int(string name) => _root.GetProperty(name).GetInt32();

    internal bool Flag(string name) => _root.GetProperty(name).GetBoolean();

    internal double[] Vector(string name) =>
        _root.GetProperty(name).EnumerateArray().Select(v => v.GetDouble()).ToArray();

    internal int[] Integers(string name) =>
        _root.GetProperty(name).EnumerateArray().Select(v => v.GetInt32()).ToArray();

    internal double[,] Matrix(string name)
    {
        var rows = _root.GetProperty(name).EnumerateArray()
            .Select(r => r.EnumerateArray().Select(v => v.GetDouble()).ToArray())
            .ToArray();

        var matrix = new double[rows.Length, rows[0].Length];
        for (int i = 0; i < rows.Length; i++)
            for (int j = 0; j < rows[i].Length; j++)
                matrix[i, j] = rows[i][j];

        return matrix;
    }

    /// <summary>A stack of matrices — k covariances, each d x d.</summary>
    internal double[][,] Cube(string name)
    {
        return _root.GetProperty(name).EnumerateArray()
            .Select(slice =>
            {
                var rows = slice.EnumerateArray()
                    .Select(r => r.EnumerateArray().Select(v => v.GetDouble()).ToArray())
                    .ToArray();

                var matrix = new double[rows.Length, rows[0].Length];
                for (int i = 0; i < rows.Length; i++)
                    for (int j = 0; j < rows[i].Length; j++)
                        matrix[i, j] = rows[i][j];

                return matrix;
            })
            .ToArray();
    }

    internal IEnumerable<Fixture> Rows(string name) =>
        _root.GetProperty(name).EnumerateArray().Select(e => new Fixture(e));
}
