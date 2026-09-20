using System.Text;

namespace OtterLogic.MachineLearning.Data;

/// <summary>
/// A dataset on disk: one <c>schema.json</c>, and under <c>models/</c> one CSV per
/// model the rows came from.
/// <code>
/// my-dataset/
///   schema.json
///   models/2024-017.csv
///   models/2025-003.csv
/// </code>
/// <para>
/// One file per model rather than one growing table, for three reasons that all come
/// from how the data is gathered — a model at a time, from Grasshopper, over months.
/// Grasshopper re-solves on every upstream change, so an append would add the same
/// rows again on every slider move; a file keyed by model is simply overwritten. The
/// file name <em>is</em> the group, so the split that has to be by project needs
/// nothing extra from the user. And curating the set is deleting a file.
/// </para>
/// <para>
/// CSV because every tool a user might check the data with opens it, and because at
/// the sizes involved — a hundred thousand rows by forty columns is about 30 MB —
/// nothing faster is needed yet.
/// </para>
/// </summary>
public static class DatasetFolder
{
    private const string SchemaFile = "schema.json";
    private const string ModelsFolder = "models";

    /// <summary>
    /// Writes one model's rows, replacing whatever that model wrote before, and
    /// creates or checks the folder's schema.
    /// </summary>
    /// <param name="folder">The dataset folder. Created if it is not there.</param>
    /// <param name="modelId">
    /// Names the file and becomes the group of every row in it. Use something that
    /// identifies the project for good — a job number — because writing the same
    /// model under two names puts its rows in twice, on both sides of a split.
    /// </param>
    /// <param name="dataset">The rows. Its <see cref="Dataset.Groups"/> are ignored; the group is the model.</param>
    /// <returns>What was written, and any classes the folder had not seen before.</returns>
    /// <exception cref="InvalidOperationException">The rows do not match the schema already in the folder.</exception>
    public static DatasetWriteResult Write(string folder, string modelId, Dataset dataset)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        RequireFolder(folder);
        RequireModelId(modelId);

        if (dataset.RowCount == 0)
            throw new ArgumentException("There are no rows to write.", nameof(dataset));

        string schemaPath = Path.Combine(folder, SchemaFile);
        DatasetSchema? existing = File.Exists(schemaPath) ? DatasetSchema.FromJson(File.ReadAllText(schemaPath)) : null;
        DatasetSchema schema = existing?.Merge(dataset.Schema) ?? dataset.Schema;

        var added = existing?.ClassesAddedBy(schema) ?? Array.Empty<(string, string)>();

        Directory.CreateDirectory(Path.Combine(folder, ModelsFolder));
        string path = ModelPath(folder, modelId);

        // Data first, schema second. If the second write is interrupted the folder
        // holds a label the schema does not list, which the next read refuses by
        // name; the other order could leave a schema promising rows that never
        // landed, which nothing would notice.
        ReplaceFile(path, RenderRows(dataset));

        string json = schema.ToJson();
        if (existing is null || existing.ToJson() != json)
            ReplaceFile(schemaPath, json);

        return new DatasetWriteResult(path, dataset.RowCount, added);
    }

    /// <summary>The folder's schema.</summary>
    public static DatasetSchema ReadSchema(string folder)
    {
        RequireFolder(folder);

        string path = Path.Combine(folder, SchemaFile);
        if (!File.Exists(path))
            throw new FileNotFoundException(
                $"'{folder}' is not a dataset folder: it has no {SchemaFile}. One is created the first time a "
                + "model is written there.", path);

        return DatasetSchema.FromJson(File.ReadAllText(path));
    }

    /// <summary>The models in the folder, in the order they are read.</summary>
    public static IReadOnlyList<string> Models(string folder)
    {
        RequireFolder(folder);

        string models = Path.Combine(folder, ModelsFolder);
        if (!Directory.Exists(models))
            return Array.Empty<string>();

        // Ordinal, so the row order of a read is the same on every machine and a
        // seeded split downstream picks the same projects everywhere.
        return Directory.EnumerateFiles(models, "*.csv")
            .Select(file => Path.GetFileNameWithoutExtension(file)!)
            .OrderBy(id => id, StringComparer.Ordinal)
            .ToArray();
    }

    /// <summary>Every row of every model, each row's group set to the model it came from.</summary>
    /// <exception cref="InvalidDataException">A file does not match the schema. The message names the file.</exception>
    public static Dataset Read(string folder)
    {
        var schema = ReadSchema(folder);
        var models = Models(folder);

        if (models.Count == 0)
            throw new InvalidDataException($"'{folder}' has a schema and no models under {ModelsFolder}/.");

        var columns = schema.Columns;
        var featureIndex = Positions(columns, c => c.Role == ColumnRole.Feature);
        int? idIndex = Positions(columns, c => c.Role == ColumnRole.Id).Cast<int?>().FirstOrDefault();

        var features = new List<double[]>();
        var ids = new List<string>();
        var groups = new List<string>();
        var numbers = columns.Where(c => c.Role == ColumnRole.Target && c.Kind == ColumnKind.Number)
            .ToDictionary(c => c.Name, _ => new List<double>());
        var labels = columns.Where(c => c.Role == ColumnRole.Target && c.Kind == ColumnKind.Category)
            .ToDictionary(c => c.Name, _ => new List<string>());

        foreach (var model in models)
        {
            string file = $"{ModelsFolder}/{model}.csv";
            List<string[]> records;
            using (var reader = new StreamReader(ModelPath(folder, model), Encoding.UTF8))
            {
                try
                {
                    records = Csv.Read(reader);
                }
                catch (InvalidDataException ex)
                {
                    throw new InvalidDataException($"{file}: {ex.Message}", ex);
                }
            }

            if (records.Count == 0)
                throw new InvalidDataException($"{file} is empty.");

            var header = records[0];
            if (header.Length != columns.Count
                || header.Where((name, j) => !string.Equals(name, columns[j].Name, StringComparison.OrdinalIgnoreCase)).Any())
                throw new InvalidDataException(
                    $"{file} has the columns [{string.Join(", ", header)}] and schema.json says "
                    + $"[{string.Join(", ", columns.Select(c => c.Name))}]. Re-export that model, or remove the file.");

            for (int r = 1; r < records.Count; r++)
            {
                var record = records[r];
                if (record.Length != columns.Count)
                    throw new InvalidDataException(
                        $"{file}, line {r + 1}: {record.Length} values for {columns.Count} columns.");

                var row = new double[featureIndex.Length];
                for (int f = 0; f < featureIndex.Length; f++)
                    row[f] = ParseNumber(record[featureIndex[f]], columns[featureIndex[f]], file, r);
                features.Add(row);

                for (int j = 0; j < columns.Count; j++)
                {
                    var column = columns[j];
                    if (column.Role != ColumnRole.Target)
                        continue;

                    if (column.Kind == ColumnKind.Number)
                    {
                        numbers[column.Name].Add(ParseNumber(record[j], column, file, r));
                        continue;
                    }

                    // The schema lists every class on purpose. A label it does not
                    // know is a hand-edited file or a half-finished write, and
                    // taking it in quietly would hand out a class index that
                    // depends on which file happened to be read first.
                    if (!column.Classes.Contains(record[j], StringComparer.Ordinal))
                        throw new InvalidDataException(
                            $"{file}, line {r + 1}: '{record[j]}' is not one of the classes schema.json lists for "
                            + $"'{column.Name}' ({string.Join(", ", column.Classes)}). Re-export that model so "
                            + "the class is recorded.");

                    labels[column.Name].Add(record[j]);
                }

                ids.Add(idIndex is int at ? record[at] : string.Empty);
                groups.Add(model);
            }
        }

        var block = new double[features.Count, featureIndex.Length];
        for (int i = 0; i < features.Count; i++)
            for (int j = 0; j < featureIndex.Length; j++)
                block[i, j] = features[i][j];

        return Dataset.Create(
            schema, block,
            numbers.ToDictionary(p => p.Key, p => p.Value.ToArray()),
            labels.ToDictionary(p => p.Key, p => p.Value.ToArray()),
            idIndex is null ? null : ids.ToArray(),
            groups.ToArray());
    }

    private static string RenderRows(Dataset dataset)
    {
        var columns = dataset.Schema.Columns;
        var numberTargets = columns.Where(c => c.Role == ColumnRole.Target && c.Kind == ColumnKind.Number)
            .ToDictionary(c => c.Name, c => dataset.NumberTarget(c.Name));
        var classTargets = columns.Where(c => c.Role == ColumnRole.Target && c.Kind == ColumnKind.Category)
            .ToDictionary(c => c.Name, c => dataset.ClassTarget(c.Name));

        using var writer = new StringWriter();
        Csv.WriteRow(writer, columns.Select(c => c.Name));

        var fields = new string[columns.Count];
        for (int i = 0; i < dataset.RowCount; i++)
        {
            int feature = 0;
            for (int j = 0; j < columns.Count; j++)
            {
                var column = columns[j];
                fields[j] = column.Role switch
                {
                    ColumnRole.Feature => Csv.Number(dataset.Features[i, feature++]),
                    ColumnRole.Id => dataset.Ids[i],
                    _ when column.Kind == ColumnKind.Number => Csv.Number(numberTargets[column.Name][i]),
                    _ => classTargets[column.Name][i],
                };
            }

            Csv.WriteRow(writer, fields);
        }

        return writer.ToString();
    }

    /// <summary>
    /// Writes beside the target and moves over it, so a reader — the trainer, or a
    /// sync client — never finds half a file. Skipped entirely when the content is
    /// unchanged, because Grasshopper will ask for the same write hundreds of times
    /// and each one would otherwise be an upload.
    /// </summary>
    private static void ReplaceFile(string path, string content)
    {
        var encoding = new UTF8Encoding(encoderShouldEmitUTF8Identifier: false);

        if (File.Exists(path) && File.ReadAllText(path, encoding) == content)
            return;

        string staging = path + ".writing";
        File.WriteAllText(staging, content, encoding);
        File.Move(staging, path, overwrite: true);
    }

    private static double ParseNumber(string text, DatasetColumn column, string file, int record)
        => Csv.TryNumber(text, out double value)
            ? value
            : throw new InvalidDataException(
                $"{file}, line {record + 1}: '{text}' is not a finite number, in column '{column.Name}'.");

    private static int[] Positions(IReadOnlyList<DatasetColumn> columns, Func<DatasetColumn, bool> wanted)
        => Enumerable.Range(0, columns.Count).Where(j => wanted(columns[j])).ToArray();

    private static string ModelPath(string folder, string modelId)
        => Path.Combine(folder, ModelsFolder, modelId + ".csv");

    private static void RequireFolder(string folder)
    {
        if (string.IsNullOrWhiteSpace(folder))
            throw new ArgumentException("No dataset folder was given.", nameof(folder));
    }

    private static void RequireModelId(string modelId)
    {
        if (string.IsNullOrWhiteSpace(modelId))
            throw new ArgumentException("No model ID was given. It names the file and becomes the group.", nameof(modelId));

        if (modelId != modelId.Trim() || modelId.EndsWith('.'))
            throw new ArgumentException($"Model ID '{modelId}' starts or ends with a space, or ends with a dot.", nameof(modelId));

        // Checked against a fixed list rather than Path.GetInvalidFileNameChars,
        // which differs by platform: an ID accepted on one machine must be one
        // every machine sharing the folder can open.
        int bad = modelId.IndexOfAny("<>:\"/\\|?*".ToCharArray());
        if (bad >= 0 || modelId.Any(char.IsControl))
            throw new ArgumentException(
                $"Model ID '{modelId}' cannot be a file name. Avoid < > : \" / \\ | ? * and control characters.",
                nameof(modelId));

        if (modelId.Length > 100)
            throw new ArgumentException("Model ID is longer than 100 characters.", nameof(modelId));
    }
}

/// <summary>What <see cref="DatasetFolder.Write"/> did.</summary>
/// <param name="Path">The CSV written.</param>
/// <param name="Rows">Rows in it.</param>
/// <param name="ClassesAdded">
/// Classes this write introduced to a folder that already had a schema. Worth
/// surfacing every time: a genuinely new class is news, and a class that differs
/// from an existing one by a capital letter is a typo about to become a category.
/// </param>
public sealed record DatasetWriteResult(
    string Path, int Rows, IReadOnlyList<(string Column, string Class)> ClassesAdded);
