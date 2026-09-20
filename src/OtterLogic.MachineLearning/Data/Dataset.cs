namespace OtterLogic.MachineLearning.Data;

/// <summary>
/// A table of samples held against a <see cref="DatasetSchema"/>: features as one
/// rectangular array, targets by column, and the group every row came from.
/// <para>
/// Columnar rather than a list of row objects because that is the shape everything
/// downstream wants. A model takes the feature block whole, and takes one target at
/// a time; nothing ever asks for "row 40 as an object".
/// </para>
/// </summary>
public sealed class Dataset
{
    private readonly Dictionary<string, double[]> _numbers;
    private readonly Dictionary<string, string[]> _classes;

    private Dataset(
        DatasetSchema schema, double[,] features,
        Dictionary<string, double[]> numbers, Dictionary<string, string[]> classes,
        string[] ids, string[] groups)
    {
        Schema = schema;
        Features = features;
        _numbers = numbers;
        _classes = classes;
        Ids = ids;
        Groups = groups;
    }

    /// <summary>What the columns are. Its class lists cover every label in this table.</summary>
    public DatasetSchema Schema { get; }

    /// <summary>n x f, columns in the order of <see cref="DatasetSchema.Features"/>.</summary>
    public double[,] Features { get; }

    /// <summary>One identifier per row; empty strings when the schema has no identifier column.</summary>
    public string[] Ids { get; }

    /// <summary>
    /// Which group each row belongs to — in a dataset folder, the model it was
    /// exported from. Empty strings on a table that has not been through a folder yet.
    /// <para>
    /// Carried per row because it is what a split has to respect. Samples from one
    /// model are strongly alike, so a model tested on rows whose neighbours it
    /// trained on reports an accuracy it will never reach on a new project.
    /// </para>
    /// </summary>
    public string[] Groups { get; }

    /// <summary>Number of rows.</summary>
    public int RowCount => Features.GetLength(0);

    /// <summary>
    /// Builds a table and checks it: every column the schema names is supplied, every
    /// array is the same length, and every number is finite.
    /// </summary>
    /// <param name="schema">The columns. Classes met in <paramref name="classTargets"/> are added to it.</param>
    /// <param name="features">n x f, in schema feature order.</param>
    /// <param name="numberTargets">One array per number target, keyed by column name. Null when there are none.</param>
    /// <param name="classTargets">One array per class target, keyed by column name. Null when there are none.</param>
    /// <param name="ids">One per row. Required exactly when the schema has an identifier column.</param>
    /// <param name="groups">One per row, or null for a table not yet written to a folder.</param>
    public static Dataset Create(
        DatasetSchema schema, double[,] features,
        IReadOnlyDictionary<string, double[]>? numberTargets = null,
        IReadOnlyDictionary<string, string[]>? classTargets = null,
        string[]? ids = null, string[]? groups = null)
    {
        ArgumentNullException.ThrowIfNull(schema);
        ArgumentNullException.ThrowIfNull(features);
        schema.Validate();

        int n = features.GetLength(0);
        var featureColumns = schema.Features;

        if (features.GetLength(1) != featureColumns.Count)
            throw new ArgumentException(
                $"The schema names {featureColumns.Count} features and each row holds {features.GetLength(1)} values.",
                nameof(features));

        for (int i = 0; i < n; i++)
            for (int j = 0; j < featureColumns.Count; j++)
                if (!double.IsFinite(features[i, j]))
                    throw new ArgumentException(
                        $"Row {i} holds {features[i, j]} for '{featureColumns[j].Name}'. Every feature must be a "
                        + "finite number — decide what a missing value means before it gets here.",
                        nameof(features));

        var numbers = new Dictionary<string, double[]>(StringComparer.OrdinalIgnoreCase);
        var classes = new Dictionary<string, string[]>(StringComparer.OrdinalIgnoreCase);
        var columns = schema.Columns.ToArray();

        for (int c = 0; c < columns.Length; c++)
        {
            var column = columns[c];
            if (column.Role != ColumnRole.Target)
                continue;

            if (column.Kind == ColumnKind.Number)
            {
                var values = Find(numberTargets, column.Name)
                    ?? throw new ArgumentException($"No values were supplied for number target '{column.Name}'.");
                RequireLength(values.Length, n, column.Name);

                for (int i = 0; i < n; i++)
                    if (!double.IsFinite(values[i]))
                        throw new ArgumentException($"Row {i} holds {values[i]} for target '{column.Name}'.");

                numbers[column.Name] = values;
            }
            else
            {
                var labels = Find(classTargets, column.Name)
                    ?? throw new ArgumentException($"No labels were supplied for class target '{column.Name}'.");
                RequireLength(labels.Length, n, column.Name);

                for (int i = 0; i < n; i++)
                    if (string.IsNullOrWhiteSpace(labels[i]))
                        throw new ArgumentException(
                            $"Row {i} has no label for '{column.Name}'. A row with nothing to learn from does "
                            + "not belong in a training set — leave it out.");

                // Order first seen, appended to whatever the schema already had, so
                // an index handed out earlier never moves.
                var known = new List<string>(column.Classes);
                var lookup = new HashSet<string>(known, StringComparer.Ordinal);
                foreach (var label in labels)
                    if (lookup.Add(label))
                        known.Add(label);

                columns[c] = column with { Classes = known };
                classes[column.Name] = labels;
            }
        }

        bool wantsIds = schema.Id is not null;
        if (wantsIds && ids is null)
            throw new ArgumentException($"The schema has an identifier column, '{schema.Id!.Name}', and no identifiers were supplied.");
        if (!wantsIds && ids is not null)
            throw new ArgumentException("Identifiers were supplied, but the schema has no identifier column.");
        if (ids is not null)
            RequireLength(ids.Length, n, schema.Id!.Name);
        if (groups is not null)
            RequireLength(groups.Length, n, "groups");

        return new Dataset(
            schema with { Columns = columns }, features, numbers, classes,
            ids ?? Blank(n), groups ?? Blank(n));
    }

    /// <summary>
    /// Builds a table from what a front-end has to hand: names, a block of features,
    /// and targets still as text.
    /// <para>
    /// Here rather than in an adaptor because both the turning of text into numbers
    /// and the wording when it fails are the same whichever front-end is asking.
    /// Which targets are numbers is stated, never guessed — see
    /// <see cref="ColumnKind.Category"/> for why a column of 0 and 1 must not be
    /// taken for a quantity.
    /// </para>
    /// </summary>
    /// <param name="featureNames">One per feature column.</param>
    /// <param name="features">n x f.</param>
    /// <param name="targetNames">One per target column. Empty for a table with nothing to predict yet.</param>
    /// <param name="targets">n x t, as text. Ignored when there are no target names.</param>
    /// <param name="targetIsNumber">One per target: true for a quantity, false for a class.</param>
    /// <param name="ids">One per row, or null for no identifier column.</param>
    /// <param name="extractorVersion">See <see cref="DatasetSchema.ExtractorVersion"/>.</param>
    /// <param name="idName">What to call the identifier column.</param>
    public static Dataset FromColumns(
        IReadOnlyList<string> featureNames, double[,] features,
        IReadOnlyList<string> targetNames, string[,]? targets, IReadOnlyList<bool> targetIsNumber,
        string[]? ids = null, string extractorVersion = "1", string idName = "id")
    {
        ArgumentNullException.ThrowIfNull(featureNames);
        ArgumentNullException.ThrowIfNull(features);
        ArgumentNullException.ThrowIfNull(targetNames);
        ArgumentNullException.ThrowIfNull(targetIsNumber);

        int n = features.GetLength(0);

        if (featureNames.Count != features.GetLength(1))
            throw new ArgumentException(
                $"There are {featureNames.Count} feature names and each sample holds {features.GetLength(1)} values. "
                + "Every column needs a name — it is what catches two of them being swapped later.");

        if (targetIsNumber.Count != targetNames.Count)
            throw new ArgumentException(
                $"There are {targetNames.Count} targets and {targetIsNumber.Count} number-or-class flags.");

        if (targetNames.Count > 0)
        {
            if (targets is null)
                throw new ArgumentException("Targets are named but no target values were supplied.");
            if (targets.GetLength(0) != n)
                throw new ArgumentException($"There are {n} samples and target values for {targets.GetLength(0)}.");
            if (targets.GetLength(1) != targetNames.Count)
                throw new ArgumentException(
                    $"There are {targetNames.Count} target names and each sample holds {targets.GetLength(1)} target values.");
        }

        var columns = new List<DatasetColumn>();
        if (ids is not null)
            columns.Add(DatasetColumn.Id(idName));
        columns.AddRange(featureNames.Select(name => DatasetColumn.Feature(name)));

        var numbers = new Dictionary<string, double[]>();
        var classes = new Dictionary<string, string[]>();

        for (int t = 0; t < targetNames.Count; t++)
        {
            string name = targetNames[t];
            if (targetIsNumber[t])
            {
                var values = new double[n];
                for (int i = 0; i < n; i++)
                    if (!Csv.TryNumber(targets![i, t] ?? string.Empty, out values[i]))
                        throw new ArgumentException(
                            $"Target '{name}' is marked as a number and sample {i} holds '{targets[i, t]}'. "
                            + "If it is a class, say so rather than marking it a number.");

                columns.Add(DatasetColumn.NumberTarget(name));
                numbers[name] = values;
            }
            else
            {
                var labels = new string[n];
                for (int i = 0; i < n; i++)
                    labels[i] = (targets![i, t] ?? string.Empty).Trim();

                columns.Add(DatasetColumn.ClassTarget(name));
                classes[name] = labels;
            }
        }

        var schema = new DatasetSchema { ExtractorVersion = extractorVersion, Columns = columns };
        return Create(schema, features, numbers, classes, ids);
    }

    /// <summary>
    /// The targets as text again, n x t in schema order — the shape
    /// <see cref="FromColumns"/> takes, for a front-end that shows them or passes them on.
    /// </summary>
    public string[,] TargetsAsText()
    {
        var columns = Schema.Targets;
        var text = new string[RowCount, columns.Count];

        for (int t = 0; t < columns.Count; t++)
        {
            bool number = columns[t].Kind == ColumnKind.Number;
            for (int i = 0; i < RowCount; i++)
                text[i, t] = number ? Csv.Number(_numbers[columns[t].Name][i]) : _classes[columns[t].Name][i];
        }

        return text;
    }

    /// <summary>
    /// What is in the table, in the few lines worth reading before fitting anything:
    /// how much, from where, how each class target is balanced, and any feature that
    /// never changes.
    /// </summary>
    public IReadOnlyList<string> Describe()
    {
        var lines = new List<string>();
        var groups = RowsPerGroup();

        lines.Add($"{RowCount} rows, {Schema.Features.Count} features, {Schema.Targets.Count} targets, "
            + $"extractor version {Schema.ExtractorVersion}.");

        if (groups.Count > 1 || (groups.Count == 1 && groups[0].Group.Length > 0))
        {
            lines.Add($"{groups.Count} group(s):");
            lines.AddRange(groups.Select(g => $"  {g.Group}: {g.Rows} rows"));
        }

        foreach (var target in Schema.Targets)
        {
            if (target.Kind == ColumnKind.Number)
            {
                var values = _numbers[target.Name];
                lines.Add(FormattableString.Invariant($"Target '{target.Name}' (number): {values.Min():G6} to {values.Max():G6}."));
                continue;
            }

            var counts = ClassCounts(target.Name);
            lines.Add($"Target '{target.Name}' (class): " + string.Join(", ", counts.Select(c =>
                FormattableString.Invariant($"{c.Class} {c.Rows} ({100.0 * c.Rows / RowCount:0.#}%)"))) + ".");

            double largest = (double)counts.Max(c => c.Rows) / RowCount;
            if (largest >= 0.8)
                lines.Add(FormattableString.Invariant($"  Always answering the commonest class would score {100.0 * largest:0.#}% on this. ")
                    + "Judge a model by balanced accuracy, not accuracy.");
        }

        var constant = ConstantFeatures();
        if (constant.Count > 0)
            lines.Add($"Never changes, so carries nothing to learn from: {string.Join(", ", constant)}.");

        return lines;
    }

    /// <summary>The values of a number target.</summary>
    public double[] NumberTarget(string name)
    {
        var column = RequireTarget(name, ColumnKind.Number);
        return _numbers[column.Name];
    }

    /// <summary>The labels of a class target, as written.</summary>
    public string[] ClassTarget(string name)
    {
        var column = RequireTarget(name, ColumnKind.Category);
        return _classes[column.Name];
    }

    /// <summary>
    /// The labels of a class target as indices into that column's
    /// <see cref="DatasetColumn.Classes"/> — what a classifier is trained on.
    /// </summary>
    public int[] ClassIndices(string name)
    {
        var column = RequireTarget(name, ColumnKind.Category);
        var index = new Dictionary<string, int>(StringComparer.Ordinal);
        for (int k = 0; k < column.Classes.Count; k++)
            index[column.Classes[k]] = k;

        return _classes[column.Name].Select(label => index[label]).ToArray();
    }

    /// <summary>The rows at <paramref name="rows"/>, in that order, as a table of their own.</summary>
    public Dataset Select(IReadOnlyList<int> rows)
    {
        ArgumentNullException.ThrowIfNull(rows);

        int f = Features.GetLength(1);
        var features = new double[rows.Count, f];
        for (int i = 0; i < rows.Count; i++)
        {
            if (rows[i] < 0 || rows[i] >= RowCount)
                throw new ArgumentOutOfRangeException(nameof(rows), rows[i], $"There are {RowCount} rows.");

            for (int j = 0; j < f; j++)
                features[i, j] = Features[rows[i], j];
        }

        // The schema goes across whole, classes included: a subset that happens to
        // hold no sample of a class must still number the others the same way.
        return new Dataset(
            Schema, features,
            _numbers.ToDictionary(p => p.Key, p => Pick(p.Value, rows), StringComparer.OrdinalIgnoreCase),
            _classes.ToDictionary(p => p.Key, p => Pick(p.Value, rows), StringComparer.OrdinalIgnoreCase),
            Pick(Ids, rows), Pick(Groups, rows));
    }

    /// <summary>Row count per group, groups in the order first met.</summary>
    public IReadOnlyList<(string Group, int Rows)> RowsPerGroup()
        => Groups.GroupBy(g => g, StringComparer.Ordinal).Select(g => (g.Key, g.Count())).ToArray();

    /// <summary>
    /// Row count per class of a class target, in schema class order, zeros included.
    /// <para>
    /// Worth looking at before training anything. A target that is nine-tenths one
    /// class makes ninety per cent accuracy the score of a model that has learned
    /// nothing, and a class spelt two ways shows up here as two classes.
    /// </para>
    /// </summary>
    public IReadOnlyList<(string Class, int Rows)> ClassCounts(string name)
    {
        var column = RequireTarget(name, ColumnKind.Category);
        var counts = new int[column.Classes.Count];
        foreach (int k in ClassIndices(name))
            counts[k]++;

        return column.Classes.Select((label, k) => (label, counts[k])).ToArray();
    }

    /// <summary>
    /// Names of the features that hold one value on every row. They carry nothing to
    /// learn from, and usually mean an input upstream is wired to a constant.
    /// </summary>
    public IReadOnlyList<string> ConstantFeatures()
    {
        var names = new List<string>();
        var columns = Schema.Features;

        for (int j = 0; j < columns.Count; j++)
        {
            bool constant = true;
            for (int i = 1; i < RowCount && constant; i++)
                constant = Features[i, j] == Features[0, j];

            if (constant && RowCount > 1)
                names.Add(columns[j].Name);
        }

        return names;
    }

    private DatasetColumn RequireTarget(string name, ColumnKind kind)
    {
        var column = Schema.Column(name);
        if (column.Role != ColumnRole.Target || column.Kind != kind)
            throw new ArgumentException(
                $"'{column.Name}' is a {column.Kind} {column.Role}, not a {kind} target.", nameof(name));

        return column;
    }

    private static T[]? Find<T>(IReadOnlyDictionary<string, T[]>? source, string name)
    {
        if (source is null)
            return null;

        foreach (var pair in source)
            if (string.Equals(pair.Key, name, StringComparison.OrdinalIgnoreCase))
                return pair.Value;

        return null;
    }

    private static void RequireLength(int actual, int expected, string what)
    {
        if (actual != expected)
            throw new ArgumentException($"'{what}' has {actual} values for {expected} rows.");
    }

    private static T[] Pick<T>(T[] source, IReadOnlyList<int> rows)
    {
        var picked = new T[rows.Count];
        for (int i = 0; i < rows.Count; i++)
            picked[i] = source[rows[i]];

        return picked;
    }

    private static string[] Blank(int n)
    {
        var blank = new string[n];
        Array.Fill(blank, string.Empty);
        return blank;
    }
}
