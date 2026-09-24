using OtterLogic.Dataset.Data;

namespace OtterLogic.MachineLearning.Training;

/// <summary>
/// Writes samples that arrived on a wire into a dataset folder the trainer can
/// read, one model per group.
/// <para>
/// The trainer reads folders and nothing else, on purpose: one format to read,
/// one set of checks, and a job a person can re-run from a terminal by pointing at
/// the folder. So samples wired straight into a component go through a folder
/// too — a temporary one under the job's work folder — and the trainer never knows
/// the difference.
/// </para>
/// </summary>
public static class SampleFolder
{
    /// <summary>The one model every row goes into when the samples came with no groups.</summary>
    public const string UngroupedModel = "samples";

    /// <summary>
    /// Writes <paramref name="dataset"/> into <paramref name="folder"/>, one model per
    /// distinct group.
    /// </summary>
    /// <param name="folder">Created if it is not there. Expected empty: an existing schema there is checked against, not replaced.</param>
    /// <param name="dataset">The rows.</param>
    /// <param name="groups">
    /// One per row, or null to use the dataset's own. Blank groups all go into one
    /// model called <see cref="UngroupedModel"/>; a group name that cannot be a file
    /// name has those characters replaced.
    /// </param>
    /// <returns>The model IDs written, in the order first met.</returns>
    public static IReadOnlyList<string> Write(string folder, SampleTable dataset, IReadOnlyList<string>? groups = null)
    {
        ArgumentNullException.ThrowIfNull(dataset);
        if (string.IsNullOrWhiteSpace(folder))
            throw new ArgumentException("No folder was given.", nameof(folder));
        if (dataset.RowCount == 0)
            throw new ArgumentException("There are no rows to write.", nameof(dataset));

        groups ??= dataset.Groups;
        if (groups.Count != dataset.RowCount)
            throw new ArgumentException(
                $"There are {dataset.RowCount} rows and {groups.Count} groups; give one per row or none.", nameof(groups));

        var rowsByModel = new Dictionary<string, List<int>>(StringComparer.Ordinal);
        var order = new List<string>();
        for (int i = 0; i < groups.Count; i++)
        {
            string model = ModelId(groups[i]);
            if (!rowsByModel.TryGetValue(model, out var rows))
            {
                rows = new List<int>();
                rowsByModel[model] = rows;
                order.Add(model);
            }

            rows.Add(i);
        }

        Directory.CreateDirectory(folder);
        foreach (string model in order)
            DatasetFolder.Write(folder, model, dataset.Select(rowsByModel[model]));

        return order;
    }

    /// <summary>How many distinct groups <paramref name="groups"/> names, blanks counted as one.</summary>
    public static int DistinctGroups(IReadOnlyList<string>? groups)
        => groups is null ? 0 : groups.Select(ModelId).Distinct(StringComparer.Ordinal).Count();

    /// <summary>A group name as the file name its rows are written under.</summary>
    public static string ModelId(string? group)
    {
        if (string.IsNullOrWhiteSpace(group))
            return UngroupedModel;

        // The same fixed list DatasetFolder refuses, rather than the platform's own:
        // an ID written on one machine must open on every machine sharing the folder.
        var name = group.Trim().Select(c => "<>:\"/\\|?*".Contains(c) || char.IsControl(c) ? '_' : c).ToArray();
        string id = new string(name).TrimEnd('.', ' ');
        return id.Length == 0 ? UngroupedModel : id;
    }
}
