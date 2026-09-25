namespace OtterLogic.MachineLearning.Inference.Export;

/// <summary>
/// One node of a tree on its way into a graph: a branch compares one feature
/// against a threshold and goes to <see cref="TrueChild"/> when
/// <c>x[feature] &lt;= threshold</c>, else to <see cref="FalseChild"/>; a leaf has
/// no children and carries one weight per target.
/// </summary>
/// <param name="Feature">Input column compared at a branch; ignored at a leaf.</param>
/// <param name="Threshold">Compared as float32, which is what the graph computes in.</param>
/// <param name="TrueChild">Index of the node taken when the comparison holds, or -1 at a leaf.</param>
/// <param name="FalseChild">Index of the node taken otherwise, or -1 at a leaf.</param>
/// <param name="LeafWeights">One per target at a leaf; empty at a branch.</param>
public sealed record TreeEnsembleNode(int Feature, double Threshold, int TrueChild, int FalseChild, IReadOnlyList<double> LeafWeights)
{
    public bool IsLeaf => TrueChild < 0;
}

/// <summary>
/// A set of trees summed or averaged into one or more targets — the shape every
/// tree learner exports as, whether it boosts or bags.
/// <para>
/// Written as one <c>TreeEnsembleRegressor</c> from <c>ai.onnx.ml</c> with one
/// target per output column, and never as the classifier operator. A classifier's
/// probabilities come from the same regressor followed by a softmax, a sigmoid
/// pair, or nothing at all when the leaves already hold class fractions — the
/// same heads the network and the linear model use, so one head is right once
/// rather than two, and the classifier operator's own quirks (a binary case that
/// wants one weight and a post-transform) never enter into it.
/// </para>
/// </summary>
public sealed class TreeEnsemble
{
    /// <param name="trees">Each a list of nodes, the root at index 0, children by index into the same list.</param>
    /// <param name="targetCount">Columns of the output; every leaf carries this many weights.</param>
    /// <param name="baseValues">Added to every row's output, one per target; the initial prediction a boosting round improves on. Null for zeros.</param>
    /// <param name="average">True to average the trees rather than sum them — a forest rather than a boosting.</param>
    public TreeEnsemble(IReadOnlyList<IReadOnlyList<TreeEnsembleNode>> trees, int targetCount, IReadOnlyList<double>? baseValues = null, bool average = false)
    {
        Trees = trees ?? throw new ArgumentNullException(nameof(trees));
        TargetCount = targetCount;
        BaseValues = baseValues ?? new double[Math.Max(targetCount, 0)];
        Average = average;
    }

    public IReadOnlyList<IReadOnlyList<TreeEnsembleNode>> Trees { get; }
    public int TargetCount { get; }
    public IReadOnlyList<double> BaseValues { get; }
    public bool Average { get; }

    /// <summary>Checks the trees are well formed against a graph taking <paramref name="featureCount"/> features.</summary>
    /// <exception cref="ArgumentException">A tree refers outside itself, a leaf carries the wrong number of weights, or a branch compares a feature the input lacks.</exception>
    public void Validate(int featureCount)
    {
        if (TargetCount < 1)
            throw new ArgumentException("A tree ensemble needs at least one target.");
        if (Trees.Count == 0)
            throw new ArgumentException("A tree ensemble needs at least one tree.");
        if (BaseValues.Count != TargetCount)
            throw new ArgumentException($"There are {BaseValues.Count} base values for {TargetCount} targets.");

        for (int t = 0; t < Trees.Count; t++)
        {
            var nodes = Trees[t];
            if (nodes.Count == 0)
                throw new ArgumentException($"Tree {t} has no nodes.");

            for (int i = 0; i < nodes.Count; i++)
            {
                var node = nodes[i];
                if (node.IsLeaf)
                {
                    if (node.LeafWeights.Count != TargetCount)
                        throw new ArgumentException($"Tree {t}, node {i} is a leaf with {node.LeafWeights.Count} weights for {TargetCount} targets.");
                    if (node.FalseChild >= 0)
                        throw new ArgumentException($"Tree {t}, node {i} is a leaf with a false child.");
                }
                else
                {
                    if (node.Feature < 0 || node.Feature >= featureCount)
                        throw new ArgumentException($"Tree {t}, node {i} compares feature {node.Feature} and the input has {featureCount}.");
                    if (node.TrueChild >= nodes.Count || node.FalseChild < 0 || node.FalseChild >= nodes.Count)
                        throw new ArgumentException($"Tree {t}, node {i} points outside the tree.");
                    if (!double.IsFinite(node.Threshold))
                        throw new ArgumentException($"Tree {t}, node {i} has threshold {node.Threshold}.");
                }
            }
        }
    }

    /// <summary>
    /// The threshold as the graph will hold it: the largest float32 no greater than
    /// the double. A tree's threshold is the midpoint of two float32 feature values;
    /// when those two are adjacent floats the midpoint has no float32 of its own,
    /// and rounding to nearest can land on the upper value — which then goes left
    /// in the graph and right in the model. Rounding down instead keeps every
    /// float32 input on the same side of both.
    /// </summary>
    internal static float Float32Threshold(double threshold)
    {
        float rounded = (float)threshold;
        return rounded > threshold ? MathF.BitDecrement(rounded) : rounded;
    }

    /// <summary>The operator's attributes: the trees flattened into the parallel arrays the specification asks for.</summary>
    internal IEnumerable<OnnxAttribute> Attributes()
    {
        var treeIds = new List<long>();
        var nodeIds = new List<long>();
        var featureIds = new List<long>();
        var modes = new List<string>();
        var values = new List<float>();
        var trueIds = new List<long>();
        var falseIds = new List<long>();
        var missing = new List<long>();

        var targetTreeIds = new List<long>();
        var targetNodeIds = new List<long>();
        var targetIds = new List<long>();
        var targetWeights = new List<float>();

        for (int t = 0; t < Trees.Count; t++)
        {
            var nodes = Trees[t];
            for (int i = 0; i < nodes.Count; i++)
            {
                var node = nodes[i];
                treeIds.Add(t);
                nodeIds.Add(i);
                // The specification wants every node array filled, leaves included; a
                // leaf's feature, threshold and children carry nothing and are zero.
                featureIds.Add(node.IsLeaf ? 0 : node.Feature);
                modes.Add(node.IsLeaf ? "LEAF" : "BRANCH_LEQ");
                values.Add(node.IsLeaf ? 0f : Float32Threshold(node.Threshold));
                trueIds.Add(node.IsLeaf ? 0 : node.TrueChild);
                falseIds.Add(node.IsLeaf ? 0 : node.FalseChild);
                // A NaN input follows the false branch, which is where an x <= t
                // comparison sends it in the C# too. OnnxModel refuses NaN anyway.
                missing.Add(0);

                if (node.IsLeaf)
                {
                    for (int k = 0; k < TargetCount; k++)
                    {
                        targetTreeIds.Add(t);
                        targetNodeIds.Add(i);
                        targetIds.Add(k);
                        targetWeights.Add((float)node.LeafWeights[k]);
                    }
                }
            }
        }

        yield return OnnxAttribute.String("aggregate_function", Average ? "AVERAGE" : "SUM");
        yield return OnnxAttribute.Floats("base_values", BaseValues.Select(v => (float)v));
        yield return OnnxAttribute.Int("n_targets", TargetCount);
        yield return OnnxAttribute.Ints("nodes_falsenodeids", falseIds);
        yield return OnnxAttribute.Ints("nodes_featureids", featureIds);
        yield return OnnxAttribute.Ints("nodes_missing_value_tracks_true", missing);
        yield return OnnxAttribute.Strings("nodes_modes", modes);
        yield return OnnxAttribute.Ints("nodes_nodeids", nodeIds);
        yield return OnnxAttribute.Ints("nodes_treeids", treeIds);
        yield return OnnxAttribute.Ints("nodes_truenodeids", trueIds);
        yield return OnnxAttribute.Floats("nodes_values", values);
        yield return OnnxAttribute.String("post_transform", "NONE");
        yield return OnnxAttribute.Ints("target_ids", targetIds);
        yield return OnnxAttribute.Ints("target_nodeids", targetNodeIds);
        yield return OnnxAttribute.Ints("target_treeids", targetTreeIds);
        yield return OnnxAttribute.Floats("target_weights", targetWeights);
    }
}
