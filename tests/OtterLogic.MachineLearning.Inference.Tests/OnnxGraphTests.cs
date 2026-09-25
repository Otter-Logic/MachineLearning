using OtterLogic.MachineLearning.Inference.Export;
using Xunit;

namespace OtterLogic.MachineLearning.Inference.Tests;

/// <summary>
/// The writing half of the ONNX seam, checked by the reading half: a graph built
/// here is written to a file, opened with <see cref="OnnxModel"/> — which is ONNX
/// Runtime parsing what the hand-written protobuf says — and must answer what the
/// same arithmetic in C# answers. Each test writes to a fixed name under the temp
/// folder and leaves it there, so <c>onnx.checker</c> can be run over the files by
/// hand when the writer changes.
/// </summary>
public class OnnxGraphTests
{
    private static readonly string Folder = Path.Combine(Path.GetTempPath(), "otterlogic-onnx-tests");

    private static string Output(string name)
    {
        Directory.CreateDirectory(Folder);
        return Path.Combine(Folder, name + ".onnx");
    }

    private static readonly double[,] Rows =
    {
        { 10.0, 1.0, 1.0 },
        { 12.5, 2.0, 0.9 },
        { 20.0, 0.6, 1.1 },
        { 30.0, 0.5, 1.0 },
        { 15.0, 1.5, 1.0 },
        { 0.0, 0.0, 0.0 },
    };

    private static readonly double[] Centre = { 15.0, 1.0, 1.0 };
    private static readonly double[] Scale = { 5.0, 0.5, 0.1 };

    private static ModelMetadata Regressor() => new()
    {
        Task = ModelTask.Regression,
        Features = new[] { "span", "sag", "rest_factor" },
        Target = "deflection",
        Trainer = new TrainerInfo("otterlogic", "0.1.0", "linear"),
    };

    private static ModelMetadata Classifier(int classes) => new()
    {
        Task = ModelTask.Classification,
        Features = new[] { "span", "sag", "rest_factor" },
        Target = "kind",
        Classes = Enumerable.Range(0, classes).Select(c => "class" + c).ToArray(),
    };

    private static double[,] Standardised()
    {
        int n = Rows.GetLength(0);
        var z = new double[n, 3];
        for (int i = 0; i < n; i++)
            for (int j = 0; j < 3; j++)
                z[i, j] = (Rows[i, j] - Centre[j]) / Scale[j];
        return z;
    }

    [Fact]
    public void LinearRegressorAnswersWhatTheArithmeticSays()
    {
        double[,] weights = { { 2.0 }, { -3.0 }, { 0.5 } };
        double[] bias = { 40.0 };

        var graph = new OnnxGraph(3);
        string z = graph.Standardise(graph.Input, Centre, Scale);
        string y = graph.Dense(z, weights, bias);
        graph.ValueOutput(y);

        var standardised = Standardised();
        var expected = new double[Rows.GetLength(0)];
        for (int i = 0; i < expected.Length; i++)
            expected[i] = bias[0] + 2.0 * standardised[i, 0] - 3.0 * standardised[i, 1] + 0.5 * standardised[i, 2];

        string path = Output("linear-regressor");
        long bytes = OnnxExport.Write(graph.ToBytes(Regressor()), path, model => OnnxExport.ExpectValues(model, Rows, expected));

        Assert.True(bytes > 0);
        Assert.True(File.Exists(path));
        Assert.False(File.Exists(path + ".writing"));

        using var reopened = OnnxModel.Load(path);
        Assert.Equal(ModelTask.Regression, reopened.Task);
        Assert.Equal(new[] { "span", "sag", "rest_factor" }, reopened.FeatureNames);
        Assert.Equal("linear", reopened.Metadata!.Trainer!.ModelType);

        var prediction = reopened.Predict(Rows);
        for (int i = 0; i < expected.Length; i++)
            Assert.Equal(expected[i], prediction.Values[i], 1e-4);
    }

    [Fact]
    public void RescaledRegressorUndoesATargetStandardisation()
    {
        double[,] weights = { { 1.0 }, { 0.0 }, { 0.0 } };
        var graph = new OnnxGraph(3);
        string z = graph.Standardise(graph.Input, Centre, Scale);
        string y = graph.Rescale(graph.Dense(z, weights, new[] { 0.0 }), scale: 100.0, offset: 1000.0);
        graph.ValueOutput(y);

        var standardised = Standardised();
        var expected = Enumerable.Range(0, Rows.GetLength(0)).Select(i => 1000.0 + 100.0 * standardised[i, 0]).ToArray();

        OnnxExport.Write(graph.ToBytes(Regressor()), Output("rescaled-regressor"), model => OnnxExport.ExpectValues(model, Rows, expected));
    }

    [Fact]
    public void SoftmaxClassifierAnswersWhatTheArithmeticSays()
    {
        double[,] weights = { { 1.0, -1.0, 0.0 }, { 0.0, 1.0, -1.0 }, { -1.0, 0.0, 1.0 } };
        double[] bias = { 0.1, 0.0, -0.1 };

        var graph = new OnnxGraph(3);
        string z = graph.Standardise(graph.Input, Centre, Scale);
        string p = graph.Softmax(graph.Dense(z, weights, bias));
        graph.ClassificationOutputs(p, 3);

        var (labels, probabilities) = SoftmaxOf(Standardised(), weights, bias);

        string path = Output("softmax-classifier");
        OnnxExport.Write(graph.ToBytes(Classifier(3)), path, model => OnnxExport.ExpectClasses(model, Rows, labels, probabilities));

        using var reopened = OnnxModel.Load(path);
        Assert.Equal(ModelTask.Classification, reopened.Task);
        var prediction = reopened.Predict(Rows);
        Assert.Equal(labels, prediction.LabelIndices);
        Assert.Equal(labels.Select(l => "class" + l), prediction.Labels);
        for (int i = 0; i < labels.Length; i++)
            Assert.Equal(probabilities[i, labels[i]], prediction.Confidence[i], 1e-5);
    }

    [Fact]
    public void SigmoidPairMakesTwoClassProbabilitiesFromOneScore()
    {
        double[,] weights = { { 1.0 }, { -0.5 }, { 0.25 } };
        var graph = new OnnxGraph(3);
        string z = graph.Standardise(graph.Input, Centre, Scale);
        string p = graph.SigmoidPair(graph.Dense(z, weights, new[] { 0.2 }));
        graph.ClassificationOutputs(p, 2);

        var standardised = Standardised();
        int n = Rows.GetLength(0);
        var probabilities = new double[n, 2];
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            double s = 0.2 + standardised[i, 0] - 0.5 * standardised[i, 1] + 0.25 * standardised[i, 2];
            double sigma = 1.0 / (1.0 + Math.Exp(-s));
            probabilities[i, 0] = 1.0 - sigma;
            probabilities[i, 1] = sigma;
            labels[i] = sigma > 0.5 ? 1 : 0;
        }

        OnnxExport.Write(graph.ToBytes(Classifier(2)), Output("sigmoid-classifier"), model => OnnxExport.ExpectClasses(model, Rows, labels, probabilities));
    }

    /// <summary>
    /// Two stumps on different features, summed onto a base value: the smallest
    /// boosting there is. Node order is deliberately not depth-first, to check the
    /// children are followed by index and not by position.
    /// </summary>
    [Fact]
    public void TreeEnsembleSumsLeavesOntoTheBaseValue()
    {
        var treeA = new TreeEnsembleNode[]
        {
            new(0, 14.0, 1, 2, Array.Empty<double>()),
            new(0, 0, -1, -1, new[] { -1.0 }),
            new(0, 0, -1, -1, new[] { 2.0 }),
        };
        var treeB = new TreeEnsembleNode[]
        {
            new(1, 0.75, 2, 1, Array.Empty<double>()),
            new(0, 0, -1, -1, new[] { 10.0 }),
            new(0, 0, -1, -1, new[] { -10.0 }),
        };

        var ensemble = new TreeEnsemble(new[] { treeA, treeB }, targetCount: 1, baseValues: new[] { 100.0 });
        var graph = new OnnxGraph(3);
        graph.ValueOutput(graph.TreeEnsemble(graph.Input, ensemble));

        var expected = new double[Rows.GetLength(0)];
        for (int i = 0; i < expected.Length; i++)
            expected[i] = 100.0 + (Rows[i, 0] <= 14.0 ? -1.0 : 2.0) + (Rows[i, 1] <= 0.75 ? -10.0 : 10.0);

        string path = Output("tree-regressor");
        OnnxExport.Write(graph.ToBytes(Regressor() with { Trainer = new TrainerInfo("otterlogic", "0.1.0", "boostedTrees") }),
            path, model => OnnxExport.ExpectValues(model, Rows, expected));

        using var reopened = OnnxModel.Load(path);
        var prediction = reopened.Predict(Rows);
        Assert.Equal(expected, prediction.Values.Select(v => Math.Round(v, 4)));
    }

    [Fact]
    public void AveragedTreeEnsembleWithClassFractionsIsAForestClassifier()
    {
        // Two trees whose leaves hold class fractions over three classes.
        var treeA = new TreeEnsembleNode[]
        {
            new(0, 14.0, 1, 2, Array.Empty<double>()),
            new(0, 0, -1, -1, new[] { 1.0, 0.0, 0.0 }),
            new(0, 0, -1, -1, new[] { 0.0, 0.5, 0.5 }),
        };
        var treeB = new TreeEnsembleNode[]
        {
            new(2, 0.95, 1, 2, Array.Empty<double>()),
            new(0, 0, -1, -1, new[] { 0.2, 0.8, 0.0 }),
            new(0, 0, -1, -1, new[] { 0.0, 0.0, 1.0 }),
        };

        var ensemble = new TreeEnsemble(new[] { treeA, treeB }, targetCount: 3, average: true);
        var graph = new OnnxGraph(3);
        graph.ClassificationOutputs(graph.TreeEnsemble(graph.Input, ensemble), 3);

        int n = Rows.GetLength(0);
        var probabilities = new double[n, 3];
        var labels = new int[n];
        for (int i = 0; i < n; i++)
        {
            var a = Rows[i, 0] <= 14.0 ? treeA[1].LeafWeights : treeA[2].LeafWeights;
            var b = Rows[i, 2] <= 0.95 ? treeB[1].LeafWeights : treeB[2].LeafWeights;
            int best = 0;
            for (int c = 0; c < 3; c++)
            {
                probabilities[i, c] = (a[c] + b[c]) / 2.0;
                if (probabilities[i, c] > probabilities[i, best]) best = c;
            }
            labels[i] = best;
        }

        OnnxExport.Write(graph.ToBytes(Classifier(3)), Output("forest-classifier"), model => OnnxExport.ExpectClasses(model, Rows, labels, probabilities));
    }

    [Fact]
    public void MultiTargetTreeEnsembleFeedsASoftmax()
    {
        // One tree with three targets per leaf: a multiclass boosting round.
        var tree = new TreeEnsembleNode[]
        {
            new(1, 1.25, 1, 2, Array.Empty<double>()),
            new(0, 0, -1, -1, new[] { 0.5, -0.5, 0.0 }),
            new(0, 0, -1, -1, new[] { -1.0, 0.0, 1.0 }),
        };
        double[] baseValues = { 0.1, 0.2, -0.3 };
        var ensemble = new TreeEnsemble(new[] { tree }, 3, baseValues);

        var graph = new OnnxGraph(3);
        string scores = graph.TreeEnsemble(graph.Input, ensemble);
        graph.ClassificationOutputs(graph.Softmax(scores), 3);

        int n = Rows.GetLength(0);
        var raw = new double[n, 3];
        for (int i = 0; i < n; i++)
        {
            var leaf = Rows[i, 1] <= 1.25 ? tree[1].LeafWeights : tree[2].LeafWeights;
            for (int c = 0; c < 3; c++) raw[i, c] = baseValues[c] + leaf[c];
        }
        var (labels, probabilities) = SoftmaxOf(raw, null, null);

        OnnxExport.Write(graph.ToBytes(Classifier(3)), Output("boosted-classifier"), model => OnnxExport.ExpectClasses(model, Rows, labels, probabilities));
    }

    [Fact]
    public void ADisagreeingModelIsRefusedAndNothingIsLeftBehind()
    {
        double[,] weights = { { 1.0 }, { 0.0 }, { 0.0 } };
        var graph = new OnnxGraph(3);
        graph.ValueOutput(graph.Dense(graph.Input, weights, new[] { 0.0 }));

        string path = Output("refused");
        File.Delete(path);
        var wrong = Enumerable.Repeat(12345.0, Rows.GetLength(0)).ToArray();

        var ex = Assert.Throws<InvalidDataException>(() =>
            OnnxExport.Write(graph.ToBytes(Regressor()), path, model => OnnxExport.ExpectValues(model, Rows, wrong)));

        Assert.Contains("disagrees", ex.Message);
        Assert.False(File.Exists(path));
        Assert.False(File.Exists(path + ".writing"));
    }

    [Fact]
    public void TheMetadataMustMatchTheGraph()
    {
        var graph = new OnnxGraph(3);
        graph.ValueOutput(graph.Dense(graph.Input, new double[,] { { 1.0 }, { 0.0 }, { 0.0 } }, new[] { 0.0 }));

        Assert.Throws<ArgumentException>(() => graph.ToBytes(Regressor() with { Features = new[] { "a", "b" } }));
        Assert.Throws<ArgumentException>(() => graph.ToBytes(Classifier(2)));
    }

    [Fact]
    public void AGraphNeedsAHeadBeforeItIsWritten()
    {
        var graph = new OnnxGraph(3);
        graph.Dense(graph.Input, new double[,] { { 1.0 }, { 0.0 }, { 0.0 } }, new[] { 0.0 });
        Assert.Throws<InvalidOperationException>(() => graph.ToBytes(Regressor()));
    }

    [Fact]
    public void AStandardisationWithAZeroScaleIsRefused()
    {
        var graph = new OnnxGraph(3);
        Assert.Throws<ArgumentException>(() => graph.Standardise(graph.Input, Centre, new[] { 5.0, 0.0, 0.1 }));
    }

    [Fact]
    public void AnIllFormedTreeIsRefused()
    {
        var pointsOutside = new TreeEnsembleNode[] { new(0, 1.0, 1, 5, Array.Empty<double>()), new(0, 0, -1, -1, new[] { 1.0 }) };
        Assert.Throws<ArgumentException>(() => new TreeEnsemble(new[] { pointsOutside }, 1).Validate(3));

        var wrongFeature = new TreeEnsembleNode[] { new(3, 1.0, 1, 1, Array.Empty<double>()), new(0, 0, -1, -1, new[] { 1.0 }) };
        Assert.Throws<ArgumentException>(() => new TreeEnsemble(new[] { wrongFeature }, 1).Validate(3));

        var wrongWeights = new TreeEnsembleNode[] { new(0, 0, -1, -1, new[] { 1.0, 2.0 }) };
        Assert.Throws<ArgumentException>(() => new TreeEnsemble(new[] { wrongWeights }, 1).Validate(3));
    }

    private static (int[] Labels, double[,] Probabilities) SoftmaxOf(double[,] inputs, double[,]? weights, double[]? bias)
    {
        int n = inputs.GetLength(0);
        int k = weights?.GetLength(1) ?? inputs.GetLength(1);
        var probabilities = new double[n, k];
        var labels = new int[n];
        var scores = new double[k];

        for (int i = 0; i < n; i++)
        {
            for (int c = 0; c < k; c++)
            {
                if (weights is null)
                {
                    scores[c] = inputs[i, c];
                }
                else
                {
                    scores[c] = bias![c];
                    for (int j = 0; j < inputs.GetLength(1); j++)
                        scores[c] += inputs[i, j] * weights[j, c];
                }
            }

            double max = scores.Max();
            double sum = 0.0;
            for (int c = 0; c < k; c++) sum += Math.Exp(scores[c] - max);
            int best = 0;
            for (int c = 0; c < k; c++)
            {
                probabilities[i, c] = Math.Exp(scores[c] - max) / sum;
                if (probabilities[i, c] > probabilities[i, best]) best = c;
            }
            labels[i] = best;
        }

        return (labels, probabilities);
    }
}
