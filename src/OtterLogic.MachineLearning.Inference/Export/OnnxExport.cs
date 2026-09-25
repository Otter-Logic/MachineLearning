namespace OtterLogic.MachineLearning.Inference.Export;

/// <summary>
/// Writes a model file the way the Python trainer used to: to a staging name
/// beside the target, checked by running it, and renamed over only once it has
/// answered the same as the model it was made from.
/// <para>
/// The check is the whole point. A graph written by hand can be wrong in ways
/// that still load and run — a threshold on the wrong feature, weights transposed,
/// a softmax on the wrong axis — and the failure mode is a model that answers
/// confidently and wrongly for months. Running the file through ONNX Runtime on
/// the rows the model was scored on, and refusing to keep it if the two disagree
/// beyond float32 tolerance, turns that into an error at training time.
/// </para>
/// </summary>
public static class OnnxExport
{
    /// <summary>
    /// Writes <paramref name="bytes"/> to <paramref name="outputPath"/> if the model
    /// passes <paramref name="check"/>.
    /// </summary>
    /// <param name="bytes">The file, from <see cref="OnnxGraph.ToBytes"/>.</param>
    /// <param name="outputPath">Where it goes; <c>.onnx</c>, in a folder that exists.</param>
    /// <param name="check">Given the model opened from the staging file; throws <see cref="InvalidDataException"/> to refuse it.</param>
    /// <returns>The file's length in bytes.</returns>
    /// <exception cref="InvalidDataException">The file did not load, or <paramref name="check"/> refused it. Nothing was written to <paramref name="outputPath"/>.</exception>
    public static long Write(byte[] bytes, string outputPath, Action<OnnxModel> check)
    {
        ArgumentNullException.ThrowIfNull(bytes);
        ArgumentNullException.ThrowIfNull(check);
        if (string.IsNullOrWhiteSpace(outputPath))
            throw new ArgumentException("No output path was given.", nameof(outputPath));

        string staging = outputPath + ".writing";
        try
        {
            File.WriteAllBytes(staging, bytes);
            using (var model = OnnxModel.Load(staging))
                check(model);

            // Move over the target, so a reader polling the file never sees half of one.
            File.Move(staging, outputPath, overwrite: true);
            return bytes.LongLength;
        }
        finally
        {
            if (File.Exists(staging))
            {
                try { File.Delete(staging); } catch (IOException) { } catch (UnauthorizedAccessException) { }
            }
        }
    }

    /// <summary>
    /// Refuses a regressor whose answers on <paramref name="rows"/> differ from
    /// <paramref name="expected"/> by more than <paramref name="tolerance"/> of the
    /// largest expected magnitude (at least one), which is the float32 allowance the
    /// Python trainer used.
    /// </summary>
    public static void ExpectValues(OnnxModel model, double[,] rows, IReadOnlyList<double> expected, double tolerance = 1e-4)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.Task != ModelTask.Regression)
            throw new InvalidDataException("The exported model is a classifier and a regressor was written.");

        var prediction = model.Predict(rows);
        if (prediction.Values.Length != expected.Count)
            throw new InvalidDataException($"The exported model answered {prediction.Values.Length} values for {expected.Count} rows.");

        double scale = 1.0;
        foreach (double v in expected) scale = Math.Max(scale, Math.Abs(v));

        double worst = 0.0;
        for (int i = 0; i < expected.Count; i++)
            worst = Math.Max(worst, Math.Abs(prediction.Values[i] - expected[i]));

        if (worst > tolerance * scale)
            throw new InvalidDataException(
                $"The exported model disagrees with the fitted one: the largest difference on the checked rows is "
                + $"{worst:G6} against values up to {scale:G6}. The export is broken, so no model was written.");
    }

    /// <summary>
    /// Refuses a classifier whose probabilities on <paramref name="rows"/> differ
    /// from <paramref name="probabilities"/> by more than <paramref name="tolerance"/>,
    /// or whose labels differ where the two best probabilities are not a near tie.
    /// A probability within tolerance can still tip a near tie the other way, and
    /// that is float32, not a fault.
    /// </summary>
    public static void ExpectClasses(OnnxModel model, double[,] rows, IReadOnlyList<int> labels, double[,] probabilities, double tolerance = 1e-4)
    {
        ArgumentNullException.ThrowIfNull(model);
        if (model.Task != ModelTask.Classification)
            throw new InvalidDataException("The exported model is a regressor and a classifier was written.");

        var prediction = model.Predict(rows);
        int n = probabilities.GetLength(0);
        int k = probabilities.GetLength(1);
        if (prediction.SampleCount != n || labels.Count != n)
            throw new InvalidDataException($"The exported model answered {prediction.SampleCount} rows for {n} checked.");
        if (n > 0 && prediction.Probabilities.GetLength(1) != k)
            throw new InvalidDataException(
                $"The exported model returns {prediction.Probabilities.GetLength(1)} probabilities per row and the fitted one {k}.");

        double worst = 0.0;
        for (int i = 0; i < n; i++)
            for (int c = 0; c < k; c++)
                worst = Math.Max(worst, Math.Abs(prediction.Probabilities[i, c] - probabilities[i, c]));

        if (worst > tolerance)
            throw new InvalidDataException(
                $"The exported model disagrees with the fitted one: probabilities differ by up to {worst:G6}. "
                + "The export is broken, so no model was written.");

        for (int i = 0; i < n; i++)
        {
            if (prediction.LabelIndices[i] == labels[i])
                continue;

            double best = double.NegativeInfinity, second = double.NegativeInfinity;
            for (int c = 0; c < k; c++)
            {
                double p = probabilities[i, c];
                if (p > best) { second = best; best = p; }
                else if (p > second) second = p;
            }

            if (best - second > 1e-3)
                throw new InvalidDataException(
                    $"The exported model answers class {prediction.LabelIndices[i]} on checked row {i} where the fitted one "
                    + $"answers {labels[i]}, and it is not a near tie. The export is broken, so no model was written.");
        }
    }
}
