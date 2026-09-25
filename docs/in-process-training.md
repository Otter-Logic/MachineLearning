# Training in-process — the decision

Decided 2026-09-25. Supersedes the trainer half of
[user-trained-models-plan.md](user-trained-models-plan.md) and the whole of
[user-trained-models-phase-1.md](user-trained-models-phase-1.md), which stay as
the record of what was built and why it was taken out. The ONNX half of both
documents stands: a model is one `.onnx` carrying its own metadata, and
`OnnxModel` reads it without knowing what made it.

## What changed

OtterTrain fits its models in C#, inside Rhino, on a background task. There is no
Python process, no runtime bundle, no release manifest and no install menu. The
four learners on the wire — Boosted Trees, Random Forest, Neural Network, Linear
Model — are C# in the Supervised repo, and the one entry point that fits, scores,
exports and checks a model is `TrainRun.Fit` there. The `.onnx` is written by C#
too, through `OnnxGraph` in the Inference project, and is run back through ONNX
Runtime before it is kept.

## Why the Python trainer went

The phase-1 design was sound on its own terms and it was built: a private CPython
with scikit-learn and skl2onnx, zipped, hashed, attached to a release, fetched on
first use into `%LOCALAPPDATA%`. It was taken out before it shipped for three
reasons, in the order they weigh.

1. **Version drift is the failure a user meets.** A bundle pins every package on
   the day it is built. The plug-in's C# reader, the trainer's exporter and ONNX
   Runtime's opset support all move on their own schedules, and the `export.py`
   in this repo already carried one workaround for a skl2onnx-against-protobuf
   break that appeared between two minor versions. Each such break is a user
   whose Train stopped working after an update they did not make, and the
   diagnosis is in a Python traceback they cannot read.
2. **A download is not an install.** A Yak install is an unzip. The runtime had
   to arrive later, by a click, from GitHub, through whatever proxy the firm runs,
   at a hundred megabytes or more. `InstallFromFile` existed for the machines
   that could not reach GitHub at all. Every one of those paths is support load.
3. **It was a release pipeline of its own.** `build-bundle.ps1`, a manifest that
   had to sit on the *latest* release, a version in two files that had to agree,
   and a `gh release create` cut by hand — for a plug-in whose every other part
   ships in one `.yak`.

What was given up is the reach of scikit-learn and the road to PyTorch training
inside OtterTrain. That road is not closed: a model trained anywhere still comes
in as an `.onnx`, and OtterPredict runs it. What OtterTrain itself offers is now
the tabular case done well — trees, a forest, a small network, a line — which is
where a parameter sweep of a few thousand rows lives anyway.

## What it cost to bring in-process

- **An ONNX writer.** ONNX Runtime does not write models, and Google.Protobuf was
  not taken as a dependency for the reason Accord and MathNet were not: a second
  copy of a common assembly at another version is a conflict waiting for a user
  with another plug-in. `ProtobufWriter` is the wire format in a few dozen lines;
  `OnnxGraph` builds the handful of operators the learners need; `TreeEnsemble`
  flattens trees into the `ai.onnx.ml` operator. Every learner exports as one of
  two graph shapes — an affine chain, or a tree ensemble — into one of two heads.
- **Verify on write, still.** `OnnxExport.Write` runs the staged file through
  ONNX Runtime on the held-out rows and refuses it if the answers differ from the
  fitted model's beyond float32 tolerance. That is the check `export.verify` did
  in Python, and it is what makes a hand-written protobuf safe to ship.
- **Three algorithms.** A decision tree, boosting and a forest over it, and a
  multilayer perceptron — tested against scikit-learn: exactly where the answer
  is unique (a tree, a boosting), by score where a random stream is involved (a
  forest, an optimiser).
- **Trees see float32.** scikit-learn casts inputs to float32 at fit and predict,
  and the ONNX tree operator compares float32 thresholds. The C# does the same at
  the door, so all three branch identically on every row.

## What stays the same

The `.onnx` contract, byte for byte: `features` in; `value`, or `label` and
`probabilities`, out; the metadata record under `metadata_props["otterlogic"]`.
The Inference tests still open a scikit-learn-made model and must answer what
Python answered, and now also open a PyTorch-made one carrying no OtterLogic
metadata — the proof that a deep-learning model from elsewhere still runs.
