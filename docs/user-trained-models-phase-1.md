# Phase 1 — create and use an ONNX model

Decided 2026-09-23. Status: **steps 1 to 3 built** the same day — the contract,
Predict and Train, with the trainer package and its tests; **step 4 built except
for cutting the release** — the bundle build script, the install API and its tests
exist, and no release carries a bundle yet. Compiled and tested from xunit and
pytest; not yet opened in Grasshopper. See
[What steps 1 to 3 turned out to be](#what-steps-1-to-3-turned-out-to-be) and
[What changed after phase 1](#what-changed-after-phase-1), which supersedes the
Model Type input described below.
Refines the phase 1 line of [user-trained-models-plan.md](user-trained-models-plan.md);
where the two differ, this file wins and the reasons are recorded here.

Phase 0 built everything up to the ONNX boundary and nothing across it. There is
no ONNX Runtime reference anywhere in the organisation, no Inference project, no
Train or Predict component, and the one Python script that writes an `.onnx`
(`Rhino3D/python/training/train_surrogate.py`) is a developer script whose output
nothing in C# can read. Phase 1 is the boundary itself, crossed in both
directions.

## What it is for

One person in a firm has a dataset and a question. They press Train, choose a
model type, and get a file. That file goes on a network share, into an email, or
into a project folder. Everyone else drops the file into Predict and gets answers.
Nobody in the second group installs anything beyond OtterLogic.

Two consequences shape everything below:

- **The model is one file.** A file that has to travel with a sidecar is a file
  that arrives without it. Everything Predict needs to know — feature names and
  order, the target, the classes, how well it did — is written *into* the
  `.onnx`, in its metadata, and there is no `.json` beside it. This reverses the
  plan's sidecar decision; the reasoning is under [The model file](#the-model-file).
- **Two components, not five.** Predict, and Train. The plan's Load Model, ML
  Environment, Train Classifier and Train Regressor fold into those two; see
  [What was stripped](#what-was-stripped) for where each went.

## Use: Predict

One component, under **Machine Learning**, Inference tier. It is the whole
experience for most users, so it has to work with nothing wired but a path.

| | Name | Shape | Notes |
|---|---|---|---|
| in | Model | text, item | Path to the `.onnx`. |
| in | Inputs | number, tree | One branch per sample, the shape every other component takes. Optional: with nothing wired, the component still loads the model and fills Report, which is how a recipient finds out what to wire. |
| out | Prediction | text or number, list | Class names for a classifier, values for a regressor. Text for classes for the same reason the baselines use text: a class is a name, and the commonest are called 0 and 1. |
| out | Confidence | number, list | A classifier's probability for the class it chose. Empty for a regressor — a boosted tree has no honest number to give here, and a made-up one would be read as if it were honest. |
| out | Feature Names | text, list | The columns the model expects, in the order it expects them. |
| out | Report | text, item | Target, task, classes, the trainer and version, when it was trained, on how many rows and groups, and the held-out score. |

What it checks: the number of columns in each branch against the number of
features in the model. Names cannot be checked — there are none on the wire — so
the Report and Feature Names outputs exist so that a person can. When phase 2's
Member Features produces a named table, Predict can gain a second input that
matches by name; that is the point to add it, not now.

What it holds: one ONNX Runtime session per component, rebuilt only when the
file's path or last-write time changes, disposed when the component is removed.
A session is expensive to build and cheap to run, and Grasshopper re-solves on
every slider move.

A model without OtterLogic metadata still runs, with a warning: the input is
taken from the graph's first input, the output from its first output, feature
names are blank. It costs nothing, since it is the same code path with empty
names, and it means an `.onnx` from anywhere can be tried.

## Create: Train

One component, under **Supervised Learning**, Methods tier. It reads the dataset
folder phase 0 writes, so curation stays what it was — delete a file — and the
split is by group, as it must be.

| | Name | Shape | Notes |
|---|---|---|---|
| in | Dataset | text, item | The folder. |
| in | Target | text, item | A target column's name. Optional when the schema has one target. Whether it is a class or a number comes from the schema — declared in phase 0, never inferred — so one component trains both. |
| in | Model Type | enum | Boosted Trees (default) or Neural Network. A dropdown component in the quinary tier, as the other enums are. *Superseded: see [What changed after phase 1](#what-changed-after-phase-1) — a Learner value with its settings replaces the bare enum.* |
| in | Holdout | number, item | Share of groups held back for the score. A quarter by default, as Split By Group. |
| in | Model | text, item | Where to write the `.onnx`. |
| in | Run | boolean, item | Training starts on the rising edge, is cancelled on the falling one, and never restarts because something upstream changed. |
| out | Model | text, item | The file written, once training ends. |
| out | Report | text, item | The score on held-out groups, the confusion matrix or residual summary, the rows and groups used, the time taken. |
| out | Status | text, item | What the trainer is doing now, while it runs. |

**Train never blocks the canvas.** It writes a job file, starts the trainer as a
separate process with no window, and polls a progress file on a timer. When the
process ends it expires its own solution and the outputs fill. Everything below
`SolveInstance` is in `OtterLogic.MachineLearning.Training`, with no Grasshopper
reference, so it can be tested on the trainer alone.

**Runtime install is a context-menu item, not a component.** On the first press
Train reports that the training runtime is not installed and its menu offers
*Install training runtime (about 100 MB)* and *Install from file…*. The second is
the answer to proxies that block GitHub: IT downloads the archive once and
points every machine at a share. Both unpack to
`%LOCALAPPDATA%\OtterLogic\trainer\<version>\`, hash-checked, no admin rights,
no PATH change. This is the plan's step 3 with the separate component removed.

Model types are offered one at a time, each gated on a round-trip parity test
(below). Two to start:

| Model Type | Classifier | Regressor | Why |
|---|---|---|---|
| Boosted Trees | `HistGradientBoostingClassifier` | `HistGradientBoostingRegressor` | On member-level tables under roughly 10k rows this is usually the most accurate thing available and trains in seconds. The default. |
| Neural Network | `MLPClassifier` | `MLPRegressor` | The comparison a user will ask for, and the shape phase 3 grows from. |

Both are scikit-learn, exported by skl2onnx, inside a `Pipeline` whose first step
is a `StandardScaler` fitted on the training rows only. That bakes scaling into
the graph, which is the phase 0 decision that standardisation lives inside each
model, carried across the boundary. LightGBM is not in phase 1: it needs a
second converter for a gain that does not show on tables this size.

## The model file

An `.onnx` file carries a `metadata_props` map of strings. OtterLogic writes one
entry, key `otterlogic`, holding this JSON:

```json
{
  "format": 1,
  "task": "classification",
  "features": ["length_ratio", "angle_to_vertical", "members_at_node"],
  "target": "rel_My",
  "classes": ["0", "1"],
  "extractorVersion": "1",
  "trainer": { "name": "otterlogic-trainer", "version": "0.1.0", "modelType": "boostedTrees" },
  "trained": "2026-09-23T14:02:11Z",
  "data": { "rows": 8412, "groups": 7, "holdoutGroups": ["2025-003-office-block"] },
  "score": { "accuracy": 0.94, "baseline": 0.71 }
}
```

`classes` is absent for a regressor and `score` holds `mae` and `r2` instead.
`extractorVersion` is the dataset schema's, copied through, so Predict can
refuse rows from a different extractor exactly as Write Dataset refuses them into
a folder.

The graph's interface is fixed, so Predict never has to guess:

| | Name | Type and shape |
|---|---|---|
| input | `features` | float32 `[N, F]` |
| regression output | `value` | float32 `[N, 1]` |
| classification outputs | `label` | int64 `[N]`, an index into `classes` |
| | `probabilities` | float32 `[N, C]` |

skl2onnx wraps classifier probabilities in a sequence of maps by default; the
exporter turns that off (`zipmap: False`) so `probabilities` is a plain tensor.

Why metadata rather than a sidecar: the sidecar buys human readability at the
cost of being separable, and Predict's Report gives the readability back. ONNX
Runtime's C# API exposes the map directly (`ModelMetadata.CustomMetadataMap`),
so the read is a dictionary lookup. A future format bumps `format`, and Predict
refuses a number it does not know rather than mis-reading it.

`OtterLogic.MachineLearning.Inference` owns `ModelMetadata` — the record above,
its JSON, and its validation — and it is the one type Python and C# both have to
agree on. The Python side is the same record written by hand, and the parity
test is what keeps them agreeing.

## The trainer

One Python package, `otterlogic_trainer`, in this repo under `python/trainer/`.
The plan left open whether paradigm repos contribute modules; the answer for
phase 1 is that they do not, because one package is what the bundle build has to
gather and there is nothing yet that would justify a second. When DeepLearning
adds a graph model in phase 3 it adds a module here and a Model Type there.

The protocol between Train and the trainer is two files in a working folder:

- **`job.json`**, written by Train: dataset folder, target, model type, holdout
  share, seed, output path.
- **`progress.jsonl`**, written by the trainer, one line at a time:
  `{"stage": "reading", "message": "7 models, 8412 rows"}`, then fitting, then
  scoring, then exporting, then either `{"done": true, "model": "…", "score": {…}}`
  or `{"error": "…"}`. Train shows the latest `message` on Status and the final
  line on Report.

The trainer reads every CSV against `schema.json`, tags rows with their group,
splits by group with the fixed seed, fits the pipeline on the training groups,
scores on the held-out ones, exports, writes the metadata into the file, and then
**reloads the file with onnxruntime and predicts the held-out rows again**. If
that disagrees with scikit-learn beyond float32 tolerance the export is broken
and the job fails rather than shipping a model that would be quietly wrong.

`Training.Runtime` in C# finds the trainer in this order: an `OTTERLOGIC_TRAINER`
environment variable, for development against a checkout's `.venv`; then the
installed folder under `%LOCALAPPDATA%`. The variable exists so that Train and
Predict can be built and proven before the bundle exists.

The bundle is python-build-standalone 3.12 — matching the fixtures — with numpy,
scikit-learn, skl2onnx, onnx and onnxruntime, zipped, hashed, and attached to a
GitHub release of this repo. It changes when the trainer changes, not with every
plug-in release. PyTorch is not in it; phase 3 adds a second, larger bundle.

## Where the code lives

| Repo | Adds |
|---|---|
| MachineLearning | `src/OtterLogic.MachineLearning.Inference/` — its own project, because of the native binaries: `OnnxModel` (load, metadata, predict), `ModelMetadata`. `Training/` in the main project: `TrainingJob`, `TrainerProcess`, `Runtime` (find, install, verify hash), `ModelType`. `python/trainer/` — the package. |
| Supervised | Nothing in C#. The Evaluate components already score whatever Predict returns. |
| Rhino3D | `Components/MachineLearning/PredictComponent.cs`, `Components/SupervisedLearning/TrainComponent.cs`, the Model Type dropdown. `Microsoft.ML.OnnxRuntime` natives beside the `.gha`. |

The Inference project does **not** set `ExcludeAssets="runtime"` on the ONNX
Runtime package: those natives genuinely must ship. It is separate from the main
project so that a test runner for k-means never loads them.

## Build order

1. **The contract.** `ModelMetadata` in C# with its JSON and tests; the same
   record in Python; the fixed input and output names. Half a day, and both
   halves depend on it.
2. **Use.** The Inference project, `OnnxModel`, the Predict component. Proven by
   rewriting `train_surrogate.py` to write the metadata and fixed names, training
   on a synthetic sweep, and predicting the held-out rows in Grasshopper to
   float32 agreement with Python. The C# parity fixture — rows in, values out —
   is committed under `tests/OtterLogic.MachineLearning.Inference.Tests/Fixtures/`.
3. **Create, against a checkout.** The trainer package, the job protocol,
   `TrainerProcess`, the Train component, found through the environment variable.
   Proven by the same sweep from the Write Dataset component through to Evaluate
   Regression, and by a classification target through to Evaluate Classification.
4. **Install.** The bundle build script, the release, `Runtime.Install`, the two
   menu items. Proven on a machine with no Python on it.

Steps 2 and 3 can run in parallel once 1 is done. Step 2 ships on its own if 3
is late: a model trained from a terminal is still a model everyone else can use.

## Checks before it ships

- **ONNX Runtime beside another plug-in.** Untested, and the one item that could
  change packaging. Load OtterLogic next to a plug-in that carries its own
  `onnxruntime.dll` and see which one wins. Do this in step 2, first thing.
- **Yak package size** with the natives included.
- **Defender** on the unpacked `python.exe`: first-run time, and whether it
  quarantines. Document the folder either way.
- **The grep** of MachineLearning for domain vocabulary still comes back empty.
  Nothing in Train or Predict knows what a release is.

## What was stripped

| Planned | Became | Why |
|---|---|---|
| Load Model + Predict | Predict | Load Model's job was to say what the model expects; Predict's Report says it with nothing wired. A separate model object earns its place when several Predicts share one session, which no phase 1 use has. |
| ML Environment component | Train's context menu | A component that exists to be used once, then never again, is a permanent ribbon slot for a one-off. The menu appears on the component that needs the runtime, at the moment it is needed. |
| Train Classifier + Train Regressor | Train | The schema already says whether a target is a class or a number. Two components would ask the user to repeat what the dataset knows. |
| Sidecar `.json` | Metadata inside the `.onnx` | A model is one file, so it cannot arrive incomplete. |
| Install from file | Kept, as a menu item | It is the only answer to a blocked proxy and costs one file dialog. |
| LightGBM, GPU bundle, cross-validation by group, feature matching by name, a Load Model object | Later | Each waits for a reason: a table LightGBM beats sklearn on, a model that needs a GPU, a fifth project, a named table to match against. |

## Open

- **Score in the Report versus Evaluate.** Train reports its own held-out score
  because a user needs to see it without wiring six more components. Evaluate
  stays the way to compare a trained model against the C# baselines on the same
  held-out groups; the two should agree, and a test should say so.
- **Cancellation.** Killing the trainer process on Run's falling edge is easy;
  cleaning up a half-written `.onnx` is the detail to get right. Write to a
  temporary name and rename on success.
- **Neural Network options.** Hidden layer sizes and iterations are fixed in
  phase 1 (one layer of 64, 500 iterations, no early stopping — scikit-learn's
  needs a validation slice that small datasets cannot spare). A user who needs
  more is in phase 3 territory. Revisit if the sweep says otherwise.

## What steps 1 to 3 turned out to be

Built as planned, with these decisions made along the way:

- **The metadata record is exactly as drawn above**, twice: `ModelMetadata` in
  the Inference project and `metadata.py` in the trainer. `score` is a flat map of
  names to numbers rather than a typed record, so a trainer can report what suits
  the task without the C# type changing; the names are the Evaluate components'.
- **The parity fixture is two real models.** `make_inference_fixtures.py` has the
  trainer train a classifier and a regressor on a synthetic sweep and records what
  onnxruntime in Python answers for five probe rows; the C# tests open the same
  files and must answer the same. Committed as `.onnx` files of about 65 KB each.
- **skl2onnx 1.20 and protobuf 7 disagree**, and every boosted-tree export failed
  on it: the tree converter passes a Python `False` where protobuf 7 now insists
  on an integer. The trainer coerces the attribute before the node is built
  (`export.py`, a dozen lines with the reason beside them). Pinning `onnx` did not
  help — the check is protobuf's — and pinning scikit-learn back would have
  changed the fixture environment for the whole repo.
- **Train's runtime menu says where it looked and stops there.** Installing from
  it is step 4. Until then `OTTERLOGIC_TRAINER` names a checkout's virtual
  environment with the trainer installed into it, and Rhino inherits the variable
  from the shell that started it.
- **The natives are copied up beside the `.gha`** by a target in the Grasshopper
  project, because Rhino's probing does not know `runtimes/win-x64/native`. The
  managed assembly's own folder is probed, and that is where they now sit.
- **The C# end-to-end tests run the real trainer** when a runtime is found — the
  environment variable, or the checkout's own `python/.venv` — and pass with a
  note when none is, so a hosted runner stays green. The Python tests run in a
  second CI job so a broken wheel does not read as a C# failure.
- **Prediction is a generic output.** Class names go out as text and values as
  numbers on one Prediction output, which the Evaluate components read either way.

Not done, and worth knowing before trusting it: neither component has been
opened in Grasshopper, so the polling re-solve in Train and the file-change cache
in Predict are exercised only by reading the code; the ONNX Runtime conflict with a
second plug-in carrying its own copy is still untested; and no release carries a
bundle yet, so the install menu items have nothing to fetch until one is cut.

## What changed after phase 1

Three things moved once the pieces above were in hand. Each is recorded here
because it contradicts a line earlier in this file, and this section wins.

**Learners replace Model Type.** The bare enum could name a method and nothing
about it, and the first thing a user asks of a neural network is how big. So
`ModelType` is gone and `Training/Learner.cs` holds an abstract record `Learner`
with four concrete ones, each carrying only what a person who has looked the
method up would expect to set:

| Learner | `type` | Settings (default) |
|---|---|---|
| Boosted Trees | `boostedTrees` | `trees` (100), `depth` (0 = no limit) |
| Neural Network | `neuralNetwork` | `hiddenLayers` ([64]), `iterations` (500) |
| Linear Model | `linear` | `regularisation` (1.0): ridge's alpha, and 1/C for logistic regression |
| Nearest Neighbours | `nearestNeighbours` | `neighbours` (5) |

`TrainingJob.Learner` goes into `job.json` as one object with a `type`
discriminator — `"learner": {"type": "boostedTrees", "trees": 100, "depth": 0}` —
which is what `models.py` reads; everything else about a method (learning rate,
tolerance, early stopping) stays fixed in the trainer, because a knob nobody can
read is a knob set wrong. The two new learners earn their place as comparisons: a
linear model that does as well as boosted trees says the relationship is a line
and the trees are not earning their keep, and nearest neighbours assumes nothing
about its shape. All four pass the round-trip parity check on both tasks. The
model's metadata still records the type under `trainer.modelType`, because the
C# `ModelMetadata` reads that key and the committed fixtures carry it.

**Rows can be held out, with a caveat.** The plan held out whole groups and
nothing else, and it still does whenever there are groups to hold. But samples
wired straight into Train from a sweep may come with no group at all, and a
score is better than none. `TrainingJob.HoldoutBy` is `Group` unless
`TrainerProcess.StartOnSamples` — which writes wired samples into a dataset folder
under the job's work folder, one model per distinct group, through
`SampleFolder` — finds fewer than two distinct groups, when it chooses `Row`. The
trainer then holds out rows at random with the seed and puts one line first in
the report: rows were held out at random because no groups were given, so the
score may be optimistic, because rows from one model resemble each other. The
number is never shown without the sentence.

**The install API is built; the release is not cut.** `TrainerRuntime` now has
the whole of step 4 but the release: `InstallAsync` reads the manifest at
`releases/latest/download/trainer-manifest.json` (a `TrainerBundle`: version, URL,
SHA-256, size), downloads the zip, checks its hash and unpacks it;
`InstallFromFile` takes the same zip from a share. A bundle is a zip with
`bundle.json` at its root naming its version and where its `python.exe` is. Both
unpack into `InstallRoot/<version>.installing` and rename into place at the end,
so a cancelled or failed unpack leaves nothing `Find()` could mistake for an
install, and a zip entry that climbs out of its folder is refused before anything
is written. `python/trainer/build-bundle.ps1` builds the zip on
python-build-standalone 3.12, `pip install`s the checkout into it, and writes the
manifest; the trainer answers `--version` so the build, and later an installer,
can prove a runtime works without a job. What remains is to run the script and
attach what it writes to a release tagged `trainer-v<version>`.
