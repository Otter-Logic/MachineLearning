# User-trained models — plan

Decided 2026-09-20. Status: **phase 0 built** the same day — the dataset contract
here, the three baselines and the scores in Supervised, ten components in Rhino3D.
Compiled and unit-tested; not yet run inside Grasshopper. Phases 1 to 4 are
planned. See [What phase 0 turned out to be](#what-phase-0-turned-out-to-be).

The aim is a set of Grasshopper components that let an end user feed in their own
data, train their own model on their own machine, and use it inside Grasshopper —
what LunchBox ML offered, without Accord.NET, and with a route to graph and
self-supervised models that LunchBox never had.

## A third kind of model

[architecture.md](architecture.md) sorts models into two kinds by one question:
*are there numbers that had to be learned from data the user does not have?* This
plan adds the case that question does not cover — numbers learned from data **only
the user has**.

|  | Fitted at solve time | Learned offline, shipped | Learned by the user |
|---|---|---|---|
| Example | Gaussian mixture, k-NN, ridge | a GraphSAGE we train | a release classifier trained on a firm's past models |
| Where do the parameters come from? | the data on the wire | our corpus | the user's dataset folder |
| Where does training run? | — | developer machine | **the user's machine, outside Rhino** |
| What crosses into Grasshopper? | nothing | `.onnx` + sidecar | `.onnx` + sidecar |
| Runtime language | C# | C# via ONNX | C# via ONNX |

The third column reuses the second column's bridge unchanged. That is the point
of it: a model is an `.onnx` and a sidecar `.json` whoever trained it, so Predict
is one component and never learns where its model came from.

It does retire one sentence. [python/README.md](../python/README.md) opens with
"no user of the plug-in installs any of this". That stays true of the fixtures and
of anything that runs at solve time; it stops being true of training. Update that
file and the two-kinds table in architecture.md when phase 1 lands, not before.

## Decisions

### ONNX is the boundary; the trainer behind it is replaceable

The contract is **job `.json` in, `.onnx` + sidecar `.json` out**, with progress
as JSON lines on a file the Train component polls. Nothing on the C# side knows
what produced the model. That is what lets scikit-learn, LightGBM and PyTorch sit
behind one Train component and one Predict component, and it is what keeps the
trainer decision below reversible.

ONNX carries the computation only. Column names, column order, class lists, the
feature-extractor version and the training metrics travel in the sidecar, and
Load Model asserts on them. Scaling is fitted on the training rows and baked into
the ONNX graph, so C# never has to repeat it and cannot repeat it wrongly.

Each model type offered needs its export tested once — skl2onnx, onnxmltools and
`torch.onnx` each have operators they handle badly. A model type is not offered
until a round-trip parity test passes.

### Bundled Python trainer, out of process

Training runs in a private Python the plug-in ships as a bundle. The user never
installs or manages Python, never sees pip or a terminal, and nothing touches a
Python already on the machine.

What was rejected, and why:

- **ML.NET as the core.** Its one real advantage is training with nothing to
  install, and the pure C# baselines give that. It has no route to graph or
  self-supervised models, and `IDataView` is the wrong shape for lists on a wire.
  Its LightGBM trainer does export ONNX, so it remains a credible *additional*
  backend behind the same contract if zero-download training ever proves to
  matter.
- **TorchSharp.** It does not improve the user's experience: libtorch is ~250 MB
  either way, so it cannot go in the Yak package and still needs a first-use
  download. What it changes is the developer's experience — one language — and
  the price is no ONNX export (so either libtorch inside Rhino for every
  predict-only user, or a hand-written exporter per architecture), no gradient
  boosting, and no PyTorch Geometric or reference implementations to borrow for
  phases 3 and 4.
- **Rhino 8's own CPython** (`~/.rhinocode`). No download, but it is Python 3.9,
  which current scikit-learn and PyTorch have dropped; McNeel owns the folder and
  may change it; and a pip install still meets the corporate proxy. The
  environment depended on should be one we control.
- **PyTorch in-process** via Rhino's embedded Python. A training loop in
  `SolveInstance` blocks the canvas and a GPU driver fault becomes a Rhino crash.

### Gradient boosting before neural nets

PyTorch is one backend, not the headline. On member-level tables under roughly
10k rows gradient boosting is usually both more accurate and far quicker to train
than a neural net, so the Train component has a model-type dropdown and its
default is boosted trees. PyTorch earns its place when the input has structure a
vector cannot hold — the element graph — and for self-supervised pretraining.

### Supervised first, self-supervised second

Supervised first because the labels already exist: past analysis models carry
their releases and section assignments. Evaluation is a confusion matrix on
held-out projects, which a user can read. It also closes the loop the
unsupervised work opened — clustering proposes labels, the user corrects them, a
classifier learns from the corrections.

Self-supervised second, in DeepLearning: masked-attribute prediction or a graph
autoencoder over the element graph. Its embeddings become the deferred
autoencoder view in `MultiViewClustering`, and pretrained features for users with
few labels.

A supervised model is also a QA tool before any of that: where a trained model
confidently disagrees with the modelled release or section, flag the member.

## What the user experiences

1. **Install OtterLogic from the package manager.** Clustering, the C# baselines
   and Predict all work immediately. ONNX Runtime's natives are small enough to
   ship inside the package.
2. **Receive a trained model.** Drop in the `.onnx` and `.json`, wire up Predict.
   Nothing else is needed. In practice one person per firm trains and everyone
   else predicts, so most users stop here.
3. **Press Train for the first time.** The component reports that the training
   runtime is not installed and offers an Install toggle. One archive, pinned to
   a version and checked against a hash, downloads from GitHub releases and
   unpacks under `%LOCALAPPDATA%\OtterLogic\trainer\<version>\`. No admin rights,
   no PATH change.
4. **Train.** The runtime changes only when the trainer changes, not with every
   plug-in release.

The package manager cannot do step 3 for us — a Yak install is an unzip with no
post-install hook — and the runtime does not belong in the `.yak`, where every
plug-in update would re-download it.

The download is an explicit click, not silent. Hundreds of megabytes fetched
without asking is poor behaviour on a firm network, and Rhino 8's script editor
has already taught users the pattern.

| Bundle | Contents | Rough size | Needed for |
|---|---|---|---|
| Light | Python, scikit-learn, LightGBM, ONNX converters | ~100 MB | all tabular training — phases 1 and 2 |
| Deep learning | adds CPU PyTorch | ~250 MB more | phase 3 onwards |
| GPU | CUDA PyTorch | 2 GB+ | opt-in only |

Sizes are estimates until the first bundle is built.

Known friction, and the answer to each:

- **Proxies that block GitHub.** An "install from file" input, so IT downloads
  the archive once and points the component at a network share.
- **Antivirus.** Defender will scan an unpacked `python.exe`; the first run is
  slow and occasionally quarantined. Sign the archive and document the folder.
- **ONNX Runtime conflicts.** Another Grasshopper plug-in may load a different
  version of the native DLL. Untested; check before phase 1 ships.

## How a model becomes data

Two things are easy to conflate. The **source** is whatever the model lives in —
Robot, GSA, Karamba, IFC, a results CSV. The **dataset** is what comes out after
feature extraction, and it is always a table.

```
Robot / GSA / Karamba / IFC             sources — many, messy, software-specific
        ↓   existing Grasshopper plug-ins; a dedicated reader later
lines, surfaces, supports, attributes   generic wires — what the Insight Engine already takes
        ↓   feature extraction, C#, in the domain toolkit
one row per sample                      the dataset — one format, software-agnostic
```

**A row is the thing being predicted about.** A member end for releases, so two
rows per member; a member for sizing; a whole design for a sweep surrogate.

```
member_id, end, length_ratio, angle_to_vertical, members_at_node, is_support_node, centrality, … | rel_My, rel_Mz, rel_Mx
```

**One CSV per model, in a dataset folder:**

```
datasets/member-releases/
  schema.json                        column names, roles (feature / target / id), units,
                                     class lists, feature-extractor version
  models/2024-017-stadium-roof.csv
  models/2025-003-office-block.csv
  edges/2024-017-stadium-roof.csv    phase 3 only: source_id, target_id
```

- **Writes are idempotent.** Grasshopper re-solves constantly; an append would
  duplicate rows on every slider move. A file keyed by model ID is overwritten.
- **The group comes free.** The filename is the project, and the trainer splits
  by it. Members within one model are strongly correlated, so a random row split
  reports an accuracy the model will never reach on a new project. There is no
  option to split by row.
- **Curation is deleting a file.**
- **CSV is enough.** 100k rows by 40 columns is about 30 MB. Parquet can wait
  for a reason.

The trainer reads every CSV, validates each against `schema.json`, tags rows with
their group, splits by group, fits scaling on the training rows only, trains, and
exports. At predict time a new model passes through the same extractor into the
same columns, then through ONNX.

Consequences that shape the design:

- **Feature extraction lives in C#, versioned, and the same code produces
  training rows and inference rows.** Two implementations would diverge and
  predictions would be wrong with no error raised. The sidecar records the
  extractor version and Load Model refuses a mismatch.
- **Features must not depend on the source software or the model's scale.**
  Ratios, angles, connectivity, normalised demand — not bar numbers, not raw
  section-name strings.
- **Section group is a harder target than it looks.** Cluster 3 in one model
  means nothing in another, so a raw group index cannot be learned across
  projects. The target needs a canonical form — section family, size rank within
  the model, a normalised section property. Releases mean the same thing
  everywhere, which is why they are the first target.
- **IFC rarely carries labels.** Most exports are the physical model (`IfcBeam`,
  `IfcColumn`), not the structural analysis view, so IFC supplies features but
  seldom releases. Analysis-software models are the training source; IFC is an
  inference-time input later.
- **Parsers are deferred.** Accept generic geometry and attribute lists first, so
  whatever plug-in the user already has can feed the dataset on day one. A
  dedicated Robot reader is a later convenience, not part of the core.

## Use cases, ranked

| Rank | Use case | Verdict |
|---|---|---|
| 1 | Analysis model → per-member attributes: end releases, then section group and member role | Thousands of rows per model, labels free, features already produced by the Insight Engine, every prediction checkable by an engineer. |
| 2 | Analysis results → sizing | An early-stage first guess or an optimiser seed, never a substitute for the code check. Predict a required property as a regression and pick from the catalogue with a fixed rule. It will learn the firm's conservatism along with its judgement. |
| 3 | Fabrication: part or connection family from corrected cluster labels; hours or cost from part features | The same trainer with different columns — the test that the tools really are generic. |
| 4 | IFC → analysis model | Not end to end: mostly deterministic geometry, paired data is rare, errors are intolerable. The slices that suit learning are per-element classifications — load-bearing, role, release — which is use case 1 with BIM features. |
| Example file | Parameter-sweep surrogate, e.g. a Karamba sweep → deflection | Unlimited data and no domain code, so it exercises the whole pipeline end to end. Ships as the worked example, not as a product. |

## Components

Two panels. The dataset sits under **Machine Learning**, which `Core.Sections`
already reserves for "dataset capture, feature preparation, inference" — what
every paradigm does before its method. The methods and their scores sit under
**Supervised Learning**, because raw methods go under their paradigm.

| Panel | Tier | Components | Built |
|---|---|---|---|
| Machine Learning | Data | Write Dataset · Read Dataset · Split By Group | phase 0 |
| Machine Learning | Inference | Load Model · Predict (label and confidence) | phase 1 |
| Machine Learning | Setup | ML Environment (runtime present, version, CPU or GPU, install, install from file) | phase 1 |
| Supervised Learning | Methods | Nearest Neighbour Classifier · Nearest Neighbour Regressor · Ridge Regression · Logistic Regression | phase 0 |
| Supervised Learning | Methods | Train Classifier · Train Regressor | phase 1 |
| Supervised Learning | Evaluation | Evaluate Classification · Evaluate Regression | phase 0 |

An earlier draft of this table put the dataset components under Supervised
Learning and had separate Build and Summary components. Build folded into Write —
a table is validated whether or not it is written, so the Write toggle is the only
difference — and Summary became the Report output of both Write and Read, so it is
seen without being asked for.

Train never blocks the canvas: it launches the trainer process, polls the
progress file, and expires its own solution when the job ends. It is driven by a
Run toggle rather than by upstream changes, the seed is fixed, and the model path
and a hash of the file are outputs so a definition can tell when the model
changed.

## Where the code lives

| Repo | Holds |
|---|---|
| MachineLearning | `Data` — table, schema, roles, group, folder IO, later edges. `Inference` — its own project, because of the native binaries. `Training` — the job protocol, the process launcher, runtime discovery and install; no Grasshopper reference. The Python trainer's entry point and protocol. |
| Supervised | The C# baselines, options records, metrics. The tabular Python trainer modules. |
| DeepLearning | The graph and self-supervised Python trainer modules, and whatever C# they need to feed `MultiViewClustering`. |
| StructuralDesign | A generic Member Features table built from the Insight Engine's graph. Users wire their own labels; no engineering rule is baked in. |
| Rhino3D | Every component. |

The usual test applies: anything that needs to know what a release is belongs in
StructuralDesign, and a grep of the generic layers for domain vocabulary should
still come back empty.

## Build order

0. **Dataset contract, C# baselines, Evaluate.** No Python, no ONNX. Gives
   `Supervised` its first real algorithm, gives LunchBox-style fit-and-predict on
   day one, and sets the baseline every trained model must beat.
1. **Inference project, Python trainer (boosted trees and an MLP), runtime
   bundle and install, Train and Predict.** Proven on the sweep surrogate.
2. **Member Features, the end-release worked example, QA by disagreement.** Then
   section group, once its canonical target is settled.
3. **DeepLearning: a GNN node classifier on the element graph, then the
   self-supervised embedding view.**
4. **IFC-derived features and the fabrication examples.**

## What phase 0 turned out to be

Built as planned, with these decisions made along the way that later phases
inherit:

- **Class lists are append-only, in the order first seen.** The position of a
  class in `schema.json` is the index a model is trained on, so a class met for
  the first time in the tenth model goes on the end and every earlier index keeps
  its meaning. The price is that a misspelt class stays in the schema until it is
  edited out by hand, which is why Write Dataset warns on every new class.
- **Whether a target is a number or a class is declared, never inferred.** Classes
  are usually written 0 and 1.
- **Features are numbers only.** A categorical feature is refused with the advice
  to one-hot it. Revisit if the boosted-tree trainer wants native categoricals.
- **Holdout is by share of groups, not of rows**, so which projects are held back
  does not depend on which happen to be large. The row share is reported.
  Cross-validation by group — leave one project out — is the better use of five
  projects and is **not built**; it wants a component that takes a method, which
  the fit-and-predict shape does not offer yet.
- **An unchanged write does not touch the file**, and files use `\n` and
  round-trip doubles, so the same rows give the same bytes on any machine. With
  the repos inside OneDrive that is the difference between one upload and one per
  slider move.
- **The logistic fit is always a softmax**, two classes included, and its L2
  penalty is mandatory: cleanly separable classes — usual on small rule-made data —
  have no best fit without it.
- **Standardisation lives inside each model**, fitted on the training rows only.
  The ONNX models in phase 1 should do the same, baked into the graph.

Not done, and worth knowing before trusting it: the components compile against
the pinned Grasshopper API but have not been opened in Grasshopper; there are no
icons for them, so they show the fallback; and the measured gap between a split by
group and a split by row on real models — the number that belongs in
`GroupSplit`'s doc comment — has not been measured, because there is no real
dataset yet.

## Open questions

- **GNN export.** PyTorch Geometric's scatter operators export to ONNX badly.
  The phase 3 model should be written as plain-torch message passing over the
  propagation operator `WeightedGraph` already builds, and that needs proving
  before phase 3 is designed around it.
- **Canonical section-group target.** Family, within-model size rank, or a
  normalised property — decide with real models in hand, in phase 2.
- **Where the Python trainer modules live.** One package in this repo with
  paradigm modules contributed from Supervised and DeepLearning, or one package
  per paradigm repo. The bundle build has to gather them either way.
- **Yak package size limit.** Unverified; only matters if ONNX Runtime's natives
  turn out larger than expected.
- **Python version for the bundle.** Pin one, and match the fixtures' 3.12 unless
  a wheel forces otherwise.
