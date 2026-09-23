# MachineLearning

The shared base every learning paradigm needs: feature preparation,
decomposition, the dataset contract, and the ONNX plumbing.

A layer rather than a domain. It sits above
[Core](https://github.com/Otter-Logic/Core) and
[Graphs](https://github.com/Otter-Logic/Graphs) and below the paradigm repos, and any
of them may reference it. See [docs/architecture.md](docs/architecture.md) for why
that exception exists.

```
                  Core     Graphs
                        ↑
                 MachineLearning          ← this repo
                        ↑
   Unsupervised   Supervised   Reinforcement   DeepLearning
                        ↑
       StructuralForm   StructuralDesign   Fabrication
```

## What is here

**Preprocessing** — `FeaturePipeline`: log1p, standardisation, per-column
weights, and the constant-column check, with an inverse so an answer can be
mapped back into the units it arrived in. Every paradigm scales features, which
is what makes this shared rather than clustering's.

**Decomposition** — `PrincipalComponents` and the `SymmetricEigen` Jacobi solver
under it. PCA is a preprocessing step for a regressor as readily as it is an
unsupervised tool in its own right, so it lives below both. `LeadingEigen` finds
the few largest eigenpairs of a large sparse operator without forming it —
Chebyshev-filtered subspace iteration, fast enough for a spectral clustering of
3,000 samples in under half a second.

**Shapes** — `ShapeSignature`: describes every outline in a population by the
same row of numbers, learned from the population rather than chosen in advance.
Each outline is read at a fixed count of points evenly along its length, aligned
to the population's own average — where the reading starts is searched for
continuously, since a closed outline has no first point and the corner it was
listed from is an accident of drawing — and decomposed. A shape is then where it
sits among the others, which catches a notch, a curve or a raked corner without
any of them having been anticipated.

The scores are scaled so the distance between two rows *is* the root-mean-square
distance between the two outlines, so a tolerance on the signature is a tolerance
in model units. What counts as the same shape is the caller's to say: whether a
mirror is the same thing, whether size matters, and how many turns of it are
still it. Fabrication's panel typology is the first consumer, and a supervised
model over shapes would want exactly this, which is the test for living here.

**Distances** — `Euclidean`: the distance between two rows, and the exact
k-nearest search built on it, nearest first with ties to the lower index.
Every paradigm measures it — clustering to place a sample, a graph to decide
which samples to join, a nearest-neighbour regressor when there is one — and it
had been written out in six places above, one of them inside k-means. One copy
summed in one order is also what keeps results bit-identical wherever it is used.

**Distances** also holds `NeighbourGraph.Of`, the k-nearest-neighbour graph of a
sample matrix with scikit-learn's symmetrisation — the way into the graph methods
for a caller with only a point cloud. `NeighbourGraph.ByDistance` is the same
neighbours weighed the other way round — each edge the distance it spans, kept
if either end chose it, left out where a caller's predicate says it is blocked —
for a graph that is about to be routed over rather than clustered on, where a
high weight has to mean far and not alike. The graph they return, `WeightedGraph`, and
everything that runs on one — shortest paths, centrality, cut vertices, potential
flow, condensation — moved down into [Graphs](https://github.com/Otter-Logic/Graphs),
because a toolkit routing a toolpath wants them with no machine learning in
sight. This one constructor stayed: it measures a distance between samples, and
Graphs references nothing, so it cannot. `LeadingEigen` and everything above
still receive the same graph type, now from one layer further down.

**Data** — the dataset contract. A `DatasetSchema` says which columns a table has,
in which order, for what, and which version of the feature code produced them; a
`Dataset` holds rows against it; and `DatasetFolder` keeps one on disk as a
`schema.json` beside one CSV per model the rows came from. One file per model
because that is how the data is gathered — a model at a time, from Grasshopper,
over months — so a re-solve overwrites rather than appends, the file name is the
group, and curating the set is deleting a file. The schema is checked on every
write, which is what stops the twentieth model going in with two columns swapped.
`GroupSplit` holds back whole groups for testing and deliberately offers no split
by row: rows from one model are near-copies of each other, and a score from a
random split measures recognition rather than prediction. See
[docs/user-trained-models-plan.md](docs/user-trained-models-plan.md).

**Training** — the protocol between the Train component and the trainer process:
`TrainingJob` is the `job.json` a run starts from, and a `Learner` — boosted trees,
a neural network, a linear model or nearest neighbours, each with the few settings
a person would expect to set — is the method in it; `TrainerProgress` is one line
of the `progress.jsonl` it reports through; `TrainerProcess` starts the process and
polls the file without ever waiting on it, writing wired samples into a folder
first through `SampleFolder` when they did not come from one; and `TrainerRuntime`
finds the private Python it runs in, or installs one from a release bundle.
Nothing here references Grasshopper, so it is tested by running the real trainer
from xunit. The trainer itself is `python/trainer/`, and `build-bundle.ps1` beside
it makes the runtime a user installs.

**Inference** — `OtterLogic.MachineLearning.Inference`, a separate project because
it is the one genuine dependency seam: `Microsoft.ML.OnnxRuntime` has native
binaries, and a toolkit doing nothing but k-means should not drag them onto a
test runner. `OnnxModel` opens a model file and answers for rows; `ModelMetadata`
is what the file says about itself — features in order, target, classes, how it
scored — carried *inside* the `.onnx` under one `metadata_props` key rather than
in a sidecar, so a model is one file and cannot arrive incomplete. The graph
interface is fixed (`features` in; `value`, or `label` and `probabilities`, out),
which is what lets scikit-learn today and PyTorch later sit behind one reader. See
[docs/user-trained-models-phase-1.md](docs/user-trained-models-phase-1.md).

## What is deliberately not here

**The algorithms.** They moved out to the paradigm repos —
[Unsupervised](https://github.com/Otter-Logic/Unsupervised) has the clustering
today. What stays is what more than one paradigm needs, which is the same test
Core applies one layer down: *would a second paradigm plausibly need this?* If
no, it belongs in the paradigm repo.

**Any opinion about a discipline.** A toolkit owns its own feature extraction,
because what a truss considers a feature is not what a nesting problem does.

## Two kinds of model, and only one of them has a bridge

This is the distinction the repo is organised around.

|  | Fitted at solve time | Learned offline |
|---|---|---|
| Example | Gaussian mixture, PCA | GraphSAGE |
| Where do the parameters come from? | the data on the wire | gradient descent over a corpus |
| Is there anything to ship? | **no** | one `.onnx`, carrying its own metadata |
| Language at runtime | C# only | C# inference, Python training |
| Crosses the ONNX boundary? | **never** | always |

One question decides which you are looking at: *are there numbers that had to be
learned from data the user does not have?* If yes, it is trained in `/python` and
shipped as a frozen graph. If no, it is an algorithm and it is written in C#.

## Layout

```
src/OtterLogic.MachineLearning/
  Preprocessing/    scaling, weighting, the constant-column check
  Decomposition/    PCA, the symmetric eigensolver, the leading-eigenvector solver
  Shapes/           outlines in, a learned row of numbers per outline out
  Distances/        Euclidean distance, the k-nearest search, and the neighbour graph built on it
  Data/             the dataset contract: schema, table, folder on disk, split by group
  Training/         the job and progress protocol, the process launcher, runtime discovery
src/OtterLogic.MachineLearning.Inference/
                    ONNX Runtime behind OnnxModel and ModelMetadata - the one project with natives
python/
  fixtures/         scikit-learn reference fixtures for the C# tests, and the inference parity fixtures
  trainer/          otterlogic_trainer, the process behind Train; ships as a private bundle
tests/              xunit; runs anywhere, no Rhino needed
docs/               architecture and design notes
```

## Dependencies

None today, deliberately. A small symmetric eigensolve is sixty lines of Jacobi
rotation and a large sparse one a filtered subspace iteration, neither worth
carrying MathNet for.
`Microsoft.ML.OnnxRuntime` is the one exception, confined to the Inference
project — leave `ExcludeAssets="runtime"` off it, unlike RhinoCommon, because its
native binaries genuinely must sit next to the `.gha`, and the Grasshopper
project copies them up from `runtimes/win-x64/native` for exactly that reason.
