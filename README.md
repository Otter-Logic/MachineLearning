# MachineLearning

The shared base every learning paradigm needs: feature preparation,
decomposition, the dataset contract, and the ONNX plumbing.

A layer rather than a domain. It sits above
[Core](https://github.com/Otter-Logic/Core) and below the paradigm repos, and any
of them may reference it. See [docs/architecture.md](docs/architecture.md) for why
that exception exists.

```
                       Core
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

**Graphs** — `WeightedGraph`: which samples are related and how strongly, as
sorted sparse rows, with the nearest-neighbour graph of a point cloud and the
symmetric normalised propagation every graph method shares. Spectral clustering,
constrained hierarchies and message passing in Unsupervised consume it today, and
a trained graph network in DeepLearning will want exactly the same input — which
is the test for living here.

**Data** *(planned)* — the dataset contract, the sweep recorder, train/test
split. Phase one of [docs/machine-learning.md](docs/machine-learning.md), and the
part people skip and then regret.

**Inference** *(planned)* — a separate project, because it is the one genuine
dependency seam. `Microsoft.ML.OnnxRuntime` has native binaries, and a toolkit
doing nothing but k-means should not drag them onto a test runner. Model loading,
the sidecar column-order assertion, and tensor marshalling go here.

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
| Is there anything to ship? | **no** | `.onnx` + sidecar |
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
  Graphs/           the weighted graph every graph method consumes
python/             development only - never ships, never installed by a user
  fixtures/         scikit-learn reference fixtures for the C# tests
tests/              xunit; runs anywhere, no Rhino needed
models/             .onnx artefacts, once there are learned models
docs/               architecture and design notes
```

## Dependencies

None today, deliberately. A small symmetric eigensolve is sixty lines of Jacobi
rotation and a large sparse one a filtered subspace iteration, neither worth
carrying MathNet for.
`Microsoft.ML.OnnxRuntime` arrives with the Inference project, and that one is
unavoidable — leave `ExcludeAssets="runtime"` off it, unlike RhinoCommon, because
its native binaries genuinely must sit next to the `.gha`.
