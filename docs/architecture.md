# Architecture

## Where this sits

```
                    OtterLogic.Core
              small, stable, slow-moving
                          ↑
              OtterLogic.MachineLearning          ← this repo
        the shared base every paradigm needs
                          ↑
   Unsupervised   Supervised   Reinforcement   DeepLearning
      one repo per paradigm — siblings, never referencing each other
                          ↑
   OtterLogic.StructuralForm   OtterLogic.StructuralDesign   (+ Fabrication, ...)
                    ↑              ↑
        OtterLogic.Rhino    OtterLogic.Grasshopper
```

The rule everywhere else in OtterLogic is that domains never reference each
other, and that a domain needing another domain means something belongs in Core.
This stack is the one sanctioned exception.

Machine learning is not a discipline alongside trusses and nesting; it is
something every discipline wants to apply to its own results. Left as a domain it
would be unreachable — Fabrication could not cluster its panels without either a
domain-to-domain reference or its own copy of the algorithm.

Pushing it down into Core instead was the alternative, and it fails Core's own
test: it would drag a native ONNX runtime into the foundation every domain
compiles against whether or not it does any inference, and turn Core into the
grab-bag it is meant not to be.

So this sits between the two. It references Core; the paradigm repos above
reference it; it references none of them. One-way, so there is still no cycle.

## What this repo is, and what it is not

This is the **base**, not the algorithms. It holds what *more than one paradigm*
needs and nothing else:

| Namespace | Holds | Why it is shared |
|---|---|---|
| `Preprocessing` | `FeaturePipeline` — log1p, standardisation, weights, the constant-column check, and the inverse of all of it | every paradigm scales its features |
| `Decomposition` | `PrincipalComponents`, the `SymmetricEigen` Jacobi solver under it, and `LeadingEigen` for large sparse operators | PCA is a preprocessing step for a regressor as readily as it is an unsupervised tool; a spectral embedding is as useful to a graph network as to a clustering |
| `Graphs` | `WeightedGraph` — sparse weighted edges, the nearest-neighbour graph, the normalised propagation operator | spectral clustering and message passing in Unsupervised read it today; a trained graph network in DeepLearning will read the same thing |
| `Data` *(planned)* | dataset contract, sweep recorder, train/test split | a supervised sweep and an RL rollout both need somewhere to put rows |
| `Inference` *(planned)* | ONNX load, the sidecar column-order assertion, tensor marshalling | every learned model, whatever trained it, comes back the same way |

The algorithms live one layer up. Clustering is in
[Unsupervised](https://github.com/Otter-Logic/Unsupervised); regression and
classification will be in `Supervised`, and so on.

One test keeps this honest, and it is Core's test applied one layer higher:
*would a second paradigm plausibly need this?* If no, it belongs in the paradigm
repo. The failure mode to avoid is not drift — it is this becoming the grab-bag
that Core is not allowed to be.

### Why paradigm repos rather than one big one

Because they are the units that gain algorithms independently, and because the
alternative puts a supervised gradient-boosting import next to an HDBSCAN
condensed tree with nothing in common but the word "learning".

The cost of the split is real and worth naming: `FeaturePipeline` and
`PrincipalComponents` are used by more than one paradigm, so they had to land
*below* all of them rather than beside any one. That is exactly what this repo is
now for, and it is why the layer exists rather than the paradigm repos sitting
directly on Core.

Siblings never reference each other. If `Supervised` ever needs something in
`Unsupervised` — a k-means for feature construction, say — that something moves
down here. Same rule as Core, same reason.

### The one seam that earns a project boundary

`Inference` is a separate project inside this repo, not just a namespace, because
it is the only part with a dependency difference. `Microsoft.ML.OnnxRuntime` ships
native binaries; everything else here is arithmetic over arrays. Keeping them
apart means a toolkit doing nothing but k-means never drags a native runtime onto
a test runner that has no use for it.

Leave `ExcludeAssets="runtime"` **off** that package reference — unlike
RhinoCommon, its native binaries genuinely must be copied next to the `.gha`.

## Two kinds of model, and only one of them has a bridge

This is the distinction the whole stack is organised around.

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

A Gaussian mixture is emphatically the second kind. Its means, covariances and
mixing weights are computed from the input by EM, on every solve. There is no
training phase, nothing to export, and nothing for PyTorch to do. Pushing it
through ONNX would freeze a *fitted* model's predict step, which is not what the
tool is for.

That is why phase one ships with no Python and no ONNX at all.

## No dependencies, deliberately

This repo references nothing. Not Accord.NET, which was archived read-only in
November 2020 with its last stable release in 2017 — and which LunchBox already
loads into Grasshopper, so shipping a second copy at a different version is an
assembly conflict waiting for a user who has both. Not ML.NET, which is healthy
and actively released but offers only a randomised PCA built for matrices far
larger than six columns, and wants data as a lazy columnar `IDataView` rather
than a list on a wire. Not MathNet.Numerics, which would be carried for a small
symmetric eigensolve that is sixty lines of Jacobi rotation and a sparse leading
one that is a filtered subspace iteration.

`Microsoft.ML.OnnxRuntime` arrives with `Inference`, and that one is unavoidable.

## The preprocessing, and why it is the part that matters

```
log1p  ->  standardise  ->  weights  ->  PCA (+ whiten)
```

The preprocessing is what decides whether a grouping — or a regression — is any
good, and it is the part no library hands you. Fitted to raw six-degree-of-freedom
magnitudes a mixture will return an answer; it will usually be a poor one, because
the input breaks two assumptions at once: heavily right-skewed where the model
expects something Gaussian, and forces beside moments with no shared scale.

*Which* of these steps suits *what* data is deliberately not decided here. That
is a claim about a discipline, and it belongs in the toolkit that has one —
[StructuralDesign](https://github.com/Otter-Logic/StructuralDesign) chooses no
log transform and three components because it knows the columns are demands.
This repo offers the steps and takes no view, which is why every switch on
`FeaturePipeline` defaults off.

### Weighting only works because PCA is there

A mixture with free covariance is invariant to scaling a column. Scale column *j*
by *w* and the fitted mean and standard deviation scale with it, so every
component's density picks up the same factor and it cancels in the
responsibilities:

```
p'(wx) = (1/w) p(x)   for every component   =>   r_ik unchanged
```

Weights applied to a bare mixture would therefore do nothing at convergence. An
eigendecomposition is not scale invariant: a column with more variance pulls the
principal axes toward it. So weights go before the decomposition, and that is
where their effect comes from.

The invariance also breaks under spherical covariance, under the regularisation
floor, and through the initialisation — but PCA is the mechanism being relied on
here.

## Grasshopper

Components live in the Rhino3D repo — nothing here touches the Grasshopper API,
and nothing in this repo has a component of its own yet. The scaling and
decomposition steps become components under **Machine Learning** when the Data
and Inference work gives them something to sit beside.
