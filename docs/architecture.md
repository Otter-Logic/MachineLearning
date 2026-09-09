# Architecture

## Where this sits

```
                    OtterLogic.Core
              small, stable, slow-moving
                          ↑
              OtterLogic.MachineLearning          ← this repo
        cross-cutting — a toolkit may reference this one
                          ↑
              OtterLogic.StructuralForm           (+ Fabrication, FormFinding, ...)
                    ↑              ↑
        OtterLogic.Rhino    OtterLogic.Grasshopper
```

The rule everywhere else in OtterLogic is that domains never reference each
other, and that a domain needing another domain means something belongs in Core.
This layer is the one sanctioned exception.

Machine learning is not a discipline alongside trusses and nesting; it is
something every discipline wants to apply to its own results. Left as a domain it
would be unreachable — Fabrication could not cluster its panels without either a
domain-to-domain reference or its own copy of the algorithm.

Pushing it down into Core instead was the alternative, and it fails Core's own
test: it would drag a native ONNX runtime into the foundation every domain
compiles against whether or not it does any inference, and turn Core into the
grab-bag it is meant not to be.

So this sits between the two. It references Core; toolkits may reference it; it
references no toolkit. One-way, so there is still no cycle. It is also a toolkit
in its own right — the clustering components ship under the **Machine Learning**
section — because being a shared layer does not stop it being a tool.

## Two kinds of model, and only one of them has a bridge

This is the distinction the whole repo is organised around.

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

The clustering pipeline references nothing. Not Accord.NET, which was archived
read-only in November 2020 with its last stable release in 2017 — and which
LunchBox already loads into Grasshopper, so shipping a second copy at a different
version is an assembly conflict waiting for a user who has both. Not ML.NET,
which is healthy and actively released but has no Gaussian mixture at all, offers
only a randomised PCA built for matrices far larger than six columns, and wants
data as a lazy columnar `IDataView` rather than a list on a wire. Not
MathNet.Numerics, which would be carried for a 6x6 symmetric eigensolve that is
sixty lines of Jacobi rotation.

`Microsoft.ML.OnnxRuntime` arrives when inference does, and that one is
unavoidable.

## What the pipeline does, and why in that order

```
log1p  ->  standardise  ->  weights  ->  PCA (+ whiten)  ->  Gaussian mixture
```

The preprocessing is the part that decides whether a grouping is any good, and it
is the part no library hands you. Fitted to raw six-degree-of-freedom magnitudes
a mixture will return an answer; it will usually be a poor one, because the input
breaks two assumptions at once — heavily right-skewed where the model expects
something Gaussian, and forces beside moments with no shared scale.

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

### Things that are not optional

- **Log-space arithmetic with log-sum-exp.** At six dimensions with well-separated
  groups, computing densities and then normalising underflows every term to zero.
- **A regularisation floor on every covariance diagonal.** Without it a component
  can collapse onto duplicate rows, its variance heading to zero and the
  likelihood to infinity. Structural models are full of identical members, so
  this is a realistic input rather than a contrived one.
- **A fixed seed.** Grasshopper re-solves on any upstream change, and a component
  returning different groups from identical inputs is unusable.
- **Ordering components by mixing weight before returning them.** EM numbers its
  components in whatever order initialisation produced. Without a canonical
  order, a small upstream change permutes them and every downstream colour and
  geometry assignment jumps for no visible reason.

## What this repo is, and what it is not

This is the **engine**. It holds the methods, each exposing every setting, and it
takes no view on which one anybody should use.

The layer that takes a view is
[Clustering Tool](https://github.com/Otter-Logic/Clustering_Tool), a toolkit
above this one. It runs all three clustering methods over structural demand data
and picks between them, so that a user with analysis results needs no opinion
about covariance shapes. Its rules about *which* method suits *what* data belong
there and not here — this repo would be wrong to have an opinion, because
Fabrication clustering panels wants the same algorithms and different judgement.

```
Core  ->  MachineLearning  ->  Clustering Tool
          (this repo)          (the end product)
```

## The three clustering methods

Not three implementations of one idea. Each assumes something different about
what a cluster is, and which assumption holds is a property of the data.

| | Told how many? | Every sample placed? | Cluster shape |
|---|---|---|---|
| `KMeans` | yes | yes | round, similar size |
| `GaussianMixture` | yes | yes, softly | any ellipsoid, may overlap |
| `Hdbscan` | **no** | **no** | any shape, density-defined |

`KMeans` is public in its own right but is also EM's seeder — there is one copy,
and the mixture calls it. A second would be a second thing to keep in step with
the scikit-learn parity fixtures.

`Hdbscan` is core distances, a mutual reachability MST by Prim's, a condensed
cluster tree, and excess-of-mass extraction. The MST is O(n²) with no distance
matrix held in memory: at a few thousand members in three components that is
milliseconds and a few kilobytes, and it avoids the spatial index the reference
implementation needs at scale.

`ClusterQuality` holds silhouette and Davies-Bouldin. Both exclude noise rather
than scoring it as a cluster — noise is not a group and has no centre, and
counting it as one would punish HDBSCAN for the thing it exists to do. Both are
also documented as what they are: compactness measures, which makes them useful
for comparing partitions and actively misleading for comparing *algorithms*.

## Grasshopper

Components live in the Rhino3D repo — nothing here touches the Grasshopper API.

Under **Machine Learning**, the three methods raw, one component each, following
LunchBoxML's shape so a definition already wired for those can be retargeted:
Training Inputs in, Result out.

- **K-Means Clustering** — clusters, restarts, seed. Out: result, clusters,
  centroids, inertia.
- **Gaussian Mixture** — components, covariance, restarts, seed. Out: result,
  groups, probability, confidence, centres, likelihood, BIC.
- **HDBSCAN Clustering** — minimum cluster size, minimum samples. Out: result,
  clusters, noise, probability, stability, noise fraction.
- **Covariance Type** — an `EnumValueList` dropdown, following `TrussType`.

Grasshopper only. These are wire-data tools with no document-level shape, so
nothing here gets a Rhino command or a toolbar button.

The **centres** output matters more than it looks. A grouping nobody can name is a
grouping nobody will act on, and mapping the cluster centres back into the
original units is what lets somebody say "group three is the high-moment family".
