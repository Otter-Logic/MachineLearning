# MachineLearning

Clustering, dataset capture and model inference for the OtterLogic Rhino tools.

A layer rather than a domain: it sits above [Core](https://github.com/Otter-Logic/Core)
and below the toolkits, and a toolkit may reference it. See
[docs/architecture.md](docs/architecture.md) for why that exception exists.

## What is here

**Design grouping** — principal component analysis and a Gaussian mixture model,
written in C#, fitted at solve time on the values coming down a Grasshopper wire.
Given the six-degree-of-freedom demand on each member, it finds the families of
members that want the same connection detail, and says how confident it is about
each one.

Nothing about it is pre-trained. A mixture computes its own parameters from the
data in front of it, so there are no weights to ship and no Python at runtime —
this is an algorithm, not a model.

## Layout

```
src/OtterLogic.MachineLearning/   the library, published as a NuGet package
  Clustering/                     PCA, EM, the preprocessing pipeline
python/                           development only - never ships, never installed by a user
  fixtures/                       scikit-learn reference fixtures for the C# tests
tests/                            xunit; runs anywhere, no Rhino needed
models/                           .onnx artefacts, once there are learned models
docs/                             architecture and design notes
```

## The pipeline

```
log1p  ->  standardise  ->  weights  ->  PCA (+ whiten)  ->  Gaussian mixture
```

Each step earns its place, and the order matters:

- **log1p** because member demands are heavily right-skewed. A handful of members
  carry most of the load and a long tail carries very little, which is not a
  shape a Gaussian describes. Without this, one component swallows the tail while
  the rest split hairs among the small values. It changes results more than any
  other single setting.
- **standardise** because forces in kN sit beside moments in kNm with no shared
  scale.
- **weights** to say that moments matter more than shears, or to drop a column
  outright. Applied here, before the decomposition, because that is the only
  place they do anything — see below.
- **PCA** to decorrelate, to whiten, and to drop directions a structure genuinely
  has no variance in. At six columns this is not dimensionality reduction.
- **mixture** with diagonal covariance, which is the right partner for whitened
  components.

### Why the weights go before the PCA

A Gaussian mixture with free covariance is **invariant** to scaling a column.
Multiply column *j* by *w* and the fitted mean and standard deviation scale with
it; every component's density picks up the same constant factor and it cancels in
the responsibility normaliser. The grouping comes out identical.

```
p'(wx) = (1/w) p(x)     for every component, so r_ik is unchanged
```

So a weight input bolted onto a bare mixture would appear broken. What makes
weighting real is the eigendecomposition: PCA is emphatically not scale
invariant, and a column carrying more variance pulls the principal axes toward
itself. Weight applied before PCA changes the rotation, changes which directions
survive the variance threshold, and therefore changes the answer.

## How it compares to scikit-learn

The tests check two different things, because "does it agree with sklearn"
is two questions.

**Exactness.** Started from identical parameters, the C# EM matches
`sklearn.mixture.GaussianMixture` to 1e-8 on weights, means, covariances and
responsibilities, for diagonal, full and spherical covariance — and agrees on
iteration count, convergence flag, parameter count, BIC and AIC. Pinning the
start is what makes this a test of arithmetic rather than of luck.

**Quality.** Left to find their own starting points, the two are level. On 300
members in four families, both reach BIC 2308.480 at k = 4 and produce the
*identical* partition (adjusted Rand index 1.0000). Across a sweep of k:

| k | scikit-learn BIC | OtterLogic BIC |
|---|---|---|
| 2 | 2988.907 | 2988.907 |
| 3 | 2654.840 | 2654.840 |
| 4 | 2308.480 | 2308.480 |
| 5 | 2164.635 | 2164.635 |
| 6 | 2063.465 | 2066.528 |
| 7 | 1970.054 | **1970.045** |
| 8 | 1898.042 | **1898.023** |

Neither dominates. The differences at k >= 6 are the local-optimum lottery, not a
difference in capability.

## Working on it

```
dotnet test
```

The tests need no Rhino and no licence — the clustering pipeline is pure numerics
— so unlike the StructuralForm tests, these run on a hosted CI runner.

To regenerate the reference fixtures, see [python/README.md](python/README.md).
