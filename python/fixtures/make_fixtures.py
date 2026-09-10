"""Generate scikit-learn reference fixtures for the C# decomposition and graph tests.

Run this, commit the JSON it writes, and the C# tests assert against it. Nothing
here runs at plug-in runtime — scikit-learn is a reference implementation, not a
dependency.

PCA is deterministic up to a sign, so this is a straight exactness check: given
the same standardised input, the C# decomposition should match scikit-learn's to
machine precision once the component signs are put in a canonical order. The
graph fixture is the same kind of check for the nearest-neighbour graph, the
normalised propagation operator and the leading-eigenvector solver.

The clustering fixtures moved to the Unsupervised repo along with the algorithms
they test.

    python python/fixtures/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA

OUT = Path(__file__).resolve().parents[2] / "tests" / "OtterLogic.MachineLearning.Tests" / "Fixtures"

DOF_NAMES = ["Fx", "Fy", "Fz", "Mx", "My", "Mz"]

# Relative demand across the six degrees of freedom for four member families,
# loosely shaped like what a moment frame produces: axial-dominated columns,
# beams carrying shear and major-axis moment, lighter secondary beams working
# about the minor axis, and transfer members with everything at once.
PROFILES = np.array([
    [1.00, 0.10, 0.08, 0.02, 0.35, 0.05],
    [0.15, 0.55, 0.10, 0.05, 0.80, 0.12],
    [0.10, 0.12, 0.50, 0.06, 0.10, 0.75],
    [0.60, 0.45, 0.40, 0.30, 0.55, 0.50],
])
SHARES = np.array([0.35, 0.30, 0.20, 0.15])


def make_members(n: int = 300, seed: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Six unsigned magnitudes per member, log-normally scaled within a family.

    The log-normal scale is the point. Real member demands are heavily
    right-skewed — a few members carry most of the load and a long tail carries
    very little — and that skew is what the log transform in the C# pipeline
    exists to handle. Fixtures drawn from a tidy Gaussian would test nothing.
    """
    rng = np.random.default_rng(seed)

    family = rng.choice(len(PROFILES), size=n, p=SHARES)
    scale = np.exp(rng.normal(loc=3.2, scale=0.9, size=n))
    noise = 1.0 + rng.normal(loc=0.0, scale=0.18, size=(n, 6))

    x = np.abs(PROFILES[family] * scale[:, None] * noise)
    return x, family


def standardise(x: np.ndarray) -> np.ndarray:
    """log1p then z-score with the population standard deviation.

    Population rather than sample, matching scikit-learn's StandardScaler and
    the C# FeaturePipeline.
    """
    logged = np.log1p(x)
    return (logged - logged.mean(axis=0)) / logged.std(axis=0)


def pca_fixture(x: np.ndarray, whiten: bool) -> dict:
    pca = PCA(n_components=None, whiten=whiten, svd_solver="full")
    z = pca.fit_transform(x)

    return {
        "name": f"pca_{'whiten' if whiten else 'plain'}",
        "whiten": whiten,
        "x": x.tolist(),
        "expected": {
            "mean": pca.mean_.tolist(),
            "explained_variance": pca.explained_variance_.tolist(),
            "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
            # Sign is arbitrary in an eigendecomposition, so the C# side applies
            # a convention (largest-magnitude entry positive) and this applies
            # the same one before comparing. Without it the test fails on a
            # difference that means nothing.
            "components": canonical_signs(pca.components_).tolist(),
            "transformed": (z * sign_flips(pca.components_)).tolist(),
        },
    }


def sign_flips(components: np.ndarray) -> np.ndarray:
    """+1 or -1 per component, making the largest-magnitude loading positive."""
    dominant = np.argmax(np.abs(components), axis=1)
    signs = np.sign(components[np.arange(components.shape[0]), dominant])
    signs[signs == 0] = 1.0
    return signs


def canonical_signs(components: np.ndarray) -> np.ndarray:
    return components * sign_flips(components)[:, None]


def graph_fixture(k: int = 6, clusters: int = 3, seed: int = 11) -> dict:
    """The nearest-neighbour graph, its normalised propagation, and its spectrum.

    Three exactness checks in one, all against dense reference arithmetic:

    *The graph.* scikit-learn's ``kneighbors_graph`` symmetrised as
    ``0.5 * (A + A.T)``, which is what ``SpectralClustering`` does with it - so an
    edge both ends chose weighs one and a one-sided edge a half.

    *The operator.* ``D^-1/2 (A + sI) D^-1/2 X`` for s = 0 (spectral clustering's
    normalised adjacency) and s = 1 (a graph convolution's renormalised one),
    computed densely.

    *The spectrum.* numpy's ``eigh`` on the normalised adjacency, shifted by the
    identity and halved so it is positive semi-definite, which is the operator the
    C# leading-eigenvector solver is handed.
    """
    from sklearn.datasets import make_blobs
    from sklearn.neighbors import kneighbors_graph

    x, _ = make_blobs(n_samples=90, n_features=3, centers=clusters, cluster_std=1.1, random_state=seed)

    connectivity = kneighbors_graph(x, n_neighbors=k, mode="connectivity", include_self=False)
    affinity = (0.5 * (connectivity + connectivity.T)).toarray()

    rows, cols = np.nonzero(np.triu(affinity, k=1))
    edges = [[int(a), int(b), float(affinity[a, b])] for a, b in zip(rows, cols)]

    def propagate(self_weight: float) -> np.ndarray:
        a = affinity + self_weight * np.eye(len(x))
        degree = a.sum(axis=1)
        scale = np.where(degree > 0, 1.0 / np.sqrt(degree), 0.0)
        return (scale[:, None] * a * scale[None, :]) @ x

    degree = affinity.sum(axis=1)
    scale = 1.0 / np.sqrt(degree)
    normalised = scale[:, None] * affinity * scale[None, :]
    shifted = 0.5 * (normalised + np.eye(len(x)))
    values, vectors = np.linalg.eigh(shifted)
    order = np.argsort(values)[::-1]

    return {
        "name": "graph",
        "neighbours": k,
        "x": x.tolist(),
        "expected": {
            "edges": edges,
            "propagate_plain": propagate(0.0).tolist(),
            "propagate_self": propagate(1.0).tolist(),
            # Values only. Well-separated blobs give a graph with an eigenvalue of
            # one repeated once per component, and inside a repeated eigenvalue
            # any rotation of the eigenvectors is as correct as any other - so the
            # C# side checks its vectors by residual, not against these.
            "leading_values": values[order][: clusters + 2].tolist(),
        },
    }


def write(fixture: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{fixture['name']}.json"
    path.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"  {path.relative_to(OUT.parents[3])}  ({path.stat().st_size // 1024} KB)")


def main() -> None:
    raw, _ = make_members()
    prepared = standardise(raw)

    print("writing fixtures:")
    write(pca_fixture(prepared, whiten=True))
    write(pca_fixture(prepared, whiten=False))
    write(graph_fixture())
    print(f"\n{raw.shape[0]} samples, {raw.shape[1]} columns")


if __name__ == "__main__":
    main()
