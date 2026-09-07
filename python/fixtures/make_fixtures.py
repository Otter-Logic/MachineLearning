"""Generate scikit-learn reference fixtures for the C# clustering tests.

Run this, commit the JSON it writes, and the C# tests assert against it. Nothing
here runs at plug-in runtime — scikit-learn is a reference implementation, not a
dependency.

Two different questions are being asked, and they need different fixtures:

*Exactness.* Given identical starting parameters, does the C# EM compute the
same trajectory as scikit-learn's? Pinning the initialisation removes the local
optimum from the comparison, so any disagreement is arithmetic and is a bug.

*Quality.* Left to find its own starting point, does the C# pipeline reach as
good a fit as scikit-learn's default? This one is not expected to match
parameter for parameter — two non-convex optimisers climbing different hills —
so it is scored on BIC and on how far the two labellings agree.

    python python/fixtures/make_fixtures.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

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


def initial_parameters(x: np.ndarray, k: int, covariance_type: str) -> dict:
    """A starting point both implementations can reproduce exactly.

    Evenly spaced rows as means, uniform mixing weights, and one shared
    isotropic covariance. Deliberately not k-means: the point of this fixture is
    to take initialisation out of the comparison entirely.
    """
    n, d = x.shape
    indices = [i * n // k for i in range(k)]

    means = x[indices].copy()
    weights = np.full(k, 1.0 / k)
    spread = float(x.var(axis=0).mean())

    covariances = np.array([np.eye(d) * spread for _ in range(k)])

    return {
        "weights": weights,
        "means": means,
        "covariances": covariances,
        "spread": spread,
    }


def em_fixture(x: np.ndarray, k: int, covariance_type: str,
               reg_covar: float = 1e-6, tol: float = 1e-4, max_iter: int = 200) -> dict:
    init = initial_parameters(x, k, covariance_type)
    d = x.shape[1]

    # scikit-learn takes precisions, not covariances. The initial covariance is
    # isotropic so inverting it is exact and introduces no drift of its own.
    if covariance_type == "full":
        precisions_init = np.array([np.eye(d) / init["spread"] for _ in range(k)])
    elif covariance_type == "diag":
        precisions_init = np.full((k, d), 1.0 / init["spread"])
    elif covariance_type == "spherical":
        precisions_init = np.full(k, 1.0 / init["spread"])
    else:
        raise ValueError(covariance_type)

    gm = GaussianMixture(
        n_components=k,
        covariance_type=covariance_type,
        reg_covar=reg_covar,
        tol=tol,
        max_iter=max_iter,
        n_init=1,
        weights_init=init["weights"],
        means_init=init["means"],
        precisions_init=precisions_init,
        random_state=0,
    )
    gm.fit(x)

    responsibilities = gm.predict_proba(x)

    return {
        "name": f"em_{covariance_type}",
        "covariance_type": covariance_type,
        "components": k,
        "reg_covar": reg_covar,
        "tolerance": tol,
        "max_iterations": max_iter,
        "x": x.tolist(),
        "init": {
            "weights": init["weights"].tolist(),
            "means": init["means"].tolist(),
            "covariances": init["covariances"].tolist(),
        },
        "expected": {
            "weights": gm.weights_.tolist(),
            "means": gm.means_.tolist(),
            "covariances": expand_covariances(gm.covariances_, covariance_type, k, d).tolist(),
            "responsibilities": responsibilities.tolist(),
            # score() is the mean log-likelihood under the final parameters,
            # which is one M-step later than sklearn's own lower_bound_.
            "mean_log_likelihood": float(gm.score(x)),
            "bic": float(gm.bic(x)),
            "aic": float(gm.aic(x)),
            "iterations": int(gm.n_iter_),
            "converged": bool(gm.converged_),
            "parameter_count": int(gm._n_parameters()),
        },
    }


def expand_covariances(covariances: np.ndarray, covariance_type: str, k: int, d: int) -> np.ndarray:
    """Everything as k full matrices, so the C# side never has to branch."""
    if covariance_type == "full":
        return covariances
    if covariance_type == "diag":
        return np.array([np.diag(covariances[c]) for c in range(k)])
    if covariance_type == "spherical":
        return np.array([np.eye(d) * covariances[c] for c in range(k)])
    raise ValueError(covariance_type)


def quality_fixture(raw: np.ndarray, family: np.ndarray, k: int = 4) -> dict:
    """scikit-learn's default pipeline, doing its own initialisation.

    This is the honest comparison: both implementations get the same data and
    the same preprocessing, then each finds its own starting point ten times
    over and keeps its best. Matching parameter for parameter would be a
    coincidence; the question is whether the two reach an equally good fit and
    agree about which member belongs with which.
    """
    prepared = standardise(raw)

    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    z = pca.fit_transform(prepared)
    z = z * sign_flips(pca.components_)

    gm = GaussianMixture(
        n_components=k,
        covariance_type="diag",
        n_init=10,
        reg_covar=1e-6,
        tol=1e-4,
        max_iter=200,
        random_state=0,
    )
    labels = gm.fit_predict(z)

    return {
        "name": "quality",
        "components": k,
        "raw": raw.tolist(),
        "true_family": family.tolist(),
        "columns": DOF_NAMES,
        "expected": {
            "retained_components": int(pca.n_components_),
            "explained_variance_ratio": float(pca.explained_variance_ratio_.sum()),
            "labels": labels.tolist(),
            "bic": float(gm.bic(z)),
            "aic": float(gm.aic(z)),
            "mean_log_likelihood": float(gm.score(z)),
            "mean_confidence": float(gm.predict_proba(z).max(axis=1).mean()),
            "iterations": int(gm.n_iter_),
        },
    }


def sweep_fixture(raw: np.ndarray, lo: int = 2, hi: int = 8) -> dict:
    """BIC across a range of k, for the group-count component."""
    prepared = standardise(raw)
    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    z = pca.fit_transform(prepared) * sign_flips(pca.components_)

    rows = []
    for k in range(lo, hi + 1):
        gm = GaussianMixture(
            n_components=k, covariance_type="diag", n_init=10,
            reg_covar=1e-6, tol=1e-4, max_iter=200, random_state=0,
        )
        gm.fit(z)
        rows.append({"groups": k, "bic": float(gm.bic(z)), "aic": float(gm.aic(z))})

    return {"name": "sweep", "minimum": lo, "maximum": hi, "raw": raw.tolist(), "expected": rows}


def write(fixture: dict) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{fixture['name']}.json"
    path.write_text(json.dumps(fixture, indent=1), encoding="utf-8")
    print(f"  {path.relative_to(OUT.parents[3])}  ({path.stat().st_size // 1024} KB)")


def main() -> None:
    raw, family = make_members()
    prepared = standardise(raw)

    pca = PCA(n_components=0.99, whiten=True, svd_solver="full")
    whitened = pca.fit_transform(prepared) * sign_flips(pca.components_)

    print("writing fixtures:")
    write(pca_fixture(prepared, whiten=True))
    write(pca_fixture(prepared, whiten=False))
    write(em_fixture(whitened, k=4, covariance_type="diag"))
    write(em_fixture(whitened, k=4, covariance_type="full"))
    write(em_fixture(whitened, k=3, covariance_type="spherical"))
    write(quality_fixture(raw, family))
    write(sweep_fixture(raw))
    print(f"\n{raw.shape[0]} members, {raw.shape[1]} columns, "
          f"{pca.n_components_} principal components retained "
          f"({pca.explained_variance_ratio_.sum() * 100:.2f}% of variance)")


if __name__ == "__main__":
    main()
