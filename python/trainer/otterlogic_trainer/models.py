"""The learners on offer, by the ``type`` a ``Learner`` in C# writes into ``job.json``.

A learner arrives as one JSON object — ``{"type": "boostedTrees", "trees": 100,
"depth": 0}`` — and the setting names here are exactly the camelCase property
names of the C# records in ``Training/Learner.cs``. Change one side and the other
must follow; the C# end-to-end tests are what notice.

Every pipeline starts with a ``StandardScaler`` fitted on the training rows only,
so the scaling is baked into the exported graph and the C# side never repeats it —
and cannot repeat it wrongly. Trees do not need it, but the network, the linear
model and nearest neighbours all do, and one shape for every pipeline is simpler
than three.

A learner is listed here only once ``export.verify`` passes for it on both tasks.
"""

from __future__ import annotations

from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

BOOSTED_TREES = "boostedTrees"
NEURAL_NETWORK = "neuralNetwork"
LINEAR = "linear"
NEAREST_NEIGHBOURS = "nearestNeighbours"

LEARNER_TYPES = (BOOSTED_TREES, NEURAL_NETWORK, LINEAR, NEAREST_NEIGHBOURS)

DEFAULT_LEARNER = {"type": BOOSTED_TREES, "trees": 100, "depth": 0}

TASKS = ("classification", "regression")


def build(learner: dict, task: str, seed: int) -> Pipeline:
    """The pipeline for ``learner`` on ``task``, unfitted."""
    if task not in TASKS:
        raise ValueError(f"'{task}' is not a task. The tasks are: {', '.join(TASKS)}.")

    kind = _kind(learner)
    classify = task == "classification"

    if kind == BOOSTED_TREES:
        trees = _positive_int(learner, "trees", 100)
        depth = _count(learner, "depth", 0)
        common = dict(max_iter=trees, max_depth=None if depth == 0 else depth, random_state=seed)
        estimator = HistGradientBoostingClassifier(**common) if classify else HistGradientBoostingRegressor(**common)

    elif kind == NEURAL_NETWORK:
        layers = _layers(learner)
        iterations = _positive_int(learner, "iterations", 500)
        # No early stopping: scikit-learn's carves a validation slice off the
        # training rows, which a dataset of a few hundred rows cannot spare.
        common = dict(hidden_layer_sizes=layers, max_iter=iterations, random_state=seed)
        estimator = MLPClassifier(**common) if classify else MLPRegressor(**common)

    elif kind == LINEAR:
        regularisation = _regularisation(learner)
        # One knob for both tasks: ridge's alpha pulls the coefficients toward
        # zero as it grows, and logistic regression's C does the opposite, so C
        # is its reciprocal and "more regularisation" means the same thing on
        # either. The iteration cap is generous because lbfgs on standardised
        # columns converges in tens of steps; the default of 100 warns on a
        # badly separable target rather than fitting it.
        estimator = (
            LogisticRegression(C=1.0 / regularisation, max_iter=1000, random_state=seed)
            if classify
            else Ridge(alpha=regularisation, random_state=seed)
        )

    else:
        neighbours = _positive_int(learner, "neighbours", 5)
        estimator = (
            KNeighborsClassifier(n_neighbors=neighbours) if classify else KNeighborsRegressor(n_neighbors=neighbours)
        )

    return Pipeline([("scale", StandardScaler()), ("model", estimator)])


def describe(learner: dict) -> str:
    """The learner and the settings that matter, in a few words for a progress line: "boosted trees, 100 trees"."""
    kind = _kind(learner)
    if kind == BOOSTED_TREES:
        depth = _count(learner, "depth", 0)
        return f"boosted trees, {_positive_int(learner, 'trees', 100)} trees" + (f" of depth {depth}" if depth else "")
    if kind == NEURAL_NETWORK:
        layers = ", ".join(str(w) for w in _layers(learner))
        return f"neural network, hidden layers of {layers}, up to {_positive_int(learner, 'iterations', 500)} iterations"
    if kind == LINEAR:
        return f"linear model, regularisation {_regularisation(learner):g}"
    return f"nearest neighbours, {_positive_int(learner, 'neighbours', 5)} neighbours"


def _kind(learner: dict) -> str:
    if not isinstance(learner, dict):
        raise ValueError(
            f"The learner must be an object with a 'type', not {type(learner).__name__}. "
            f"The types are: {', '.join(LEARNER_TYPES)}."
        )
    kind = str(learner.get("type") or "")
    if kind not in LEARNER_TYPES:
        raise ValueError(f"'{kind}' is not a learner type. The choices are: {', '.join(LEARNER_TYPES)}.")
    return kind


def _positive_int(learner: dict, name: str, default: int) -> int:
    value = int(learner.get(name, default))
    if value < 1:
        raise ValueError(f"'{name}' must be at least 1, not {value}.")
    return value


def _count(learner: dict, name: str, default: int) -> int:
    value = int(learner.get(name, default))
    if value < 0:
        raise ValueError(f"'{name}' is a count, or 0 for no limit; not {value}.")
    return value


def _layers(learner: dict) -> tuple[int, ...]:
    layers = tuple(int(w) for w in (learner.get("hiddenLayers") or (64,)))
    if any(w < 1 for w in layers):
        raise ValueError("Every hidden layer needs at least one neuron.")
    return layers


def _regularisation(learner: dict) -> float:
    value = float(learner.get("regularisation", 1.0))
    if not value > 0.0:
        raise ValueError(f"Regularisation must be a positive number, not {value}.")
    return value
