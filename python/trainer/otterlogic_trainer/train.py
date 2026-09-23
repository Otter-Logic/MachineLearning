"""One training run: read, split, fit, score, export, check, write."""

from __future__ import annotations

import os
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    r2_score,
)

from . import NAME, __version__
from .dataset import Table, read_folder
from .export import export, verify
from .metadata import Metadata
from .models import DEFAULT_LEARNER, build, describe
from .progress import Progress
from .split import Split, holdout, holdout_rows

HOLDOUT_BY = ("group", "row")

ROW_HOLDOUT_CAVEAT = (
    "Rows were held out at random because no groups were given, so the score may be optimistic: "
    "rows from one model resemble each other, and a held-out row's neighbours were trained on. "
    "Wire groups to score on whole models instead."
)


class JobError(ValueError):
    """Something about the request, not the code — reported to the user as it is."""


def run(job: dict, progress: Progress) -> dict:
    started = time.perf_counter()

    folder = Path(str(job.get("datasetFolder") or "")).expanduser()
    output = Path(str(job.get("outputPath") or "")).expanduser()
    learner = job.get("learner") or dict(DEFAULT_LEARNER)
    holdout_by = str(job.get("holdoutBy") or "group")
    fraction = float(job.get("holdoutFraction", 0.25))
    seed = int(job.get("seed", 1))
    target_name = job.get("target")

    if not folder.is_dir():
        raise JobError(f"There is no dataset folder at '{folder}'.")
    if output.suffix.lower() != ".onnx":
        raise JobError(f"The model file must end in .onnx; '{output}' does not.")
    if not output.parent.is_dir():
        raise JobError(f"The folder for the model, '{output.parent}', does not exist.")
    if holdout_by not in HOLDOUT_BY:
        raise JobError(f"'{holdout_by}' is not a way to hold out. The choices are: {', '.join(HOLDOUT_BY)}.")

    # The learner is checked before anything is read, so a job with a typo in
    # it fails in a moment rather than after a minute of reading.
    try:
        learner_text = describe(learner)
    except (ValueError, TypeError) as ex:
        raise JobError(str(ex)) from ex

    # -- read ---------------------------------------------------------------
    progress.stage("reading", f"Reading {folder}")
    table = read_folder(folder)
    target = _choose_target(table, target_name)
    task = "classification" if target.kind == "category" else "regression"
    feature_names = [c.name for c in table.schema.features]

    y_raw = table.targets[target.name]
    if task == "classification":
        classes = list(target.classes)
        present = sorted({str(v) for v in y_raw}, key=classes.index)
        if len(present) < 2:
            raise JobError(
                f"Every row's '{target.name}' is '{present[0] if present else ''}', so there is nothing to tell apart."
            )
        index = {name: i for i, name in enumerate(classes)}
        y = np.asarray([index[str(v)] for v in y_raw], dtype=np.int64)
    else:
        classes = []
        y = np.asarray(y_raw, dtype=np.float64)

    group_count = len(set(table.groups))
    progress.stage(
        "reading",
        f"{group_count} model{'s' if group_count != 1 else ''}, {table.row_count} rows, "
        f"{len(feature_names)} features. Predicting {target.name} ({task}).",
    )

    # -- split --------------------------------------------------------------
    try:
        split = (
            holdout(table.groups, fraction, seed)
            if holdout_by == "group"
            else holdout_rows(table.row_count, fraction, seed)
        )
    except ValueError as ex:
        raise JobError(str(ex)) from ex

    x_train, x_test = table.features[split.train_rows], table.features[split.test_rows]
    y_train, y_test = y[split.train_rows], y[split.test_rows]

    if task == "classification" and len(np.unique(y_train)) < 2:
        held = ", ".join(split.test_groups) if split.test_groups else f"{len(split.test_rows)} rows"
        raise JobError(
            f"After holding out {held} every training row has the same '{target.name}'. "
            "Add a model where it differs, or lower the holdout."
        )

    progress.stage("fitting", f"Fitting {learner_text} on {_trained_on(split)}; holding out {_held_out(split)}.")

    # -- fit ----------------------------------------------------------------
    try:
        pipeline = build(learner, task, seed)
    except (ValueError, TypeError) as ex:
        raise JobError(str(ex)) from ex
    pipeline.fit(x_train, y_train)

    # -- score --------------------------------------------------------------
    progress.stage("scoring", f"Scoring on {len(split.test_rows)} held-out rows.")
    report: list[str] = []
    if holdout_by == "row":
        # First, because it qualifies every number that follows.
        report.append(ROW_HOLDOUT_CAVEAT)
    report.append(f"Trained on {_trained_on(split)}; scored on {_held_out(split)}.")

    if task == "classification":
        predicted = pipeline.predict(x_test).astype(np.int64)
        probabilities = pipeline.predict_proba(x_test)
        score = _classification_score(y_test, predicted, len(classes))
        report += _classification_report(y_test, predicted, classes, score)
        expected = {"labels": predicted, "probabilities": probabilities}
    else:
        predicted = pipeline.predict(x_test).astype(np.float64).reshape(-1)
        score = _regression_score(y_test, predicted)
        report += _regression_report(y_test, predicted, score, target.unit)
        expected = {"values": predicted}

    # -- export -------------------------------------------------------------
    progress.stage("exporting", "Exporting to ONNX.")
    metadata = Metadata(
        task=task,
        features=feature_names,
        target=target.name,
        classes=classes,
        extractor_version=table.schema.extractor_version,
        trainer_name=NAME,
        trainer_version=__version__,
        # The metadata key is still "modelType": the C# ModelMetadata reads it
        # and the committed fixtures carry it. What goes in it is the learner's type.
        model_type=str(learner["type"]),
        trained=datetime.now(timezone.utc),
        rows=table.row_count,
        groups=group_count,
        holdout_groups=split.test_groups,
        score=score,
    )
    model_bytes = export(pipeline, task, len(feature_names), len(classes), metadata.to_json())

    # -- check --------------------------------------------------------------
    progress.stage("checking", "Running the exported model against the fitted one.")
    verify(model_bytes, x_test, task, expected)

    # -- write --------------------------------------------------------------
    # Beside and rename over, so a reader never finds half a file.
    staging = output.with_name(output.name + ".writing")
    staging.write_bytes(model_bytes)
    os.replace(staging, output)

    elapsed = time.perf_counter() - started
    report.append(f"Wrote {output} ({len(model_bytes) / 1024:.0f} KB) in {elapsed:.1f} s.")

    return {"model": output, "score": score, "report": report}


def _trained_on(split: Split) -> str:
    if split.train_groups:
        return f"{len(split.train_rows)} rows from {len(split.train_groups)} models"
    return f"{len(split.train_rows)} rows"


def _held_out(split: Split) -> str:
    if split.test_groups:
        return (
            f"{len(split.test_rows)} rows from {len(split.test_groups)} held-out "
            f"({', '.join(split.test_groups)})"
        )
    return f"{len(split.test_rows)} rows held out at random"


def _choose_target(table: Table, name: str | None):
    targets = table.schema.targets
    if name:
        column = table.schema.column(str(name))
        if column.role != "target":
            raise JobError(f"'{column.name}' is a {column.role} column, not a target.")
        return column

    if len(targets) == 1:
        return targets[0]
    if not targets:
        raise JobError("The dataset has no target column, so there is nothing to predict.")
    raise JobError(
        "The dataset has several targets — " + ", ".join(c.name for c in targets) + " — so name the one to predict."
    )


def _classification_score(actual: np.ndarray, predicted: np.ndarray, class_count: int) -> dict[str, float]:
    counts = np.bincount(actual, minlength=class_count)
    return {
        "accuracy": float(np.mean(actual == predicted)),
        "noInformationRate": float(counts.max() / counts.sum()) if counts.sum() else 0.0,
        "balancedAccuracy": float(balanced_accuracy_score(actual, predicted)),
        "macroF1": float(f1_score(actual, predicted, average="macro", zero_division=0)),
    }


def _classification_report(actual, predicted, classes: list[str], score: dict[str, float]) -> list[str]:
    lines = [
        f"Accuracy {score['accuracy']:.3f} against {score['noInformationRate']:.3f} for always answering the "
        f"commonest class; balanced accuracy {score['balancedAccuracy']:.3f}; macro F1 {score['macroF1']:.3f}.",
    ]
    if score["accuracy"] <= score["noInformationRate"]:
        lines.append("Warning: the model does not beat always answering the commonest class.")

    labels = list(range(len(classes)))
    matrix = confusion_matrix(actual, predicted, labels=labels)
    width = max(len(c) for c in classes)
    lines.append("Confusion (rows are true, columns predicted): " + "  ".join(classes))
    for i, name in enumerate(classes):
        lines.append(f"  {name.rjust(width)}  " + "  ".join(str(int(v)) for v in matrix[i]))
    return lines


def _regression_score(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    residuals = predicted - actual
    return {
        "rSquared": float(r2_score(actual, predicted)) if actual.size > 1 else 0.0,
        "meanAbsoluteError": float(mean_absolute_error(actual, predicted)),
        "rootMeanSquaredError": float(np.sqrt(np.mean(residuals**2))),
    }


def _regression_report(actual, predicted, score: dict[str, float], unit: str | None) -> list[str]:
    residuals = predicted - actual
    worst = int(np.argmax(np.abs(residuals))) if residuals.size else 0
    suffix = f" {unit}" if unit else ""
    lines = [
        f"R² {score['rSquared']:.3f}; mean absolute error {score['meanAbsoluteError']:.4g}{suffix}; "
        f"root mean squared error {score['rootMeanSquaredError']:.4g}{suffix}.",
        f"Largest miss {abs(residuals[worst]):.4g}{suffix} on held-out row {worst} "
        f"(predicted {predicted[worst]:.4g}, actual {actual[worst]:.4g}).",
    ]
    if score["rSquared"] <= 0:
        lines.append("Warning: the model does no better than answering with the average.")
    return lines
