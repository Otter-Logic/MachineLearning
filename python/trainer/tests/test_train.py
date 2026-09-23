"""Trains every learner on both tasks from a synthetic dataset folder and checks the file round-trips.

The folder is written here by hand, in the layout ``DatasetFolder`` in C# writes,
so the test doubles as a check that the Python reader agrees with it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest

from otterlogic_trainer import NAME, __version__
from otterlogic_trainer.dataset import read_folder
from otterlogic_trainer.metadata import KEY
from otterlogic_trainer.models import LEARNER_TYPES, build, describe
from otterlogic_trainer.progress import Progress
from otterlogic_trainer.split import holdout, holdout_rows
from otterlogic_trainer.train import ROW_HOLDOUT_CAVEAT, JobError, run

FEATURES = ["span", "sag", "rest_factor", "load"]
GROUPS = ["2024-017", "2024-031", "2025-003", "2025-008", "2025-011", "2025-019"]

# One of each, with settings a user might actually pick, so the test proves the
# setting names reach scikit-learn and not only that the default does.
LEARNERS = {
    "boostedTrees": {"type": "boostedTrees", "trees": 60, "depth": 4},
    "neuralNetwork": {"type": "neuralNetwork", "hiddenLayers": [32, 16], "iterations": 400},
    "linear": {"type": "linear", "regularisation": 0.5},
    "nearestNeighbours": {"type": "nearestNeighbours", "neighbours": 7},
}


def make_folder(root: Path, task: str, rows_per_group: int = 60, seed: int = 7) -> Path:
    """A sweep-like dataset: features drawn per group, targets a known function plus noise."""
    rng = np.random.default_rng(seed)
    folder = root / f"dataset-{task}"
    (folder / "models").mkdir(parents=True)

    if task == "classification":
        target = {"name": "stiff", "role": "target", "kind": "category", "classes": ["0", "1"]}
    else:
        target = {"name": "deflection", "role": "target", "kind": "number", "unit": "mm"}

    schema = {
        "formatVersion": 1,
        "extractorVersion": "1",
        "columns": [{"name": f, "role": "feature", "kind": "number"} for f in FEATURES]
        + [target]
        + [{"name": "member", "role": "id", "kind": "category"}],
    }
    (folder / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    for g, group in enumerate(GROUPS):
        span = rng.uniform(8, 30, rows_per_group) + g          # each project a little different
        sag = rng.uniform(0.5, 3.0, rows_per_group)
        rest = rng.uniform(0.8, 1.2, rows_per_group)
        load = rng.uniform(1, 10, rows_per_group)
        deflection = 0.02 * span**2 * load / (sag * rest) + rng.normal(0, 2, rows_per_group)

        lines = [",".join(FEATURES + [target["name"], "member"])]
        for i in range(rows_per_group):
            if task == "classification":
                y = "1" if deflection[i] < np.median(deflection) else "0"
            else:
                y = repr(float(deflection[i]))
            lines.append(",".join([repr(float(v)) for v in (span[i], sag[i], rest[i], load[i])] + [y, f"m{i}"]))
        (folder / "models" / f"{group}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return folder


class QuietProgress(Progress):
    def __init__(self) -> None:  # noqa: D107 - never touches disk
        self.lines: list[dict] = []

    def _write(self, record: dict) -> None:
        self.lines.append(record)

    def close(self) -> None:
        pass


def test_reader_agrees_with_layout(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "classification")
    table = read_folder(folder)

    assert table.row_count == 60 * len(GROUPS)
    assert [c.name for c in table.schema.features] == FEATURES
    assert sorted(set(table.groups)) == GROUPS
    assert set(table.targets["stiff"]) == {"0", "1"}
    assert table.ids[0] == "m0"


def test_holdout_is_by_group() -> None:
    groups = [g for g in GROUPS for _ in range(3)]
    split = holdout(groups, 0.25, seed=1)

    assert len(split.test_groups) == 2          # round(0.25 * 6) = 2, not 1.5 rows of something
    assert not set(split.train_groups) & set(split.test_groups)
    assert all(groups[i] in split.test_groups for i in split.test_rows)
    assert holdout(groups, 0.25, seed=1) == split   # seeded


def test_holdout_rows_is_seeded_and_leaves_a_row_each_side() -> None:
    split = holdout_rows(40, 0.25, seed=3)

    assert len(split.test_rows) == 10
    assert sorted(split.train_rows + split.test_rows) == list(range(40))
    assert split.train_groups == [] and split.test_groups == []
    assert holdout_rows(40, 0.25, seed=3) == split
    assert holdout_rows(40, 0.25, seed=4) != split

    tiny = holdout_rows(2, 0.01, seed=1)
    assert len(tiny.train_rows) == 1 and len(tiny.test_rows) == 1
    with pytest.raises(ValueError):
        holdout_rows(1, 0.25)


def test_every_learner_type_has_a_test_setting() -> None:
    assert set(LEARNERS) == set(LEARNER_TYPES)


def test_learner_settings_reach_scikit_learn() -> None:
    trees = build(LEARNERS["boostedTrees"], "regression", 1).steps[-1][1]
    assert (trees.max_iter, trees.max_depth) == (60, 4)
    assert build({"type": "boostedTrees"}, "regression", 1).steps[-1][1].max_depth is None

    net = build(LEARNERS["neuralNetwork"], "classification", 1).steps[-1][1]
    assert (net.hidden_layer_sizes, net.max_iter) == ((32, 16), 400)

    ridge = build(LEARNERS["linear"], "regression", 1).steps[-1][1]
    logistic = build(LEARNERS["linear"], "classification", 1).steps[-1][1]
    assert ridge.alpha == 0.5 and logistic.C == 2.0

    assert build(LEARNERS["nearestNeighbours"], "classification", 1).steps[-1][1].n_neighbors == 7

    assert describe(LEARNERS["boostedTrees"]) == "boosted trees, 60 trees of depth 4"
    assert describe({"type": "linear"}) == "linear model, regularisation 1"


def test_unknown_learner_lists_the_choices() -> None:
    with pytest.raises(ValueError) as caught:
        build({"type": "randomForest"}, "regression", 1)
    for kind in LEARNER_TYPES:
        assert kind in str(caught.value)
    with pytest.raises(ValueError):
        build({"type": "boostedTrees", "trees": 0}, "regression", 1)


@pytest.mark.parametrize("task", ["classification", "regression"])
@pytest.mark.parametrize("kind", LEARNER_TYPES)
def test_train_round_trips(tmp_path: Path, task: str, kind: str) -> None:
    folder = make_folder(tmp_path, task)
    output = tmp_path / f"{task}-{kind}.onnx"
    progress = QuietProgress()
    learner = LEARNERS[kind]

    result = run(
        {"datasetFolder": str(folder), "learner": learner, "holdoutFraction": 0.25, "seed": 1,
         "outputPath": str(output)},
        progress,
    )

    assert output.is_file()
    assert result["model"] == output
    assert [line["stage"] for line in progress.lines][:2] == ["reading", "reading"]
    fitting = next(line["message"] for line in progress.lines if line["stage"] == "fitting")
    assert describe(learner) in fitting
    assert result["report"][0].startswith("Trained on")
    assert "held-out" in result["report"][0]

    model = onnx.load(str(output))
    props = {p.key: p.value for p in model.metadata_props}
    metadata = json.loads(props[KEY])

    assert metadata["format"] == 1
    assert metadata["task"] == task
    assert metadata["features"] == FEATURES
    assert metadata["trainer"]["modelType"] == kind
    assert metadata["trainer"]["version"] == __version__
    assert metadata["data"]["groups"] == len(GROUPS)
    assert len(metadata["data"]["holdoutGroups"]) == 2
    assert metadata["extractorVersion"] == "1"

    session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
    x = np.array([[12.0, 1.5, 1.0, 5.0], [25.0, 0.8, 0.9, 9.0]], dtype=np.float32)

    if task == "classification":
        assert metadata["classes"] == ["0", "1"]
        assert metadata["score"]["accuracy"] > metadata["score"]["noInformationRate"]
        labels, probabilities = session.run(["label", "probabilities"], {"features": x})
        assert labels.shape == (2,) and labels.dtype == np.int64
        assert probabilities.shape == (2, 2)
        assert np.allclose(probabilities.sum(axis=1), 1.0, atol=1e-5)
    else:
        assert "classes" not in metadata
        assert metadata["score"]["rSquared"] > 0.5
        (values,) = session.run(["value"], {"features": x})
        assert values.shape == (2, 1)
        assert values[1, 0] > values[0, 0]   # longer span, more load: more deflection


def test_default_learner_is_boosted_trees(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "regression")
    output = tmp_path / "default.onnx"
    run({"datasetFolder": str(folder), "outputPath": str(output)}, QuietProgress())

    metadata = json.loads({p.key: p.value for p in onnx.load(str(output)).metadata_props}[KEY])
    assert metadata["trainer"]["modelType"] == "boostedTrees"


def test_holdout_by_row_says_so_first(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "classification")
    output = tmp_path / "rows.onnx"
    progress = QuietProgress()

    result = run(
        {"datasetFolder": str(folder), "learner": {"type": "linear"}, "holdoutBy": "row", "holdoutFraction": 0.2,
         "outputPath": str(output)},
        progress,
    )

    assert result["report"][0] == ROW_HOLDOUT_CAVEAT
    assert "held out at random" in result["report"][1]
    assert "72 rows held out" in result["report"][1]          # 0.2 of 360
    fitting = next(line["message"] for line in progress.lines if line["stage"] == "fitting")
    assert "held out at random" in fitting

    metadata = json.loads({p.key: p.value for p in onnx.load(str(output)).metadata_props}[KEY])
    assert metadata["data"]["holdoutGroups"] == []
    assert metadata["data"]["groups"] == len(GROUPS)


def test_holdout_by_row_works_with_one_group(tmp_path: Path) -> None:
    """The case it exists for: samples wired in with no groups all land in one model."""
    folder = make_folder(tmp_path, "regression")
    keep = folder / "models" / f"{GROUPS[0]}.csv"
    for other in GROUPS[1:]:
        (folder / "models" / f"{other}.csv").unlink()
    assert keep.is_file()

    with pytest.raises(JobError, match="at least two groups"):
        run({"datasetFolder": str(folder), "outputPath": str(tmp_path / "never.onnx")}, QuietProgress())

    result = run(
        {"datasetFolder": str(folder), "holdoutBy": "row", "outputPath": str(tmp_path / "one.onnx")}, QuietProgress()
    )
    assert result["report"][0] == ROW_HOLDOUT_CAVEAT
    assert (tmp_path / "one.onnx").is_file()


def test_bad_learner_is_a_job_error(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "regression")
    with pytest.raises(JobError, match="not a learner type"):
        run(
            {"datasetFolder": str(folder), "learner": {"type": "forest"}, "outputPath": str(tmp_path / "x.onnx")},
            QuietProgress(),
        )
    with pytest.raises(JobError, match="not a way to hold out"):
        run({"datasetFolder": str(folder), "holdoutBy": "column", "outputPath": str(tmp_path / "x.onnx")}, QuietProgress())


def test_bad_target_is_a_job_error(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "regression")
    with pytest.raises(JobError):
        run({"datasetFolder": str(folder), "target": "span", "outputPath": str(tmp_path / "x.onnx")}, QuietProgress())


def test_command_line_writes_progress(tmp_path: Path) -> None:
    folder = make_folder(tmp_path, "regression")
    work = tmp_path / "job"
    work.mkdir()
    job = work / "job.json"
    job.write_text(json.dumps({
        "datasetFolder": str(folder), "learner": {"type": "boostedTrees", "trees": 50},
        "outputPath": str(tmp_path / "cli.onnx"),
    }), encoding="utf-8")

    completed = subprocess.run([sys.executable, "-m", "otterlogic_trainer", str(job)], capture_output=True, text=True)
    assert completed.returncode == 0, completed.stderr

    lines = [json.loads(l) for l in (work / "progress.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines[-1]["done"] is True
    assert Path(lines[-1]["model"]).is_file()
    assert "rSquared" in lines[-1]["score"]
    assert any(line.get("stage") == "checking" for line in lines)


def test_command_line_reports_a_missing_folder(tmp_path: Path) -> None:
    job = tmp_path / "job.json"
    job.write_text(json.dumps({"datasetFolder": str(tmp_path / "nowhere"), "outputPath": str(tmp_path / "x.onnx")}))

    completed = subprocess.run([sys.executable, "-m", "otterlogic_trainer", str(job)], capture_output=True, text=True)
    assert completed.returncode == 1

    last = json.loads((tmp_path / "progress.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert "nowhere" in last["error"]


def test_command_line_version(tmp_path: Path) -> None:
    completed = subprocess.run(
        [sys.executable, "-m", "otterlogic_trainer", "--version"], capture_output=True, text=True, cwd=tmp_path
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.strip() == f"{NAME} {__version__}"
    assert not (tmp_path / "progress.jsonl").exists()    # a version check is not a job

    usage = subprocess.run([sys.executable, "-m", "otterlogic_trainer"], capture_output=True, text=True)
    assert usage.returncode == 2 and "--version" in usage.stderr
