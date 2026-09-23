"""Writes the inference parity fixtures into the Inference test project.

Two small models trained by the trainer itself — a classifier and a regressor,
boosted trees — plus ``inference.json`` holding probe rows and what onnxruntime in
Python answered for them. The C# tests open the same files and must get the same
answers: that is the contract between the two sides of the ONNX boundary, and it is
what catches a metadata key renamed on one side only, or an output that changed
shape.

Commit what this writes. Regenerate only when the trainer's export or the metadata
record changes — otherwise the tests assert against whatever was generated last.

    python fixtures/make_inference_fixtures.py
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np
import onnxruntime as ort

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trainer"))

from otterlogic_trainer.progress import Progress  # noqa: E402
from otterlogic_trainer.train import run  # noqa: E402

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "OtterLogic.MachineLearning.Inference.Tests" / "Fixtures"

FEATURES = ["span", "sag", "rest_factor", "load"]
GROUPS = ["2024-017", "2024-031", "2025-003", "2025-008"]

PROBE = np.array(
    [
        [10.0, 1.0, 1.0, 2.0],      # short and light: stiff
        [12.5, 2.0, 0.9, 5.0],
        [20.0, 0.6, 1.1, 8.0],      # long, shallow, loaded: not
        [30.0, 0.5, 1.0, 10.0],     # the extreme of that
        [15.0, 1.5, 1.0, 1.0],
    ],
    dtype=np.float32,
)


class Silent(Progress):
    def __init__(self) -> None:  # noqa: D107
        pass

    def _write(self, record: dict) -> None:
        pass

    def close(self) -> None:
        pass


def make_folder(root: Path, task: str) -> Path:
    rng = np.random.default_rng(11)
    folder = root / task
    (folder / "models").mkdir(parents=True)

    target = (
        {"name": "stiff", "role": "target", "kind": "category", "classes": ["no", "yes"]}
        if task == "classification"
        else {"name": "deflection", "role": "target", "kind": "number", "unit": "mm"}
    )
    schema = {
        "formatVersion": 1,
        "extractorVersion": "fixture-1",
        "columns": [{"name": f, "role": "feature", "kind": "number"} for f in FEATURES] + [target],
    }
    (folder / "schema.json").write_text(json.dumps(schema, indent=2), encoding="utf-8")

    for g, group in enumerate(GROUPS):
        n = 80
        span = rng.uniform(8, 30, n) + g
        sag = rng.uniform(0.5, 3.0, n)
        rest = rng.uniform(0.8, 1.2, n)
        load = rng.uniform(1, 10, n)
        deflection = 0.02 * span**2 * load / (sag * rest) + rng.normal(0, 2, n)
        lines = [",".join(FEATURES + [target["name"]])]
        for i in range(n):
            y = ("yes" if deflection[i] < 60 else "no") if task == "classification" else repr(float(deflection[i]))
            lines.append(",".join(repr(float(v)) for v in (span[i], sag[i], rest[i], load[i])) + "," + y)
        (folder / "models" / f"{group}.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")

    return folder


def main() -> None:
    FIXTURES.mkdir(parents=True, exist_ok=True)
    expected: dict = {"features": FEATURES, "rows": PROBE.tolist()}

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for task, name in (("classification", "classifier"), ("regression", "regressor")):
            folder = make_folder(root, task)
            output = root / f"{name}.onnx"
            run(
                {"datasetFolder": str(folder), "learner": {"type": "boostedTrees"}, "holdoutFraction": 0.25, "seed": 1,
                 "outputPath": str(output)},
                Silent(),
            )
            shutil.copyfile(output, FIXTURES / f"{name}.onnx")

            session = ort.InferenceSession(str(output), providers=["CPUExecutionProvider"])
            if task == "classification":
                labels, probabilities = session.run(["label", "probabilities"], {"features": PROBE})
                expected[name] = {
                    "labels": [int(v) for v in labels],
                    "probabilities": np.asarray(probabilities, dtype=np.float64).tolist(),
                }
            else:
                (values,) = session.run(["value"], {"features": PROBE})
                expected[name] = {"values": np.asarray(values, dtype=np.float64).reshape(-1).tolist()}

            print(f"wrote {FIXTURES / f'{name}.onnx'} ({output.stat().st_size / 1024:.0f} KB)")

    (FIXTURES / "inference.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"wrote {FIXTURES / 'inference.json'}")


if __name__ == "__main__":
    main()
