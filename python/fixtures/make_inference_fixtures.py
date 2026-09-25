"""Writes the inference parity fixtures into the Inference test project.

Three small models made by tools that are not OtterLogic, plus ``inference.json``
holding probe rows and what onnxruntime in Python answered for them. The C# side
opens the same files and must get the same answers: that is the contract across
the ONNX boundary, and it is what catches a metadata key renamed on one side only,
or an output that changed shape.

- ``classifier.onnx`` and ``regressor.onnx`` — scikit-learn boosted trees behind a
  ``StandardScaler``, exported with skl2onnx and carrying the OtterLogic metadata
  record. Written here the way the retired Python trainer wrote them, so the
  reader keeps proving it reads a model it did not make.
- ``external-mlp.onnx`` — a small network with *no* OtterLogic metadata, in the
  shape ``torch.onnx.export`` gives a ``torch.nn.Sequential`` of Linear and ReLU
  layers: Gemm nodes with a dynamic batch dimension and an output named by the
  exporter. Built with ``onnx.helper`` so that regenerating the fixtures needs no
  PyTorch; if ``torch`` is importable it is used instead, and the file is the
  same shape either way. This is the fixture for the promise that a deep-learning
  model trained elsewhere still runs in OtterPredict.

Commit what this writes. Regenerate only when the metadata record or the reader's
graph contract changes — otherwise the tests assert against whatever was generated
last rather than against a fixed reference.

    python fixtures/make_inference_fixtures.py
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
from onnx import TensorProto, helper, numpy_helper
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FIXTURES = Path(__file__).resolve().parents[2] / "tests" / "OtterLogic.MachineLearning.Inference.Tests" / "Fixtures"

FEATURES = ["span", "sag", "rest_factor", "load"]
GROUPS = ["2024-017", "2024-031", "2025-003", "2025-008"]

# The graph interface OnnxModel reads, fixed on the C# side.
INPUT, VALUE, LABEL, PROBABILITIES = "features", "value", "label", "probabilities"
METADATA_KEY = "otterlogic"

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


def _tolerate_booleans_in_int_attributes() -> None:
    """skl2onnx 1.20 against protobuf 7: the tree converter puts Python booleans in an
    integer list attribute, which protobuf 7 refuses. Coerce them before the node is
    built. Harmless once skl2onnx fixes it."""
    from skl2onnx.common._container import ModelComponentContainer

    if getattr(ModelComponentContainer, "_otterlogic_coerces_bools", False):
        return
    original = ModelComponentContainer.add_node

    def add_node(self, op_type, inputs, outputs, *args, **attrs):
        for key, value in attrs.items():
            if isinstance(value, (list, tuple)) and any(isinstance(v, (bool, np.bool_)) for v in value):
                attrs[key] = [int(v) if isinstance(v, (bool, np.bool_)) else v for v in value]
        return original(self, op_type, inputs, outputs, *args, **attrs)

    ModelComponentContainer.add_node = add_node
    ModelComponentContainer._otterlogic_coerces_bools = True


def make_table(seed: int = 11) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Four groups of eighty rows; a deflection that grows with span squared and load."""
    rng = np.random.default_rng(seed)
    x, deflection, groups = [], [], []
    for g, group in enumerate(GROUPS):
        n = 80
        span = rng.uniform(8, 30, n) + g
        sag = rng.uniform(0.5, 3.0, n)
        rest = rng.uniform(0.8, 1.2, n)
        load = rng.uniform(1, 10, n)
        x.append(np.column_stack([span, sag, rest, load]))
        deflection.append(0.02 * span**2 * load / (sag * rest) + rng.normal(0, 2, n))
        groups += [group] * n
    return np.vstack(x), np.concatenate(deflection), np.asarray(groups)


def metadata(task: str, target: str, classes: list[str], rows: int, score: dict[str, float]) -> str:
    record: dict = {
        "format": 1,
        "task": task,
        "features": FEATURES,
        "target": target,
        "extractorVersion": "fixture-1",
        "trainer": {"name": "scikit-learn-fixture", "version": "1", "modelType": "boostedTrees"},
        "trained": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
        "data": {"rows": rows, "groups": len(GROUPS), "holdoutGroups": [GROUPS[-1]]},
        "score": score,
    }
    if task == "classification":
        record["classes"] = classes
    return json.dumps(record, separators=(",", ":"))


def export_sklearn(pipeline: Pipeline, task: str, class_count: int, metadata_json: str) -> bytes:
    initial_types = [(INPUT, FloatTensorType([None, len(FEATURES)]))]
    if task == "classification":
        estimator = pipeline.steps[-1][1]
        model = to_onnx(
            pipeline,
            initial_types=initial_types,
            final_types=[(LABEL, Int64TensorType([None])), (PROBABILITIES, FloatTensorType([None, class_count]))],
            options={id(estimator): {"zipmap": False}},
        )
    else:
        model = to_onnx(pipeline, initial_types=initial_types, final_types=[(VALUE, FloatTensorType([None, 1]))])

    model.producer_name = "scikit-learn-fixture"
    entry = model.metadata_props.add()
    entry.key = METADATA_KEY
    entry.value = metadata_json
    onnx.checker.check_model(model)
    return model.SerializeToString()


def external_mlp(x: np.ndarray, y: np.ndarray) -> bytes:
    """A regressor with no OtterLogic metadata, as PyTorch would export one."""
    try:
        import torch  # noqa: F401

        return _torch_mlp(x, y)
    except ImportError:
        return _helper_mlp(x, y)


def _torch_mlp(x: np.ndarray, y: np.ndarray) -> bytes:
    import io

    import torch

    torch.manual_seed(0)
    mean, std = x.mean(axis=0), x.std(axis=0)
    xt = torch.tensor((x - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32).reshape(-1, 1)
    net = torch.nn.Sequential(torch.nn.Linear(4, 16), torch.nn.ReLU(), torch.nn.Linear(16, 1))
    optimiser = torch.optim.Adam(net.parameters(), lr=1e-2)
    for _ in range(500):
        optimiser.zero_grad()
        loss = torch.nn.functional.mse_loss(net(xt), yt)
        loss.backward()
        optimiser.step()
    # Fold the standardisation into the first layer so the file takes raw features.
    with torch.no_grad():
        first = net[0]
        first.bias.copy_(first.bias - (first.weight @ torch.tensor(mean / std, dtype=torch.float32)))
        first.weight.copy_(first.weight / torch.tensor(std, dtype=torch.float32))
    buffer = io.BytesIO()
    torch.onnx.export(net, torch.zeros(1, 4), buffer, input_names=["x"], output_names=["y"],
                      dynamic_axes={"x": {0: "batch"}, "y": {0: "batch"}}, opset_version=17)
    return buffer.getvalue()


def _helper_mlp(x: np.ndarray, y: np.ndarray) -> bytes:
    """The same graph shape torch.onnx.export writes for Linear-ReLU-Linear, built directly.

    The weights are fitted the cheap way — random ReLU features then a least-squares
    read-out — because what is under test is the reader's handling of a graph it did
    not make, not the fit. The standardisation is folded into the first layer so the
    file takes raw features, as an exported network would.
    """
    rng = np.random.default_rng(0)
    mean, std = x.mean(axis=0), x.std(axis=0)
    z = (x - mean) / std
    w1 = rng.normal(0.0, 1.0, (16, 4))            # torch keeps Linear weights as [out, in]
    b1 = rng.normal(0.0, 0.5, 16)
    hidden = np.maximum(z @ w1.T + b1, 0.0)
    design = np.column_stack([hidden, np.ones(len(z))])
    solution, *_ = np.linalg.lstsq(design, y, rcond=None)
    w2, b2 = solution[:16].reshape(1, 16), solution[16:]

    w1_raw = w1 / std                              # fold (x - mean) / std into the layer
    b1_raw = b1 - w1_raw @ mean

    nodes = [
        helper.make_node("Gemm", ["x", "fc1.weight", "fc1.bias"], ["h"], alpha=1.0, beta=1.0, transB=1, name="/fc1/Gemm"),
        helper.make_node("Relu", ["h"], ["a"], name="/act/Relu"),
        helper.make_node("Gemm", ["a", "fc2.weight", "fc2.bias"], ["y"], alpha=1.0, beta=1.0, transB=1, name="/fc2/Gemm"),
    ]
    graph = helper.make_graph(
        nodes, "main_graph",
        [helper.make_tensor_value_info("x", TensorProto.FLOAT, ["batch", 4])],
        [helper.make_tensor_value_info("y", TensorProto.FLOAT, ["batch", 1])],
        initializer=[
            numpy_helper.from_array(w1_raw.astype(np.float32), "fc1.weight"),
            numpy_helper.from_array(b1_raw.astype(np.float32), "fc1.bias"),
            numpy_helper.from_array(w2.astype(np.float32), "fc2.weight"),
            numpy_helper.from_array(b2.astype(np.float32), "fc2.bias"),
        ],
    )
    model = helper.make_model(graph, producer_name="pytorch", producer_version="2.4.0",
                              opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 8
    onnx.checker.check_model(model, full_check=True)
    return model.SerializeToString()


def main() -> None:
    _tolerate_booleans_in_int_attributes()
    FIXTURES.mkdir(parents=True, exist_ok=True)
    expected: dict = {"features": FEATURES, "rows": PROBE.tolist()}

    x, deflection, groups = make_table()
    train = groups != GROUPS[-1]
    stiff = np.where(deflection < 60, "yes", "no")
    classes = ["no", "yes"]
    y_class = np.asarray([classes.index(s) for s in stiff], dtype=np.int64)

    classifier = Pipeline([("scale", StandardScaler()), ("model", HistGradientBoostingClassifier(random_state=1))])
    classifier.fit(x[train], y_class[train])
    accuracy = float((classifier.predict(x[~train]) == y_class[~train]).mean())
    counts = np.bincount(y_class[~train], minlength=2)
    (FIXTURES / "classifier.onnx").write_bytes(export_sklearn(
        classifier, "classification", 2,
        metadata("classification", "stiff", classes, len(x),
                 {"accuracy": accuracy, "noInformationRate": float(counts.max() / counts.sum())})))

    regressor = Pipeline([("scale", StandardScaler()), ("model", HistGradientBoostingRegressor(random_state=1))])
    regressor.fit(x[train], deflection[train])
    predicted = regressor.predict(x[~train])
    residual = deflection[~train] - predicted
    r2 = 1.0 - float((residual**2).sum() / ((deflection[~train] - deflection[~train].mean())**2).sum())
    (FIXTURES / "regressor.onnx").write_bytes(export_sklearn(
        regressor, "regression", 0,
        metadata("regression", "deflection", [], len(x), {"rSquared": r2, "meanAbsoluteError": float(np.abs(residual).mean())})))

    (FIXTURES / "external-mlp.onnx").write_bytes(external_mlp(x[train], deflection[train]))

    for name, outputs in (("classifier", [LABEL, PROBABILITIES]), ("regressor", [VALUE]), ("external-mlp", None)):
        session = ort.InferenceSession(str(FIXTURES / f"{name}.onnx"), providers=["CPUExecutionProvider"])
        input_name = session.get_inputs()[0].name
        answers = session.run(outputs, {input_name: PROBE})
        if name == "classifier":
            expected[name] = {
                "labels": [int(v) for v in answers[0]],
                "probabilities": np.asarray(answers[1], dtype=np.float64).tolist(),
            }
        else:
            key = "regressor" if name == "regressor" else "externalMlp"
            expected[key] = {"values": np.asarray(answers[0], dtype=np.float64).reshape(-1).tolist()}
        print(f"wrote {FIXTURES / f'{name}.onnx'} ({(FIXTURES / f'{name}.onnx').stat().st_size / 1024:.0f} KB)")

    (FIXTURES / "inference.json").write_text(json.dumps(expected, indent=2), encoding="utf-8")
    print(f"wrote {FIXTURES / 'inference.json'}")


if __name__ == "__main__":
    main()
