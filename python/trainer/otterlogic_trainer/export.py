"""scikit-learn pipeline in, ``.onnx`` bytes out, with the graph interface the C# side expects.

The interface is fixed so ``OnnxModel`` never has to guess:

    input   features       float32 [rows, features]
    output  value          float32 [rows, 1]            regression
    output  label          int64   [rows]               classification, index into classes
    output  probabilities  float32 [rows, classes]      classification

``verify`` runs the exported graph through onnxruntime on the held-out rows and
compares with scikit-learn. skl2onnx handles some operators badly, and the failure
is a model that runs and is quietly wrong; this is the check that turns it into an
error before the file is written.
"""

from __future__ import annotations

import numpy as np
import onnx
from skl2onnx import to_onnx
from skl2onnx.common.data_types import FloatTensorType, Int64TensorType
from sklearn.pipeline import Pipeline

from .metadata import KEY

INPUT = "features"


def _tolerate_booleans_in_int_attributes() -> None:
    """Works around skl2onnx 1.20 against protobuf 7.

    The tree-ensemble converter fills ``nodes_missing_value_tracks_true`` with a
    Python ``False`` for every leaf, and protobuf 7 refuses a boolean in an integer
    list where earlier versions quietly took it. Every boosted-tree export fails
    with "Field onnx.AttributeProto.ints: Expected an int, got a boolean". Coercing
    the list before the node is built is the whole fix. Remove this when skl2onnx
    passes integers itself — the test suite will say so, because the patch then
    does nothing.
    """
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


_tolerate_booleans_in_int_attributes()
VALUE = "value"
LABEL = "label"
PROBABILITIES = "probabilities"


def export(pipeline: Pipeline, task: str, feature_count: int, class_count: int, metadata_json: str) -> bytes:
    initial_types = [(INPUT, FloatTensorType([None, feature_count]))]

    if task == "classification":
        estimator = pipeline.steps[-1][1]
        model = to_onnx(
            pipeline,
            initial_types=initial_types,
            final_types=[(LABEL, Int64TensorType([None])), (PROBABILITIES, FloatTensorType([None, class_count]))],
            # ZipMap wraps the probabilities in a sequence of dictionaries, which
            # is convenient in Python and unreadable as a tensor. Off, so the
            # probabilities come out as rows by classes.
            options={id(estimator): {"zipmap": False}},
        )
    else:
        model = to_onnx(
            pipeline,
            initial_types=initial_types,
            final_types=[(VALUE, FloatTensorType([None, 1]))],
        )

    model.producer_name = "otterlogic-trainer"
    model.doc_string = "An OtterLogic model. Its metadata_props['otterlogic'] says what it predicts and what it takes."

    # One key, holding the whole record. Anything reading the file finds it in the
    # same place ONNX Runtime exposes as CustomMetadataMap.
    entry = model.metadata_props.add()
    entry.key = KEY
    entry.value = metadata_json

    onnx.checker.check_model(model)
    return model.SerializeToString()


def verify(model_bytes: bytes, x: np.ndarray, task: str, expected: dict[str, np.ndarray]) -> None:
    """Runs the graph and raises if it disagrees with scikit-learn beyond float32 tolerance."""
    import onnxruntime as ort

    session = ort.InferenceSession(model_bytes, providers=["CPUExecutionProvider"])
    inputs = {INPUT: x.astype(np.float32)}

    if task == "regression":
        (values,) = session.run([VALUE], inputs)
        values = np.asarray(values, dtype=np.float64).reshape(-1)
        want = expected["values"].astype(np.float64).reshape(-1)
        scale = max(1.0, float(np.max(np.abs(want))) if want.size else 1.0)
        worst = float(np.max(np.abs(values - want))) if want.size else 0.0
        if worst > 1e-4 * scale:
            raise RuntimeError(
                f"The exported model disagrees with the fitted one: the largest difference on the held-out rows "
                f"is {worst:.6g} against values up to {scale:.6g}. The export is broken, so no model was written."
            )
        return

    labels, probabilities = session.run([LABEL, PROBABILITIES], inputs)
    labels = np.asarray(labels).reshape(-1)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    want_labels = expected["labels"].reshape(-1)
    want_probabilities = expected["probabilities"].astype(np.float64)

    if probabilities.shape != want_probabilities.shape:
        raise RuntimeError(
            f"The exported model returns probabilities of shape {probabilities.shape}; scikit-learn's are "
            f"{want_probabilities.shape}."
        )

    worst = float(np.max(np.abs(probabilities - want_probabilities))) if probabilities.size else 0.0
    if worst > 1e-4:
        raise RuntimeError(
            f"The exported model disagrees with the fitted one: probabilities differ by up to {worst:.6g}. "
            "The export is broken, so no model was written."
        )

    # A probability within tolerance can still tip a near-tie the other way. Only a
    # disagreement that is not a near-tie is an export fault.
    disagree = labels != want_labels
    if disagree.any():
        rows = np.flatnonzero(disagree)
        for i in rows:
            top = np.sort(want_probabilities[i])[::-1]
            margin = top[0] - top[1] if top.size > 1 else 1.0
            if margin > 1e-3:
                raise RuntimeError(
                    f"The exported model answers class {labels[i]} on held-out row {i} where scikit-learn answers "
                    f"{want_labels[i]}, and it is not a near-tie. The export is broken, so no model was written."
                )
