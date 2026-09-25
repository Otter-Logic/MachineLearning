# Python

Development only. Nothing here ships, and nothing here runs on a user's machine:
the plug-in trains its models in C#, and it reads models through ONNX Runtime.

- **Reference fixtures** — scikit-learn as a *reference implementation to test
  the C# against*, never as a dependency.
- **Inference parity fixtures** — models written by tools that are not
  OtterLogic, which the C# reader must answer for exactly as onnxruntime in
  Python does. This is the proof that the reader is a reader of ONNX, not of one
  writer's ONNX.

There used to be a third job here, the trainer behind OtterTrain, shipped as a
private Python bundle. It was taken out on 2026-09-25 before it shipped; see
[docs/in-process-training.md](../docs/in-process-training.md).

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Fixtures

```
python fixtures/make_fixtures.py             # decomposition: pca_plain.json, pca_whiten.json
python fixtures/make_inference_fixtures.py   # classifier.onnx, regressor.onnx, external-mlp.onnx, inference.json
```

Commit what they produce. Regenerate only when the fixture data or the comparison
changes — not on every run, or the tests assert against whatever was generated
last rather than against a fixed reference.

The decomposition fixtures are a straight exactness check: PCA is deterministic
up to a sign, so the C# should match scikit-learn to machine precision once the
component signs are in a canonical order. The clustering fixtures live in
[Unsupervised](https://github.com/Otter-Logic/Unsupervised) with the algorithms,
and the supervised ones — trees, boosting, the network — in
[Supervised](https://github.com/Otter-Logic/Supervised).

The inference fixtures are the contract across the ONNX boundary. Two scikit-learn
models carry the OtterLogic metadata record and stand in for any writer that does;
one small network carries none and stands in for a PyTorch export. The C# tests
open all three and must answer what Python answered, to float32 tolerance. A
metadata key renamed on one side only, or an output that changed shape, fails here
rather than in a user's definition.
