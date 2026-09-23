# Python

Two jobs, and a user of the plug-in installs neither of them by hand.

- **Reference fixtures** — scikit-learn as a *reference implementation to test
  the C# against*, never as a dependency. Development only.
- **The trainer** — `trainer/`, the process behind the Train component. It ships
  as a private bundle the plug-in fetches on request; nothing here is installed
  into a Python the user already has, and Predict needs none of it.

That second job retired a sentence this file used to open with: it is no longer
true that nothing Python ever reaches a user's machine. What is still true is
that nothing Python runs *inside Rhino*, and that a user who only receives models
and runs them installs nothing beyond OtterLogic.

## Setup

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pip install -e trainer[test]
```

## Fixtures

```
python fixtures/make_fixtures.py             # decomposition: pca_plain.json, pca_whiten.json
python fixtures/make_inference_fixtures.py   # inference parity: classifier.onnx, regressor.onnx, inference.json
```

Commit what they produce. Regenerate only when the fixture data or the comparison
changes — not on every run, or the tests assert against whatever was generated
last rather than against a fixed reference.

The decomposition fixtures are a straight exactness check: PCA is deterministic
up to a sign, so the C# should match scikit-learn to machine precision once the
component signs are in a canonical order. The clustering fixtures moved to
[Unsupervised](https://github.com/Otter-Logic/Unsupervised) with the algorithms.

The inference fixtures are the contract across the ONNX boundary. The trainer
trains two small models and records what onnxruntime in Python answers for a few
probe rows; the C# tests open the same files and must answer the same, to float32
tolerance. A metadata key renamed on one side only, or an output that changed
shape, fails here rather than in a user's definition.

## The trainer

See [trainer/README.md](trainer/README.md) for the job and progress protocol,
the four learners, pointing the plug-in at this checkout with `OTTERLOGIC_TRAINER`,
and `build-bundle.ps1`, which makes the zip a user's plug-in installs. Its own
tests run with `pytest trainer`.

Note the asymmetry the whole design rests on: a *learned* model has weights that
had to be found from data, so it is trained here, out of process, and crosses
into the plug-in as one `.onnx` file. A Gaussian mixture has no such weights — it
computes its parameters from whatever is on the wire, every solve — so it is C#,
and nothing about it comes through this directory.
