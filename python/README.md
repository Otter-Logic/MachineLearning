# Python

**No user of the plug-in installs any of this.** Not IronPython, not Rhino 8's
embedded CPython, not a system interpreter. Everything here runs on a developer's
machine and stops there; everything that ships is C# end to end.

Two jobs, present and future.

## Today: reference fixtures

scikit-learn is here as a *reference implementation to test the C# against*, not
as a dependency. `fixtures/make_fixtures.py` writes JSON into the test project;
the C# tests assert against it.

```
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python fixtures/make_fixtures.py
```

Commit the JSON it produces. Regenerate it when the fixture data or the
comparison changes — not on every run, or the tests are asserting against
whatever was generated last rather than against a fixed reference.

What is generated here is the decomposition fixtures — **`pca_plain.json`** and
**`pca_whiten.json`**. PCA is deterministic up to a sign, so this is a straight
exactness check: the C# should match scikit-learn to machine precision once the
component signs are put in a canonical order.

The clustering fixtures — `em_*.json`, `quality.json`, `sweep.json` — moved to
[Unsupervised](https://github.com/Otter-Logic/Unsupervised) along with the
algorithms they test. Both generators still build their input from the same
`make_members`, which is why the two files look alike at the top.

## Later: training

When learned models arrive — GraphSAGE first — this is where they get trained,
and `torch` joins `requirements.txt`. The one artefact that crosses into the
plug-in is a `.onnx` file dropped into `/models`, with a sidecar `.json`
recording feature order and normalisation. Training never runs inside Rhino; see
`Rhino3D/docs/machine-learning.md`.

Note the asymmetry, because it is the whole design: a *learned* model has weights
that had to be found from data the user does not have, so it must be trained here
and shipped. A Gaussian mixture has no such weights — it computes its parameters
from whatever is on the wire, every solve — so it is C#, and nothing about it
comes through this directory.
