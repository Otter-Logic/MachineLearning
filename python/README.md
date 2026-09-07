# Python

**No user of the plug-in installs any of this.** Not IronPython, not Rhino 8's
embedded CPython, not a system interpreter. Everything here runs on a developer's
machine and stops there; the clustering pipeline that ships is C# end to end.

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

The fixtures answer two separate questions, and it is worth keeping them
separate:

- **`em_*.json`** pin the initial parameters, so both implementations start from
  exactly the same place. From there they should agree to machine precision, and
  any drift is an arithmetic bug. Nothing about local optima enters into it.
- **`quality.json`** and **`sweep.json`** let each implementation initialise
  itself, ten restarts each, and compare on BIC and on how far the two labellings
  agree. Matching parameter for parameter here would be a coincidence — these are
  two non-convex optimisers on the same surface.

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
