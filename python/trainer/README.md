# otterlogic-trainer

The process behind the Train component. A dataset folder in, one `.onnx` out, with
everything the model knows about itself written inside the file.

```
python -m otterlogic_trainer job.json
python -m otterlogic_trainer --version
```

`--version` prints `otterlogic-trainer 0.2.0` and exits 0. It exists so that an
installer can prove a runtime works — the interpreter starts, the package imports —
without inventing a job to give it; the bundle build runs it before zipping.

`job.json` is what `OtterLogic.MachineLearning.Training.TrainingJob` writes:

```json
{
  "datasetFolder": "C:/datasets/member-releases",
  "target": "rel_My",
  "learner": { "type": "boostedTrees", "trees": 100, "depth": 0 },
  "holdoutBy": "group",
  "holdoutFraction": 0.25,
  "seed": 1,
  "outputPath": "C:/models/releases.onnx"
}
```

Beside it the trainer appends `progress.jsonl`, one line per stage and a last
line that is either `{"done": true, "model": …, "score": {…}, "report": […]}` or
`{"error": "…"}`. The C# launcher, `TrainerProcess`, polls that file.

What a run does, in order: read every CSV against `schema.json`; hold out whole
models with the seed — or rows, see below; fit a `StandardScaler` and the chosen
learner on the training rows only; score on the held-out rows; export to ONNX with
the fixed input `features` and outputs `value` or `label` + `probabilities`; write
the metadata record into the file; **run the exported graph through onnxruntime on
the held-out rows and refuse to write it if it disagrees with scikit-learn**; then
write beside the target and rename over it.

## Learners

`learner` is one object with a `type` and that learner's settings beside it. The
names are the camelCase property names of the `Learner` records in
`Training/Learner.cs`; `models.py` reads exactly these and nothing else is tunable.
Every learner sits behind a `StandardScaler` in the exported graph, so the C# side
never scales and cannot scale wrongly.

| `type` | Settings (default) | Classifier | Regressor |
|---|---|---|---|
| `boostedTrees` | `trees` (100), `depth` (0 = no limit) | `HistGradientBoostingClassifier` | `HistGradientBoostingRegressor` |
| `neuralNetwork` | `hiddenLayers` ([64]), `iterations` (500) | `MLPClassifier` | `MLPRegressor` |
| `linear` | `regularisation` (1.0) | `LogisticRegression`, C = 1/regularisation | `Ridge`, alpha = regularisation |
| `nearestNeighbours` | `neighbours` (5) | `KNeighborsClassifier` | `KNeighborsRegressor` |

A missing `learner` means boosted trees with the defaults. An unknown `type` is a
job error that lists the four. The model's metadata records the type under
`trainer.modelType` — the key the C# `ModelMetadata` reads and the committed
fixtures carry, so it did not change name when the job did.

A learner is added only when the round-trip check in `export.verify` passes for it
on both tasks. skl2onnx has operators it converts badly, and the failure mode is a
model that runs and is quietly wrong. All four above pass on both tasks with
skl2onnx 1.20 and onnxruntime 1.23.

## Holding out rows

`holdoutBy` is `group` unless the job says `row`. By group is the honest score:
rows from one model are near-copies of each other, so a row held out at random has
its twins in the training set and the score measures recognition, not prediction.

`row` exists for samples that arrived on a wire with no groups at all — the C#
`TrainerProcess.StartOnSamples` chooses it when fewer than two distinct groups were
given. The report's first line then says that rows were held out at random and the
score may be optimistic, so the number is never read without its caveat, and
`data.holdoutGroups` in the metadata is empty.

## Development

Point the plug-in at a checkout instead of an installed bundle:

```
cd python
py -3.12 -m venv .venv
.venv\Scripts\activate
pip install -e trainer[test]
set OTTERLOGIC_TRAINER=%CD%\.venv
```

Rhino inherits the variable from the shell that starts it. Run the tests with
`pytest trainer`. They build a synthetic dataset folder, train every learner on
both tasks, check the file round-trips through onnxruntime, and run the command
line both ways.

## The bundle

A user of the plug-in installs no Python. Train's context menu fetches a bundle: a
python-build-standalone CPython 3.12 with this package and its wheels installed,
zipped, hashed, and attached to a GitHub release of this repo.

```
cd python\trainer
.\build-bundle.ps1 -PythonSha256 <hash published beside the archive>
```

The script (Windows PowerShell 5.1 is enough) downloads the `install_only`
archive named by `-PythonUrl`, unpacks it, `pip install`s this checkout into it,
trims what the trainer never uses (the interpreter's test suite, IDLE, Tk), and
writes two files to `dist\`:

- `otterlogic-trainer-<version>-win-x64.zip` — the bundle. At its root sits
  `bundle.json`, `{"version": "0.2.0", "python": "python/python.exe"}`, which is
  how `TrainerRuntime` recognises a bundle and finds its interpreter; the
  interpreter itself sits one folder down because that is how the archive unpacks.
- `trainer-manifest.json` — `version`, `url`, `sha256`, `bytes`: the shape of
  `TrainerRuntime.TrainerBundle`. The URL is where the zip will be once it is
  attached to a release tagged `trainer-v<version>`.

Cutting the release is by hand, and the script prints the `gh release create`
line. The manifest must sit on the repo's *latest* release, because
`TrainerRuntime.ManifestUrl` is `releases/latest/download/trainer-manifest.json`.
`TrainerRuntime.InstallAsync` reads it, downloads the zip, checks the hash and
unpacks into `%LOCALAPPDATA%\OtterLogic\trainer\<version>\`;
`InstallFromFile` takes the same zip from a share, for a machine that cannot reach
GitHub. Both unpack into a `<version>.installing` folder that is renamed into place
only at the end, so nothing half-done is ever found.

The version in `pyproject.toml` and `__init__.py` must agree; the build refuses to
zip when `--version` answers anything else. Bump both when the trainer changes —
the bundle is released on the trainer's version, not the plug-in's.
