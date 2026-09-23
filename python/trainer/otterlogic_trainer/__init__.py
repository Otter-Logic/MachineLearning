"""The OtterLogic trainer: a dataset folder in, a self-describing ``.onnx`` out.

Runs as ``python -m otterlogic_trainer job.json`` in a private Python the plug-in
finds through ``OtterLogic.MachineLearning.Training.TrainerRuntime``. Nothing in
Rhino imports it; the two sides talk through ``job.json``, ``progress.jsonl`` and
the model file, and that is the whole contract.
"""

from __future__ import annotations

__version__ = "0.2.0"

NAME = "otterlogic-trainer"
