"""Appends one JSON line per event to ``progress.jsonl``, which the C# launcher polls.

A file rather than stdout so that nothing blocks on either side and the history is
there to read when a job goes wrong. Every line is flushed as it is written,
because the reader is looking at the file while it grows.
"""

from __future__ import annotations

import json
from pathlib import Path


class Progress:
    def __init__(self, path: Path) -> None:
        self._path = path
        self._handle = path.open("a", encoding="utf-8")

    def stage(self, stage: str, message: str) -> None:
        self._write({"stage": stage, "message": message})

    def done(self, model: Path, score: dict[str, float], report: list[str]) -> None:
        self._write({
            "stage": "done",
            "message": "Done.",
            "done": True,
            "model": str(model),
            "score": {k: float(v) for k, v in score.items()},
            "report": list(report),
        })

    def error(self, message: str, stage: str | None = None) -> None:
        line = {"error": message}
        if stage:
            line["stage"] = stage
        self._write(line)

    def close(self) -> None:
        self._handle.close()

    def _write(self, record: dict) -> None:
        self._handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._handle.flush()
