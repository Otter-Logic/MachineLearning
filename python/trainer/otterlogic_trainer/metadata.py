"""The record a model carries about itself — the twin of ``ModelMetadata`` in C#.

Written into the ``.onnx`` under the key ``otterlogic`` in ``metadata_props``. Keys
are camelCase because that is what the C# JSON options read; the inference parity
fixture is what keeps the two sides agreeing, so a change here is a change there.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone

KEY = "otterlogic"
FORMAT = 1


@dataclass
class Metadata:
    task: str                                   # classification | regression
    features: list[str]
    target: str
    classes: list[str] = field(default_factory=list)
    extractor_version: str | None = None
    trainer_name: str | None = None
    trainer_version: str | None = None
    model_type: str | None = None
    trained: datetime | None = None
    rows: int | None = None
    groups: int | None = None
    holdout_groups: list[str] = field(default_factory=list)
    score: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict:
        record: dict = {
            "format": FORMAT,
            "task": self.task,
            "features": list(self.features),
            "target": self.target,
        }
        if self.task == "classification":
            record["classes"] = list(self.classes)
        if self.extractor_version is not None:
            record["extractorVersion"] = self.extractor_version
        if self.trainer_name is not None:
            record["trainer"] = {
                "name": self.trainer_name,
                "version": self.trainer_version or "",
                "modelType": self.model_type or "",
            }
        if self.trained is not None:
            when = self.trained.astimezone(timezone.utc).replace(microsecond=0)
            record["trained"] = when.isoformat().replace("+00:00", "Z")
        if self.rows is not None and self.groups is not None:
            record["data"] = {
                "rows": int(self.rows),
                "groups": int(self.groups),
                "holdoutGroups": list(self.holdout_groups),
            }
        if self.score:
            record["score"] = {k: float(v) for k, v in self.score.items()}
        return record

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, separators=(",", ":"))
