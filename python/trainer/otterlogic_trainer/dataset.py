"""Reads a dataset folder the way ``OtterLogic.MachineLearning.Data.DatasetFolder`` writes it.

A folder is ``schema.json`` and one CSV per model under ``models/``. The checks here
are the C# reader's checks, because a file the plug-in would refuse must be refused
here too — a trainer that is more lenient than the reader trains on rows the user
never sees, and the model is wrong with no error raised.
"""

from __future__ import annotations

import csv
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

SCHEMA_FILE = "schema.json"
MODELS_FOLDER = "models"
CURRENT_FORMAT = 1


@dataclass(frozen=True)
class Column:
    name: str
    role: str            # feature | target | id
    kind: str            # number | category
    unit: str | None = None
    classes: tuple[str, ...] = ()


@dataclass(frozen=True)
class Schema:
    extractor_version: str
    columns: tuple[Column, ...]

    @property
    def features(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.role == "feature")

    @property
    def targets(self) -> tuple[Column, ...]:
        return tuple(c for c in self.columns if c.role == "target")

    def column(self, name: str) -> Column:
        for column in self.columns:
            if column.name.lower() == name.lower():
                return column
        names = ", ".join(c.name for c in self.columns)
        raise ValueError(f"No column named '{name}'. The columns are: {names}.")


@dataclass
class Table:
    schema: Schema
    features: np.ndarray                         # float64, rows by features
    targets: dict[str, np.ndarray] = field(default_factory=dict)   # name -> float64 or str array
    groups: list[str] = field(default_factory=list)
    ids: list[str] = field(default_factory=list)

    @property
    def row_count(self) -> int:
        return int(self.features.shape[0])


def read_schema(folder: Path) -> Schema:
    path = folder / SCHEMA_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"'{folder}' is not a dataset folder: it has no {SCHEMA_FILE}. "
            "One is created the first time a model is written there."
        )

    raw = json.loads(path.read_text(encoding="utf-8"))
    if raw.get("formatVersion", CURRENT_FORMAT) != CURRENT_FORMAT:
        raise ValueError(
            f"{SCHEMA_FILE} is format {raw.get('formatVersion')} and this trainer reads format {CURRENT_FORMAT}."
        )

    columns = tuple(
        Column(
            name=c["name"],
            role=c.get("role", "feature"),
            kind=c.get("kind", "number"),
            unit=c.get("unit"),
            classes=tuple(c.get("classes") or ()),
        )
        for c in raw.get("columns", [])
    )

    schema = Schema(extractor_version=str(raw.get("extractorVersion", "1")), columns=columns)
    _validate_schema(schema)
    return schema


def _validate_schema(schema: Schema) -> None:
    seen: set[str] = set()
    for column in schema.columns:
        if not column.name or column.name != column.name.strip():
            raise ValueError(f"Column '{column.name}' has no name, or a space at the start or end of it.")
        key = column.name.lower()
        if key in seen:
            raise ValueError(f"Two columns are both called '{column.name}'.")
        seen.add(key)
        if column.role not in ("feature", "target", "id"):
            raise ValueError(f"Column '{column.name}' has the role '{column.role}', which is not one.")
        if column.kind not in ("number", "category"):
            raise ValueError(f"Column '{column.name}' has the kind '{column.kind}', which is not one.")
        if column.role == "feature" and column.kind != "number":
            raise ValueError(f"Feature '{column.name}' is a category. Features are numbers.")

    if not schema.features:
        raise ValueError("The dataset has no feature columns.")


def models(folder: Path) -> list[str]:
    """Model IDs in the folder, in the order the C# reader takes them: ordinal by name."""
    models_dir = folder / MODELS_FOLDER
    if not models_dir.is_dir():
        return []
    return sorted(p.stem for p in models_dir.glob("*.csv"))


def read_folder(folder: Path) -> Table:
    schema = read_schema(folder)
    ids = models(folder)
    if not ids:
        raise ValueError(f"'{folder}' has a schema and no models under {MODELS_FOLDER}/.")

    columns = schema.columns
    feature_index = [j for j, c in enumerate(columns) if c.role == "feature"]
    id_index = next((j for j, c in enumerate(columns) if c.role == "id"), None)
    target_index = [(j, c) for j, c in enumerate(columns) if c.role == "target"]

    rows: list[list[float]] = []
    targets: dict[str, list] = {c.name: [] for _, c in target_index}
    groups: list[str] = []
    row_ids: list[str] = []

    for model in ids:
        file = f"{MODELS_FOLDER}/{model}.csv"
        path = folder / MODELS_FOLDER / f"{model}.csv"
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            records = [r for r in csv.reader(handle) if any(field.strip() for field in r)]

        if not records:
            raise ValueError(f"{file} is empty.")

        header = records[0]
        if len(header) != len(columns) or any(
            h.strip().lower() != c.name.lower() for h, c in zip(header, columns)
        ):
            raise ValueError(
                f"{file} has the columns [{', '.join(header)}] and {SCHEMA_FILE} says "
                f"[{', '.join(c.name for c in columns)}]. Re-export that model, or remove the file."
            )

        for r, record in enumerate(records[1:], start=2):
            if len(record) != len(columns):
                raise ValueError(f"{file}, line {r}: {len(record)} values for {len(columns)} columns.")

            rows.append([_number(record[j], columns[j], file, r) for j in feature_index])

            for j, column in target_index:
                if column.kind == "number":
                    targets[column.name].append(_number(record[j], column, file, r))
                else:
                    if record[j] not in column.classes:
                        raise ValueError(
                            f"{file}, line {r}: '{record[j]}' is not one of the classes {SCHEMA_FILE} lists for "
                            f"'{column.name}' ({', '.join(column.classes)}). Re-export that model so the class is recorded."
                        )
                    targets[column.name].append(record[j])

            row_ids.append(record[id_index] if id_index is not None else "")
            groups.append(model)

    features = np.asarray(rows, dtype=np.float64).reshape(len(rows), len(feature_index))
    packed = {
        name: (np.asarray(values, dtype=np.float64) if schema.column(name).kind == "number" else np.asarray(values, dtype=object))
        for name, values in targets.items()
    }
    return Table(schema=schema, features=features, targets=packed, groups=groups, ids=row_ids)


def _number(text: str, column: Column, file: str, line: int) -> float:
    try:
        value = float(text)
    except ValueError:
        value = math.nan
    if not math.isfinite(value):
        raise ValueError(f"{file}, line {line}: '{text}' is not a finite number, in column '{column.name}'.")
    return value
