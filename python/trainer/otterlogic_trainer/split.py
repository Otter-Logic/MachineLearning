"""Holds back whole groups by preference, and rows only when there are no groups to hold.

Rows from one model are near-copies of each other, so a row held out at random has
its twins in the training set and the score measures recognition, not prediction.
That is why ``holdout`` is by group, the same rule as ``GroupSplit`` in C#, and why
``holdout_rows`` exists only for samples that arrived with no groups at all: a
score that may be optimistic is better than none, as long as the report says so.

The shuffle is seeded and the group list is sorted first, so the choice depends on
which groups exist and the seed, not on the order the files were read.

The random stream is Python's, not .NET's, so the same seed picks different groups
here and in Split By Group. That is fine: which groups were held out is written into
the model, and a score is a score on those.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass


@dataclass(frozen=True)
class Split:
    train_rows: list[int]
    test_rows: list[int]
    train_groups: list[str]
    test_groups: list[str]

    @property
    def test_row_fraction(self) -> float:
        total = len(self.train_rows) + len(self.test_rows)
        return len(self.test_rows) / total if total else 0.0


def holdout(groups: list[str], fraction: float = 0.25, seed: int = 1) -> Split:
    """Whole groups held out, at least one on each side."""
    _require_fraction(fraction)

    distinct = sorted(set(groups))
    if len(distinct) < 2:
        have = "is one" if len(distinct) == 1 else "are none"
        raise ValueError(
            f"Need at least two groups to hold one out, and there {have}. A score from rows of the same "
            "group the model trained on says nothing about the next one — add another model to the dataset first."
        )

    rng = random.Random(seed)
    rng.shuffle(distinct)

    held = _held(fraction, len(distinct))
    test = set(distinct[:held])
    train_rows = [i for i, g in enumerate(groups) if g not in test]
    test_rows = [i for i, g in enumerate(groups) if g in test]

    return Split(
        train_rows=train_rows,
        test_rows=test_rows,
        train_groups=sorted(distinct[held:]),
        test_groups=sorted(test),
    )


def holdout_rows(row_count: int, fraction: float = 0.25, seed: int = 1) -> Split:
    """Rows at random, for samples that came with no groups. At least one row on each side; no groups either side."""
    _require_fraction(fraction)
    if row_count < 2:
        have = "is one" if row_count == 1 else "are none"
        raise ValueError(f"Need at least two rows to hold one out, and there {have}.")

    rows = list(range(row_count))
    random.Random(seed).shuffle(rows)
    held = _held(fraction, row_count)

    # Membership is random; order is the file's, as it is after a group split,
    # so the fitted model sees rows the same way either route.
    return Split(train_rows=sorted(rows[held:]), test_rows=sorted(rows[:held]), train_groups=[], test_groups=[])


def _require_fraction(fraction: float) -> None:
    if not 0.0 < fraction < 1.0:
        raise ValueError("The holdout fraction must be between 0 and 1, exclusive.")


def _held(fraction: float, count: int) -> int:
    # Round half away from zero, as the C# side does, then keep at least one
    # on each side.
    held = int(math.floor(fraction * count + 0.5))
    return max(1, min(held, count - 1))
