"""``python -m otterlogic_trainer job.json``

Reads the job, writes ``progress.jsonl`` beside it, and exits 0 with a ``done``
line or 1 with an ``error`` line. A traceback goes to stderr, which the launcher
keeps for its log; the user sees the error line.

``python -m otterlogic_trainer --version`` prints the package name and version
and exits 0. It exists so an installer can prove a runtime works — the
interpreter starts, the package imports — without inventing a job to give it.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path

from . import NAME, __version__
from .progress import Progress
from .train import JobError, run

USAGE = "usage: python -m otterlogic_trainer job.json | --version"


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv

    if argv == ["--version"]:
        print(f"{NAME} {__version__}")
        return 0

    if len(argv) != 1 or argv[0].startswith("-"):
        print(USAGE, file=sys.stderr)
        return 2

    job_file = Path(argv[0])
    progress = Progress(job_file.with_name("progress.jsonl"))
    try:
        job = json.loads(job_file.read_text(encoding="utf-8"))
        result = run(job, progress)
        progress.done(result["model"], result["score"], result["report"])
        return 0
    except JobError as ex:
        progress.error(str(ex))
        return 1
    except Exception as ex:  # noqa: BLE001 - anything else is still reported, with its traceback kept
        traceback.print_exc()
        progress.error(f"{type(ex).__name__}: {ex}")
        return 1
    finally:
        progress.close()


if __name__ == "__main__":
    sys.exit(main())
