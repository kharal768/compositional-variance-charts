"""Where the data, caches, results and manuscript live.

Every script used to hard-code the author's own directories, so a clone would
fail on the first file it opened. Each location now comes from an environment
variable, defaulting to a directory beside or inside this repository.

    WMMON_HOME     parent of data/, cache/ and results/  (default: repository root)
    WMMON_PAPER    the manuscript directory               (default: ../paper)

To reproduce, place LSWMD.pkl under $WMMON_HOME/data/ and run the scripts; they
write intermediate files to cache/ and outputs to results/.
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO = Path(__file__).resolve().parents[1]

WMMON_HOME = Path(os.environ.get("WMMON_HOME", _REPO)).resolve()
WMMON_PAPER = Path(os.environ.get("WMMON_PAPER", _REPO.parent / "paper")).resolve()
DATA = WMMON_HOME / "data"
CACHE = WMMON_HOME / "cache"
RESULTS = WMMON_HOME / "results"
