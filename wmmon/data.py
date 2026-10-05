"""WM-811K (LSWMD) loading, cleaning and lot-stream construction.

The raw file is distributed as a pickled pandas DataFrame (``LSWMD.pkl``,
~214 MB) with one row per wafer and the columns:

    waferMap       2-D numpy array, values {0: outside wafer, 1: good die,
                   2: failed die}; dimensions vary across wafers
    dieSize        number of dies on the wafer
    lotName        identifier of the production lot (e.g. 'lot1')
    waferIndex     position of the wafer within its lot (nominally 1..25)
    trainTestLabel original train/test split flag ('Training'/'Test'/[])
    failureType    expert pattern label, one of the nine classes or []

Unlabelled rows carry an empty list in ``failureType``; roughly 79 % of the
wafers are unlabelled.  Both label columns are stored inconsistently in the
public file (sometimes ``[['Center']]``, sometimes ``'Center'``, sometimes
``[]``), so every accessor here is defensive.

Nothing in this module fabricates data.  Rows that cannot be parsed are
dropped and counted, and the counts are reported by :func:`load_lswmd`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

#: Canonical class order.  ``none`` is kept as a class: it is the dominant
#: category (~85 % of labelled wafers) and excluding it would destroy the
#: composition that this pipeline monitors.
CLASSES: tuple[str, ...] = (
    "Center",
    "Donut",
    "Edge-Loc",
    "Edge-Ring",
    "Loc",
    "Near-full",
    "Random",
    "Scratch",
    "none",
)

CLASS_INDEX = {c: i for i, c in enumerate(CLASSES)}

#: Lower-cased lookup so that variant spellings in the public file
#: ('Edge-Loc' vs 'edge-loc' vs 'Edge_Loc') resolve to one class.
_CLASS_LOOKUP = {c.lower().replace("_", "-"): c for c in CLASSES}


def _scalarise(value):
    """Unwrap the nested-list encoding used by the public LSWMD file."""
    while isinstance(value, (list, tuple, np.ndarray)):
        if len(value) == 0:
            return None
        value = value[0]
    if isinstance(value, (bytes, np.bytes_)):
        value = value.decode("utf-8", errors="ignore")
    if isinstance(value, float) and np.isnan(value):
        return None
    return value


def parse_label(value) -> str | None:
    """Return a canonical class name, or ``None`` when the wafer is unlabelled."""
    raw = _scalarise(value)
    if raw is None:
        return None
    key = str(raw).strip().lower().replace("_", "-")
    return _CLASS_LOOKUP.get(key)


@dataclass
class LoadReport:
    """Row-level accounting for a load, so nothing is silently discarded."""

    n_rows_raw: int = 0
    n_dropped_bad_map: int = 0
    n_dropped_no_lot: int = 0
    n_rows_kept: int = 0
    n_labelled: int = 0
    n_lots: int = 0
    label_counts: dict[str, int] = field(default_factory=dict)

    def render(self) -> str:
        lines = [
            f"rows in file        : {self.n_rows_raw:,}",
            f"dropped (bad map)   : {self.n_dropped_bad_map:,}",
            f"dropped (no lot id) : {self.n_dropped_no_lot:,}",
            f"rows kept           : {self.n_rows_kept:,}",
            f"labelled rows       : {self.n_labelled:,}",
            f"lots                : {self.n_lots:,}",
            "label counts:",
        ]
        total = max(self.n_labelled, 1)
        for name, count in sorted(
            self.label_counts.items(), key=lambda kv: -kv[1]
        ):
            lines.append(f"  {name:<12s} {count:>8,}  ({100 * count / total:5.2f}%)")
        return "\n".join(lines)


def load_lswmd(
    path: str | Path,
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, LoadReport]:
    """Load ``LSWMD.pkl`` into a tidy frame.

    Returns a frame with columns ``wafer_map`` (2-D uint8 array), ``die_size``,
    ``lot``, ``wafer_index`` and ``label`` (``None`` when unlabelled), plus a
    :class:`LoadReport` recording exactly what was dropped.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download WM-811K (LSWMD.pkl) from the Kaggle "
            "mirror or the MIR Lab page and pass its path with --data."
        )

    raw = pd.read_pickle(path)
    if not isinstance(raw, pd.DataFrame):
        raise TypeError(f"expected a DataFrame in {path}, got {type(raw)!r}")

    report = LoadReport(n_rows_raw=len(raw))
    if max_rows is not None:
        raw = raw.iloc[:max_rows]

    records = []
    for row in raw.itertuples(index=False):
        wmap = getattr(row, "waferMap", None)
        wmap = np.asarray(wmap) if wmap is not None else None
        if wmap is None or wmap.ndim != 2 or wmap.size == 0:
            report.n_dropped_bad_map += 1
            continue

        lot = _scalarise(getattr(row, "lotName", None))
        if lot is None:
            report.n_dropped_no_lot += 1
            continue

        label = parse_label(getattr(row, "failureType", None))
        wafer_index = _scalarise(getattr(row, "waferIndex", None))
        die_size = _scalarise(getattr(row, "dieSize", None))

        records.append(
            {
                "wafer_map": wmap.astype(np.uint8),
                "die_size": float(die_size) if die_size is not None else np.nan,
                "lot": str(lot),
                "wafer_index": float(wafer_index)
                if wafer_index is not None
                else np.nan,
                "label": label,
            }
        )

    df = pd.DataFrame.from_records(records)
    report.n_rows_kept = len(df)
    if len(df):
        labelled = df["label"].notna()
        report.n_labelled = int(labelled.sum())
        report.n_lots = int(df["lot"].nunique())
        report.label_counts = (
            df.loc[labelled, "label"].value_counts().to_dict()
        )
    return df, report


def lot_order(df: pd.DataFrame) -> list[str]:
    """Production order of lots.

    WM-811K carries no timestamp.  The public file preserves the order in
    which lots were exported, and ``lotName`` is of the form ``lot<k>``; where
    that numeric suffix parses it is used as the sort key, otherwise the fallback is first
    appearance in the file.  Both choices are assumptions about the production
    sequence and must be stated as such in any paper that uses them --- see
    :func:`permuted_lot_order` for the robustness check.
    """
    lots = list(dict.fromkeys(df["lot"].tolist()))

    def numeric_key(name: str):
        digits = "".join(ch for ch in name if ch.isdigit())
        return int(digits) if digits else None

    keys = [numeric_key(l) for l in lots]
    if all(k is not None for k in keys):
        return [l for _, l in sorted(zip(keys, lots))]
    log.warning("lotName has no numeric suffix; falling back to file order")
    return lots


def permuted_lot_order(df: pd.DataFrame, seed: int = 0) -> list[str]:
    """A random lot order, for the sequence-assumption robustness experiment."""
    rng = np.random.default_rng(seed)
    lots = np.array(lot_order(df), dtype=object)
    rng.shuffle(lots)
    return list(lots)


def split_labelled(
    df: pd.DataFrame,
    train_fraction: float = 0.6,
    seed: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split into (classifier-train, classifier-test, monitoring-stream).

    The split is **by lot**, not by wafer.  Wafers within a lot are correlated,
    so a wafer-level split leaks information into the test set and inflates
    classifier accuracy --- a reviewer will look for this.

    The monitoring stream is every remaining lot, labelled or not: the whole
    point of the method is that it runs without labels.
    """
    labelled_lots = sorted(df.loc[df["label"].notna(), "lot"].unique())
    rng = np.random.default_rng(seed)
    shuffled = np.array(labelled_lots, dtype=object)
    rng.shuffle(shuffled)

    n_train = int(round(train_fraction * len(shuffled)))
    train_lots = set(shuffled[:n_train])
    test_lots = set(shuffled[n_train:])

    labelled = df["label"].notna()
    train = df[labelled & df["lot"].isin(train_lots)].copy()
    test = df[labelled & df["lot"].isin(test_lots)].copy()
    stream = df[~df["lot"].isin(train_lots)].copy()
    return train, test, stream
