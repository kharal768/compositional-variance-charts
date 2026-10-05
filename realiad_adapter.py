"""Real-IAD adapter.

Real-IAD ships as per-object image directories plus JSON metadata:

    Real-IAD/
      realiad_1024/<category>/*.jpg
      realiad_jsons/realiad_jsons/<category>.json

Two warnings before you build on this.

**There are no lots.**  WM-811K has ``lotName``: a real production batch, which
is what makes a per-lot composition a meaningful unit.  Real-IAD has no batch
or timestamp field at all.  Any "stream" over it is constructed by grouping
consecutive samples into pseudo-batches, which is an assumption strictly weaker
than WM-811K's.  A reviewer who knows the dataset will spot this, so state it
plainly: Real-IAD supports a cross-sector *sensitivity* experiment, not a second
independent validation of the monitoring premise.

**The JSON schema is not assumed here.**  The field names below are candidates
inferred from the dataset's published description, not verified against the
file --- the adapter could not be tested against real Real-IAD data.  Run

    python realiad_adapter.py --root /path/to/Real-IAD --category audiojack --inspect

first.  It prints the actual keys and a sample record; fix ``FIELD_CANDIDATES``
to match before trusting anything downstream.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np

#: Candidate key names, tried in order.  Extend after running --inspect.
FIELD_CANDIDATES = {
    "image": ("image_path", "img_path", "image", "file_name", "path"),
    "defect": ("defect_name", "defect_code", "anomaly_class", "defect", "label"),
    "category": ("category", "class_name", "object"),
    "view": ("view", "camera", "view_id"),
}

#: Label used for defect-free samples.  Kept as a class: the composition needs
#: the normal share, exactly as ``none`` is kept for WM-811K.
NORMAL_LABEL = "ok"

#: Values that different releases use to mean "no defect".
NORMAL_ALIASES = {"ok", "good", "normal", "0", "none", ""}


def _resolve(record: dict, kind: str) -> str | None:
    for key in FIELD_CANDIDATES[kind]:
        if key in record and record[key] not in (None, ""):
            return str(record[key])
    return None


def load_category(root: str | Path, category: str) -> list[dict]:
    """Read one category's JSON into a flat list of records."""
    root = Path(root)
    candidates = [
        root / "realiad_jsons" / "realiad_jsons" / f"{category}.json",
        root / "realiad_jsons" / f"{category}.json",
        root / f"{category}.json",
    ]
    for path in candidates:
        if path.exists():
            break
    else:
        raise FileNotFoundError(
            "no JSON found for category "
            f"{category!r}; looked in: " + ", ".join(str(c) for c in candidates)
        )

    with open(path) as fh:
        blob = json.load(fh)

    # The file may be a dict of splits or a bare list; accept either.
    if isinstance(blob, dict):
        records: list[dict] = []
        for value in blob.values():
            if isinstance(value, list):
                records.extend(r for r in value if isinstance(r, dict))
        if not records:
            records = [blob]
    elif isinstance(blob, list):
        records = [r for r in blob if isinstance(r, dict)]
    else:
        raise TypeError(f"unexpected JSON root type {type(blob)!r} in {path}")
    return records


def inspect(root: str | Path, category: str) -> None:
    """Print the real schema so FIELD_CANDIDATES can be corrected."""
    records = load_category(root, category)
    print(f"records: {len(records):,}")
    keys = Counter(k for r in records for k in r)
    print("\nkeys (count):")
    for key, count in keys.most_common():
        print(f"  {key:<24s} {count:,}")
    print("\nfirst record:")
    print(json.dumps(records[0], indent=2)[:1500])
    print("\nresolution with current FIELD_CANDIDATES:")
    for kind in FIELD_CANDIDATES:
        print(f"  {kind:<10s} -> {_resolve(records[0], kind)}")
    labels = Counter(_resolve(r, "defect") for r in records)
    print("\ndefect values:")
    for value, count in labels.most_common(20):
        print(f"  {str(value):<24s} {count:,}")


def defect_classes(records: list[dict]) -> list[str]:
    """Canonical class list: normal first, then observed defect names sorted."""
    found = set()
    for record in records:
        raw = _resolve(record, "defect")
        key = (raw or "").strip().lower()
        if key not in NORMAL_ALIASES:
            found.add(raw)
    return [NORMAL_LABEL] + sorted(found)


def class_of(record: dict) -> str:
    raw = _resolve(record, "defect")
    key = (raw or "").strip().lower()
    return NORMAL_LABEL if key in NORMAL_ALIASES else str(raw)


def build_pseudo_batches(
    records: list[dict],
    batch_size: int = 25,
    seed: int | None = None,
) -> tuple[np.ndarray, list[str], list[dict]]:
    """Group records into pseudo-batches of ``batch_size``.

    ``batch_size=25`` matches the nominal WM-811K lot, so the two experiments
    are at least comparable in the number of items per composition.

    ``seed`` shuffles first.  Shuffling destroys whatever weak ordering the
    files carry and makes the batches exchangeable, which is the conservative
    choice: it removes any chance that an apparent detection is an artefact of
    the export order rather than the injected degradation.  Report which you
    used.
    """
    records = list(records)
    if seed is not None:
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(records))
        records = [records[i] for i in order]

    batch_ids = np.array(
        [f"batch{i // batch_size:06d}" for i in range(len(records))], dtype=object
    )
    sequence = list(dict.fromkeys(batch_ids.tolist()))
    return batch_ids, sequence, records


def image_paths(root: str | Path, records: list[dict], resolution: int = 256) -> list[Path]:
    """Absolute paths to the image files, at the chosen resolution directory."""
    root = Path(root)
    base = root / f"realiad_{resolution}"
    out = []
    for record in records:
        relative = _resolve(record, "image")
        if relative is None:
            raise KeyError(
                "no image path field resolved; run --inspect and fix "
                "FIELD_CANDIDATES['image']"
            )
        path = Path(relative)
        out.append(path if path.is_absolute() else base / path)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--category", required=True)
    parser.add_argument("--inspect", action="store_true")
    parser.add_argument("--batch-size", type=int, default=25)
    parser.add_argument("--resolution", type=int, default=256)
    parser.add_argument("--shuffle-seed", type=int, default=None)
    args = parser.parse_args()

    if args.inspect:
        inspect(args.root, args.category)
        return

    records = load_category(args.root, args.category)
    classes = defect_classes(records)
    batch_ids, sequence, ordered = build_pseudo_batches(
        records, args.batch_size, args.shuffle_seed
    )
    paths = image_paths(args.root, ordered, args.resolution)
    missing = sum(1 for p in paths[:200] if not p.exists())

    print(f"category         : {args.category}")
    print(f"records          : {len(records):,}")
    print(f"classes ({len(classes)})   : {classes}")
    print(f"pseudo-batches   : {len(sequence):,} of {args.batch_size}")
    print(f"missing images   : {missing}/200 sampled")
    print(f"example path     : {paths[0]}")
    if missing:
        print(
            "\nImages not found. Check --resolution matches the directory you "
            "extracted, and that the JSON paths are relative to it."
        )


if __name__ == "__main__":
    main()
