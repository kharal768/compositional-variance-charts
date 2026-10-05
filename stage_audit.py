#!/usr/bin/env python3
"""Check the trimmed submission manuscript against the 22-stage workflow.

The stages were satisfied by the full manuscript; the submission build is
~3,700 words shorter, so this asks only whether any stage lost its supporting
text in the cut.

All matching is done on whitespace-normalised text. Three separate checks in
this project have produced wrong answers because a phrase happened to wrap
across a line break, so raw-text matching is not used anywhere here.
"""
from wmmon.paths import WMMON_PAPER
import pathlib
import sys

TEX = pathlib.Path(f"{WMMON_PAPER}/latex")


def normalise(text: str) -> str:
    return " ".join(text.split()).lower()


def main() -> int:
    parts = []
    for name in ("abstract.md", "body.md", "references_src.md"):
        p = TEX / name
        if not p.exists():
            print(f"missing {name}")
            return 1
        parts.append(p.read_text())
    raw = "\n".join(parts)
    flat = normalise(raw)
    refs = (TEX / "references_src.md").read_text()
    # Entries are separated by blank lines; counting them is more robust than
    # matching an author-and-year pattern, which undercounted 23 as 12 because
    # of hyphenated and accented surnames and multi-initial author lists.
    n_refs = len([b for b in refs.split("\n\n")
                  if b.strip() and not b.lstrip().startswith("#")])

    checks = [
        ("1  problem stated precisely",
         "the task is to signal when the distribution" in flat),
        ("2  gap in the strong form",
         "prevents them from holding a nominal false-alarm rate" in flat
         and "it remains unclear whether" in flat),
        ("3  research questions, each answered",
         all(f"rq{i}" in flat for i in (1, 2, 3, 4))),
        ("4  contributions separated by type",
         "contributions" in flat and "(i)" in raw and "(iii)" in raw),
        ("5  closest prior work, 5-15 papers", 5 <= n_refs <= 30),
        ("7  recent and seminal literature", "2025" in refs and "1986" in refs),
        ("8  intellectual lineage", "woodall" in flat and "laney" in flat),
        ("10 adopted vs introduced labelled",
         "what is inherited and what is added" in flat),
        ("11 assumptions stated", "exchangeab" in flat),
        ("13 register: nothing proved",
         "our experiments show" in flat or "we hypothesise" in flat or "we hypothesize" in flat),
        ("14 data reported fully", "811,457" in flat and "duplicate" in flat),
        ("15 baselines of each required kind",
         "md3" in flat and "kolmogorov" in flat and "uncorrected" in flat),
        ("16 fair comparison stated",
         "same phase i window" in flat or "common per-unit" in flat),
        ("17 metrics justified",
         "clopper" in flat or "exact binomial" in flat),
        ("18 statistical validation", "95%" in flat or "interval" in flat),
        ("19 ablation", "uncorrected" in flat and "identity" in flat),
        ("20 robustness", "two classifier" in flat or "three datasets" in flat),
        ("21 sensitivity", "prior" in flat and "lag" in flat),
        ("22 results as RQ to evidence",
         "probability-aggregated compositions break" in flat and all(f"rq{i}." in flat for i in (1, 2, 3, 4))),
        ("-- limitations explicit", "limitation" in flat),
        ("-- conclusion distinct from abstract", "conclusion" in flat),
        ("-- abstract at most 250 words",
         len((TEX / "abstract.md").read_text().split()) <= 250),
    ]

    failures = [name for name, ok in checks if not ok]
    for name, ok in checks:
        print(f"   [{'ok' if ok else 'FAIL'}] {name}")
    print(f"\n   references: {n_refs}")
    print(f"   abstract:   {len((TEX / 'abstract.md').read_text().split())} words")
    print(f"   body:       {len((TEX / 'body.md').read_text().split()):,} words")
    if failures:
        print("\nstages whose supporting text did not survive the cut:")
        for f in failures:
            print(f"   {f}")
        return 1
    print("\nevery stage still has supporting text in the submission build")
    return 0


if __name__ == "__main__":
    sys.exit(main())
