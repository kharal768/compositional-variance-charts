#!/usr/bin/env python3
"""Pre-submission audit of the manuscript.

Three checks, in order of how much damage the failure would do.

1. **Key numeric claims against source data.** Each assertion below names a
   figure that appears in the manuscript text and the result file it must come
   from. A mismatch means the text has drifted from the computation, which is
   the failure mode that survives proofreading.

2. **Superseded figures still asserted as current.** Several results in this
   project were withdrawn after later work - the soft-aggregation calibration
   numbers above all. They are retained in the text as a record, which is
   correct, but only where the surrounding sentence marks them as superseded.
   This check finds withdrawn numbers in files that do not carry the
   corresponding retraction marker.

3. **Invisible and non-ASCII characters.** Zero-width spaces, non-breaking
   spaces, soft hyphens and directional marks survive copy-paste and break
   LaTeX silently. Legitimate typography (en dashes, times signs, Greek) is
   listed separately rather than flagged.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME, WMMON_PAPER

import json
import sys
import unicodedata
from pathlib import Path

PAPER = Path(f"{WMMON_PAPER}")
RESULTS = Path(f"{WMMON_HOME}/results")

# --- 1. claims: (description, file, literal as it appears, loader, expected) ---
def load(name):
    """Return the row list, whether the file stores one directly or nests it.

    Result files were written at different times and are not uniform: some are
    a bare list of rows, others wrap the rows under a key alongside metadata.
    The audit should not fail because of that.
    """
    data = json.load(open(RESULTS / name))
    if isinstance(data, dict):
        for key in ("rows", "runs", "results"):
            if key in data and isinstance(data[key], list):
                return data[key]
        raise KeyError(f"{name}: no row list found")
    return data


def claims():
    out = []

    cm = {(r["dataset"], r["unit"], r["mode"], r["lag"]): r
          for r in load("composition_mode.json")}
    out.append(("WM-811K unit 24 hard lag4 variance components",
                "6.4", cm[("wm811k", 24, "hard", 4)]["varcomp"]))
    out.append(("WM-811K unit 48 hard lag4 variance components",
                "13.5", cm[("wm811k", 48, "hard", 4)]["varcomp"]))
    out.append(("WM-811K unit 24 hard uncorrected",
                "184.2", cm[("wm811k", 24, "hard", 4)]["uncorrected"]))
    out.append(("WM-811K unit 48 hard uncorrected",
                "627.8", cm[("wm811k", 48, "hard", 4)]["uncorrected"]))
    out.append(("WM-811K unit 24 hard lag1 variance components",
                "34.3", cm[("wm811k", 24, "hard", 1)]["varcomp"]))
    out.append(("WM-811K unit 24 soft lag4 variance components",
                "77.1", cm[("wm811k", 24, "soft", 4)]["varcomp"]))

    ps = {(r["stream"], r["unit"], r["prior"], r["lag"]): r
          for r in load("prior_sensitivity.json")}
    out.append(("prior 0.1 lag4 rate",
                "4.3", ps[("lot-structured", 24, 0.1, 4)]["rate"]))
    out.append(("prior 0.1 lag4 sigma_Z",
                "1.81", ps[("lot-structured", 24, 0.1, 4)]["sigma_Z"]))
    out.append(("item-shuffled sigma_Z at prior 0.1",
                "0.99", ps[("item-shuffled", 24, 0.1, 1)]["sigma_Z"]))

    ec = {(r["stream"], r["param"]): r for r in load("estimator_comparison.json")}
    out.append(("Dirichlet alpha=20 uncorrected",
                "184.4", ec[("dirichlet", 20.0)]["rate_uncorrected"]))
    out.append(("pure multinomial, successive differences",
                "6.6", ec[("multinomial", None)]["rate_corrected"]))
    out.append(("pure multinomial, Dirichlet-multinomial route",
                "29.7", ec[("multinomial", None)]["rate_dm"]))

    vc = {(r["dataset"], r["unit"], r["lag"]): r
          for r in load("varcomp_calibration.json")}
    out.append(("soft-mode varcomp WM-811K unit 24 lag 1",
                "139.2", vc[("wm811k", 24, 1)]["varcomp"]))
    return out


def check_claims(text_all: str) -> int:
    failures = 0
    print("1. NUMERIC CLAIMS AGAINST SOURCE DATA")
    for desc, literal, value in claims():
        matches_source = abs(float(literal) - float(value)) < 0.06
        in_text = literal in text_all
        status = "ok" if (matches_source and in_text) else "FAIL"
        if status == "FAIL":
            failures += 1
        note = ""
        if not matches_source:
            note = f"  <-- source says {value:.4g}"
        elif not in_text:
            note = "  <-- not found in manuscript text"
        print(f"   [{status}] {desc:<52s} {literal:>8s}{note}")
    return failures


# --- 2. superseded figures ---
WITHDRAWN = {
    "119.9": "6.2",    # residual attributed to heterogeneity, later explained
    "212.0": "6.2",    # soft-aggregation dispersion-matrix rate
    "76.2": "6.2",     # soft-aggregation varcomp
}
# Documents that record the history of the work quote withdrawn figures on
# purpose; they are not manuscript text.
RECORDS = ("review", "stage2", "sop_compliance", "SUBMISSION_JQT", "references")

# Claims demoted or withdrawn by later results. Prose reads as fluent whether or
# not it is current, so the check is mechanical.
STALE_PHRASES = [
    ("lag separation is necessary and is new", "demoted in 6.14"),
    ("restores nominal coverage where dispersion is homogeneous", "superseded by 6.12"),
    ("dispersion-corrected compositional control chart", "renamed: components-of-variance"),
    ("covers the nominal rate on all three datasets", "over-scoped, see Stage 24 A2"),
    ("share one cause", "over-scoped, see Stage 24 A5"),
    ("does not appear to exist", "narrowed to the surveyed set"),
    ("which we show is necessary when consecutive units are clustered", "lag estimator retracted in 6.6"),
    ("0.97 at every lag", "probability-aggregated value; counts give 0.62"),
    ("ranged from 0.97 to 5.06", "probability-aggregated range"),
    ("residual 212", "withdrawn framing"),
    ("Drift accounts for about a quarter", "withdrawn sliding-window result"),
    ("shown sufficient on one stream", "unification narrowed; mechanism is unit-size dependent"),
    ("common cause of three", "unification narrowed; mechanism is unit-size dependent"),
    ("on three datasets, and that it is", "the failure is on one batch-structured stream"),
    ("earlier drafts of this work", "manuscript text must not refer to its own drafts"),
    ("*unverified*", "working note printed in the manuscript"),
    ("not having obtained the original", "working note printed in the manuscript"),
    ("before this paragraph is submitted", "working note printed in the manuscript"),
    ("If, on reading", "working note printed in the manuscript"),
    ("An earlier version", "manuscript text must not refer to its own drafts"),
    ("the eight questions the workflow", "writing-workflow note printed in the manuscript"),
    ("No result in this paper is proved", "Proposition 1 is proved"),
    ("no underlying generative model", "Section 4.4 defines a two-level model"),
    ("consult only the abstract", "Yashchin has now been read in full"),
    ("An identifiable variance decomposition", "reframed after reading Yashchin 1995"),
    ("To our knowledge no compositional chart", "narrowed to the surveyed set"),
]


def check_superseded(files: dict[str, str]) -> int:
    print("\n2. SUPERSEDED FIGURES AND CLAIMS STILL ASSERTED AS CURRENT")
    failures = 0
    for name, text in files.items():
        if ("supplementary" in name.lower() or "stage2" in name or "review" in name
                or name in ("sop_compliance.md", "SUBMISSION_JQT.md", "references.md")):
            continue
        # MANUSCRIPT.md concatenates the supplementary file, whose superseded
        # claims are deliberate. Check only the part before it.
        if name == "MANUSCRIPT.md":
            marker = "# Supplementary material"
            text = text.split(marker)[0] if marker in text else text
        for phrase, why in STALE_PHRASES:
            if phrase in text:
                print(f"   [FAIL] {name}: \"{phrase}\" - {why}")
                failures += 1
    for name, text in files.items():
        if any(r in name for r in RECORDS) or "supplementary" in name.lower():
            continue
        for number, marker in WITHDRAWN.items():
            if number in text and marker not in text and "supersed" not in text.lower():
                print(f"   [FAIL] {name}: quotes {number} with no reference to Section {marker}")
                failures += 1
    if failures == 0:
        print("   [ok]   every withdrawn figure sits in a file that marks it as such")
    return failures


# --- 3. character sweep ---
SUSPECT = {
    "\u200b": "zero-width space", "\u200c": "zero-width non-joiner",
    "\u200d": "zero-width joiner", "\ufeff": "byte-order mark",
    "\u00a0": "non-breaking space", "\u00ad": "soft hyphen",
    "\u2060": "word joiner", "\u202f": "narrow no-break space",
    "\u200e": "left-to-right mark", "\u200f": "right-to-left mark",
}


def check_characters(files: dict[str, str]) -> int:
    print("\n3. INVISIBLE AND NON-ASCII CHARACTERS")
    failures = 0
    benign: dict[str, int] = {}
    for name, text in files.items():
        for i, ch in enumerate(text):
            if ch in SUSPECT:
                line = text[:i].count("\n") + 1
                print(f"   [FAIL] {name}:{line} {SUSPECT[ch]} (U+{ord(ch):04X})")
                failures += 1
            elif ord(ch) > 127:
                benign[ch] = benign.get(ch, 0) + 1
    if failures == 0:
        print("   [ok]   no zero-width, non-breaking or directional characters")
    if benign:
        shown = ", ".join(
            f"{ch} U+{ord(ch):04X} {unicodedata.name(ch, '?').lower()} x{n}"
            for ch, n in sorted(benign.items(), key=lambda kv: -kv[1])[:12])
        print(f"   note   non-ASCII typography present, review for the target "
              f"template: {shown}")
    return failures


def assembled_is_current() -> tuple[bool, str]:
    """Is MANUSCRIPT.md rebuilt from the current sections?

    The assembled file is derived, so it can fall behind its sources; a stale
    assembly is the file most likely to be sent to a co-author.
    """
    # Single master: the submission build. The long-form section files are
    # archived and are not assembled from.
    order = ["latex/abstract.md", "latex/body.md"]
    parts = [(PAPER / n).read_text().rstrip() for n in order if (PAPER / n).exists()]
    for extra in ("figures/captions.md", "SUPPLEMENTARY.md", "latex/references_src.md"):
        q = PAPER / extra
        if q.exists():
            parts.append("---\n\n" + q.read_text().rstrip())
    expected = "\n\n---\n\n".join(parts) + "\n"
    target = PAPER / "MANUSCRIPT.md"
    if not target.exists():
        return False, "MANUSCRIPT.md does not exist"
    if target.read_text() != expected:
        target.write_text(expected)
        return False, "MANUSCRIPT.md was stale and has been rebuilt; re-run the audit"
    return True, "MANUSCRIPT.md matches its sources"


def main() -> int:
    current, note = assembled_is_current()
    print(f"0. ASSEMBLY\n   [{'ok' if current else 'REBUILT'}] {note}\n")
    files = {p.name: p.read_text() for p in sorted(PAPER.glob("*.md"))}
    files.update({f"figures/{p.name}": p.read_text()
                  for p in sorted((PAPER / "figures").glob("*.md"))})
    text_all = "\n".join(files.values())
    print(f"auditing {len(files)} manuscript files "
          f"({sum(len(t) for t in files.values()):,} characters)\n")

    failures = 0 if current else 1
    failures += check_claims(text_all)
    failures += check_superseded(files)
    failures += check_characters(files)

    print(f"\n{'=' * 70}")
    if failures:
        print(f"{failures} check(s) failed - do not package until resolved")
    else:
        print("all checks passed")
    print("Not covered by this script: whether the citations resolve to the "
          "works claimed. That requires the primary sources.")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
