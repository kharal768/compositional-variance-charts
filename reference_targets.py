#!/usr/bin/env python3
"""Do cross-references land on the section they mean?

The dangling-reference check in submission_check.py catches pointers to sections
that do not exist. It cannot catch a pointer to a section that exists but is the
wrong one: after renumbering, "the threshold of Section 4.6" pointed at the
covariate-conditioning subsection instead of the precision rule, and the check
passed it because Section 4.6 exists.

This compares the words around each reference with the text of every section,
by TF-IDF cosine similarity, and flags a reference when the section it names is
not among the closest matches for its own context. It does not understand
meaning; it measures whether the reference and its target talk about the same
things. Each flag needs reading. What it replaces is reading every reference.

    python reference_targets.py [--body PATH] [--top K]
"""
from __future__ import annotations
from wmmon.paths import WMMON_PAPER

import argparse
import pathlib
import re
import sys

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

PAPER = pathlib.Path(f"{WMMON_PAPER}")


def sections(body: str, supp: str) -> dict[str, tuple[str, str]]:
    """Map section keys to (heading, text). Keys: '4.4', '6.2', 'S1', ..."""
    out: dict[str, tuple[str, str]] = {}
    for part in re.split(r"(?m)^(?=#{1,3} )", body):
        head, _, text = part.partition("\n")
        m = re.match(r"#{1,3}\s+(\d+(?:\.\d+)*)\.?\s+(.*)", head)
        if m:
            out[m.group(1)] = (m.group(2).strip(), text)
            f = re.search(r"Finding (\d)", m.group(2))
            if f:
                out[f"Finding {f.group(1)}"] = (m.group(2).strip(), text)
    for part in re.split(r"(?m)^(?=### S\d+)", supp):
        head, _, text = part.partition("\n")
        m = re.match(r"###\s+(S\d+)\.\s+(.*)", head)
        if m:
            out[m.group(1)] = (m.group(2).strip(), text)
    return out


REF = re.compile(
    r"(?:Section |Sections?\s+)(\d+\.\d+(?:\.\d+)?)"          # Section 4.4, Section 6.2
    r"|Supplementary\s+(S\d+)"                             # Supplementary S1
    r"|(Finding \d)")                                     # Finding 1


def references(body: str, window: int = 35):
    flat = " ".join(body.split())
    for m in REF.finditer(flat):
        key = next(g for g in m.groups() if g)
        words_before = flat[:m.start()].split()[-window:]
        words_after = flat[m.end():].split()[:window]
        yield key, " ".join(words_before + words_after), m.start()


def owning_section(body: str, pos_in_flat: int) -> str | None:
    """Which section a reference sits in, so self-references are not flagged."""
    flat_lines = []
    current = None
    offset = 0
    for line in body.split("\n"):
        m = re.match(r"#{1,3}\s+(\d+(?:\.\d+)*)", line)
        if m:
            current = m.group(1)
        n = len(" ".join(line.split())) + 1
        flat_lines.append((offset, current))
        offset += n
    owner = None
    for off, sec in flat_lines:
        if off <= pos_in_flat:
            owner = sec
    return owner


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--body", default=str(PAPER / "latex" / "body.md"))
    ap.add_argument("--top", type=int, default=3)
    ap.add_argument("--mark-correct", nargs="*", default=None,
                    help="record flagged references as read and correct")
    args = ap.parse_args()
    if args.mark_correct:
        import json
        rp = pathlib.Path(__file__).with_name("reference_targets_reviewed.json")
        data = json.loads(rp.read_text()) if rp.exists() else {}
        for d in args.mark_correct:
            data[d] = "correct"
        rp.write_text(json.dumps(data, indent=2))
        print(f"recorded {len(args.mark_correct)} reference(s) as reviewed")
        return 0

    body = pathlib.Path(args.body).read_text()
    supp = (PAPER / "SUPPLEMENTARY.md").read_text()
    secs = sections(body, supp)
    keys = list(secs)
    docs = [f"{secs[k][0]} {secs[k][1]}" for k in keys]
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, min_df=1)
    mat = vec.fit_transform(docs)

    import hashlib
    import json
    reviewed_path = pathlib.Path(__file__).with_name("reference_targets_reviewed.json")
    reviewed = json.loads(reviewed_path.read_text()) if reviewed_path.exists() else {}

    flagged = []
    checked = 0
    for key, context, pos in references(body):
        if key not in secs:
            continue  # absent targets are submission_check.py's job
        checked += 1
        sims = cosine_similarity(vec.transform([context]), mat).ravel()
        # The words around a reference also describe the section it sits in,
        # which then outranks the real target. Exclude the enclosing section.
        owner = owning_section(body, pos)
        if owner and owner != key and not key.startswith(owner + "."):
            for i, k in enumerate(keys):
                if k == owner or (owner and k.startswith(owner + ".")):
                    sims[i] = -1.0
        order = sims.argsort()[::-1]
        ranked = [keys[i] for i in order]
        # Parent and child sections are the same target for this purpose.
        family = {k for k in keys if k == key or k.startswith(key + ".")
                  or key.startswith(k + ".")}
        rank = next(i for i, k in enumerate(ranked) if k in family) + 1
        if rank > args.top:
            digest = hashlib.sha1(f"{key}|{context}".encode()).hexdigest()[:16]
            if reviewed.get(digest) == "correct":
                continue
            best = ranked[0]
            flagged.append((key, rank, best, context, digest))

    print(f"checked {checked} cross-references against {len(keys)} sections\n")
    if not flagged:
        print(f"every reference's target is among the top {args.top} matches for its context")
        return 0
    print(f"{len(flagged)} reference(s) whose named target is not a close match - read these:\n")
    for key, rank, best, context, digest in flagged:
        print(f"  -> {key}  (ranked {rank}; closest match is {best}: "
              f"\"{secs[best][0][:50]}\")  id {digest}")
        print(f"     \"...{context[:230]}...\"\n")
    print("After reading a flag and confirming it is correct, record it with")
    print("    python reference_targets.py --mark-correct ID [ID ...]")
    print("so later runs show only new or changed references.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
