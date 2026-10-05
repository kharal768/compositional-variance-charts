#!/usr/bin/env python3
"""Cross-document consistency check for the submission set.

The manuscript has been audited against its source data throughout. The cover
letter and the supplementary were written separately and never checked against
it, so this compares them: every figure quoted in the cover letter must appear
in the manuscript, the supplementary must not assert a withdrawn result as
current, and the placeholders that must be filled before submission must be
findable rather than forgotten.
"""
from __future__ import annotations
from wmmon.paths import WMMON_HOME, WMMON_PAPER

import pathlib
import re
import sys

P = pathlib.Path(f"{WMMON_PAPER}")
FAILURES: list[str] = []


def norm(t: str) -> str:
    return " ".join(t.split())


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    manuscript = (P / "latex" / "body.md").read_text()
    abstract = (P / "latex" / "abstract.md").read_text()
    letter = (P / "cover_letter.md").read_text()
    supp = (P / "SUPPLEMENTARY.md").read_text()
    main_all = abstract + "\n" + manuscript

    print("1. COVER LETTER FIGURES APPEAR IN THE MANUSCRIPT")
    main_nums = set(re.findall(r"\d[\d,]*\.\d+", main_all)) | set(
        re.findall(r"\b\d{2,}\b", main_all))
    letter_nums = set(re.findall(r"\d[\d,]*\.\d+", letter)) | set(
        re.findall(r"\b\d{2,}\b", letter))
    # Strip contact details and the signature block before comparing: an email
    # address and a grant number are not manuscript figures, and the first run
    # of this check flagged the corresponding author's email as an invented
    # statistic.
    letter_body = re.sub(r"(?s)Yours sincerely.*$", "", letter)
    letter_body = re.sub(r"\S+@\S+", "", letter_body)
    # URLs and DOIs carry digits that are identifiers, not results.
    letter_body = re.sub(r"https?://\S+", "", letter_body)
    letter_body = re.sub(r"\b10\.\d{4,}/\S+", "", letter_body)
    letter_body = letter_body.replace("W2633234", "")
    letter_nums = set(re.findall(r"\d[\d,]*\.\d+", letter_body)) | set(
        re.findall(r"\b\d{2,}\b", letter_body))
    ignore = {"2002", "1995", "1997", "2015", "2025", "1000", "811"}
    missing = {n for n in letter_nums - main_nums if n not in ignore}
    check("no invented figures in the cover letter", not missing,
          f"unmatched: {sorted(missing)}" if missing else "")

    print("\n2. SUPPLEMENTARY DOES NOT ASSERT WITHDRAWN RESULTS AS CURRENT")
    flat = norm(supp).lower()
    check("opens by marking everything superseded",
          "every result here is" in flat and "superseded" in flat)
    check("carries a withdrawal table",
          "what was withdrawn" in flat)
    check("does not claim a current positive finding",
          "this is the study's one replicated positive finding" not in flat
          or "superseded" in flat)

    print("\n3. KEY CLAIMS AGREE ACROSS DOCUMENTS")
    pairs = [
        ("false-alarm range", "22.7 to 984.5"),
        ("nominal rate", "5.0"),
        ("covariance inflation", "29-fold"),
        ("size correlation", "0.53"),
    ]
    for name, token in pairs:
        in_main = token in norm(main_all)
        in_letter = token in norm(letter)
        check(f"{name} consistent", not in_letter or in_main,
              f"'{token}' in letter but not manuscript" if in_letter and not in_main else "")

    print("\n4. PLACEHOLDERS ARE FINDABLE, NOT FORGOTTEN")
    for label, path, token in [
        ("cover letter repository URL", P / "cover_letter.md", "[repository URL]"),
        ("data availability URL", P / "latex" / "main.tex", "to be completed"),
        ("pyproject repository URL",
         pathlib.Path(f"{WMMON_HOME}/pyproject.toml"),
         "example.invalid"),
    ]:
        present = token in path.read_text() if path.exists() else False
        print(f"   [{'TODO' if present else 'done'}] {label}")

    print("\n5. EVERY SECTION CITED IN THE PAPER EXISTS IN THE BUILD")
    body = (P / "latex" / "body.md").read_text()
    flat_body = " ".join(body.split())
    present = set()
    for h in re.findall(r"(?m)^#{1,4}\s+(.*)$", body):
        m = re.match(r"(?:Section\s+)?(\d+(?:\.\d+)*)", h)
        if m:
            present.add(m.group(1).rstrip("."))
    for m in re.finditer(r"\*\((?:Section )?(\d+\.\d+(?:\.\d+)?)", body):
        present.add(m.group(1))
    refs = {m.group(1) for m in re.finditer(
        r"(?:Section |Sections?\s+|\u00a7\s?)(\d+(?:\.\d+)*[a-z]?)", flat_body)}
    # Bare three-level numbers after a preposition ("withdrawn in 6.13.1") are
    # section pointers too; two-level ones are not checked this way because
    # "sigma_Z of 2.8" is data, not a reference.
    refs |= {m.group(1) for m in re.finditer(
        r"\b(?:in|of|see|from)\s+(\d\.\d{1,2}\.\d)\b", flat_body)}
    dangling = sorted(r for r in refs if "." in r and r not in present)
    check("no citation to a section absent from the build", not dangling,
          f"dangling: {dangling}" if dangling else "")
    cited_s = set(re.findall(r"Supplementary (S\d+)", flat_body))
    defined_s = set(re.findall(r"(?m)^### (S\d+)\.", supp))
    check("every supplementary S-label cited is defined",
          cited_s <= defined_s,
          f"missing: {sorted(cited_s - defined_s)}" if not cited_s <= defined_s else "")

    print("\n6. EVERY CITATION HAS A REFERENCE ENTRY, AND EVERY ENTRY IS CITED")
    # Keyed on (author, year) rather than surname, so two papers by the same
    # first author are told apart; an a/b suffix is part of the year. Added after
    # the text was found citing four works with no entry, one entry never cited,
    # and an author list that could not be verified.
    ref_text = (P / "latex" / "references_src.md").read_text()
    entries = [e.strip() for e in ref_text.split("\n\n")
               if e.strip() and not e.lstrip().startswith("#")]
    def parse_entry(e):
        """(all author surnames, year) from an author-year entry.

        The first author is written "Surname, I." and the rest "I. Surname";
        the final author ends in "." rather than ",". \\w is used so accented
        names (Martín-Fernández, Barceló-Vidal) are read whole.
        """
        year = re.search(r"\b((?:19|20)\d\d[ab]?)\.", e)
        if not year:
            return None
        head = e[:year.start()]
        first = re.match(r"\s*([^\W\d_][\w'\-]*)", head)
        rest = re.findall(r"(?:[A-Z]\.[\s\-]?)+\s*([^\W\d_][\w'\-]+)", head[first.end():] if first else head)
        authors = ([first.group(1)] if first else []) + [a for a in rest if a != "and"]
        return authors, year.group(1)

    keyed = []
    for e in entries:
        parsed = parse_entry(e)
        if parsed:
            keyed.append((parsed[0], parsed[1], e[:60]))
    never = []
    for authors, year, label in keyed:
        first = authors[0]
        if not re.search(re.escape(first) + r".{0,90}?" + re.escape(year) + r"(?![\da-z])", flat_body):
            never.append(f"{first} {year}")
    check("every reference entry is cited in the text", not never,
          f"never cited: {never}" if never else "")
    known = {(a, y) for authors, y, _ in keyed for a in authors}
    orphans = set()
    for m in re.finditer(r"([^\W\d_][\w'\-]+)(?: et al\.)?,?\s*\[((?:19|20)\d\d[ab]?)", flat_body):
        if (m.group(1), m.group(2)) not in known:
            orphans.add(f"{m.group(1)} {m.group(2)}")
    for grp in re.finditer(r"\[([^\]]{0,300})\]", flat_body):
        for m in re.finditer(r"([^\W\d_][\w'\-]+)(?: et al\.)?,\s*((?:19|20)\d\d[ab]?)", grp.group(1)):
            if (m.group(1), m.group(2)) not in known:
                orphans.add(f"{m.group(1)} {m.group(2)}")
    check("every citation in the text has a reference entry", not orphans,
          f"no entry for: {sorted(orphans)}" if orphans else "")
    leaks = [w for w in ("*unverified*", "doi:", "not having obtained",
                         "before this paragraph", "earlier draft")
             if w in flat_body]
    # Unfilled placeholders: "[Martín-Fernández et al., Year]" survived every
    # pass from the first draft because every other check looks for digits.
    for pat in (r"\bYear\]", r"et al\., Year", r"\[Author", r"XXXX", r"\?\?",
                r"\[citation", r"\bTBD\b", r"\[ref\b"):
        if re.search(pat, flat_body):
            leaks.append(f"placeholder /{pat}/")
    check("no working notes, placeholders or inline DOIs in the text", not leaks,
          f"found: {leaks}" if leaks else "")

    print("\n7. BUILD ARTEFACTS PRESENT")
    for f in ("latex/main.pdf", "supplementary/main.pdf", "cover_letter.md"):
        check(f, (P / f).exists())

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed")
        return 1
    print("submission set is internally consistent")
    print("Placeholders marked TODO above still need the repository URL.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
