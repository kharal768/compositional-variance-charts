#!/usr/bin/env python3
"""Stage 24 on the submission build: every major claim against its evidence.

Each claim below is a statement the manuscript makes, the result file it rests
on, the value in that file, and the literal that must appear in the manuscript.
A claim passes only if all three agree: the file says it, the number matches,
and the paper states it. A claim that the paper makes but no file supports
cannot be expressed here - which is the point: adding it forces a source.
"""
from __future__ import annotations
from wmmon.paths import WMMON_HOME, WMMON_PAPER

import json
import pathlib
import sys

R = pathlib.Path(f"{WMMON_HOME}/results")
TEX = pathlib.Path(f"{WMMON_PAPER}/latex")
FAIL: list[str] = []


def load(name):
    d = json.load(open(R / name))
    if isinstance(d, dict) and "rows" in d and isinstance(d["rows"], list):
        return d, d["rows"]
    return d, d


def row(rows, **kw):
    for r in rows:
        if all(r.get(k) == v for k, v in kw.items()):
            return r
    raise KeyError(kw)


def claim(name, source, value, literal, tol=0.06):
    text = " ".join((TEX / "abstract.md").read_text().split()) + " " + \
        " ".join((TEX / "body.md").read_text().split())
    ok_val = abs(float(literal.replace(",", "")) - value) <= tol * max(1, abs(value)) \
        if tol < 1 else abs(float(literal) - value) <= tol
    ok_txt = literal in text
    status = "ok" if ok_val and ok_txt else "FAIL"
    if status == "FAIL":
        FAIL.append(name)
    why = "" if ok_val else f" (source says {value:.4g})"
    why += "" if ok_txt else " (not stated in the manuscript)"
    print(f"   [{status}] {name:<58s} {literal:>8s}  <- {source}{why}")


def main() -> int:
    _, cm = load("composition_mode.json")
    _, ms = load("min_unit_size.json")
    ag, _ = load("aggregation_covariance.json")
    _, ps = load("prior_sensitivity.json")
    _, em = load("em_estimator.json")
    _, ec = load("estimator_comparison.json")
    _, idf = load("identifiability_sweep.json")
    rb, _ = load("review_batch2.json")
    rr, _ = load("review_responses.json")

    print("A. The calibration failure and its cause")
    claim("uncorrected, counts, WM-811K unit 24", "composition_mode",
          row(cm, dataset="wm811k", unit=24, mode="hard", lag=4)["uncorrected"], "184.2", 0.06)
    claim("uncorrected, counts, WM-811K unit 48", "composition_mode",
          row(cm, dataset="wm811k", unit=48, mode="hard", lag=4)["uncorrected"], "627.8", 0.06)
    claim("uncorrected, counts, digits (holds nominal)", "composition_mode",
          row(cm, dataset="digits", mode="hard")["uncorrected"], "0.0", 0.06)
    claim("uncorrected, probabilities, lowest (WM-811K 24)", "composition_mode",
          row(cm, dataset="wm811k", unit=24, mode="soft", lag=1)["uncorrected"], "683.1", 0.06)

    print("\nB. The correction")
    claim("corrected, counts, lag 4, WM-811K unit 24", "composition_mode",
          row(cm, dataset="wm811k", unit=24, mode="hard", lag=4)["varcomp"], "6.4", 0.06)
    claim("corrected, counts, lag 4, WM-811K unit 48", "composition_mode",
          row(cm, dataset="wm811k", unit=48, mode="hard", lag=4)["varcomp"], "13.5", 0.06)
    claim("counts alone insufficient (lag 1)", "composition_mode",
          row(cm, dataset="wm811k", unit=24, mode="hard", lag=1)["varcomp"], "34.3", 0.06)
    claim("lag alone insufficient (probabilities)", "composition_mode",
          row(cm, dataset="wm811k", unit=24, mode="soft", lag=4)["varcomp"], "77.1", 0.06)

    print("\nC. The two conditions")
    claim("aggregation: trace ratio probabilities/counts", "aggregation_covariance",
          ag["trace_ratio_soft_over_hard"], "29", 0.5)
    claim("aggregation: largest-diagonal ratio", "aggregation_covariance",
          ag["max_diagonal_ratio_soft_over_hard"], "75", 0.5)
    single = row(ms, min_size=1, centring="single Phase I mean")
    claim("size correlation, single centre", "min_unit_size",
          single["rho"], "0.532", 0.001)
    fixed = row(ms, min_size=5, centring="per-unit correction")
    claim("size correlation, corrected, units >= 5", "min_unit_size",
          fixed["rho"], "0.012", 0.001)

    print("\nD. Robustness")
    claim("coverage at prior 0.1, lag 4", "prior_sensitivity",
          row(ps, stream="lot-structured", unit=24, prior=0.1, lag=4)["rate"], "4.3", 0.06)
    emr = row(em, dataset="wm811k", unit=24, estimator="em")
    claim("EM batch share, unit 24", "em_estimator",
          100 * emr["batch"], "20.6", 0.06)
    md3 = rb["md3"]["rows"]
    claim("MD3 under a common empirical threshold", "review_batch2",
          next(r for r in md3 if "empirical" in r["method"])["rate"], "27.8", 0.06)
    claim("MD3 under its own theta=2 rule", "review_batch2",
          next(r for r in md3 if "own" in r["method"])["rate"], "601.7", 0.06)

    print("\nE. Validation and limits")
    mult = next(r for r in ec if r["stream"] == "multinomial")
    claim("sigma_Z on the pure multinomial stream", "estimator_comparison",
          mult["sigma_diff"], "1.07", 0.006)
    claim("uncorrected, simulated multinomial", "estimator_comparison",
          mult["rate_uncorrected"], "30.6", 0.06)
    sc = rr["sampling_covariance"]
    n25 = next(r for r in sc if r["n"] == 25)
    claim("closed-form covariance error at n=25 (percent)", "review_responses",
          100 * n25["max_rel_error"], "41", 1.0)
    r200 = next(r for r in idf if r["n_unit"] == 200)
    claim("total recovered within 13% at trace ratio 0.21", "identifiability_sweep",
          100 * r200["total_rel_error"], "13", 1.0)

    print("\nF0. Section 6.1 and the mechanism, on count-aggregated compositions")
    ct = json.load(open(R / "calibration_transfer_permuted_counts.json"))["rows"]
    cs = json.load(open(R / "calibration_transfer_permuted_counts_itemshuffled.json"))["rows"]
    lot = {r["median_wafers_per_unit"]: r for r in ct}
    shf = {r["median_wafers_per_unit"]: r for r in cs}
    claim("chi2 at 22 wafers, lot-structured", "calibration_transfer_counts",
          lot[22]["chi2_per_1000"], "69.6", 0.06)
    claim("chi2 at 22 wafers, batch structure removed", "..._itemshuffled",
          shf[22]["chi2_per_1000"], "60.7", 0.06)
    claim("chi2 at 65 wafers, batch structure removed", "..._itemshuffled",
          shf[65]["chi2_per_1000"], "2.4", 0.06)
    claim("MD3 at 65 wafers, lot-structured", "calibration_transfer_counts",
          lot[65]["md3_theta2_per_1000"], "330.1", 0.06)
    mc = json.load(open(R / "mechanism_counts.json"))["rows"]
    m48 = next(r for r in mc if r["mode"] == "hard" and r["unit"] == 48)
    claim("uncorrected chart restored by permutation, unit 48", "mechanism_counts",
          m48["item_shuffled"]["rate"], "9.0", 0.06)
    f3 = json.load(open(R / "figure_data.json"))["fig3"]["series"]
    claim("lag-4 dispersion, lot-structured (Figure 3)", "figure_data",
          f3["WM-811K, lot-structured"][2], "1.29", 0.006)
    claim("dispersion, item-shuffled (Figure 3)", "figure_data",
          f3["WM-811K, item-shuffled"][0], "0.62", 0.006)

    print("\nF. Every method the abstract claims is defined in Section 4")
    body = (TEX / "body.md").read_text()
    s4 = " ".join(body.split("# 4. Method")[1].split("# 5.")[0].split()).lower()
    abstract = " ".join((TEX / "abstract.md").read_text().split()).lower()
    # A method named in the abstract but absent from the method section is a
    # claim without its evidence.
    for term, required in [("maximum likelihood", "maximum likelihood"),
                           ("identifiable", "identified"),
                           ("closed form", "closed form"),
                           ("unit size", "unit size")]:
        if term in abstract:
            ok = required in s4
            if not ok:
                FAIL.append(f"abstract claims '{term}' but Section 4 never defines it")
            print(f"   [{'ok' if ok else 'FAIL'}] abstract says '{term}' -> Section 4 contains '{required}'")

    print("\nG. The introduction states the current method, not a superseded one")
    s1 = " ".join(body.split("# 1. Introduction")[1].split("# 2.")[0].split()).lower()
    # The contribution list must describe the method as it now stands.
    for term in ("identif", "maximum likelihood", "counts"):
        ok = term in s1
        if not ok:
            FAIL.append(f"introduction never mentions '{term}'")
        print(f"   [{'ok' if ok else 'FAIL'}] introduction mentions '{term}'")

    print("\n" + "=" * 74)
    if FAIL:
        print(f"{len(FAIL)} claim(s) not supported as stated:")
        for f in FAIL:
            print(f"   {f}")
        return 1
    print("every major claim checked is supported by its result file and stated as such")
    return 0


if __name__ == "__main__":
    sys.exit(main())
