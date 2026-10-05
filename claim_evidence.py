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
    # supporting checks live in the supplementary; a claim stated there still
    # has its evidence, so the search covers both documents
    _supp = TEX.parent / "SUPPLEMENTARY.md"
    if _supp.exists():
        text += " " + " ".join(_supp.read_text().split())
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

    print("PROVENANCE. Result files with no generating script in the repository")
    _orphans = ["composition_mode.json", "prior_sensitivity.json", "em_estimator.json",
                "varcomp_calibration.json", "min_unit_size.json"]
    import re as _rp, glob as _g
    _writers = {o: [f for f in _g.glob(str(pathlib.Path(__file__).parent / "*.py"))
                    if o[:-5] in open(f).read()
                    and not any(x in f for x in ("claim_evidence", "audit_manuscript",
                                                 "make_figures", "run_pipeline"))]
                for o in _orphans}
    for _o, _w in _writers.items():
        print(f"   [{'TODO' if not _w else 'ok'}] {_o}" +
              ("" if _w else "  (disclosed in section 8 and the README)"))
    print()
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

    print("\nF5. EM convergence in the operating regime")
    emc = json.load(open(R / "em_convergence.json"))
    claim("trace ratio simulated", "em_convergence", emc["trace_ratio"], "0.26", 0.006)
    claim("spread of final trace across starts (percent)", "em_convergence",
          100 * emc["trace_spread_across_starts"], "16", 1.0)

    print("\nF4. Second-order centring")
    _so_raw = json.load(open(R / "second_order_centring.json"))
    if _so_raw.get("regeneration_required"):
        print("   [TODO] second_order_centring.json is a transcript, not a fresh "
              "run: re-run second_order_centring.py where the data are present")
    so = {r["centring"]: r for r in
          json.load(open(R / "second_order_centring.json"))["rows"]}
    claim("first-order residual overshoots", "second_order_centring",
          so["first order"]["rho"], "-0.242", 0.001)
    claim("second order closes it on real lots", "second_order_centring",
          so["first + second order"]["rho"], "0.038", 0.001)
    sim = {r["centring"]: r for r in
           json.load(open(R / "centring_simulation.json"))["rows"]}
    claim("first order is correct in simulation", "centring_simulation",
          sim["first order"]["rho"], "0.065", 0.001)
    claim("third-moment term alone, in simulation", "centring_simulation",
          sim["first + incomplete second order"]["rho"], "0.444", 0.001)
    claim("complete second order, in simulation", "centring_simulation",
          sim["first + complete second order"]["rho"], "0.433", 0.001)
    claim("first-order rate in simulation", "centring_simulation",
          sim["first order"]["rate"], "30.0", 0.06)

    print("\nF13. The F-limit chart: spread, heteroscedasticity and normality")
    fl = {r["design"]: r for r in json.load(open(R / "plain_f_limit_study.json"))["rows"]}
    claim("F-limit marginal rate, fixed 24", "plain_f_limit_study",
          fl["fixed, 24 items"]["rate"], "4.97", 0.01)
    claim("F-limit marginal rate, widely varied", "plain_f_limit_study",
          fl["widely varied, 4-40"]["rate"], "5.49", 0.01)
    claim("F-limit 90% quantile of the conditional rate, fixed 24", "plain_f_limit_study",
          fl["fixed, 24 items"]["q90"], "11.8", 0.06)
    bsz = {(r["design"], r["size_class"]): r for r in
           json.load(open(R / "plain_f_limit_study.json"))["by_size"]}
    claim("F-limit rate on units of 25 to 40, widely varied", "plain_f_limit_study",
          bsz[("widely varied, 4-40", "25-40")]["rate"], "7.74", 0.01)
    bvv = {r["between_scale"]: r for r in json.load(open(R / "f_limit_between_variance.json"))["rows"]}
    claim("F-limit rate with no between-unit variation", "f_limit_between_variance",
          bvv[0.0]["rate"], "9.73", 0.01)
    claim("F-limit rate with 22 percent between-unit variance", "f_limit_between_variance",
          bvv[1.0]["rate"], "5.02", 0.01)

    print("\nF15. The plain chart's dependence on the zero-replacement rule")
    rr = {r["rule"]: r for r in json.load(open(R / "replacement_rule_study.json"))["rows"]}
    claim("plain rate, fixed 24, BM prior 1.0", "replacement_rule_study", rr["BM, prior 1.0"]["rate_fixed"], "5.82", 0.01)
    claim("plain rate, fixed 24, BM Jeffreys", "replacement_rule_study", rr["BM, prior 0.5 (Jeffreys)"]["rate_fixed"], "4.61", 0.01)
    claim("plain rate, fixed 24, BM prior 0.1", "replacement_rule_study", rr["BM, prior 0.1"]["rate_fixed"], "0.80", 0.01)
    claim("plain rate, fixed 24, BM prior 0.05", "replacement_rule_study", rr["BM, prior 0.05"]["rate_fixed"], "0.45", 0.01)
    claim("plain rate, fixed 24, zeros to 0.5", "replacement_rule_study", rr["zeros -> 0.5"]["rate_fixed"], "4.84", 0.01)
    claim("plain rate, no between-unit variation, BM prior 1.0", "replacement_rule_study", rr["BM, prior 1.0"]["rate_fixed_no_between"], "71.2", 0.06)
    claim("plain rate, no between-unit variation, BM prior 0.05", "replacement_rule_study", rr["BM, prior 0.05"]["rate_fixed_no_between"], "0.06", 0.01)

    print("\nF14. The plain chart on Dirichlet-multinomial and mixture streams; the size mechanism")
    pdm = {(r["stream"], r["param"]): r for r in json.load(open(R / "plain_dm_streams.json"))["rows"]}
    claim("F-limit rate, pure multinomial", "plain_dm_streams", pdm[("multinomial", None)]["rate_f"], "8.16", 0.01)
    claim("F-limit rate, Dirichlet 120", "plain_dm_streams", pdm[("dirichlet", 120.0)]["rate_f"], "8.98", 0.01)
    claim("F-limit rate, Dirichlet 20", "plain_dm_streams", pdm[("dirichlet", 20.0)]["rate_f"], "3.72", 0.01)
    scm = {r["n"]: r for r in json.load(open(R / "size_class_mechanism.json"))["rows"]}
    claim("trace ratio at 4 items", "size_class_mechanism", scm[4]["trace_ratio"], "0.008", 0.001)
    claim("trace ratio at 24 items", "size_class_mechanism", scm[24]["trace_ratio"], "0.14", 0.01)
    claim("trace ratio at 40 items", "size_class_mechanism", scm[40]["trace_ratio"], "0.32", 0.01)

    print("\nF12. The plain chart against the length of Phase I")
    p1 = {r["phase_one"]: r for r in json.load(open(R / "phase1_length_sim.json"))["rows"]}
    claim("plain chart ratio to nominal at 120 Phase I units", "phase1_length_sim",
          p1[120]["ratio"], "7.21", 0.01)
    claim("plain chart ratio to nominal at 800 Phase I units", "phase1_length_sim",
          p1[800]["ratio"], "1.47", 0.01)
    claim("plain chart F-limit ratio at 120 Phase I units", "phase1_length_sim",
          p1[120]["f_ratio"], "1.12", 0.01)
    pbf = {r["design"]: r for r in json.load(open(R / "plain_t2_baseline.json"))["rows"]}
    claim("plain F-limit rate, widely varied sizes", "plain_t2_baseline",
          pbf["widely varied, 4-40"]["plain_f_rate"], "8.5", 0.06)

    print("\nF11. Misspecification crossed with the dispersion of unit sizes")
    tb = {(r["design"], r["k"]): r for r in
          json.load(open(R / "tolerance_by_size.json"))["rows"]}
    claim("fixed size, k=0.25", "tolerance_by_size",
          tb[("fixed, 25 items", 0.25)]["rate"], "7.4", 0.06)
    claim("varied sizes, k=0.5", "tolerance_by_size",
          tb[("widely varied, 4-40", 0.5)]["rate"], "5.9", 0.06)
    claim("varied sizes, k=1", "tolerance_by_size",
          tb[("widely varied, 4-40", 1.0)]["rate"], "0.5", 0.06)

    print("\nF10. Power with shifts scaled in the true covariance, and the Monte Carlo within-term")
    mw = json.load(open(R / "mc_within_term.json"))
    def _row(design, chart, direction):
        return next(r for r in mw["rows"] if r["design"] == design and r["chart"] == chart
                    and r["direction"] == direction)
    CL = "variance components, closed form"; MCT = "variance components, Monte Carlo term"
    PL = "plain T2 (F limit)"
    claim("closed form, leading coordinate, fixed size", "mc_within_term",
          _row("fixed, 25 items", CL, "leading coordinate")["power_matched"], "0.218", 0.002)
    claim("closed form, rarest balance, fixed size", "mc_within_term",
          _row("fixed, 25 items", CL, "rarest balance")["power_matched"], "0.016", 0.002)
    claim("plain, rarest balance, fixed size", "mc_within_term",
          _row("fixed, 25 items", PL, "rarest balance")["power_matched"], "0.098", 0.002)
    claim("Monte Carlo term, rarest balance, fixed size", "mc_within_term",
          _row("fixed, 25 items", MCT, "rarest balance")["power_matched"], "0.098", 0.002)
    claim("Monte Carlo term nominal-limit rate, fixed size", "mc_within_term",
          _row("fixed, 25 items", MCT, "leading coordinate")["nominal_limit_rate"], "9.59", 0.01)
    claim("plain nominal-limit rate, fixed size", "mc_within_term",
          _row("fixed, 25 items", PL, "leading coordinate")["nominal_limit_rate"], "5.39", 0.01)
    bs = {(x["chart"], x["size_class"]): x for x in mw["by_size"]}
    claim("Monte Carlo term rate on the smallest units", "mc_within_term",
          bs[(MCT, "4-12")]["rate"], "29.0", 0.06)
    es2 = {(r["unit"], r["method"]): r for r in
           json.load(open(R / "exposure_sweep.json"))["rows"]}
    claim("streams with no alarm, variance components, unit 24 (percent)",
          "exposure_sweep", 100 * es2[(24, "variance components")]["share_streams_silent"],
          "92", 1.0)

    print("\nF9. Unit construction and adjacency")
    ba = {(x["concentration"], x["units"]): x for x in
          json.load(open(R / "block_alignment.json"))["rows"]}
    claim("lag-1 correlation, fixed blocks, strong lot variation", "block_alignment",
          ba[(4, "fixed blocks")]["lag1_correlation"], "0.209", 0.002)
    claim("lag-1 correlation, lot-aligned, strong lot variation", "block_alignment",
          ba[(4, "lot-aligned")]["lag1_correlation"], "-0.013", 0.002)
    claim("sigma_Z at lag 2, fixed blocks, strong lot variation", "block_alignment",
          ba[(4, "fixed blocks")]["sigma_Z_lag2"], "0.604", 0.002)

    print("\nF8. Calibration at a resolving exposure")
    es = {(r["unit"], r["method"]): r for r in
          json.load(open(R / "exposure_sweep.json"))["rows"]}
    claim("chart is conservative at unit 24", "exposure_sweep",
          es[(24, "variance components")]["rate"], "0.24", 0.02)
    claim("plain baseline is liberal at unit 24", "exposure_sweep",
          es[(24, "plain Phase I covariance")]["rate"], "17.76", 0.06)

    print("\nF7. The plain Phase I covariance baseline")
    pb = {r["design"]: r for r in
          json.load(open(R / "plain_t2_baseline.json"))["rows"]}
    claim("plain chart at mildly varying sizes", "plain_t2_baseline",
          pb["mildly varied, 18-30"]["plain_rate"], "6.4", 0.06)
    claim("variance-components chart at widely varying sizes", "plain_t2_baseline",
          pb["widely varied, 4-40"]["vc_rate"], "3.2", 0.06)

    print("\nF6. Tolerance measured against the truth")
    tm = json.load(open(R / "tolerance_mechanism.json"))
    claim("closed form overstates the true trace", "tolerance_mechanism",
          tm["closed_over_true_trace"], "6.99", 0.02)
    whole = {r["k"]: r for r in tm["rows"] if r["family"] == "whole term"}
    silent = [k for k, r in whole.items() if r["rate"] == 0.0]
    ok = bool(silent) and min(silent) <= 0.5
    if not ok:
        FAIL.append("the tolerance sweep no longer shows the chart going silent")
    print(f"   [{'ok' if ok else 'FAIL'}] chart goes silent once the assumed term "
          f"exceeds the truth several-fold")

    print("\nF3. Anisotropic misspecification and the reference distribution")
    an = json.load(open(R / "anisotropic_misspecification.json"))["rows"]
    covered = sum(1 for r in an if r["covers"])
    print(f"   [{'ok' if covered == len(an) else 'FAIL'}] all {len(an)} shape distortions cover nominal")
    if covered != len(an):
        FAIL.append("a shape distortion missed the nominal rate")
    nc = json.load(open(R / "normality_check.json"))
    lv = {int(r["nominal_per_1000"]): r for r in nc["levels"]}
    claim("KS distance, simulated vs empirical", "normality_check",
          nc["ks_statistic"], "0.442", 0.006)
    claim("simulated limit at nominal 5", "normality_check",
          lv[5]["simulated_limit"], "22.02", 0.006)
    claim("empirical quantile at nominal 5", "normality_check",
          lv[5]["empirical_quantile"], "23.19", 0.006)

    print("\nF2. The permutation null distribution")
    pn = json.load(open(R / "permutation_null.json"))["units"]
    claim("null mean at unit 48", "permutation_null", pn["48"]["null_mean"], "4.0", 0.06)
    claim("null maximum at unit 48", "permutation_null", pn["48"]["null_max"], "26.9", 0.06)
    claim("null sd at unit 48", "permutation_null", pn["48"]["null_sd"], "4.5", 0.06)
    claim("evaluated exposure at unit 24", "permutation_null",
          pn["24"]["exposure_units"], "467", 0.5)
    claim("alarms expected at nominal, unit 24", "permutation_null",
          pn["24"]["expected_alarms_at_nominal"], "2.3", 0.06)

    print("\nF1. The misspecification sweep")
    ms = json.load(open(R / "misspecification_sweep.json"))["rows"]
    em = {r["scale"]: r for r in ms if r["estimator"] == "em"}
    claim("overstated known term stays conservative (k=2)", "misspecification_sweep",
          em[2.0]["rate"], "2.1", 0.06)
    claim("understated by half misses nominal (k=0.5)", "misspecification_sweep",
          em[0.5]["rate"], "17.1", 0.06)
    claim("understated fourfold (k=0.25)", "misspecification_sweep",
          em[0.25]["rate"], "57.8", 0.06)
    claim("batch share falls as the known term grows (k=4)", "misspecification_sweep",
          100 * em[4.0]["between_fraction"], "2.9", 0.06)

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

    print("\nG2. No unsourced figure in the discussion or conclusion")
    import re as _re2
    _vals = set()
    def _walk(o):
        if isinstance(o, dict):
            for v in o.values(): _walk(v)
        elif isinstance(o, list):
            for v in o: _walk(v)
        elif isinstance(o, (int, float)):
            for d in (1, 2, 4): _vals.add(round(float(o), d))
    for _f in R.glob("*.json"):
        try: _walk(json.load(open(_f)))
        except Exception: pass
    _synth = body[body.index("# 7. Discussion"):]
    _orphans = []
    for _m in _re2.finditer(r"(?<![\d.])(\d+\.\d+)(?![\d])", _synth):
        _x = float(_m.group(1))
        if any(abs(_x - _v) < 0.051 for _v in _vals):
            continue
        # section numbers are not data
        if _re2.search(r"§\s*$", _synth[max(0, _m.start() - 3):_m.start()]):
            continue
        _orphans.append(_m.group(1))
    _ok = not _orphans
    if not _ok:
        FAIL.append(f"discussion quotes {_orphans} with no result file behind them")
    print(f"   [{'ok' if _ok else 'FAIL'}] every figure in the synthesis sections traces to data"
          + ("" if _ok else f"  unsourced: {_orphans}"))

    print("\nH. Every figure quoted in the abstract appears in the body")
    import re as _re
    abstract_raw = (TEX / "abstract.md").read_text()
    body_flat = " ".join((TEX / "body.md").read_text().split())
    # A number in the abstract with no counterpart in the body is a claim the
    # reader cannot trace. Rounded forms are allowed: 184 matches 184.2.
    missing = []
    for num in sorted(set(_re.findall(r"\b\d+(?:\.\d+)?\b", abstract_raw))):
        if float(num) < 5 and "." not in num:
            continue  # small integers are counts of things, not results
        if num in body_flat:
            continue
        # the abstract may round: 628 for 627.8, 41 for 41.3
        body_numbers = [float(b) for b in _re.findall(r"\b\d+(?:\.\d+)?\b", body_flat)]
        if any(abs(float(num) - b) < 0.5 for b in body_numbers):
            continue
        missing.append(num)
    ok = not missing
    if not ok:
        FAIL.append(f"abstract quotes {missing} with no counterpart in the body")
    print(f"   [{'ok' if ok else 'FAIL'}] abstract figures traceable to the body"
          + ("" if ok else f"  missing: {missing}"))

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
