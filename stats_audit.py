#!/usr/bin/env python3
"""Statistical rigor audit of this study's results.

Everything computed here is a CHECK on the results in `results/`, not a new
finding to be folded into them. Nothing in this file edits or replaces a
reported number; it recomputes from the stored per-replication records and
reports where the reporting is weaker than the data allows or stronger than it
supports.

Four things are checked:

1. **False-alarm rates against nominal.** Rates have been quoted as point
   estimates with no interval. They are binomial counts; an exact interval is
   available and should be reported.

2. **Detection delay is right-censored.** "Never detected" is a censored
   observation, not a missing one. Every median delay in this study is computed
   over the detected subset only, which is biased upward in coverage and
   downward in delay --- a method that detects 3 of 12 at a median of 51 lots
   looks better than one detecting 12 of 12 at 93, and is not. Censoring-aware
   summaries are computed here for comparison.

3. **Paired comparisons without correction.** Method comparisons have been made
   by eye across roughly 6 methods x 6 modes x 3 budgets. Paired tests with a
   family-wise correction are computed, with the paired structure respected:
   the same change points are used for every method.

4. **Replication independence.** The twelve "replications" are twelve change
   points on ONE stream. They share most of their lots. Treating them as
   independent overstates precision, and the overlap is quantified.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import glob
import json
import os
from itertools import combinations

import numpy as np
from scipy import stats


def clopper_pearson(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return float(lo), float(hi)


def section(title: str) -> None:
    print(f"\n{'=' * 78}\n{title}\n{'=' * 78}")


# ---------------------------------------------------------------- 1. FA rates
section("1. False-alarm rates: point estimates quoted without intervals")

path = f"{WMMON_HOME}/results/calibration_transfer_permuted.json"
data = json.load(open(path))
nominal = data["nominal_per_1000"]
print(f"nominal {nominal:.1f} per 1,000 units. Exact (Clopper-Pearson) 95% CI,")
print("reconstructing the alarm count from rate x exposure.\n")
print(f"{'unit':>6s}{'rule':>12s}{'alarms/units':>16s}{'rate/1000':>11s}"
      f"{'95% CI /1000':>22s}{'excludes nominal':>18s}")
for row in data["rows"]:
    n = row["n_units"]
    phase_one = max(60, int(0.4 * n))
    exposure = n - phase_one - 20
    for rule, key in (("chi2", "chi2_per_1000"),
                      ("MD3 theta=2", "md3_theta2_per_1000"),
                      ("bootstrap", "bootstrap_per_1000")):
        k = int(round(row[key] * exposure / 1000.0))
        lo, hi = clopper_pearson(k, exposure)
        flag = "yes" if (lo * 1000 > nominal or hi * 1000 < nominal) else "NO"
        print(f"{row['median_wafers_per_unit']:>6d}{rule:>12s}"
              f"{f'{k}/{exposure}':>16s}{row[key]:>11.2f}"
              f"{f'[{lo * 1000:.1f}, {hi * 1000:.1f}]':>22s}{flag:>18s}")
print("\nCheck: the bootstrap rule at the natural unit (22 wafers) is the only")
print("cell whose interval comes close to nominal. Report intervals, not points.")

# ------------------------------------------------------------- 2. censoring
section("2. Detection delay is right-censored; medians are over detected only")

files = sorted(glob.glob(f"{WMMON_HOME}/results/cnn_replicate_*_perm.json"))
print("Reported median (detected only) vs a censoring-aware summary.")
print("'restricted mean' truncates undetected runs at the evaluation horizon,")
print("which is conservative but unbiased in the direction that matters.\n")
print(f"{'mode':<20s}{'method':<20s}{'det':>6s}{'median(det)':>13s}"
      f"{'restricted mean':>17s}")
horizon = 400.0
rows = []
for f in files:
    d = json.load(open(f))
    mode = os.path.basename(f)[15:-10]
    for name, by in d["runs"].items():
        delays = [r["delay"] for r in by["1.0"]]
        finite = [x for x in delays if np.isfinite(x)]
        capped = [min(x, horizon) if np.isfinite(x) else horizon for x in delays]
        rows.append((mode, name, len(finite), len(delays),
                     np.median(finite) if finite else np.nan, float(np.mean(capped))))
for mode, name, k, n, med, rmean in rows:
    if name in ("ilr_mewma", "ks_confidence"):
        print(f"{mode:<20s}{name:<20s}{f'{k}/{n}':>6s}"
              f"{med if np.isfinite(med) else float('nan'):>13.0f}{rmean:>17.0f}")
print("\nCheck: on modes where ILR-MEWMA detects 1-2 of 12, the reported median")
print("is computed from one or two runs. Those cells should carry the detection")
print("count in the same table cell, or be shown as censored, not as a median.")

# ------------------------------------------------- 3. paired tests + correction
section("3. Paired comparisons across change points, Holm-corrected")

print("Wilcoxon signed-rank on paired per-change-point delays, undetected runs")
print("set to the horizon. One family per mode (all method pairs involving")
print("ilr_mewma); Holm correction within the family.\n")
for f in files:
    d = json.load(open(f))
    mode = os.path.basename(f)[15:-10]
    runs = d["runs"]
    if "ilr_mewma" not in runs:
        continue
    base = [min(r["delay"], horizon) if np.isfinite(r["delay"]) else horizon
            for r in runs["ilr_mewma"]["1.0"]]
    tests = []
    for name, by in runs.items():
        if name == "ilr_mewma":
            continue
        other = [min(r["delay"], horizon) if np.isfinite(r["delay"]) else horizon
                 for r in by["1.0"]]
        if len(other) != len(base) or np.allclose(base, other):
            continue
        try:
            p = float(stats.wilcoxon(base, other).pvalue)
        except ValueError:
            continue
        diff = float(np.median(np.array(base) - np.array(other)))
        tests.append((name, diff, p))
    if not tests:
        continue
    order = np.argsort([t[2] for t in tests])
    m = len(tests)
    print(f"{mode}  (n = {len(base)} paired change points)")
    adj_prev = 0.0
    for rank, idx in enumerate(order):
        name, diff, p = tests[idx]
        adj = min(1.0, max(adj_prev, (m - rank) * p))
        adj_prev = adj
        verdict = "ILR faster" if diff < 0 else "ILR slower"
        star = "*" if adj < 0.05 else " "
        print(f"   vs {name:<22s} median diff {diff:>+7.1f} lots  "
              f"p={p:.4f}  Holm p={adj:.4f} {star} {verdict}")
print("\nCheck: unstarred rows are differences the data does not distinguish at")
print("alpha=0.05 after correction. Several comparisons quoted as wins in the")
print("narrative fall here.")

# --------------------------------------------------------- 4. independence
section("4. The twelve replications are not independent")

cps = [450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
total = 1200
overlaps = []
for a, b in combinations(cps, 2):
    shared = total - max(a, b) + min(a, b) - 0  # lots common to both pre/post splits
    overlaps.append(1.0 - abs(a - b) / total)
print(f"change points {cps[0]}-{cps[-1]} on a single {total}-lot stream")
print(f"mean pairwise agreement in the pre/post split: {np.mean(overlaps):.3f}")
print(f"minimum: {min(overlaps):.3f}")
print("\nCheck: two runs whose change points differ by 50 lots classify 96% of")
print("the stream identically. These are twelve views of one stream, not twelve")
print("independent experiments. Quartile ranges across them describe sensitivity")
print("to change-point placement, and must not be read as sampling variability")
print("or converted into a standard error.")

print("\n" + "=" * 78)
print("Summary of gaps this audit found in the study's own reporting")
print("=" * 78)
for line in [
    "No interval estimates anywhere. Every rate and delay is a bare point.",
    "Medians over the detected subset are compared with medians over full sets.",
    "No multiple-comparison control across ~100 informal method comparisons.",
    "Replications are dependent; ranges across them are not sampling error.",
    "No power or sample-size justification for 12 change points or 6 modes.",
]:
    print(f"  - {line}")
