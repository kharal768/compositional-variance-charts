#!/usr/bin/env python3
"""In which order are the fixed blocks cut, and does it change the published rates?

The cached CNN probabilities (cnnproba_clean_1200_perm.npz, written by replicate_cnn.py) are stored in the *row order of
the stream table restricted to the selected lots*, i.e. the order of the original WM-811K file. The permutation only
chooses which 1,200 lots are kept. mechanism_counts.py then cuts consecutive rows into blocks of 24 or 48. This script

  1. rebuilds the row order and reports whether it is lot-contiguous, how it relates to the permuted and the natural lot
     order, and whether wafers within a lot are in waferIndex order;
  2. re-cuts the same probabilities in four orders (file order as cached, permuted lot order, natural lot order, fully
     shuffled) and reports the uncorrected chart's false-alarm rate for blocks of 24 and 48;
  3. compares file order with results/mechanism_counts.json, to confirm that it is the order the published rates used.

It needs the WM-811K subset (data/subset.pkl) and the cached probabilities, so it cannot run in the public archive.
Run:  python check_fixed_block_order.py --stream-lots 1200 --seed 0
"""
from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, laney_coda as L

NOM = 5.0
RESULTS = Path(f"{WMMON_HOME}/results")
CACHE = Path(f"{WMMON_HOME}/cache")


def cpi(k, n):
    lo = stats.beta.ppf(0.025, k, n - k + 1) * 1000 if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) * 1000 if k < n else 1000.0
    return lo, hi


def run(proba, unit, mode="hard"):
    """Identical to mechanism_counts.run: consecutive rows form units of `unit` items."""
    n = len(proba) // unit
    p = proba[: n * unit]
    ids = np.array([f"u{i // unit:06d}" for i in range(len(p))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n)]
    fr = composition.build_stream(ids, p, seq, mode=mode)
    C = composition.ilr_matrix(fr); P = composition.proportion_matrix(fr)
    S = fr["lot_size"].to_numpy()
    ph = max(20 * C.shape[1], int(0.4 * n)); ex = n - ph - 20
    fit = L.calibrate(C[:ph], P[:ph], S[:ph], target_far=NOM / 1000, seed=0)
    k = int(L.run_uncorrected(fit, C[ph:], P[ph:], S[ph:])["alarm"][20:].sum())
    lo, hi = cpi(k, ex)
    return {"rate": 1000 * k / ex, "ci": [lo, hi], "exposure": ex,
            "covers": bool(lo <= NOM <= hi)}


def order_diagnostics(lots, wafer_index, position, natural_rank):
    """Properties of the cached row order. `position` and `natural_rank` are per row (the row's lot)."""
    lots = np.asarray(lots, dtype=object)
    wafer_index = np.asarray(wafer_index, dtype=float)
    n = len(lots)
    runs = int(np.sum(lots[1:] != lots[:-1])) + 1
    distinct = len(set(lots.tolist()))
    first = {}
    for i, l in enumerate(lots.tolist()):
        first.setdefault(l, i)
    firsts = np.array([first[l] for l in first])
    pos_l = np.array([position[first[l]] for l in first], dtype=float)
    nat_l = np.array([natural_rank[first[l]] for l in first], dtype=float)
    out = {"rows": n, "distinct_lots": distinct, "runs_of_equal_lot": runs,
           "lot_contiguous": bool(runs == distinct)}
    if distinct > 2:
        out["spearman_file_order_vs_permuted_position"] = float(stats.spearmanr(firsts, pos_l)[0])
        out["spearman_file_order_vs_natural_order"] = float(stats.spearmanr(firsts, nat_l)[0])
    mono = 0; total = 0
    for l, i0 in first.items():
        idx = np.flatnonzero(lots == l)
        w = wafer_index[idx]
        if np.isnan(w).any():
            continue
        total += 1; mono += int(np.all(np.diff(w) >= 0))
    out["lots_with_nondecreasing_waferIndex"] = f"{mono} of {total}"
    return out


def reorder(kind, position, natural_rank, wafer_index, seed=0):
    n = len(position)
    wi = np.where(np.isnan(np.asarray(wafer_index, dtype=float)), 1e9, wafer_index)
    if kind == "file order (as cached)":
        return np.arange(n)
    if kind == "permuted lot order":
        return np.lexsort((np.arange(n), wi, np.asarray(position)))
    if kind == "natural lot order":
        return np.lexsort((np.arange(n), wi, np.asarray(natural_rank)))
    if kind == "fully shuffled":
        return np.random.default_rng(seed).permutation(n)
    raise ValueError(kind)


KINDS = ["file order (as cached)", "permuted lot order", "natural lot order", "fully shuffled"]


def main() -> int:
    from wmmon import data
    ap = argparse.ArgumentParser()
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    _, _, stream = data.split_labelled(df, seed=a.seed)
    stream = stream.reset_index(drop=True)
    sequence = data.permuted_lot_order(stream, seed=a.seed)[: a.stream_lots]      # as replicate_cnn.py
    keep = set(sequence)
    lots_all = stream["lot"].to_numpy(dtype=object)
    rows = np.flatnonzero(np.array([str(l) in keep for l in lots_all]))
    lots = lots_all[rows]
    perm_pos = {l: i for i, l in enumerate(sequence)}
    nat_rank = {l: i for i, l in enumerate(data.lot_order(stream))}
    position = np.array([perm_pos[str(l)] for l in lots])
    natural = np.array([nat_rank[str(l)] for l in lots])
    wafer_index = stream["wafer_index"].to_numpy(dtype=float)[rows]

    proba = np.load(CACHE / f"cnnproba_clean_{a.stream_lots}_perm.npz")["p"]
    if len(proba) != len(rows):
        print(f"MISMATCH: the cached array has {len(proba)} rows but the selection gives {len(rows)}. "
              "The cache was made with a different subset, seed or split; the order cannot be reconstructed.")
        return 1

    diag = order_diagnostics(lots, wafer_index, position, natural)
    print("row order of the cached array:")
    for k, v in diag.items():
        print(f"  {k:<46} {v}")

    results = {}
    print(f"\n{'order':<26}{'unit':>5}{'rate per 1,000':>16}{'95% interval':>20}{'exposure':>10}")
    for kind in KINDS:
        order = reorder(kind, position, natural, wafer_index)
        for unit in (24, 48):
            r = run(proba[order], unit)
            results[f"{kind}|{unit}"] = r
            print(f"{kind:<26}{unit:>5}{r['rate']:>16.1f}   [{r['ci'][0]:6.1f}, {r['ci'][1]:6.1f}]{r['exposure']:>10}")

    pub = RESULTS / "mechanism_counts.json"
    if pub.exists():
        rows_p = {(r["mode"], r["unit"]): r for r in json.load(open(pub))["rows"]}
        print("\nagreement of file order with the published lot-structured rates (hard counts):")
        for unit in (24, 48):
            mine = results[f"file order (as cached)|{unit}"]["rate"]; pubr = rows_p[("hard", unit)]["lot_structured"]["rate"]
            print(f"  unit {unit}: this run {mine:.1f}  published {pubr:.1f}  "
                  f"{'MATCH' if abs(mine - pubr) < 0.05 else 'DIFFERENT: the published rates used another order'}")
    json.dump({"diagnostics": diag, "rates": results}, open(RESULTS / "fixed_block_order_check.json", "w"), indent=2)
    print("\nwrote results/fixed_block_order_check.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
