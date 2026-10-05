#!/usr/bin/env python3
"""Which component actually causes the improvement?

The proposed monitor bundles four things: an ILR transform, lot-size
standardisation, a bootstrap control limit, and conditional VAR prewhitening.
Every comparison so far has pitted the whole bundle against outside baselines,
which cannot tell whether the compositional geometry is doing the work or
whether the calibration is.

Two axes are separated here because they answer different questions and are
measured by different quantities.

**Axis 1 -- representation and scaling (measured by detection delay).**
A 2x2 over {ILR, raw proportions} x {lot-size standardised, not}. Thresholds
are set empirically from a held-out clean segment at a fixed false-alarm
budget, identically for every cell, so the internal limit rule is out of play
and only the representation and scaling differ. If the ILR cells do not beat
the proportion cells here, the log-ratio treatment is not the contribution.

**Axis 2 -- the control limit (measured by realised false-alarm rate).**
Calibration cannot be judged by delay: its job is to hit a nominal rate. It is
evaluated by running each limit rule on undegraded lots and comparing the
realised rate to the nominal one.

Feature caches from the replication runs are reused, so this is cheap.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matched_comparison as mc
from wmmon import classifier, composition, data, monitor
from wmmon.monitor import _mewma_statistics  # noqa: F401

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)

CHANGE_POINTS = [450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
TARGETS = [1.0, 5.0]
LAM = 0.2
PHASE_ONE = 300


def mewma_score(matrix: np.ndarray, sizes: np.ndarray, rate_invariant: bool,
                phase_one: int = PHASE_ONE, lam: float = LAM) -> np.ndarray:
    """MEWMA statistic on an arbitrary coordinate matrix.

    Deliberately one code path for every cell of the ablation: the same
    estimator, the same smoothing constant, the same Phase I window. The only
    things that vary are the matrix passed in (log-ratio coordinates or raw
    proportions) and whether observations are scaled by their lot size. A
    pseudo-inverse is used because the raw-proportion covariance is singular by
    construction -- rows sum to one -- which is itself part of what the ILR
    transform exists to avoid.
    """
    matrix = np.asarray(matrix, dtype=float)
    sizes = np.asarray(sizes, dtype=float)
    reference = matrix[:phase_one]
    mean = reference.mean(axis=0)
    scale = (
        np.sqrt(sizes / np.median(sizes[:phase_one]))
        if rate_invariant
        else np.ones(len(sizes))
    )
    standardised = (matrix - mean) * scale[:, None]
    # Same covariance estimator as the production monitor (Ledoit-Wolf
    # shrinkage), not a plain sample covariance. A different estimator here
    # would silently make the full cell something other than the method being
    # ablated.
    cov = monitor._shrunk_covariance(standardised[:phase_one], None)
    cov_inv = np.linalg.pinv(np.atleast_2d(cov))
    # The recursion starts AFTER Phase I, exactly as monitor.run_monitor does,
    # and the returned series is indexed from the first Phase II lot. Returning
    # the full-length series instead would shift every delay by phase_one lots.
    return _mewma_statistics(standardised[phase_one:], cov_inv, lam)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+",
                    default=["resolution_loss", "defect_remix", "new_signature"])
    ap.add_argument("--confined-modes", nargs="*", default=["defect_remix"])
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-seconds", type=int, default=250)
    args = ap.parse_args()
    t0 = time.time()

    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, _, stream = data.split_labelled(df, seed=args.seed)
    stream = stream.reset_index(drop=True)
    del df

    model = classifier.train_inspection_model(
        np.load(CACHE / "train.npz")["x"], train["label"].tolist(), CLASSES,
        seed=args.seed,
    )
    blocks = []
    for path in sorted(CACHE.glob("stream_part_*.npz")):
        with np.load(path) as z:
            blocks.append((int(z["lo"]), z["x"]))
    blocks.sort(key=lambda t: t[0])
    x_all = np.vstack([b for _, b in blocks])

    sequence = data.lot_order(stream)[: args.stream_lots]
    keep = set(sequence)
    lots_all = stream["lot"].to_numpy(dtype=object)
    rows = np.flatnonzero(np.array([str(l) in keep for l in lots_all]))
    lots = lots_all[rows]
    position = np.array([{l: i for i, l in enumerate(sequence)}[str(l)] for l in lots])
    x_clean = x_all[rows]

    proba_clean, frame_clean = mc.build(model, x_clean, lots, sequence)
    coords_clean = composition.ilr_matrix(frame_clean)
    props_clean = composition.proportion_matrix(frame_clean)
    sizes = frame_clean["lot_size"].to_numpy()

    CELLS = {
        "ilr + lot-size scaling": ("ilr", True),
        "ilr only": ("ilr", False),
        "proportions + lot-size scaling": ("prop", True),
        "proportions only": ("prop", False),
    }

    def scores(coords, props):
        return {
            name: mewma_score(coords if rep == "ilr" else props, sizes, ri)
            for name, (rep, ri) in CELLS.items()
        }

    clean_scores = scores(coords_clean, props_clean)
    out = {"change_points": [], "axis1": {}, "axis2": {}}

    # ---------------- Axis 2: the control limit ----------------------------
    print("Axis 2 -- control limit rules on undegraded lots")
    print(f"{'rule':<34s}{'limit':>10s}{'realised FA/1000':>20s}  (nominal 5.00)")
    warmup = 20
    n_eval = len(coords_clean) - PHASE_ONE - warmup
    for label, kwargs in [
        ("bootstrap + prewhiten auto", dict(prewhiten="auto")),
        ("bootstrap, prewhiten forced on", dict(prewhiten=True)),
        ("bootstrap, prewhiten off", dict(prewhiten=False)),
    ]:
        fit = monitor.calibrate(coords_clean[:PHASE_ONE], sizes[:PHASE_ONE],
                                lam=LAM, target_far=0.005, seed=args.seed, **kwargs)
        alarm = monitor.run_monitor(
            fit, coords_clean[PHASE_ONE:], sizes[PHASE_ONE:])["alarm"]
        rate = 1000.0 * int(alarm[warmup:].sum()) / n_eval
        out["axis2"][label] = {"limit": fit.limit, "realised_fa_per_1000": rate}
        print(f"{label:<34s}{fit.limit:>10.2f}{rate:>20.2f}")

    chi2 = monitor.chi_square_limit(coords_clean.shape[1], 0.005)
    stat = mewma_score(coords_clean, sizes, True)[PHASE_ONE:]
    rate = 1000.0 * int((stat[warmup:] > chi2).sum()) / n_eval
    out["axis2"]["asymptotic chi-square limit"] = {
        "limit": chi2, "realised_fa_per_1000": rate}
    print(f"{'asymptotic chi-square limit':<34s}{chi2:>10.2f}{rate:>20.2f}")

    # ---------------- Axis 1: representation and scaling -------------------
    print("\nAxis 1 -- representation and scaling, thresholds matched\n")
    for mode in args.modes:
        confine = "_conf" if mode in args.confined_modes else ""
        per_cell: dict[str, dict[str, list]] = {c: {str(t): [] for t in TARGETS}
                                                for c in CELLS}
        used = []
        for cp_lot in CHANGE_POINTS:
            if time.time() - t0 > args.budget_seconds:
                break
            path = CACHE / (f"deg_{mode}{confine}_s1.0_r0_cp{cp_lot}_seed{args.seed}.npz")
            if not path.exists():
                continue
            post = position >= cp_lot
            x_deg = x_clean.copy()
            x_deg[post] = np.load(path)["x"]
            _, frame_deg = mc.build(model, x_deg, lots, sequence)
            deg_scores = scores(composition.ilr_matrix(frame_deg),
                                composition.proportion_matrix(frame_deg))
            cp = cp_lot - PHASE_ONE
            for cell in CELLS:
                clean, deg = clean_scores[cell], deg_scores[cell]
                for target in TARGETS:
                    thr = float(np.quantile(clean[cp:], 1.0 - target / 1000.0,
                                            method="higher"))
                    hits = np.flatnonzero(deg[cp:] > thr)
                    per_cell[cell][str(target)].append(
                        float(hits[0]) if len(hits) else float("inf"))
            used.append(cp_lot)

        out["change_points"] = used
        out["axis1"][mode] = {}
        print(f"--- {mode}{confine}   ({len(used)} change points)")
        print(f"{'cell':<32s}" + "".join(f"{f'@{t:g}/1000':>22s}" for t in TARGETS))
        for cell in CELLS:
            cells_text = []
            out["axis1"][mode][cell] = {}
            for target in TARGETS:
                delays = per_cell[cell][str(target)]
                finite = [d for d in delays if np.isfinite(d)]
                out["axis1"][mode][cell][str(target)] = {
                    "n_runs": len(delays), "n_detected": len(finite),
                    "median": float(np.median(finite)) if finite else None,
                    "min": float(min(finite)) if finite else None,
                    "max": float(max(finite)) if finite else None,
                }
                cells_text.append(
                    f"{len(finite)}/{len(delays)}  med "
                    f"{np.median(finite):.0f}" if finite else f"0/{len(delays)}  never"
                )
            print(f"{cell:<32s}" + "".join(f"{c:>22s}" for c in cells_text))
        print()

    RESULTS.mkdir(parents=True, exist_ok=True)
    with open(RESULTS / "ablation.json", "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote results/ablation.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
