#!/usr/bin/env python3
"""Test the imbalance mechanism on WM-811K by rebalancing it downward.

The `digits` sweep found compositional detection failing below a dominant-class
share of roughly 0.4 and improving monotonically above it. That was one dataset
moving from balanced toward imbalanced. This moves the primary dataset in the
opposite direction: WM-811K natively sits near 0.83 predicted-`none`, and here
that share is reduced by subsampling `none` wafers out of the stream.

If the mechanism is real, ILR-MEWMA should degrade as the share falls and
should approach failure near 0.4, while confidence-based detection stays flat.
If instead it holds up at low dominance on WM-811K, the threshold found on
`digits` is dataset-specific and must be reported as such.

Everything runs from cached CNN probabilities, so no re-classification is
needed: subsampling is an index selection on an existing probability matrix.
Only the dominant class is thinned --- every other wafer is kept, and the
degradation, the change points and the protocol are untouched.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

import matched_comparison as mc
from wmmon import composition, data, monitor
from wmmon.monitor import _mewma_statistics, _shrunk_covariance

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)
SHARES = [None, 0.70, 0.55, 0.40, 0.30]   # None = native
UNIT_WAFERS = 24
HORIZON = 200.0
TARGET = 1.0
CHANGE_POINTS = [450, 550, 650, 750, 850, 950]


def proportion_statistic(props, sizes, phase_one):
    mean = props[:phase_one].mean(axis=0)
    scale = np.sqrt(sizes / np.median(sizes[:phase_one]))
    std = (props - mean) * scale[:, None]
    cov_inv = np.linalg.pinv(np.atleast_2d(_shrunk_covariance(std[:phase_one], None)))
    return _mewma_statistics(std[phase_one:], cov_inv, 0.2)


def main() -> None:
    t0 = time.time()
    seed = 0
    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    _, _, stream = data.split_labelled(df, seed=seed)
    stream = stream.reset_index(drop=True)

    sequence = data.permuted_lot_order(stream, seed=seed)[:1200]
    keep = set(sequence)
    lots_all = stream["lot"].to_numpy(dtype=object)
    rows_idx = np.flatnonzero(np.array([str(l) in keep for l in lots_all]))
    lots = lots_all[rows_idx]
    lot_pos = {l: i for i, l in enumerate(sequence)}
    position = np.array([lot_pos[str(l)] for l in lots])

    proba_clean = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
    none_idx = CLASSES.index("none")
    predicted = proba_clean.argmax(axis=1)
    native = float((predicted == none_idx).mean())
    print(f"stream {len(proba_clean):,} wafers | native predicted-none share "
          f"{native:.2f}", flush=True)

    rng = np.random.default_rng(seed)
    other = np.flatnonzero(predicted != none_idx)
    dominant = np.flatnonzero(predicted == none_idx)

    out_rows = []
    for share in SHARES:
        if share is None:
            selected = np.arange(len(proba_clean))
        else:
            n_keep = int(round(len(other) * share / (1.0 - share)))
            if n_keep > len(dominant):
                print(f"  share {share}: needs more dominant wafers than exist, skip")
                continue
            selected = np.sort(np.concatenate(
                [other, rng.choice(dominant, size=n_keep, replace=False)]))

        sel_pos = position[selected]
        # Units are consecutive runs of retained wafers, so unit size is held
        # fixed while the class mix changes. Grouping by original lot instead
        # would shrink units as the dominant class is thinned, confounding unit
        # size with balance.
        n_units = len(selected) // UNIT_WAFERS
        if n_units < 260:
            print(f"  share {share}: only {n_units} units, skip")
            continue
        selected = selected[: n_units * UNIT_WAFERS]
        sel_pos = sel_pos[: n_units * UNIT_WAFERS]
        unit_ids = np.array([f"u{i // UNIT_WAFERS:06d}" for i in range(len(selected))],
                            dtype=object)
        unit_seq = [f"u{i:06d}" for i in range(n_units)]
        phase_one = max(160, int(0.4 * n_units))

        def score(p_full):
            p = p_full[selected]
            frame = composition.build_stream(unit_ids, p, unit_seq, mode="soft")
            coords = composition.ilr_matrix(frame)
            props = composition.proportion_matrix(frame)
            sizes = frame["lot_size"].to_numpy()
            conf = np.array([p[i * UNIT_WAFERS:(i + 1) * UNIT_WAFERS].max(axis=1).mean()
                             for i in range(n_units)])[phase_one:]
            fit = monitor.calibrate(coords[:phase_one], sizes[:phase_one], lam=0.2,
                                    target_far=0.005, seed=seed)
            return {
                "ilr_mewma": monitor.run_monitor(
                    fit, coords[phase_one:], sizes[phase_one:])["statistic"],
                "proportion_mewma": proportion_statistic(props, sizes, phase_one),
                "ks_confidence": mc.score_ks(conf),
            }, frame

        clean, frame_clean = score(proba_clean)
        realised = float(composition.proportion_matrix(frame_clean).mean(axis=0)[none_idx])

        per_method: dict[str, list[float]] = {}
        used = 0
        for cp_lot in CHANGE_POINTS:
            path = CACHE / (f"cnnproba_defect_remix_conf_perm_s1.0_r0_"
                            f"cp{cp_lot}_seed{seed}.npz")
            if not path.exists():
                continue
            p_deg = proba_clean.copy()
            p_deg[position >= cp_lot] = np.load(path)["p"]
            scores, _ = score(p_deg)
            # the change point in unit coordinates for this subsample
            k = int(np.searchsorted(sel_pos, cp_lot) // UNIT_WAFERS) - phase_one
            if k <= 5 or k >= n_units - phase_one - 20:
                continue
            used += 1
            for name, s in scores.items():
                thr = float(np.quantile(clean[name][k:], 1.0 - TARGET / 1000.0,
                                        method="higher"))
                hits = np.flatnonzero(s[k:] > thr)
                per_method.setdefault(name, []).append(
                    float(hits[0]) if len(hits) else float("inf"))

        row = {"target_share": share, "realised_share": realised,
               "n_units": n_units, "n_change_points": used, "methods": {}}
        for name, delays in per_method.items():
            fin = [d for d in delays if np.isfinite(d)]
            row["methods"][name] = {
                "n_detected": len(fin), "n_runs": len(delays),
                "restricted_mean": float(np.mean(
                    [min(d, HORIZON) if np.isfinite(d) else HORIZON for d in delays]))}
        out_rows.append(row)
        cells = "  ".join(
            f"{n.split('_')[0]}:{row['methods'][n]['n_detected']}/"
            f"{row['methods'][n]['n_runs']}"
            f"({row['methods'][n]['restricted_mean']:.0f})"
            for n in per_method)
        print(f"  none share {realised:.2f} | {n_units:>4} units | {cells}", flush=True)

    print(f"\n{'none share':>12s}{'units':>8s}{'ILR det':>10s}{'ILR RM':>9s}"
          f"{'KS det':>9s}{'KS RM':>8s}")
    for r in out_rows:
        m = r["methods"]
        print(f"{r['realised_share']:>12.2f}{r['n_units']:>8d}"
              f"{m['ilr_mewma']['n_detected']:>7d}/{m['ilr_mewma']['n_runs']}"
              f"{m['ilr_mewma']['restricted_mean']:>9.0f}"
              f"{m['ks_confidence']['n_detected']:>6d}/{m['ks_confidence']['n_runs']}"
              f"{m['ks_confidence']['restricted_mean']:>8.0f}")

    if len(out_rows) >= 4:
        xs = [r["realised_share"] for r in out_rows]
        ys = [r["methods"]["ilr_mewma"]["restricted_mean"] for r in out_rows]
        rho, p = stats.spearmanr(xs, ys)
        print(f"\nSpearman(none share, ILR restricted-mean delay) = "
              f"{rho:+.3f}, p = {p:.4f}")

    RESULTS.mkdir(exist_ok=True)
    json.dump({"dataset": "wm811k", "native_share": native, "unit": UNIT_WAFERS,
               "rows": out_rows}, open(RESULTS / "imbalance_wm811k.json", "w"), indent=2)
    print(f"\nwrote results/imbalance_wm811k.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
