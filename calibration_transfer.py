#!/usr/bin/env python3
"""Calibration transfer as a function of monitoring unit size.

Two established rules for label-free / compositional monitoring were found to
misfire on this data: the asymptotic chi-square limit for an ILR chart
(272.7 false alarms per 1,000 against a nominal 5.0) and MD3's theta*sigma rule
at theta=2, the sensitivity its authors use (492 per 1,000 against 2.7). Both
have the same diagnosis: a reference spread estimated at one scale, applied at
another.

If that diagnosis is right it makes a prediction: the error should shrink as
the monitoring unit grows, and vanish once the unit is large enough that the
asymptotics the rules rest on actually hold. If the error is flat in unit size,
the diagnosis is wrong and the rules are simply miscalibrated here.

Monitoring units are built by grouping consecutive production lots, so unit
size varies while everything else --- classifier, stream, ordering --- is held
fixed. For each unit size this reports the realised in-control false-alarm rate
of three rules against their common nominal target:

  chi2       asymptotic quantile for the ILR-MEWMA statistic
  md3        theta * sigma, sigma estimated out-of-fold at the unit's own scale
  bootstrap  the empirical calibration used throughout this study

The stream is undegraded throughout, so every alarm counted is a false alarm.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import KFold

from wmmon import classifier, composition, data, md3, monitor

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)
# Groupings are capped at 4 lots per unit. Beyond that the 2,860-lot stream
# leaves fewer than 20(D-1) = 160 units for Phase I, the VAR(1) filter cannot be
# fitted, and the bootstrap column stops measuring the method this study
# proposes. The 8- and 16-lot rows from the first run are withdrawn for that
# reason rather than reported with a caveat.
GROUPINGS = [1, 2, 4]
NOMINAL_PER_1000 = 5.0
THETA = 2.0


def out_of_fold_indicator(x, y, seed=0, n_splits=5):
    ind = np.zeros(len(x))
    for tr, te in KFold(n_splits=n_splits, shuffle=True, random_state=seed).split(x):
        ens = md3.train_random_subspace(x[tr], [y[i] for i in tr], CLASSES, seed=seed)
        ind[te] = md3.margin_indicator(ens.probabilities(x[te]))
    return ind


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["hard", "soft"], default="hard",
                    help="composition aggregation; 'soft' reproduces the original "
                         "probability-aggregated run, which Finding 1 shows is "
                         "misspecified")
    ap.add_argument("--shuffle-items", action="store_true",
                    help="destroy batch structure: reassign wafers to lot slots at "
                         "random, keeping every unit's size unchanged")
    ap.add_argument("--permute", action="store_true",
                    help="shuffle lot order: removes sequence-driven drift while "
                         "preserving every lot's composition, giving a stream that "
                         "is in control by construction")
    args = ap.parse_args()
    t0 = time.time()
    seed = 0
    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, _, stream = data.split_labelled(df, seed=seed)
    stream = stream.reset_index(drop=True)

    x_train = np.load(CACHE / "train.npz")["x"]
    y_train = train["label"].tolist()
    model = classifier.train_inspection_model(x_train, y_train, CLASSES, seed=seed)
    ens = md3.train_random_subspace(x_train, y_train, CLASSES, seed=seed)
    oof = out_of_fold_indicator(x_train, y_train, seed)
    print(f"reference material ready ({time.time() - t0:.0f}s)", flush=True)

    blocks = []
    for path in sorted(CACHE.glob("stream_part_*.npz")):
        with np.load(path) as z:
            blocks.append((int(z["lo"]), z["x"]))
    blocks.sort(key=lambda t: t[0])
    x_all = np.vstack([b for _, b in blocks])

    sequence = (data.permuted_lot_order(stream, seed=seed) if args.permute
                else data.lot_order(stream))
    lots = stream["lot"].to_numpy(dtype=object)
    proba = model.predict_proba(x_all)
    indicator_stream = md3.margin_indicator(ens.probabilities(x_all))
    if args.shuffle_items:
        # Move each wafer's data into a random wafer slot. `pos` below is left
        # untouched, so every unit keeps exactly the size it had; only which
        # wafers share a unit changes. That isolates batch structure from unit
        # size, which a re-blocking into fixed-size units would confound.
        order = np.random.default_rng(seed).permutation(len(proba))
        proba = proba[order]
        indicator_stream = indicator_stream[order]

    # wafer -> its position in the lot sequence
    lot_position = {l: i for i, l in enumerate(sequence)}
    pos = np.array([lot_position[str(l)] for l in lots])

    rng = np.random.default_rng(seed)
    rows = []

    for group in GROUPINGS:
        unit_of = pos // group
        unit_ids = [f"u{u:06d}" for u in unit_of]
        unit_sequence = [f"u{u:06d}" for u in sorted(set(unit_of))]

        frame = composition.build_stream(
            np.array(unit_ids, dtype=object), proba, unit_sequence, mode=args.mode
        )
        coords = composition.ilr_matrix(frame)
        sizes = frame["lot_size"].to_numpy()
        n_units = len(frame)
        phase_one = max(60, int(0.4 * n_units))
        if n_units - phase_one < 100:
            print(f"group {group}: only {n_units} units, skipping")
            continue
        warmup = 20
        exposure = n_units - phase_one - warmup

        # --- bootstrap (this study) ---
        fit = monitor.calibrate(coords[:phase_one], sizes[:phase_one], lam=0.2,
                                target_far=NOMINAL_PER_1000 / 1000.0, seed=seed)
        run = monitor.run_monitor(fit, coords[phase_one:], sizes[phase_one:])
        boot_rate = 1000.0 * int(run["alarm"][warmup:].sum()) / exposure

        # --- asymptotic chi-square on the same statistic ---
        chi2 = monitor.chi_square_limit(coords.shape[1], NOMINAL_PER_1000 / 1000.0)
        chi_rate = 1000.0 * int((run["statistic"][warmup:] > chi2).sum()) / exposure

        # --- MD3 native rule, sigma estimated at this unit's own scale ---
        unit_size = int(np.median(sizes))
        draws = [
            float(oof[rng.choice(len(oof), size=unit_size, replace=False)].mean())
            for _ in range(2000)
        ]
        sigma = float(np.std(draws, ddof=1))
        density = md3.lot_margin_density(
            indicator_stream, np.array(unit_ids, dtype=object), unit_sequence
        )
        score = md3.md3_score(density, phase_one, lam=0.2)
        md3_rate = 1000.0 * int((score[warmup:] > THETA * sigma).sum()) / exposure

        rows.append({
            "lots_per_unit": group, "n_units": n_units,
            "median_wafers_per_unit": unit_size,
            "chi2_per_1000": chi_rate,
            "md3_theta2_per_1000": md3_rate,
            "bootstrap_per_1000": boot_rate,
            "md3_sigma": sigma, "chi2_limit": chi2, "bootstrap_limit": fit.limit,
        })
        print(f"  group {group:>2} | {n_units:>5} units of ~{unit_size:>4} wafers "
              f"| chi2 {chi_rate:7.2f} | md3 {md3_rate:7.2f} | boot {boot_rate:6.2f}",
              flush=True)

    print(f"\nrealised in-control false alarms per 1,000 units "
          f"(nominal {NOMINAL_PER_1000:.1f})")
    print(f"{'wafers/unit':>12s}{'units':>8s}{'chi2':>10s}{'MD3 th=2':>10s}"
          f"{'bootstrap':>11s}")
    for r in rows:
        print(f"{r['median_wafers_per_unit']:>12d}{r['n_units']:>8d}"
              f"{r['chi2_per_1000']:>10.2f}{r['md3_theta2_per_1000']:>10.2f}"
              f"{r['bootstrap_per_1000']:>11.2f}")

    RESULTS.mkdir(exist_ok=True)
    tag = (("_permuted" if args.permute else "") + ("_counts" if args.mode == "hard" else "")
           + ("_itemshuffled" if args.shuffle_items else ""))
    json.dump({"nominal_per_1000": NOMINAL_PER_1000, "theta": THETA,
               "permuted": bool(args.permute), "mode": args.mode,
               "items_shuffled": bool(args.shuffle_items), "rows": rows},
              open(RESULTS / f"calibration_transfer{tag}.json", "w"), indent=2)
    print(f"\nwrote results/calibration_transfer{tag}.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
