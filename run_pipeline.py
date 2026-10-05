#!/usr/bin/env python3
"""End-to-end experiment: WM-811K -> inspection model -> ILR monitoring layer.

Usage
-----
    python run_pipeline.py --data /path/to/LSWMD.pkl --out results/

    # quick check on a subset first
    python run_pipeline.py --data LSWMD.pkl --max-rows 40000 --out results/

Every number the pipeline reports is written to ``results/metrics.json`` so the
manuscript can be audited against it line by line rather than retyped.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wmmon import baselines, classifier, composition, data, drift, monitor


def _prewhiten_arg(mode: str):
    return {"auto": "auto", "on": True, "off": False}[mode]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="path to LSWMD.pkl")
    p.add_argument("--out", default="results", help="output directory")
    p.add_argument("--max-rows", type=int, default=None)
    p.add_argument("--lam", type=float, default=0.2, help="MEWMA smoothing")
    p.add_argument("--target-far", type=float, default=0.005)
    p.add_argument("--phase-one", type=int, default=300, help="Phase I lots")
    # Three-way, not a boolean. "auto" fits the VAR(1) filter only when a
    # holdout test says it helps; "on" forces it. Forcing it on a stream that
    # does not want it quadrupled the false-alarm rate on real WM-811K
    # (55.1 vs 14.4 per 1,000 lots), so "on" is an ablation, never a default.
    p.add_argument("--prewhiten", choices=["auto", "on", "off"], default="auto")
    p.add_argument("--stream-lots", type=int, default=1500, help="Phase II lots")
    p.add_argument("--composition-mode", choices=["soft", "hard"], default="soft")
    p.add_argument(
        "--degradation",
        nargs="*",
        default=["resolution_loss", "rotation", "edge_exclusion", "new_signature"],
    )
    p.add_argument("--ramp-lots", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    results: dict = {"config": vars(args)}
    t0 = time.time()

    print("[1/6] loading WM-811K")
    frame, report = data.load_lswmd(args.data, max_rows=args.max_rows)
    print(report.render())
    results["load"] = {
        "rows_raw": report.n_rows_raw,
        "rows_kept": report.n_rows_kept,
        "labelled": report.n_labelled,
        "lots": report.n_lots,
        "label_counts": report.label_counts,
    }

    print("[2/6] splitting by lot and training the inspection module")
    train, test, stream = data.split_labelled(frame, seed=args.seed)
    print(f"  train {len(train):,} wafers | test {len(test):,} | stream {len(stream):,}")

    from wmmon import features as feat

    x_train = feat.extract_batch(train["wafer_map"].tolist())
    x_test = feat.extract_batch(test["wafer_map"].tolist())
    model = classifier.train_inspection_model(
        x_train, train["label"].tolist(), list(data.CLASSES), seed=args.seed
    )
    quality = classifier.evaluate(model, x_test, test["label"].tolist())
    print(f"  macro-F1 {quality['macro_f1']:.3f} | weighted-F1 {quality['weighted_f1']:.3f}")
    print(quality["report"])
    results["inspection_model"] = {
        k: v for k, v in quality.items() if k != "report"
    }

    print("[3/6] building the in-control lot stream")
    sequence = data.lot_order(stream)
    n_needed = args.phase_one + args.stream_lots
    sequence = sequence[:n_needed]
    subset = stream[stream["lot"].isin(set(sequence))].copy()
    maps = subset["wafer_map"].tolist()
    lots = subset["lot"].to_numpy(dtype=object)

    x_stream = feat.extract_batch(maps)
    proba = model.predict_proba(x_stream)
    baseline_stream = composition.build_stream(
        lots, proba, sequence, mode=args.composition_mode
    )
    coords = composition.ilr_matrix(baseline_stream)
    sizes = baseline_stream["lot_size"].to_numpy()
    print(f"  {len(baseline_stream):,} lots | median lot size {np.median(sizes):.0f}")

    print("[4/6] Phase I calibration")
    fit = monitor.calibrate(
        coords[: args.phase_one],
        sizes[: args.phase_one],
        lam=args.lam,
        target_far=args.target_far,
        rate_invariant=True,
        prewhiten=_prewhiten_arg(args.prewhiten),
        seed=args.seed,
    )
    fit_naive = monitor.calibrate(
        coords[: args.phase_one],
        sizes[: args.phase_one],
        lam=args.lam,
        target_far=args.target_far,
        rate_invariant=False,
        prewhiten=_prewhiten_arg(args.prewhiten),
        seed=args.seed,
    )
    chi2 = monitor.chi_square_limit(fit.dim, args.target_far)
    print(
        f"  bootstrap limit {fit.limit:.2f} (achieved FAR {fit.achieved_far:.4f}) "
        f"| naive chi2 limit {chi2:.2f}"
    )
    results["calibration"] = {
        "limit_bootstrap": fit.limit,
        "limit_chi2": chi2,
        "target_far": fit.target_far,
        "achieved_far": fit.achieved_far,
        "dim": fit.dim,
        "reference_lot_size": fit.reference_lot_size,
        "prewhiten_mode": args.prewhiten,
        "prewhiten_applied": fit.phi is not None,
        "lag1_dependence": fit.lag1_dependence,
    }

    print("[5/6] in-control run (false-alarm check)")
    in_control = monitor.run_monitor(fit, coords[args.phase_one :], sizes[args.phase_one :])
    ic_metrics = monitor.detection_metrics(in_control["alarm"], None, warmup=20)
    print(f"  false alarms per 1000 lots: {ic_metrics['far_per_1000_lots']:.2f}")
    results["in_control"] = ic_metrics

    print("[6/6] degradation experiments")
    change_point = args.phase_one + 200
    experiments = {}
    for mode in args.degradation:
        degraded_maps = drift.apply_degradation(
            maps, lots, sequence, mode, change_point,
            ramp_lots=args.ramp_lots, seed=args.seed,
        )
        x_deg = feat.extract_batch(degraded_maps, progress_every=0)
        proba_deg = model.predict_proba(x_deg)
        deg_stream = composition.build_stream(
            lots, proba_deg, sequence, mode=args.composition_mode
        )
        deg_coords = composition.ilr_matrix(deg_stream)
        deg_props = composition.proportion_matrix(deg_stream)
        deg_sizes = deg_stream["lot_size"].to_numpy()

        offset = args.phase_one
        run = monitor.run_monitor(fit, deg_coords[offset:], deg_sizes[offset:])
        run_naive = monitor.run_monitor(
            fit_naive, deg_coords[offset:], deg_sizes[offset:]
        )
        cp = change_point - offset

        confidence = baselines.max_confidence_stream(proba_deg, lots, sequence)[offset:]
        methods = {
            "ilr_mewma_rate_invariant": run["alarm"],
            "ilr_mewma_no_rate_correction": run_naive["alarm"],
            "proportion_mewma": baselines.proportion_mewma(
                deg_props[offset:], deg_sizes[offset:],
                lam=args.lam, phase_one=150, target_far=args.target_far,
                seed=args.seed,
            ),
            "page_hinkley_confidence": baselines.page_hinkley(confidence),
            "adwin_confidence": baselines.adwin(confidence),
            "ks_confidence": baselines.ks_two_sample(confidence),
            "mmd_ilr": baselines.mmd_detector(deg_coords[offset:], seed=args.seed),
        }
        experiments[mode] = {
            name: monitor.detection_metrics(alarm, cp, warmup=20)
            for name, alarm in methods.items()
        }
        summary = ", ".join(
            f"{k}={v['detection_delay']}" for k, v in experiments[mode].items()
        )
        print(f"  {mode}: delay (lots) {summary}")

        np.savez_compressed(
            out / f"stream_{mode}.npz",
            statistic=run["statistic"],
            limit=run["limit"],
            coords=deg_coords,
            proportions=deg_props,
            sizes=deg_sizes,
            change_point=cp,
        )

    results["experiments"] = experiments
    results["runtime_seconds"] = time.time() - t0

    with open(out / "metrics.json", "w") as fh:
        json.dump(results, fh, indent=2, default=str)
    print(f"\nwrote {out / 'metrics.json'} in {results['runtime_seconds']:.0f}s")


if __name__ == "__main__":
    main()
