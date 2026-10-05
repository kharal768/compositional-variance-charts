#!/usr/bin/env python3
"""Detectability and false-alarm experiments under the Sethi protocol.

    python run_sethi.py --which top      # true drift -- must be detected
    python run_sethi.py --which bottom   # irrelevant change -- must be ignored

Both are reported: a detector that catches the top-25% rotation and also fires
on the bottom-25% rotation has not demonstrated anything, which is the failure
Sethi & Kantardzic document for feature-space detectors.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np

import matched_comparison as mc
from wmmon import baselines, classifier, composition, data, md3, monitor, sethi

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)
CHANGE_POINTS = [450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
TARGETS = [1.0, 5.0, 20.0]
PHASE_ONE = 300


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--which", choices=["top", "bottom"], default="top")
    ap.add_argument("--fraction", type=float, default=0.25)
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--budget-seconds", type=int, default=240)
    args = ap.parse_args()
    t0 = time.time()

    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, test, stream = data.split_labelled(df, seed=args.seed)
    stream = stream.reset_index(drop=True)

    x_train = np.load(CACHE / "train.npz")["x"]
    model = classifier.train_inspection_model(
        x_train, train["label"].tolist(), CLASSES, seed=args.seed
    )
    rs = md3.train_random_subspace(
        x_train, train["label"].tolist(), CLASSES, seed=args.seed
    )

    order = sethi.rank_features(x_train, train["label"].tolist(), seed=args.seed)
    columns = sethi.select_columns(order, args.which, args.fraction)
    print(f"rotating {len(columns)} of {x_train.shape[1]} features "
          f"({args.which} {100 * args.fraction:.0f}% by information gain)", flush=True)

    # Sanity check the premise: the top rotation must cost accuracy and the
    # bottom rotation must not. If that does not hold, the experiment says
    # nothing about the detectors.
    q_clean = classifier.evaluate(model, x_train, train["label"].tolist())
    q_rot = classifier.evaluate(
        model, sethi.rotate(x_train, columns), train["label"].tolist()
    )
    print(f"in-sample macro-F1: clean {q_clean['macro_f1']:.3f} "
          f"-> rotated {q_rot['macro_f1']:.3f}", flush=True)

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

    off = PHASE_ONE
    proba_clean, frame_clean = mc.build(model, x_clean, lots, sequence)
    coords_clean = composition.ilr_matrix(frame_clean)
    props_clean = composition.proportion_matrix(frame_clean)
    sizes = frame_clean["lot_size"].to_numpy()
    fit = monitor.calibrate(coords_clean[:off], sizes[:off], lam=0.2,
                            target_far=0.005, seed=args.seed)
    conf_clean = baselines.max_confidence_stream(proba_clean, lots, sequence)[off:]

    def scorers(coords, props, conf, proba, feats):
        return {
            "ilr_mewma": lambda: monitor.run_monitor(
                fit, coords[off:], sizes[off:])["statistic"],
            "proportion_mewma": lambda: mc._proportion_statistic(props, sizes, off),
            "md3_conf": lambda: md3.md3_score(
                md3.lot_margin_density(md3.margin_indicator(proba), lots, sequence),
                off, lam=0.2),
            "md3_rs": lambda: md3.md3_score(
                md3.lot_margin_density(
                    md3.margin_indicator(rs.probabilities(feats)), lots, sequence),
                off, lam=0.2),
            "adwin_confidence": lambda: mc.score_adwin(conf),
            "ks_confidence": lambda: mc.score_ks(conf),
        }

    clean_scores = {
        k: f() for k, f in
        scorers(coords_clean, props_clean, conf_clean, proba_clean, x_clean).items()
    }

    out = {"protocol": "sethi", "which": args.which, "fraction": args.fraction,
           "n_features_rotated": int(len(columns)),
           "macro_f1_clean": q_clean["macro_f1"], "macro_f1_rotated": q_rot["macro_f1"],
           "change_points": [], "runs": {}}

    for cp_lot in CHANGE_POINTS:
        if time.time() - t0 > args.budget_seconds:
            break
        post = position >= cp_lot
        x_deg = sethi.induce(x_clean, post, columns)
        proba_deg, frame_deg = mc.build(model, x_deg, lots, sequence)
        coords_deg = composition.ilr_matrix(frame_deg)
        props_deg = composition.proportion_matrix(frame_deg)
        conf_deg = baselines.max_confidence_stream(proba_deg, lots, sequence)[off:]
        cp = cp_lot - off

        for name, fn in scorers(coords_deg, props_deg, conf_deg,
                                proba_deg, x_deg).items():
            deg, clean = fn(), clean_scores[name]
            for target in TARGETS:
                thr = float(np.quantile(clean[cp:], 1.0 - target / 1000.0,
                                        method="higher"))
                hits = np.flatnonzero(deg[cp:] > thr)
                delay = float(hits[0]) if len(hits) else float("inf")
                out["runs"].setdefault(name, {}).setdefault(str(target), []).append(
                    {"change_point": cp_lot, "delay": delay})
        out["change_points"].append(cp_lot)
        print(f"  change point {cp_lot} done [{time.time() - t0:.0f}s]", flush=True)

    verdict = "detections (higher is better)" if args.which == "top" \
        else "FALSE detections (LOWER is better)"
    print(f"\n{args.which}-25% rotation over {len(out['change_points'])} "
          f"change points --- {verdict}")
    print(f"{'method':<20s}" + "".join(f"{f'@{t:g}/1000':>22s}" for t in TARGETS))
    summary = {}
    for name, by_target in out["runs"].items():
        cells = []
        summary[name] = {}
        for target in TARGETS:
            delays = [r["delay"] for r in by_target[str(target)]]
            finite = [d for d in delays if np.isfinite(d)]
            summary[name][str(target)] = {
                "n_runs": len(delays), "n_detected": len(finite),
                "median": float(np.median(finite)) if finite else None}
            cells.append(f"{len(finite)}/{len(delays)} med {np.median(finite):.0f}"
                         if finite else f"0/{len(delays)} none")
        print(f"{name:<20s}" + "".join(f"{c:>22s}" for c in cells))
    out["summary"] = summary

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"sethi_{args.which}25.json"
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
