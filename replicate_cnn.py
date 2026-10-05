#!/usr/bin/env python3
"""Replication with the CNN inspection module in place of the tree ensemble.

Same protocol as replicate.py --- matched false-alarm calibration on a held-out
clean segment, replication across change points --- but the monitored model is
now the CNN. The monitoring layer is unchanged: it consumes class probabilities
and never sees an image, a feature vector, or a label.

MD3-RS keeps its own random-subspace ensemble over the handcrafted features,
because that is how the method is defined --- a feature-bagged ensemble whose
disagreement gives blindspot density. Swapping it to the CNN would not be MD3.
MD3-conf reads the margin off whichever model is deployed, so it follows the
CNN here.

Per-mode, per-change-point *probabilities* are cached rather than images: nine
floats per wafer instead of two 32x32 planes, which is what makes repeated runs
cheap.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

import matched_comparison as mc
from wmmon import baselines, cnn, composition, data, drift, md3, monitor

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)
DEFAULT_CHANGE_POINTS = [450, 500, 550, 600, 650, 700, 750, 800, 850, 900, 950, 1000]
TARGETS = [1.0, 5.0, 20.0]
PHASE_ONE = 300


def load_cnn() -> cnn.CNNInspectionModel:
    model = cnn.WaferCNN(len(CLASSES))
    state = torch.load(CACHE / "cnn_ckpt.pt", weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    return cnn.CNNInspectionModel(model=model, classes=CLASSES)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--confine-to-defective", action="store_true")
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--ramp-lots", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--change-points", type=int, nargs="*", default=DEFAULT_CHANGE_POINTS)
    ap.add_argument("--permute", action="store_true",
                    help="evaluate on a shuffled lot order. Every lot keeps its "
                         "composition, but the product-mix drift that runs through "
                         "the natural sequence is destroyed, so the pre-change "
                         "stream is in control by construction and a false alarm "
                         "is genuinely a false alarm.")
    ap.add_argument("--budget-seconds", type=int, default=240)
    args = ap.parse_args()
    t0 = time.time()

    df, _ = data.load_lswmd(f"{WMMON_HOME}/data/subset.pkl")
    train, _, stream = data.split_labelled(df, seed=args.seed)
    stream = stream.reset_index(drop=True)

    model = load_cnn()
    x_feat_train = np.load(CACHE / "train.npz")["x"]
    rs = md3.train_random_subspace(
        x_feat_train, train["label"].tolist(), CLASSES, seed=args.seed
    )

    sequence = (data.permuted_lot_order(stream, seed=args.seed) if args.permute
                else data.lot_order(stream))[: args.stream_lots]
    ptag = "_perm" if args.permute else ""
    keep = set(sequence)
    lots_all = stream["lot"].to_numpy(dtype=object)
    rows = np.flatnonzero(np.array([str(l) in keep for l in lots_all]))
    lots = lots_all[rows]
    position = np.array([{l: i for i, l in enumerate(sequence)}[str(l)] for l in lots])
    maps = [np.asarray(m) for m in stream["wafer_map"].to_numpy()[rows]]

    # handcrafted features for MD3-RS, assembled from the existing blocks
    blocks = []
    for path in sorted(CACHE.glob("stream_part_*.npz")):
        with np.load(path) as z:
            blocks.append((int(z["lo"]), z["x"]))
    blocks.sort(key=lambda t: t[0])
    x_feat = np.vstack([b for _, b in blocks])[rows]

    clean_path = CACHE / f"cnnproba_clean_{args.stream_lots}{ptag}.npz"
    if clean_path.exists():
        proba_clean = np.load(clean_path)["p"]
    else:
        proba_clean = model.predict_proba(cnn.encode_batch(maps))
        np.savez_compressed(clean_path, p=proba_clean)
    print(f"clean probabilities {proba_clean.shape} ({time.time() - t0:.0f}s)", flush=True)

    off, warmup = PHASE_ONE, 20
    frame_clean = composition.build_stream(lots, proba_clean, sequence, mode="soft")
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
        scorers(coords_clean, props_clean, conf_clean, proba_clean, x_feat).items()
    }

    confine = "_conf" if args.confine_to_defective else ""
    out = {"mode": args.mode, "model": "cnn", "permuted": bool(args.permute),
           "confined": bool(args.confine_to_defective),
           "change_points": [], "runs": {}}

    for cp_lot in args.change_points:
        if time.time() - t0 > args.budget_seconds:
            print(f"budget reached after {len(out['change_points'])} change points",
                  flush=True)
            break
        post = position >= cp_lot
        # The permutation changes which lots fall after the change point, so it
        # must appear in the cache key or a permuted run silently reuses the
        # natural-order probabilities.
        ppath = CACHE / (f"cnnproba_{args.mode}{confine}{ptag}_s{args.severity}"
                         f"_r{args.ramp_lots}_cp{cp_lot}_seed{args.seed}.npz")
        if ppath.exists():
            proba_post = np.load(ppath)["p"]
        else:
            maps_post = [maps[i] for i in np.flatnonzero(post)]
            degraded = drift.apply_degradation(
                maps_post, lots[post], sequence, args.mode, change_point=cp_lot,
                max_severity=args.severity, ramp_lots=args.ramp_lots, seed=args.seed)
            if args.confine_to_defective:
                normal = proba_clean[post].argmax(axis=1) == CLASSES.index("none")
                for i in np.flatnonzero(normal):
                    degraded[i] = maps_post[i]
            proba_post = model.predict_proba(cnn.encode_batch(degraded))
            np.savez_compressed(ppath, p=proba_post)

        proba_deg = proba_clean.copy()
        proba_deg[post] = proba_post
        # MD3-RS runs on handcrafted features, so it needs the degraded ones;
        # where those are unavailable it is reported as not evaluated rather
        # than silently scored on clean features.
        feat_path = CACHE / (f"deg_{args.mode}{confine}{ptag}_s{args.severity}"
                             f"_r{args.ramp_lots}_cp{cp_lot}_seed{args.seed}.npz")
        feats_deg = x_feat.copy()
        rs_available = feat_path.exists()
        if rs_available:
            feats_deg[post] = np.load(feat_path)["x"]

        frame_deg = composition.build_stream(lots, proba_deg, sequence, mode="soft")
        coords_deg = composition.ilr_matrix(frame_deg)
        props_deg = composition.proportion_matrix(frame_deg)
        conf_deg = baselines.max_confidence_stream(proba_deg, lots, sequence)[off:]
        cp = cp_lot - off

        print(f"\nchange point {cp_lot} [{time.time() - t0:.0f}s]"
              f"{'' if rs_available else '  (md3_rs skipped: no degraded features)'}",
              flush=True)
        for name, fn in scorers(coords_deg, props_deg, conf_deg,
                                proba_deg, feats_deg).items():
            if name == "md3_rs" and not rs_available:
                continue
            deg, clean = fn(), clean_scores[name]
            parts = []
            for target in TARGETS:
                thr = float(np.quantile(clean[cp:], 1.0 - target / 1000.0,
                                        method="higher"))
                hits = np.flatnonzero(deg[cp:] > thr)
                delay = float(hits[0]) if len(hits) else float("inf")
                fa = 1000.0 * int((deg[warmup:cp] > thr).sum()) / max(cp - warmup, 1)
                out["runs"].setdefault(name, {}).setdefault(str(target), []).append(
                    {"change_point": cp_lot, "delay": delay, "fa_per_1000": fa})
                parts.append(f"@{target:g}: " + ("never" if not np.isfinite(delay)
                                                 else f"{delay:.0f}"))
            print(f"  {name:<20s} " + "  ".join(parts), flush=True)
        out["change_points"].append(cp_lot)

    print(f"\nmedian delay (range) over {len(out['change_points'])} change points")
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
                "median": float(np.median(finite)) if finite else None,
                "min": float(min(finite)) if finite else None,
                "max": float(max(finite)) if finite else None}
            cells.append(f"{len(finite)}/{len(delays)} med {np.median(finite):.0f}"
                         if finite else f"0/{len(delays)} never")
        print(f"{name:<20s}" + "".join(f"{c:>22s}" for c in cells))
    out["summary"] = summary

    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"cnn_replicate_{args.mode}{confine}{ptag}.json"
    if path.exists():
        prev = json.load(open(path))
        if len(prev.get("change_points", [])) > len(out["change_points"]):
            print(f"\nNOT overwriting {path}: it holds more change points.")
            return
    json.dump(out, open(path, "w"), indent=2)
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
