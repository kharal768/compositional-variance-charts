#!/usr/bin/env python3
"""Controlled-degradation experiments on the real WM-811K stream.

Run one mode per invocation so each fits inside the foreground time budget:

    python run_degradation.py --mode new_signature
    python run_degradation.py --mode resolution_loss
    ...

Only lots at or after the change point are re-classified: degradation does not
touch earlier lots, so their features are reused from the cache. That cuts each
experiment from a full 47,281-wafer pass to roughly 9,600 wafers.

Results are counted in *episodes*, not raw alarms. The MEWMA statistic persists
above its limit for several lots after one excursion, so raw alarm counts turn a
single event into ten and overstate the false-alarm rate.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np

from wmmon import baselines, classifier, composition, data, drift, monitor
from wmmon import features as feat

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)


def assembled_stream(expected: int) -> np.ndarray:
    blocks = []
    for path in sorted(CACHE.glob("stream_part_*.npz")):
        with np.load(path) as z:
            blocks.append((int(z["lo"]), z["x"]))
    blocks.sort(key=lambda t: t[0])
    x = np.vstack([b for _, b in blocks])
    if len(x) != expected:
        raise ValueError(f"assembled {len(x)}, expected {expected}")
    return x


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=sorted(drift.TRANSFORMS))
    ap.add_argument("--phase-one", type=int, default=300)
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--change-point", type=int, default=800)
    ap.add_argument("--ramp-lots", type=int, default=0)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument(
        "--confine-to-defective", action="store_true",
        help="apply the degradation only to wafers the undegraded classifier "
             "already calls defective, so the normal share is preserved by "
             "construction rather than by a fail-fraction heuristic",
    )
    ap.add_argument("--seed", type=int, default=0)
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

    x_all = assembled_stream(len(stream))
    sequence = data.lot_order(stream)[: args.stream_lots]
    keep = set(sequence)
    lots_all = stream["lot"].to_numpy(dtype=object)
    in_window = np.array([str(l) in keep for l in lots_all])

    rows = np.flatnonzero(in_window)
    lots = lots_all[rows]
    position_of = {lot: i for i, lot in enumerate(sequence)}
    lot_position = np.array([position_of[str(l)] for l in lots])
    post = lot_position >= args.change_point

    print(f"stream: {len(sequence):,} lots | {len(rows):,} wafers "
          f"| {int(post.sum()):,} wafers after the change point", flush=True)

    # --- baseline (undegraded) stream ---------------------------------------
    proba_clean = model.predict_proba(x_all[rows])
    frame_clean = composition.build_stream(lots, proba_clean, sequence, mode="soft")
    coords_clean = composition.ilr_matrix(frame_clean)
    sizes = frame_clean["lot_size"].to_numpy()

    fit = monitor.calibrate(
        coords_clean[: args.phase_one], sizes[: args.phase_one],
        lam=0.2, target_far=0.005, seed=args.seed,
    )
    clean_run = monitor.run_monitor(
        fit, coords_clean[args.phase_one :], sizes[args.phase_one :]
    )
    clean = monitor.episode_metrics(clean_run["alarm"], None, warmup=20)
    print(f"undegraded control: {clean['false_alarm_episodes']:.0f} episodes "
          f"= {clean['episodes_per_1000_lots']:.2f} per 1,000 lots "
          f"(nominal 5.00)", flush=True)

    # --- degraded features (cached per mode) --------------------------------
    confine = "_conf" if args.confine_to_defective else ""
    # The change point and seed determine WHICH wafers are degraded and how, so
    # both belong in the cache key. Without them a rerun at a different change
    # point silently reuses the wrong features whenever the array lengths happen
    # to match, and the run looks clean.
    cache_path = CACHE / (
        f"deg_{args.mode}{confine}_s{args.severity}_r{args.ramp_lots}"
        f"_cp{args.change_point}_seed{args.seed}.npz"
    )
    if cache_path.exists():
        x_post = np.load(cache_path)["x"]
        print(f"using cached degraded features {cache_path.name}", flush=True)
    else:
        maps_post = [np.asarray(m) for m in stream["wafer_map"].to_numpy()[rows[post]]]
        degraded = drift.apply_degradation(
            maps_post, lots[post], sequence, args.mode,
            change_point=args.change_point, max_severity=args.severity,
            ramp_lots=args.ramp_lots, seed=args.seed,
        )
        if args.confine_to_defective:
            # Restore the wafers the clean classifier already called normal.
            # Without this, rearranging sparse background failures on a wafer
            # that scored 'none' flips it to a defect class, and the normal
            # share collapses --- which is a defect-level change, exactly what
            # this mode is supposed to avoid.
            clean_pred = proba_clean[post].argmax(axis=1)
            normal = clean_pred == CLASSES.index("none")
            for i in np.flatnonzero(normal):
                degraded[i] = maps_post[i]
            print(f"confined: {int(normal.sum()):,} of {len(degraded):,} post-change "
                  f"wafers left untouched (clean prediction 'none')", flush=True)
        print(f"degrading {len(degraded):,} wafers ({time.time() - t0:.0f}s)", flush=True)
        x_post = feat.extract_batch(degraded, progress_every=0, n_jobs=1)
        np.savez_compressed(cache_path, x=x_post)
        print(f"cached -> {cache_path.name} ({time.time() - t0:.0f}s)", flush=True)

    x_deg = x_all[rows].copy()
    x_deg[post] = x_post
    proba_deg = model.predict_proba(x_deg)

    frame_deg = composition.build_stream(lots, proba_deg, sequence, mode="soft")
    coords_deg = composition.ilr_matrix(frame_deg)
    props_deg = composition.proportion_matrix(frame_deg)

    off = args.phase_one
    cp = args.change_point - off
    confidence = baselines.max_confidence_stream(proba_deg, lots, sequence)[off:]

    methods = {
        "ilr_mewma": monitor.run_monitor(fit, coords_deg[off:], sizes[off:])["alarm"],
        "proportion_mewma": baselines.proportion_mewma(
            props_deg[off:], sizes[off:], lam=0.2, phase_one=200,
            target_far=0.005, seed=args.seed,
        ),
        "page_hinkley_confidence": baselines.page_hinkley(confidence),
        "adwin_confidence": baselines.adwin(confidence),
        "ks_confidence": baselines.ks_two_sample(confidence),
        "mmd_ilr": baselines.mmd_detector(coords_deg[off:], seed=args.seed),
    }

    out = {
        "mode": args.mode,
        "severity": args.severity,
        "ramp_lots": args.ramp_lots,
        "change_point_lot": args.change_point,
        "undegraded_control": clean,
        "class_shift": {
            CLASSES[j]: {
                "before": float(props_deg[args.phase_one : args.change_point, j].mean()),
                "after": float(props_deg[args.change_point :, j].mean()),
            }
            for j in range(len(CLASSES))
        },
        "methods": {},
    }

    print(f"\n{'method':<26s} {'delay':>7s} {'FA episodes':>12s} {'per 1000':>9s}")
    for name, alarm in methods.items():
        m = monitor.episode_metrics(alarm, cp, warmup=20)
        out["methods"][name] = m
        delay = "never" if not np.isfinite(m["detection_delay"]) else f"{m['detection_delay']:.0f}"
        print(f"{name:<26s} {delay:>7s} {m['false_alarm_episodes']:>12.0f} "
              f"{m['episodes_per_1000_lots']:>9.2f}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = ("_conf" if args.confine_to_defective else "") + (f"_ramp{args.ramp_lots}" if args.ramp_lots else "")
    path = RESULTS / f"degradation_{args.mode}{suffix}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
