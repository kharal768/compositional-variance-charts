#!/usr/bin/env python3
"""Compare detectors at a matched false-alarm rate.

The first comparison was not a comparison.  Three baselines showed a detection
delay of zero while firing 3 to 12 times the nominal false-alarm rate on
undegraded lots --- a detector that alarms constantly detects everything
instantly, and putting its delay in the same table as a calibrated method
misrepresents both.

Here every detector is reduced to a score where larger means more evidence of
change.  One threshold per detector is chosen on the *undegraded* stream so that
all of them realise the same false-alarm episode budget.  Only then is the delay
on degraded streams comparable.

Caveat to carry into the paper: the threshold is chosen on the same undegraded
stream the false-alarm rate is later quoted from, which flatters every method
equally.  A held-out calibration split would be stricter and is the right thing
to do once the experiment is final.

Run one mode at a time:

    python matched_comparison.py --mode new_signature
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import argparse
import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import baselines, classifier, composition, data, monitor

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
CLASSES = list(data.CLASSES)

#: False-alarm episodes per 1,000 lots that every detector must respect.
FA_BUDGET_PER_1000 = 5.0


# --------------------------------------------------------------------------
# Score functions: higher = more evidence of change. One threshold each.
# --------------------------------------------------------------------------

def score_page_hinkley(values: np.ndarray, delta: float = 0.005,
                       burn_in: int = 30) -> np.ndarray:
    """Page-Hinkley test statistic, before thresholding."""
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values))
    if len(values) <= burn_in:
        return out
    baseline = values[:burn_in].mean()
    cumulative = minimum = 0.0
    for t in range(burn_in, len(values)):
        cumulative += values[t] - baseline - delta
        minimum = min(minimum, cumulative)
        out[t] = cumulative - minimum
    return out


def score_ks(values: np.ndarray, reference_size: int = 100,
             window: int = 30) -> np.ndarray:
    """-log10 p-value of a sliding-window KS test against a fixed reference."""
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values))
    if len(values) <= reference_size + window:
        return out
    reference = values[:reference_size]
    for t in range(reference_size + window, len(values)):
        p = stats.ks_2samp(reference, values[t - window : t]).pvalue
        out[t] = -np.log10(max(p, 1e-12))
    return out


def score_mmd(features: np.ndarray, reference_size: int = 100, window: int = 30,
              n_permutations: int = 120, stride: int = 5,
              seed: int = 0) -> np.ndarray:
    """-log10 permutation p-value of an RBF MMD two-sample test.

    Evaluated every ``stride`` lots and held between evaluations, so this
    detector's delay carries up to ``stride - 1`` lots of granularity penalty.
    That is a property of the baseline, not of the comparison, and is reported.
    """
    features = np.asarray(features, dtype=float)
    out = np.zeros(len(features))
    if len(features) <= reference_size + window:
        return out

    reference = features[:reference_size]
    pooled = np.vstack([reference, features[reference_size : reference_size + window]])
    distances = ((pooled[:, None, :] - pooled[None, :, :]) ** 2).sum(-1)
    gamma = 1.0 / max(np.median(distances[distances > 0]), 1e-9)

    rng = np.random.default_rng(seed)
    current_score = 0.0
    for t in range(reference_size + window, len(features)):
        if (t - reference_size - window) % stride == 0:
            current = features[t - window : t]
            observed = baselines._rbf_mmd2(reference, current, gamma)
            combined = np.vstack([reference, current])
            null = np.empty(n_permutations)
            for b in range(n_permutations):
                perm = rng.permutation(len(combined))
                null[b] = baselines._rbf_mmd2(
                    combined[perm[: len(reference)]],
                    combined[perm[len(reference) :]],
                    gamma,
                )
            p = (null >= observed).mean()
            current_score = -np.log10(max(p, 1.0 / n_permutations))
        out[t] = current_score
    return out


def score_adwin(values: np.ndarray, confidence: float = 0.05,
                min_window: int = 10, max_window: int = 400) -> np.ndarray:
    """Largest normalised mean gap over all cut points in the adaptive window.

    ADWIN's own parameter is a confidence level, which makes a clean threshold
    sweep awkward; this exposes the underlying evidence directly so it can be
    calibrated like the others.
    """
    values = np.asarray(values, dtype=float)
    out = np.zeros(len(values))
    window: list[float] = []
    for t, value in enumerate(values):
        window.append(float(value))
        if len(window) > max_window:
            window = window[-max_window:]
        if len(window) < 2 * min_window:
            continue
        arr = np.asarray(window)
        n = len(arr)
        variance = arr.var(ddof=1) + 1e-12
        best = 0.0
        for split in range(min_window, n - min_window + 1):
            left, right = arr[:split], arr[split:]
            harmonic = 1.0 / (1.0 / len(left) + 1.0 / len(right))
            epsilon = np.sqrt(2.0 * variance * np.log(2.0 / confidence) / harmonic)
            best = max(best, abs(left.mean() - right.mean()) / max(epsilon, 1e-12))
        out[t] = best
        if best > 1.0:
            window = list(arr[len(arr) // 2 :])
    return out


# --------------------------------------------------------------------------

def episodes(alarm: np.ndarray, merge_gap: int = 1) -> int:
    return len(monitor._episodes(alarm, merge_gap))


def choose_threshold(scores: np.ndarray, warmup: int, target_rate: float) -> float:
    """Threshold giving the target *per-lot* false-alarm rate on clean data.

    Budgeting false-alarm *episodes* instead is gameable: a detector whose score sits above the line continuously produces
    one unbroken run, and one run counts as a single episode however many lots
    it covers. Every baseline duly collapsed to a near-zero threshold, alarmed
    on almost every lot, and scored a detection delay of zero --- the same
    artefact this script exists to remove, reintroduced through the calibration
    rule. The per-lot rate cannot be gamed that way. Episodes remain a
    reporting statistic, never the constraint.
    """
    tail = scores[warmup:]
    if not len(tail):
        return np.inf
    return float(np.quantile(tail, 1.0 - target_rate, method="higher"))


def build(model, x, lots, sequence):
    proba = model.predict_proba(x)
    frame = composition.build_stream(lots, proba, sequence, mode="soft")
    return proba, frame


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True)
    ap.add_argument("--phase-one", type=int, default=300)
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--change-point", type=int, default=800)
    ap.add_argument("--ramp-lots", type=int, default=0)
    ap.add_argument("--severity", type=float, default=1.0)
    ap.add_argument("--confine-to-defective", action="store_true")
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
    position_of = {lot: i for i, lot in enumerate(sequence)}
    post = np.array([position_of[str(l)] for l in lots]) >= args.change_point

    x_clean = x_all[rows]
    confine = "_conf" if args.confine_to_defective else ""
    # The change point and seed determine WHICH wafers are degraded and how, so
    # both belong in the cache key. Without them a rerun at a different change
    # point silently reuses the wrong features whenever the array lengths happen
    # to match, and the run looks clean.
    cache_path = CACHE / (
        f"deg_{args.mode}{confine}_s{args.severity}_r{args.ramp_lots}"
        f"_cp{args.change_point}_seed{args.seed}.npz"
    )
    if not cache_path.exists():
        raise FileNotFoundError(
            f"{cache_path.name} missing; run run_degradation.py --mode {args.mode} first"
        )
    x_deg = x_clean.copy()
    x_deg[post] = np.load(cache_path)["x"]

    off, warmup = args.phase_one, 20
    cp = args.change_point - off

    proba_clean, frame_clean = build(model, x_clean, lots, sequence)
    proba_deg, frame_deg = build(model, x_deg, lots, sequence)
    coords_clean = composition.ilr_matrix(frame_clean)
    coords_deg = composition.ilr_matrix(frame_deg)
    props_clean = composition.proportion_matrix(frame_clean)
    props_deg = composition.proportion_matrix(frame_deg)
    sizes = frame_clean["lot_size"].to_numpy()

    fit = monitor.calibrate(coords_clean[:off], sizes[:off], lam=0.2,
                            target_far=0.005, seed=args.seed)

    conf_clean = baselines.max_confidence_stream(proba_clean, lots, sequence)[off:]
    conf_deg = baselines.max_confidence_stream(proba_deg, lots, sequence)[off:]

    print(f"scoring ({time.time() - t0:.0f}s)", flush=True)
    scorers = {
        "ilr_mewma": (
            lambda c, p, s: monitor.run_monitor(fit, c[off:], sizes[off:])["statistic"],
        ),
        "proportion_mewma": (
            lambda c, p, s: _proportion_statistic(p, sizes, off),
        ),
        "page_hinkley_confidence": (lambda c, p, s: score_page_hinkley(s),),
        "adwin_confidence": (lambda c, p, s: score_adwin(s),),
        "ks_confidence": (lambda c, p, s: score_ks(s),),
        "mmd_ilr": (lambda c, p, s: score_mmd(c[off:], seed=args.seed),),
    }

    n_eval = len(coords_clean) - off - warmup
    target_rate = FA_BUDGET_PER_1000 / 1000.0
    print(f"evaluated lots {n_eval} -> target {target_rate:.4f} alarms per lot "
          f"({FA_BUDGET_PER_1000:.1f} per 1,000)\n")

    out = {
        "mode": args.mode,
        "ramp_lots": args.ramp_lots,
        "fa_budget_per_1000": FA_BUDGET_PER_1000,
        "target_rate_per_lot": FA_BUDGET_PER_1000 / 1000.0,
        "n_eval_lots": int(n_eval),
        "methods": {},
    }

    print(f"{'method':<26s} {'thresh':>10s} {'FA lots':>8s} {'/1000':>7s} "
          f"{'FA ep':>6s} {'delay':>7s}")
    for name, (fn,) in scorers.items():
        clean_scores = fn(coords_clean, props_clean, conf_clean)
        threshold = choose_threshold(clean_scores, warmup, target_rate)
        clean_alarm = clean_scores > threshold
        n_fa = episodes(clean_alarm[warmup:])
        fa_lots = int(clean_alarm[warmup:].sum())

        deg_scores = fn(coords_deg, props_deg, conf_deg)
        deg_alarm = deg_scores > threshold
        hits = np.flatnonzero(deg_alarm[cp:])
        delay = float(hits[0]) if len(hits) else float("inf")

        entry = {
            "threshold": threshold,
            "false_alarm_episodes_clean": n_fa,
            "false_alarm_lots_clean": fa_lots,
            "fa_lots_per_1000_clean": 1000.0 * fa_lots / n_eval,
            "fa_episodes_per_1000_clean": 1000.0 * n_fa / n_eval,
            "detection_delay": delay,
        }
        out["methods"][name] = entry
        print(f"{name:<26s} {threshold:>10.3f} {fa_lots:>8d} "
              f"{entry['fa_lots_per_1000_clean']:>7.2f} {n_fa:>6d} "
              f"{'never' if not np.isfinite(delay) else f'{delay:.0f}':>7s}", flush=True)

    RESULTS.mkdir(parents=True, exist_ok=True)
    suffix = ("_conf" if args.confine_to_defective else "") + (f"_ramp{args.ramp_lots}" if args.ramp_lots else "")
    path = RESULTS / f"matched_{args.mode}{suffix}.json"
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path} ({time.time() - t0:.0f}s)")


def _proportion_statistic(props, sizes, phase_one_len):
    """MEWMA statistic on raw proportions, exposed as a score.

    This baseline must receive the same treatment as the proposed monitor or
    the comparison is worthless. Giving it a plain sample covariance and a 200-lot reference window while the ILR chart gets Ledoit-Wolf shrinkage and 300 lots is a handicap that alone accounts for a
    277-versus-2-lot difference in median delay on one mode -- the baseline was
    losing to its own tuning, not to the method. Estimator, reference window
    and smoothing constant are now identical; only the coordinates differ.
    """
    from wmmon.monitor import _mewma_statistics, _shrunk_covariance

    reference = props[:phase_one_len]
    mean = reference.mean(axis=0)
    scale = np.sqrt(sizes / np.median(sizes[:phase_one_len]))
    standardised = (props - mean) * scale[:, None]
    cov = _shrunk_covariance(standardised[:phase_one_len], None)
    cov_inv = np.linalg.pinv(np.atleast_2d(cov))
    # Chart from the first Phase II lot, as the ILR monitor does. Passing this
    # function a pre-sliced array made its reference window lots 300-600 while
    # the ILR chart used 0-300, so the two were estimated on different data.
    return _mewma_statistics(standardised[phase_one_len:], cov_inv, 0.2)


if __name__ == "__main__":
    main()
