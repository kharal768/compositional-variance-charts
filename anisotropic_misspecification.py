#!/usr/bin/env python3
"""Misspecify the shape of the known term, not only its scale.

The scalar sweep multiplies the sampling covariance by a constant, which cannot
distort its shape. Real approximation error at small counts need not be
isotropic: the delta-method and zero-replacement errors differ by balance. This
applies anisotropic perturbations instead - stretching one log-ratio direction
while compressing another, holding the determinant fixed so the overall scale is
unchanged - and records the realised in-control rate.

Three families are tried:
  leading    stretch the first balance, compress the rest
  trailing   stretch the last balance (the rarest parts) and compress the rest
  random     a random orthogonal rotation of a fixed stretch, averaged over draws

A chart tolerant only of scale error would fail here; one tolerant of shape
error would not.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, varcomp as VC
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL, UNIT = 5.0, 24


def distortion(dim, kind, strength, rng=None):
    """A unit-determinant distortion matrix: shape changes, scale does not."""
    d = np.ones(dim)
    if kind == "leading":
        d[0] = strength
    elif kind == "trailing":
        d[-1] = strength
    elif kind == "random":
        d = rng.uniform(1.0 / strength, strength, size=dim)
    d = d / np.exp(np.mean(np.log(d)))          # determinant 1
    M = np.diag(d)
    if kind == "random":
        Q, _ = np.linalg.qr(rng.standard_normal((dim, dim)))
        M = Q @ M @ Q.T
    return M


def evaluate(coords, props, sizes, phase_one, warmup, M):
    """Calibrate and run with the sampling term distorted by M."""
    import wmmon.varcomp as V
    original = V.sampling_covariance

    def distorted(ref, n, basis):
        S = original(ref, n, basis)
        return M @ S @ M.T

    V.sampling_covariance = distorted
    try:
        fit = V.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                          target_far=NOMINAL / 1000.0, estimator="em", lag=4, seed=0)
        out = V.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:])
    finally:
        V.sampling_covariance = original
    k = int(out["alarm"][warmup:].sum())
    n = len(coords) - phase_one - warmup
    lo = stats.beta.ppf(0.025, k, n - k + 1) * 1000 if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) * 1000 if k < n else 1000.0
    return 1000.0 * k / n, (lo, hi), float(fit.between_fraction)


def main() -> int:
    proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
    n_units = len(proba) // UNIT
    p = proba[: n_units * UNIT]
    ids = np.array([f"u{i // UNIT:06d}" for i in range(len(p))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n_units)]
    fr = composition.build_stream(ids, p, seq, mode="hard")
    coords = composition.ilr_matrix(fr)
    props = composition.proportion_matrix(fr)
    sizes = fr["lot_size"].to_numpy()
    dim = coords.shape[1]
    phase_one, warmup = max(160, int(0.4 * n_units)), 20

    rows = []
    print(f"WM-811K, {UNIT} wafers per unit, nominal {NOMINAL:.1f} per 1,000")
    print("determinant held at 1, so only the shape of the known term changes\n")
    print(f"{'distortion':>12}{'strength':>10}{'rate':>9}{'95% CI':>18}{'covers':>9}{'batch share':>13}")
    rng = np.random.default_rng(0)
    for kind in ("leading", "trailing", "random"):
        for strength in (1.5, 2.0, 4.0):
            M = distortion(dim, kind, strength, rng)
            rate, ci, share = evaluate(coords, props, sizes, phase_one, warmup, M)
            rows.append({"kind": kind, "strength": strength, "rate": rate,
                         "ci": list(ci), "covers": bool(ci[0] <= NOMINAL <= ci[1]),
                         "between_fraction": share})
            print(f"{kind:>12}{strength:>10.1f}{rate:>9.1f}"
                  f"{f'[{ci[0]:.1f}, {ci[1]:.1f}]':>18}"
                  f"{'yes' if ci[0] <= NOMINAL <= ci[1] else 'NO':>9}{100*share:>12.1f}%")
    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "unit": UNIT, "rows": rows},
              open(RESULTS / "anisotropic_misspecification.json", "w"), indent=2)
    print("\nwrote results/anisotropic_misspecification.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
