#!/usr/bin/env python3
"""A dispersion model, rather than a dispersion constant.

Sections 6.9 to 6.9.2 reduced WM-811K's in-control rate from 683 to 149.5 per
1,000 against a nominal 5, and located the residual: a single sigma_Z assumes
every monitoring unit is over-dispersed to the same degree, while WM-811K units
are each dominated by one or two production lots that differ in product, die
size and map geometry.

This replaces the constant with a model. The whitened residual magnitude of a
unit is regressed on a unit covariate, and each unit is rescaled by its own
fitted dispersion before the residual matrix is estimated. If die size is the
right covariate, conditioning on it should absorb part of the heterogeneity and
the realised rate should fall further. If it does not, die size is the wrong
covariate and that is worth knowing before any random-effects model is built
around it.

The covariate is deliberately chosen in advance and not searched over: with
811 units and a handful of candidate covariates, picking the best-performing
one after the fact would manufacture an effect. Die size is the choice, on the
grounds that die size varied by a factor of three across
windows of the stream while lot size did not.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
import time
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import composition, data, laney_coda as L

CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0
UNIT = 24


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo * 1000, hi * 1000


def fit_dispersion_model(w, covariate, n_bins=6):
    """Per-unit dispersion factor from a binned regression of |w|^2 on x.

    Binned rather than parametric: the relationship between die size and
    dispersion has no reason to be linear, and a smooth with few degrees of
    freedom is less likely to absorb signal than a flexible one. Returns the
    bin edges and each bin's factor so the same model can be applied to
    Phase II units.
    """
    magnitude = (w ** 2).sum(axis=1) / w.shape[1]
    edges = np.quantile(covariate, np.linspace(0, 1, n_bins + 1))
    edges[0] -= 1e-9
    edges[-1] += 1e-9
    idx = np.clip(np.digitize(covariate, edges[1:-1]), 0, n_bins - 1)
    factors = np.ones(n_bins)
    for b in range(n_bins):
        sel = idx == b
        if sel.sum() >= 5:
            factors[b] = float(np.mean(magnitude[sel]))
    factors = np.clip(factors, 1e-6, None)
    factors /= float(np.exp(np.mean(np.log(factors))))  # geometric mean 1
    return edges, factors


def apply_dispersion_model(covariate, edges, factors):
    idx = np.clip(np.digitize(covariate, edges[1:-1]), 0, len(factors) - 1)
    return factors[idx]


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
    die = stream["die_size"].to_numpy()[rows_idx]
    proba = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]

    n_units = len(proba) // UNIT
    proba = proba[: n_units * UNIT]
    die = die[: n_units * UNIT]
    ids = np.array([f"u{i // UNIT:06d}" for i in range(len(proba))], dtype=object)
    seq = [f"u{i:06d}" for i in range(n_units)]
    frame = composition.build_stream(ids, proba, seq, mode="soft")
    coords = composition.ilr_matrix(frame)
    props = composition.proportion_matrix(frame)
    sizes = frame["lot_size"].to_numpy()
    unit_die = np.array([np.nanmedian(die[i * UNIT:(i + 1) * UNIT])
                         for i in range(n_units)])
    print(f"{n_units} units | die size per unit: median {np.nanmedian(unit_die):.0f}, "
          f"IQR {np.nanpercentile(unit_die, 25):.0f}-{np.nanpercentile(unit_die, 75):.0f}",
          flush=True)

    phase_one = max(160, int(0.4 * n_units))
    warmup = 20
    exposure = n_units - phase_one - warmup
    results = []

    for lag in (1, 4):
        # --- constant dispersion ---
        fit = L.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                          target_far=NOMINAL / 1000.0, lag=lag, seed=seed)
        base = L.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:])
        k = int(base["alarm"][warmup:].sum())
        lo, hi = clopper_pearson(k, exposure)
        results.append({"model": "constant", "lag": lag, "rate": 1000.0 * k / exposure,
                        "ci": [lo, hi], "covers": bool(lo <= NOMINAL <= hi),
                        "sigma_Z": fit.inflation})
        print(f"  constant     lag {lag}: rate {1000.0 * k / exposure:7.1f} "
              f"CI [{lo:.1f}, {hi:.1f}]", flush=True)

        # --- dispersion conditioned on die size ---
        w_train = L.whiten(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                           fit.mean, fit.basis, fit.reference)
        edges, factors = fit_dispersion_model(w_train, unit_die[:phase_one])
        scale_train = np.sqrt(apply_dispersion_model(unit_die[:phase_one],
                                                     edges, factors))
        w_scaled = w_train / scale_train[:, None]
        sigma_w = L.dispersion_matrix(w_scaled, lag=lag)
        inv = np.linalg.pinv(sigma_w)

        rng = np.random.default_rng(seed)
        root = L._sqrt(sigma_w)
        draws = rng.standard_normal((4000, sigma_w.shape[0])) @ root.T
        limit = float(np.quantile(np.einsum("ij,jk,ik->i", draws, inv, draws),
                                  1.0 - NOMINAL / 1000.0))

        w_test = L.whiten(coords[phase_one:], props[phase_one:], sizes[phase_one:],
                          fit.mean, fit.basis, fit.reference)
        scale_test = np.sqrt(apply_dispersion_model(unit_die[phase_one:],
                                                    edges, factors))
        w_test = w_test / scale_test[:, None]
        statistic = np.einsum("ij,jk,ik->i", w_test, inv, w_test)
        k2 = int((statistic[warmup:] > limit).sum())
        lo2, hi2 = clopper_pearson(k2, exposure)
        spread = float(factors.max() / factors.min())
        results.append({"model": "die-size-conditioned", "lag": lag,
                        "rate": 1000.0 * k2 / exposure, "ci": [lo2, hi2],
                        "covers": bool(lo2 <= NOMINAL <= hi2),
                        "factor_spread": spread,
                        "factors": [float(f) for f in factors]})
        print(f"  conditioned  lag {lag}: rate {1000.0 * k2 / exposure:7.1f} "
              f"CI [{lo2:.1f}, {hi2:.1f}] | dispersion factors span "
              f"{spread:.2f}x across die-size bins", flush=True)

    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "unit": UNIT, "n_units": n_units,
               "rows": results}, open(RESULTS / "dispersion_model.json", "w"),
              indent=2)
    print(f"\nwrote results/dispersion_model.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
