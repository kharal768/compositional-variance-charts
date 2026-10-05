#!/usr/bin/env python3
"""A Monte Carlo within-unit term, and power with shifts scaled in the true covariance.

Two corrections to earlier experiments, and one remedy.

1. Shift scaling. Shifts were previously scaled to a Mahalanobis length in the closed-form total
   covariance, which overstates rare-balance variance, so a nominally equal-difficulty shift was
   larger in true standard-deviation units toward the rarest balance. Here shifts are scaled in the
   true total covariance, estimated from a large in-control simulation of the same design, and each is
   also reported in natural units: the change in the largest class's proportion and the relative
   change in the rarest class's.

2. The remedy. The within-unit covariance is obtained by simulating multinomial counts at the Phase
   I composition and the unit's size, with the same zero replacement, instead of from the asymptotic
   closed form. It is still a known function of composition and size, needing no replication, and it
   removes the closed form's overstatement.

3. The plain chart is given the textbook F-based Phase II limit as well as an empirical one.

Charts: variance components with the closed form, variance components with the Monte Carlo term, and
plain T^2. Limits: nominal (the chart's own), matched (pooled over in-control streams) and held-out.
"""
from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import varcomp as VC
from wmmon.composition import (bayesian_multiplicative_replacement, ilr, ilr_basis,
                               ilr_inverse)
from wmmon.laney_coda import sampling_covariance as CLOSED
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
CACHE = Path("/tmp/mc_within_term_cache")
STREAMS, T, NOMINAL = 16, 1000, 5.0
FIT, PHASE_ONE, WARMUP = 240, 400, 20
M_LEN = 2.5
MC_DRAWS = 6000
P = np.array([0.808, 0.053, 0.028, 0.024, 0.020, 0.017, 0.017, 0.017, 0.017])
P = P / P.sum()
V = ilr_basis(len(P)); DIM = len(P) - 1


def make_mc(basis):
    cache = {}

    def mc(ref, n, b):
        key = (int(round(n)), np.asarray(ref).tobytes())
        if key not in cache:
            rng = np.random.default_rng(1000 + int(round(n)))
            counts = rng.multinomial(int(round(n)), np.asarray(ref) / np.sum(ref),
                                     size=MC_DRAWS).astype(float)
            z = ilr(bayesian_multiplicative_replacement(counts), b)
            cache[key] = np.cov(z, rowvar=False)
        return cache[key]
    return mc


def one_stream(seed, sizes_fn, B):
    rng = np.random.default_rng(seed)
    sizes = sizes_fn(rng)
    b = rng.multivariate_normal(np.zeros(DIM), B, size=T)
    counts = np.array([rng.multinomial(int(n), P) for n in sizes], dtype=float)
    props = bayesian_multiplicative_replacement(counts)
    return ilr(props) + b, props, sizes.astype(float)


def true_total(sizes_fn, B, n_sim=60000):
    """Pooled in-control covariance of the coordinates for a design, by simulation."""
    rng = np.random.default_rng(99)
    sizes = np.asarray(sizes_fn(rng, n_sim) if sizes_fn.__code__.co_argcount == 2 else sizes_fn(rng))
    zs = []
    for n in np.unique(sizes):
        k = int((sizes == n).sum())
        counts = rng.multinomial(int(n), P, size=k).astype(float)
        zs.append(ilr(bayesian_multiplicative_replacement(counts)) +
                  rng.multivariate_normal(np.zeros(DIM), B, size=k))
    return np.cov(np.vstack(zs), rowvar=False)


def setup():
    base = CLOSED(P, 25.0, V)
    B = np.diag(np.linspace(1.0, 0.3, DIM)); B *= 0.30 * np.trace(base) / np.trace(B)
    designs = {"fixed, 25 items": (lambda rng, n=T: np.full(n, 25)),
               "widely varied, 4-40": (lambda rng, n=T: rng.integers(4, 41, size=n))}
    rng0 = np.random.default_rng(0)
    raw = {"leading coordinate": np.eye(DIM)[0], "rarest balance": np.eye(DIM)[-1],
           "equal-weight diagonal": np.ones(DIM) / np.sqrt(DIM),
           "random direction": rng0.standard_normal(DIM)}
    return B, designs, raw


def shifts_for(sizes_fn, B, raw):
    inv = np.linalg.inv(true_total(sizes_fn, B))
    out = {}
    for name, u in raw.items():
        u = u / np.linalg.norm(u)
        out[name] = M_LEN * u / np.sqrt(u @ inv @ u)
    return out


def compute(design_index):
    """Simulate and cache the statistics of every chart for each stream of one design."""
    B, designs, raw = setup()
    design = list(designs)[design_index]
    sizes_fn = designs[design]
    shifts = shifts_for(sizes_fn, B, raw)
    CACHE.mkdir(exist_ok=True)
    for s in range(STREAMS):
        f = CACHE / f"d{design_index}_s{s}.pkl"
        if f.exists():
            continue
        coords, props, sizes = one_stream(6000 + s, sizes_fn, B)
        rec = {}
        for variant in ("closed", "mc"):
            VC.sampling_covariance = make_mc(V) if variant == "mc" else CLOSED
            try:
                fit = VC.calibrate(coords[:FIT], props[:FIT], sizes[:FIT],
                                   target_far=NOMINAL / 1000.0, estimator="em",
                                   seed=0, n_bootstrap=6000)
                def st(c, sl):
                    return np.asarray(VC.run(fit, c, props[sl], sizes[sl])["statistic"])
                ph2 = coords[PHASE_ONE:]; sl2 = slice(PHASE_ONE, None)
                rec[variant] = {"limit": float(fit.limit),
                                "hold": st(coords[FIT:PHASE_ONE], slice(FIT, PHASE_ONE)),
                                "base": st(ph2, sl2),
                                "shift": {k: st(ph2 + v, sl2) for k, v in shifts.items()}}
            finally:
                VC.sampling_covariance = CLOSED
        mu = coords[:FIT].mean(axis=0)
        invs = np.linalg.pinv(np.cov(coords[:FIT], rowvar=False))
        def pl(c):
            d = c - mu
            return np.einsum("ij,jk,ik->i", d, invs, d)
        ph2 = coords[PHASE_ONE:]
        f_lim = (DIM * (FIT + 1) * (FIT - 1) / (FIT * (FIT - DIM))) * stats.f.ppf(
            1 - NOMINAL / 1000, DIM, FIT - DIM)
        rec["plain"] = {"limit": float(f_lim), "hold": pl(coords[FIT:PHASE_ONE]),
                        "base": pl(ph2), "shift": {k: pl(ph2 + v) for k, v in shifts.items()}}
        pickle.dump(rec, open(f, "wb"))
        print(f"design {design_index} stream {s + 1}/{STREAMS}", flush=True)


def aggregate():
    B, designs, raw = setup()
    out = {"nominal_per_1000": NOMINAL, "streams": STREAMS, "m": M_LEN, "rows": [], "natural_units": []}
    for di, (design, sizes_fn) in enumerate(designs.items()):
        for name, v in shifts_for(sizes_fn, B, raw).items():
            ps = ilr_inverse((ilr(P[None, :], V) + v[None, :]), V)[0]
            out["natural_units"].append({
                "design": design, "direction": name,
                "largest_class_change": float(ps[0] - P[0]),
                "rarest_class_relative_change": float(ps[-1] / P[-1] - 1.0),
                "total_variation": float(0.5 * np.abs(ps - P).sum())})
        per = [pickle.load(open(CACHE / f"d{di}_s{s}.pkl", "rb")) for s in range(STREAMS)]
        for chart, label in (("closed", "variance components, closed form"),
                             ("mc", "variance components, Monte Carlo term"),
                             ("plain", "plain T2 (F limit)")):
            base_all = np.concatenate([r[chart]["base"][WARMUP:] for r in per])
            lim_matched = float(np.quantile(base_all, 1 - NOMINAL / 1000))
            nominal_rate = float(np.mean(np.concatenate(
                [r[chart]["base"][WARMUP:] > r[chart]["limit"] for r in per]))) * 1000
            ho_lims = [float(np.quantile(r[chart]["hold"], 1 - NOMINAL / 1000)) for r in per]
            ho_rate = float(np.mean(np.concatenate(
                [r[chart]["base"][WARMUP:] > l for r, l in zip(per, ho_lims)]))) * 1000
            silent = float(np.mean([(r[chart]["base"][WARMUP:] > r[chart]["limit"]).sum() == 0 for r in per]))
            for name in raw:
                pm = float(np.mean(np.concatenate([r[chart]["shift"][name][WARMUP:] > lim_matched for r in per])))
                pn = float(np.mean(np.concatenate([r[chart]["shift"][name][WARMUP:] > r[chart]["limit"] for r in per])))
                ph = float(np.mean(np.concatenate([r[chart]["shift"][name][WARMUP:] > l for r, l in zip(per, ho_lims)])))
                out["rows"].append({"design": design, "chart": label, "direction": name,
                                    "power_matched": pm, "power_nominal": pn, "power_heldout": ph,
                                    "nominal_limit_rate": nominal_rate, "heldout_limit_rate": ho_rate,
                                    "share_streams_silent": silent})
    json.dump(out, open(RESULTS / "mc_within_term.json", "w"), indent=2)
    print("aggregated", len(out["rows"]), "rows")


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "aggregate"
    if mode == "compute":
        compute(int(sys.argv[2]))
    else:
        aggregate()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
