#!/usr/bin/env python3
"""Does EM converge, and to the same place, in the regime the paper operates in?

The manuscript says the estimate "stabilises within a few hundred iterations"
without evidence, and the concern is specific: at the low trace ratio of the
WM-811K cells the between-unit component is weakly identified, so EM might
settle on a flat or degenerate solution that happens to reproduce the target
false-alarm rate.

The regime is reproduced here with known truth - the dimension and trace ratio
of the 24-wafer cell - and EM is run from several starting points. Reported: the
log-likelihood trace, the iteration at which successive changes fall below
tolerance, the spread of the final estimate across starts, and the error against
the truth. Convergence to a common point from dispersed starts is evidence
against a local-optimum problem; a flat likelihood would show as a large spread
with small likelihood differences.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from wmmon.composition import ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

RESULTS = Path(f"{WMMON_HOME}/results")
T, DIM, N_UNIT = 700, 8, 24
TARGET_RATIO = 0.26          # the measured trace ratio of the 24-wafer cell


def log_likelihood(z, mu, between, sampling):
    total = 0.0
    for t in range(len(z)):
        cov = sampling[t] + between
        sign, logdet = np.linalg.slogdet(cov)
        d = z[t] - mu
        total += -0.5 * (logdet + d @ np.linalg.solve(cov, d))
    return float(total)


def em(z, mu, sampling, start, iterations=600, tol=1e-6):
    """EM for the between-unit covariance, recording the likelihood each step."""
    between = start.copy()
    trace = []
    stop = None
    for i in range(iterations):
        acc = np.zeros_like(between)
        for t in range(len(z)):
            cov = sampling[t] + between
            inv = np.linalg.inv(cov)
            d = z[t] - mu
            post_mean = between @ inv @ d
            post_cov = between - between @ inv @ between
            acc += np.outer(post_mean, post_mean) + post_cov
        between = acc / len(z)
        between = (between + between.T) / 2
        ll = log_likelihood(z, mu, between, sampling)
        trace.append(ll)
        if stop is None and i > 1 and abs(trace[-1] - trace[-2]) < tol * abs(trace[-2]):
            stop = i + 1
    return between, trace, stop


def main() -> int:
    rng = np.random.default_rng(0)
    p = np.full(DIM + 1, 0.5 / DIM)
    p[0] = 0.5
    V = ilr_basis(DIM + 1)
    S = sampling_covariance(p, float(N_UNIT), V)
    B = np.diag(np.linspace(1.0, 0.25, DIM))
    B *= TARGET_RATIO * np.trace(S) / np.trace(B)
    ratio = np.trace(B) / np.trace(S)
    sampling = np.array([S] * T)
    z = (rng.multivariate_normal(np.zeros(DIM), B, size=T)
         + rng.multivariate_normal(np.zeros(DIM), S, size=T))
    mu = z.mean(axis=0)

    print(f"dimension {DIM}, {N_UNIT} items per unit, {T} units, "
          f"trace ratio {ratio:.3f}\n")
    starts = {"scaled identity, small": 0.01 * np.trace(S) / DIM * np.eye(DIM),
              "scaled identity, large": 2.0 * np.trace(S) / DIM * np.eye(DIM),
              "sample covariance minus sampling":
                  np.cov(z, rowvar=False) - S,
              "random positive definite": None}
    rows = []
    finals = []
    print(f"{'start':>34}{'iter to tol':>13}{'final ll':>14}{'trace err':>11}")
    for name, start in starts.items():
        if start is None:
            A = rng.standard_normal((DIM, DIM))
            start = A @ A.T * np.trace(S) / DIM / DIM
        start = (start + start.T) / 2
        w, Vv = np.linalg.eigh(start)
        start = Vv @ np.diag(np.clip(w, 1e-9, None)) @ Vv.T
        est, trace, stop = em(z, mu, sampling, start)
        err = abs(np.trace(est) - np.trace(B)) / np.trace(B)
        finals.append(est)
        rows.append({"start": name, "iterations_to_tol": stop,
                     "final_loglik": trace[-1], "trace_error": float(err),
                     "loglik_first": trace[0], "loglik_last": trace[-1],
                     "delta_last_10": float(trace[-1] - trace[-11])})
        print(f"{name:>34}{str(stop):>13}{trace[-1]:>14.2f}{err:>11.3f}")
    spread = max(abs(np.trace(a) - np.trace(b))
                 for a in finals for b in finals) / np.trace(B)
    ll_spread = max(r["final_loglik"] for r in rows) - min(r["final_loglik"] for r in rows)
    print(f"\nspread of final trace across starts: {spread:.4f} of the truth")
    print(f"spread of final log-likelihood across starts: {ll_spread:.4f}")
    print(f"log-likelihood change over the last ten iterations: "
          f"{rows[0]['delta_last_10']:.2e}")
    json.dump({"dim": DIM, "n_unit": N_UNIT, "T": T, "trace_ratio": float(ratio),
               "trace_spread_across_starts": float(spread),
               "loglik_spread_across_starts": float(ll_spread), "rows": rows},
              open(RESULTS / "em_convergence.json", "w"), indent=2)
    print("\nwrote results/em_convergence.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
