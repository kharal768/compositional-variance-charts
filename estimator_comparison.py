#!/usr/bin/env python3
"""Successive differences versus a fitted Dirichlet-multinomial.

The empirical Bayes scheme of Shiau, Chen & Feltz (2005) models category
fractions as multinomial with a Dirichlet prior, which is the generative model
behind the over-dispersion this paper measures. A reviewer will reasonably ask
why the dispersion is estimated from successive differences rather than by
fitting that model, and the honest answer requires a comparison rather than an
argument.

Both estimators are compared on simulated streams where the truth is known. For
a Dirichlet-multinomial with n trials and concentration alpha0, the variance of
the counts is inflated over the multinomial by

    c = (n + alpha0) / (1 + alpha0),

so the model-based analogue of Laney's sigma_Z is sqrt(c). alpha0 is estimated
from Phase I counts by method of moments, which is the standard estimator and
does not require an optimiser.

Three questions:
  Q1  Which estimator recovers the true dispersion more accurately?
  Q2  Does either hold the nominal false-alarm rate better?
  Q3  What happens when the generative assumption is violated --- when
      over-dispersion is present but not Dirichlet?

Q3 is the one that matters for the paper. A model-based estimator should win
when its model is right; the question is what it costs when the model is wrong.
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME

import json
from pathlib import Path

import numpy as np

from wmmon import laney_coda as L
from wmmon.composition import bayesian_multiplicative_replacement, ilr

RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0


def dm_concentration(counts: np.ndarray) -> float:
    """Method-of-moments estimate of the Dirichlet-multinomial concentration.

    Uses the mean observed over-dispersion of the category proportions relative
    to the multinomial variance, averaged over categories weighted by their
    mean proportion. Returns a large value when the data are indistinguishable
    from multinomial, which maps to an inflation factor of one.
    """
    counts = np.asarray(counts, dtype=float)
    n = counts.sum(axis=1)
    n_bar = float(n.mean())
    p = counts / n[:, None]
    p_bar = p.mean(axis=0)
    observed = p.var(axis=0, ddof=1)
    expected = p_bar * (1 - p_bar) / n_bar
    keep = (expected > 0) & (p_bar > 1e-4)
    if not keep.any():
        return 1e9
    ratio = float(np.average(observed[keep] / expected[keep], weights=p_bar[keep]))
    if ratio <= 1.0 + 1e-6:
        return 1e9
    # ratio = (n + a) / (1 + a)  ->  a = (n - ratio) / (ratio - 1)
    alpha0 = (n_bar - ratio) / (ratio - 1.0)
    return float(max(alpha0, 1e-6))


def dm_sigma(counts: np.ndarray) -> float:
    alpha0 = dm_concentration(counts)
    n_bar = float(np.asarray(counts, dtype=float).sum(axis=1).mean())
    return float(np.sqrt((n_bar + alpha0) / (1.0 + alpha0)))


def make_stream(kind, param, k=4000, n=25, seed=0):
    """Dirichlet-multinomial, or an over-dispersed stream that is not Dirichlet."""
    rng = np.random.default_rng(seed)
    p0 = np.array([.5, .2, .15, .1, .05])
    rows = []
    for _ in range(k):
        if kind == "multinomial":
            p = p0
        elif kind == "dirichlet":
            p = rng.dirichlet(param * p0)
        elif kind == "mixture":
            # Over-dispersed but not Dirichlet: the composition switches
            # between two regimes. Same marginal mean, heavier tails, and a
            # correlation structure a Dirichlet cannot represent.
            shift = np.array([param, -param, 0.0, 0.0, 0.0])
            p = np.clip(p0 + (shift if rng.random() < 0.5 else -shift), 1e-3, None)
            p = p / p.sum()
        else:
            raise ValueError(kind)
        rows.append(rng.multinomial(n, p))
    counts = np.array(rows, dtype=float)
    closed = bayesian_multiplicative_replacement(counts)
    return counts, ilr(closed), closed, np.full(k, float(n))


def true_sigma(kind, param, n=25):
    if kind == "multinomial":
        return 1.0
    if kind == "dirichlet":
        return float(np.sqrt((n + param) / (1.0 + param)))
    return float("nan")  # no closed form for the mixture


def evaluate(counts, coords, closed, sizes, phase_one=800, lag=1, seed=0):
    fit = L.calibrate(coords[:phase_one], closed[:phase_one], sizes[:phase_one],
                      target_far=NOMINAL / 1000.0, lag=lag, seed=seed)
    corrected = L.run(fit, coords[phase_one:], closed[phase_one:], sizes[phase_one:])
    uncorrected = L.run_uncorrected(fit, coords[phase_one:], closed[phase_one:],
                                    sizes[phase_one:])

    # Dirichlet-multinomial route: same whitening, dispersion scaled by the
    # fitted inflation instead of measured from residuals.
    sigma_dm = dm_sigma(counts[:phase_one])
    w = L.whiten(coords[phase_one:], closed[phase_one:], sizes[phase_one:],
                 fit.mean, fit.basis, fit.reference)
    from scipy import stats as st
    limit = st.chi2.ppf(1.0 - NOMINAL / 1000.0, fit.dim)
    stat_dm = np.einsum("ij,ij->i", w, w) / (sigma_dm ** 2)
    return {
        "sigma_diff": fit.inflation,
        "sigma_dm": sigma_dm,
        "rate_corrected": 1000.0 * float(corrected["alarm"].mean()),
        "rate_dm": 1000.0 * float((stat_dm > limit).mean()),
        "rate_uncorrected": 1000.0 * float(uncorrected["alarm"].mean()),
    }


def main() -> None:
    # The Dirichlet concentrations span the full sweep plotted in Figure 1, so
    # that the figure and the manuscript table are both read from this file
    # rather than each re-simulating the same experiment.
    cases = [("multinomial", None), ("dirichlet", 120.0), ("dirichlet", 60.0),
             ("dirichlet", 30.0), ("dirichlet", 20.0), ("dirichlet", 12.0),
             ("dirichlet", 8.0), ("mixture", 0.05), ("mixture", 0.10),
             ("mixture", 0.15)]
    rows = []
    print(f"{'stream':<20s}{'true':>7s}{'diff':>8s}{'DM':>8s}"
          f"{'rate diff':>11s}{'rate DM':>10s}{'uncorr':>9s}")
    for kind, param in cases:
        counts, coords, closed, sizes = make_stream(kind, param)
        r = evaluate(counts, coords, closed, sizes)
        t = true_sigma(kind, param)
        r.update(stream=kind, param=param, true_sigma=t)
        rows.append(r)
        label = kind if param is None else f"{kind} {param:g}"
        print(f"{label:<20s}{(f'{t:.2f}' if np.isfinite(t) else '  ?'):>7s}"
              f"{r['sigma_diff']:>8.2f}{r['sigma_dm']:>8.2f}"
              f"{r['rate_corrected']:>11.1f}{r['rate_dm']:>10.1f}"
              f"{r['rate_uncorrected']:>9.1f}", flush=True)

    fin = [r for r in rows if np.isfinite(r["true_sigma"])]
    if fin:
        e_diff = np.mean([abs(r["sigma_diff"] - r["true_sigma"]) for r in fin])
        e_dm = np.mean([abs(r["sigma_dm"] - r["true_sigma"]) for r in fin])
        print(f"\nQ1 mean |error| in sigma_Z where the truth is known: "
              f"successive differences {e_diff:.3f}, Dirichlet-multinomial {e_dm:.3f}")
    mix = [r for r in rows if r["stream"] == "mixture"]
    if mix:
        print("Q3 non-Dirichlet over-dispersion, realised rate (nominal 5.0):")
        for r in mix:
            print(f"   mixture {r['param']:.2f}: differences {r['rate_corrected']:7.1f} "
                  f"| Dirichlet-multinomial {r['rate_dm']:7.1f}")

    RESULTS.mkdir(exist_ok=True)
    json.dump({"nominal_per_1000": NOMINAL, "rows": rows},
              open(RESULTS / "estimator_comparison.json", "w"), indent=2)
    print("\nwrote results/estimator_comparison.json")


if __name__ == "__main__":
    main()
