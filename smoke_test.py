#!/usr/bin/env python3
"""Smoke test: does the installation work, and do the core claims reproduce?

Runs in under a minute on one core and needs no downloaded data. It checks the
three properties the paper's method rests on, each against a value stated in
the manuscript, so a failure here means the environment does not reproduce the
published results rather than merely that something crashed.

    python smoke_test.py
"""
from __future__ import annotations

import sys

import numpy as np

FAILURES: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"   [{'ok' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def main() -> int:
    print("1. IMPORTS")
    try:
        from wmmon import composition, laney_coda, varcomp  # noqa: F401
        from wmmon.composition import bayesian_multiplicative_replacement, ilr, ilr_basis
        from wmmon.laney_coda import sampling_covariance
        check("wmmon package imports", True)
    except Exception as exc:  # pragma: no cover
        check("wmmon package imports", False, f"{type(exc).__name__}: {exc}")
        return 1

    print("\n2. ILR BASIS matches the published Helmert form")
    D = 5
    V = ilr_basis(D)
    ref = np.zeros((D - 1, D))
    for i in range(1, D):
        s = np.sqrt(i / (i + 1))
        ref[i - 1, :i] = s / i
        ref[i - 1, i] = -s
    check("basis equals the closed form", float(np.abs(V - ref).max()) < 1e-12)
    check("rows orthonormal", bool(np.allclose(V @ V.T, np.eye(D - 1))))
    check("rows sum to zero", bool(np.allclose(V @ np.ones(D), 0)))

    print("\n3. SAMPLING COVARIANCE equals V diag(p)^-1 V' / n")
    p = np.array([0.5, 0.2, 0.15, 0.1, 0.05])
    ours = sampling_covariance(p, 1000.0, V)
    theirs = (1.0 / 1000.0) * (V @ np.diag(1.0 / p) @ V.T)
    check("matches the closed form", float(np.abs(ours - theirs).max()) < 1e-12)

    print("\n4. REDUCTION PROPERTY: zero between-unit term recovers the "
          "uncorrected chart")
    fit = varcomp.VarCompFit(mean=np.zeros(D - 1), between=np.zeros((D - 1, D - 1)),
                             reference=p, basis=V, limit=1e9, lag=1)
    rng = np.random.default_rng(0)
    z = rng.standard_normal((200, D - 1))
    sizes = np.full(200, 25.0)
    stat = varcomp.run(fit, z, None, sizes)["statistic"]
    inv = np.linalg.pinv(sampling_covariance(p, 25.0, V))
    expected = np.array([z[t] @ inv @ z[t] for t in range(200)])
    check("statistic identical", float(np.abs(stat - expected).max()) < 1e-10)

    print("\n5. CALIBRATION on a simulated over-dispersed stream")
    print("   (manuscript: corrected within a factor of three of "
          "nominal,\n    uncorrected two orders of magnitude out)")
    n, k, nominal = 25, 4000, 5.0
    rng = np.random.default_rng(0)
    rows = [rng.multinomial(n, rng.dirichlet(20.0 * p)) for _ in range(k)]
    counts = np.array(rows, dtype=float)
    closed = bayesian_multiplicative_replacement(counts)
    coords = ilr(closed)
    sizes = np.full(k, float(n))
    fit = laney_coda.calibrate(coords[:800], closed[:800], sizes[:800],
                               target_far=nominal / 1000.0, seed=0)
    corrected = 1000 * float(laney_coda.run(
        fit, coords[800:], closed[800:], sizes[800:])["alarm"].mean())
    uncorrected = 1000 * float(laney_coda.run_uncorrected(
        fit, coords[800:], closed[800:], sizes[800:])["alarm"].mean())
    print(f"        sigma_Z {fit.inflation:.2f} | corrected {corrected:.1f} | "
          f"uncorrected {uncorrected:.1f} per 1,000 (nominal {nominal:.1f})")
    check("dispersion detected above 1", fit.inflation > 1.2)
    check("corrected within 3x nominal", corrected <= 3 * nominal)
    check("uncorrected at least 10x nominal", uncorrected >= 10 * nominal)

    print("\n" + "=" * 62)
    if FAILURES:
        print(f"{len(FAILURES)} check(s) failed:")
        for f in FAILURES:
            print(f"   {f}")
        return 1
    print("all checks passed - the installation reproduces the core results")
    return 0


if __name__ == "__main__":
    sys.exit(main())
