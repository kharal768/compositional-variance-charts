#!/usr/bin/env python3
"""Generate the manuscript figures.

Every panel is recomputed from source here rather than plotted from a table, so
a figure cannot drift away from the result it illustrates. Stage 21 requires
each figure to answer a scientific question; the question is stated in the
caption text emitted alongside each file.

Fig. 1  Does the measured dispersion track the true dispersion, and does
        correcting for it restore the nominal rate?
Fig. 2  Does the correction hold across real datasets, and where does it fail?
Fig. 3  Is the adjacent-difference estimator biased under batch clustering?
"""

from __future__ import annotations
from wmmon.paths import WMMON_HOME, WMMON_PAPER

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from wmmon import composition, laney_coda as L
from wmmon.composition import bayesian_multiplicative_replacement, ilr

OUT = Path(f"{WMMON_PAPER}/figures")
CACHE = Path(f"{WMMON_HOME}/cache")
RESULTS = Path(f"{WMMON_HOME}/results")
NOMINAL = 5.0
plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3,
                     "figure.dpi": 200, "savefig.bbox": "tight"})


def clopper_pearson(k, n, alpha=0.05):
    lo = stats.beta.ppf(alpha / 2, k, n - k + 1) if k > 0 else 0.0
    hi = stats.beta.ppf(1 - alpha / 2, k + 1, n - k) if k < n else 1.0
    return lo * 1000, hi * 1000


# ---------------------------------------------------------------- Figure 1
def figure_one():
    rng = np.random.default_rng(0)
    n, k = 25, 4000
    p0 = np.array([.5, .2, .15, .1, .05])
    alphas = [None, 120.0, 60.0, 30.0, 20.0, 12.0, 8.0]
    sig, cor, unc, label = [], [], [], []
    for a in alphas:
        rows = [rng.multinomial(n, p0 if a is None else rng.dirichlet(a * p0))
                for _ in range(k)]
        c = bayesian_multiplicative_replacement(np.array(rows, float))
        C, S = ilr(c), np.full(k, n)
        fit = L.calibrate(C[:800], c[:800], S[:800],
                          target_far=NOMINAL / 1000.0, seed=0)
        sig.append(fit.inflation)
        cor.append(1000 * L.run(fit, C[800:], c[800:], S[800:])["alarm"].mean())
        unc.append(1000 * L.run_uncorrected(
            fit, C[800:], c[800:], S[800:])["alarm"].mean())
        label.append("multinomial" if a is None else f"{a:g}")

    fig, ax = plt.subplots(1, 2, figsize=(7.2, 2.8))
    ax[0].plot(sig, "o-", color="#1f4e79")
    ax[0].axhline(1.0, ls="--", color="grey", lw=1)
    ax[0].set_xticks(range(len(label)))
    ax[0].set_xticklabels(label, rotation=45, ha="right")
    ax[0].set_xlabel("Dirichlet concentration $\\alpha$ (left: pure multinomial)")
    ax[0].set_ylabel("measured $\\hat\\sigma_Z$")
    ax[0].set_title("(a) dispersion is measured, not assumed", fontsize=9)

    ax[1].semilogy(sig, unc, "s-", color="#a33", label="uncorrected")
    ax[1].semilogy(sig, cor, "o-", color="#1f4e79", label="corrected")
    ax[1].axhline(NOMINAL, ls="--", color="grey", lw=1, label="nominal 5.0")
    ax[1].set_xlabel("measured $\\hat\\sigma_Z$")
    ax[1].set_ylabel("false alarms per 1,000 units")
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title("(b) realised rate against nominal", fontsize=9)
    fig.savefig(OUT / "fig1_simulation.pdf")
    fig.savefig(OUT / "fig1_simulation.png")
    plt.close(fig)
    return {"sigma": sig, "corrected": cor, "uncorrected": unc, "labels": label}


# ---------------------------------------------------------------- Figure 2
def figure_two():
    """Count-aggregated results only.

    Probability-aggregated compositions, for which the reference covariance in this paper is not valid, are not shown.
    """
    rows = [r for r in json.load(open(RESULTS / "composition_mode.json"))
            if r["mode"] == "hard" and (r["lag"] == 4 or r["dataset"] != "wm811k")]
    names = [f"{r['dataset']}\nunit {r['unit']}" for r in rows]
    unc = [r["uncorrected"] for r in rows]
    sig = [r["sigmaZ"] for r in rows]
    vc = [r["varcomp"] for r in rows]
    lo = [r["ci"][0] for r in rows]
    hi = [r["ci"][1] for r in rows]
    x = np.arange(len(rows))
    floor = 0.3  # log axis needs a visible floor for exact zeros

    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    ax.bar(x - 0.26, np.maximum(unc, floor), width=0.25,
           color="#a33", label="uncorrected")
    ax.bar(x, np.maximum(sig, floor), width=0.25,
           color="#c98", label="dispersion matrix")
    ax.bar(x + 0.26, np.maximum(vc, floor), width=0.25,
           color="#1f4e79", label="variance components")
    err = np.vstack([np.maximum(np.array(vc) - np.array(lo), 0),
                     np.maximum(np.array(hi) - np.array(vc), 0)])
    ax.errorbar(x + 0.26, np.maximum(vc, floor), yerr=err, fmt="none",
                ecolor="black", elinewidth=0.8, capsize=2)
    ax.axhline(NOMINAL, ls="--", color="grey", lw=1, label="nominal 5.0")
    ax.set_yscale("log")
    ax.set_ylim(floor * 0.7, 2000)
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=7)
    ax.set_ylabel("false alarms per 1,000 units")
    ax.legend(frameon=False, fontsize=7.5, ncol=2)
    ax.set_title("In-control rate, count-aggregated compositions; "
                 "95% intervals on the corrected chart", fontsize=9)
    # Annotate every bar drawn at the floor, not only
    # the third, which invited reading the first two as small non-zero values.
    for xi, (a, b_, c) in enumerate(zip(unc, sig, vc)):
        for off, v in ((-0.26, a), (0.0, b_), (0.26, c)):
            if v < floor:
                ax.text(x[xi] + off, floor * 0.72, "0", ha="center", fontsize=6)
    fig.savefig(OUT / "fig2_datasets.pdf")
    fig.savefig(OUT / "fig2_datasets.png")
    plt.close(fig)
    return rows


# ---------------------------------------------------------------- Figure 3
def figure_three():
    import phase1_ratio_sweep as P

    def coords_of(proba, unit):
        n = len(proba) // unit
        p = proba[: n * unit]
        ids = np.array([f"u{i // unit:06d}" for i in range(len(p))], dtype=object)
        seq = [f"u{i:06d}" for i in range(n)]
        fr = composition.build_stream(ids, p, seq, mode="hard")
        return (composition.ilr_matrix(fr),
                composition.proportion_matrix(fr),
                fr["lot_size"].to_numpy())

    def sigma_at_lag(C, Pr, S, ph, lag):
        mean = C[:ph].mean(axis=0)
        ref = Pr[:ph].mean(axis=0)
        w = L.whiten(C[:ph], Pr[:ph], S[:ph], mean, None, ref)
        d = w[lag:] - w[:-lag]
        sig = (d.T @ d) / (2.0 * len(d))
        return float(np.sqrt(np.trace(sig) / len(sig)))

    lags = [1, 2, 4, 8, 16]
    series = {}
    pw = np.load(CACHE / "cnnproba_clean_1200_perm.npz")["p"]
    C, Pr, S = coords_of(pw, 24)
    series["WM-811K, lot-structured"] = [sigma_at_lag(C, Pr, S, 324, l) for l in lags]
    rng = np.random.default_rng(0)
    C2, P2, S2 = coords_of(pw[rng.permutation(len(pw))], 24)
    series["WM-811K, item-shuffled"] = [sigma_at_lag(C2, P2, S2, 324, l) for l in lags]
    for nm, u in (("digits", 3), ("image_segments", 4)):
        C3, P3, S3 = coords_of(P.probabilities(nm, 0), u)
        series[nm] = [sigma_at_lag(C3, P3, S3, 180, l) for l in lags]

    fig, ax = plt.subplots(figsize=(5.4, 3.0))
    styles = {"WM-811K, lot-structured": ("o-", "#a33"),
              "WM-811K, item-shuffled": ("s--", "#1f4e79"),
              "digits": ("^:", "#555"),
              "image_segments": ("v:", "#888")}
    for name, values in series.items():
        m, c = styles[name]
        ax.plot(lags, values, m, color=c, label=name)
    ax.axhline(1.0, ls="--", color="grey", lw=1)
    ax.set_xscale("log", base=2)
    ax.set_xticks(lags)
    ax.set_xticklabels(lags)
    ax.set_xlabel("lag $\\ell$ between differenced units")
    ax.set_ylabel("measured $\\hat\\sigma_Z$")
    ax.legend(frameon=False, fontsize=7.5)
    ax.set_title("Only the batch-structured stream rises with lag\n"
                 "(count-aggregated compositions)", fontsize=9)
    fig.savefig(OUT / "fig3_lag.pdf")
    fig.savefig(OUT / "fig3_lag.png")
    plt.close(fig)
    return {"lags": lags, "series": series}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    one = figure_one()
    two = figure_two()
    three = figure_three()
    json.dump({"fig1": one, "fig2": two, "fig3": three},
              open(RESULTS / "figure_data.json", "w"), indent=2, default=str)

    captions = f"""# Figure captions

**Figure 1.** Validation against known dispersion. Streams of five parts and 25
items per monitoring unit, drawn multinomially or from a Dirichlet-multinomial
with concentration $\\alpha$; 800 units for Phase I, 3,200 evaluated; nominal
{NOMINAL:.1f} false alarms per 1,000 units. (a) The measured dispersion
$\\hat\\sigma_Z$ rises monotonically as $\\alpha$ falls and reads
{one['sigma'][0]:.2f} on the pure multinomial stream, where the correct value is
1.00 (dashed). (b) Realised false-alarm rate, log scale, against $\\hat\\sigma_Z$.
The uncorrected chart degrades from {one['uncorrected'][0]:.1f} to
{one['uncorrected'][-1]:.1f} per 1,000; the corrected chart stays between
{min(one['corrected']):.1f} and {max(one['corrected']):.1f}.

**Figure 2.** In-control false-alarm rate on count-aggregated compositions, log
scale, nominal {NOMINAL:.1f} dashed, with exact Clopper-Pearson 95% intervals on
the variance-components bars. WM-811K cells use lag 4; the auxiliary datasets
use lag 1. On WM-811K the variance-components chart covers nominal
(6.4 and 13.5 per 1,000) where the uncorrected chart realises 184.2 and 627.8.
On `digits` and `image_segments` all three charts realise exactly zero false
alarms and are drawn at the axis floor; the intervals there are wide because
exposure is short, so those two cells show that the correction does no harm,
not that it is needed. Probability-aggregated values, for which the reference covariance used here is not valid, are not shown.

**Figure 3.** Measured dispersion as a function of the lag between differenced
units, count-aggregated. Adjacent differencing ($\\ell=1$) assumes consecutive
units are independent draws. Only the lot-structured stream rises with lag, from
{three['series']['WM-811K, lot-structured'][0]:.2f} to
{max(three['series']['WM-811K, lot-structured']):.2f}, plateauing near
$\\ell=4$; the same data shuffled at item level is flat at
{three['series']['WM-811K, item-shuffled'][0]:.2f}. The contrast between the two
WM-811K arms is the result; their absolute level is not. At 24 items per unit
and nine classes, 47% of count cells are zero before replacement, and the
measured dispersion depends strongly on the zero-replacement prior: on the
item-shuffled stream it reads 0.62, 0.99, 1.63 and 2.31 at priors of 0.5, 0.1,
0.01 and 0.001. A value below 1.00 means the replacement has shrunk rare parts
toward a constant and the chart is conservative in those balances. The prior is
therefore a tuning parameter of the dispersion estimate under sparse counts,
which this paper does not resolve; both arms of this figure use the same prior
(0.5), so the comparison is unaffected.
"""
    (OUT / "captions.md").write_text(captions)
    print("figures written to", OUT)
    for f in sorted(OUT.iterdir()):
        print("  ", f.name)


if __name__ == "__main__":
    main()
