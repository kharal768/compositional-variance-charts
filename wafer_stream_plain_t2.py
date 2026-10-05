#!/usr/bin/env python3
"""The plain individuals T^2 with the textbook F limit, on the wafer stream.

Every claim in the manuscript that this chart is calibrated is simulated. This runs it on
the stream itself, in four settings, and reports it beside two comparators on the same units:

  plain T2, F limit           Phase I mean and sample covariance, textbook Phase II limit
  plain T2, empirical limit   the same statistic, 0.995 quantile of the Phase I statistics
  multinomial limit           the statistic built from the multinomial sampling covariance with
                              the asymptotic chi-square limit, i.e. the failure the paper reports

Settings
  fixed blocks     24 and 48 consecutive wafers, cut irrespective of lot boundaries
  lot-aligned      whole consecutive lots grouped until the unit reaches 24 or 48 wafers
  real lots        each lot is a unit, 1 to 25 wafers; also reported with lots below 4 wafers removed
  deduplicated     exact duplicate maps removed from the stream, then fixed blocks of 24

Each setting is repeated over several random orderings of the same lots, which changes the Phase I
window and so gives the spread of the in-control rate over Phase I draws on the real stream. The
repeats reuse the same wafers, so their exposures overlap and the pooled interval is optimistic;
the per-repeat distribution is the more honest summary.

Usage
    python reduce_lswmd.py --data /path/to/LSWMD.pkl        # once; writes data/subset.pkl
    python wafer_stream_plain_t2.py                          # features, classifier, charts

Feature extraction is the slow step (cached afterwards). The classifier is the gradient-boosting
model on engineered features, not the convolutional network used for the cached probabilities
elsewhere, so the numbers will not match those experiments to the decimal; the multinomial-limit
column is the check that the setting reproduces the failure the paper reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon import classifier, composition, data, features
from wmmon.composition import ilr_basis
from wmmon.laney_coda import sampling_covariance
from wmmon.paths import WMMON_HOME

NOMINAL = 5.0
WARMUP = 20


def cp(k: int, n: int):
    lo = stats.beta.ppf(0.025, k, n - k + 1) * 1000 if k else 0.0
    hi = stats.beta.ppf(0.975, k + 1, n - k) * 1000 if k < n else 1000.0
    return float(lo), float(hi)


def stream_probabilities(stream, lots, seed, n_jobs, cache, train_df):
    """Class probabilities for every wafer in the chosen lots, cached."""
    key = cache / f"plain_t2_proba_{len(lots)}_{seed}.npz"
    mask = stream["lot"].astype(str).isin(set(lots)).to_numpy()
    sub = stream[mask].reset_index(drop=True)
    if key.exists():
        z = np.load(key, allow_pickle=True)
        return sub, z["p"]
    xt_path = cache / f"plain_t2_train_{seed}.npz"
    if xt_path.exists():
        xt = np.load(xt_path)["x"]
    else:
        print("extracting training features ...", flush=True)
        xt = features.extract_batch(train_df["wafer_map"].tolist(), n_jobs=n_jobs)
        np.savez_compressed(xt_path, x=xt)
    model = classifier.train_inspection_model(
        xt, train_df["label"].tolist(), list(data.CLASSES), seed=seed)
    print(f"extracting features for {len(sub):,} stream wafers ...", flush=True)
    xs = features.extract_batch(sub["wafer_map"].tolist(), n_jobs=n_jobs)
    p = model.predict_proba(xs)
    np.savez_compressed(key, p=p)
    return sub, p


def duplicate_mask(maps) -> np.ndarray:
    """True for the second and later occurrence of an exact-duplicate map."""
    seen, dup = set(), np.zeros(len(maps), dtype=bool)
    for i, m in enumerate(maps):
        a = np.asarray(m)
        k = (a.shape, a.tobytes())
        if k in seen:
            dup[i] = True
        else:
            seen.add(k)
    return dup


def unit_ids(order_lots, lot_of, setting, size):
    """Per-wafer unit id for wafers already arranged in (lot sequence, wafer) order."""
    n = len(lot_of)
    if setting == "fixed":
        return np.arange(n) // size
    pos = {l: i for i, l in enumerate(order_lots)}
    lot_idx = np.array([pos[str(l)] for l in lot_of])
    if setting == "real":
        return lot_idx
    if setting == "aligned":
        # group whole consecutive lots until the group reaches `size` wafers
        sizes = np.bincount(lot_idx, minlength=len(order_lots))
        group, cur, tot = np.zeros(len(order_lots), dtype=int), 0, 0
        for i, s in enumerate(sizes):
            group[i] = cur
            tot += s
            if tot >= size:
                cur += 1
                tot = 0
        return group[lot_idx]
    raise ValueError(setting)


def charts(proba, ids, prior=0.5):
    """Counts of in-control alarms for the three charts on one arrangement."""
    uniq = list(dict.fromkeys(ids.tolist()))
    frame = composition.build_stream(
        np.array([f"u{u:06d}" for u in ids], dtype=object), proba,
        [f"u{u:06d}" for u in uniq], mode="hard", prior=prior)
    z = composition.ilr_matrix(frame)
    props = composition.proportion_matrix(frame)
    n = frame["lot_size"].to_numpy().astype(float)
    D = z.shape[1]
    ph = max(20 * D, int(0.4 * len(z)))
    if len(z) - ph - WARMUP < 50 or ph <= D + 2:
        return None
    mu = z[:ph].mean(axis=0)
    inv = np.linalg.pinv(np.cov(z[:ph], rowvar=False))
    d = z - mu
    stat = np.einsum("ij,jk,ik->i", d, inv, d)
    f_lim = (D * (ph + 1) * (ph - 1) / (ph * (ph - D))) * stats.f.ppf(1 - NOMINAL / 1000, D, ph - D)
    e_lim = float(np.quantile(stat[:ph], 1 - NOMINAL / 1000))
    ref = props[:ph].mean(axis=0)
    basis = ilr_basis(D + 1)
    cache = {}
    ms = np.empty(len(z))
    for t in range(len(z)):
        k = float(n[t])
        if k not in cache:
            cache[k] = np.linalg.pinv(sampling_covariance(ref, k, basis))
        ms[t] = d[t] @ cache[k] @ d[t]
    m_lim = float(stats.chi2.ppf(1 - NOMINAL / 1000, D))
    sl = slice(ph + WARMUP, None)
    exposure = len(z) - ph - WARMUP
    return {"exposure": int(exposure), "phase_one": int(ph),
            "plain_f": int((stat[sl] > f_lim).sum()),
            "plain_empirical": int((stat[sl] > e_lim).sum()),
            "multinomial_limit": int((ms[sl] > m_lim).sum()),
            "median_unit_size": float(np.median(n))}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=f"{WMMON_HOME}/data/subset.pkl")
    ap.add_argument("--stream-lots", type=int, default=1200)
    ap.add_argument("--repeats", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-jobs", type=int, default=None)
    ap.add_argument("--priors", type=float, nargs="+", default=[0.5, 0.1, 1.0],
                    help="Dirichlet priors of the Bayesian-multiplicative zero replacement; the plain chart's "
                         "calibration depends on it (the first is the primary and the Jeffreys prior is 0.5)")
    ap.add_argument("--cache", default=f"{WMMON_HOME}/cache")
    ap.add_argument("--out", default=f"{WMMON_HOME}/results/wafer_stream_plain_t2.json")
    a = ap.parse_args()
    cache = Path(a.cache); cache.mkdir(parents=True, exist_ok=True)

    df, _ = data.load_lswmd(a.data)
    train, _, stream = data.split_labelled(df, seed=a.seed)
    stream = stream.reset_index(drop=True)
    lots = [str(l) for l in data.permuted_lot_order(stream, seed=a.seed)[: a.stream_lots]]
    sub, proba = stream_probabilities(stream, lots, a.seed, a.n_jobs, cache, train)
    print(f"{len(sub):,} wafers in {len(lots)} lots; real lot sizes "
          f"{int(sub.groupby('lot').size().min())}-{int(sub.groupby('lot').size().max())}", flush=True)
    sub["lot"] = sub["lot"].astype(str)
    dup = duplicate_mask(sub["wafer_map"].tolist())
    print(f"exact-duplicate maps: {dup.sum():,} ({100 * dup.mean():.1f}%)", flush=True)

    settings = [("fixed blocks, 24", "fixed", 24, False, 1), ("fixed blocks, 48", "fixed", 48, False, 1),
                ("lot-aligned, 24", "aligned", 24, False, 1), ("lot-aligned, 48", "aligned", 48, False, 1),
                ("real lots", "real", 0, False, 1), ("real lots, 4 or more wafers", "real", 0, False, 4),
                ("deduplicated, fixed blocks, 24", "fixed", 24, True, 1)]
    out = {"nominal_per_1000": NOMINAL, "wafers": int(len(sub)), "lots": len(lots),
           "duplicate_share": float(dup.mean()), "repeats": a.repeats, "rows": []}
    print(f"\n{'setting':<32}{'units':>7}{'expos.':>8}{'plain F':>9}{'plain emp.':>12}{'multinomial':>13}"
          f"{'  F: q10/median/q90 across orderings':>38}")
    for name, kind, size, dedup, min_size in settings:
        rec = []
        for r in range(a.repeats):
            rng = np.random.default_rng(1000 * a.seed + r)
            order = list(rng.permutation(lots))
            pos = {l: i for i, l in enumerate(order)}
            keep = ~dup if dedup else np.ones(len(sub), dtype=bool)
            # arrange the kept wafers in (lot position, wafer index) order
            idx = np.flatnonzero(keep)
            fr = sub.iloc[idx].copy(); fr["_p"] = fr["lot"].map(pos)
            order_idx = np.lexsort((fr["wafer_index"].to_numpy(), fr["_p"].to_numpy()))
            fr = fr.iloc[order_idx]; pr = proba[idx[order_idx]]
            if min_size > 1:
                sizes = fr.groupby("lot")["lot"].transform("size").to_numpy()
                m = sizes >= min_size
                fr, pr = fr[m], pr[m]
            lot_of = fr["lot"].to_numpy()
            ids = unit_ids(order, lot_of, kind, size)
            resd = {pri: charts(pr, ids, prior=pri) for pri in a.priors}
            if all(v is not None for v in resd.values()):
                rec.append(resd)
        if not rec:
            print(f"{name:<32}  too few units")
            continue
        primary = a.priors[0]
        row = {"setting": name, "repeats": len(rec), "primary_prior": primary, "by_prior": {}}
        for pri in a.priors:
            rp = [r[pri] for r in rec]
            tot = sum(x["exposure"] for x in rp)
            pf = np.array([1000.0 * x["plain_f"] / x["exposure"] for x in rp])
            entry = {"pooled_exposure": int(tot), "mean_exposure": int(np.mean([x["exposure"] for x in rp])),
                     "mean_unit_size": float(np.mean([x["median_unit_size"] for x in rp]))}
            for k in ("plain_f", "plain_empirical", "multinomial_limit"):
                kk = sum(x[k] for x in rp)
                lo, hi = cp(kk, tot)
                entry[k] = {"alarms": kk, "rate": 1000.0 * kk / tot, "ci": [lo, hi]}
            entry["plain_f_across_orderings"] = {"q10": float(np.quantile(pf, .1)),
                                                 "median": float(np.quantile(pf, .5)),
                                                 "q90": float(np.quantile(pf, .9)),
                                                 "share_above_2x": float(np.mean(pf > 2 * NOMINAL))}
            row["by_prior"][str(pri)] = entry
        e0 = row["by_prior"][str(primary)]
        row.update({"mean_exposure": e0["mean_exposure"], "pooled_exposure": e0["pooled_exposure"],
                    "mean_unit_size": e0["mean_unit_size"]})
        out["rows"].append(row)
        a0 = e0["plain_f_across_orderings"]
        print(f"{name:<32}{rec[0][primary]['phase_one']:>7}{e0['mean_exposure']:>8}{e0['plain_f']['rate']:>9.1f}"
              f"{e0['plain_empirical']['rate']:>12.1f}{e0['multinomial_limit']['rate']:>13.1f}"
              f"{a0['q10']:>14.1f}/{a0['median']:.1f}/{a0['q90']:.1f}", flush=True)
    print(f"\nplain chart with the F limit, rate per 1,000, by zero-replacement prior "
          f"(nominal {NOMINAL:.1f}; the paper's simulations find it depends strongly on this choice)")
    print(f"{'setting':<32}" + "".join(f"{'prior ' + str(p):>12}" for p in a.priors))
    for r in out["rows"]:
        print(f"{r['setting']:<32}" + "".join(f"{r['by_prior'][str(p)]['plain_f']['rate']:>12.1f}" for p in a.priors))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump(out, open(a.out, "w"), indent=2)
    print(f"\nwrote {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
