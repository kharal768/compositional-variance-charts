#!/usr/bin/env python3
"""Variance inflation of a Dirichlet-multinomial over the multinomial.

For n trials and total concentration alpha the covariance is the multinomial one
times (alpha + n) / (alpha + 1) (Kartiosuo et al., 2025, Corollary 3). The
manuscript quotes this factor at the ends of the concentration sweep of the
validation experiment, so it is computed here and stored rather than left as an
unsourced figure in the text.
"""
import json
from pathlib import Path

from wmmon.paths import WMMON_HOME

N = 25
rows = [{"alpha": a, "inflation": (a + N) / (a + 1.0)}
        for a in (120.0, 60.0, 30.0, 20.0, 12.0, 8.0)]
for r in rows:
    print(f"alpha {r['alpha']:>6.0f}  inflation {r['inflation']:.2f}")
out = Path(f"{WMMON_HOME}/results")
out.mkdir(exist_ok=True)
json.dump({"n": N, "rows": rows}, open(out / "dm_inflation.json", "w"), indent=2)
