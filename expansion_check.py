#!/usr/bin/env python3
"""The expansion of E[log p-hat] used for size-dependent centring, against the exact value.

For X ~ Bin(K, p) and p-hat = X/K,
    E[log p-hat] = log p - (1-p)/(2Kp) + (1-p)(1-2p)/(3K^2 p^2) - 3(1-p)^2/(4K^2 p^2) + O(K^-3),
where the last two terms are both of order K^-2. The exact expectation is computed by
summation, conditional on X >= 1. Reported: the error of the first-order series, of the
series with only the third-moment term, and of the series with both second-order terms,
and the expansion parameter 1/(Kp), which must be small for any of them to be valid.
"""
import json
from pathlib import Path

import numpy as np
from scipy import stats

from wmmon.paths import WMMON_HOME

cases = [(200, .3), (200, .1), (100, .3), (100, .1), (50, .3), (50, .2),
         (25, .3), (25, .1), (25, .017)]
rows = []
print(f"{'K':>4}{'p':>7}{'1/(Kp)':>9}{'exact':>11}{'err 1st':>10}{'err 3rd-only':>14}{'err complete':>14}")
for K, p in cases:
    x = np.arange(1, K + 1)
    pm = stats.binom.pmf(x, K, p)
    exact = float((pm * np.log(x / K)).sum() / pm.sum())
    first = np.log(p) - (1 - p) / (2 * K * p)
    third = first + (1 - p) * (1 - 2 * p) / (3 * K ** 2 * p ** 2)
    complete = third - 3 * (1 - p) ** 2 / (4 * K ** 2 * p ** 2)
    rows.append({"K": K, "p": p, "expansion_parameter": 1.0 / (K * p), "exact": exact,
                 "error_first": abs(first - exact), "error_third_only": abs(third - exact),
                 "error_complete": abs(complete - exact)})
    r = rows[-1]
    print(f"{K:>4}{p:>7.3f}{r['expansion_parameter']:>9.2f}{exact:>11.5f}"
          f"{r['error_first']:>10.2e}{r['error_third_only']:>14.2e}{r['error_complete']:>14.2e}")
out = Path(f"{WMMON_HOME}/results")
out.mkdir(exist_ok=True)
json.dump({"rows": rows}, open(out / "expansion_check.json", "w"), indent=2)
