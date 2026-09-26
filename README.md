# Components-of-variance charts for compositional classifier output

Code for the manuscript *Components-of-variance charts for compositional classifier output under batch structure*.

Start here:

    pip install -e .          # the chart: numpy, scipy, pandas
    python smoke_test.py

For the experiments, the CNN inspection model, or the auxiliary datasets:

    pip install -e ".[experiments]"   # scikit-learn, matplotlib, scikit-image
    pip install -e ".[cnn]"           # torch
    pip install -e ".[auxiliary]"     # river
    pip install -e ".[all]"

`requirements.txt` pins the exact versions the reported results were produced
with, for reproduction rather than as minimum requirements.

The smoke test runs in under a minute, needs no downloaded data, and checks the
three properties the method rests on against values stated in the manuscript:
the ILR basis matches its published closed form, the sampling covariance matches
`V diag(p)^-1 V' / n`, a zero between-unit term reproduces the uncorrected chart
exactly, and a simulated over-dispersed stream is brought from ~184 to ~2 false
alarms per 1,000 against a nominal 5.

## The method

`wmmon/varcomp.py` is the chart. `wmmon/laney_coda.py` is the dispersion-matrix
variant and the whitening both share. Minimal use:

```python
from wmmon import composition, varcomp

frame  = composition.build_stream(unit_ids, probabilities, sequence, mode="hard")
coords = composition.ilr_matrix(frame)
props  = composition.proportion_matrix(frame)
sizes  = frame["lot_size"].to_numpy()

fit = varcomp.calibrate(coords[:phase_one], props[:phase_one], sizes[:phase_one],
                        target_far=0.005, estimator="em")
out = varcomp.run(fit, coords[phase_one:], props[phase_one:], sizes[phase_one:])
out["alarm"]        # boolean per unit
fit.between_fraction  # share of variance attributable to batch
```

Two conditions, both established in the paper and both easy to get wrong:

- **`mode="hard"`** (the default). The composition must be counts of arg-max
  predictions, not column sums of predicted probabilities. Summing probabilities inflates the
  reference covariance 29-fold overall and 75-fold in the rarest balance, and the chart silently miscalibrates (Finding 1).
- **Unit sizes.** If monitoring units vary materially in size, the centre must
  depend on unit size; see `variable_unit_size.py`. The diagnostic is the rank
  correlation between unit size and the charting statistic, which should be near
  zero (Finding 3).

`estimator="em"` is the default and needs no `lag`. `estimator="moment"` is
cheaper but requires a lag chosen by the rule given in the method section, and can return a
between-unit covariance dominated by negative eigenvalues on small units.

## Where files live

Every script reads and writes through `wmmon/paths.py`, so nothing is tied to
one machine. Two environment variables set the locations:

| Variable | Holds | Default |
|---|---|---|
| `WMMON_HOME` | `data/`, `cache/`, `results/` | the repository root |
| `WMMON_PAPER` | the manuscript, for the audit scripts | `../paper` |

Put `LSWMD.pkl` in `$WMMON_HOME/data/`. Scripts write intermediate files to
`cache/` and results to `results/`, both created on first use and both excluded
from version control.

## Data

| Dataset | Source |
|---|---|
| WM-811K | Public. Place `LSWMD.pkl` where `wmmon/data.py` expects it; `reduce_lswmd.py` builds the working subset |
| digits | Bundled with scikit-learn, downloaded automatically |
| image_segments | Bundled with `river`, downloaded automatically |

`legacy_pickle.py` is a compatibility shim: `LSWMD.pkl` was written by pandas 0.x
and needs `encoding='latin1'` to load under modern pandas.

## Reproducing the results

Each script writes JSON to `results/` and prints a table. The manuscript's
figures and result tables are regenerated from those files, never transcribed.

| Claim | Script |
|---|---|
| Calibration across datasets and aggregations | `composition_mode` path in `calibration_across_datasets.py` |
| Dispersion estimator vs Dirichlet-multinomial | `estimator_comparison.py` |
| Variance components, EM vs moment | `dispersion_model.py`, `sliding_calibration.py` |
| Identifiability threshold | `identifiability_sweep.py` |
| Variable unit sizes | `variable_unit_size.py` |
| Lag dependence under clustering | `phase1_ratio_sweep.py` |
| Figures | `make_figures.py` |
| Manuscript integrity checks | `audit_manuscript.py`, `stage_audit.py` |

Scripts in this directory that produced results **later withdrawn** are kept so
the record is complete, not because their findings stand: `imbalance_sweep.py`,
`imbalance_wm811k.py`, `run_sethi.py`, `run_md3_native.py`. The manuscript's
supplementary section says which claims were withdrawn and why.

## Cost

One core, no GPU, nine classes, 24 items per unit: a Phase I fit over 324 units
takes about 1.1 s, dominated by simulating the control limit; scoring a unit
thereafter takes about 2.6 microseconds. The CNN inspection model is the only
component that benefits from a GPU, and it is not needed to reproduce any
calibration result if cached probabilities are used.

## Repository layout

    wmmon/            the package - varcomp.py is the chart, laney_coda.py the
                      dispersion-matrix variant, composition.py the ILR layer
    *.py              experiment scripts, one per claim (see the table above)
    smoke_test.py     one-minute check that the install reproduces the results
    pyproject.toml    packaging metadata
    requirements.txt  exact pinned versions used for the reported results

## Caveats

The closed-form sampling covariance is asymptotic. At 25 items per unit it is
about 41% out entry-wise, which the measured dispersion term absorbs; the chart
holds its rate but `sigma_Z` at that unit size is a ratio to an approximate
baseline. Under sparse counts `sigma_Z` also depends on the zero-replacement
prior, while the realised false-alarm rate does not.
