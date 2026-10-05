# Code for: Multinomial control limits fail for classifier-output compositions on batch-structured streams

Code for the manuscript *Multinomial control limits fail for classifier-output compositions on batch-structured streams*.

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
  reference covariance 29-fold overall and 75-fold in its largest diagonal entry, and the chart silently miscalibrates (Section 6.2).
- **Unit sizes.** If monitoring units vary materially in size, the centre must
  depend on unit size; see `variable_unit_size.py`. The diagnostic is the rank
  correlation between unit size and the charting statistic, which should be near
  zero (Supplementary S12).

`estimator="moment"` is the default in `varcomp.calibrate`; `estimator="em"` is what
the paper recommends and what the newer experiment scripts pass explicitly. Only the
moment estimator uses `lag`, which it needs because successive differences understate total
variance when neighbouring units are alike; it can also return a between-unit covariance
dominated by negative eigenvalues on small units. The EM estimator fits the marginal
covariance directly, ignores `lag`, and is positive semi-definite by construction.

## Where files live

Every script reads and writes through `wmmon/paths.py`, so nothing is tied to
one machine. Two environment variables set the locations:

| Variable | Holds | Default |
|---|---|---|
| `WMMON_HOME` | `data/`, `cache/`, `results/` | the repository root |
| `WMMON_PAPER` | the manuscript, for the audit scripts | `../paper` |

Put `LSWMD.pkl` in `$WMMON_HOME/data/`. Scripts write intermediate files to
`cache/`, created on first use and excluded from version control, and results to
`results/`. The JSON files in `results/` are committed, so that every stored value the paper
cites can be inspected, including the five whose scripts were not preserved (see Provenance);
running a script overwrites its file.

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
| Uncorrected rates and the batch-structure permutation, per-unit chart (Table 3) | `mechanism_counts.py`, `permutation_null.py` |
| Chi-square, MD3 and bootstrap rules, by unit size (Table 2); these use the smoothed ILR-MEWMA statistic, not the per-unit chart | `calibration_transfer.py` |
| Order in which fixed blocks are cut (needs the WM-811K subset and the cached probabilities; not run in the public archive) | `check_fixed_block_order.py` |
| Lag and aggregation ablation (`composition_mode.json`) | no script preserved; see Provenance |
| Dispersion estimator vs Dirichlet-multinomial | `estimator_comparison.py` |
| Variance components, EM vs moment (`em_estimator.json`) | no script preserved; `misspecification_sweep.py` reproduces 6.4 at 24 wafers |
| Identifiability threshold | `identifiability_sweep.py` |
| Plain individuals chart, spread and heteroscedasticity | `plain_f_limit_study.py`, `f_limit_between_variance.py`, `phase1_length_sim.py`, `plain_t2_baseline.py`, `plain_dm_streams.py`, `size_class_mechanism.py`, `replacement_rule_study.py` |
| Monte Carlo within-unit term and power | `mc_within_term.py` |
| Variable unit sizes | `variable_unit_size.py` (all-lots rows); rank correlations and minimum-size rows (`min_unit_size.json`): no script preserved |
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

## Run this next: the plain chart on the wafer stream

Every claim in the paper that a plain individuals T^2 with the textbook F limit is calibrated is
simulated. `wafer_stream_plain_t2.py` runs it on the stream, in four settings (fixed blocks of 24 and 48,
lot-aligned units, real lots, a deduplicated stream), beside the plain chart with an empirical limit and the
uncorrected multinomial-limit chart on the same units, and repeats each over random lot orderings to give the
spread over Phase I draws.

    python reduce_lswmd.py --data /path/to/LSWMD.pkl     # once
    python wafer_stream_plain_t2.py                       # add --priors 0.5 0.1 1.0 0.05 to widen the sweep

It was tested end to end on synthetic data from `make_synthetic_lswmd.py` and against hand-computed unit
groupings and a truly multinomial stream, and has not been run on WM-811K. It also reports the plain chart under several zero-replacement priors, because the simulations find its
calibration depends strongly on that choice. It trains the gradient-boosting
model on engineered features, not the network used for the cached probabilities elsewhere, so its numbers
will not match those experiments exactly; the multinomial-limit column is the check that the setting
reproduces the failure the paper reports.

## Provenance

Five stored result files were written by runs whose scripts were not preserved:
`composition_mode.json` (the lag and aggregation ablation), `prior_sensitivity.json`
(the zero-replacement prior sweep), `em_estimator.json` (moment against EM),
`varcomp_calibration.json` (soft-aggregation calibration) and `min_unit_size.json` (the
rank correlations and minimum-size rows). The checks in `claim_evidence.py` read them and
report each one. Several headline values are independently reproduced by scripts that are
here: the uncorrected rates 184.2 and 627.8 (`mechanism_counts.py`, `permutation_null.py`),
the corrected rate 6.4 at 24 wafers (`misspecification_sweep.py`), the aggregation ratios 29
and 75 (`aggregation_covariance.py`) and the all-lots rates of the unit-size experiment
(`variable_unit_size.py`). The lag ablation, the corrected rate at 48 wafers, the prior
sweep and the rank correlations rest on the stored files alone.

Two further sets of figures quoted in the manuscript are not stored in `results/`: the held-out scores of the
gradient-boosting model (Section 5.2; the network's scores are in `results/cnn_quality.json` and match the manuscript) and
the duplicate statistics of Section 5.1 (unique maps, within-stream duplicates, train-to-stream leakage). The data counts of
Section 5.1 are in `results/census.json`.

## Software

`requirements.txt` pins the versions the reported results were produced with, including PyTorch for the network of
Section 5.2; `pyproject.toml` gives lower bounds only.
