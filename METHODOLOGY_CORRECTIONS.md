# Methodology Corrections Ledger

This file records the issues found during the manuscript/code audit and separates
text/code corrections from items that still require regenerated outputs or
missing provenance artifacts.

## Corrected in the manuscript

- Replaced operating-point language to state the stricter protocol now used by
  the runners: candidate contamination percentiles are converted to validation
  score cutoffs, and the selected numeric cutoff is applied unchanged to test.
- Clarified that the OCSVM grid tunes window length and `nu`, while using
  scikit-learn's `gamma="scale"` throughout and subsampling large training
  matrices for tractability.
- Reworded the VAE method: training uses an ELBO-style objective, but scoring
  uses masked-tail Negative-Binomial negative log-likelihood rather than the
  full-window ELBO.
- Reworded BOCPD-residual: the implemented score is predictive surprise
  `-log p(r_t | r_1:t-1)`, not a combined score with run-length-zero posterior.
- Reworded Farrington Flexible: the inlined implementation keeps the trend term
  when enough non-zero history is available, rather than admitting it only after
  a significance test.
- Updated ensemble wording to match the validation-selected OR-vote search over
  two- to five-method subsets.

## Requires a rerun before final numerical claims

- **Strict validation-threshold outputs.** The standalone score-based runners now
  use validation-selected numeric cutoffs on test. Regenerate all affected
  per-method CSVs, unified tables, robustness tables, ensemble summaries, and
  manuscript tables/figures before treating the numerical claims as final.
- **Residual-feature outputs.** `run_residual_ml.py` now reproduces IF, KNN, LOF,
  and OCSVM residual-feature variants using the raw-count tabular grids and the
  strict validation-cutoff protocol. Regenerate Table 10 and any downstream
  residual-feature discussion from the new CSVs.
- **Rerun command.** With the synthetic datasets available under a common root,
  run `DATA_ROOT=/path/to/data-root ./rerun_methodology_corrections.sh`. Set
  `MAGS="small"` for a single magnitude, `RUN_RESIDUAL=0` to skip residual
  ablations, or `RUN_DEEP=1` to retrain/recompute the VAE outputs as well.
- **NB-HMM emission dispersion.** The current implementation ties emissions to a
  Pearson dispersion estimate from the seasonal baseline. Any change to a
  different Negative-Binomial parameterisation should be treated as a method
  change and rerun across all magnitudes.

## Provenance gaps to close

- The deep-model hyperparameter-search notes refer to external Colab helpers
  (`select_best_hp.py`, `cloud/run_hpsearch_colab.py`) that are not tracked in
  this checkout.
- Bootstrap and correlation-analysis claims should be backed by tracked scripts
  or the relevant input artifacts (`results/headline_bootstrap_per_signal.csv`,
  signal-characteristic tables) before submission.
