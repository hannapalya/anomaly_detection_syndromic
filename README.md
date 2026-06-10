# Unsupervised Anomaly Detection for Syndromic Surveillance

Code for the chapter *"Unsupervised Anomaly Detection Methods for Early Outbreak
Detection from Syndromic Surveillance."* It benchmarks 11 unsupervised anomaly
detectors and their OR-vote ensembles against the Farrington Flexible baseline on
simulated daily syndromic-surveillance data, across three outbreak magnitudes
(small / medium / large), using the evaluation metrics of Noufaily et al. (2019).

> **These scripts are the methods of record for the paper.** The earlier
> notebooks (`Farrington/LSTM/OCSVM/IF_LSTM_ensemble/Ensemble_3way_voting`) have
> been moved to [`legacy/`](legacy/) — they implement an older, divergent
> methodology and do **not** reproduce the paper. See [`legacy/README.md`](legacy/README.md).

## Methods

| Family | Detectors | Script |
|--------|-----------|--------|
| Density / distance on tabular features | Isolation Forest | `IsolationForest.py` |
| | K-Nearest Neighbours | `run_knn.py` |
| | Local Outlier Factor | `run_lof.py` |
| | One-Class SVM | `run_ocsvm.py` |
| Reconstruction-based deep learning | LSTM Autoencoder | `cache_lstm_ae_prod_scores.py` |
| | Variational Autoencoder (NegBin) | `run_vae_count.py` |
| Seasonal-baseline statistical | Farrington Flexible (α=0.01) | `run_farrington_custom.R` → `collect_farrington_metrics.py` |
| | CUSUM (NB seasonal) | `run_cusum.py` |
| | NB-HMM | `run_nbhmm.py` |
| | BOCPD-residual | `run_bocpd_residual.py` |
| | RateChange-residual | `run_ratechange_residual.py` |
| Ensembles | OR-vote search + validation-based selection | `search_ensembles.py`, `fix_ensemble_val_selection.py` |
| Residual-feature variant | IF / KNN / LOF / OCSVM on NB-baseline residuals | `run_residual_ml.py` |

Shared infrastructure: `anom_common.py` (20-d feature construction, the
`split_60_20_20` split, and the R-comparator metrics) and `r_comparator_metrics.py`.

## Setup

```bash
pip install -r requirements.txt
```

The Farrington baseline additionally needs **R ≥ 4.0** with the `surveillance`
package (tested with 1.24.1):

```r
install.packages("surveillance")
```

The deep-learning detectors (LSTM-AE, VAE) use PyTorch and benefit from a GPU;
they were trained on Colab. The `cloud/` helpers (`compute_lstm_alone_metrics.py`,
`compute_vae_alone_metrics.py`) turn the cached per-day scores into per-signal
metric tables.

## Data

Not committed (large, regenerable). See **[DATA.md](DATA.md)** for the expected
`big_signal_datasets_{small,medium,large}/` layout and the train/val/test split.

## Running

The whole pipeline is driven by the `Makefile`. The outbreak magnitude is selected
by `SYND_DATA_DIR` (handled for you by the make targets).

```bash
make smoke                  # fast env check (seconds; no data/GPU/R) — run this first
make all                    # full pipeline over all 3 magnitudes, then build tables
make pipeline MAG=medium    # detectors + aggregation for one magnitude
make detectors MAG=small    # just run the 11 detectors for one magnitude
make tables                 # build the cross-magnitude / per-signal / timely tables
```

`make smoke` synthesizes a tiny series if no dataset is present, so a fresh clone
can verify the Python environment, `anom_common`, scikit-learn, and the metric
functions in a few seconds before committing to the full multi-hour run.

`./run_all.sh` is a one-command wrapper for `make all`.

> The deep-learning (PyTorch/GPU) and Farrington (R) steps have the extra
> prerequisites noted under Setup. The `make` targets encode the canonical run
> order; they have not been executed end-to-end inside this checkout, so on a
> fresh machine expect to satisfy those prerequisites first.

## Table → script → output map

All per-method runners write `results/<Method>_per_sig_big_<MAG>.csv`, which the
aggregators consume.

| Table(s) | Produced by | Output |
|----------|-------------|--------|
| Table 1 (individuals + ensembles, medium) | `collect_all_results.py --magnitude medium` | `results/ALL_METHODS_unified_big_medium.csv` |
| Table 2 (sens/timeliness × 3 magnitudes)   | `build_robustness_table.py`                 | `results/ROBUSTNESS_table.csv` |
| Table 3, 8 (ensembles, val-selected)       | `fix_ensemble_val_selection.py` (+ `search_ensembles.py`) | `results/ensemble_val_vs_test_selection.csv` |
| Tables 6, 7 (per-signal sensitivity)       | `generate_per_signal_table.py`              | LaTeX to stdout |
| Table 10 (residual-feature variant)        | `run_residual_ml.py`                        | `results/{IF,KNN}_residual_per_sig_big_<MAG>.csv` |
| Tables 11 / 11M / 11L, 12 (timely detection)| `analyse_timely_detection.py`              | `results/timely_detection_summary_<MAG>.csv`, `results/outbreak_duration_dist_<MAG>.csv` |
| Table 9 (recommended configs)              | assembled by hand from Tables 2 & 8         | — |

## Reproducibility notes

- Fixed seed `RNG_STATE = 42` for the simulation split; every method trains,
  tunes, and evaluates on identical partitions.
- Per-method alarm thresholds are tuned on the **validation** partition by
  maximising `2·sensitivity + 3·specificity` subject to specificity ≥ 0.97.
  Farrington runs at its fixed operating point (α = 0.01).
- Ensemble configurations are **selected on validation and reported on test**
  (no selection-on-test bias), and Farrington alarms are magnitude-aligned.
- Evaluation window: the final 49 weeks, absolute day range `[2205, 2548)`.
