# Legacy notebooks (superseded)

These notebooks are an **earlier prototype** and are kept only for history. They
do **not** reproduce the paper and should not be cited as the methods of record.
Use the scripts in the repository root instead (see the top-level `README.md`).

How they differ from the paper scripts:

| Notebook | Paper replacement | Why it differs |
|----------|-------------------|----------------|
| `Farrington.ipynb` | `run_farrington_custom.R` + `collect_farrington_metrics.py` | Same custom Farrington implementation, but evaluated outside the unified R-comparator pipeline. |
| `LSTM.ipynb` | `cache_lstm_ae_prod_scores.py` | Keras (vs PyTorch); reads splits from IF CSVs rather than the seeded `split_60_20_20`; bakes in its own thresholding; also contains two divergent architecture variants. |
| `OCSVM.ipynb` | `run_ocsvm.py` | Uses only **4 features** (vs the 20-d vector), a **0.95** specificity floor (vs 0.97), and does not use the R-comparator metrics. |
| `IF_LSTM_ensemble.ipynb` | `search_ensembles.py` + `fix_ensemble_val_selection.py` | A per-signal model *chooser* (IF or LSTM), not an OR-vote ensemble. |
| `Ensemble_3way_voting.ipynb` | `search_ensembles.py` + `fix_ensemble_val_selection.py` | Fixed majority-of-3 vote (IF+LSTM+OCSVM); no Farrington, no validation-based selection, no alarm-magnitude alignment. |

The paper-version corrections — R-comparator metrics, the true 60/20/20 split,
validation-based ensemble selection, and Farrington magnitude alignment — live
only in the root scripts.
