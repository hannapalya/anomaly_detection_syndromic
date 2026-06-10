# Per-signal model-HP search on Colab (VAE, LSTM-AE, BOCPD)

This gives the three detectors that were previously run at a single fixed
configuration a **per-signal model search** comparable to the tabular methods,
so their reported numbers are a tuned result rather than an un-tuned floor.

## What it does

For each method, several candidate model configs are trained; for each signal
the config that maximises the **validation `2·sens + 3·spec`** (specificity floor
0.97) is selected — exactly the rule used to tune KNN / Isolation Forest / etc.

| Method | Search space (grid) | # configs | Selection |
|--------|---------------------|:---------:|-----------|
| VAE     | latent ∈ {4,8,16} × hidden ∈ {32,64} (subset) | 4 | per signal, external (`select_best_hp.py`) |
| LSTM-AE | latent ∈ {4,8} × width ∈ {1.0,0.5} | 4 | per signal, external (`select_best_hp.py`) |
| BOCPD   | hazard ∈ {1/30, 1/50, 1/100} | 3 | per signal, internal (run_bocpd_residual.py) |

`width` scales the BiLSTM hidden sizes (128/64/32) uniformly. Grids are defined
at the top of `cloud/run_hpsearch_colab.py` — edit them to taste.

## Steps

1. **Build the bundle locally** (includes code + datasets + the modified runners
   + `select_best_hp.py`):
   ```bash
   bash cloud/make_cloud_bundle.sh cloud_bundle_hpsearch.tgz   # ~50 MB
   ```
2. **Upload** `cloud_bundle_hpsearch.tgz` to your Google Drive (default path the notebook
   expects: `MyDrive/cloud_bundle_hpsearch.tgz`).
3. Open **`cloud/colab_hpsearch.ipynb`** in Colab. Set a **GPU** runtime
   (Runtime → Change runtime type → A100 or L4; Pro+ gives the long runtimes this
   needs). Set `MAG` in the config cell, then Run all.
4. The final cell writes `MyDrive/hpsearch_results/hpsearch_{MAG}.tgz`.

Approx A100 runtime for `small`: VAE ~1.5 h, LSTM-AE ~3 h, BOCPD ~1.5 h (CPU).
If the session disconnects, re-run the driver cell with `--skip` for the methods
that already finished, e.g. inside the notebook change the run cell to
`['...','--mag',MAG,'--skip','vae','lstm']` to run only BOCPD.

## Importing results back

```bash
# from the repo root, after downloading hpsearch_{MAG}.tgz
tar xzf hpsearch_small.tgz                 # restores results/*.csv + cloud_out/* + score_cache/*
```
This overwrites:
- `results/VAE_per_sig_big_{MAG}.csv`, `results/LSTM_AE_per_sig_big_{MAG}.csv`,
  `results/BOCPD_residual_alone_results_big_{MAG}.csv` — the tuned per-signal metrics
- `results/{VAE,LSTM_AE}_hp_selection_{MAG}.csv` — which config won, per signal
- `cloud_out/{vae,lstm_ae}_{MAG}/`, `score_cache/bocpd_resid_{MAG}/` — the winning scores

Then locally re-run the downstream comparison so the writeup tables/figures pick
up the tuned numbers:
```bash
/Users/u5585063/miniconda3/bin/python analyse_timely_detection.py   # PSD table (Table 11)
/Users/u5585063/miniconda3/bin/python make_tufte_figures.py         # scorecard etc.
# then update Table 1 / Table 6 / unified CSV with the new VAE/LSTM-AE/BOCPD means
```

## Output conventions (for reference)

- VAE writes scores into `VAE_OUT_DIR` (higher = more normal).
- LSTM-AE writes into `LSTM_OUT_DIR` (higher = more anomalous; the metric scripts negate).
- Config-tagged dirs: `cloud_out/{vae,lstm_ae}_{mag}__{tag}/`; the winner is staged
  into the canonical `cloud_out/{vae,lstm_ae}_{mag}/`.
