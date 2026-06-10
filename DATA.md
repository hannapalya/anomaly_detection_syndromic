# Data

The experiments run on simulated daily syndromic-surveillance count data
generated with the simulation framework of Noufaily et al. (2019), which builds
on Noufaily et al. (2013). The data is **not committed** to this repository
because it is large (~104 MB per outbreak magnitude) and is a regenerable
synthetic artifact.

## What the pipeline expects

Three dataset directories, one per outbreak magnitude, in the repository root:

```
big_signal_datasets_small/
big_signal_datasets_medium/
big_signal_datasets_large/
```

Each directory contains, for every signal `1..16`:

| File | Shape | Meaning |
|------|-------|---------|
| `simulated_totals_sig{N}.csv`              | 2548 days × 500 sims | observed daily counts (baseline + injected outbreaks) |
| `simulated_outbreaks_sig{N}.csv`           | 2548 days × 500 sims | injected outbreak counts (the ground-truth labels) |
| `simulated_seasonal_outbreaks_sig{N}.csv`  | (signals 5, 6, 15 only) | recurring seasonal outbreaks |

- **16 signals**, each **500 simulations** of **2548 days** (7 years × 364 days).
- The three directories differ only in the outbreak magnitude parameter `m`
  (small / medium / large); the baselines are otherwise comparable.
- Detectors load **only** `simulated_totals` (input) and `simulated_outbreaks`
  (labels). The `simulated_seasonal_outbreaks` files are **not** merged into the
  evaluation labels.

## Selecting the magnitude

The magnitude is chosen at runtime via the `SYND_DATA_DIR` environment variable,
which the scripts read (and from which `MAG_TAG` is derived):

```bash
SYND_DATA_DIR=big_signal_datasets_small  python IsolationForest.py   # small
SYND_DATA_DIR=big_signal_datasets_large  python IsolationForest.py   # large
```

The `Makefile` sets this for you (`make all`, or `make pipeline MAG=medium`).

## Train / validation / test split

The split is **60 / 20 / 20 over whole simulations, per signal**, with a fixed
seed (`RNG_STATE = 42`), applied by `split_60_20_20()` in `anom_common.py`. With
500 simulations this yields **300 train / 100 validation / 100 test** per signal.
Splitting at the level of whole 7-year series (not across time) prevents temporal
leakage. The R Farrington baseline reads the same split via `splits_for_r.json`,
which `export_splits_for_r.py` writes.

## Obtaining the data

The datasets are produced by the Noufaily simulation framework. If you do not
already have them, contact the author for the `big_signal_datasets_*`
directories, or regenerate them from the simulation framework and place them in
the repository root with the filenames above.

A 100-simulation sample (`signal_datasets/`) was used by the reference
comparison; this study uses the 500-simulation sets for higher per-signal
precision.
