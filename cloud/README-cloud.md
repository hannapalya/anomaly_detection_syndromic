# Cloud-compute orchestration for the syndromic-surveillance robustness study

The local laptop runs the fast/medium methods (NB-HMM, CUSUM, RC-residual,
Noufaily, KNN, Farrington-custom) for all three outbreak magnitudes. The
slow methods are off-loaded to cloud compute:

| Where | Methods | Instance type | Approx wallclock per magnitude |
|---|---|---|---|
| GPU instance | LSTM-AE, VAE | RTX 3090 / A5000 (24GB) | 1.5 hr |
| CPU instance | IF, KNN, LOF, OCSVM, OneClass-RF | 16+ cores, 32GB RAM | 3–6 hr |

## What you transfer to each instance

A tarball produced by:

```bash
cd /Users/u5585063/anomaly_detection_syndromic
tar czf cloud_bundle.tgz \
    *.py *.R *.json \
    big_signal_datasets_small big_signal_datasets_medium big_signal_datasets_large \
    cloud/
ls -lh cloud_bundle.tgz   # expect ~60-80MB
```

Then `scp cloud_bundle.tgz user@INSTANCE:/workspace/` and on the instance:

```bash
mkdir -p /workspace/syndromic && cd /workspace/syndromic
tar xzf /workspace/cloud_bundle.tgz
pip install -r cloud/requirements.txt
```

## On the GPU instance

```bash
cd /workspace/syndromic
bash cloud/run_cloud_gpu.sh small medium large    # or pick subset
# Output: cloud_out_gpu.tgz containing per-signal score CSVs + per-magnitude summary CSVs.
```

## On the CPU instance

```bash
cd /workspace/syndromic
bash cloud/run_cloud_cpu.sh small medium large    # or pick subset
# Can also run a subset of methods:
METHODS="lof oneclass_rf" bash cloud/run_cloud_cpu.sh small medium large
# Output: cloud_out_cpu.tgz containing per-signal alarm CSVs + per-magnitude summary CSVs.
```

## Back on the local laptop

```bash
scp user@GPU:/workspace/syndromic/cloud_out_gpu.tgz .
scp user@CPU:/workspace/syndromic/cloud_out_cpu.tgz .
tar xzf cloud_out_gpu.tgz -C /Users/u5585063/anomaly_detection_syndromic/
tar xzf cloud_out_cpu.tgz -C /Users/u5585063/anomaly_detection_syndromic/

# Then aggregate per-magnitude:
cd /Users/u5585063/anomaly_detection_syndromic
python collect_all_results.py --magnitude small
python collect_all_results.py --magnitude medium
python collect_all_results.py --magnitude large

# Per-magnitude unified CSVs are written to:
#   results/ALL_METHODS_unified_big_small.csv
#   results/ALL_METHODS_unified_big_medium.csv
#   results/ALL_METHODS_unified_big_large.csv

# Build the side-by-side robustness table:
python build_robustness_table.py
# -> results/ROBUSTNESS_table.csv  (methods × magnitudes)
```

## Key conventions

- **`SYND_DATA_DIR`** environment variable controls which outbreak-magnitude
  directory is read. `anom_common.py` reads it at import time. Set it before
  invoking any Python runner. The R script (`run_farrington_custom.R`) also
  honours it.
- **Splits are identical across magnitudes** because RNG_STATE=42 + 500 sims
  produces the same simulation-index lists regardless of outbreak content.
- **Score CSVs are named per-signal** (e.g. `lstm_ae_val_scores_signal_3.csv`).
  Magnitude is encoded in the `cloud_out/<method>_<magnitude>/` subdirectory.
- **Summary CSVs are named per-magnitude** (e.g. `results/LSTM_AE_per_sig_big_large.csv`).

## Sanity-check commands on the cloud instance

```bash
python -c "from anom_common import DATA_DIR, MAG_TAG; print(DATA_DIR, MAG_TAG)"
# Should print whatever SYND_DATA_DIR is set to.

python -c "import torch; print('CUDA:', torch.cuda.is_available(), 'device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```
