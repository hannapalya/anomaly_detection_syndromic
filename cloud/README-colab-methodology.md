# Methodology-correction reruns on Google Colab

Use this when you want to regenerate outputs after the strict
validation-cutoff and residual-feature runner corrections without heating up
the laptop.

## 1. Build the bundle locally

From the repo that contains the data directories:

```bash
cd /Users/u5585063/anomaly_detection_syndromic
DATA_ROOT=/Users/u5585063/anomaly_detection_syndromic \
  bash cloud/make_methodology_colab_bundle.sh
```

This writes `cloud_methodology_bundle.tgz`. Upload that tarball to Google Drive,
for example to `MyDrive/cloud_methodology_bundle.tgz`.

## 2. Colab setup cells

In Colab, use a CPU high-RAM runtime for sklearn/residual methods and a GPU
runtime only if running VAE/deep-model work.

```python
from google.colab import drive
drive.mount("/content/drive")
```

```bash
!tar xzf /content/drive/MyDrive/cloud_methodology_bundle.tgz -C /content
%cd /content/syndromic
!pip install -q -r cloud/requirements.txt
```

## 3. Run safe chunks

Start with small chunks and let each cell pack its outputs back to Drive.

Residual IF/KNN, all signals for small magnitude:

```bash
!python cloud/run_methodology_colab.py \
  --magnitudes small \
  --methods residual \
  --residual-methods IF,KNN \
  --residual-signals 1-16 \
  --pack /content/drive/MyDrive/syndromic_results/residual_if_knn_small.tgz
```

Residual LOF/OCSVM should be split more cautiously:

```bash
!python cloud/run_methodology_colab.py \
  --magnitudes small \
  --methods residual \
  --residual-methods LOF \
  --residual-signals 1-4 \
  --pack /content/drive/MyDrive/syndromic_results/residual_lof_small_1_4.tgz
```

Strict-cutoff reruns for non-residual score-based methods:

```bash
!python cloud/run_methodology_colab.py \
  --magnitudes small \
  --methods if,knn,lof,ocsvm,nbhmm,cusum,bocpd,ratechange \
  --aggregate \
  --pack /content/drive/MyDrive/syndromic_results/core_small.tgz
```

For medium/large, change `--magnitudes`. For multiple magnitudes, use
`--magnitudes small,medium,large`, but only after confirming one magnitude fits
within the Colab session timeout.

## 4. Bring results back

Download the `.tgz` files from Drive, then from the repo root:

```bash
tar xzf residual_if_knn_small.tgz
tar xzf core_small.tgz
python collect_all_results.py --magnitude small
python build_robustness_table.py
python generate_per_signal_table.py
```

If you ran multiple residual chunks, extract them all before rebuilding tables.

## Notes

- Colab sessions can disconnect. Rerun the same chunk with `--merge`; residual
  per-signal CSVs are merged by signal.
- LOF and OCSVM are the slowest CPU chunks. Prefer one method and a few signals
  per session.
- The helper packs `results/`, `score_cache/`, `cloud_out/`, and root-level
  alarm/score CSVs so downstream ensemble scripts can use the returned outputs.
