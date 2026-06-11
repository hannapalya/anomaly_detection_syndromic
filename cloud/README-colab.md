# Running the robustness study on Google Colab

For the current methodology-correction reruns (strict validation cutoffs plus
the four-method residual-feature runner), use
`cloud/README-colab-methodology.md`. The notes below describe the older
robustness-study split between GPU and CPU notebooks.

Three notebooks. Run them on separate Colab sessions in parallel if you can:

| Notebook | Methods | Runtime needed | Wallclock estimate |
|---|---|---|---|
| `colab_gpu_runner.ipynb` | LSTM-AE, VAE | GPU (T4 or A100) | T4: 4-6 hr / A100: 1-2 hr |
| **`colab_fast_runner.ipynb`** | **NB-HMM, CUSUM, RC-residual, Noufaily/OR-vote, Matrix Profile, KNN, Farrington (R)** | CPU | 3-6 hr (all 3 magnitudes) |
| `colab_cpu_runner.ipynb` | IF, OCSVM, LOF, OneClass-RF | CPU (high-RAM if available) | depends; see ⚠️ in notebook |

## ⚠️ Realistic recommendation

Colab CPU runtimes only have **2 cores free, ~8 cores Pro**, and a hard
session timeout (~12 hr Pro+). The sklearn methods (especially LOF and
OneClass-RF) want lots of cores via `n_jobs=-1`. Estimates per magnitude
on Colab CPU:

| Method | Cores 2 (free) | Cores 8 (Pro) |
|---|---|---|
| IF | ~3 hr | ~1 hr |
| OCSVM | ~6 hr | ~2 hr |
| KNN | ~3 hr | ~1 hr |
| LOF | ~24+ hr | ~6-8 hr |
| OneClass-RF | ~12+ hr | ~3-4 hr |

Across 3 magnitudes that's 30-60 wallclock hours — too much for one session.

**Suggested split:**

1. **Colab GPU notebook** → LSTM-AE and VAE on all 3 magnitudes. Fits in
   one Pro session.
2. **Local laptop (overnight)** → KNN, NB-HMM, CUSUM, RC-residual,
   Noufaily, Farrington-custom (already running via
   `run_robustness_local.sh`).
3. **Colab CPU notebook (Pro, high-RAM, one method at a time)** → IF and
   OCSVM, on one magnitude per session. LOF and OneClass-RF: skip or do
   locally if you have a few days.

## Step-by-step

### 1. On your laptop: build the bundle

```bash
cd /Users/u5585063/anomaly_detection_syndromic
bash cloud/make_cloud_bundle.sh
# Produces cloud_bundle.tgz (~60-80 MB)
```

### 2. Upload to Drive

Upload `cloud_bundle.tgz` to your Google Drive (any folder works; the
notebooks default to `/content/drive/MyDrive/cloud_bundle.tgz`).

### 3. Open the notebook in Colab

- Drag-drop the `.ipynb` file into Colab, or upload via File menu, or
  open from Drive after uploading.
- **Set the runtime type**: Runtime → Change runtime type → GPU (T4/A100)
  for `colab_gpu_runner.ipynb`, or CPU (high-RAM) for
  `colab_cpu_runner.ipynb`.
- Edit `BUNDLE_PATH` and `DRIVE_OUT_DIR` at the top of the notebook if
  the bundle isn't at the default path.
- Run all cells.

### 4. Download results to laptop

The notebooks save `cloud_out_gpu.tgz` / `cloud_out_cpu.tgz` to
`MyDrive/syndromic_results/`. Download from Drive into the repo root,
then:

```bash
cd /Users/u5585063/anomaly_detection_syndromic
tar xzf cloud_out_gpu.tgz
tar xzf cloud_out_cpu.tgz  # if you ran the CPU notebook
python collect_all_results.py --magnitude small
python collect_all_results.py --magnitude medium
python collect_all_results.py --magnitude large
python build_robustness_table.py
```

## Notes / gotchas

- **The CPU notebook lets you scope methods and magnitudes per session.**
  Edit the `METHODS = [...]` and `MAGNITUDES = [...]` cell at the top to
  control what runs. E.g., one session per method per magnitude lets you
  stay under Colab's session timeout.
- **Drive sync:** Saving to Drive can be slow. Each magnitude's
  per-sim CSVs are ~30 KB × 32 files = ~1 MB; bundled tarball is
  ~10-30 MB. Should sync in seconds.
- **Reproducibility:** `RNG_STATE=42` is fixed everywhere; same sims
  selected on cloud as on laptop. Outputs from cloud are directly
  comparable to local outputs.
- **`SYND_DATA_DIR` env var** is set by the notebook cells; you don't
  need to touch it manually.
- **If a session crashes mid-run**, just rerun. The notebooks idempotently
  rewrite per-sig CSVs.

## Files in this folder

- `colab_gpu_runner.ipynb` — Colab GPU notebook (LSTM-AE + VAE).
- `colab_cpu_runner.ipynb` — Colab CPU notebook (IF / OCSVM / KNN / LOF / OneClass-RF).
- `make_cloud_bundle.sh` — produces `cloud_bundle.tgz` for upload.
- `requirements.txt` — pinned Python deps.
- `compute_lstm_alone_metrics.py` — post-processing script the GPU
  notebook calls to compute per-magnitude LSTM-AE alone metrics from
  cached scores.
- `run_cloud_gpu.sh`, `run_cloud_cpu.sh` — legacy Runpod-style shell scripts
  (not needed for Colab; ignore unless you also want to run on Runpod).
- `README-cloud.md` — original Runpod-style instructions (kept for
  reference).
- `README-colab.md` — this file.
