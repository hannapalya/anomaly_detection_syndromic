#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

PY="${PY:-python3}"
DATA_ROOT="${DATA_ROOT:-/Users/u5585063/anomaly_detection_syndromic}"
MAGS="${MAGS:-small medium large}"
RUN_DEEP="${RUN_DEEP:-0}"
RUN_RESIDUAL="${RUN_RESIDUAL:-1}"

for mag in $MAGS; do
  export SYND_DATA_DIR="${DATA_ROOT}/big_signal_datasets_${mag}"
  if [[ ! -d "$SYND_DATA_DIR" ]]; then
    echo "Missing data directory: $SYND_DATA_DIR" >&2
    exit 1
  fi

  echo "=== Corrected rerun: magnitude=${mag} ==="
  "$PY" IsolationForest.py
  "$PY" run_knn.py
  "$PY" run_lof.py
  "$PY" run_ocsvm.py
  "$PY" run_nbhmm.py
  "$PY" run_cusum.py
  "$PY" run_bocpd_residual.py
  "$PY" run_ratechange_residual.py
  if [[ "$RUN_RESIDUAL" == "1" ]]; then
    "$PY" run_residual_ml.py
  fi

  if [[ "$RUN_DEEP" == "1" ]]; then
    "$PY" run_vae_count.py
    "$PY" cloud/compute_vae_alone_metrics.py --magnitudes "$mag"
  fi

  "$PY" run_unsup_or_ensemble.py
  "$PY" collect_all_results.py --magnitude "$mag"
done

"$PY" build_robustness_table.py
"$PY" generate_per_signal_table.py > results/per_signal_tables.tex

for mag in $MAGS; do
  export SYND_DATA_DIR="${DATA_ROOT}/big_signal_datasets_${mag}"
  "$PY" analyse_timely_detection.py
  "$PY" fix_ensemble_val_selection.py
done
