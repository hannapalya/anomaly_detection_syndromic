# Reproduction pipeline for the syndromic anomaly-detection paper.
# See README.md and DATA.md. Outbreak magnitude is selected via SYND_DATA_DIR.
#
#   make all                 # full pipeline over all magnitudes + build tables
#   make pipeline MAG=medium # detectors + aggregation for one magnitude
#   make detectors MAG=small # just the 11 detectors for one magnitude
#   make tables              # cross-magnitude / per-signal / timely tables

PY  ?= python3
MAG ?= small
DATA_DIR := big_signal_datasets_$(MAG)
export SYND_DATA_DIR := $(DATA_DIR)

.PHONY: all pipeline detectors aggregate tables splits smoke clean

## Fast environment check (seconds, no data/GPU/R needed). Run this first.
smoke:
	$(PY) smoke_test.py

## Full reproduction: every magnitude, then the cross-magnitude tables.
all: splits
	for m in small medium large; do $(MAKE) pipeline MAG=$$m; done
	$(MAKE) tables

## One magnitude: run all detectors, then aggregate into the unified table.
pipeline: detectors aggregate

## Export the shared 60/20/20 split for the R Farrington baseline.
splits:
	$(PY) export_splits_for_r.py

## The 11 detectors (+ residual-feature variant) for magnitude $(MAG).
## Deep-learning (PyTorch/GPU) and Farrington (R) steps need extra setup; see README.
detectors:
	@echo ">>> Detectors for magnitude=$(MAG)  (SYND_DATA_DIR=$(DATA_DIR))"
	$(PY) IsolationForest.py
	$(PY) run_knn.py
	$(PY) run_lof.py
	$(PY) run_ocsvm.py
	$(PY) run_nbhmm.py
	$(PY) run_cusum.py
	$(PY) run_bocpd_residual.py
	$(PY) run_ratechange_residual.py
	$(PY) run_residual_ml.py
	$(PY) cache_lstm_ae_prod_scores.py && $(PY) cloud/compute_lstm_alone_metrics.py
	$(PY) run_vae_count.py            && $(PY) cloud/compute_vae_alone_metrics.py
	Rscript run_farrington_custom.R 0.01 test
	Rscript run_farrington_custom.R 0.01 val
	$(PY) collect_farrington_metrics.py
	$(PY) run_unsup_or_ensemble.py

## Collapse the per-method per-signal CSVs into one unified table for $(MAG).
aggregate:
	$(PY) collect_all_results.py --magnitude $(MAG)

## Cross-magnitude and per-signal tables (run after all magnitudes are aggregated).
tables:
	$(PY) build_robustness_table.py
	$(PY) generate_per_signal_table.py
	for m in small medium large; do \
		SYND_DATA_DIR=big_signal_datasets_$$m $(PY) analyse_timely_detection.py; \
		SYND_DATA_DIR=big_signal_datasets_$$m $(PY) fix_ensemble_val_selection.py; \
	done

## Remove generated outputs (keeps the source data).
clean:
	rm -rf results/*.csv score_cache cloud_out *_inputs splits_for_r.json
