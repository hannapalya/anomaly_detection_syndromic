#!/usr/bin/env python3
"""Read the per-sim Farrington alarm CSVs produced by run_farrington_custom.R,
compute R-comparator metrics using the existing Python pipeline, and save
a summary.

Handles both naming conventions:
  - farrington_custom_alarms_signal_{S}.csv   (from run_farrington_custom.R)
  - farrington_alarms_signal_{S}.csv          (from older run_farrington.R)

Also supports searching in a subdirectory (e.g. cloud_out/fast_small/) via
the --csv-dir argument.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Ensure working directory is on path (needed on Colab)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, MAG_TAG,
)


def find_alarm_csv(csv_dir, suffix, S, alpha_suffix=None):
    """Try 'farrington_custom_{suffix}_signal_{S}.csv' first, then fallback
    to 'farrington_{suffix}_signal_{S}.csv'.

    If alpha_suffix is given (e.g. 'a005'), prefer the alpha-tagged version
    'farrington_custom_{alpha_suffix}_{suffix}_signal_{S}.csv'.
    """
    candidates = []
    if alpha_suffix:
        candidates.append(f"farrington_custom_{alpha_suffix}_{suffix}_signal_{S}.csv")
    for prefix in ("farrington_custom", "farrington"):
        candidates.append(f"{prefix}_{suffix}_signal_{S}.csv")
    for fname in candidates:
        p = os.path.join(csv_dir, fname)
        if os.path.exists(p):
            return p
    return None


def main(use_alarmall=False, csv_dir=".", mag_tag=None, alpha_suffix=None, out_label=None):
    rng = np.random.RandomState(RNG_STATE)
    summary = {}
    suffix = "alarmsall" if use_alarmall else "alarms"
    label = out_label or ("Farrington_alarmsall" if use_alarmall else "Farrington_alarms")
    tag = mag_tag or MAG_TAG

    for S in SIGNALS:
        Xsig, Ysig = load_data(S)
        sims = []
        for sim_idx, col in enumerate(Xsig.columns):
            x = Xsig[col].to_numpy(np.float32, copy=False)
            y = Ysig[col].to_numpy(np.int32,  copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{sim_idx}", sim_idx=sim_idx))
        train_sims, val_sims, test_sims = split_60_20_20(sims, rng)

        ap = find_alarm_csv(csv_dir, suffix, S, alpha_suffix=alpha_suffix)
        if ap is None:
            print(f"[sig {S}] no alarm CSV found in {csv_dir} for {suffix}; skip")
            continue
        A = pd.read_csv(ap).to_numpy().astype(int)   # [343, n_test_sims]
        if A.shape[1] != len(test_sims):
            print(f"[sig {S}] A.shape[1]={A.shape[1]} != len(test_sims)={len(test_sims)}; skip")
            continue
        O_full_test = np.stack([d['y'] for d in test_sims], axis=1)

        m = dict(
            sensitivity=compute_sensitivity_R(A, O_full_test),
            specificity=compute_specificity_R(A, O_full_test, IDX_RANGE),
            fpr=compute_fpr_R(A, O_full_test, IDX_RANGE),
            pod=compute_pod_R(A, O_full_test),
            timeliness=compute_timeliness_R(A, O_full_test),
        )
        summary[S] = m
        print(f"[sig {S}] sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
              f"fpr={m['fpr']:.3f} pod={m['pod']:.3f} tim={m['timeliness']:.3f}")

    if summary:
        df = pd.DataFrame.from_dict(summary, orient="index")
        print(f"\n=== {label} means (mag={tag}) ===")
        print(df.mean(numeric_only=True))
        os.makedirs("results", exist_ok=True)
        outp = f"results/{label}_under_pipeline_big_{tag}.csv"
        df.to_csv(outp)
        print(f"Saved {outp}")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv-dir", default=".",
                        help="Directory containing farrington_*_alarms_signal_*.csv files")
    parser.add_argument("--magnitude", default=None,
                        help="Override magnitude tag (small/medium/large) for output filename")
    parser.add_argument("--alpha-suffix", default=None,
                        help="If R script was run with a non-default alpha, the suffix used "
                             "in the alarm filenames (e.g. 'a005' for alpha=0.005).")
    parser.add_argument("--out-label", default=None,
                        help="Override the output filename label (default: Farrington_alarms[all])")
    args = parser.parse_args()

    tag = args.magnitude  # None means use MAG_TAG from env
    print(f"=== Farrington CUSTOM (alarm: with limit54 filter) [dir={args.csv_dir}] ===")
    main(use_alarmall=False, csv_dir=args.csv_dir, mag_tag=tag,
         alpha_suffix=args.alpha_suffix, out_label=args.out_label)
    print(f"\n\n=== Farrington CUSTOM (alarmsall: without limit54 filter) [dir={args.csv_dir}] ===")
    main(use_alarmall=True, csv_dir=args.csv_dir, mag_tag=tag,
         alpha_suffix=args.alpha_suffix,
         out_label=(args.out_label + "all" if args.out_label else None))
