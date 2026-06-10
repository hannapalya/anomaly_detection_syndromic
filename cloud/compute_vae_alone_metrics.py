#!/usr/bin/env python3
"""Compute "VAE alone" R-comparator metrics for each magnitude from cached
val/test score CSVs (produced by run_vae_count.py). Equivalent to
compute_lstm_alone_metrics.py but for VAE; VAE caches already store -NLL
(higher = more normal), so scores are used as-is, not negated.
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)
for p in (_repo_root, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

from anom_common import (
    split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    TRAIN_DAYS, VALID_DAYS,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
)


def score_method_for_mag(mag, cache_dir):
    os.environ["SYND_DATA_DIR"] = f"big_signal_datasets_{mag}"
    import importlib
    import anom_common as _ac
    importlib.reload(_ac)
    _load = _ac.load_data

    rng = np.random.RandomState(RNG_STATE)
    rows = {}
    for S in SIGNALS:
        Xs, Ys = _load(S)
        sims = []
        for i, c in enumerate(Xs.columns):
            x = Xs[c].to_numpy(np.float32)
            y = Ys[c].to_numpy(np.int32)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(x=x, y=y, sim=f"s{S}_{i}", sim_idx=i))
        _, val_sims, test_sims = split_60_20_20(sims, rng)

        v_fp = os.path.join(cache_dir, f"vae_val_scores_signal_{S}.csv")
        t_fp = os.path.join(cache_dir, f"vae_test_scores_signal_{S}.csv")
        if not (os.path.exists(v_fp) and os.path.exists(t_fp)):
            print(f"  [sig {S}] score CSVs not found in {cache_dir}; skip")
            continue
        # VAE cache convention: HIGHER = more normal (already -NLL). Use as-is.
        v = pd.read_csv(v_fp).to_numpy().flatten(order="F")
        t = pd.read_csv(t_fp).to_numpy()
        val_lengths = [VALID_DAYS] * len(val_sims)
        c = tune_contamination_threshold(val_sims, val_lengths, v,
                                         spec_target=SPEC_TARGET,
                                         w_sens=W_SENS, w_spec=W_SPEC)
        thr = np.percentile(v, c * 100)
        A = (t <= thr).astype(int)
        O = np.stack([d['y'] for d in test_sims], axis=1)
        rows[S] = dict(
            sensitivity=compute_sensitivity_R(A, O),
            specificity=compute_specificity_R(A, O, IDX_RANGE),
            fpr=compute_fpr_R(A, O, IDX_RANGE),
            pod=compute_pod_R(A, O),
            timeliness=compute_timeliness_R(A, O),
            contamination=c,
        )
        print(f"  [sig {S}] sens={rows[S]['sensitivity']:.3f} "
              f"spec={rows[S]['specificity']:.3f} tim={rows[S]['timeliness']:.3f}")
    return pd.DataFrame.from_dict(rows, orient="index")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--magnitudes", nargs="*", default=["small", "medium", "large"])
    parser.add_argument("--score-dir", default=None,
                        help="Override score directory (default: cloud_out/vae_{mag})")
    args = parser.parse_args()

    for mag in args.magnitudes:
        cache_dir = args.score_dir or f"cloud_out/vae_{mag}"
        if not os.path.isdir(cache_dir):
            print(f"skip {mag}: {cache_dir} not found")
            continue
        print(f"\n=== VAE alone, magnitude={mag} (scores from {cache_dir}) ===")
        df = score_method_for_mag(mag, cache_dir)
        if df.empty:
            print(f"  No signals processed for {mag}")
            continue
        print(df.round(3))
        print("Means:", df.mean(numeric_only=True).round(3).to_dict())
        os.makedirs("results", exist_ok=True)
        outp = f"results/VAE_per_sig_big_{mag}.csv"
        df.to_csv(outp)
        print(f"Saved {outp}")
