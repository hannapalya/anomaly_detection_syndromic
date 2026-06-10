#!/usr/bin/env python3
"""After LSTM-AE caching on each magnitude, compute the "LSTM-AE alone"
R-comparator metrics: tune contamination threshold on val, evaluate on test.

Works both locally and on Colab (adds working directory to sys.path).
Also handles score CSVs that live directly in the working directory (not
just under cloud_out/lstm_ae_{mag}/).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd

# Ensure the repo root is importable (needed on Colab where cwd may differ)
_script_dir = os.path.dirname(os.path.abspath(__file__))
_repo_root = os.path.dirname(_script_dir)  # parent of cloud/
for p in (_repo_root, os.getcwd()):
    if p not in sys.path:
        sys.path.insert(0, p)

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
)


def score_method_for_mag(mag, cache_dir, score_kind="lstm_ae"):
    """Load cached scores and compute R-comparator metrics.

    The function reloads anom_common after setting SYND_DATA_DIR so that
    load_data picks up the correct dataset directory.
    """
    # Set env BEFORE reloading anom_common
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

        v_fp = os.path.join(cache_dir, f"lstm_ae_val_scores_signal_{S}.csv")
        t_fp = os.path.join(cache_dir, f"lstm_ae_test_scores_signal_{S}.csv")
        if not (os.path.exists(v_fp) and os.path.exists(t_fp)):
            print(f"  [sig {S}] score CSVs not found in {cache_dir}; skip")
            continue
        # LSTM cache convention: HIGHER = more anomalous (reconstruction MSE)
        # tune_contamination_threshold convention: HIGHER = more normal -> negate.
        v = -pd.read_csv(v_fp).to_numpy().flatten(order="F")
        t = -pd.read_csv(t_fp).to_numpy()
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
    df = pd.DataFrame.from_dict(rows, orient="index")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--magnitudes", nargs="*", default=["small", "medium", "large"],
                        help="Magnitude(s) to process")
    parser.add_argument("--score-dir", default=None,
                        help="Override score directory (default: cloud_out/lstm_ae_{mag})")
    args = parser.parse_args()

    for mag in args.magnitudes:
        cache_dir = args.score_dir or f"cloud_out/lstm_ae_{mag}"
        if not os.path.isdir(cache_dir):
            print(f"skip {mag}: {cache_dir} not found")
            continue
        print(f"\n=== LSTM-AE alone, magnitude={mag} (scores from {cache_dir}) ===")
        df = score_method_for_mag(mag, cache_dir)
        if df.empty:
            print(f"  No signals processed for {mag}")
            continue
        print(df.round(3))
        print("Means:", df.mean(numeric_only=True).round(3).to_dict())
        os.makedirs("results", exist_ok=True)
        outp = f"results/LSTM_AE_per_sig_big_{mag}.csv"
        df.to_csv(outp)
        print(f"Saved {outp}")
