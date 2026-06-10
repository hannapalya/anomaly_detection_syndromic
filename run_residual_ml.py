#!/usr/bin/env python3
"""
Residual-based ML feature variant for IF and KNN.

Pipeline per (signal, sim):
  1. Fit a Negative-Binomial seasonal+DOW baseline on the first 6 years
     (TRAIN_DAYS) of THIS sim's count series.
  2. Compute Pearson residuals on the entire 7-year series — these strip
     out the seasonal cycle, day-of-week effects, and baseline magnitude.
  3. Build the standard 20-dim sliding-window feature matrix on the
     residual stream instead of the raw counts.
  4. Train IF and KNN on the residual features from train sims.
  5. Score val and test using the same residual features.

This is the variant that should help ML methods on highly seasonal signals
(rhinitis, heat stroke, insect bites, arthropod bites) where vanilla
features confound seasonal peaks with outbreak anomalies.

Outputs:
  results/IF_residual_per_sig_big_{MAG}.csv
  results/KNN_residual_per_sig_big_{MAG}.csv

Per-magnitude (set via SYND_DATA_DIR env var).
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

from anom_common import (
    load_data, split_60_20_20,
    create_residual_features, fit_nb_baseline, pearson_residuals,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
    MAG_TAG,
)


WINDOW_SIZE = 14   # sliding window for feature engineering
N_TREES = 200      # IF
K_NEI = 10         # KNN


class KNNAnomaly:
    def __init__(self, k=K_NEI):
        self.k = k
        self._nn = NearestNeighbors(n_neighbors=k, n_jobs=-1)

    def fit(self, X):
        self._nn.fit(X)
        return self

    def score_samples(self, X):
        d, _ = self._nn.kneighbors(X)
        return -d.mean(axis=1)   # higher = more normal (smaller distance)


def build_features_for_sim(d, window_size):
    """Return tuple (train_feats, val_feats, test_feats, val_labels_aligned,
    test_labels_aligned).

    train_feats: residual features on FIRST TRAIN_DAYS (no outbreak contamination)
    val_feats:   residual features on last VALID_DAYS (used to tune threshold)
    test_feats:  residual features on test window [ABS_START:ABS_END]

    Baseline parameters are fitted on first TRAIN_DAYS of THIS sim.
    """
    x = np.asarray(d['x'], dtype=np.float64)
    y = np.asarray(d['y'], dtype=np.int32)

    # Fit baseline on training portion (no outbreak contamination)
    params, dispersion = fit_nb_baseline(x[:TRAIN_DAYS])
    resid_full = pearson_residuals(x, params, dispersion, start=0)

    # Build features on three windows. For ML training we use the residual
    # stream of the training portion (with WINDOW_SIZE-1 days of context).
    from anom_common import create_features

    # Train: residuals on first TRAIN_DAYS
    train_feats = create_features(resid_full[:TRAIN_DAYS], window_size)

    # Val: take last VALID_DAYS days WITH a window of context before
    val_ctx_start = len(x) - VALID_DAYS - (window_size - 1)
    if val_ctx_start < 0:
        val_feats = None; val_labels = None
    else:
        val_resid = resid_full[val_ctx_start: len(x)]
        val_feats = create_features(val_resid, window_size)
        # Align labels: features start at window_size-1 days into the ctx
        # which equals val_ctx_start + (window_size-1) = len(x) - VALID_DAYS
        val_labels = y[len(x) - VALID_DAYS: len(x)]

    # Test: ABS_START:ABS_END with leading window context
    test_ctx_start = ABS_START - (window_size - 1)
    if test_ctx_start < 0:
        test_feats = None; test_labels = None
    else:
        test_resid = resid_full[test_ctx_start: ABS_END]
        test_feats = create_features(test_resid, window_size)
        test_labels = y[ABS_START:ABS_END]

    return train_feats, val_feats, test_feats, val_labels, test_labels


def evaluate_signal(S, methods=('IF', 'KNN')):
    rng = np.random.RandomState(RNG_STATE)
    # Burn through previous signals to keep RNG aligned with other runners
    for prior in SIGNALS:
        if prior == S:
            break
        Xp, _ = load_data(prior)
        sims_p = []
        for i, c in enumerate(Xp.columns):
            x = Xp[c].to_numpy(np.float32, copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims_p.append(dict(x=x, y=None, sim_idx=i))
        if sims_p:
            split_60_20_20(sims_p, rng)

    Xsig, Ysig = load_data(S)
    sims = []
    for i, c in enumerate(Xsig.columns):
        x = Xsig[c].to_numpy(np.float32, copy=False)
        y = Ysig[c].to_numpy(np.int32, copy=False)
        if len(x) >= TRAIN_DAYS + VALID_DAYS:
            sims.append(dict(x=x, y=y, sim_idx=i))

    train_sims, val_sims, test_sims = split_60_20_20(sims, rng)

    # ---- Build train features ----
    train_blocks = []
    for d in train_sims:
        feats, _, _, _, _ = build_features_for_sim(d, WINDOW_SIZE)
        if feats is not None and len(feats):
            train_blocks.append(feats)
    if not train_blocks:
        return None
    Xtr = np.concatenate(train_blocks, axis=0)

    # ---- Build val features ----
    val_feats_list, val_labels_list = [], []
    for d in val_sims:
        _, vf, _, vl, _ = build_features_for_sim(d, WINDOW_SIZE)
        if vf is None or vl is None:
            continue
        val_feats_list.append(vf)
        val_labels_list.append(vl)

    # ---- Build test features ----
    test_feats_list, test_labels_list, test_sim_indices = [], [], []
    for d in test_sims:
        _, _, tf, _, tl = build_features_for_sim(d, WINDOW_SIZE)
        if tf is None or tl is None:
            continue
        test_feats_list.append(tf)
        test_labels_list.append(tl)
        test_sim_indices.append(d['sim_idx'])

    if not val_feats_list or not test_feats_list:
        return None

    scaler = StandardScaler().fit(Xtr)
    Xtr_s = scaler.transform(Xtr)

    # Build val/test matrices per-sim so we can split scores back
    val_lengths = [len(v) for v in val_feats_list]
    test_lengths = [len(t) for t in test_feats_list]
    Xval = scaler.transform(np.concatenate(val_feats_list, axis=0))
    Xte = scaler.transform(np.concatenate(test_feats_list, axis=0))

    # Stack labels for R-comparator metrics
    O_full_val = np.stack([d['y'] for d in val_sims], axis=1)
    O_full_test = np.stack([d['y'] for d in test_sims], axis=1)

    results_out = {}

    for method in methods:
        if method == 'IF':
            mdl = IsolationForest(
                n_estimators=N_TREES,
                contamination=0.03,
                random_state=RNG_STATE,
                n_jobs=-1,
                max_samples=min(20000, len(Xtr_s)),
            ).fit(Xtr_s)
            val_scores = mdl.decision_function(Xval)   # higher = more normal
            te_scores  = mdl.decision_function(Xte)
        elif method == 'KNN':
            mdl = KNNAnomaly(k=K_NEI).fit(Xtr_s)
            val_scores = mdl.score_samples(Xval)
            te_scores  = mdl.score_samples(Xte)
        else:
            continue

        # Tune contamination threshold on validation (higher=normal convention)
        c_best = tune_contamination_threshold(
            val_sims, val_lengths, val_scores,
            spec_target=SPEC_TARGET, w_sens=W_SENS, w_spec=W_SPEC,
        )
        thr = np.percentile(val_scores, c_best * 100)

        # Predict on test using same percentile threshold (computed against val
        # distribution, applied to test scores). For fair comparison apply to
        # test distribution directly.
        thr_test = np.percentile(te_scores, c_best * 100)
        yhat_te = (te_scores <= thr_test).astype(int)

        # Split into per-sim columns and column-stack into matrix
        offset = 0
        A_list = []
        for L in test_lengths:
            A_list.append(yhat_te[offset:offset + L])
            offset += L
        A = np.column_stack(A_list)

        m = dict(
            sensitivity=compute_sensitivity_R(A, O_full_test),
            specificity=compute_specificity_R(A, O_full_test, IDX_RANGE),
            fpr=compute_fpr_R(A, O_full_test, IDX_RANGE),
            pod=compute_pod_R(A, O_full_test),
            timeliness=compute_timeliness_R(A, O_full_test),
            contamination=c_best,
        )
        results_out[method] = m

        # Cache per-sim alarms + scores for downstream ensemble search
        import os as _os
        cache_dir = f"score_cache/{method.lower()}_residual_{MAG_TAG}"
        _os.makedirs(cache_dir, exist_ok=True)
        # Build val scores matrix and test scores matrix (per-sim columns)
        val_score_mat = np.column_stack([
            val_scores[sum(val_lengths[:i]):sum(val_lengths[:i + 1])]
            for i in range(len(val_lengths))
        ])
        test_score_mat = np.column_stack([
            te_scores[sum(test_lengths[:i]):sum(test_lengths[:i + 1])]
            for i in range(len(test_lengths))
        ])
        # Convention: higher = MORE NORMAL for these scorers (matches anom_common stacker convention)
        pd.DataFrame(val_score_mat,
                     columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
                     ).to_csv(f"{cache_dir}/val_scores_signal_{S}.csv", index=False)
        pd.DataFrame(test_score_mat,
                     columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
                     ).to_csv(f"{cache_dir}/test_scores_signal_{S}.csv", index=False)
        pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]
                     ).to_csv(f"{cache_dir}/alarms_signal_{S}.csv", index=False)
        # Backward-compat flat path for small magnitude
        if MAG_TAG == "small":
            pd.DataFrame(val_score_mat,
                         columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
                         ).to_csv(f"{method.lower()}_residual_val_scores_signal_{S}.csv", index=False)
            pd.DataFrame(test_score_mat,
                         columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
                         ).to_csv(f"{method.lower()}_residual_test_scores_signal_{S}.csv", index=False)
            pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]
                         ).to_csv(f"{method.lower()}_residual_alarms_signal_{S}.csv", index=False)

    return results_out


if __name__ == "__main__":
    print(f"Residual-feature ML runner — magnitude={MAG_TAG}")
    print(f"  Methods: IF, KNN  |  window_size={WINDOW_SIZE}  trees={N_TREES}  k={K_NEI}")
    print("=" * 80)

    if_rows = []
    knn_rows = []
    for S in SIGNALS:
        print(f"[sig {S}] processing...", flush=True)
        try:
            res = evaluate_signal(S)
        except Exception as e:
            import traceback
            print(f"[sig {S}] FAILED: {e}", flush=True)
            traceback.print_exc()
            continue
        if res is None:
            print(f"[sig {S}] skipped (no valid features)")
            continue
        if 'IF' in res:
            m = res['IF']
            print(f"  IF-residual:  sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
                  f"pod={m['pod']:.3f} tim={m['timeliness']:.3f}", flush=True)
            if_rows.append(dict(signal=S, **m))
        if 'KNN' in res:
            m = res['KNN']
            print(f"  KNN-residual: sens={m['sensitivity']:.3f} spec={m['specificity']:.3f} "
                  f"pod={m['pod']:.3f} tim={m['timeliness']:.3f}", flush=True)
            knn_rows.append(dict(signal=S, **m))

    if if_rows:
        df_if = pd.DataFrame(if_rows).set_index('signal')
        print("\n=== IF-RESIDUAL (per signal) ===")
        print(df_if)
        print("Means:\n", df_if.mean(numeric_only=True))
        df_if.to_csv(f"results/IF_residual_per_sig_big_{MAG_TAG}.csv")
        print(f"Wrote results/IF_residual_per_sig_big_{MAG_TAG}.csv")

    if knn_rows:
        df_knn = pd.DataFrame(knn_rows).set_index('signal')
        print("\n=== KNN-RESIDUAL (per signal) ===")
        print(df_knn)
        print("Means:\n", df_knn.mean(numeric_only=True))
        df_knn.to_csv(f"results/KNN_residual_per_sig_big_{MAG_TAG}.csv")
        print(f"Wrote results/KNN_residual_per_sig_big_{MAG_TAG}.csv")
