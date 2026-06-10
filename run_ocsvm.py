#!/usr/bin/env python3
"""
One-Class SVM on tabular features (per-signal HP grid, R-comparator metrics).

Mirrors run_knn.py / run_lof.py / run_oneclass_rf.py:
- 60/20/20 split with RNG_STATE=42 (matches run_stacked_meta and other runners)
- Tabular features via anom_common.create_features
- HP grid: (window, gamma, nu)
- Threshold tuned by anom_common.tune_contamination_threshold (spec >= 0.97)
- Saves per-signal alarm/outbreak CSVs + summary CSV
"""

import os
import time
import numpy as np
import pandas as pd
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from anom_common import (
    load_data, split_60_20_20, create_features,
    sens_spec, pod_anyhit, timeliness_single, split_by_lengths,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    compute_sensitivity, compute_specificity, compute_pod, compute_timeliness,
    tune_contamination_threshold,
    SIGNALS, TRAIN_DAYS, VALID_DAYS, RNG_STATE,
    ABS_START, ABS_END, WIN_LEN, IDX_RANGE,
    SPEC_TARGET, W_SENS, W_SPEC,
    MAG_TAG,
)


# Smaller HP grid for OCSVM (it's O(N^2) — 9-config grid would be slow)
HP_GRID = [
    (7,  "scale", 0.03),
    (7,  "scale", 0.05),
    (14, "scale", 0.03),
    (14, "scale", 0.05),
    (21, "scale", 0.05),
]


if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)
    summary_all, rows_val, rows_test = {}, [], []

    for S in SIGNALS:
        print(f"\n--- Signal {S} (OCSVM) ---", flush=True)
        Xsig, Ysig = load_data(S)
        sims = []
        for sim_idx, col in enumerate(Xsig.columns):
            x = Xsig[col].to_numpy(np.float32, copy=False)
            y = Ysig[col].to_numpy(np.int32,  copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{sim_idx}", sim_idx=sim_idx))
        if not sims:
            continue
        train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
        print(f"  Splits: {len(train_sims)} train, {len(val_sims)} val, {len(test_sims)} test", flush=True)

        def build_train(window_size):
            parts = [create_features(d['x'][:TRAIN_DAYS], window_size) for d in train_sims]
            parts = [p for p in parts if len(p)]
            return np.concatenate(parts) if parts else np.empty((0, 20), np.float32)

        def build_window(sims_, window_size, abs_start_fn, abs_end_fn):
            XL, YL, lengths, used = [], [], [], []
            for d in sims_:
                a = abs_start_fn(d) if callable(abs_start_fn) else abs_start_fn
                b = abs_end_fn(d) if callable(abs_end_fn) else abs_end_fn
                ctx_start = a - (window_size - 1)
                if ctx_start < 0:
                    continue
                x_ctx = d['x'][ctx_start:b]
                feats = create_features(x_ctx, window_size)
                if len(feats) == b - a:
                    y_al = d['y'][a:b].astype(int)
                    XL.append(feats); YL.append(y_al); lengths.append(b - a); used.append(d)
            X = np.concatenate(XL) if XL else np.empty((0, 20), np.float32)
            Y = np.concatenate(YL) if YL else np.empty((0,), np.int32)
            return X, Y, lengths, used

        best = dict(score=-1.0, params=None, model=None, scaler=None,
                    val_scores=None, val_lengths=None, val_sims_used=None,
                    Yval=None, thr_val=None, c=None)

        for (WIN, GAMMA, NU) in HP_GRID:
            t0 = time.time()
            Xtr = build_train(WIN)
            if not len(Xtr):
                continue
            # Subsample training data if huge — OCSVM is O(N^2), 600K points is impractical.
            if len(Xtr) > 30000:
                idx = np.random.RandomState(RNG_STATE).choice(len(Xtr), 30000, replace=False)
                Xtr = Xtr[idx]
            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)

            try:
                ocs = OneClassSVM(kernel="rbf", gamma=GAMMA, nu=NU, cache_size=500).fit(Xtr_s)
            except Exception as e:
                print(f"  HP win={WIN} gamma={GAMMA} nu={NU}: FAILED {e}", flush=True)
                continue

            Xval, Yval, val_lengths, val_sims_used = build_window(
                val_sims, WIN, lambda d: len(d['x']) - VALID_DAYS, lambda d: len(d['x']))
            if not len(Xval) or len(val_sims_used) == 0:
                continue
            Xval_s = scaler.transform(Xval)
            val_scores = ocs.decision_function(Xval_s)  # higher = more normal
            c = tune_contamination_threshold(val_sims_used, val_lengths, val_scores,
                                             spec_target=SPEC_TARGET,
                                             w_sens=W_SENS, w_spec=W_SPEC)
            thr = np.percentile(val_scores, c * 100)
            yhat = (val_scores <= thr).astype(int)
            O_full_val = np.stack([d['y'] for d in val_sims_used], axis=1)
            A_list, ofs = [], 0
            for L in val_lengths:
                A_list.append(yhat[ofs:ofs + L]); ofs += L
            A_val = np.column_stack(A_list)
            sens = compute_sensitivity_R(A_val, O_full_val)
            spec = compute_specificity_R(A_val, O_full_val, IDX_RANGE)
            score = (W_SENS * sens + W_SPEC * spec) if (not np.isnan(spec) and spec >= SPEC_TARGET) \
                     else (spec if not np.isnan(spec) else -1.0)
            print(f"  win={WIN} gamma={GAMMA} nu={NU}: sens={sens:.3f} spec={spec:.3f} "
                  f"score={score:.3f} c={c} ({time.time()-t0:.1f}s)", flush=True)
            if score > best['score']:
                best.update(score=score, params=(WIN, GAMMA, NU),
                            model=ocs, scaler=scaler,
                            val_scores=val_scores, val_lengths=val_lengths,
                            val_sims_used=val_sims_used, Yval=Yval, thr_val=thr, c=c)

        if best['params'] is None:
            print(f"  no usable config; skip"); continue
        WIN, GAMMA, NU = best['params']
        scaler = best['scaler']; ocs = best['model']; c_best = best['c']
        print(f"  BEST: win={WIN} gamma={GAMMA} nu={NU} c={c_best}", flush=True)

        # Per-sim val rows
        for d, yh, y in zip(best['val_sims_used'],
                            split_by_lengths((best['val_scores'] <= best['thr_val']).astype(int), best['val_lengths']),
                            split_by_lengths(best['Yval'], best['val_lengths'])):
            if len(yh) == 0:
                continue
            s_i, sp_i = sens_spec(y, yh)
            rows_val.append(dict(split="val", model="ocsvm", signal=S, sim=d['sim'],
                                 window=WIN, gamma=GAMMA, nu=NU, contamination=c_best, thr=best['thr_val'],
                                 sens=s_i, spec=sp_i, pod=pod_anyhit(yh, y),
                                 timeliness=timeliness_single(yh, y), n_points=len(yh)))

        # Test on ABS window
        Xte, Yte, test_lengths, test_sims_used = build_window(
            test_sims, WIN, ABS_START, ABS_END)
        if not len(Xte):
            print("  no test features; skip"); continue
        Xte_s = scaler.transform(Xte)
        test_scores = ocs.decision_function(Xte_s)
        thr_test = np.percentile(test_scores, c_best * 100)
        yhat_test = (test_scores <= thr_test).astype(int)
        A_list, ofs = [], 0
        for L in test_lengths:
            A_list.append(yhat_test[ofs:ofs + L]); ofs += L
        A = np.column_stack(A_list)
        O = np.column_stack([d['y'][ABS_START:ABS_END] for d in test_sims_used])

        cache_dir = f"score_cache/ocsvm_{MAG_TAG}"
        os.makedirs(cache_dir, exist_ok=True)
        pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
            f"{cache_dir}/ocsvm_alarms_signal_{S}.csv", index=False)
        pd.DataFrame(O, columns=[f"sim_{i}" for i in range(O.shape[1])]).to_csv(
            f"{cache_dir}/ocsvm_outbreaks_signal_{S}.csv", index=False)

        # Cache continuous scores
        test_score_mat_cols = []
        _ofs = 0
        for _L in test_lengths:
            test_score_mat_cols.append(test_scores[_ofs:_ofs + _L])
            _ofs += _L
        test_score_mat = np.column_stack(test_score_mat_cols)
        pd.DataFrame(test_score_mat,
                     columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
                     ).to_csv(f"{cache_dir}/ocsvm_test_scores_signal_{S}.csv", index=False)
        val_score_mat_cols = []
        _ofs = 0
        for _L in best['val_lengths']:
            val_score_mat_cols.append(best['val_scores'][_ofs:_ofs + _L])
            _ofs += _L
        val_score_mat = np.column_stack(val_score_mat_cols)
        pd.DataFrame(val_score_mat,
                     columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
                     ).to_csv(f"{cache_dir}/ocsvm_val_scores_signal_{S}.csv", index=False)
        if MAG_TAG == 'small':
            pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
                f"ocsvm_alarms_signal_{S}.csv", index=False)
            pd.DataFrame(test_score_mat,
                         columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
                         ).to_csv(f"ocsvm_test_scores_signal_{S}.csv", index=False)
            pd.DataFrame(val_score_mat,
                         columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
                         ).to_csv(f"ocsvm_val_scores_signal_{S}.csv", index=False)

        O_full_test = np.stack([d['y'] for d in test_sims_used], axis=1)
        sens_R = compute_sensitivity_R(A, O_full_test)
        spec_R = compute_specificity_R(A, O_full_test, IDX_RANGE)
        fpr_R  = compute_fpr_R(A, O_full_test, IDX_RANGE)
        pod_R  = compute_pod_R(A, O_full_test)
        tim_R  = compute_timeliness_R(A, O_full_test)
        print(f"  R-COMPARATOR -> Sens={sens_R:.3f} Spec={spec_R:.3f} FPR={fpr_R:.3f} "
              f"POD={pod_R:.3f} Tim={tim_R:.3f}", flush=True)

        for d, yh, ytrue in zip(test_sims_used,
                                split_by_lengths(yhat_test, test_lengths),
                                [d['y'][ABS_START:ABS_END] for d in test_sims_used]):
            s_i, sp_i = sens_spec(ytrue, yh)
            rows_test.append(dict(split="test", model="ocsvm", signal=S, sim=d['sim'],
                                  window=WIN, gamma=GAMMA, nu=NU, contamination=c_best, thr=thr_test,
                                  sens=s_i, spec=sp_i, pod=pod_anyhit(yh, ytrue),
                                  timeliness=timeliness_single(yh, ytrue), n_points=len(yh)))

        summary_all[S] = dict(sensitivity=sens_R, specificity=spec_R, fpr=fpr_R,
                              pod=pod_R, timeliness=tim_R, window=WIN,
                              gamma=GAMMA, nu=NU, contamination=c_best)

    if summary_all:
        df = pd.DataFrame.from_dict(summary_all, orient="index")
        print("\n=== SUMMARY (OCSVM) ===\n", df, flush=True)
        print("\nMeans:\n", df.mean(numeric_only=True))
        df.to_csv("OCSVM_Tuned_all_days_per_sig.csv")
        os.makedirs("results", exist_ok=True)
        df.to_csv(f"results/OCSVM_Tuned_per_sig_big_{MAG_TAG}.csv")
    if rows_val:
        pd.DataFrame(rows_val).to_csv("OCSVM_per_sim_val.csv", index=False)
    if rows_test:
        pd.DataFrame(rows_test).to_csv("OCSVM_per_sim_test.csv", index=False)
