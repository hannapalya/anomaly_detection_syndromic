#!/usr/bin/env python3
"""
Rate-change detector, but on Pearson residuals from a seasonal NB baseline.

The original RateChange detector (run_ratechange.py) uses EWMA of raw counts.
Counts are heavily seasonal, so the EWMA-z is dominated by seasonal cycles.

This variant:
  1. Fits a Negative-Binomial seasonal GLM baseline on outbreak-free training-period
     days (reusing CUSUMDetector.fit_baseline).
  2. Computes Pearson residual: r_t = (y_t - mu_t) / sqrt(dispersion * mu_t).
  3. Applies EWMA-z to the residuals: z_t = (r_t - EWMA(r))/sqrt(var(EWMA)+eps).
  4. Score = -clip(z_t, 0, inf) (higher = more normal; the percentile-tuning
     convention is "score <= threshold => alarm").

Expected: much better than raw EWMA because seasonal noise is removed.
"""

import time
import numpy as np
import pandas as pd

from anom_common import (
    load_data, split_60_20_20,
    sens_spec, pod_anyhit, timeliness_single, split_by_lengths,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    SIGNALS, TRAIN_DAYS, VALID_DAYS, RNG_STATE,
    ABS_START, ABS_END, WIN_LEN, IDX_RANGE,
    SPEC_TARGET, W_SENS, W_SPEC,
    MAG_TAG,
)
from run_cusum import CUSUMDetector


class RateChangeResidualDetector:
    def __init__(self, alpha=0.2, eps=1e-6):
        self.alpha = alpha
        self.eps = eps

    def _ewma(self, x):
        mu = np.zeros_like(x, dtype=float)
        if len(x) == 0:
            return mu
        mu[0] = x[0]
        for t in range(1, len(x)):
            mu[t] = self.alpha * x[t-1] + (1 - self.alpha) * mu[t-1]
        return mu

    def decision_function_series(self, d):
        cu = CUSUMDetector(k=0.5)
        cu.fit_baseline(d['x'], d['y'])
        mu = cu.baseline(len(d['x']))
        disp = cu.dispersion_
        x = np.asarray(d['x'], dtype=float)
        # Pearson residual
        r = (x - mu) / np.sqrt(disp * mu + self.eps)
        # EWMA-z on residuals
        mu_r = self._ewma(r)
        # Use a robust variance proxy (1.0 since residuals are already normalised)
        z = (r - mu_r) / 1.0
        z_pos = np.clip(z, 0.0, None)
        return -z_pos


if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)
    summary_all, rows_val, rows_test = {}, [], []
    ALPHA = 0.2

    for S in SIGNALS:
        print(f"\n--- Signal {S} (RateChange-residual) ---", flush=True)
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
        rc = RateChangeResidualDetector(alpha=ALPHA)

        t0 = time.time()
        val_scores_concat, val_lengths, Yval_concat = [], [], []
        for d in val_sims:
            full = rc.decision_function_series(d)
            tail = full[-VALID_DAYS:]
            val_scores_concat.extend(tail); val_lengths.append(VALID_DAYS)
            Yval_concat.extend(d['y'][-VALID_DAYS:])
        val_scores_concat = np.asarray(val_scores_concat, dtype=float)
        Yval_concat = np.asarray(Yval_concat, dtype=int)
        c_best = tune_contamination_threshold(val_sims, val_lengths, val_scores_concat,
                                              spec_target=SPEC_TARGET,
                                              w_sens=W_SENS, w_spec=W_SPEC)
        thr_val = np.percentile(val_scores_concat, c_best * 100)
        print(f"  c={c_best} thr={thr_val:.3f} (val in {time.time()-t0:.1f}s)", flush=True)

        # Per-sim val rows
        yhat_v = (val_scores_concat <= thr_val).astype(int)
        for d, yh, y in zip(val_sims,
                            split_by_lengths(yhat_v, val_lengths),
                            split_by_lengths(Yval_concat, val_lengths)):
            if len(yh) == 0: continue
            s_i, sp_i = sens_spec(y, yh)
            rows_val.append(dict(split="val", model="ratechange_residual", signal=S, sim=d['sim'],
                                 alpha=ALPHA, contamination=c_best, thr=thr_val,
                                 sens=s_i, spec=sp_i, pod=pod_anyhit(yh, y),
                                 timeliness=timeliness_single(yh, y), n_points=len(yh)))

        # Test on ABS window
        per_sim_scores, per_sim_labels = [], []
        for d in test_sims:
            full = rc.decision_function_series(d)
            if len(full) < ABS_END: continue
            per_sim_scores.append(full[ABS_START:ABS_END])
            per_sim_labels.append(d['y'][ABS_START:ABS_END].astype(int))
        if not per_sim_scores:
            continue
        scores_concat = np.concatenate(per_sim_scores)
        thr_test = np.percentile(scores_concat, c_best * 100)
        yhat_concat = (scores_concat <= thr_test).astype(int)
        A_list, ofs = [], 0
        for y_win in per_sim_labels:
            A_list.append(yhat_concat[ofs:ofs + WIN_LEN]); ofs += WIN_LEN
        A = np.column_stack(A_list)
        O = np.column_stack(per_sim_labels)

        pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
            f"ratechange_residual_alarms_signal_{S}.csv", index=False)
        pd.DataFrame(O, columns=[f"sim_{i}" for i in range(O.shape[1])]).to_csv(
            f"ratechange_residual_outbreaks_signal_{S}.csv", index=False)

        O_full_test = np.stack([d['y'] for d in test_sims], axis=1)
        sens_R = compute_sensitivity_R(A, O_full_test)
        spec_R = compute_specificity_R(A, O_full_test, IDX_RANGE)
        fpr_R  = compute_fpr_R(A, O_full_test, IDX_RANGE)
        pod_R  = compute_pod_R(A, O_full_test)
        tim_R  = compute_timeliness_R(A, O_full_test)
        print(f"  R-COMPARATOR -> Sens={sens_R:.3f} Spec={spec_R:.3f} FPR={fpr_R:.3f} "
              f"POD={pod_R:.3f} Tim={tim_R:.3f}", flush=True)

        for d, yh, ytrue in zip(test_sims, A_list, per_sim_labels):
            s_i, sp_i = sens_spec(ytrue, yh)
            rows_test.append(dict(split="test", model="ratechange_residual", signal=S, sim=d['sim'],
                                  alpha=ALPHA, contamination=c_best, thr=thr_test,
                                  sens=s_i, spec=sp_i, pod=pod_anyhit(yh, ytrue),
                                  timeliness=timeliness_single(yh, ytrue), n_points=len(yh)))

        summary_all[S] = dict(sensitivity=sens_R, specificity=spec_R, fpr=fpr_R,
                              pod=pod_R, timeliness=tim_R, contamination=c_best, alpha=ALPHA)

    if summary_all:
        df = pd.DataFrame.from_dict(summary_all, orient="index")
        print("\n=== SUMMARY (RateChange-residual) ===\n", df, flush=True)
        print("\nMeans:\n", df.mean(numeric_only=True))
        df.to_csv(f"results/RateChangeResidual_per_sig_big_{MAG_TAG}.csv")
    if rows_val:
        pd.DataFrame(rows_val).to_csv("RateChangeResidual_per_sim_val.csv", index=False)
    if rows_test:
        pd.DataFrame(rows_test).to_csv("RateChangeResidual_per_sim_test.csv", index=False)
