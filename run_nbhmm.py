#!/usr/bin/env python3
"""
2-state Negative-Binomial Hidden Markov Model for syndromic surveillance.

States:
  0 = normal,  1 = outbreak.

Emissions:
  State 0: NB(mean = mu_0(t), dispersion = alpha_0)
    - mu_0(t) = seasonal NB-GLM baseline fit on outbreak-free TRAINING-period
      days (reuses CUSUMDetector.fit_baseline / .baseline).
    - alpha_0 = Pearson dispersion from CUSUM (cu.dispersion_).
  State 1: NB(mean = c * mu_0(t), dispersion = alpha_1 = alpha_0)
    - c is a free outbreak-elevation factor, picked from {1.5, 2.0, 3.0} on val.

Transitions (fixed):
  A = [[0.99, 0.01],
       [0.08, 0.92]]   ~ outbreaks last ~12 days on average
  pi = [0.95, 0.05]

Inference:
  Forward algorithm in log-space (online, as required for surveillance).
  Score per day = posterior P(state_t = 1 | y_{1:t}).

Outputs (per signal S):
  nbhmm_val_scores_signal_{S}.csv  shape (343, n_val_sims)  higher=NORMAL
  nbhmm_test_scores_signal_{S}.csv shape (343, n_test_sims) higher=NORMAL

Summary:
  results/NBHMM_per_sig_big_{MAG_TAG}.csv  with R-comparator test metrics.

Truly unsupervised: outbreak labels y are only used (a) to mask outbreak-free
training-period days when fitting the seasonal baseline (same as CUSUM /
BOCPD-residual) and (b) for evaluation. They are NOT used to fit the HMM
emissions or transitions.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.stats import nbinom
from scipy.special import logsumexp

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    sens_spec, pod_anyhit, timeliness_single,
    tune_contamination_threshold, split_by_lengths,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
    MAG_TAG,
)
from run_cusum import CUSUMDetector


# Hyper-parameters
C_GRID = [1.5, 2.0, 3.0]

# 2x2 transition matrix and initial state distribution
A_MAT = np.array([[0.99, 0.01],
                  [0.08, 0.92]], dtype=np.float64)
PI = np.array([0.95, 0.05], dtype=np.float64)
LOG_A = np.log(A_MAT)
LOG_PI = np.log(PI)


def nb_logpmf(y, mu, alpha):
    """NB log-pmf with mean=mu and dispersion alpha (NB2 parameterisation).

    scipy.stats.nbinom uses (n, p) with E[X] = n*(1-p)/p, Var = n*(1-p)/p^2.
    Setting n = alpha and p = alpha/(alpha+mu) gives:
        E[X] = mu,  Var = mu + mu^2/alpha
    where alpha = 1/dispersion in some conventions. Here we follow the spec:
    "scipy.stats.nbinom.logpmf(y, n=alpha, p=alpha/(alpha+mu))".
    """
    mu = np.maximum(mu, 1e-9)
    alpha = max(float(alpha), 1e-6)
    p = alpha / (alpha + mu)
    return nbinom.logpmf(y, n=alpha, p=p)


def _logsumexp2(a, b):
    """Fast logsumexp of two scalars (numerically stable)."""
    m = a if a > b else b
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def forward_posterior(y, mu0_seq, alpha0, c, alpha1=None):
    """Forward algorithm in log-space. Returns per-day P(state_t=1 | y_{1:t}).

    Vectorises emission log-pmfs over t up-front, then runs the 2-state
    forward recursion as a tight Python loop.
    """
    if alpha1 is None:
        alpha1 = alpha0
    y = np.asarray(y, dtype=np.int64)
    mu0_seq = np.asarray(mu0_seq, dtype=np.float64)
    mu0_seq = np.maximum(mu0_seq, 1e-9)
    mu1_seq = np.maximum(c * mu0_seq, 1e-9)
    a0 = max(float(alpha0), 1e-6)
    a1 = max(float(alpha1), 1e-6)

    # Vectorised NB log-pmfs across full time series
    p0 = a0 / (a0 + mu0_seq)
    p1 = a1 / (a1 + mu1_seq)
    log_b0_arr = nbinom.logpmf(y, n=a0, p=p0)
    log_b1_arr = nbinom.logpmf(y, n=a1, p=p1)

    T = len(y)
    posteriors = np.empty(T, dtype=np.float64)

    # Unpack constants for speed
    A00, A01 = LOG_A[0, 0], LOG_A[0, 1]
    A10, A11 = LOG_A[1, 0], LOG_A[1, 1]
    log_pi0, log_pi1 = LOG_PI[0], LOG_PI[1]

    # t = 0
    la0 = log_pi0 + log_b0_arr[0]
    la1 = log_pi1 + log_b1_arr[0]
    norm = _logsumexp2(la0, la1)
    posteriors[0] = np.exp(la1 - norm)
    la0 -= norm
    la1 -= norm

    for t in range(1, T):
        # next_log_alpha[0] = log_b0 + logsumexp(la0+A00, la1+A10)
        # next_log_alpha[1] = log_b1 + logsumexp(la0+A01, la1+A11)
        n0 = log_b0_arr[t] + _logsumexp2(la0 + A00, la1 + A10)
        n1 = log_b1_arr[t] + _logsumexp2(la0 + A01, la1 + A11)
        norm = _logsumexp2(n0, n1)
        posteriors[t] = np.exp(n1 - norm)
        la0 = n0 - norm
        la1 = n1 - norm

    return posteriors


def get_baseline(d, k=0.5):
    """Reuse CUSUMDetector to obtain mu_0(t) and dispersion alpha_0."""
    cu = CUSUMDetector(k=k)
    cu.fit_baseline(d['x'], d['y'])
    mu = cu.baseline(len(d['x']))
    alpha0 = max(float(cu.dispersion_), 1.0)
    # NB2 parameterisation: scipy's "n" (size) parameter; with NB GLM Pearson
    # dispersion phi, the size is approximately 1/phi if mu set; but we follow
    # the spec literally: pass dispersion straight through as alpha. This
    # gives Var = mu + mu^2/alpha (high alpha -> closer to Poisson).
    return mu.astype(np.float64), alpha0


def evaluate_signal(S, log_path=None, save_cache=True):
    rng = np.random.RandomState(RNG_STATE)

    # Walk prior signals to consume RNG so splits match run_stacked_meta /
    # cache_*_scores scripts.
    for prior in SIGNALS:
        if prior == S:
            break
        Xp, _ = load_data(prior)
        sims_p = []
        for i, c in enumerate(Xp.columns):
            x = Xp[c].to_numpy(np.float32, copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims_p.append(dict(x=x, y=None, sim=f"sig{prior}_sim{i}", sim_idx=i))
        if sims_p:
            split_60_20_20(sims_p, rng)

    Xsig, Ysig = load_data(S)
    sims = []
    for i, col in enumerate(Xsig.columns):
        x = Xsig[col].to_numpy(np.float32, copy=False)
        y = Ysig[col].to_numpy(np.int32, copy=False)
        if len(x) >= TRAIN_DAYS + VALID_DAYS:
            sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{i}", sim_idx=i))
    if not sims:
        print(f"[sig {S}] no complete sims; skip.", flush=True)
        return None

    train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
    print(f"[sig {S}] splits: {len(train_sims)} train, {len(val_sims)} val, {len(test_sims)} test", flush=True)

    O_full_val = np.stack([d['y'] for d in val_sims], axis=1)
    O_full_test = np.stack([d['y'] for d in test_sims], axis=1)

    # Pre-compute baselines once per sim (independent of c).
    val_baselines = []
    for d in val_sims:
        mu, a0 = get_baseline(d)
        val_baselines.append((mu, a0))
    test_baselines = []
    for d in test_sims:
        mu, a0 = get_baseline(d)
        test_baselines.append((mu, a0))

    best = dict(score=-1e9, c=None, val_block=None, test_block=None,
                contamination=None, val_metrics=None, test_metrics=None)

    for c_val in C_GRID:
        t0 = time.time()
        # Score val tails
        val_cols = []
        for d, (mu, a0) in zip(val_sims, val_baselines):
            post = forward_posterior(d['x'], mu, a0, c_val)
            val_cols.append(post[len(d['x']) - VALID_DAYS:])
        val_block = np.column_stack(val_cols)  # (343, n_val_sims) higher=ANOMALOUS

        # Score test ABS window
        test_cols = []
        for d, (mu, a0) in zip(test_sims, test_baselines):
            post = forward_posterior(d['x'], mu, a0, c_val)
            test_cols.append(post[ABS_START:ABS_END])
        test_block = np.column_stack(test_cols)

        # Convention for tune_contamination_threshold: higher = NORMAL.
        # Posterior is higher = anomaly, so negate.
        val_for_tune = (-val_block).flatten(order="F")
        val_lengths = [VALID_DAYS] * len(val_sims)
        c_thr = tune_contamination_threshold(val_sims, val_lengths, val_for_tune,
                                             spec_target=SPEC_TARGET,
                                             w_sens=W_SENS, w_spec=W_SPEC)

        thr_val_neg = np.percentile(val_for_tune, c_thr * 100)
        # Alarm when val_for_tune <= thr_val_neg, i.e., posterior >= -thr_val_neg
        thr_anom_v = -thr_val_neg
        yhat_v = (val_block.flatten(order="F") >= thr_anom_v).astype(int)
        A_list, ofs = [], 0
        for L in val_lengths:
            A_list.append(yhat_v[ofs:ofs + L]); ofs += L
        A_v = np.column_stack(A_list)
        sens_v = compute_sensitivity_R(A_v, O_full_val)
        spec_v = compute_specificity_R(A_v, O_full_val, IDX_RANGE)
        score_v = (W_SENS * (sens_v if not np.isnan(sens_v) else 0.0)
                   + W_SPEC * (spec_v if not np.isnan(spec_v) else 0.0))

        # Test eval at the validation-selected numeric threshold.
        test_for_tune = (-test_block).flatten(order="F")
        thr_test_neg = thr_val_neg
        thr_anom_t = -thr_test_neg
        yhat_t = (test_block.flatten(order="F") >= thr_anom_t).astype(int)
        A_tlist, ofs = [], 0
        for L in [WIN_LEN] * len(test_sims):
            A_tlist.append(yhat_t[ofs:ofs + L]); ofs += L
        A_t = np.column_stack(A_tlist)
        m = dict(sensitivity=compute_sensitivity_R(A_t, O_full_test),
                 specificity=compute_specificity_R(A_t, O_full_test, IDX_RANGE),
                 fpr=compute_fpr_R(A_t, O_full_test, IDX_RANGE),
                 pod=compute_pod_R(A_t, O_full_test),
                 timeliness=compute_timeliness_R(A_t, O_full_test))
        elapsed = time.time() - t0
        print(f"  c={c_val} cont={c_thr:.3f}  val sens={sens_v:.3f} spec={spec_v:.3f}  "
              f"test sens={m['sensitivity']:.3f} tim={m['timeliness']:.3f} "
              f"({elapsed:.1f}s)", flush=True)

        if score_v > best['score']:
            best.update(score=score_v, c=c_val, val_block=val_block,
                        test_block=test_block, contamination=c_thr,
                        val_metrics=dict(sens=sens_v, spec=spec_v),
                        test_metrics=m)

    if save_cache:
        # Convention: higher = MORE NORMAL -> negate the posterior.
        import os as _os
        cache_dir = f"score_cache/nbhmm_{MAG_TAG}"
        _os.makedirs(cache_dir, exist_ok=True)
        val_csv = f"{cache_dir}/nbhmm_val_scores_signal_{S}.csv"
        test_csv = f"{cache_dir}/nbhmm_test_scores_signal_{S}.csv"
        pd.DataFrame(-best['val_block'],
                     columns=[f"sim_{i}" for i in range(best['val_block'].shape[1])]
                     ).to_csv(val_csv, index=False)
        pd.DataFrame(-best['test_block'],
                     columns=[f"sim_{i}" for i in range(best['test_block'].shape[1])]
                     ).to_csv(test_csv, index=False)
        # Backward-compat flat path for small magnitude
        if MAG_TAG == 'small':
            pd.DataFrame(-best['val_block'],
                         columns=[f"sim_{i}" for i in range(best['val_block'].shape[1])]
                         ).to_csv(f"nbhmm_val_scores_signal_{S}.csv", index=False)
            pd.DataFrame(-best['test_block'],
                         columns=[f"sim_{i}" for i in range(best['test_block'].shape[1])]
                         ).to_csv(f"nbhmm_test_scores_signal_{S}.csv", index=False)
        print(f"[sig {S}] BEST c={best['c']}  saved {val_csv}, {test_csv}", flush=True)

    if log_path:
        with open(log_path, "a") as f:
            f.write(f"sig {S} best c={best['c']} contam={best['contamination']:.3f} "
                    f"test sens={best['test_metrics']['sensitivity']:.3f} "
                    f"tim={best['test_metrics']['timeliness']:.3f}\n")

    return dict(signal=S, c=best['c'],
                contamination=best['contamination'],
                **best['test_metrics'])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=str, default=None,
                    help="Comma list, e.g. '1,2,3'. Default: all SIGNALS.")
    ap.add_argument("--log", type=str, default="nbhmm_cache.log")
    args = ap.parse_args()
    target = SIGNALS if args.signals is None else [int(s) for s in args.signals.split(",")]
    print(f"NB-HMM: signals {target}", flush=True)
    open(args.log, "w").write(f"start: {target}\n")

    rows = []
    for S in target:
        try:
            r = evaluate_signal(S, args.log)
            if r:
                rows.append(r)
        except Exception as e:
            import traceback
            print(f"[sig {S}] FAILED: {e}", flush=True)
            traceback.print_exc()

    if rows:
        df = pd.DataFrame(rows, columns=["signal", "c", "sensitivity",
                                          "specificity", "fpr", "pod",
                                          "timeliness", "contamination"])
        print("\n=== NB-HMM (per signal) ===")
        print(df)
        means = df.mean(numeric_only=True)
        print("\nMeans:", means.to_dict())
        os.makedirs("results", exist_ok=True)
        df.to_csv(f"results/NBHMM_per_sig_big_{MAG_TAG}.csv", index=False)
        print(f"Saved: results/NBHMM_per_sig_big_{MAG_TAG}.csv", flush=True)
    print("ALL DONE.", flush=True)
