#!/usr/bin/env python3
"""
Fully-unsupervised OR-vote ensemble.

Combines per-day alarms from multiple unsupervised detectors, each individually
tuned for spec >= 0.97 via anom_common.tune_contamination_threshold. Then
OR-votes across various detector subsets and computes R-comparator metrics
on the test ABS window.

Detectors used (truly unsupervised at training):
  - IF (per-signal-tuned, from cache)
  - LSTM-AE (production cache)
  - NB-HMM (cache)
  - BOCPD-residual (cache)
  - CUSUM (live; NB seasonal baseline)
  - Noufaily-quantile (live, built here): per-day -log survival under NB(mu_t,alpha)

All caches use convention "higher = more normal". CUSUM produced live, same convention.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import nbinom

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
    MAG_TAG,
)
from run_cusum import CUSUMDetector


# ---------- Noufaily-quantile detector (per-day NB upper-tail survival) ----------

def noufaily_score(d):
    """Return per-day score for full series: higher = more normal.
    Uses NB seasonal baseline fit on outbreak-free training-period days,
    predicts mu_t, alpha. Score = log P(Y >= y_t | NB) (survival probability)."""
    cu = CUSUMDetector(k=0.5)              # only used for its baseline-fit & params
    cu.fit_baseline(d['x'], d['y'])
    mu = cu.baseline(len(d['x']))
    alpha = cu.dispersion_                  # variance multiplier; not directly NB shape
    # scipy NB parameterization: NB(n=r, p) with mean r*(1-p)/p, var = mean*(1+mean/r).
    # For NB(mean=mu, var=mu*(1+mu/r)) with overdispersion phi: var = mu + mu^2/r.
    # Match Pearson dispersion alpha to: phi := alpha (since var/mu ≈ alpha for CUSUM).
    # If dispersion alpha >= 1: r = mu / max(alpha - 1, 1e-3). If alpha <= 1: use Poisson approx.
    if alpha > 1.001:
        r = mu / (alpha - 1.0)
        r = np.clip(r, 1e-3, 1e8)
        p = r / (r + mu)
        # Survival = P(Y >= y) = 1 - cdf(y-1).
        y = np.asarray(d['x'], dtype=int)
        log_s = np.log(np.clip(nbinom.sf(y - 1, n=r, p=p), 1e-12, 1.0))
    else:
        from scipy.stats import poisson
        y = np.asarray(d['x'], dtype=int)
        log_s = np.log(np.clip(poisson.sf(y - 1, mu=mu), 1e-12, 1.0))
    # higher log_s = more normal (high probability of seeing this value or higher).
    return log_s.astype(np.float32)


# ---------- Generic load/tune ----------

def alarms_from_cache(val_csv, test_csv, val_sims, test_sims, val_lengths, test_lengths,
                     higher_is_normal=True):
    v = pd.read_csv(val_csv).to_numpy()
    t = pd.read_csv(test_csv).to_numpy()
    if not higher_is_normal:
        v = -v; t = -t
    val_flat = v.flatten(order="F")
    c = tune_contamination_threshold(val_sims, val_lengths, val_flat,
                                     spec_target=SPEC_TARGET, w_sens=W_SENS, w_spec=W_SPEC)
    thr = np.percentile(val_flat, c * 100)
    A = (t <= thr).astype(int)
    return A, c, thr


def alarms_from_live(scorer_full_series, sims_val, sims_test,
                     val_lengths, test_lengths):
    """scorer_full_series(d) returns higher=normal per-day full series."""
    vcols, tcols = [], []
    for d in sims_val:
        s = scorer_full_series(d)
        vcols.append(s[len(d['x']) - VALID_DAYS:])
    for d in sims_test:
        s = scorer_full_series(d)
        tcols.append(s[ABS_START:ABS_END])
    v = np.column_stack(vcols); t = np.column_stack(tcols)
    val_flat = v.flatten(order="F")
    c = tune_contamination_threshold(sims_val, val_lengths, val_flat,
                                     spec_target=SPEC_TARGET, w_sens=W_SENS, w_spec=W_SPEC)
    thr = np.percentile(val_flat, c * 100)
    A = (t <= thr).astype(int)
    return A, c, thr


def metrics(A, O_full):
    return dict(
        sensitivity=compute_sensitivity_R(A, O_full),
        specificity=compute_specificity_R(A, O_full, IDX_RANGE),
        fpr=compute_fpr_R(A, O_full, IDX_RANGE),
        pod=compute_pod_R(A, O_full),
        timeliness=compute_timeliness_R(A, O_full),
    )


# ---------- MAIN ----------

if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)

    rows_per_method = []   # (method, signal, metrics dict)

    for S in SIGNALS:
        print(f"\n=== Signal {S} ===", flush=True)
        Xsig, Ysig = load_data(S)
        sims = []
        for i, c in enumerate(Xsig.columns):
            x = Xsig[c].to_numpy(np.float32, copy=False)
            y = Ysig[c].to_numpy(np.int32,  copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{i}", sim_idx=i))
        if not sims:
            continue
        train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
        val_lengths = [VALID_DAYS] * len(val_sims)
        test_lengths = [WIN_LEN] * len(test_sims)
        O_full_test = np.stack([d['y'] for d in test_sims], axis=1)

        det_alarms = {}

        # IF cache
        if os.path.exists(f"if_tuned_val_scores_signal_{S}.csv"):
            A, c, thr = alarms_from_cache(f"if_tuned_val_scores_signal_{S}.csv",
                                          f"if_tuned_test_scores_signal_{S}.csv",
                                          val_sims, test_sims, val_lengths, test_lengths,
                                          higher_is_normal=True)
            det_alarms['if'] = A

        # LSTM-AE cache: cache stores reconstruction MSE (higher=anomalous)
        if os.path.exists(f"lstm_ae_val_scores_signal_{S}.csv"):
            A, c, thr = alarms_from_cache(f"lstm_ae_val_scores_signal_{S}.csv",
                                          f"lstm_ae_test_scores_signal_{S}.csv",
                                          val_sims, test_sims, val_lengths, test_lengths,
                                          higher_is_normal=False)
            det_alarms['lstm'] = A

        # NB-HMM cache: saved as -posterior (higher = normal)
        if os.path.exists(f"nbhmm_val_scores_signal_{S}.csv"):
            A, c, thr = alarms_from_cache(f"nbhmm_val_scores_signal_{S}.csv",
                                          f"nbhmm_test_scores_signal_{S}.csv",
                                          val_sims, test_sims, val_lengths, test_lengths,
                                          higher_is_normal=True)
            det_alarms['nbhmm'] = A

        # BOCPD-residual cache (higher = normal)
        if os.path.exists(f"bocpd_resid_val_scores_signal_{S}.csv"):
            A, c, thr = alarms_from_cache(f"bocpd_resid_val_scores_signal_{S}.csv",
                                          f"bocpd_resid_test_scores_signal_{S}.csv",
                                          val_sims, test_sims, val_lengths, test_lengths,
                                          higher_is_normal=True)
            det_alarms['bocpd_resid'] = A

        # CUSUM live (higher = normal)
        cu_master = CUSUMDetector(k=0.5)
        def cusum_scorer(d):
            cu = CUSUMDetector(k=0.5); cu.fit_baseline(d['x'], d['y'])
            return cu.decision_function_series(d['x'])
        A, c, thr = alarms_from_live(cusum_scorer, val_sims, test_sims, val_lengths, test_lengths)
        det_alarms['cusum'] = A

        # Noufaily-quantile live (higher = normal)
        t0 = time.time()
        A, c, thr = alarms_from_live(noufaily_score, val_sims, test_sims, val_lengths, test_lengths)
        det_alarms['noufaily'] = A
        print(f"  Noufaily-quantile fit: {time.time()-t0:.1f}s", flush=True)

        # Individual metrics
        for name, A in det_alarms.items():
            m = metrics(A, O_full_test); m['method'] = name; m['signal'] = S
            rows_per_method.append(m)

        # OR subsets
        subsets = [
            ('OR_if_lstm_nbhmm',         ('if', 'lstm', 'nbhmm')),
            ('OR_if_nbhmm_noufaily',     ('if', 'nbhmm', 'noufaily')),
            ('OR_if_lstm_noufaily',      ('if', 'lstm', 'noufaily')),
            ('OR_if_lstm_nbhmm_noufaily', ('if', 'lstm', 'nbhmm', 'noufaily')),
            ('OR_all6',                  ('if', 'lstm', 'nbhmm', 'bocpd_resid', 'cusum', 'noufaily')),
            ('OR_strong3_fast',          ('noufaily', 'if', 'bocpd_resid')),
        ]
        for sname, keys in subsets:
            usable = [det_alarms[k] for k in keys if k in det_alarms]
            if len(usable) < 2:
                continue
            A_or = np.maximum.reduce(usable)
            m = metrics(A_or, O_full_test); m['method'] = sname; m['signal'] = S
            rows_per_method.append(m)

        print(f"  done {S}", flush=True)

    df = pd.DataFrame(rows_per_method)
    print("\n=== PER-METHOD MEANS ===")
    means = df.groupby('method').agg({
        'sensitivity': 'mean', 'specificity': 'mean',
        'fpr': 'mean', 'pod': 'mean', 'timeliness': 'mean',
    }).round(3)
    print(means)
    df.to_csv('Unsup_OR_per_signal.csv', index=False)
    means.reset_index().to_csv(f'results/Unsup_OR_summary_big_{MAG_TAG}.csv', index=False)
    print(f"\nSaved: results/Unsup_OR_summary_big_{MAG_TAG}.csv")
