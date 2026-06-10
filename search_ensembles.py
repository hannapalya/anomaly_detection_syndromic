#!/usr/bin/env python3
"""
Combinatorial ensemble search over cached alarm/score CSVs.

Tries all 2/3/4-method subsets of available detectors with:
  - OR-vote (any detector fires)
  - Majority vote (≥ceil(N/2) detectors fire)
  - ≥2-of-N vote

Reports R-comparator metrics. Flags cheap vs expensive combos.
"""

import os
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
)

# ---------- Detector definitions ----------

DETECTORS = {
    # name: (alarm_csv_pattern, score_csv_patterns, higher_is_normal, cost_tier)
    # alarm_csv_pattern: f-string with {S} for signal number. If exists, use directly.
    # score_csv_patterns: (val_pat, test_pat) if we need to threshold from scores.
    'IF':        dict(alarm='isolation_forest_alarms_signal_{S}.csv',
                      outbreak='isolation_forest_outbreaks_signal_{S}.csv',
                      cost='medium'),
    'KNN':       dict(alarm='knn_alarms_signal_{S}.csv',
                      outbreak='knn_outbreaks_signal_{S}.csv',
                      cost='medium'),
    'OCSVM':     dict(alarm='ocsvm_alarms_signal_{S}.csv',
                      outbreak='ocsvm_outbreaks_signal_{S}.csv',
                      cost='medium'),
    'LOF':       dict(alarm='lof_alarms_signal_{S}.csv',
                      outbreak='lof_outbreaks_signal_{S}.csv',
                      cost='expensive'),
    'CUSUM':     dict(alarm='cusum_alarms_signal_{S}.csv',
                      outbreak='cusum_outbreaks_signal_{S}.csv',
                      cost='cheap'),
    'Farrington':dict(alarm='farrington_custom_alarms_signal_{S}.csv',
                      outbreak='farrington_custom_outbreaks_signal_{S}.csv',
                      cost='cheap'),
    'RateChange':dict(alarm='ratechange_residual_alarms_signal_{S}.csv',
                      outbreak='ratechange_residual_outbreaks_signal_{S}.csv',
                      cost='cheap'),
    # Score-based — need to threshold from cached val/test scores
    'LSTM-AE':   dict(val_scores='lstm_ae_val_scores_signal_{S}.csv',
                      test_scores='lstm_ae_test_scores_signal_{S}.csv',
                      higher_is_normal=False, cost='expensive'),
    'NB-HMM':    dict(val_scores='nbhmm_val_scores_signal_{S}.csv',
                      test_scores='nbhmm_test_scores_signal_{S}.csv',
                      higher_is_normal=True, cost='cheap'),
    'BOCPD':     dict(val_scores='bocpd_resid_val_scores_signal_{S}.csv',
                      test_scores='bocpd_resid_test_scores_signal_{S}.csv',
                      higher_is_normal=True, cost='cheap'),
    'VAE':       dict(val_scores='cloud_out/vae_small/vae_val_scores_signal_{S}.csv',
                      test_scores='cloud_out/vae_small/vae_test_scores_signal_{S}.csv',
                      higher_is_normal=True, cost='expensive'),
}

COST_ORDER = {'cheap': 0, 'medium': 1, 'expensive': 2}


def load_alarm_outbreak_pair(det_info, S, val_sims, test_sims):
    """Load or compute test-window alarm matrix for one detector, one signal.
    Returns (A, O) each shape (WIN_LEN, n_test_sims), or (None, None)."""

    if 'alarm' in det_info:
        alarm_path = det_info['alarm'].format(S=S)
        outbreak_path = det_info['outbreak'].format(S=S)
        if not os.path.exists(alarm_path) or not os.path.exists(outbreak_path):
            return None, None
        A = pd.read_csv(alarm_path).to_numpy(dtype=int)
        O = pd.read_csv(outbreak_path).to_numpy(dtype=int)
        return A, O

    elif 'val_scores' in det_info:
        val_path = det_info['val_scores'].format(S=S)
        test_path = det_info['test_scores'].format(S=S)
        if not os.path.exists(val_path) or not os.path.exists(test_path):
            return None, None
        v = pd.read_csv(val_path).to_numpy()
        t = pd.read_csv(test_path).to_numpy()
        if not det_info.get('higher_is_normal', True):
            v = -v; t = -t
        val_flat = v.flatten(order="F")
        val_lengths = [v.shape[0]] * v.shape[1]
        c = tune_contamination_threshold(val_sims, val_lengths, val_flat,
                                         spec_target=SPEC_TARGET, w_sens=W_SENS, w_spec=W_SPEC)
        thr = np.percentile(val_flat, c * 100)
        A = (t <= thr).astype(int)
        # Outbreak from test sims
        O = np.column_stack([d['y'][ABS_START:ABS_END] for d in test_sims])
        return A, O

    return None, None


def metrics(A, O_full):
    return dict(
        sensitivity=compute_sensitivity_R(A, O_full),
        specificity=compute_specificity_R(A, O_full, IDX_RANGE),
        fpr=compute_fpr_R(A, O_full, IDX_RANGE),
        pod=compute_pod_R(A, O_full),
        timeliness=compute_timeliness_R(A, O_full),
    )


if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)

    # Step 1: Load all alarm matrices per signal
    print("Loading alarm matrices for all detectors × signals...", flush=True)
    # alarms[det_name][S] = (A, O)
    alarms = defaultdict(dict)
    available_per_signal = defaultdict(set)

    for S in SIGNALS:
        Xsig, Ysig = load_data(S)
        sims = []
        for i, c in enumerate(Xsig.columns):
            x = Xsig[c].to_numpy(np.float32, copy=False)
            y = Ysig[c].to_numpy(np.int32, copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{i}", sim_idx=i))
        if not sims:
            continue
        train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
        O_full_test = np.stack([d['y'] for d in test_sims], axis=1)

        for dname, dinfo in DETECTORS.items():
            A, O = load_alarm_outbreak_pair(dinfo, S, val_sims, test_sims)
            if A is not None:
                alarms[dname][S] = (A, O_full_test)
                available_per_signal[S].add(dname)

    # Which detectors cover all 16 signals?
    full_coverage = [d for d in DETECTORS if len(alarms[d]) == 16]
    partial_coverage = [d for d in DETECTORS if 0 < len(alarms[d]) < 16]
    print(f"\nFull coverage (16/16): {full_coverage}")
    print(f"Partial coverage: {[(d, len(alarms[d])) for d in partial_coverage]}")

    # Step 2: Individual detector metrics
    print("\n=== INDIVIDUAL DETECTORS ===")
    indiv_rows = []
    for dname in sorted(DETECTORS.keys()):
        if not alarms[dname]:
            continue
        sig_metrics = []
        for S in SIGNALS:
            if S not in alarms[dname]:
                continue
            A, O_full = alarms[dname][S]
            sig_metrics.append(metrics(A, O_full))
        if sig_metrics:
            mean_m = {k: np.nanmean([m[k] for m in sig_metrics]) for k in sig_metrics[0]}
            mean_m['method'] = dname
            mean_m['cost'] = DETECTORS[dname]['cost']
            mean_m['n_signals'] = len(sig_metrics)
            indiv_rows.append(mean_m)
    df_indiv = pd.DataFrame(indiv_rows).sort_values('sensitivity', ascending=False)
    print(df_indiv[['method', 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness', 'cost', 'n_signals']].to_string(index=False))

    # Step 3: Enumerate ensemble combinations
    # Use only detectors with full coverage for fair comparison
    candidates = sorted(full_coverage)
    print(f"\nCandidates for ensemble search (full coverage): {candidates}")

    ensemble_rows = []

    for size in [2, 3, 4, 5]:
        for combo in itertools.combinations(candidates, size):
            combo_names = list(combo)
            combo_cost = max(COST_ORDER[DETECTORS[d]['cost']] for d in combo_names)
            cost_label = {0: 'cheap', 1: 'medium', 2: 'expensive'}[combo_cost]

            # For each signal, compute ensemble alarms
            sig_metrics_or = []
            sig_metrics_maj = []
            sig_metrics_ge2 = []

            for S in SIGNALS:
                if any(S not in alarms[d] for d in combo_names):
                    continue
                As = [alarms[d][S][0] for d in combo_names]
                O_full = alarms[combo_names[0]][S][1]

                # OR vote
                A_or = np.maximum.reduce(As)
                sig_metrics_or.append(metrics(A_or, O_full))

                # ≥2-of-N vote
                A_sum = np.sum(As, axis=0)
                A_ge2 = (A_sum >= 2).astype(int)
                sig_metrics_ge2.append(metrics(A_ge2, O_full))

                # Majority vote (≥ceil(N/2))
                majority_thr = int(np.ceil(len(As) / 2))
                A_maj = (A_sum >= majority_thr).astype(int)
                sig_metrics_maj.append(metrics(A_maj, O_full))

            if not sig_metrics_or:
                continue

            for strategy, sig_m in [('OR', sig_metrics_or), ('≥2', sig_metrics_ge2),
                                     ('majority', sig_metrics_maj)]:
                mean_m = {k: np.nanmean([m[k] for m in sig_m]) for k in sig_m[0]}
                mean_m['combo'] = '+'.join(combo_names)
                mean_m['size'] = size
                mean_m['strategy'] = strategy
                mean_m['cost'] = cost_label
                mean_m['has_noufaily'] = 'Noufaily' in combo_names if 'Noufaily' in candidates else False
                mean_m['has_gpu'] = any(DETECTORS[d]['cost'] == 'expensive' for d in combo_names)
                ensemble_rows.append(mean_m)

    df_ens = pd.DataFrame(ensemble_rows)

    # Step 4: Report best ensembles
    print("\n" + "="*90)
    print("TOP 20 ENSEMBLES BY SENSITIVITY (no Noufaily, no GPU methods)")
    print("="*90)
    mask = (~df_ens['has_noufaily']) & (~df_ens['has_gpu'])
    top = df_ens[mask].sort_values('sensitivity', ascending=False).head(20)
    print(top[['combo', 'strategy', 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness', 'cost', 'size']].to_string(index=False))

    print("\n" + "="*90)
    print("TOP 20 ENSEMBLES BY SENSITIVITY (no Noufaily, allow all methods)")
    print("="*90)
    mask2 = ~df_ens['has_noufaily']
    top2 = df_ens[mask2].sort_values('sensitivity', ascending=False).head(20)
    print(top2[['combo', 'strategy', 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness', 'cost', 'size']].to_string(index=False))

    print("\n" + "="*90)
    print("BEST CHEAP ENSEMBLES (cost=cheap only, no Noufaily)")
    print("="*90)
    mask3 = (~df_ens['has_noufaily']) & (df_ens['cost'] == 'cheap')
    if mask3.any():
        top3 = df_ens[mask3].sort_values('sensitivity', ascending=False).head(15)
        print(top3[['combo', 'strategy', 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness', 'size']].to_string(index=False))
    else:
        print("No all-cheap combos found (need score-based detectors like NB-HMM, BOCPD)")

    print("\n" + "="*90)
    print("BEST MEDIUM-COST ENSEMBLES (no GPU, no Noufaily)")
    print("="*90)
    mask4 = (~df_ens['has_noufaily']) & (~df_ens['has_gpu']) & (df_ens['sensitivity'] > 0.5)
    if mask4.any():
        top4 = df_ens[mask4].sort_values('sensitivity', ascending=False).head(20)
        print(top4[['combo', 'strategy', 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness', 'cost', 'size']].to_string(index=False))

    # Pareto frontier: best sens for each FPR bucket
    print("\n" + "="*90)
    print("PARETO FRONTIER: best sensitivity at each FPR level (no Noufaily, no GPU)")
    print("="*90)
    mask5 = (~df_ens['has_noufaily']) & (~df_ens['has_gpu'])
    df_f = df_ens[mask5].copy()
    df_f['fpr_bucket'] = pd.cut(df_f['fpr'], bins=[0, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10, 0.15, 1.0])
    pareto = df_f.sort_values('sensitivity', ascending=False).groupby('fpr_bucket').first().reset_index()
    pareto = pareto.sort_values('fpr')
    print(pareto[['fpr_bucket', 'combo', 'strategy', 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness', 'cost']].to_string(index=False))

    # Also show: for each size, best OR combo without Noufaily/GPU
    print("\n" + "="*90)
    print("BEST OR-VOTE PER SIZE (no Noufaily, no GPU)")
    print("="*90)
    mask6 = (~df_ens['has_noufaily']) & (~df_ens['has_gpu']) & (df_ens['strategy'] == 'OR')
    for sz in [2, 3, 4, 5]:
        sub = df_ens[mask6 & (df_ens['size'] == sz)].sort_values('sensitivity', ascending=False).head(3)
        if not sub.empty:
            print(f"\n--- Size {sz} ---")
            print(sub[['combo', 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness', 'cost']].to_string(index=False))

    # Save full results
    df_ens.to_csv('results/ensemble_search_full.csv', index=False)
    print(f"\nSaved {len(df_ens)} ensemble configurations to results/ensemble_search_full.csv")
