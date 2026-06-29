#!/usr/bin/env python3
"""
POD + TIMELINESS joint analysis for early-warning surveillance.

The current timeliness metric conflates:
  - LATENESS when detected (relative position within outbreak)
  - MISS PENALTY (each missed outbreak contributes 1.0)

For early-warning operational design we need to look at these jointly:
  - POD (event-level recall) — higher is better
  - Days-to-detection (absolute days from outbreak start to first alarm) — lower is better
  - Timeliness (R-comparator metric) — lower is better

This analysis produces:
  1. Per-method, per-signal: POD, conditional-timeliness, mean-days-to-detection
  2. POD × Timeliness Pareto frontiers (across methods, across signals)
  3. Decomposition of timeliness into lateness vs miss penalty
  4. Earliest-detector identification per signal
  5. Ensemble inheritance: does OR-vote inherit the EARLIEST member's timing?
"""

import os
import numpy as np
import pandas as pd
from collections import defaultdict

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC, DAYS, YEARS,
    MAG_TAG,
)

# For non-small magnitudes the alarm matrices / scores live in a magnitude-specific
# input directory (staged from the cloud runs); small uses the original flat paths.
INPUT_DIR = "" if MAG_TAG == "small" else f"{MAG_TAG}_inputs/"

SIGNAL_NAMES = {
    1: "Diarrhoea, NHS 111", 2: "Arthropod bites, ED", 3: "Cardiac, ED",
    4: "ICU cardiac adm., ED", 5: "Allergic rhinitis, GP", 6: "Heat stroke, GP",
    7: "Herpes zoster, GP", 8: "Insect bite, GP", 9: "Pertussis, GP",
    10: "Pneumonia, GPOOH", 11: "Rubella, GP", 12: "Upper resp. tract, GP",
    13: "Bronchitis, GPOOH", 14: "Hepatitis, GPOOH", 15: "ILI, GPOOH",
    16: "UTI, GPOOH",
}


def load_alarm_matrix(method, S, n_test):
    # Magnitude tag derived from the INPUT_DIR (== '' for small, '<mag>_inputs/' otherwise)
    _mag = (INPUT_DIR.split('_inputs/')[0] or 'small')
    paths = {
        'Farrington': f'{INPUT_DIR}farrington_custom_alarms_signal_{S}.csv',
        'CUSUM': f'{INPUT_DIR}cusum_alarms_signal_{S}.csv',
        'IF': f'{INPUT_DIR}isolation_forest_alarms_signal_{S}.csv',
        'KNN': f'{INPUT_DIR}knn_alarms_signal_{S}.csv',
        'OCSVM': f'{INPUT_DIR}ocsvm_alarms_signal_{S}.csv',
        'LOF': f'{INPUT_DIR}lof_alarms_signal_{S}.csv',
        'RateChange': f'{INPUT_DIR}ratechange_residual_alarms_signal_{S}.csv',
        'R-IF':    f'score_cache/if_residual_{_mag}/if_residual_alarms_signal_{S}.csv',
        'R-KNN':   f'score_cache/knn_residual_{_mag}/knn_residual_alarms_signal_{S}.csv',
        'R-LOF':   f'score_cache/lof_residual_{_mag}/lof_residual_alarms_signal_{S}.csv',
        'R-OCSVM': f'score_cache/ocsvm_residual_{_mag}/ocsvm_residual_alarms_signal_{S}.csv',
    }
    if method not in paths:
        return None
    p = paths[method]
    if not os.path.exists(p):
        return None
    try:
        A = pd.read_csv(p).to_numpy(dtype=int)
        return A if A.shape[1] == n_test else None
    except Exception:
        return None


def alarms_from_scores(score_path_val, score_path_test, val_sims, higher_normal,
                       n_test, spec_target=0.975):
    if not os.path.exists(score_path_val) or not os.path.exists(score_path_test):
        return None
    v = pd.read_csv(score_path_val).to_numpy().astype(np.float64)
    t = pd.read_csv(score_path_test).to_numpy().astype(np.float64)
    if not higher_normal:
        v = -v; t = -t
    val_flat = v.flatten(order="F")
    val_lengths = [v.shape[0]] * v.shape[1]
    c = tune_contamination_threshold(val_sims, val_lengths, val_flat,
                                      spec_target=spec_target,
                                      w_sens=W_SENS, w_spec=W_SPEC)
    thr = np.percentile(val_flat, c * 100)
    return (t <= thr).astype(int)


def per_outbreak_timing(A_win, O_win, X_baseline=None):
    """
    Returns per-outbreak detection info:
      - hit: bool
      - days_to_detect: int (days from outbreak start to first alarm). NaN if missed.
      - relative_pos: float in [0, 1]. NaN if missed.
      - outbreak_length: int

    A_win and O_win are shape (WIN_LEN, n_sims). X_baseline unused.
    """
    n_days, n_sims = A_win.shape
    records = []
    for j in range(n_sims):
        # Find contiguous outbreak events
        y = O_win[:, j]
        in_event = False
        start = None
        for i in range(n_days):
            if y[i] > 0 and not in_event:
                start = i
                in_event = True
            elif (y[i] == 0 or i == n_days - 1) and in_event:
                end = i - 1 if y[i] == 0 else i
                length = end - start + 1

                # First alarm during event
                hits = np.where(A_win[start:end + 1, j] == 1)[0]
                if hits.size:
                    days_to_detect = int(hits[0])
                    rel_pos = days_to_detect / max(length - 1, 1) if length > 1 else 0.0
                    records.append((True, days_to_detect, rel_pos, length))
                else:
                    records.append((False, np.nan, np.nan, length))
                in_event = False
    return records


# ===== Load all data and alarm matrices =====
if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)

    METHODS_ALARM = ['Farrington', 'CUSUM', 'IF', 'KNN', 'OCSVM', 'LOF', 'RateChange']
    METHODS_SCORE = {
        'LSTM-AE': dict(val='lstm_ae_val_scores_signal_{S}.csv',
                        test='lstm_ae_test_scores_signal_{S}.csv', higher_normal=False),
        'NB-HMM': dict(val='nbhmm_val_scores_signal_{S}.csv',
                       test='nbhmm_test_scores_signal_{S}.csv', higher_normal=True),
        'VAE': dict(val='cloud_out/vae_small/vae_val_scores_signal_{S}.csv',
                    test='cloud_out/vae_small/vae_test_scores_signal_{S}.csv', higher_normal=True),
        'BOCPD': dict(val='bocpd_resid_val_scores_signal_{S}.csv',
                      test='bocpd_resid_test_scores_signal_{S}.csv', higher_normal=True),
    }

    print("=" * 110)
    print("POD × TIMELINESS — early-warning analysis")
    print("=" * 110)
    print("\nLoading data and alarm matrices...")

    sig_data = {}
    for S in SIGNALS:
        Xsig, Ysig = load_data(S)
        sims = []
        for i, c in enumerate(Xsig.columns):
            x = Xsig[c].to_numpy(np.float32, copy=False)
            y = Ysig[c].to_numpy(np.int32, copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(x=x, y=y, sim_idx=i))
        if not sims:
            continue
        train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
        O_full = np.stack([d['y'] for d in test_sims], axis=1)
        n_test = len(test_sims)

        alarms = {}
        for m in METHODS_ALARM:
            A = load_alarm_matrix(m, S, n_test)
            if A is not None:
                # Always slice to last WIN_LEN
                if A.shape[0] != WIN_LEN:
                    A = A[-WIN_LEN:]
                alarms[m] = A
        for m, info in METHODS_SCORE.items():
            A = alarms_from_scores(info['val'].format(S=S), info['test'].format(S=S),
                                    val_sims, info['higher_normal'], n_test)
            if A is not None:
                alarms[m] = A

        sig_data[S] = dict(O_full=O_full, n_test=n_test, alarms=alarms, val_sims=val_sims)

    METHODS = ['Farrington', 'CUSUM', 'IF', 'KNN', 'OCSVM', 'LOF',
               'LSTM-AE', 'NB-HMM', 'VAE', 'BOCPD', 'RateChange']

    # ===== Per (method, signal) timing analysis =====
    per_sig_timing = defaultdict(dict)  # method -> signal -> dict

    for S in SIGNALS:
        if S not in sig_data:
            continue
        d = sig_data[S]
        O_win = d['O_full'][-WIN_LEN:]
        for m, A in d['alarms'].items():
            records = per_outbreak_timing(A, O_win)
            n_events = len(records)
            if n_events == 0:
                continue
            hits = [r for r in records if r[0]]
            n_hits = len(hits)
            pod = n_hits / n_events
            # Conditional timeliness — mean relative pos GIVEN detected
            cond_rel = np.mean([r[2] for r in hits]) if hits else np.nan
            # Days-to-detection given detected
            mean_days = np.mean([r[1] for r in hits]) if hits else np.nan
            median_days = np.median([r[1] for r in hits]) if hits else np.nan
            # Standard timeliness metric (R-formula): (sum of rel-pos for hits + 1.0 per miss) / n_sims
            # But timeliness is per-sim not per-event; recompute using R formula
            timeliness_R = compute_timeliness_R(A, d['O_full'])
            # POD by R-formula too
            pod_R = compute_pod_R(A, d['O_full'])
            # Decomposition: miss penalty = (n_events - n_hits) / n_events
            # Lateness contribution = sum of rel-pos for hits / n_events
            miss_pen = (n_events - n_hits) / n_events
            late_pen = sum(r[2] for r in hits) / n_events if n_events else 0
            # Per-method-per-signal records
            per_sig_timing[m][S] = dict(
                pod=pod, pod_R=pod_R,
                timeliness_R=timeliness_R,
                cond_rel_pos=cond_rel,
                mean_days_to_detect=mean_days,
                median_days_to_detect=median_days,
                miss_penalty=miss_pen,
                lateness_penalty=late_pen,
                n_events=n_events, n_hits=n_hits,
            )

    # ============== TABLE: POD per method per signal ==============
    print("\n" + "=" * 130)
    print("TABLE A: POD (event-level recall) per method × signal")
    print("=" * 130)
    hdr = f"{'Sig':<3} {'Name':<22}"
    for m in METHODS:
        hdr += f" {m[:8]:>8}"
    hdr += f" {'MEAN':>7}"
    print(hdr)
    print("-" * len(hdr))
    method_pod_mean = defaultdict(list)
    for S in SIGNALS:
        if S not in sig_data:
            continue
        row = f"{S:<3} {SIGNAL_NAMES[S]:<22}"
        vals = []
        for m in METHODS:
            if S in per_sig_timing[m]:
                p = per_sig_timing[m][S]['pod']
                row += f" {p:>8.2f}"
                vals.append(p)
                method_pod_mean[m].append(p)
            else:
                row += f" {'---':>8}"
        if vals:
            row += f" {np.mean(vals):>7.2f}"
        print(row)
    print("-" * len(hdr))
    row = f"{'':<3} {'MEAN':<22}"
    for m in METHODS:
        if method_pod_mean[m]:
            row += f" {np.mean(method_pod_mean[m]):>8.2f}"
        else:
            row += f" {'---':>8}"
    print(row)

    # ============== TABLE: Mean days-to-detect (when detected) per method per signal ==============
    print("\n" + "=" * 130)
    print("TABLE B: MEAN DAYS-TO-DETECTION (absolute days from outbreak start, "
          "given outbreak detected)")
    print("       Lower = faster early-warning")
    print("=" * 130)
    hdr = f"{'Sig':<3} {'Name':<22}"
    for m in METHODS:
        hdr += f" {m[:8]:>8}"
    hdr += f" {'MEAN':>7}"
    print(hdr)
    print("-" * len(hdr))
    method_days_mean = defaultdict(list)
    for S in SIGNALS:
        if S not in sig_data:
            continue
        row = f"{S:<3} {SIGNAL_NAMES[S]:<22}"
        vals = []
        for m in METHODS:
            if S in per_sig_timing[m]:
                dy = per_sig_timing[m][S]['mean_days_to_detect']
                if not np.isnan(dy):
                    row += f" {dy:>8.2f}"
                    vals.append(dy)
                    method_days_mean[m].append(dy)
                else:
                    row += f" {'nan':>8}"
            else:
                row += f" {'---':>8}"
        if vals:
            row += f" {np.mean(vals):>7.2f}"
        print(row)
    print("-" * len(hdr))
    row = f"{'':<3} {'MEAN':<22}"
    for m in METHODS:
        if method_days_mean[m]:
            row += f" {np.mean(method_days_mean[m]):>8.2f}"
        else:
            row += f" {'---':>8}"
    print(row)

    # ============== TABLE: R-Timeliness (combined) per method per signal ==============
    print("\n" + "=" * 130)
    print("TABLE C: R-TIMELINESS (combines miss penalty + lateness; lower = better)")
    print("=" * 130)
    hdr = f"{'Sig':<3} {'Name':<22}"
    for m in METHODS:
        hdr += f" {m[:8]:>8}"
    hdr += f" {'MEAN':>7}"
    print(hdr)
    print("-" * len(hdr))
    method_tim_mean = defaultdict(list)
    for S in SIGNALS:
        if S not in sig_data:
            continue
        row = f"{S:<3} {SIGNAL_NAMES[S]:<22}"
        vals = []
        for m in METHODS:
            if S in per_sig_timing[m]:
                t = per_sig_timing[m][S]['timeliness_R']
                if not np.isnan(t):
                    row += f" {t:>8.3f}"
                    vals.append(t)
                    method_tim_mean[m].append(t)
                else:
                    row += f" {'nan':>8}"
            else:
                row += f" {'---':>8}"
        if vals:
            row += f" {np.mean(vals):>7.3f}"
        print(row)
    print("-" * len(hdr))
    row = f"{'':<3} {'MEAN':<22}"
    for m in METHODS:
        if method_tim_mean[m]:
            row += f" {np.mean(method_tim_mean[m]):>8.3f}"
        else:
            row += f" {'---':>8}"
    print(row)

    # ============== TABLE: Timeliness decomposition — miss penalty vs lateness ==============
    print("\n" + "=" * 110)
    print("TABLE D: TIMELINESS DECOMPOSITION (per method, averaged across signals)")
    print("    Timeliness_R = miss_penalty + lateness_penalty")
    print("    Which contributes more for each method?")
    print("=" * 110)
    print(f"\n{'Method':<12} {'Tim_R':>7} {'POD':>6} {'Miss pen':>9} {'Late pen':>9} "
          f"{'Cond-pos':>10} {'Days/det':>10} {'Dominant':>15}")
    print("-" * 95)
    for m in METHODS:
        if not per_sig_timing[m]:
            continue
        avg_tim = np.mean([per_sig_timing[m][S]['timeliness_R']
                            for S in per_sig_timing[m]])
        avg_pod = np.mean([per_sig_timing[m][S]['pod'] for S in per_sig_timing[m]])
        avg_miss = np.mean([per_sig_timing[m][S]['miss_penalty']
                             for S in per_sig_timing[m]])
        avg_late = np.mean([per_sig_timing[m][S]['lateness_penalty']
                             for S in per_sig_timing[m]])
        # Only consider sigs where the method detected something
        cond_rels = [per_sig_timing[m][S]['cond_rel_pos'] for S in per_sig_timing[m]
                     if not np.isnan(per_sig_timing[m][S]['cond_rel_pos'])]
        mean_days = [per_sig_timing[m][S]['mean_days_to_detect'] for S in per_sig_timing[m]
                     if not np.isnan(per_sig_timing[m][S]['mean_days_to_detect'])]
        avg_cr = np.mean(cond_rels) if cond_rels else np.nan
        avg_md = np.mean(mean_days) if mean_days else np.nan
        dominant = "miss penalty" if avg_miss > avg_late else "lateness"
        print(f"{m:<12} {avg_tim:>7.3f} {avg_pod:>6.2f} {avg_miss:>9.3f} "
              f"{avg_late:>9.3f} {avg_cr:>10.3f} {avg_md:>10.2f} {dominant:>15}")

    # ============== ANALYSIS 1: POD-TIMELINESS Pareto frontier (across methods) ==============
    print("\n" + "=" * 110)
    print("ANALYSIS 1: POD × TIMELINESS_R Pareto frontier (per method, mean across signals)")
    print("    Pareto-optimal = no other method has both higher POD AND lower timeliness")
    print("=" * 110)
    points = []
    for m in METHODS:
        if not per_sig_timing[m]:
            continue
        avg_pod = np.mean([per_sig_timing[m][S]['pod'] for S in per_sig_timing[m]])
        avg_tim = np.mean([per_sig_timing[m][S]['timeliness_R']
                            for S in per_sig_timing[m]])
        points.append((m, avg_pod, avg_tim))

    # Identify Pareto frontier (higher POD, lower timeliness)
    def is_pareto_optimal(idx):
        m_i, p_i, t_i = points[idx]
        for j, (m_j, p_j, t_j) in enumerate(points):
            if j == idx:
                continue
            if p_j >= p_i and t_j <= t_i and (p_j > p_i or t_j < t_i):
                return False
        return True

    pareto = [points[i] for i in range(len(points)) if is_pareto_optimal(i)]
    pareto = sorted(pareto, key=lambda x: x[1], reverse=True)
    dominated = sorted(
        [p for p in points if p not in pareto],
        key=lambda x: x[1], reverse=True,
    )

    print(f"\n  PARETO-OPTIMAL methods (each is best for some POD-timeliness tradeoff):")
    print(f"  {'Method':<12} {'POD':>6} {'Timeliness':>11}  Verdict")
    print(f"  {'-'*12} {'-'*6} {'-'*11}  {'-'*40}")
    for m, p, t in pareto:
        if p > 0.65:
            verdict = "high-POD anchor"
        elif t < 0.45:
            verdict = "fastest detector"
        else:
            verdict = "balanced"
        print(f"  {m:<12} {p:>6.2f} {t:>11.3f}  {verdict}")
    print(f"\n  DOMINATED methods (some other method beats them on BOTH axes):")
    print(f"  {'Method':<12} {'POD':>6} {'Timeliness':>11}  Dominated by:")
    print(f"  {'-'*12} {'-'*6} {'-'*11}  {'-'*40}")
    for m, p, t in dominated:
        # Find dominating methods
        dominators = [m2 for m2, p2, t2 in pareto if p2 >= p and t2 <= t]
        print(f"  {m:<12} {p:>6.2f} {t:>11.3f}  {', '.join(dominators)}")

    # ============== ANALYSIS 2: PER-SIGNAL fastest detector ==============
    print("\n" + "=" * 110)
    print("ANALYSIS 2: FASTEST DETECTOR per signal (lowest days-to-detect among methods "
          "with POD>=0.5)")
    print("=" * 110)
    print(f"\n  {'Sig':<3} {'Name':<22} {'Fastest':<12} {'Days':>6} "
          f"{'POD':>6}   {'Highest POD':<12} {'POD':>6} {'Days':>6}")
    print(f"  {'-'*3} {'-'*22} {'-'*12} {'-'*6} {'-'*6}   {'-'*12} {'-'*6} {'-'*6}")
    for S in SIGNALS:
        if S not in sig_data:
            continue
        # Reliable detectors (POD>=0.5)
        reliable = [
            (m, per_sig_timing[m][S]['mean_days_to_detect'],
             per_sig_timing[m][S]['pod'])
            for m in METHODS
            if S in per_sig_timing[m]
            and per_sig_timing[m][S]['pod'] >= 0.5
            and not np.isnan(per_sig_timing[m][S]['mean_days_to_detect'])
        ]
        # All detectors with positive POD
        all_dets = [
            (m, per_sig_timing[m][S]['mean_days_to_detect'],
             per_sig_timing[m][S]['pod'])
            for m in METHODS
            if S in per_sig_timing[m]
            and not np.isnan(per_sig_timing[m][S]['mean_days_to_detect'])
        ]
        if not all_dets:
            continue
        # Fastest among reliable
        if reliable:
            fast = min(reliable, key=lambda x: x[1])
        else:
            fast = ('—', np.nan, 0)
        # Highest POD
        best_pod = max(all_dets, key=lambda x: x[2])
        print(f"  {S:<3} {SIGNAL_NAMES[S]:<22} {fast[0]:<12} {fast[1]:>6.2f} "
              f"{fast[2]:>6.2f}   {best_pod[0]:<12} {best_pod[2]:>6.2f} "
              f"{best_pod[1]:>6.2f}")

    # ============== ANALYSIS 3: ENSEMBLE TIMING — does OR-vote inherit the FAST member? ==============
    print("\n" + "=" * 110)
    print("ANALYSIS 3: ENSEMBLE TIMING — does OR-vote inherit the FASTEST member's timing?")
    print("=" * 110)
    print("    For each combination, compute: ensemble days-to-detect vs min(member days)")
    print()

    ensemble_combos = [
        ('Farr+IF', ['Farrington', 'IF']),
        ('Farr+KNN', ['Farrington', 'KNN']),
        ('Farr+LSTM-AE', ['Farrington', 'LSTM-AE']),
        ('Farr+IF+NB-HMM', ['Farrington', 'IF', 'NB-HMM']),
        ('Farr+IF+LSTM-AE', ['Farrington', 'IF', 'LSTM-AE']),
        ('Farr+IF+LSTM-AE+NB-HMM', ['Farrington', 'IF', 'LSTM-AE', 'NB-HMM']),
        ('Farr+CUSUM+IF+LSTM-AE+NB-HMM', ['Farrington', 'CUSUM', 'IF', 'LSTM-AE', 'NB-HMM']),
        ('Farr+CUSUM+KNN+LSTM-AE+NB-HMM', ['Farrington', 'CUSUM', 'KNN', 'LSTM-AE', 'NB-HMM']),
    ]

    print(f"  {'Ensemble':<35} {'POD':>5} {'Tim_R':>7} {'Days/det':>9} "
          f"{'Best mem.':>10} {'Min mem days':>14}")
    print(f"  {'-'*35} {'-'*5} {'-'*7} {'-'*9} {'-'*10} {'-'*14}")

    ensemble_results = []
    for name, members in ensemble_combos:
        all_pods, all_tims, all_days = [], [], []
        member_min_days = []
        for S in SIGNALS:
            if S not in sig_data:
                continue
            mems_present = [m for m in members if m in sig_data[S]['alarms']]
            if len(mems_present) != len(members):
                continue
            # OR-vote alarm matrix
            A_or = np.maximum.reduce([sig_data[S]['alarms'][m] for m in mems_present])
            O_full = sig_data[S]['O_full']
            # POD and timeliness via R-formula
            pod = compute_pod_R(A_or, O_full)
            tim = compute_timeliness_R(A_or, O_full)
            # Days-to-detect for ensemble
            recs = per_outbreak_timing(A_or, O_full[-WIN_LEN:])
            hits = [r for r in recs if r[0]]
            ens_days = np.mean([r[1] for r in hits]) if hits else np.nan
            # Min member days
            mem_days_per = []
            for m in mems_present:
                if S in per_sig_timing[m]:
                    md = per_sig_timing[m][S]['mean_days_to_detect']
                    if not np.isnan(md):
                        mem_days_per.append(md)
            min_mem_days = min(mem_days_per) if mem_days_per else np.nan
            all_pods.append(pod)
            all_tims.append(tim)
            if not np.isnan(ens_days):
                all_days.append(ens_days)
            if not np.isnan(min_mem_days):
                member_min_days.append(min_mem_days)
        avg_pod = np.mean(all_pods) if all_pods else np.nan
        avg_tim = np.mean(all_tims) if all_tims else np.nan
        avg_days = np.mean(all_days) if all_days else np.nan
        avg_min_mem = np.mean(member_min_days) if member_min_days else np.nan
        # Best individual member
        best_mem = max(members, key=lambda m: np.mean(
            [per_sig_timing[m][S]['pod'] for S in per_sig_timing[m]]
            if per_sig_timing[m] else [0]
        ))
        ensemble_results.append((name, avg_pod, avg_tim, avg_days, avg_min_mem))
        print(f"  {name:<35} {avg_pod:>5.2f} {avg_tim:>7.3f} {avg_days:>9.2f} "
              f"{best_mem:<10} {avg_min_mem:>14.2f}")

    print("""
  IMPLICATION: ensemble days-to-detect vs min member days-to-detect.
  If ensemble days ≈ min(member days), the ensemble IS inheriting the fastest
  detector. If ensemble days > min(member days), the ensemble is being held
  back by slower members.
""")

    # ============== ANALYSIS 4: EARLY-WARNING SCORE ==============
    # Custom score: POD - alpha*timeliness, penalising lateness
    print("\n" + "=" * 110)
    print("ANALYSIS 4: EARLY-WARNING composite score")
    print("    score = POD - 0.5 * timeliness_R")
    print("    (rewards high POD AND low timeliness simultaneously)")
    print("=" * 110)

    print(f"\n  {'Method':<12} {'POD':>6} {'Tim_R':>7} {'EW score':>9}  ranking notes")
    print(f"  {'-'*12} {'-'*6} {'-'*7} {'-'*9}  {'-'*40}")
    ew_scores = []
    for m in METHODS:
        if not per_sig_timing[m]:
            continue
        avg_pod = np.mean([per_sig_timing[m][S]['pod'] for S in per_sig_timing[m]])
        avg_tim = np.mean([per_sig_timing[m][S]['timeliness_R']
                            for S in per_sig_timing[m]])
        score = avg_pod - 0.5 * avg_tim
        ew_scores.append((m, avg_pod, avg_tim, score))
    ew_scores.sort(key=lambda x: x[3], reverse=True)
    for m, p, t, s in ew_scores:
        note = ""
        if p > 0.7 and t < 0.5:
            note = "★ early-warning gold"
        elif p > 0.7 and t > 0.55:
            note = "high POD but slow"
        elif p < 0.4:
            note = "low POD limits value"
        elif t < 0.4 and p < 0.6:
            note = "fast but unreliable"
        print(f"  {m:<12} {p:>6.2f} {t:>7.3f} {s:>9.3f}  {note}")

    # ============== Save comprehensive CSV ==============
    rows = []
    for S in SIGNALS:
        if S not in sig_data:
            continue
        for m in METHODS:
            if S in per_sig_timing[m]:
                rec = per_sig_timing[m][S]
                rows.append({
                    'signal': S, 'signal_name': SIGNAL_NAMES[S],
                    'method': m, **rec,
                })
    df_out = pd.DataFrame(rows)
    df_out.to_csv('results/pod_timeliness_per_signal.csv', index=False)
    print(f"\nSaved {len(df_out)} rows to results/pod_timeliness_per_signal.csv")
