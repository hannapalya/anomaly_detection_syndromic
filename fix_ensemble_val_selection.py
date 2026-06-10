#!/usr/bin/env python3
"""
FIX #1: Re-select ensemble configurations on VALIDATION, report on TEST.

The original ensemble search (search_ensembles_v2.py) ranked ~1,881 OR-vote
configurations by TEST-set performance, then reported the best. That is
selection-on-test and optimistically biases the recommended ensemble.

This script:
  1. Builds val-window AND test-window alarms for every detector, using
     thresholds tuned ONLY on validation (the standard, leak-free procedure).
  2. Enumerates OR-vote configurations of 2-5 detectors.
  3. Computes both VAL and TEST R-comparator metrics for each configuration.
  4. Selects the winner on VALIDATION (max val-sensitivity subject to
     val-specificity >= constraint), then reports that configuration's TEST
     metrics.
  5. Reports, for contrast, the configuration that the old TEST-ranking would
     have picked, quantifying the selection-on-test optimism.

Detectors (all with aligned val + test data):
  Farrington (val alarms from run_farrington_custom.R ... val),
  IF, NB-HMM, LSTM-AE, VAE, BOCPD (cached val/test scores),
  CUSUM, Noufaily (computed live on val and test sims).

Run AFTER: SYND_DATA_DIR=big_signal_datasets_small Rscript run_farrington_custom.R 0.01 val
"""
import os
import itertools
import numpy as np
import pandas as pd
from collections import defaultdict
from scipy.stats import nbinom

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
)
from run_cusum import CUSUMDetector


def noufaily_score(d):
    cu = CUSUMDetector(k=0.5)
    cu.fit_baseline(d['x'], d['y'])
    mu = cu.baseline(len(d['x']))
    alpha = cu.dispersion_
    if alpha > 1.001:
        r = mu / (alpha - 1.0); r = np.clip(r, 1e-3, 1e8)
        p = r / (r + mu)
        y = np.asarray(d['x'], dtype=int)
        log_s = np.log(np.clip(nbinom.sf(y - 1, n=r, p=p), 1e-12, 1.0))
    else:
        from scipy.stats import poisson
        y = np.asarray(d['x'], dtype=int)
        log_s = np.log(np.clip(poisson.sf(y - 1, mu=mu), 1e-12, 1.0))
    return log_s.astype(np.float32)


def cusum_score(d, k=0.5):
    cu = CUSUMDetector(k=k)
    cu.fit_baseline(d['x'], d['y'])
    return cu.decision_function_series(d['x'])  # higher = normal


# Cached score-based detectors (val/test). higher_normal matches v4 conventions.
SCORE_DETS = {
    'IF':      dict(val='if_tuned_val_scores_signal_{S}.csv',
                    test='if_tuned_test_scores_signal_{S}.csv', higher_normal=True, cost='medium'),
    'LSTM-AE': dict(val='lstm_ae_val_scores_signal_{S}.csv',
                    test='lstm_ae_test_scores_signal_{S}.csv', higher_normal=False, cost='expensive'),
    'NB-HMM':  dict(val='nbhmm_val_scores_signal_{S}.csv',
                    test='nbhmm_test_scores_signal_{S}.csv', higher_normal=True, cost='cheap'),
    'BOCPD':   dict(val='bocpd_resid_val_scores_signal_{S}.csv',
                    test='bocpd_resid_test_scores_signal_{S}.csv', higher_normal=True, cost='cheap'),
    'VAE':     dict(val='cloud_out/vae_small/vae_val_scores_signal_{S}.csv',
                    test='cloud_out/vae_small/vae_test_scores_signal_{S}.csv', higher_normal=True, cost='expensive'),
}
# Live detectors computed on the fly
LIVE_DETS = {
    'Noufaily': dict(scorer=noufaily_score, cost='cheap'),
    'CUSUM':    dict(scorer=cusum_score, cost='cheap'),
}
# Farrington from R: val + test alarm files
FARR_VAL  = 'farrington_custom_val_alarms_signal_{S}.csv'
FARR_TEST = 'farrington_custom_alarms_signal_{S}.csv'

COST_ORDER = {'cheap': 0, 'medium': 1, 'expensive': 2}


def val_metrics(A_val, O_full_val):
    return (compute_sensitivity_R(A_val, O_full_val),
            compute_specificity_R(A_val, O_full_val, IDX_RANGE))


def test_metrics(A, O_full):
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

    # Per-detector val/test alarm matrices, per signal
    det_val_alarms = defaultdict(dict)
    det_test_alarms = defaultdict(dict)
    O_full_val = {}
    O_full_test = {}

    print("Building val + test alarms per detector...", flush=True)
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
        _, val_sims, test_sims = split_60_20_20(sims, rng)
        O_full_val[S] = np.stack([d['y'] for d in val_sims], axis=1)
        O_full_test[S] = np.stack([d['y'] for d in test_sims], axis=1)
        n_val, n_test = len(val_sims), len(test_sims)

        # ---- Score-based detectors ----
        for dname, dinfo in SCORE_DETS.items():
            vp, tp = dinfo['val'].format(S=S), dinfo['test'].format(S=S)
            if not (os.path.exists(vp) and os.path.exists(tp)):
                continue
            v = pd.read_csv(vp).to_numpy().astype(np.float64)
            t = pd.read_csv(tp).to_numpy().astype(np.float64)
            if not dinfo['higher_normal']:
                v = -v; t = -t   # make higher = normal
            # Tune threshold on val (leak-free: val only)
            val_flat = v.flatten(order='F')
            val_lengths = [v.shape[0]] * v.shape[1]
            c = tune_contamination_threshold(val_sims, val_lengths, val_flat,
                                             spec_target=SPEC_TARGET, w_sens=W_SENS, w_spec=W_SPEC)
            thr = np.percentile(val_flat, c * 100)
            det_val_alarms[dname][S] = (v <= thr).astype(int)
            det_test_alarms[dname][S] = (t <= thr).astype(int)

        # ---- Live detectors (Noufaily, CUSUM) ----
        for dname, dinfo in LIVE_DETS.items():
            scorer = dinfo['scorer']
            vcols, tcols = [], []
            for d in val_sims:
                s = scorer(d)
                vcols.append(s[len(d['x']) - VALID_DAYS:])
            for d in test_sims:
                s = scorer(d)
                tcols.append(s[ABS_START:ABS_END])
            v = np.column_stack(vcols).astype(np.float64)
            t = np.column_stack(tcols).astype(np.float64)
            # scorer convention: higher = normal already
            val_flat = v.flatten(order='F')
            val_lengths = [v.shape[0]] * v.shape[1]
            c = tune_contamination_threshold(val_sims, val_lengths, val_flat,
                                             spec_target=SPEC_TARGET, w_sens=W_SENS, w_spec=W_SPEC)
            thr = np.percentile(val_flat, c * 100)
            det_val_alarms[dname][S] = (v <= thr).astype(int)
            det_test_alarms[dname][S] = (t <= thr).astype(int)

        # ---- Farrington from R alarm files ----
        fv, ft = FARR_VAL.format(S=S), FARR_TEST.format(S=S)
        if os.path.exists(fv) and os.path.exists(ft):
            Av = pd.read_csv(fv).to_numpy(dtype=int)
            At = pd.read_csv(ft).to_numpy(dtype=int)
            if Av.shape[1] == n_val and At.shape[1] == n_test:
                det_val_alarms['Farrington'][S] = Av
                det_test_alarms['Farrington'][S] = At

        print(f"  sig {S}: {sum(1 for dd in det_val_alarms if S in det_val_alarms[dd])} detectors", flush=True)

    # Detectors with full 16-signal val+test coverage
    dets = sorted([d for d in det_val_alarms
                   if len(det_val_alarms[d]) == 16 and len(det_test_alarms.get(d, {})) == 16])
    print(f"\nDetectors with full coverage: {dets}")
    if 'Farrington' not in dets:
        print("\n*** WARNING: Farrington val alarms not found/aligned. "
              "Run: SYND_DATA_DIR=big_signal_datasets_small Rscript run_farrington_custom.R 0.01 val ***")

    # ---- Enumerate OR-vote configs, compute val + test metrics ----
    def cost_of(combo):
        order = max(COST_ORDER[SCORE_DETS[d]['cost']] if d in SCORE_DETS
                    else COST_ORDER[LIVE_DETS[d]['cost']] if d in LIVE_DETS
                    else 0 for d in combo)
        return {0: 'cheap', 1: 'medium', 2: 'expensive'}[order]

    rows = []
    for size in (2, 3, 4, 5):
        for combo in itertools.combinations(dets, size):
            val_sm, val_sp = [], []
            test_m = []
            ok = True
            for S in SIGNALS:
                if any(S not in det_val_alarms[d] or S not in det_test_alarms[d] for d in combo):
                    ok = False; break
                A_val = np.maximum.reduce([det_val_alarms[d][S] for d in combo])
                A_test = np.maximum.reduce([det_test_alarms[d][S] for d in combo])
                s_v, sp_v = val_metrics(A_val, O_full_val[S])
                val_sm.append(s_v); val_sp.append(sp_v)
                test_m.append(test_metrics(A_test, O_full_test[S]))
            if not ok:
                continue
            rows.append(dict(
                combo='+'.join(combo), size=size, cost=cost_of(combo),
                val_sens=np.nanmean(val_sm), val_spec=np.nanmean(val_sp),
                test_sens=np.nanmean([m['sensitivity'] for m in test_m]),
                test_spec=np.nanmean([m['specificity'] for m in test_m]),
                test_fpr=np.nanmean([m['fpr'] for m in test_m]),
                test_pod=np.nanmean([m['pod'] for m in test_m]),
                test_tim=np.nanmean([m['timeliness'] for m in test_m]),
            ))

    df = pd.DataFrame(rows)
    df.to_csv('results/ensemble_val_vs_test_selection.csv', index=False)
    print(f"\nEvaluated {len(df)} OR-vote configurations; saved to results/ensemble_val_vs_test_selection.csv")

    # ---- Selection comparison at several spec constraints ----
    print("\n" + "=" * 100)
    print("VAL-SELECTED vs TEST-SELECTED winner (the selection-on-test bias)")
    print("=" * 100)
    for spec_con in (0.93, 0.95, 0.96):
        print(f"\n--- Operating constraint: specificity >= {spec_con} ---")
        # Val selection: among configs with VAL spec >= constraint, max val_sens
        val_feasible = df[df['val_spec'] >= spec_con]
        test_feasible = df[df['test_spec'] >= spec_con]
        if len(val_feasible) == 0 or len(test_feasible) == 0:
            print("  no feasible configs at this constraint")
            continue
        val_winner = val_feasible.sort_values('val_sens', ascending=False).iloc[0]
        test_winner = test_feasible.sort_values('test_sens', ascending=False).iloc[0]

        print(f"  VAL-selected winner:  {val_winner['combo']}")
        print(f"    -> reported TEST: sens={val_winner['test_sens']:.3f} spec={val_winner['test_spec']:.3f} "
              f"fpr={val_winner['test_fpr']:.3f} tim={val_winner['test_tim']:.3f}  (cost={val_winner['cost']})")
        print(f"  TEST-selected winner (biased): {test_winner['combo']}")
        print(f"    -> TEST: sens={test_winner['test_sens']:.3f} spec={test_winner['test_spec']:.3f}")
        opt = test_winner['test_sens'] - val_winner['test_sens']
        same = "SAME config" if val_winner['combo'] == test_winner['combo'] else "DIFFERENT config"
        print(f"  => {same}; selection-on-test optimism = {opt:+.3f} sensitivity")

    # ---- Specifically check the paper's recommended 3-detector ensemble ----
    print("\n" + "=" * 100)
    print("Does the paper's recommended Farrington+IF+NB-HMM survive val selection?")
    print("=" * 100)
    # Best 3-detector config on val (spec>=0.95) vs the recommended one
    three = df[df['size'] == 3]
    if len(three):
        three_val = three[three['val_spec'] >= 0.95].sort_values('val_sens', ascending=False)
        print("\n  Top 5 three-detector ensembles by VALIDATION sensitivity (val spec>=0.95):")
        print(three_val.head(5)[['combo', 'val_sens', 'val_spec', 'test_sens', 'test_spec', 'cost']].to_string(index=False))
        rec = df[df['combo'].apply(lambda c: set(c.split('+')) == {'Farrington', 'IF', 'NB-HMM'})]
        if len(rec):
            r = rec.iloc[0]
            rank = (three_val['val_sens'] > r['val_sens']).sum() + 1
            print(f"\n  Farrington+IF+NB-HMM: val_sens={r['val_sens']:.3f} val_spec={r['val_spec']:.3f} "
                  f"-> test_sens={r['test_sens']:.3f} test_spec={r['test_spec']:.3f}")
            print(f"  Its rank among 3-detector configs by VAL sensitivity (spec>=0.95): #{rank}")
