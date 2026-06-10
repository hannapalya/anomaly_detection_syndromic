#!/usr/bin/env python3
"""
TIMELY-DETECTION metric: Probability of Successful Detection within d days of
outbreak onset, PSD(d).

GROUND TRUTH: each simulation contains exactly ONE injected outbreak, placed
inside the evaluation window. So an "event" is the whole sim's outbreak, spanning
[first injected day, last injected day]; the earlier contiguous-Y>0-run definition
wrongly split one outbreak into a body + 1-day satellites. We therefore work
per-sim, consistent with the R-comparator's own POD/timeliness (compute_pod_R,
compute_timeliness_R both already treat the sim's outbreak as one unit).

  onset_j  = first day with O>0 in the window (sim j)
  end_j    = last  day with O>0
  duration = end_j - onset_j + 1
  delay_j  = (first day with alarm AND O>0) - onset_j      [days; >=0]   (miss if none)

  PSD(d) = ( # sims whose delay <= d ) / ( # sims with an outbreak )

counting misses as never-detected. As d -> infinity, PSD(d) -> POD (== compute_pod_R).

Theory / naming:
  - Frisen M. (2003) Int. Stat. Rev. 71(2):403-434  -- defines PSD(d).
  - Sonesson & Bock (2003) JRSS-A 166:5-21          -- ARL1, CED, PSD.
  - Buckeridge (2007) J Biomed Inform 40:370-379     -- timeliness in syndromic surv.

Outputs:
  results/timely_detection_curve.csv      (method, day, psd_all, psd_long)
  results/timely_detection_summary.csv    (method, POD, PSD@1/3/5/7 days, CED, long versions)
  results/outbreak_duration_detection.csv  (method, bin, det_rate)  -- for the duration figure
  results/outbreak_duration_dist.csv       (bin, frac, n)           -- true (per-sim) durations
"""
import os
import numpy as np
import pandas as pd
from collections import defaultdict

from anom_common import (
    load_data, split_60_20_20, WIN_LEN, TRAIN_DAYS, VALID_DAYS, RNG_STATE, SIGNALS,
    MAG_TAG, compute_specificity_R, IDX_RANGE,
)
from analyse_pod_timeliness import load_alarm_matrix, alarms_from_scores

# small uses the original flat paths (+ cloud_out/vae_small); other magnitudes read
# from a magnitude-specific staged input dir.
INPUT_DIR = "" if MAG_TAG == "small" else f"{MAG_TAG}_inputs/"
_VAE = (f'{INPUT_DIR}vae_{{kind}}_scores_signal_{{S}}.csv' if INPUT_DIR
        else 'cloud_out/vae_small/vae_{kind}_scores_signal_{S}.csv')

METHODS_ALARM = ['Farrington', 'CUSUM', 'IF', 'KNN', 'OCSVM', 'LOF', 'RateChange']
METHODS_SCORE = {
    'LSTM-AE': dict(val=f'{INPUT_DIR}lstm_ae_val_scores_signal_{{S}}.csv',
                    test=f'{INPUT_DIR}lstm_ae_test_scores_signal_{{S}}.csv', higher_normal=False),
    'NB-HMM': dict(val=f'{INPUT_DIR}nbhmm_val_scores_signal_{{S}}.csv',
                   test=f'{INPUT_DIR}nbhmm_test_scores_signal_{{S}}.csv', higher_normal=True),
    'VAE': dict(val=_VAE.format(kind='val', S='{S}'),
                test=_VAE.format(kind='test', S='{S}'), higher_normal=True),
    'BOCPD': dict(val=f'{INPUT_DIR}bocpd_resid_val_scores_signal_{{S}}.csv',
                  test=f'{INPUT_DIR}bocpd_resid_test_scores_signal_{{S}}.csv', higher_normal=True),
}
METHODS = ['Farrington', 'CUSUM', 'IF', 'KNN', 'OCSVM', 'LOF',
           'LSTM-AE', 'NB-HMM', 'VAE', 'BOCPD', 'RateChange']

ENSEMBLE_NAME = 'OR-ensemble (IF+LSTM-AE+NB-HMM)'
ENSEMBLE_MEMBERS = ['IF', 'LSTM-AE', 'NB-HMM']

D_MAX = 20          # day offsets 0..20  ->  "days since onset" 1..21
LONG_MIN = 7        # outbreaks with duration >= 7 days


def per_sim_delay(A_win, O_win):
    """ONE record per sim that carries an outbreak: (hit, delay_days, duration).

    delay = (first day with alarm on an outbreak day) - onset.  Misses -> hit=False.
    """
    n_days, n_sims = A_win.shape
    recs = []
    for j in range(n_sims):
        oj = O_win[:, j]
        outdays = np.where(oj > 0)[0]
        if outdays.size == 0:
            continue                              # no outbreak in window for this sim
        onset, end = int(outdays[0]), int(outdays[-1])
        duration = end - onset + 1
        hit_days = np.where((A_win[:, j] == 1) & (oj > 0))[0]
        if hit_days.size:
            recs.append((True, int(hit_days[0]) - onset, duration))
        else:
            recs.append((False, np.nan, duration))
    return recs


def collect_records():
    """Returns (recs, specs):
      recs[method]  -> list of (hit, delay, duration) over all sims (one outbreak per sim)
      specs[method] -> list of per-signal specificities (R-comparator, on IDX_RANGE).
    """
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)
    recs = defaultdict(list)
    specs = defaultdict(list)
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
        _, val_sims, test_sims = split_60_20_20(sims, rng)
        O_full = np.stack([d['y'] for d in test_sims], axis=1)
        O_win = O_full[-WIN_LEN:]
        n_test = len(test_sims)

        alarms = {}
        for m in METHODS_ALARM:
            A = load_alarm_matrix(m, S, n_test)
            if A is None:
                continue
            if A.shape[0] != WIN_LEN:
                A = A[-WIN_LEN:]
            alarms[m] = A
        for m, info in METHODS_SCORE.items():
            A = alarms_from_scores(info['val'].format(S=S), info['test'].format(S=S),
                                   val_sims, info['higher_normal'], n_test)
            if A is None:
                continue
            if A.shape[0] != WIN_LEN:
                A = A[-WIN_LEN:]
            alarms[m] = A

        for m, A in alarms.items():
            recs[m].extend(per_sim_delay(A, O_win))
            sp = compute_specificity_R(A, O_full, IDX_RANGE)
            if not np.isnan(sp):
                specs[m].append(sp)
        if all(m in alarms for m in ENSEMBLE_MEMBERS):
            A_or = np.maximum.reduce([alarms[m] for m in ENSEMBLE_MEMBERS])
            recs[ENSEMBLE_NAME].extend(per_sim_delay(A_or, O_win))
            sp = compute_specificity_R(A_or, O_full, IDX_RANGE)
            if not np.isnan(sp):
                specs[ENSEMBLE_NAME].append(sp)
    return recs, specs


def psd_curve(records, length_min=None):
    if length_min is not None:
        records = [r for r in records if r[2] >= length_min]
    n = len(records)
    if n == 0:
        return 0, np.full(D_MAX + 1, np.nan)
    psd = np.array([np.mean([1.0 if (r[0] and r[1] <= d) else 0.0 for r in records])
                    for d in range(D_MAX + 1)])
    return n, psd


def ced(records):
    dd = [r[1] for r in records if r[0]]
    return float(np.mean(dd)) if dd else np.nan


DUR_BINS = [0, 1, 3, 7, 14, 10000]
DUR_LABELS = ["1", "2-3", "4-7", "8-14", "15+"]


def dur_bin(duration):
    for i in range(len(DUR_BINS) - 1):
        if DUR_BINS[i] < duration <= DUR_BINS[i + 1]:
            return DUR_LABELS[i]
    return DUR_LABELS[-1]


if __name__ == "__main__":
    print("=" * 100)
    print("TIMELY DETECTION — PSD(d), per-sim single-outbreak definition")
    print(f"  {MAG_TAG} magnitude; day offsets 0..{D_MAX}; long-outbreak threshold >= {LONG_MIN} days")
    print("=" * 100)
    print("\nReconstructing alarms (loads data + validation-tuned thresholds)...")
    recs, specs = collect_records()
    spec_mean = {m: float(np.mean(v)) for m, v in specs.items() if v}

    # ---- true (per-sim) outbreak-duration distribution ----
    # outbreak durations are method-independent (same O); use any method that has records
    durations = next(([r[2] for r in recs[m]] for m in recs if recs[m]), [])
    db = pd.Series([dur_bin(d) for d in durations])
    dist = (db.value_counts(normalize=True).reindex(DUR_LABELS).fillna(0))
    print(f"\nTRUE outbreak-duration distribution (one outbreak per sim, n={len(durations)}):")
    print("  " + "  ".join(f"{lab}d:{dist[lab]*100:4.1f}%" for lab in DUR_LABELS)
          + f"   median={np.median(durations):.0f}d")

    # ---- PSD table ----
    curve_rows, summary_rows, dur_rows = [], [], []
    print("\n" + "-" * 100)
    print(f"{'Method':<32} {'#sim':>5} {'POD':>5} | within-first-N-days PSD     "
          f"| {'CED':>5} || long(>= {LONG_MIN}d)")
    print(f"{'':<32} {'':>5} {'':>5} | {'1d':>5} {'3d':>5} {'5d':>5} {'7d':>5} "
          f"| {'days':>5} || {'#':>4} {'POD':>5} {'5d':>5}")
    print("-" * 100)
    cache, order_metric = {}, {}
    for m in [ENSEMBLE_NAME] + METHODS:
        if m not in recs or not recs[m]:
            continue
        n_all, psd_all = psd_curve(recs[m])
        n_long, psd_long = psd_curve(recs[m], length_min=LONG_MIN)
        cache[m] = (n_all, psd_all, n_long, psd_long, ced(recs[m]))
        order_metric[m] = psd_all[4]
    for m in sorted(order_metric, key=order_metric.get, reverse=True):
        n_all, psd_all, n_long, psd_long, c = cache[m]
        pod = psd_all[D_MAX]
        w1, w3, w5, w7 = psd_all[0], psd_all[2], psd_all[4], psd_all[6]
        print(f"{m:<32} {n_all:>5d} {pod:>5.2f} | {w1:>5.2f} {w3:>5.2f} {w5:>5.2f} "
              f"{w7:>5.2f} | {c:>5.2f} || {n_long:>4d} {psd_long[D_MAX]:>5.2f} {psd_long[4]:>5.2f}")
        summary_rows.append(dict(method=m, n_sims=n_all, POD=round(pod, 4),
                                 psd_within_1d=round(w1, 4), psd_within_3d=round(w3, 4),
                                 psd_within_5d=round(w5, 4), psd_within_7d=round(w7, 4),
                                 CED_days=round(c, 4),
                                 specificity=round(spec_mean.get(m, float('nan')), 4),
                                 n_long=n_long,
                                 POD_long=round(psd_long[D_MAX], 4),
                                 psd_within_5d_long=round(psd_long[4], 4)))
        for d in range(D_MAX + 1):
            curve_rows.append(dict(method=m, day=d + 1,
                                   psd_all=round(psd_all[d], 5),
                                   psd_long=round(psd_long[d], 5)))
        # detection rate by duration bin (any-hit), for the duration figure
        by_bin = defaultdict(list)
        for hit, delay, dur in recs[m]:
            by_bin[dur_bin(dur)].append(1.0 if hit else 0.0)
        for lab in DUR_LABELS:
            if by_bin[lab]:
                dur_rows.append(dict(method=m, bin=lab,
                                     det_rate=round(float(np.mean(by_bin[lab])), 4),
                                     n=len(by_bin[lab])))

    os.makedirs("results", exist_ok=True)
    SUF = "" if MAG_TAG == "small" else f"_{MAG_TAG}"   # don't clobber the small results
    pd.DataFrame(curve_rows).to_csv(f"results/timely_detection_curve{SUF}.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(f"results/timely_detection_summary{SUF}.csv", index=False)
    pd.DataFrame(dur_rows).to_csv(f"results/outbreak_duration_detection{SUF}.csv", index=False)
    pd.DataFrame([dict(bin=lab, frac=round(float(dist[lab]), 4),
                       n=int(round(dist[lab] * len(durations)))) for lab in DUR_LABELS]
                 ).to_csv(f"results/outbreak_duration_dist{SUF}.csv", index=False)
    print("-" * 100)
    print("\nPOD here == compute_pod_R (per-sim any-hit). 'within 5 days' now binds widely")
    print("because the true single outbreaks are ~2 weeks long, not 1-2 days.")
    print("\nSaved -> results/timely_detection_{curve,summary}.csv, "
          "results/outbreak_duration_{detection,dist}.csv")
