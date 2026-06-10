#!/usr/bin/env python3
"""
Isolation Forest — Fair Farrington Comparison (no bank-holiday handling)
Option A: start earlier so TEST alarms exactly cover the R window [2205:2547] (0-based inclusive)
ALWAYS splits simulations per signal as 60% train / 20% val / 20% test (random, RNG_STATE-seeded).
Adds per-simulation metrics (validation + test) for statistical analysis:
- IsolationForest_big_medium_per_sim_val.csv
- IsolationForest_big_medium_per_sim_test.csv
"""

import os, numpy as np, pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix

# ===== CONFIG =====
DATA_DIR      = os.environ.get("SYND_DATA_DIR", "big_signal_datasets_small")
# Derive magnitude tag from DATA_DIR for output naming
_mag = DATA_DIR.rsplit("_", 1)[-1] if "_" in DATA_DIR else "small"
MAG_TAG = _mag if _mag in ("small", "medium", "large") else "small"
SIGNALS       = list(range(1,17))
DAYS_PER_YEAR = 364
TRAIN_YEARS   = 6
TRAIN_DAYS    = TRAIN_YEARS * DAYS_PER_YEAR
VALID_DAYS    = 49 * 7
RNG_STATE     = 42

# Validation target
SPEC_TARGET    = 0.97
W_SENS, W_SPEC = 2.0, 3.0

# Small, fast hyperparam search per signal
HP_GRID = [
    (7,  200, 0.7, 0.6),
    (7,  500, 0.9, 0.8),
    (14, 200, 0.9, 0.6),
    (14, 500, 0.7, 0.8),
    (21, 200, 0.7, 0.8),
    (21, 500, 0.9, 0.6),
]

# Comparator constants (mirrors R)
DAYS = 7
YEARS = 7
# R used 2206:2548 (1-based, end inclusive). Python 0-based -> 2205:2548 (end exclusive).
ABS_START = 2205
ABS_END   = 2548
WIN_LEN   = ABS_END - ABS_START  # 343 = 49*7
IDX_RANGE = np.arange(ABS_START, ABS_END, dtype=int)

# ===== HELPERS =====
def load_data(sig):
    X = pd.read_csv(os.path.join(DATA_DIR, f"simulated_totals_sig{sig}.csv"))
    Y = (pd.read_csv(os.path.join(DATA_DIR, f"simulated_outbreaks_sig{sig}.csv")) > 0).astype(int)
    for c in ["date", "Date", "ds", "timestamp"]:
        if c in X.columns: X = X.drop(columns=[c])
        if c in Y.columns: Y = Y.drop(columns=[c])
    return X, Y

def split_60_20_20(sims, rng):
    """Shuffle sims and split 60% train, 20% val, 20% test."""
    idx = np.arange(len(sims))
    rng.shuffle(idx)
    n = len(sims)
    n_train = int(round(0.60*n))
    n_val   = int(round(0.20*n))
    # Ensure no overlap and all sims assigned
    n_test  = n - n_train - n_val
    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]
    train_sims = [sims[i] for i in train_idx]
    val_sims   = [sims[i] for i in val_idx]
    test_sims  = [sims[i] for i in test_idx]
    return train_sims, val_sims, test_sims

# Use vectorized create_features from anom_common (80× faster than Python loop)
try:
    from anom_common import create_features
except ImportError:
    raise ImportError("anom_common.py must be importable for vectorized create_features")

def sens_spec(y_true, y_pred):
    if len(y_true)==0: return (np.nan, np.nan)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    sens = TP/(TP+FN) if (TP+FN)>0 else np.nan
    spec = TN/(TN+FP) if (TN+FP)>0 else np.nan
    return sens, spec

def tune_contamination_threshold(val_sims, val_lengths, decision_scores,
                                 spec_target=SPEC_TARGET,
                                 w_sens=W_SENS, w_spec=W_SPEC):
    """
    Tune contamination threshold using R-comparator metrics on VALIDATION TAILS.
    Assumes each validation column length == VALID_DAYS (343).
    """
    O_full_list = [d["y"] for d in val_sims]
    O_full = np.stack(O_full_list, axis=1)

    best_c, best_score = None, -1.0
    grid = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07]

    for c in grid:
        thr = np.percentile(decision_scores, c*100)
        yhat = (decision_scores <= thr).astype(int)

        # Build alarm matrix from concatenated predictions (no padding; all equal lengths)
        A_list, offset = [], 0
        for L in val_lengths:
            if L > 0:
                A_list.append(yhat[offset:offset+L])
                offset += L
        if not A_list or len(A_list) != len(val_sims): 
            continue

        A = np.column_stack(A_list)  # shape [VALID_DAYS, n_val_sims]

        sp = compute_specificity_R(A, O_full, IDX_RANGE)
        s  = compute_sensitivity_R(A, O_full)
        if (not np.isnan(sp)) and sp >= spec_target:
            score = w_sens*s + w_spec*sp
            if score > best_score:
                best_c, best_score = c, score

    if best_c is None:
        # Fallback: pick the most specific threshold among a subset
        best_sp, best_c = -1.0, 0.02
        for c in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
            thr = np.percentile(decision_scores, c*100)
            yhat = (decision_scores <= thr).astype(int)

            A_list, offset = [], 0
            for L in val_lengths:
                if L > 0:
                    A_list.append(yhat[offset:offset+L])
                    offset += L
            if not A_list or len(A_list) != len(val_sims): 
                continue

            A = np.column_stack(A_list)
            sp = compute_specificity_R(A, O_full, IDX_RANGE)
            if (not np.isnan(sp)) and sp > best_sp:
                best_sp, best_c = sp, c
        print("NOTE: no contamination hit the specificity target; chose the most specific fallback.")
    return best_c

def _align_tail(O, T):
    return O[-T:] if len(O) >= T else np.pad(O, (T-len(O), 0))

def compute_sensitivity(A, O):
    Oa = _align_tail(O, A.shape[0])
    TP = np.logical_and(A == 1, Oa > 0).sum()
    FN = np.logical_and(A == 0, Oa > 0).sum()
    return (TP / (TP + FN)) if (TP + FN) > 0 else np.nan

def compute_specificity(A, O):
    Oa = _align_tail(O, A.shape[0])
    TN = np.logical_and(A == 0, Oa == 0).sum()
    FP = np.logical_and(A == 1, Oa == 0).sum()
    return (TN / (TN + FP)) if (TN + FP) > 0 else np.nan

def compute_pod(A, O):
    Oa = _align_tail(O, A.shape[0])
    return np.mean((np.logical_and(A == 1, Oa > 0)).sum(axis=0) > 0)

def compute_timeliness(A, O):
    Oa = _align_tail(O, A.shape[0])
    T, J = A.shape
    score = 0.0
    for j in range(J):
        y = Oa[:, j]; a = A[:, j]
        idx_out = np.where(y > 0)[0]
        if len(idx_out) == 0: score += 1.0; continue
        idx_hit = np.where((a == 1) & (y > 0))[0]
        if len(idx_hit) == 0: score += 1.0; continue
        r1, r2 = int(idx_out[0]), int(idx_out[-1])
        obs = int(idx_hit[0])
        score += (obs - r1) / (r2 - r1 + 1)
    return score / J

# R-comparator metric helpers
def compute_fpr_R(A, O_full, idx_range):
    n = A.shape[0]
    FP = np.sum((A == 1) & (O_full[-n:, :] == 0))
    N0 = np.sum(O_full[idx_range, :] == 0)
    return np.nan if N0 == 0 else FP / N0

def compute_specificity_R(A, O_full, idx_range):
    n = A.shape[0]
    TN = np.sum((A == 0) & (O_full[-n:, :] == 0))
    N0 = np.sum(O_full[idx_range, :] == 0)
    return np.nan if N0 == 0 else TN / N0

def compute_sensitivity_R(A, O_full):
    n = A.shape[0]
    TP = np.sum((A == 1) & (O_full[-n:, :] > 0))
    P  = np.sum(O_full > 0)
    return np.nan if P == 0 else TP / P

def compute_pod_R(A, O_full):
    n = A.shape[0]
    per_sim_any = np.sum((A == 1) & (O_full[-n:, :] > 0), axis=0) > 0
    return per_sim_any.mean()

def compute_timeliness_R(A, O_full, days=DAYS, years=YEARS):
    nsim = A.shape[1]; n_alarm = A.shape[0]; n_out = O_full.shape[0]
    miss = 0; score = 0.0
    w_start = 52*days*(years-1) + 3*days
    w_end   = 52*days*years
    w = slice(w_start, w_end)
    for j in range(nsim):
        idx = np.where(O_full[w, j] > 0)[0]
        if idx.size == 0: miss += 1; continue
        r1 = w_start + idx.min(); r2 = w_start + idx.max()
        alarm_hit = np.where((A[:, j] == 1) & (O_full[-n_alarm:, j] > 0))[0]
        if alarm_hit.size:
            obs_idx = n_out - n_alarm + int(alarm_hit[0])
            score += (obs_idx - r1) / (r2 - r1 + 1)
        else:
            miss += 1
    return (score + miss) / nsim if nsim > 0 else np.nan

# ---- Per-sim helpers for validation/test ----
def split_by_lengths(arr, lengths):
    out = []; i = 0; total = len(arr)
    for L in lengths:
        out.append(arr[i:min(i+L, total)])
        i += L
    return out

def pod_anyhit(yhat, ytrue):
    return float(int(((yhat==1) & (ytrue==1)).any())) if (ytrue==1).any() else np.nan

def timeliness_single(yhat, ytrue):
    idx_out = np.where(ytrue > 0)[0]
    if len(idx_out) == 0: return np.nan
    idx_hit = np.where((ytrue > 0) & (yhat > 0))[0]
    if len(idx_hit) == 0: return 1.0
    r1, r2 = int(idx_out[0]), int(idx_out[-1]); obs = int(idx_hit[0])
    return (obs - r1) / (r2 - r1 + 1)

# ===== MAIN =====
np.random.seed(RNG_STATE)
rng = np.random.RandomState(RNG_STATE)

summary_all = {}
rows_val, rows_test = [], []

for S in SIGNALS:
    print(f"\n--- Signal {S} (Isolation Forest - Fair Comparison; fixed 60/20/20 splits) ---")
    Xsig, Ysig = load_data(S)

    sims = []
    for sim_idx, col in enumerate(Xsig.columns):
        x = Xsig[col].to_numpy(np.float32, copy=False)
        y = Ysig[col].to_numpy(np.int32,  copy=False)
        if len(x) >= TRAIN_DAYS + VALID_DAYS:
            sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{sim_idx}", sim_idx=sim_idx))
    if not sims:
        print("No complete sims; skip."); continue

    # ALWAYS 60/20/20 split (seeded, no overlap)
    train_sims, val_sims, test_sims_final = split_60_20_20(sims, rng)
    print(f"  Using fixed splits: {len(train_sims)} train, {len(val_sims)} val, {len(test_sims_final)} test")

    def build_train_matrix(window_size):
        XtrL = []
        for d in train_sims:
            feats = create_features(d["x"][:TRAIN_DAYS], window_size)
            if len(feats): XtrL.append(feats)
        return np.concatenate(XtrL) if XtrL else np.empty((0,20), np.float32)

    def build_val_tail(window_size):
        """
        Build validation windows with (window_size-1) days of context so that
        the first prediction aligns with the FIRST day of the 49-week tail.
        Every per-sim validation alarm length == VALID_DAYS (343).
        """
        XvL, YvL, lengths = [], [], []
        for d in val_sims:
            x = d["x"]; y = d["y"]
            tail_start = len(x) - VALID_DAYS
            ctx_start  = tail_start - (window_size - 1)
            if ctx_start < 0:
                continue  # cannot provide the context; skip this sim
            x_ctx_tail = x[ctx_start : tail_start + VALID_DAYS]
            y_ctx_tail = y[ctx_start : tail_start + VALID_DAYS]
            feats = create_features(x_ctx_tail, window_size)  # length == VALID_DAYS
            if len(feats) == VALID_DAYS:
                y_al = y_ctx_tail[window_size-1 : window_size-1 + VALID_DAYS].astype(int)
                XvL.append(feats); YvL.append(y_al); lengths.append(len(y_al))
        Xv = np.concatenate(XvL) if XvL else np.empty((0,20), np.float32)
        Yv = np.concatenate(YvL) if YvL else np.empty((0,), np.int32)
        return Xv, Yv, lengths

    best = dict(score=-1.0, params=None, contamination=None, val_lengths=None)

    for (WINDOW_SIZE, N_EST, MAX_SAMP, MAX_FEAT) in HP_GRID:
        Xtr = build_train_matrix(WINDOW_SIZE)
        if not len(Xtr): continue
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)

        Xval, Yval, val_lengths = build_val_tail(WINDOW_SIZE)
        Xval_s = scaler.transform(Xval) if len(Xval) else Xval

        iso = IsolationForest(
            n_estimators=N_EST,
            contamination=0.05,
            max_samples=MAX_SAMP,
            max_features=MAX_FEAT,
            random_state=RNG_STATE,
            n_jobs=-1,
            bootstrap=False,
        ).fit(Xtr_s)

        if len(Xval_s):
            val_scores = iso.decision_function(Xval_s)  # higher = more normal
            c_best = tune_contamination_threshold(val_sims, val_lengths, val_scores,
                                                  spec_target=SPEC_TARGET,
                                                  w_sens=W_SENS, w_spec=W_SPEC)
            thr = np.percentile(val_scores, c_best*100)
            yhat = (val_scores <= thr).astype(int)

            # R-comparator metrics for validation (logging only), no padding
            O_full_val = np.stack([d["y"] for d in val_sims], axis=1)
            A_list, offset = [], 0
            for L in val_lengths:
                if L > 0:
                    A_list.append(yhat[offset:offset+L]); offset += L
            if A_list and len(A_list) == len(val_sims):
                A  = np.column_stack(A_list)
                s  = compute_sensitivity_R(A, O_full_val)
                sp = compute_specificity_R(A, O_full_val, IDX_RANGE)
                score = (W_SENS*s + W_SPEC*sp) if (not np.isnan(sp)) and sp >= SPEC_TARGET else (sp if not np.isnan(sp) else -1.0)
            else:
                score = -1.0
            if score > best["score"]:
                best.update(score=score,
                            params=dict(WINDOW_SIZE=WINDOW_SIZE, N_EST=N_EST,
                                        MAX_SAMP=MAX_SAMP, MAX_FEAT=MAX_FEAT,
                                        scaler=scaler, model=iso),
                            contamination=c_best,
                            val_lengths=val_lengths,
                            val_scores=val_scores,
                            Yval=Yval,
                            thr_val=thr)
        else:
            if best["params"] is None:
                best.update(score=0.0,
                            params=dict(WINDOW_SIZE=WINDOW_SIZE, N_EST=N_EST,
                                        MAX_SAMP=MAX_SAMP, MAX_FEAT=MAX_FEAT,
                                        scaler=scaler, model=iso),
                            contamination=0.02,
                            val_lengths=[],
                            val_scores=np.array([]),
                            Yval=np.array([]),
                            thr_val=np.nan)

    if best["params"] is None:
        print("  No training/validation features; skipping."); continue

    P = best["params"]; WINDOW_SIZE = P["WINDOW_SIZE"]
    print(f"  Best HP → window={WINDOW_SIZE}, n_estimators={P['N_EST']}, "
          f"max_samples={P['MAX_SAMP']}, max_features={P['MAX_FEAT']}, "
          f"contamination≈{best['contamination']:.3f}")

    # --------- PER-SIM VALIDATION METRICS (selected HP) ---------
    if len(best["val_scores"]):
        yhat_v = (best["val_scores"] <= best["thr_val"]).astype(int)
        splits_yhat = split_by_lengths(yhat_v, best["val_lengths"])
        splits_y    = split_by_lengths(best["Yval"],  best["val_lengths"])
        for d, yh, y in zip(val_sims, splits_yhat, splits_y):
            s_i, sp_i = sens_spec(y, yh)
            pod_i = pod_anyhit(yh, y)
            tim_i = timeliness_single(yh, y)
            rows_val.append(dict(
                split="val", signal=S, sim=d["sim"], window=WINDOW_SIZE,
                n_estimators=P['N_EST'], max_samples=P['MAX_SAMP'], max_features=P['MAX_FEAT'],
                contamination=best['contamination'], thr=best['thr_val'],
                sens=s_i, spec=sp_i, pod=pod_i, timeliness=tim_i, n_points=len(yh)
            ))

    # --------- Refit tuned model on all TRAIN data with chosen params ---------
    Xtr = build_train_matrix(WINDOW_SIZE)
    scaler = P["scaler"]
    Xtr_s = scaler.transform(Xtr)

    iso_tuned = IsolationForest(
        n_estimators=P['N_EST'],
        contamination=best['contamination'],
        max_samples=P['MAX_SAMP'],
        max_features=P['MAX_FEAT'],
        random_state=RNG_STATE,
        n_jobs=-1,
        bootstrap=False,
    ).fit(Xtr_s)
    print(f"  Isolation Forest trained on normal patterns from first {TRAIN_YEARS} years")

    # --------- Build TEST features for exact R window [ABS_START:ABS_END) ---------
    per_sim_preds, per_sim_labels = [], []
    Xte_concat, Yte_concat = [], []

    for d in test_sims_final:
        x = d["x"]; y = d["y"]

        # Need WINDOW_SIZE-1 days of context immediately before ABS_START
        ctx_lo = ABS_START - (WINDOW_SIZE - 1)
        if ctx_lo < 0 or ABS_END > len(x):
            continue  # skip sims that cannot supply the absolute window

        x_ctx = x[ctx_lo:ABS_END]                # includes context + target window
        feats = create_features(x_ctx, WINDOW_SIZE)
        if len(feats) != WIN_LEN:
            continue  # safety check

        y_win = y[ABS_START:ABS_END].astype(int) # align labels exactly to window

        Xte_concat.append(feats)
        Yte_concat.append(y_win)

    Xte = np.concatenate(Xte_concat) if Xte_concat else np.empty((0,20), np.float32)
    Yte = np.concatenate(Yte_concat) if Yte_concat else np.empty((0,), np.int32)
    print(f"  Testing (absolute window {ABS_START}:{ABS_END-1}): {len(Xte)} feature vectors, "
          f"{Yte.sum()} outbreaks ({100*Yte.mean():.1f}%)" if len(Yte) else "  Testing: 0")

    Xte_s = scaler.transform(Xte) if len(Xte) else Xte

    # --------- Predict on TEST ---------
    if len(Xte_s):
        test_scores = iso_tuned.decision_function(Xte_s)  # higher = more normal
        thr_test = np.percentile(test_scores, best['contamination']*100)
        yhat_concat = (test_scores <= thr_test).astype(int)

        # Split back per-simulation (each sim contributes exactly WIN_LEN rows)
        ofs = 0; per_sim_preds.clear(); per_sim_labels.clear()
        for y_win in Yte_concat:
            per_sim_preds.append(yhat_concat[ofs:ofs+WIN_LEN]); ofs += WIN_LEN
            per_sim_labels.append(y_win)

        A = np.stack(per_sim_preds, axis=1)  # shape [WIN_LEN, n_sims]
        O = np.stack(per_sim_labels, axis=1)

        # Save alarm and outbreak sequences
        alarm_filename = f"isolation_forest_alarms_signal_{S}.csv"
        pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(alarm_filename, index=False)
        print(f"Saved alarm sequences: {alarm_filename}")
        outbreak_filename = f"isolation_forest_outbreaks_signal_{S}.csv"
        pd.DataFrame(O, columns=[f"sim_{i}" for i in range(O.shape[1])]).to_csv(outbreak_filename, index=False)
        print(f"Saved outbreak sequences: {outbreak_filename}")

        # O_full for R-comparator (full series; A covers last WIN_LEN days)
        O_full_list = [d["y"] for d in test_sims_final]
        O_full = np.stack(O_full_list, axis=1)

        # R-comparator metrics (exactly like your R helper definitions)
        fpr   = compute_fpr_R(A, O_full, IDX_RANGE)
        spec_R = compute_specificity_R(A, O_full, IDX_RANGE)
        sens_R = compute_sensitivity_R(A, O_full)
        pod_R  = compute_pod_R(A, O_full)
        tim_R  = compute_timeliness_R(A, O_full, days=DAYS, years=YEARS)
        print(f"R-COMPARATOR → Sens={sens_R:.3f}, Spec={spec_R:.3f}, FPR={fpr:.3f}, POD={pod_R:.3f}, Tim={tim_R:.3f}")

        # Original metrics on the same window
        sens = compute_sensitivity(A, O)
        spec = compute_specificity(A, O)
        pod  = compute_pod(A, O)
        tim  = compute_timeliness(A, O)
        print(f"ORIGINAL → Sens={sens:.3f}, Spec={spec:.3f}, POD={pod:.3f}, Tim={tim:.3f}")

        # Per-sim TEST rows
        for d, yh, ytrue in zip(test_sims_final, per_sim_preds, per_sim_labels):
            s_i, sp_i = sens_spec(ytrue, yh)
            pod_i = pod_anyhit(yh, ytrue)
            tim_i = timeliness_single(yh, ytrue)
            rows_test.append(dict(
                split="test", signal=S, sim=d["sim"], window=WINDOW_SIZE,
                n_estimators=P['N_EST'], max_samples=P['MAX_SAMP'], max_features=P['MAX_FEAT'],
                contamination=best['contamination'], thr=thr_test,
                sens=s_i, spec=sp_i, pod=pod_i, timeliness=tim_i, n_points=len(yh)
            ))

        # Store summary
        summary_all[S] = dict(
            sensitivity=sens_R, specificity=spec_R, fpr=fpr, pod=pod_R, timeliness=tim_R,
            contamination=best['contamination'],
            win=WINDOW_SIZE, n_estimators=P['N_EST'], max_samples=P['MAX_SAMP'], max_features=P['MAX_FEAT'],
            sensitivity_orig=sens, specificity_orig=spec, pod_orig=pod, timeliness_orig=tim
        )
    else:
        print("No test features.")

# ===== SUMMARY =====
if summary_all:
    df = pd.DataFrame.from_dict(summary_all, orient="index")
    print("\n=== SUMMARY (ALL DAYS) ==="); print(df); print("\nMeans:\n", df.mean(numeric_only=True))
    _if_out = f"IsolationForest_Tuned_all_days_per_sig.csv"
    df.to_csv(_if_out); print(f"Saved: {_if_out}")
    os.makedirs("results", exist_ok=True)
    _if_std = f"results/IsolationForest_Tuned_per_sig_big_{MAG_TAG}.csv"
    df.to_csv(_if_std); print(f"Saved: {_if_std}")

    combined_summary = {}
    for signal in summary_all.keys():
        combined_summary[f"Signal_{signal}"] = {
            "Sensitivity_All": summary_all[signal]["sensitivity"],
            "Specificity_All": summary_all[signal]["specificity"],
            "POD_All": summary_all[signal]["pod"],
            "Timeliness_All": summary_all[signal]["timeliness"],
            "Contamination_All": summary_all[signal]["contamination"],
            "Win": summary_all[signal]["win"],
            "n_estimators": summary_all[signal]["n_estimators"],
            "max_samples": summary_all[signal]["max_samples"],
            "max_features": summary_all[signal]["max_features"],
        }
    pd.DataFrame.from_dict(combined_summary, orient="index").to_csv(f"IsolationForest_Tuned_results_per_sig_{MAG_TAG}.csv")
    print(f"Combined results saved to: IsolationForest_Tuned_results_per_sig_{MAG_TAG}.csv")

# ===== PER-SIM EXPORTS =====
if rows_val:
    dfv = pd.DataFrame(rows_val)
    dfv.sort_values(["signal","sim","split"], inplace=True)
    dfv.to_csv(f"IsolationForest_big_per_sim_val_{MAG_TAG}.csv", index=False)
    print(f"Saved: IsolationForest_big_per_sim_val_{MAG_TAG}.csv")
else:
    print("No per-sim VALIDATION rows to save.")

if rows_test:
    dft = pd.DataFrame(rows_test)
    dft.sort_values(["signal","sim","split"], inplace=True)
    dft.to_csv(f"IsolationForest_big_per_sim_test_{MAG_TAG}.csv", index=False)
    print(f"Saved: IsolationForest_big_per_sim_test_{MAG_TAG}.csv")
else:
    print("No per-sim TEST rows to save.")
