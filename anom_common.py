#!/usr/bin/env python3
"""
Common helpers for anomaly-detection scripts:

- Data loading
- Sim splits
- Feature creation (tabular)
- R-comparator metrics
- Validation threshold tuning, per-sim metrics
"""

import os
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix

# ===== CONFIG (same as your IF script) =====
DATA_DIR      = os.environ.get("SYND_DATA_DIR", "big_signal_datasets_small")
# Tag derived from DATA_DIR, used by runners that want a magnitude-aware filename
# e.g. DATA_DIR='big_signal_datasets_large' -> MAG_TAG='large'.
def _derive_mag_tag(d):
    bn = os.path.basename(d.rstrip("/"))
    for tag in ("small", "medium", "large"):
        if bn.endswith("_" + tag):
            return tag
    return bn
MAG_TAG       = _derive_mag_tag(DATA_DIR)
SIGNALS       = list(range(1, 17))
DAYS_PER_YEAR = 364
TRAIN_YEARS   = 6
TRAIN_DAYS    = TRAIN_YEARS * DAYS_PER_YEAR
VALID_DAYS    = 49 * 7
RNG_STATE     = 42

# Validation target
SPEC_TARGET    = 0.97
W_SENS, W_SPEC = 2.0, 3.0

# Comparator constants (mirrors R)
DAYS = 7
YEARS = 7
ABS_START = 2205
ABS_END   = 2548               # end exclusive
WIN_LEN   = ABS_END - ABS_START
IDX_RANGE = np.arange(ABS_START, ABS_END, dtype=int)


# ===== DATA / SPLIT HELPERS =====
def load_data(sig):
    X = pd.read_csv(os.path.join(DATA_DIR, f"simulated_totals_sig{sig}.csv"))
    Y = (pd.read_csv(os.path.join(DATA_DIR, f"simulated_outbreaks_sig{sig}.csv")) > 0).astype(int)
    for c in ["date", "Date", "ds", "timestamp"]:
        if c in X.columns:
            X = X.drop(columns=[c])
        if c in Y.columns:
            Y = Y.drop(columns=[c])
    return X, Y


def split_60_20_20(sims, rng):
    """Shuffle sims and split 60% train, 20% val, 20% test."""
    idx = np.arange(len(sims))
    rng.shuffle(idx)
    n = len(sims)
    n_train = int(round(0.60 * n))
    n_val   = int(round(0.20 * n))
    n_test  = n - n_train - n_val

    train_idx = idx[:n_train]
    val_idx   = idx[n_train:n_train+n_val]
    test_idx  = idx[n_train+n_val:]

    train_sims = [sims[i] for i in train_idx]
    val_sims   = [sims[i] for i in val_idx]
    test_sims  = [sims[i] for i in test_idx]
    return train_sims, val_sims, test_sims


# ===== TABULAR FEATURE CREATION (vectorized via stride_tricks) =====
def _rolling_windows(a, w):
    """Return a (n - w + 1, w) strided view of 1-D array *a*."""
    shape = (a.shape[0] - w + 1, w)
    strides = (a.strides[0], a.strides[0])
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def create_features(series, window_size):
    """Build 20-dim tabular feature matrix.

    Output shape: (len(series) - window_size + 1, 20).
    Uses numpy stride_tricks for fully vectorized rolling statistics —
    no Python-level per-day loop.
    """
    s = np.asarray(series, dtype=np.float64)
    n = len(s)
    W = window_size
    out_len = n - W + 1
    if out_len <= 0:
        return np.empty((0, 20), dtype=np.float32)

    # ---- Windowed views (no copy) ----
    winsW = _rolling_windows(s, W)                     # (out_len, W)
    # 7-day window: may start before the W-window region, so build from
    # the full series offset to align the LAST element with winsW rows.
    start7 = W - 1  # first output index in the original series
    # For each output row i, the 7-day window ends at series[start7+i]
    # and starts at max(0, start7+i-6).  When start7+i >= 6 the window is
    # full-length 7, otherwise shorter.  Build full-length view for i>=6-start7.
    wins7_full = _rolling_windows(s, 7)  # (n-6, 7) — row j covers s[j:j+7]
    # wins7_full row corresponding to output row i: j = start7 + i - 6
    # valid when j >= 0, i.e., i >= 6 - start7.  For W >= 7, start7 >= 6
    # so all rows are valid.  For W < 7 we handle partial below.

    cur = s[W - 1:]  # (out_len,) — current-day value for each output row

    # ---- Rolling stats over W-day window (axis=1 vectorised) ----
    meanW = winsW.mean(axis=1)
    maxW  = winsW.max(axis=1)
    minW  = winsW.min(axis=1)
    stdW  = winsW.std(axis=1, ddof=0)
    medW  = np.median(winsW, axis=1)
    madW  = np.median(np.abs(winsW - medW[:, None]), axis=1)
    madW  = np.where(madW > 0, madW, 1e-6)
    # Rank: fraction of window values <= cur
    rankW = (winsW <= cur[:, None]).sum(axis=1) / W

    # ---- Rolling stats over 7-day window ----
    if W >= 7:
        # wins7_full row j=start7+i-6.  start7=W-1, so j=W-7+i
        j_off = W - 7
        wins7 = wins7_full[j_off: j_off + out_len]  # (out_len, 7)
        mean7 = wins7.mean(axis=1)
        max7  = wins7.max(axis=1)
        std7  = wins7.std(axis=1, ddof=0)
        rank7 = (wins7 <= cur[:, None]).sum(axis=1) / 7.0
    else:
        # W < 7: 7-day window is partially filled for early rows.
        # Fall back to pandas rolling for the 7-day stats only.
        ps = pd.Series(s)
        r7 = ps.rolling(7, min_periods=1)
        _m7 = r7.mean().values[W - 1:]
        _x7 = r7.max().values[W - 1:]
        _s7 = r7.std(ddof=0).values[W - 1:]
        _s7 = np.nan_to_num(_s7)
        mean7 = _m7[:out_len]; max7 = _x7[:out_len]; std7 = _s7[:out_len]
        rank7 = r7.apply(lambda x: np.sum(x <= x[-1]) / len(x), raw=True).values[W - 1:]
        rank7 = rank7[:out_len]

    # ---- Lagged differences ----
    diff1  = np.empty(out_len)
    diff7  = np.empty(out_len)
    diff14 = np.empty(out_len)
    idx = np.arange(W - 1, n)  # absolute indices
    diff1[:] = s[idx] - s[np.clip(idx - 1, 0, n - 1)]
    if W - 1 < 1: diff1[0] = 0.0
    diff7[:] = s[idx] - s[np.clip(idx - 7, 0, n - 1)]
    diff7[:max(0, 7 - (W - 1))] = 0.0
    diff14[:] = s[idx] - s[np.clip(idx - 14, 0, n - 1)]
    diff14[:max(0, 14 - (W - 1))] = 0.0

    # ---- Linear detrending ----
    win_start = winsW[:, 0]
    slope = (cur - win_start) / (W + 1e-6)
    lin_resid = cur - (win_start + slope * (W - 1))

    # ---- Derived ratios ----
    eps = 1e-6
    ratio7   = cur / (mean7 + eps)
    ratioW   = cur / (meanW + eps)
    cv       = stdW / (meanW + eps)
    zscore   = (cur - meanW) / (stdW + eps)
    robust_z = 0.6745 * (cur - medW) / madW
    range_pos = (cur - minW) / (maxW - minW + eps)

    return np.column_stack([
        cur, mean7, meanW, max7, maxW,       # 0-4
        ratio7, ratioW,                       # 5-6
        rank7, rankW,                         # 7-8
        diff1, diff7, diff14,                 # 9-11
        std7, stdW, cv,                       # 12-14
        slope, lin_resid,                     # 15-16
        zscore, robust_z, range_pos,          # 17-19
    ]).astype(np.float32)


# ===== SEASONAL RESIDUAL FEATURE PIPELINE =====
def _seasonal_design_for_residuals(n, start=0, period=364, n_harmonics=2):
    """Annual + semi-annual harmonics + day-of-week design matrix.

    Returns (n, 1 + 2*n_harmonics + 6) design matrix matching the CUSUM/BOCPD
    baseline so that the residual-feature pipeline uses a consistent model.
    """
    t = np.arange(start, start + n).astype(float)
    cols = [np.ones(n)]
    for h in range(1, n_harmonics + 1):
        cols.append(np.cos(2 * np.pi * h * t / period))
        cols.append(np.sin(2 * np.pi * h * t / period))
    dow = (t.astype(int) % 7)
    for d in range(1, 7):
        cols.append((dow == d).astype(float))
    return np.column_stack(cols)


def fit_nb_baseline(x_train, train_days=None):
    """Fit a Negative-Binomial GLM with seasonal + day-of-week design on the
    outbreak-free training portion of a single sim's count series.

    Returns:
        params: coefficient vector
        dispersion: Pearson dispersion estimate (≥ 1)

    Falls back to log-mean intercept if the GLM fails to converge.
    """
    from statsmodels.genmod.generalized_linear_model import GLM
    from statsmodels.genmod import families
    x = np.asarray(x_train, dtype=float)
    n = len(x) if train_days is None else min(train_days, len(x))
    D = _seasonal_design_for_residuals(n, start=0)
    try:
        model = GLM(x[:n], D, family=families.NegativeBinomial(alpha=1.0)
                    ).fit(maxiter=50, disp=0)
        params = model.params
        mu_tr = np.exp(np.clip(D @ params, -20, 20))
        resid_sq = (x[:n] - mu_tr) ** 2 / np.maximum(mu_tr, 1e-6)
        dispersion = max(float(resid_sq.mean()), 1.0)
    except Exception:
        params = np.zeros(D.shape[1])
        params[0] = np.log(max(x[:n].mean(), 1.0))
        dispersion = 1.0
    return params, dispersion


def pearson_residuals(x_full, params, dispersion, start=0):
    """Compute Pearson residuals for the WHOLE series given a fitted baseline.

    r_t = (x_t - mu_t) / sqrt(dispersion * mu_t)
    Approximately N(0, 1) under the null hypothesis (no outbreak).
    """
    n = len(x_full)
    D = _seasonal_design_for_residuals(n, start=start)
    mu = np.exp(np.clip(D @ params, -20, 20))
    var = dispersion * mu + 1e-9
    return (x_full - mu) / np.sqrt(var)


def create_residual_features(series, train_days, window_size):
    """Residualise a count series against a seasonal+DOW NB baseline fitted on
    the FIRST `train_days` days, then build the standard 20-dim tabular
    feature matrix on the residual stream.

    This is the variant designed to help ML methods on highly seasonal signals
    (rhinitis, heat stroke, insect bites): the seasonal cycle is absorbed into
    the baseline, so the residual stream has no annual pattern for the
    sliding-window features to confound with outbreaks.

    Output shape: (len(series) - window_size + 1, 20).
    """
    s = np.asarray(series, dtype=np.float64)
    params, dispersion = fit_nb_baseline(s[:train_days])
    resid = pearson_residuals(s, params, dispersion, start=0)
    # The 20-dim feature builder treats inputs as counts; residuals can be
    # negative, but the engineered features (mean/std/diffs/z-score) are
    # all linear or scale-invariant and remain well-defined.
    return create_features(resid, window_size)


# ===== BASIC METRICS =====
def sens_spec(y_true, y_pred):
    if len(y_true) == 0:
        return (np.nan, np.nan)
    TN, FP, FN, TP = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    sens = TP/(TP+FN) if (TP+FN) > 0 else np.nan
    spec = TN/(TN+FP) if (TN+FP) > 0 else np.nan
    return sens, spec


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
        y = Oa[:, j]
        a = A[:, j]
        idx_out = np.where(y > 0)[0]
        if len(idx_out) == 0:
            score += 1.0
            continue
        idx_hit = np.where((a == 1) & (y > 0))[0]
        if len(idx_hit) == 0:
            score += 1.0
            continue
        r1, r2 = int(idx_out[0]), int(idx_out[-1])
        obs = int(idx_hit[0])
        score += (obs - r1) / (r2 - r1 + 1)
    return score / J


# ===== R-COMPARATOR METRICS =====
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
    nsim = A.shape[1]
    n_alarm = A.shape[0]
    n_out = O_full.shape[0]
    miss = 0
    score = 0.0

    w_start = 52*days*(years-1) + 3*days
    w_end   = 52*days*years
    w = slice(w_start, w_end)

    for j in range(nsim):
        idx = np.where(O_full[w, j] > 0)[0]
        if idx.size == 0:
            miss += 1
            continue
        r1 = w_start + idx.min()
        r2 = w_start + idx.max()

        alarm_hit = np.where((A[:, j] == 1) & (O_full[-n_alarm:, j] > 0))[0]
        if alarm_hit.size:
            obs_idx = n_out - n_alarm + int(alarm_hit[0])
            score += (obs_idx - r1) / (r2 - r1 + 1)
        else:
            miss += 1

    return (score + miss) / nsim if nsim > 0 else np.nan


# ===== PER-SIM HELPERS =====
def split_by_lengths(arr, lengths):
    out = []
    i = 0
    total = len(arr)
    for L in lengths:
        out.append(arr[i:min(i+L, total)])
        i += L
    return out


def pod_anyhit(yhat, ytrue):
    return float(int(((yhat == 1) & (ytrue == 1)).any())) if (ytrue == 1).any() else np.nan


def timeliness_single(yhat, ytrue):
    idx_out = np.where(ytrue > 0)[0]
    if len(idx_out) == 0:
        return np.nan
    idx_hit = np.where((ytrue > 0) & (yhat > 0))[0]
    if len(idx_hit) == 0:
        return 1.0
    r1, r2 = int(idx_out[0]), int(idx_out[-1])
    obs = int(idx_hit[0])
    return (obs - r1) / (r2 - r1 + 1)


# ===== CONTAMINATION TUNING (shared) =====
def tune_contamination_threshold(val_sims, val_lengths, decision_scores,
                                 spec_target=SPEC_TARGET,
                                 w_sens=W_SENS, w_spec=W_SPEC):
    """
    Tune contamination threshold using R-comparator metrics on VALIDATION TAILS.
    decision_scores: concatenated scores for all validation sims (higher = more normal).
    Each sim contributes length == VALID_DAYS if not skipped.
    """
    if len(decision_scores) == 0 or len(val_sims) == 0:
        return 0.02  # fallback

    O_full_list = [d["y"] for d in val_sims]
    O_full = np.stack(O_full_list, axis=1)

    best_c, best_score = None, -1.0
    grid = [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05, 0.06, 0.07]

    for c in grid:
        thr = np.percentile(decision_scores, c * 100)
        yhat = (decision_scores <= thr).astype(int)

        A_list, offset = [], 0
        for L in val_lengths:
            if L > 0:
                A_list.append(yhat[offset:offset+L])
                offset += L
        if not A_list or len(A_list) != len(val_sims):
            continue

        A = np.column_stack(A_list)  # [VALID_DAYS, n_val_sims]

        sp = compute_specificity_R(A, O_full, IDX_RANGE)
        s  = compute_sensitivity_R(A, O_full)

        if (not np.isnan(sp)) and sp >= spec_target:
            score = w_sens*s + w_spec*sp
            if score > best_score:
                best_c, best_score = c, score

    if best_c is None:
        # Fallback: pick contamination giving highest specificity
        best_sp, best_c = -1.0, 0.02
        for c in [0.01, 0.015, 0.02, 0.025, 0.03, 0.035, 0.04, 0.045, 0.05]:
            thr = np.percentile(decision_scores, c * 100)
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
        print("NOTE: no contamination_hit the specificity target; chose most specific fallback.")
    return best_c
