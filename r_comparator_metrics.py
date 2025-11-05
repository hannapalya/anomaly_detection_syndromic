#!/usr/bin/env python3
"""
R-comparator metrics module - matches Farrington R evaluation methodology.
Use these functions for consistent metric calculation across all algorithms.

These metrics use:
- Full time series (O_full) for proper denominators
- IDX_RANGE for specificity/FPR calculations
- Years 6-7 window for timeliness calculations
"""

import numpy as np

# Constants matching R/Farrington
DAYS = 7
YEARS = 7
IDX_RANGE = np.arange(2205, 2548, dtype=int)  # R: 2206:2548 (1-based), Python: 2205:2548 (0-based)


def compute_fpr_R(A, O_full, idx_range=None):
    """
    Compute False Positive Rate using R-comparator methodology.
    
    Args:
        A: Alarm matrix [n_alarms, n_sims] (predictions on test period)
        O_full: Full outbreak matrix [n_total_days, n_sims] (entire time series)
        idx_range: Index range for denominator (default: IDX_RANGE)
    
    Returns:
        FPR = FP / N0, where N0 is non-outbreak days in idx_range
    """
    if idx_range is None:
        idx_range = IDX_RANGE
    
    n = A.shape[0]
    FP = np.sum((A == 1) & (O_full[-n:, :] == 0))
    N0 = np.sum(O_full[idx_range, :] == 0)
    return np.nan if N0 == 0 else FP / N0


def compute_specificity_R(A, O_full, idx_range=None):
    """
    Compute Specificity using R-comparator methodology.
    
    Args:
        A: Alarm matrix [n_alarms, n_sims]
        O_full: Full outbreak matrix [n_total_days, n_sims]
        idx_range: Index range for denominator (default: IDX_RANGE)
    
    Returns:
        Specificity = TN / N0, where N0 is non-outbreak days in idx_range
    """
    if idx_range is None:
        idx_range = IDX_RANGE
    
    n = A.shape[0]
    TN = np.sum((A == 0) & (O_full[-n:, :] == 0))
    N0 = np.sum(O_full[idx_range, :] == 0)
    return np.nan if N0 == 0 else TN / N0


def compute_sensitivity_R(A, O_full):
    """
    Compute Sensitivity using R-comparator methodology.
    
    Args:
        A: Alarm matrix [n_alarms, n_sims]
        O_full: Full outbreak matrix [n_total_days, n_sims]
    
    Returns:
        Sensitivity = TP / P, where P is ALL outbreak days in full time series
    """
    n = A.shape[0]
    TP = np.sum((A == 1) & (O_full[-n:, :] > 0))
    P = np.sum(O_full > 0)
    return np.nan if P == 0 else TP / P


def compute_pod_R(A, O_full):
    """
    Compute Power of Detection using R-comparator methodology.
    
    Args:
        A: Alarm matrix [n_alarms, n_sims]
        O_full: Full outbreak matrix [n_total_days, n_sims]
    
    Returns:
        POD = percentage of simulations with at least one true alarm
    """
    n = A.shape[0]
    per_sim_any = np.sum((A == 1) & (O_full[-n:, :] > 0), axis=0) > 0
    return per_sim_any.mean()


def compute_timeliness_R(A, O_full, days=DAYS, years=YEARS):
    """
    Compute Timeliness using R-comparator methodology.
    
    Uses outbreaks in years 6-7 window only (matching R implementation).
    
    Args:
        A: Alarm matrix [n_alarms, n_sims]
        O_full: Full outbreak matrix [n_total_days, n_sims]
        days: Days per week (default: 7)
        years: Total years (default: 7)
    
    Returns:
        Timeliness score: 0 = perfect (alarm at start), 1 = worst (missed)
    """
    nsim = A.shape[1]
    n_alarm = A.shape[0]
    n_out = O_full.shape[0]
    miss = 0
    score = 0.0
    
    w_start = 52 * days * (years - 1) + 3 * days
    w_end = 52 * days * years
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


def compute_all_metrics_R(A, O_full, idx_range=None, days=DAYS, years=YEARS):
    """
    Compute all R-comparator metrics at once.
    
    Returns:
        dict with keys: 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness'
    """
    if idx_range is None:
        idx_range = IDX_RANGE
    
    return {
        'sensitivity': compute_sensitivity_R(A, O_full),
        'specificity': compute_specificity_R(A, O_full, idx_range),
        'fpr': compute_fpr_R(A, O_full, idx_range),
        'pod': compute_pod_R(A, O_full),
        'timeliness': compute_timeliness_R(A, O_full, days, years)
    }


# Example usage for validation (where we have partial data)
def compute_sensitivity_val(A_tail, O_tail, O_full=None):
    """
    Compute sensitivity for validation period.
    If O_full is provided, uses full time series denominator (R-comparator style).
    Otherwise, uses only validation period denominator (Original style).
    """
    if O_full is not None:
        # R-comparator: use full time series for denominator
        n = A_tail.shape[0]
        TP = np.sum((A_tail == 1) & (O_full[-n:, :] > 0))
        P = np.sum(O_full > 0)
        return np.nan if P == 0 else TP / P
    else:
        # Original: use only validation period
        TP = np.sum((A_tail == 1) & (O_tail > 0))
        P = np.sum(O_tail > 0)
        return np.nan if P == 0 else TP / P


def compute_specificity_val(A_tail, O_tail, O_full=None, idx_range=None):
    """
    Compute specificity for validation period.
    If O_full and idx_range are provided, uses R-comparator methodology.
    Otherwise, uses only validation period (Original style).
    """
    if O_full is not None and idx_range is not None:
        # R-comparator: use idx_range for denominator
        n = A_tail.shape[0]
        TN = np.sum((A_tail == 0) & (O_full[-n:, :] == 0))
        N0 = np.sum(O_full[idx_range, :] == 0)
        return np.nan if N0 == 0 else TN / N0
    else:
        # Original: use only validation period
        TN = np.sum((A_tail == 0) & (O_tail == 0))
        N0 = np.sum(O_tail == 0)
        return np.nan if N0 == 0 else TN / N0
