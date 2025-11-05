# Guide: Updating All Scripts to Use R-Comparator Metrics (Like Farrington)

## Overview

All algorithms should use the same R-comparator metrics as Farrington for fair comparison. This means:

1. **Use full time series** (`O_full`) for proper denominators
2. **Use `IDX_RANGE` (2206:2548)** for specificity/FPR calculations
3. **Use years 6-7 window** for timeliness calculations

## Shared Module

Use the `r_comparator_metrics.py` module which provides:
- `compute_sensitivity_R(A, O_full)`
- `compute_specificity_R(A, O_full, idx_range)`
- `compute_fpr_R(A, O_full, idx_range)`
- `compute_pod_R(A, O_full)`
- `compute_timeliness_R(A, O_full, days=7, years=7)`

## Key Changes Needed

### 1. Import the shared module

```python
from r_comparator_metrics import (
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R, IDX_RANGE
)
```

### 2. Build O_full (full time series) for test simulations

Instead of just using the tail/test period, you need the **full** outbreak sequences:

```python
# OLD (Original metrics):
O = np.stack(per_sim_labels, axis=1)  # Only test period

# NEW (R-comparator):
O_full_list = [d["y"] for d in test_sims]  # Full time series
O_full = np.stack(O_full_list, axis=1)  # [n_total_days, n_sims]
```

### 3. Replace metric calculations

```python
# OLD (Original metrics):
sens = compute_sensitivity_tail(A_tail, O_tail)
spec = compute_specificity_tail(A_tail, O_tail)
fpr = compute_fpr_tail(A_tail, O_tail)
pod = compute_pod_tail(A_tail, O_tail)
tim = compute_timeliness_tail(A_tail, O_tail)

# NEW (R-comparator):
sens = compute_sensitivity_R(A, O_full)
spec = compute_specificity_R(A, O_full, IDX_RANGE)
fpr = compute_fpr_R(A, O_full, IDX_RANGE)
pod = compute_pod_R(A, O_full)
tim = compute_timeliness_R(A, O_full, days=7, years=7)
```

### 4. For Validation Tuning

For validation, you have two options:
- **Option A**: Use R-comparator style (recommended for consistency)
  - Need `O_full` for validation sims too
  - Use `compute_sensitivity_val()` and `compute_specificity_val()` with `O_full` parameter
  
- **Option B**: Use Original metrics for validation (simpler, but less consistent)
  - Keep using tail-only metrics for validation tuning
  - Only switch to R-comparator for final test evaluation

**Recommendation**: Use Option A (R-comparator for validation too) to match Farrington exactly.

## Files to Update

1. **LSTM.ipynb** - Replace `compute_*_tail()` functions
2. **OCSVM.ipynb** - Replace `metric_*()` functions
3. **IF_LSTM_ensemble.ipynb** - Replace `metric_*()` functions  
4. **Ensemble_3way_voting.ipynb** - Replace `metric_*()` functions

## Example: LSTM Notebook Changes

### Current Code (around line 862-893):
```python
def compute_fpr_tail(A_tail, O_tail):
    FP = np.sum((A_tail == 1) & (O_tail == 0))
    N0 = np.sum(O_tail == 0)
    return (FP / N0) if N0 > 0 else np.nan
# ... etc
```

### Replace with:
```python
from r_comparator_metrics import (
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R, IDX_RANGE
)
```

### Current test metric calculation (around line 1076-1083):
```python
A_tail = np.column_stack(A_tail_cols)  # [TAIL_DAYS, nsim_test]
O_tail = np.column_stack(O_tail_cols)  # [TAIL_DAYS, nsim_test]

sens = compute_sensitivity_tail(A_tail, O_tail)
spec = compute_specificity_tail(A_tail, O_tail)
# ...
```

### Replace with:
```python
A_tail = np.column_stack(A_tail_cols)  # [TAIL_DAYS, nsim_test]

# Build O_full (full time series)
O_full_list = [d["y"] for d in test_sims]
O_full = np.stack(O_full_list, axis=1)  # [n_total_days, n_sims]

# Use R-comparator metrics
sens = compute_sensitivity_R(A_tail, O_full)
spec = compute_specificity_R(A_tail, O_full, IDX_RANGE)
fpr = compute_fpr_R(A_tail, O_full, IDX_RANGE)
pod = compute_pod_R(A_tail, O_full)
tim = compute_timeliness_R(A_tail, O_full, days=7, years=7)

print(f"  R-COMPARATOR → Sens={sens:.3f}, Spec={spec:.3f}, FPR={fpr:.3f}, POD={pod:.3f}, Tim={tim:.3f}")
```

## Validation Metrics

For validation tuning, you'll also want R-comparator metrics. Update the validation metric calculations similarly:

```python
# Build O_full for validation sims
O_full_val_list = [d["y"] for d in val_sims]
O_full_val = np.stack(O_full_val_list, axis=1)

# During threshold tuning:
for threshold in thresholds:
    A_val = (scores >= threshold).astype(int)
    # Stack A_val to match O_full_val
    A_val_stacked = ...  # Shape: [val_period_length, n_val_sims]
    
    sens_val = compute_sensitivity_R(A_val_stacked, O_full_val)
    spec_val = compute_specificity_R(A_val_stacked, O_full_val, IDX_RANGE)
    
    # Use spec_val for tuning decisions
```

## Summary

The key principle: **R-comparator metrics use the FULL time series** for denominators, not just the test/validation period. This matches how Farrington evaluates performance in R.
