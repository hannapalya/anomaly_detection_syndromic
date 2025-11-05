# Guide: Updating Validation Tuning to Use R-Comparator Metrics

## Overview

We've updated `IsolationForest.py` to use R-comparator metrics for validation tuning. This guide shows how to update the notebooks similarly.

## Key Principle

**Validation tuning should use the SAME evaluation methodology as testing** to ensure:
1. Thresholds are optimized for the same metric definition
2. Hyperparameters are selected based on R-comparator metrics
3. Results are directly comparable

## Changes Required

### For LSTM.ipynb

#### 1. Update `tune_aggressive_threshold()` function

**Current signature:**
```python
def tune_aggressive_threshold(y_val, anomaly_scores, spec_target=SPEC_TARGET, ...):
```

**New signature:**
```python
def tune_aggressive_threshold(anomaly_scores, val_sims, val_lengths, spec_target=SPEC_TARGET, ...):
```

**Add at the start of function body:**
```python
# Build O_full for R-comparator metrics
O_full_val = np.stack([d["y"] for d in val_sims], axis=1)
IDX_RANGE = np.arange(2205, 2548, dtype=int)
```

**Replace:**
```python
s, sp = sens_spec_for_tuning(y_val, yhat)
```

**With:**
```python
# Build alarm matrix A from yhat predictions (split by simulation lengths)
A_list = []
offset = 0
for L in val_lengths:
    if L > 0:
        A_list.append(yhat[offset:offset+L])
        offset += L

if A_list and len(A_list) == len(val_sims):
    max_len = max(len(a) for a in A_list)
    A_padded = [np.pad(a, (0, max_len - len(a)), mode='constant') if len(a) < max_len else a for a in A_list]
    A = np.column_stack(A_padded)
    s = compute_sensitivity_R(A, O_full_val)
    sp = compute_specificity_R(A, O_full_val, IDX_RANGE)
else:
    s, sp = 0.0, 0.0
```

#### 2. Update `tune_mix_and_threshold()` function

**Current signature:**
```python
def tune_mix_and_threshold(Xval, Yval, recon_val, mixes=MIX_GRID, ...):
```

**New signature:**
```python
def tune_mix_and_threshold(Xval, Yval, recon_val, val_sims, val_lengths, mixes=MIX_GRID, ...):
```

**Add at the start:**
```python
# Build O_full for R-comparator metrics
O_full_val = np.stack([d["y"] for d in val_sims], axis=1)
IDX_RANGE = np.arange(2205, 2548, dtype=int)
```

**Update the call to `tune_aggressive_threshold()`:**
```python
# OLD:
thr = tune_aggressive_threshold(Yval, scores, spec_target, w_sens, w_spec)

# NEW:
thr = tune_aggressive_threshold(scores, val_sims, val_lengths, spec_target, w_sens, w_spec)
```

**Replace `sens_spec_for_tuning(Yval, yhat)` calls:**
```python
# Build alarm matrix A (similar to tune_aggressive_threshold)
# ... (same pattern as above)
s = compute_sensitivity_R(A, O_full_val)
sp = compute_specificity_R(A, O_full_val, IDX_RANGE)
```

#### 3. Update call site

**Find:**
```python
best = tune_mix_and_threshold(
    Xval, Yval, val_rec,
    mixes=MIX_GRID, spec_target=SPEC_TARGET,
    w_sens=TUNE_WEIGHT_SENS, w_spec=TUNE_WEIGHT_SPEC
)
```

**Change to:**
```python
# Build val_lengths (validation window lengths per simulation)
val_lengths = []
for d in val_sims:
    x_tail = d["x"][-TAIL_DAYS:]
    y_tail = d["y"][-TAIL_DAYS:]
    Xv, Yv = make_seq_labels(x_tail, y_tail, SEQ, STRIDE)
    val_lengths.append(len(Yv))

best = tune_mix_and_threshold(
    Xval, Yval, val_rec, val_sims, val_lengths,
    mixes=MIX_GRID, spec_target=SPEC_TARGET,
    w_sens=TUNE_WEIGHT_SENS, w_spec=TUNE_WEIGHT_SPEC
)
```

### For OCSVM.ipynb

#### Update `tune_threshold()` function

**Current:**
```python
def tune_threshold(y_val, scores):
    ...
    s, sp = sens_spec(y_val, (scores <= thr).astype(int))
```

**New signature and implementation:**
```python
def tune_threshold(scores, val_sims, val_lengths, spec_target=SPEC_TARGET):
    # Build O_full
    O_full_val = np.stack([d["y"] for d in val_sims], axis=1)
    IDX_RANGE = np.arange(2205, 2548, dtype=int)
    
    ...
    yhat = (scores <= thr).astype(int)
    
    # Build alarm matrix A
    A_list = []
    offset = 0
    for L in val_lengths:
        if L > 0:
            A_list.append(yhat[offset:offset+L])
            offset += L
    
    if A_list and len(A_list) == len(val_sims):
        max_len = max(len(a) for a in A_list)
        A_padded = [np.pad(a, (0, max_len - len(a)), mode='constant') if len(a) < max_len else a for a in A_list]
        A = np.column_stack(A_padded)
        s = compute_sensitivity_R(A, O_full_val)
        sp = compute_specificity_R(A, O_full_val, IDX_RANGE)
    else:
        s, sp = 0.0, 0.0
```

### For Ensemble Notebooks

Similar pattern - update threshold tuning functions to:
1. Accept `val_sims` and `val_lengths` parameters
2. Build `O_full_val` from validation sims
3. Build alarm matrix `A` from predictions split by simulation
4. Use `compute_sensitivity_R()` and `compute_specificity_R()` instead of `sens_spec()`

## Summary

The pattern is consistent across all scripts:

1. **Function signature**: Add `val_sims, val_lengths` parameters
2. **Build O_full**: `O_full_val = np.stack([d["y"] for d in val_sims], axis=1)`
3. **Build alarm matrix**: Split predictions by `val_lengths` and stack into matrix
4. **Use R-comparator metrics**: Replace `sens_spec()` calls with `compute_sensitivity_R()` and `compute_specificity_R(A, O_full_val, IDX_RANGE)`
5. **Update call sites**: Pass `val_sims` and `val_lengths` when calling tuning functions

## Benefits

✅ **Consistency**: Validation and testing use the same metric definition  
✅ **Fair comparison**: Hyperparameters optimized for the same metrics used in final evaluation  
✅ **Matches Farrington**: Same evaluation methodology as the R implementation  

## Status

- ✅ **IsolationForest.py**: Updated to use R-comparator metrics for validation tuning
- ⏳ **LSTM.ipynb**: Needs manual update (guide provided above)
- ⏳ **OCSVM.ipynb**: Needs manual update
- ⏳ **Ensemble notebooks**: Need manual update
