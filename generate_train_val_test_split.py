#!/usr/bin/env python3
"""
Generate train/validation/test split indices for simulations.
Randomly selects 20 simulations for validation and 20 for testing PER SIGNAL.
Saves indices to JSON file for reuse across all algorithms.
"""

import os
import json
import numpy as np
import pandas as pd

# ===== CONFIG =====
# Try different possible data directories
POSSIBLE_DATA_DIRS = ["signal_datasets_large", "signal_datasets_medium"]
DATA_DIR = None
for d in POSSIBLE_DATA_DIRS:
    if os.path.exists(d) and os.path.exists(os.path.join(d, "simulated_totals_sig1.csv")):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        f"Could not find data directory. Tried: {POSSIBLE_DATA_DIRS}"
    )

SIGNALS = list(range(1, 17))  # Signals 1-16
RNG_STATE = 42
N_VAL_PER_SIGNAL = 20  # Number of simulations for validation per signal
N_TEST_PER_SIGNAL = 20  # Number of simulations for testing per signal

def load_signal_info(sig):
    """Load signal data and return list of valid simulation indices."""
    totals_path = os.path.join(DATA_DIR, f"simulated_totals_sig{sig}.csv")
    if not os.path.exists(totals_path):
        print(f"Warning: {totals_path} not found, skipping signal {sig}")
        return []
    
    X = pd.read_csv(totals_path)
    valid_sims = []
    
    for sim_idx, col in enumerate(X.columns):
        # Check if simulation has enough data (assuming at least some minimum length)
        x = X[col].to_numpy()
        if len(x) > 0 and not np.all(np.isnan(x)):
            valid_sims.append({
                "signal": sig,
                "sim_index": sim_idx,
                "column_name": col,
                "identifier": f"sig{sig}_sim{sim_idx}"
            })
    
    return valid_sims

def main():
    print("=" * 60)
    print("Generating Train/Validation/Test Split (PER SIGNAL)")
    print(f"Using data directory: {DATA_DIR}")
    print(f"Selecting {N_VAL_PER_SIGNAL} val + {N_TEST_PER_SIGNAL} test per signal")
    print("=" * 60)
    
    # Set random seed for reproducibility
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)
    
    # Organize by signal
    splits_by_signal = {}
    total_val = 0
    total_test = 0
    total_train = 0
    
    for sig in SIGNALS:
        print(f"\nProcessing Signal {sig}...")
        sims = load_signal_info(sig)
        
        if len(sims) < N_VAL_PER_SIGNAL + N_TEST_PER_SIGNAL:
            print(f"  ⚠ Warning: Signal {sig} has only {len(sims)} simulations, "
                  f"need at least {N_VAL_PER_SIGNAL + N_TEST_PER_SIGNAL}")
            print(f"  Skipping signal {sig}...")
            continue
        
        # Randomly shuffle simulations for this signal
        signal_sims = sims.copy()
        rng.shuffle(signal_sims)
        
        # Split into validation, test, and training sets (non-overlapping slices)
        val_indices = signal_sims[:N_VAL_PER_SIGNAL]
        test_indices = signal_sims[N_VAL_PER_SIGNAL:N_VAL_PER_SIGNAL + N_TEST_PER_SIGNAL]
        train_indices = signal_sims[N_VAL_PER_SIGNAL + N_TEST_PER_SIGNAL:]
        
        # Verify no overlap
        val_ids = {s['identifier'] for s in val_indices}
        test_ids = {s['identifier'] for s in test_indices}
        train_ids = {s['identifier'] for s in train_indices}
        
        overlap_val_test = val_ids & test_ids
        overlap_val_train = val_ids & train_ids
        overlap_test_train = test_ids & train_ids
        
        if overlap_val_test or overlap_val_train or overlap_test_train:
            raise ValueError(
                f"ERROR: Overlap detected in splits for signal {sig}!\n"
                f"  Val-Test overlap: {overlap_val_test}\n"
                f"  Val-Train overlap: {overlap_val_train}\n"
                f"  Test-Train overlap: {overlap_test_train}"
            )
        
        # Verify all simulations are accounted for
        total_split = len(val_indices) + len(test_indices) + len(train_indices)
        if total_split != len(sims):
            raise ValueError(
                f"ERROR: Split count mismatch for signal {sig}! "
                f"Expected {len(sims)}, got {total_split}"
            )
        
        splits_by_signal[sig] = {
            "validation": val_indices,
            "test": test_indices,
            "train": train_indices
        }
        
        total_val += len(val_indices)
        total_test += len(test_indices)
        total_train += len(train_indices)
        
        print(f"  ✓ Signal {sig}: {len(val_indices)} val, {len(test_indices)} test, "
              f"{len(train_indices)} train (total: {len(sims)}, ✓ no overlap)")
    
    # Create output structure organized by signal
    output = {
        "metadata": {
            "rng_state": RNG_STATE,
            "n_val_per_signal": N_VAL_PER_SIGNAL,
            "n_test_per_signal": N_TEST_PER_SIGNAL,
            "signals_processed": list(splits_by_signal.keys()),
            "total_signals": len(splits_by_signal),
            "total_val": total_val,
            "total_test": total_test,
            "total_train": total_train,
            "total_sims": total_val + total_test + total_train
        },
        "by_signal": splits_by_signal
    }
    
    # Also create flat lists for backward compatibility
    all_val = []
    all_test = []
    all_train = []
    
    for sig in sorted(splits_by_signal.keys()):
        all_val.extend(splits_by_signal[sig]["validation"])
        all_test.extend(splits_by_signal[sig]["test"])
        all_train.extend(splits_by_signal[sig]["train"])
    
    output["validation"] = all_val
    output["test"] = all_test
    output["train"] = all_train
    
    # Save to JSON file
    output_file = "train_val_test_split.json"
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"✓ Saved split indices to: {output_file}")
    
    # Create summary files
    summary_all = {
        "validation": [sim["identifier"] for sim in all_val],
        "test": [sim["identifier"] for sim in all_test],
        "train": [sim["identifier"] for sim in all_train]
    }
    
    summary_file = "train_val_test_split_summary.json"
    with open(summary_file, 'w') as f:
        json.dump(summary_all, f, indent=2)
    
    print(f"✓ Saved summary (identifiers only) to: {summary_file}")
    
    # Create per-signal summary
    summary_by_signal = {}
    for sig in sorted(splits_by_signal.keys()):
        summary_by_signal[sig] = {
            "validation": [sim["identifier"] for sim in splits_by_signal[sig]["validation"]],
            "test": [sim["identifier"] for sim in splits_by_signal[sig]["test"]],
            "train": [sim["identifier"] for sim in splits_by_signal[sig]["train"]]
        }
    
    summary_by_signal_file = "train_val_test_split_by_signal.json"
    with open(summary_by_signal_file, 'w') as f:
        json.dump(summary_by_signal, f, indent=2)
    
    print(f"✓ Saved per-signal summary to: {summary_by_signal_file}")
    
    # Print summary
    print(f"\n{'='*60}")
    print("SUMMARY:")
    print(f"  Signals processed: {len(splits_by_signal)}")
    print(f"  Total validation simulations: {total_val} ({N_VAL_PER_SIGNAL} per signal × {len(splits_by_signal)} signals)")
    print(f"  Total test simulations: {total_test} ({N_TEST_PER_SIGNAL} per signal × {len(splits_by_signal)} signals)")
    print(f"  Total training simulations: {total_train}")
    print(f"  Grand total: {total_val + total_test + total_train}")
    
    # Print examples from first signal
    if splits_by_signal:
        first_sig = sorted(splits_by_signal.keys())[0]
        print(f"\n{'='*60}")
        print(f"Sample from Signal {first_sig}:")
        print(f"  Validation: {[s['identifier'] for s in splits_by_signal[first_sig]['validation'][:3]]}")
        print(f"  Test: {[s['identifier'] for s in splits_by_signal[first_sig]['test'][:3]]}")
        print(f"  Train: {[s['identifier'] for s in splits_by_signal[first_sig]['train'][:3]]}")
    
    print(f"\n{'='*60}")
    print("Done! Use these indices in your anomaly detection algorithms.")
    print("=" * 60)

if __name__ == "__main__":
    main()
