#!/usr/bin/env python3
"""
Example: How to use the train/validation/test split indices in your algorithms.
This shows how to integrate the predefined splits into IsolationForest.py or similar scripts.
"""

import os
import numpy as np
import pandas as pd
from load_split_indices import get_val_indices, get_test_indices, get_train_indices, filter_sims_by_indices

# Example: Modified version of how you'd use it in IsolationForest.py
DATA_DIR = "signal_datasets_large"
SIGNALS = list(range(1, 17))
RNG_STATE = 42

def load_data(sig):
    """Load signal data (same as in IsolationForest.py)"""
    X = pd.read_csv(os.path.join(DATA_DIR, f"simulated_totals_sig{sig}.csv"))
    Y = (pd.read_csv(os.path.join(DATA_DIR, f"simulated_outbreaks_sig{sig}.csv")) > 0).astype(int)
    for c in ["date", "Date", "ds", "timestamp"]:
        if c in X.columns: X = X.drop(columns=[c])
        if c in Y.columns: Y = Y.drop(columns=[c])
    return X, Y

# Load the predefined split indices
val_indices = get_val_indices()
test_indices = get_test_indices()
train_indices = get_train_indices()

print("=" * 60)
print("Example: Using Predefined Split Indices")
print("=" * 60)

# Create sets for fast lookup
val_set = {(sim['signal'], sim['sim_index']) for sim in val_indices}
test_set = {(sim['signal'], sim['sim_index']) for sim in test_indices}
train_set = {(sim['signal'], sim['sim_index']) for sim in train_indices}

print(f"Validation set: {len(val_set)} simulations")
print(f"Test set: {len(test_set)} simulations")
print(f"Train set: {len(train_set)} simulations")

# Example: Collect simulations into train/val/test groups
train_sims = []
val_sims = []
test_sims = []

for sig in SIGNALS:
    print(f"\nProcessing Signal {sig}...")
    Xsig, Ysig = load_data(sig)
    
    for sim_idx, col in enumerate(Xsig.columns):
        x = Xsig[col].to_numpy(np.float32, copy=False)
        y = Ysig[col].to_numpy(np.int32, copy=False)
        
        # Check which set this simulation belongs to
        sim_key = (sig, sim_idx)
        sim_dict = {
            'signal': sig,
            'sim_index': sim_idx,
            'sim': f'sig{sig}_sim{sim_idx}',
            'x': x,
            'y': y
        }
        
        if sim_key in val_set:
            val_sims.append(sim_dict)
        elif sim_key in test_set:
            test_sims.append(sim_dict)
        elif sim_key in train_set:
            train_sims.append(sim_dict)
        # If not in any set, it might be filtered out or used for training

print(f"\n{'='*60}")
print("Summary:")
print(f"  Train simulations collected: {len(train_sims)}")
print(f"  Validation simulations collected: {len(val_sims)}")
print(f"  Test simulations collected: {len(test_sims)}")
print(f"{'='*60}")

# Example: Show some validation simulations
print("\nSample Validation Simulations:")
for sim in val_sims[:3]:
    print(f"  - {sim['sim']}: Signal {sim['signal']}, Shape {sim['x'].shape}")

print("\nSample Test Simulations:")
for sim in test_sims[:3]:
    print(f"  - {sim['sim']}: Signal {sim['signal']}, Shape {sim['x'].shape}")

# Alternative approach: Filter existing sims list
print(f"\n{'='*60}")
print("Alternative: Filtering a list of all simulations")
print(f"{'='*60}")

# Build all sims first (as you might do in your code)
all_sims = []
for sig in SIGNALS[:3]:  # Just first 3 signals for demo
    Xsig, Ysig = load_data(sig)
    for sim_idx, col in enumerate(Xsig.columns):
        x = Xsig[col].to_numpy(np.float32, copy=False)
        y = Ysig[col].to_numpy(np.int32, copy=False)
        all_sims.append({
            'signal': sig,
            'sim_idx': sim_idx,
            'sim': f'sig{sig}_sim{sim_idx}',
            'x': x,
            'y': y
        })

print(f"Total simulations collected: {len(all_sims)}")

# Filter using the helper function
filtered_val = filter_sims_by_indices(all_sims, val_indices)
filtered_test = filter_sims_by_indices(all_sims, test_indices)

print(f"Validation sims in first 3 signals: {len(filtered_val)}")
print(f"Test sims in first 3 signals: {len(filtered_test)}")

print("\n" + "="*60)
print("Now you can use train_sims, val_sims, and test_sims in your algorithm!")
print("="*60)
