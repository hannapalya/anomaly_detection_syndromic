#!/usr/bin/env python3
"""
Helper module to load train/validation/test split indices.
Use this in your anomaly detection algorithms to get the predefined splits.
Supports both per-signal and global access.
"""

import json
import os
from typing import Dict, List, Tuple, Optional

def load_split_indices(split_file: str = "train_val_test_split.json") -> Dict:
    """
    Load the train/validation/test split indices from JSON file.
    
    Args:
        split_file: Path to the JSON file containing split indices
        
    Returns:
        Dictionary with keys: 'metadata', 'by_signal', 'validation', 'test', 'train'
        - 'by_signal': dict organized by signal number
        - 'validation', 'test', 'train': flat lists of all simulations
    """
    if not os.path.exists(split_file):
        raise FileNotFoundError(
            f"Split file not found: {split_file}\n"
            f"Run generate_train_val_test_split.py first to generate it."
        )
    
    with open(split_file, 'r') as f:
        return json.load(f)

def get_val_indices(split_file: str = "train_val_test_split.json", signal: Optional[int] = None) -> List[Dict]:
    """
    Get validation simulation indices.
    
    Args:
        split_file: Path to split file
        signal: If provided, return only indices for this signal (1-16)
    
    Returns:
        List of validation simulation dictionaries
    """
    data = load_split_indices(split_file)
    if signal is not None:
        if 'by_signal' in data and str(signal) in data['by_signal']:
            return data['by_signal'][str(signal)]['validation']
        elif 'by_signal' in data and signal in data['by_signal']:
            return data['by_signal'][signal]['validation']
        else:
            return []
    return data.get('validation', [])

def get_test_indices(split_file: str = "train_val_test_split.json", signal: Optional[int] = None) -> List[Dict]:
    """
    Get test simulation indices.
    
    Args:
        split_file: Path to split file
        signal: If provided, return only indices for this signal (1-16)
    
    Returns:
        List of test simulation dictionaries
    """
    data = load_split_indices(split_file)
    if signal is not None:
        if 'by_signal' in data and str(signal) in data['by_signal']:
            return data['by_signal'][str(signal)]['test']
        elif 'by_signal' in data and signal in data['by_signal']:
            return data['by_signal'][signal]['test']
        else:
            return []
    return data.get('test', [])

def get_train_indices(split_file: str = "train_val_test_split.json", signal: Optional[int] = None) -> List[Dict]:
    """
    Get training simulation indices.
    
    Args:
        split_file: Path to split file
        signal: If provided, return only indices for this signal (1-16)
    
    Returns:
        List of training simulation dictionaries
    """
    data = load_split_indices(split_file)
    if signal is not None:
        if 'by_signal' in data and str(signal) in data['by_signal']:
            return data['by_signal'][str(signal)]['train']
        elif 'by_signal' in data and signal in data['by_signal']:
            return data['by_signal'][signal]['train']
        else:
            return []
    return data.get('train', [])

def get_signal_split(signal: int, split_file: str = "train_val_test_split.json") -> Dict:
    """
    Get all splits (train/val/test) for a specific signal.
    
    Args:
        signal: Signal number (1-16)
        split_file: Path to split file
        
    Returns:
        Dictionary with keys 'validation', 'test', 'train'
    """
    data = load_split_indices(split_file)
    if 'by_signal' in data:
        # Try both string and int keys
        if str(signal) in data['by_signal']:
            return data['by_signal'][str(signal)]
        elif signal in data['by_signal']:
            return data['by_signal'][signal]
    return {'validation': [], 'test': [], 'train': []}

def get_val_test_signal_col_pairs(split_file: str = "train_val_test_split.json", signal: Optional[int] = None) -> Tuple[List[Tuple], List[Tuple]]:
    """
    Get (signal, column_index) tuples for validation and test sets.
    
    Args:
        split_file: Path to split file
        signal: If provided, return only pairs for this signal
    
    Returns:
        Tuple of (val_pairs, test_pairs) where each is a list of (signal, col_idx) tuples
    """
    val_indices = get_val_indices(split_file, signal)
    test_indices = get_test_indices(split_file, signal)
    
    val_pairs = [(sim['signal'], sim['sim_index']) for sim in val_indices]
    test_pairs = [(sim['signal'], sim['sim_index']) for sim in test_indices]
    
    return val_pairs, test_pairs

def filter_sims_by_indices(sims: List[Dict], indices: List[Dict]) -> List[Dict]:
    """
    Filter a list of simulation dictionaries to only include those in the indices.
    
    Args:
        sims: List of simulation dicts (with 'sim' key like 'sig1_sim0')
        indices: List of simulation dicts from the split file
        
    Returns:
        Filtered list of simulations that match the indices
    """
    index_set = {sim['identifier'] for sim in indices}
    return [s for s in sims if s.get('sim') in index_set or 
            f"sig{s.get('signal', '')}_sim{s.get('sim_idx', '')}" in index_set]

# Example usage:
if __name__ == "__main__":
    print("Loading split indices...")
    data = load_split_indices()
    
    print(f"\nMetadata:")
    for key, value in data['metadata'].items():
        print(f"  {key}: {value}")
    
    # Global access
    print(f"\n{'='*60}")
    print("Global Access (all signals):")
    print(f"{'='*60}")
    print(f"Validation simulations: {len(data.get('validation', []))}")
    print(f"Test simulations: {len(data.get('test', []))}")
    print(f"Train simulations: {len(data.get('train', []))}")
    
    # Per-signal access
    if 'by_signal' in data:
        print(f"\n{'='*60}")
        print("Per-Signal Access:")
        print(f"{'='*60}")
        signals = sorted([int(k) for k in data['by_signal'].keys()])
        for sig in signals[:3]:  # Show first 3 signals
            sig_data = get_signal_split(sig)
            print(f"\nSignal {sig}:")
            print(f"  Validation: {len(sig_data['validation'])}")
            print(f"  Test: {len(sig_data['test'])}")
            print(f"  Train: {len(sig_data['train'])}")
            print(f"  Examples - Val: {[s['identifier'] for s in sig_data['validation'][:2]]}")
            print(f"  Examples - Test: {[s['identifier'] for s in sig_data['test'][:2]]}")
        if len(signals) > 3:
            print(f"\n  ... and {len(signals) - 3} more signals")
    
    # Example: Get pairs for a specific signal
    val_pairs, test_pairs = get_val_test_signal_col_pairs(signal=1)
    print(f"\n{'='*60}")
    print(f"Signal 1 pairs:")
    print(f"  Validation pairs: {len(val_pairs)}")
    print(f"  Examples: {val_pairs[:3]}")
    print(f"  Test pairs: {len(test_pairs)}")
    print(f"  Examples: {test_pairs[:3]}")
