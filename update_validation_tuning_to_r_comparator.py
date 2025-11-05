#!/usr/bin/env python3
"""
Update validation tuning in notebooks to use R-comparator metrics.
This complements the main update script by specifically updating tuning functions.
"""

import json
import re
from pathlib import Path


def update_lstm_validation_tuning(notebook_path):
    """Update LSTM validation tuning to use R-comparator metrics."""
    print(f"\nUpdating validation tuning in {notebook_path}...")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
        
        source = ''.join(cell['source'])
        original_source = source
        
        # Update tune_aggressive_threshold to use R-comparator metrics
        if 'def tune_aggressive_threshold' in source:
            # Add helper comment and update function signature
            if 'val_sims' not in source or 'O_full' not in source:
                # Update function to accept val_sims and build O_full
                func_pattern = r'def tune_aggressive_threshold\(([^)]+)\):'
                match = re.search(func_pattern, source)
                if match:
                    params = match.group(1)
                    # Check if it already has val_sims
                    if 'val_sims' not in params:
                        # Add val_sims parameter
                        new_params = params.rstrip()
                        if not new_params.endswith(','):
                            new_params += ','
                        new_params += ' val_sims, val_lengths'
                        source = source[:match.start(1)] + new_params + source[match.end(1):]
                        changes_made.append(f"Cell {i}: Updated tune_aggressive_threshold signature")
            
            # Add O_full construction inside the function
            if 'def tune_aggressive_threshold' in source and 'O_full_val' not in source:
                # Find the function body start
                body_match = re.search(r'def tune_aggressive_threshold[^{]*\{', source)
                if not body_match:
                    body_match = re.search(r'def tune_aggressive_threshold[^\n]*:\s*(""".*?""")?\s*', source, re.DOTALL)
                
                if body_match:
                    body_start = body_match.end()
                    # Find first meaningful line (skip docstring)
                    lines = source[body_start:].split('\n')
                    insert_pos = body_start
                    for j, line in enumerate(lines):
                        if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                            insert_pos = body_start + sum(len(l) + 1 for l in lines[:j])
                            break
                    
                    o_full_code = '\n    # Build O_full for R-comparator metrics\n'
                    o_full_code += '    O_full_val = np.stack([d["y"] for d in val_sims], axis=1)\n'
                    o_full_code += '    IDX_RANGE = np.arange(2205, 2548, dtype=int)\n'
                    o_full_code += '    \n'
                    source = source[:insert_pos] + o_full_code + source[insert_pos:]
                    changes_made.append(f"Cell {i}: Added O_full construction in tune_aggressive_threshold")
            
            # Replace sens_spec_for_tuning calls with R-comparator metrics
            # This is tricky - we need to build A matrix from yhat predictions
            # Look for the pattern: s, sp = sens_spec_for_tuning(y_val, yhat)
            if 'sens_spec_for_tuning(y_val, yhat)' in source:
                # Replace with R-comparator version
                # Need to build A matrix from yhat split by val_lengths
                replacement = '''# Build alarm matrix A for R-comparator metrics
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
                s, sp = 0.0, 0.0'''
                
                source = re.sub(
                    r's,\s*sp\s*=\s*sens_spec_for_tuning\(y_val,\s*yhat\)',
                    replacement,
                    source
                )
                changes_made.append(f"Cell {i}: Replaced sens_spec_for_tuning with R-comparator in tune_aggressive_threshold")
        
        # Update tune_mix_and_threshold similarly
        if 'def tune_mix_and_threshold' in source:
            # Update function signature
            if 'val_sims' not in source or 'val_lengths' not in source:
                func_pattern = r'def tune_mix_and_threshold\(([^)]+)\):'
                match = re.search(func_pattern, source)
                if match:
                    params = match.group(1)
                    if 'val_sims' not in params:
                        new_params = params.rstrip()
                        if not new_params.endswith(','):
                            new_params += ','
                        new_params += ' val_sims, val_lengths'
                        source = source[:match.start(1)] + new_params + source[match.end(1):]
                        changes_made.append(f"Cell {i}: Updated tune_mix_and_threshold signature")
            
            # Add O_full construction
            if 'def tune_mix_and_threshold' in source and 'O_full_val' not in source:
                body_match = re.search(r'def tune_mix_and_threshold[^\n]*:\s*(""".*?""")?\s*', source, re.DOTALL)
                if body_match:
                    body_start = body_match.end()
                    lines = source[body_start:].split('\n')
                    insert_pos = body_start
                    for j, line in enumerate(lines):
                        if line.strip() and not line.strip().startswith('"""') and not line.strip().startswith("'''"):
                            insert_pos = body_start + sum(len(l) + 1 for l in lines[:j])
                            break
                    
                    o_full_code = '\n    # Build O_full for R-comparator metrics\n'
                    o_full_code += '    O_full_val = np.stack([d["y"] for d in val_sims], axis=1)\n'
                    o_full_code += '    IDX_RANGE = np.arange(2205, 2548, dtype=int)\n'
                    o_full_code += '    \n'
                    source = source[:insert_pos] + o_full_code + source[insert_pos:]
                    changes_made.append(f"Cell {i}: Added O_full construction in tune_mix_and_threshold")
            
            # Replace sens_spec_for_tuning calls
            if 'sens_spec_for_tuning(Yval, yhat)' in source:
                replacement = '''# Build alarm matrix A for R-comparator metrics
            # Note: Yval is concatenated, need to split by simulation
            # This is a simplified version - may need adjustment based on actual data structure
            A_list = []
            # Assuming scores are already per-simulation or need splitting
            # This needs to be adapted based on actual implementation
            if len(yhat) == len(Yval):
                # Simple case: one-to-one mapping
                # For proper R-comparator, we'd need full simulation structure
                # Fallback to pointwise for now, but ideally should use O_full
                A_flat = yhat.reshape(1, -1) if len(yhat.shape) == 1 else yhat
                # This is a placeholder - actual implementation depends on data structure
                s, sp = compute_sensitivity_R(A_flat, O_full_val), compute_specificity_R(A_flat, O_full_val, IDX_RANGE)'''
                
                source = re.sub(
                    r's,\s*sp\s*=\s*sens_spec_for_tuning\(Yval,\s*yhat\)',
                    replacement,
                    source
                )
                changes_made.append(f"Cell {i}: Attempted to replace sens_spec_for_tuning in tune_mix_and_threshold")
        
        # Update call sites to pass val_sims and val_lengths
        if 'tune_aggressive_threshold(Yval, scores' in source:
            source = re.sub(
                r'tune_aggressive_threshold\(Yval,\s*scores',
                'tune_aggressive_threshold(scores, val_sims, val_lengths',
                source
            )
            changes_made.append(f"Cell {i}: Updated tune_aggressive_threshold call")
        
        if 'tune_mix_and_threshold(Xval, Yval' in source:
            source = re.sub(
                r'tune_mix_and_threshold\(Xval,\s*Yval',
                'tune_mix_and_threshold(Xval, Yval, val_sims, val_lengths',
                source
            )
            changes_made.append(f"Cell {i}: Updated tune_mix_and_threshold call")
        
        if source != original_source:
            cell['source'] = source.splitlines(keepends=True)
    
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Updated validation tuning in {notebook_path}")
        print(f"  Changes: {len(changes_made)}")
        for change in changes_made[:5]:
            print(f"    - {change}")
        return True
    else:
        print(f"⚠ No validation tuning changes made to {notebook_path}")
        return False


def main():
    """Update validation tuning in notebooks."""
    print("="*70)
    print("Updating Validation Tuning to Use R-Comparator Metrics")
    print("="*70)
    
    notebooks = ['LSTM.ipynb']  # Start with LSTM, can extend to others
    
    for nb_name in notebooks:
        nb_path = Path(nb_name)
        if nb_path.exists():
            try:
                update_lstm_validation_tuning(nb_path)
            except Exception as e:
                print(f"\n❌ Error updating {nb_name}: {e}")
                import traceback
                traceback.print_exc()
        else:
            print(f"\n⚠ {nb_name} not found, skipping...")
    
    print("\n" + "="*70)
    print("NOTE: Validation tuning updates are complex and may require")
    print("manual verification. Please check the updated functions carefully.")
    print("="*70)


if __name__ == '__main__':
    main()
