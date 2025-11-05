#!/usr/bin/env python3
"""
Automated script to update Jupyter notebooks to use R-comparator metrics.
This script modifies the notebook JSON files to replace Original metrics with R-comparator metrics.
"""

import json
import re
import sys
from pathlib import Path


def update_lstm_notebook(notebook_path):
    """Update LSTM.ipynb to use R-comparator metrics."""
    print(f"\n{'='*70}")
    print(f"Updating {notebook_path}")
    print(f"{'='*70}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = []
    
    # Find the main code cell (usually the large one with all the logic)
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # 1. Add import at the top of the cell (after existing imports)
        if 'import' in source and 'r_comparator_metrics' not in source:
            # Find last import statement
            import_pattern = r'(import[^\n]+\n)'
            imports = re.findall(import_pattern, source)
            if imports:
                last_import_idx = source.rfind(imports[-1]) + len(imports[-1])
                new_import = "from r_comparator_metrics import (\n"
                new_import += "    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,\n"
                new_import += "    compute_pod_R, compute_timeliness_R, IDX_RANGE\n"
                new_import += ")\n\n"
                source = source[:last_import_idx] + new_import + source[last_import_idx:]
                changes_made.append(f"Cell {i}: Added r_comparator_metrics import")
        
        # 2. Remove old metric functions
        if 'def compute_fpr_tail' in source or 'def compute_sensitivity_tail' in source:
            # Remove the old metric function definitions
            old_functions_pattern = r'# ========== COMPARATOR METRICS.*?def compute_timeliness_tail\(.*?return.*?\n\n'
            source = re.sub(old_functions_pattern, '', source, flags=re.DOTALL)
            changes_made.append(f"Cell {i}: Removed old _tail metric functions")
        
        # 3. Update test metric calculations
        # Check if we need to add O_full construction
        if 'O_tail = np.column_stack(O_tail_cols)' in source and 'O_full = np.stack(O_full_list' not in source:
            # Add O_full construction after O_tail
            o_tail_pattern = r'(O_tail = np\.column_stack\(O_tail_cols\)[^\n]*\n)'
            match = re.search(o_tail_pattern, source)
            if match:
                o_tail_pos = match.end()
                o_full_code = '\n    # Build O_full (full time series) for R-comparator metrics\n'
                o_full_code += '    O_full_list = [d["y"] for d in test_sims]\n'
                o_full_code += '    O_full = np.stack(O_full_list, axis=1)  # [n_total_days, n_sims]\n\n'
                source = source[:o_tail_pos] + o_full_code + source[o_tail_pos:]
                changes_made.append(f"Cell {i}: Added O_full construction")
        
        # Replace metric calls (handle both cases: with/without O_full already added)
        replacements = [
            (r'compute_sensitivity_tail\(A_tail,\s*O_tail\)', 'compute_sensitivity_R(A_tail, O_full)'),
            (r'compute_specificity_tail\(A_tail,\s*O_tail\)', 'compute_specificity_R(A_tail, O_full, IDX_RANGE)'),
            (r'compute_fpr_tail\(A_tail,\s*O_tail\)', 'compute_fpr_R(A_tail, O_full, IDX_RANGE)'),
            (r'compute_pod_tail\(A_tail,\s*O_tail\)', 'compute_pod_R(A_tail, O_full)'),
            (r'compute_timeliness_tail\(A_tail,\s*O_tail\)', 'compute_timeliness_R(A_tail, O_full, days=7, years=7)'),
            # Also handle cases with different spacing
            (r'compute_sensitivity_tail\(A_tail,O_tail\)', 'compute_sensitivity_R(A_tail, O_full)'),
            (r'compute_specificity_tail\(A_tail,O_tail\)', 'compute_specificity_R(A_tail, O_full, IDX_RANGE)'),
            (r'compute_fpr_tail\(A_tail,O_tail\)', 'compute_fpr_R(A_tail, O_full, IDX_RANGE)'),
            (r'compute_pod_tail\(A_tail,O_tail\)', 'compute_pod_R(A_tail, O_full)'),
            (r'compute_timeliness_tail\(A_tail,O_tail\)', 'compute_timeliness_R(A_tail, O_full, days=7, years=7)'),
        ]
        
        for old, new in replacements:
            if re.search(old, source):
                source = re.sub(old, new, source)
                changes_made.append(f"Cell {i}: Replaced metric call")
        
        # Update print statements and comments
        if 'TAIL-ONLY' in source:
            source = source.replace('TAIL-ONLY', 'R-COMPARATOR')
            changes_made.append(f"Cell {i}: Updated print label to R-COMPARATOR")
        
        # Update comment about tail-only metrics
        if 'tail-only' in source.lower() and 'comparator metrics' in source.lower():
            source = re.sub(r'#.*Comparator metrics.*tail-only.*', 
                          '# ---- Comparator metrics (R-comparator style) ----', 
                          source, flags=re.IGNORECASE)
            changes_made.append(f"Cell {i}: Updated metric comment")
        
        # Update the cell source
        cell['source'] = source.splitlines(keepends=True)
    
    # Save updated notebook
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Updated {notebook_path}")
        print(f"  Changes: {len(changes_made)}")
        for change in changes_made[:5]:  # Show first 5
            print(f"    - {change}")
        if len(changes_made) > 5:
            print(f"    ... and {len(changes_made) - 5} more")
        return True
    else:
        print(f"⚠ No changes made to {notebook_path}")
        return False


def update_ocsvm_notebook(notebook_path):
    """Update OCSVM.ipynb to use R-comparator metrics."""
    print(f"\n{'='*70}")
    print(f"Updating {notebook_path}")
    print(f"{'='*70}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # 1. Add import
        if 'import' in source and 'r_comparator_metrics' not in source:
            import_pattern = r'(import[^\n]+\n)'
            imports = re.findall(import_pattern, source)
            if imports:
                last_import_idx = source.rfind(imports[-1]) + len(imports[-1])
                new_import = "from r_comparator_metrics import (\n"
                new_import += "    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,\n"
                new_import += "    compute_pod_R, compute_timeliness_R, IDX_RANGE\n"
                new_import += ")\n\n"
                source = source[:last_import_idx] + new_import + source[last_import_idx:]
                changes_made.append(f"Cell {i}: Added r_comparator_metrics import")
        
        # 2. Remove old metric functions (metric_sensitivity, metric_specificity, etc.)
        if 'def metric_sensitivity(A, O):' in source:
            # Remove old functions
            old_funcs_pattern = r'def _align_tail\(O, T\):.*?return score / J\n'
            source = re.sub(old_funcs_pattern, '', source, flags=re.DOTALL)
            changes_made.append(f"Cell {i}: Removed old metric functions")
        
        # 3. Update test metric calculations
        if 'metric_sensitivity(A, O)' in source:
            # Find where O is built from per_sim_labels
            if 'O = np.stack(per_sim_labels' in source or 'O = np.stack(per_sim_labels' in source:
                # Add O_full construction
                o_stack_pattern = r'(O = np\.stack\(per_sim_labels[^\n]+\n)'
                match = re.search(o_stack_pattern, source)
                if match:
                    o_pos = match.end()
                    o_full_code = '\n    # Build O_full (full time series) for R-comparator metrics\n'
                    o_full_code += '    O_full_list = [d["y"] for d in test_sims]\n'
                    o_full_code += '    O_full = np.stack(O_full_list, axis=1)  # [n_total_days, n_sims]\n\n'
                    source = source[:o_pos] + o_full_code + source[o_pos:]
                    changes_made.append(f"Cell {i}: Added O_full construction")
            
            # Replace metric calls
            replacements = [
                (r'metric_sensitivity\(A, O\)', 'compute_sensitivity_R(A, O_full)'),
                (r'metric_specificity\(A, O\)', 'compute_specificity_R(A, O_full, IDX_RANGE)'),
                (r'metric_pod\(A, O\)', 'compute_pod_R(A, O_full)'),
                (r'metric_timeliness\(A, O\)', 'compute_timeliness_R(A, O_full, days=7, years=7)'),
            ]
            
            for old, new in replacements:
                if old in source:
                    source = re.sub(old, new, source)
                    changes_made.append(f"Cell {i}: Replaced {old} with {new}")
            
            # Add FPR calculation if missing
            if 'fpr' not in source or 'compute_fpr' not in source:
                # Find where sens is calculated
                sens_match = re.search(r'(sens = [^\n]+\n)', source)
                if sens_match:
                    fpr_code = '    fpr = compute_fpr_R(A, O_full, IDX_RANGE)\n'
                    source = source[:sens_match.end()] + fpr_code + source[sens_match.end():]
                    changes_made.append(f"Cell {i}: Added FPR calculation")
        
        # Update the cell source
        cell['source'] = source.splitlines(keepends=True)
    
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Updated {notebook_path}")
        print(f"  Changes: {len(changes_made)}")
        for change in changes_made[:5]:
            print(f"    - {change}")
        if len(changes_made) > 5:
            print(f"    ... and {len(changes_made) - 5} more")
        return True
    else:
        print(f"⚠ No changes made to {notebook_path}")
        return False


def update_ensemble_notebook(notebook_path):
    """Update ensemble notebooks to use R-comparator metrics."""
    print(f"\n{'='*70}")
    print(f"Updating {notebook_path}")
    print(f"{'='*70}")
    
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)
    
    changes_made = []
    
    for i, cell in enumerate(nb['cells']):
        if cell['cell_type'] != 'code':
            continue
            
        source = ''.join(cell['source'])
        
        # 1. Add import
        if 'import' in source and 'r_comparator_metrics' not in source:
            import_pattern = r'(import[^\n]+\n)'
            imports = re.findall(import_pattern, source)
            if imports:
                last_import_idx = source.rfind(imports[-1]) + len(imports[-1])
                new_import = "from r_comparator_metrics import (\n"
                new_import += "    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,\n"
                new_import += "    compute_pod_R, compute_timeliness_R, IDX_RANGE\n"
                new_import += ")\n\n"
                source = source[:last_import_idx] + new_import + source[last_import_idx:]
                changes_made.append(f"Cell {i}: Added r_comparator_metrics import")
        
        # 2. Remove old metric functions
        if 'def metric_sensitivity(A, O):' in source:
            old_funcs_pattern = r'def _align_tail\(O, T\):.*?return score / J\n'
            source = re.sub(old_funcs_pattern, '', source, flags=re.DOTALL)
            changes_made.append(f"Cell {i}: Removed old metric functions")
        
        # 3. Update test metric calculations
        if 'metric_sensitivity(A, O)' in source:
            # Find where O is built
            if 'O = np.stack(' in source:
                # Add O_full construction
                o_stack_pattern = r'(O = np\.stack\([^)]+\)\n)'
                match = re.search(o_stack_pattern, source)
                if match:
                    o_pos = match.end()
                    o_full_code = '\n    # Build O_full (full time series) for R-comparator metrics\n'
                    o_full_code += '    O_full_list = [d["y"] for d in test_sims]\n'
                    o_full_code += '    O_full = np.stack(O_full_list, axis=1)  # [n_total_days, n_sims]\n\n'
                    source = source[:o_pos] + o_full_code + source[o_pos:]
                    changes_made.append(f"Cell {i}: Added O_full construction")
            
            # Replace metric calls
            replacements = [
                (r'metric_sensitivity\(A, O\)', 'compute_sensitivity_R(A, O_full)'),
                (r'metric_specificity\(A, O\)', 'compute_specificity_R(A, O_full, IDX_RANGE)'),
                (r'metric_pod\(A, O\)', 'compute_pod_R(A, O_full)'),
                (r'metric_timeliness\(A, O\)', 'compute_timeliness_R(A, O_full, days=7, years=7)'),
            ]
            
            for old, new in replacements:
                if old in source:
                    source = re.sub(old, new, source)
                    changes_made.append(f"Cell {i}: Replaced {old} with {new}")
            
            # Add FPR if missing
            if 'fpr' not in source or 'compute_fpr' not in source:
                sens_match = re.search(r'(sens = [^\n]+\n)', source)
                if sens_match:
                    fpr_code = '    fpr = compute_fpr_R(A, O_full, IDX_RANGE)\n'
                    source = source[:sens_match.end()] + fpr_code + source[sens_match.end():]
                    changes_made.append(f"Cell {i}: Added FPR calculation")
        
        # Update the cell source
        cell['source'] = source.splitlines(keepends=True)
    
    if changes_made:
        with open(notebook_path, 'w', encoding='utf-8') as f:
            json.dump(nb, f, indent=1, ensure_ascii=False)
        print(f"✓ Updated {notebook_path}")
        print(f"  Changes: {len(changes_made)}")
        for change in changes_made[:5]:
            print(f"    - {change}")
        if len(changes_made) > 5:
            print(f"    ... and {len(changes_made) - 5} more")
        return True
    else:
        print(f"⚠ No changes made to {notebook_path}")
        return False


def main():
    """Main function to update all notebooks."""
    print("="*70)
    print("Automated Notebook Update: Converting to R-Comparator Metrics")
    print("="*70)
    
    notebooks = {
        'LSTM.ipynb': update_lstm_notebook,
        'OCSVM.ipynb': update_ocsvm_notebook,
        'IF_LSTM_ensemble.ipynb': update_ensemble_notebook,
        'Ensemble_3way_voting.ipynb': update_ensemble_notebook,
    }
    
    updated = []
    skipped = []
    
    for nb_name, update_func in notebooks.items():
        nb_path = Path(nb_name)
        if not nb_path.exists():
            print(f"\n⚠ {nb_name} not found, skipping...")
            skipped.append(nb_name)
            continue
        
        try:
            if update_func(nb_path):
                updated.append(nb_name)
        except Exception as e:
            print(f"\n❌ Error updating {nb_name}: {e}")
            import traceback
            traceback.print_exc()
            skipped.append(nb_name)
    
    # Summary
    print(f"\n{'='*70}")
    print("SUMMARY")
    print(f"{'='*70}")
    print(f"✓ Successfully updated: {len(updated)}")
    for nb in updated:
        print(f"  - {nb}")
    
    if skipped:
        print(f"\n⚠ Skipped/Failed: {len(skipped)}")
        for nb in skipped:
            print(f"  - {nb}")
    
    if updated:
        print(f"\n✓ Backup recommendation: Commit changes or create backups before running the notebooks!")
    
    print(f"\n{'='*70}")


if __name__ == '__main__':
    main()
