#!/usr/bin/env python3
"""
Generate per-signal analysis tables for the write-up.
Produces:
  1. Per-signal sensitivity table (all key methods x 16 signals) at small magnitude
  2. Cross-signal variation statistics (range, CV, best/worst signals per method)
  3. Per-signal sensitivity at large magnitude for methods that have it
"""
import pandas as pd
import numpy as np

RESULTS = "results"

SIGNAL_NAMES = {
    1: "Diarrhoea, NHS 111",
    2: "Arthropod bites, ED",
    3: "Cardiac, ED",
    4: "ICU cardiac adm., ED",
    5: "Allergic rhinitis, GP",
    6: "Heat stroke, GP",
    7: "Herpes zoster, GP",
    8: "Insect bite, GP",
    9: "Pertussis, GP",
    10: "Pneumonia, GPOOH",
    11: "Rubella, GP",
    12: "Upper resp. tract, GP",
    13: "Bronchitis, GPOOH",
    14: "Hepatitis, GPOOH",
    15: "ILI, GPOOH",
    16: "UTI, GPOOH",
}

def load_per_sig(path, sig_col=None, sens_col='sensitivity', spec_col='specificity',
                 pod_col='pod', tim_col='timeliness', fpr_col='fpr', skip_header=False):
    """Load per-signal results CSV. Returns dict signal -> metrics."""
    df = pd.read_csv(path)
    # Clean column names
    df.columns = [c.strip() for c in df.columns]
    result = {}
    for _, row in df.iterrows():
        if sig_col:
            s = int(row[sig_col])
        else:
            # First column is signal index
            s = int(row.iloc[0])
        result[s] = {
            'sensitivity': float(row[sens_col]) if sens_col in df.columns else np.nan,
            'specificity': float(row[spec_col]) if spec_col in df.columns else np.nan,
            'pod': float(row[pod_col]) if pod_col in df.columns else np.nan,
            'timeliness': float(row[tim_col]) if tim_col in df.columns else np.nan,
            'fpr': float(row[fpr_col]) if fpr_col in df.columns else np.nan,
        }
    return result


def load_farrington(path):
    """Load Farrington results (different format)."""
    df = pd.read_csv(path, skiprows=1)
    df.columns = ['signal', 'sensitivity', 'specificity', 'fpr', 'pod', 'timeliness', 'extra']
    result = {}
    for _, row in df.iterrows():
        s = int(row['signal'])
        result[s] = {
            'sensitivity': float(row['sensitivity']),
            'specificity': float(row['specificity']),
            'pod': float(row['pod']),
            'timeliness': float(row['timeliness']),
            'fpr': float(row['fpr']),
        }
    return result


def load_nbhmm(path):
    """Load NB-HMM results."""
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        s = int(row['signal'])
        result[s] = {
            'sensitivity': float(row['sensitivity']),
            'specificity': float(row['specificity']),
            'pod': float(row['pod']),
            'timeliness': np.nan,  # NB-HMM doesn't always store timeliness
            'fpr': float(row['fpr']),
        }
    return result


def load_bocpd(path):
    """Load BOCPD results."""
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        s = int(row['signal'])
        result[s] = {
            'sensitivity': float(row['sensitivity']),
            'specificity': float(row['specificity']),
            'pod': float(row['pod']),
            'timeliness': float(row['timeliness']),
            'fpr': float(row['fpr']),
        }
    return result


# ---- Load all small-magnitude per-signal results ----
methods = {}

methods['IF'] = load_per_sig(f"{RESULTS}/IsolationForest_Tuned_per_sig_big_small.csv")
methods['KNN'] = load_per_sig(f"{RESULTS}/KNN_Tuned_per_sig_big_small.csv")
methods['OCSVM'] = load_per_sig(f"{RESULTS}/OCSVM_Tuned_per_sig_big_small.csv")
methods['LOF'] = load_per_sig(f"{RESULTS}/LOF_Tuned_per_sig_big_small.csv")
methods['LSTM-AE'] = load_per_sig(f"{RESULTS}/LSTM_AE_per_sig_big_small.csv")
methods['NB-HMM'] = load_nbhmm(f"{RESULTS}/NBHMM_per_sig_big_small.csv")
methods['CUSUM'] = load_per_sig(f"{RESULTS}/CUSUM_per_sig_big_small.csv")
methods['Farrington'] = load_farrington(f"{RESULTS}/Farrington_results_big_small.csv")
methods['VAE'] = load_per_sig(f"{RESULTS}/VAE_per_sig_big_small.csv")
methods['OCSVM'] = load_per_sig(f"{RESULTS}/OCSVM_Tuned_per_sig_big_small.csv")
methods['RateChange'] = load_per_sig(f"{RESULTS}/RateChangeResidual_per_sig_big_small.csv")

# Matrix Profile: different format, need to pick best regime per signal
_mp_df = pd.read_csv(f"{RESULTS}/MatrixProfile_per_sig_big_small.csv")
_mp_tuned = _mp_df[_mp_df['regime']=='C_tuned'].copy()
if len(_mp_tuned) < 16:
    _mp_tuned = _mp_df.sort_values('sensitivity', ascending=False).groupby('signal').first().reset_index()
methods['MatrixProfile'] = {}
for _, row in _mp_tuned.iterrows():
    s = int(row['signal'])
    methods['MatrixProfile'][s] = {
        'sensitivity': float(row['sensitivity']),
        'specificity': float(row['specificity']),
        'pod': float(row['pod']),
        'timeliness': float(row['timeliness']),
        'fpr': float(row['fpr']),
    }

# BOCPD-residual: different format (starts at signal 2)
try:
    _bocpd_df = pd.read_csv(f"{RESULTS}/BOCPD_residual_alone_results_big_small.csv")
    methods['BOCPD'] = {}
    for _, row in _bocpd_df.iterrows():
        s = int(row['signal'])
        methods['BOCPD'][s] = {
            'sensitivity': float(row['sensitivity']),
            'specificity': float(row['specificity']),
            'pod': float(row['pod']),
            'timeliness': float(row['timeliness']),
            'fpr': float(row['fpr']),
        }
except Exception as e:
    print(f"BOCPD load warning: {e}")

# Also load large magnitude where available
methods_large = {}
methods_large['IF'] = load_per_sig(f"{RESULTS}/IsolationForest_Tuned_per_sig_big_large.csv")
methods_large['KNN'] = load_per_sig(f"{RESULTS}/KNN_Tuned_per_sig_big_large.csv")
methods_large['OCSVM'] = load_per_sig(f"{RESULTS}/OCSVM_Tuned_per_sig_big_large.csv")
methods_large['LSTM-AE'] = load_per_sig(f"{RESULTS}/LSTM_AE_per_sig_big_large.csv")
methods_large['NB-HMM'] = load_nbhmm(f"{RESULTS}/NBHMM_per_sig_big_large.csv")
methods_large['CUSUM'] = load_per_sig(f"{RESULTS}/CUSUM_per_sig_big_large.csv")
methods_large['VAE'] = load_per_sig(f"{RESULTS}/VAE_per_sig_big_large.csv")
methods_large['RateChange'] = load_per_sig(f"{RESULTS}/RateChangeResidual_per_sig_big_large.csv")
# Matrix Profile large: same special format
_mp_df_l = pd.read_csv(f"{RESULTS}/MatrixProfile_per_sig_big_large.csv")
_mp_tuned_l = _mp_df_l[_mp_df_l['regime']=='C_tuned'].copy()
if len(_mp_tuned_l) < 16:
    _mp_tuned_l = _mp_df_l.sort_values('sensitivity', ascending=False).groupby('signal').first().reset_index()
methods_large['MatrixProfile'] = {}
for _, row in _mp_tuned_l.iterrows():
    s = int(row['signal'])
    methods_large['MatrixProfile'][s] = {
        'sensitivity': float(row['sensitivity']),
        'specificity': float(row['specificity']),
        'pod': float(row['pod']),
        'timeliness': float(row['timeliness']),
        'fpr': float(row['fpr']),
    }

# ============ TABLE 1: Per-signal sensitivity at small magnitude ============
print("="*120)
print("TABLE: Per-signal sensitivity at SMALL magnitude")
print("="*120)

method_order = ['Farrington', 'IF', 'KNN', 'OCSVM', 'LOF', 'LSTM-AE', 'NB-HMM', 'VAE', 'CUSUM']

# Header
hdr = f"{'Signal':>4} {'Name':<25}"
for m in method_order:
    hdr += f" {m:>10}"
print(hdr)
print("-"*len(hdr))

for s in range(1, 17):
    row = f"{s:>4} {SIGNAL_NAMES[s]:<25}"
    for m in method_order:
        if s in methods.get(m, {}):
            val = methods[m][s]['sensitivity']
            row += f" {val:>10.3f}"
        else:
            row += f" {'---':>10}"
    print(row)

# Mean row
row = f"{'':>4} {'MEAN':<25}"
for m in method_order:
    vals = [methods[m][s]['sensitivity'] for s in range(1, 17) if s in methods.get(m, {})]
    if vals:
        row += f" {np.mean(vals):>10.3f}"
    else:
        row += f" {'---':>10}"
print("-"*len(hdr))
print(row)

# ============ TABLE 2: Cross-signal variation ============
print("\n")
print("="*120)
print("TABLE: Cross-signal variation in sensitivity (small magnitude)")
print("="*120)

hdr2 = f"{'Method':<12} {'Mean':>7} {'Std':>7} {'CV':>7} {'Min':>7} {'Max':>7} {'Range':>7} {'Best sig':>10} {'Worst sig':>10}"
print(hdr2)
print("-"*len(hdr2))

for m in method_order:
    sens_dict = {s: methods[m][s]['sensitivity'] for s in range(1, 17) if s in methods.get(m, {})}
    if not sens_dict:
        continue
    vals = list(sens_dict.values())
    mean_v = np.mean(vals)
    std_v = np.std(vals)
    cv = std_v / mean_v if mean_v > 0 else 0
    min_v = np.min(vals)
    max_v = np.max(vals)
    best_s = max(sens_dict, key=sens_dict.get)
    worst_s = min(sens_dict, key=sens_dict.get)
    print(f"{m:<12} {mean_v:>7.3f} {std_v:>7.3f} {cv:>7.2f} {min_v:>7.3f} {max_v:>7.3f} {max_v-min_v:>7.3f} {best_s:>4} ({SIGNAL_NAMES[best_s][:15]}) {worst_s:>4} ({SIGNAL_NAMES[worst_s][:15]})")


# ============ TABLE 3: Signals that are consistently hard/easy ============
print("\n")
print("="*120)
print("TABLE: Signal difficulty ranking (mean sensitivity across methods, small magnitude)")
print("="*120)

sig_means = {}
for s in range(1, 17):
    vals = []
    for m in method_order:
        if s in methods.get(m, {}):
            vals.append(methods[m][s]['sensitivity'])
    if vals:
        sig_means[s] = np.mean(vals)

sorted_sigs = sorted(sig_means.items(), key=lambda x: x[1], reverse=True)
print(f"{'Rank':>4} {'Signal':>6} {'Name':<25} {'Mean Sens':>10} {'Category':<15}")
for rank, (s, mean_s) in enumerate(sorted_sigs, 1):
    cat = "easy" if mean_s > 0.7 else ("medium" if mean_s > 0.4 else "hard")
    print(f"{rank:>4} {s:>6} {SIGNAL_NAMES[s]:<25} {mean_s:>10.3f} {cat:<15}")


# ============ TABLE 4: Per-signal sensitivity at LARGE magnitude (key methods) ============
print("\n")
print("="*120)
print("TABLE: Per-signal sensitivity at LARGE magnitude")
print("="*120)

lm_order = ['IF', 'KNN', 'OCSVM', 'LSTM-AE', 'NB-HMM', 'VAE', 'CUSUM']

hdr = f"{'Signal':>4} {'Name':<25}"
for m in lm_order:
    hdr += f" {m:>10}"
print(hdr)
print("-"*len(hdr))

for s in range(1, 17):
    row = f"{s:>4} {SIGNAL_NAMES[s]:<25}"
    for m in lm_order:
        if s in methods_large.get(m, {}):
            val = methods_large[m][s]['sensitivity']
            row += f" {val:>10.3f}"
        else:
            row += f" {'---':>10}"
    print(row)

row = f"{'':>4} {'MEAN':<25}"
for m in lm_order:
    vals = [methods_large[m][s]['sensitivity'] for s in range(1, 17) if s in methods_large.get(m, {})]
    if vals:
        row += f" {np.mean(vals):>10.3f}"
    else:
        row += f" {'---':>10}"
print("-"*len(hdr))
print(row)


# ============ TABLE 5: Per-signal sensitivity improvement (small -> large) ============
print("\n")
print("="*120)
print("TABLE: Per-signal sensitivity change (large - small) for IF, KNN, LSTM-AE")
print("="*120)

for m in ['IF', 'KNN', 'LSTM-AE', 'NB-HMM']:
    print(f"\n--- {m} ---")
    for s in range(1, 17):
        if s in methods.get(m, {}) and s in methods_large.get(m, {}):
            s_small = methods[m][s]['sensitivity']
            s_large = methods_large[m][s]['sensitivity']
            delta = s_large - s_small
            print(f"  Sig {s:>2} ({SIGNAL_NAMES[s]:<25}): {s_small:.3f} -> {s_large:.3f}  (delta = {delta:+.3f})")


# ============ CSV output for easy plotting ============
rows = []
for mag_label, mdict in [('small', methods), ('large', methods_large)]:
    for m_name, m_data in mdict.items():
        for s in range(1, 17):
            if s in m_data:
                rows.append({
                    'method': m_name,
                    'signal': s,
                    'signal_name': SIGNAL_NAMES[s],
                    'magnitude': mag_label,
                    'sensitivity': m_data[s]['sensitivity'],
                    'specificity': m_data[s].get('specificity', np.nan),
                    'pod': m_data[s].get('pod', np.nan),
                    'timeliness': m_data[s].get('timeliness', np.nan),
                    'fpr': m_data[s].get('fpr', np.nan),
                })

df_out = pd.DataFrame(rows)
df_out.to_csv('results/per_signal_all_methods.csv', index=False)
print(f"\nSaved per-signal results to results/per_signal_all_methods.csv ({len(df_out)} rows)")
