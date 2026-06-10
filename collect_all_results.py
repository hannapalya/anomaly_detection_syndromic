#!/usr/bin/env python3
"""Read every available per-signal results CSV and produce one unified table.

Usage:
    python collect_all_results.py [--magnitude {small,medium,large}]

The script substitutes "big_medium" in each source path with "big_<MAG>"
when --magnitude is provided. Filed missing for a given magnitude are noted
as 'missing'. Output: results/ALL_METHODS_unified_big_<MAG>.csv.
"""

import argparse
import os
import pandas as pd
import numpy as np


# (label, csv_path) — order matters for the printed table.
# Paths use the "big_medium" suffix as a template; the --magnitude argument
# rewrites that to big_<MAG>.
SOURCES = [
    # ── Core unsupervised methods (robustness study) ──────────────────
    # Sequential/statistical
    ("NB-HMM",                      "results/NBHMM_per_sig_big_medium.csv"),
    ("CUSUM (NB seasonal)",         "results/CUSUM_per_sig_big_medium.csv"),
    ("Farrington (custom α=0.01)",  "results/Farrington_alarms_under_pipeline_big_medium.csv"),
    ("Noufaily-quantile",           "results/Unsup_OR_summary_big_medium.csv:noufaily"),
    ("RateChange-residual",         "results/RateChangeResidual_per_sig_big_medium.csv"),
    ("BOCPD-residual",              "results/Unsup_OR_summary_big_medium.csv:bocpd_resid"),
    ("Matrix Profile",              "results/MatrixProfile_per_sig_big_medium.csv"),
    # Tabular/feature-based ML
    ("IsolationForest (tuned)",     "results/IsolationForest_Tuned_per_sig_big_medium.csv"),
    ("KNN (per-sig tuned)",         "results/KNN_Tuned_per_sig_big_medium.csv"),
    ("LOF (per-sig tuned)",         "results/LOF_Tuned_per_sig_big_medium.csv"),
    ("OCSVM (per-sig tuned)",       "results/OCSVM_Tuned_per_sig_big_medium.csv"),
    # Deep unsupervised
    ("LSTM-AE (production)",        "results/LSTM_AE_per_sig_big_medium.csv"),
    ("VAE (NegBin)",                "results/VAE_per_sig_big_medium.csv"),
    # ── OR-vote ensembles ─────────────────────────────────────────────
    ("OR-vote (all6)",              "results/Unsup_OR_summary_big_medium.csv:OR_all6"),
    ("OR-vote (if+lstm+nbhmm+noufaily)", "results/Unsup_OR_summary_big_medium.csv:OR_if_lstm_nbhmm_noufaily"),
    ("OR-vote (if+lstm+nbhmm)",    "results/Unsup_OR_summary_big_medium.csv:OR_if_lstm_nbhmm"),
    # ── Supervised (reference only, single magnitude) ─────────────────
    ("Stacker v5 (XGB)",            "results/StackedMetaV5_xgb_results_big_medium.csv"),
    ("Stacker v8 (+BOCPD-residual)","results/StackedMetaV8_BOCPDresid_xgb_results_big_medium.csv"),
    # ── Legacy paths (IF has non-standard naming) ────────────────────
    # NOTE: "per_sig_med" is the legacy small-magnitude IF. Kept for backward compat.
    # TODO: re-run IF with proper magnitude-aware naming per new convention.
]

# Legacy path mapping: some older results use non-standard naming.
# If the standard _big_{MAG} path doesn't exist, try these fallbacks.
LEGACY_FALLBACKS = {
    # No longer needed — IF uses standard naming now
}


METRIC_COLS = {"sensitivity", "Sensitivity_All", "specificity", "Specificity_All",
               "fpr", "FPR_All", "pod", "POD_All", "timeliness", "Timeliness_All"}


def normalize(df):
    """Lowercase metric column names; coerce numeric."""
    rename = {}
    for c in df.columns:
        cl = c.strip().lower()
        if cl in ("sensitivity_all",): rename[c] = "sensitivity"
        elif cl in ("specificity_all",): rename[c] = "specificity"
        elif cl in ("fpr_all",): rename[c] = "fpr"
        elif cl in ("pod_all",): rename[c] = "pod"
        elif cl in ("timeliness_all",): rename[c] = "timeliness"
        elif cl == "sensitivity": rename[c] = "sensitivity"
        elif cl == "specificity": rename[c] = "specificity"
        elif cl == "fpr": rename[c] = "fpr"
        elif cl == "pod": rename[c] = "pod"
        elif cl == "timeliness": rename[c] = "timeliness"
    df = df.rename(columns=rename)
    for k in ("sensitivity", "specificity", "fpr", "pod", "timeliness"):
        if k in df.columns:
            df[k] = pd.to_numeric(df[k], errors='coerce')
    return df


def get_means(path):
    # path may be "filename" or "filename:method_filter"
    filt = None
    if ":" in path:
        path, filt = path.split(":", 1)
    if not os.path.exists(path):
        # Try legacy fallback
        path = LEGACY_FALLBACKS.get(path, path)
        if not os.path.exists(path):
            return None
    df = pd.read_csv(path, skip_blank_lines=True)
    df = normalize(df)
    if filt is not None and "method" in df.columns:
        df = df[df["method"] == filt]
        if df.empty:
            return None
    keep = [c for c in ("sensitivity", "specificity", "fpr", "pod", "timeliness") if c in df.columns]
    if not keep:
        return None
    sub = df[keep].dropna(how="all")
    if len(sub) == 0:
        return None
    return sub.mean(skipna=True).to_dict()


def _rewrite_for_mag(path, mag):
    """Replace 'big_medium' (and 'big_small', 'big_large') with 'big_<mag>'.
    Also handles the legacy IF naming '_per_sig_med' → '_per_sig_<mag>'."""
    if path is None:
        return None
    # Replace any magnitude tag with the requested one
    for old_tag in ("big_medium", "big_small", "big_large"):
        path = path.replace(old_tag, f"big_{mag}")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--magnitude", default="medium",
                    choices=["small", "medium", "large"],
                    help="Outbreak magnitude; legacy 'big_medium' files map to medium.")
    args = ap.parse_args()
    MAG = args.magnitude

    rows = []
    for label, path in SOURCES:
        path = _rewrite_for_mag(path, MAG)
        if path is None:
            continue
        m = get_means(path)
        if m is None:
            rows.append(dict(method=label, **{k: float('nan') for k in ("sensitivity", "specificity", "fpr", "pod", "timeliness")}, status="missing"))
        else:
            rows.append(dict(method=label, status="ok", **m))

    df = pd.DataFrame(rows)
    # Fill FPR from (1 - specificity) when FPR was not explicitly saved.
    # Both denominators are non-outbreak days in IDX_RANGE under R-comparator metrics.
    mask = df["fpr"].isna() & df["specificity"].notna()
    df.loc[mask, "fpr"] = (1 - df.loc[mask, "specificity"]).round(4)
    df_show = df[["method", "sensitivity", "specificity", "pod", "timeliness", "fpr", "status"]]
    df_show = df_show.sort_values("sensitivity", ascending=False, na_position="last")
    pd.options.display.float_format = "{:.3f}".format
    print(f"\n=== UNIFIED COMPARISON @ magnitude={MAG} (sorted by sensitivity, desc) ===")
    print(df_show.to_string(index=False))
    outp = f"results/ALL_METHODS_unified_big_{MAG}.csv"
    df_show.to_csv(outp, index=False)
    print(f"\nSaved: {outp}")
