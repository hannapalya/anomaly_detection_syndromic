#!/usr/bin/env python3
"""After unified per-magnitude tables exist, produce a side-by-side robustness
table: methods × magnitudes, showing how each method's sens/timeliness/POD
evolves with outbreak magnitude.

Reads:
  results/ALL_METHODS_unified_big_small.csv
  results/ALL_METHODS_unified_big_medium.csv
  results/ALL_METHODS_unified_big_large.csv
Writes:
  results/ROBUSTNESS_table.csv      (wide format)
  results/ROBUSTNESS_table_long.csv (long format for plotting)
Prints:
  Markdown-formatted table to stdout.
"""

import os
import pandas as pd


MAGNITUDES = ["small", "medium", "large"]
METRICS    = ["sensitivity", "specificity", "pod", "timeliness", "fpr"]


def load_one(mag):
    fp = f"results/ALL_METHODS_unified_big_{mag}.csv"
    if not os.path.exists(fp):
        print(f"[!] Missing {fp}; skipping {mag}")
        return None
    df = pd.read_csv(fp)
    df["magnitude"] = mag
    return df


def main():
    dfs = [load_one(m) for m in MAGNITUDES]
    dfs = [d for d in dfs if d is not None]
    if not dfs:
        print("No magnitude files found.")
        return
    long_df = pd.concat(dfs, ignore_index=True)
    long_df.to_csv("results/ROBUSTNESS_table_long.csv", index=False)

    # Wide: one row per method, columns = (magnitude, metric)
    wide = long_df.pivot_table(
        index="method", columns="magnitude",
        values=METRICS, aggfunc="first")
    # Reorder so each magnitude block is grouped together
    wide = wide.reorder_levels([1, 0], axis=1)
    desired = [(m, k) for m in MAGNITUDES for k in METRICS if (m, k) in wide.columns]
    wide = wide.reindex(columns=desired)
    # Sort methods by mean sensitivity across magnitudes available
    sens_cols = [(m, "sensitivity") for m in MAGNITUDES if (m, "sensitivity") in wide.columns]
    wide["__sort"] = wide[sens_cols].mean(axis=1)
    wide = wide.sort_values("__sort", ascending=False).drop(columns="__sort")
    wide.to_csv("results/ROBUSTNESS_table.csv")

    pd.options.display.float_format = "{:.3f}".format

    # Compact print: sens-only by magnitude
    sens_only = long_df.pivot_table(
        index="method", columns="magnitude", values="sensitivity", aggfunc="first")
    sens_only = sens_only.reindex(columns=[m for m in MAGNITUDES if m in sens_only.columns])
    sens_only["mean"] = sens_only.mean(axis=1)
    sens_only = sens_only.sort_values("mean", ascending=False)
    print("\n=== ROBUSTNESS: sensitivity by magnitude ===")
    print(sens_only.round(3).to_string())

    tim_only = long_df.pivot_table(
        index="method", columns="magnitude", values="timeliness", aggfunc="first")
    tim_only = tim_only.reindex(columns=[m for m in MAGNITUDES if m in tim_only.columns])
    print("\n=== ROBUSTNESS: timeliness by magnitude (lower is better) ===")
    print(tim_only.round(3).reindex(sens_only.index).to_string())

    print("\nSaved: results/ROBUSTNESS_table.csv (wide) + results/ROBUSTNESS_table_long.csv")


if __name__ == "__main__":
    main()
