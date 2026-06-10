#!/usr/bin/env python3
"""
Smoke test — verify a fresh checkout can run the core pipeline.

Runs in a few seconds and needs NO GPU, NO R, and NO downloaded data: if the
real datasets (big_signal_datasets_*) are absent it synthesizes a tiny series.
It exercises the shared stack that every tabular detector depends on:

    anom_common.create_features  ->  sklearn IsolationForest  ->  R-comparator metrics

A PASS means the Python environment, anom_common, scikit-learn, and the metric
functions all work. It does NOT cover the deep-learning (PyTorch/GPU) or
Farrington (R) detectors — those are reported as optional availability checks.

Usage:  python smoke_test.py      (or:  make smoke)
"""
import sys
import numpy as np


def _versions():
    import platform
    print(f"Python      {platform.python_version()}")
    for mod in ("numpy", "pandas", "sklearn", "scipy", "statsmodels"):
        try:
            m = __import__(mod)
            print(f"{mod:<11} {getattr(m, '__version__', '?')}")
        except Exception as e:  # pragma: no cover
            print(f"{mod:<11} MISSING ({e})")
            return False
    return True


def _load_or_synthesize(anom):
    """Return (totals[days,nsims], outbreaks01[days,nsims], source_str)."""
    days = anom.ABS_END  # 2548 — long enough for the absolute IDX_RANGE window
    nsims = 6
    rng = np.random.RandomState(anom.RNG_STATE)
    try:
        X, Y = anom.load_data(1)
        tot = X.to_numpy(dtype=float)[:days, :nsims]
        out = (Y.to_numpy() > 0).astype(int)[:days, :nsims]
        if tot.shape[1] >= nsims and tot.shape[0] >= days:
            return tot[:days, :nsims], out[:days, :nsims], f"real data ({anom.DATA_DIR}, signal 1)"
    except FileNotFoundError:
        pass
    # Synthesize: Poisson baseline + a multi-day outbreak spike inside the window.
    tot = rng.poisson(25, size=(days, nsims)).astype(float)
    out = np.zeros((days, nsims), dtype=int)
    win = np.arange(anom.ABS_START, anom.ABS_END)
    for j in range(nsims):
        onset = int(rng.choice(win[: len(win) - 12]))
        length = int(rng.randint(6, 12))
        tot[onset:onset + length, j] += rng.poisson(40, size=length)
        out[onset:onset + length, j] = 1
    return tot, out, "synthetic (no dataset found — environment check only)"


def main():
    print("=" * 60)
    print("SMOKE TEST — syndromic anomaly-detection pipeline")
    print("=" * 60)
    if not _versions():
        print("\nFAIL: a required package is missing (see requirements.txt).")
        return 1

    try:
        import anom_common as anom
        from sklearn.ensemble import IsolationForest
    except Exception as e:
        print(f"\nFAIL: could not import core modules: {e}")
        return 1

    tot, out, source = _load_or_synthesize(anom)
    print(f"\nData source : {source}")
    print(f"Window      : [{anom.ABS_START}, {anom.ABS_END})  (WIN_LEN={anom.WIN_LEN})")

    W = 14
    nsims = tot.shape[1]
    n_train = max(2, nsims // 2)

    # --- Features (the 20-d tabular vector used by IF/KNN/LOF/OCSVM) ---
    feats = [anom.create_features(tot[:, j], W) for j in range(nsims)]
    assert feats[0].shape[1] == 20, f"expected 20 features, got {feats[0].shape[1]}"
    print(f"Features    : {feats[0].shape[1]}-d  (rows/sim={feats[0].shape[0]})")

    # --- Fit IF on 'train' sims, score 'test' sims, alarm on the window tail ---
    clf = IsolationForest(n_estimators=100, random_state=anom.RNG_STATE)
    clf.fit(np.vstack(feats[:n_train]))

    A_cols, O_cols = [], []
    for j in range(n_train, nsims):
        s = clf.score_samples(feats[j])          # lower = more anomalous
        thr = np.percentile(s, 5)                # ~5% contamination
        alarms = (s <= thr).astype(int)
        A_cols.append(alarms[-anom.WIN_LEN:])    # last 343 rows == days [2205,2548)
        O_cols.append(out[:, j])
    A = np.column_stack(A_cols)                  # [WIN_LEN, n_test]
    O_full = np.column_stack(O_cols)             # [days,    n_test]

    # --- R-comparator metrics ---
    m = dict(
        sensitivity=anom.compute_sensitivity_R(A, O_full),
        specificity=anom.compute_specificity_R(A, O_full, anom.IDX_RANGE),
        fpr=anom.compute_fpr_R(A, O_full, anom.IDX_RANGE),
        pod=anom.compute_pod_R(A, O_full),
        timeliness=anom.compute_timeliness_R(A, O_full),
    )
    print("\nMetrics (mini IF on a few sims — values are not meaningful, only that they compute):")
    for k, v in m.items():
        print(f"  {k:<12} {v:.3f}")

    for k, v in m.items():
        assert np.isfinite(v), f"{k} is not finite ({v})"
        assert -1e-9 <= v <= 1 + 1e-9, f"{k} out of [0,1]: {v}"

    # --- Optional heavy deps (not required for a PASS) ---
    print("\nOptional detectors:")
    try:
        import torch
        print(f"  torch       {torch.__version__}  (LSTM-AE / VAE available)")
    except Exception:
        print("  torch       not installed — LSTM-AE / VAE will be skipped")
    import shutil
    print(f"  Rscript     {'found' if shutil.which('Rscript') else 'not found — Farrington will be skipped'}")

    print("\n" + "=" * 60)
    print("SMOKE TEST PASSED")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
