#!/usr/bin/env python3
"""
CUSUM (Cumulative Sum) detector for syndromic surveillance.

- Fixed seasonal baseline mu_t fitted by Negative-Binomial GLM on outbreak-free
  training-period days (annual + semi-annual harmonics + day-of-week).
- Pearson-style standardised residual: r_t = (x_t - mu_t) / sqrt(dispersion * mu_t + eps).
- One-sided recursive CUSUM: S_t = max(0, S_{t-1} + r_t - k).
- Score = -S_t (higher = more normal) so percentile-based contamination tuning
  in anom_common applies unchanged.
- Same 60/20/20 split, validation on last VALID_DAYS, test on [ABS_START:ABS_END).
"""

import os
import numpy as np
import pandas as pd
from statsmodels.genmod.generalized_linear_model import GLM
from statsmodels.genmod import families
from anom_common import *

# Linear secular trend term (makes the baseline a faithful Serfling cyclical
# regression). OFF by default so existing results are unchanged; enable with
# SYND_BASELINE_TREND=1 or by setting run_cusum._SEASONAL_TREND = True.
_SEASONAL_TREND = os.environ.get("SYND_BASELINE_TREND", "0") == "1"


def _seasonal_design(n, start=0):
    t = np.arange(start, start + n, dtype=float)
    period = 364.0
    cols = [
        np.ones(n),
        np.sin(2*np.pi*t/period), np.cos(2*np.pi*t/period),
        np.sin(4*np.pi*t/period), np.cos(4*np.pi*t/period),
    ]
    if _SEASONAL_TREND:
        cols.append(t / period)  # linear trend, scaled to ~years (Serfling secular term)
    dow = (t.astype(int) % 7)
    for d in range(1, 7):  # 6 dummies, Sunday=0 as reference
        cols.append((dow == d).astype(float))
    return np.column_stack(cols)


class CUSUMDetector:
    def __init__(self, k=0.5, eps=1e-6, train_days=TRAIN_DAYS):
        self.k = k
        self.eps = eps
        self.train_days = train_days
        self.params_ = None

    def fit_baseline(self, x_full, y_full):
        x = np.asarray(x_full, dtype=float)
        y = np.asarray(y_full, dtype=int)
        n_train = min(self.train_days, len(x))
        X_train = _seasonal_design(n_train, start=0)
        mask = (y[:n_train] == 0)
        if mask.sum() < 50:
            mask = np.ones(n_train, dtype=bool)  # fallback: use all train days
        try:
            model = GLM(x[:n_train][mask], X_train[mask],
                        family=families.NegativeBinomial(alpha=1.0)).fit(maxiter=50, disp=0)
            self.params_ = model.params
            # Pearson dispersion gives variance scale used in residual standardisation
            mu_train = np.exp(np.clip(X_train[mask] @ self.params_, -20, 20))
            resid = (x[:n_train][mask] - mu_train) ** 2 / np.maximum(mu_train, 1e-6)
            self.dispersion_ = max(float(resid.mean()), 1.0)
        except Exception:
            # Fallback: log-mean intercept only
            self.params_ = np.zeros(X_train.shape[1])
            self.params_[0] = np.log(max(x[:n_train][mask].mean(), 1.0))
            self.dispersion_ = 1.0

    def baseline(self, n_total):
        X_full = _seasonal_design(n_total, start=0)
        eta = X_full @ self.params_
        eta = np.clip(eta, -20, 20)
        return np.exp(eta)

    def decision_function_series(self, series, y_for_fit=None):
        x = np.asarray(series, dtype=float)
        if len(x) == 0:
            return np.array([], dtype=float)
        if self.params_ is None:
            self.fit_baseline(x, y_for_fit if y_for_fit is not None else np.zeros_like(x))
        mu = self.baseline(len(x))
        var = self.dispersion_ * mu + self.eps
        r = (x - mu) / np.sqrt(var)
        S = np.zeros_like(r)
        prev = 0.0
        for t in range(len(r)):
            prev = max(0.0, prev + r[t] - self.k)
            S[t] = prev
        return -S


if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)

    summary_all = {}
    rows_val, rows_test = [], []

    K_GRID = [0.1, 0.15, 0.25, 0.5, 1.0]

    for S in SIGNALS:
        print(f"\n--- Signal {S} (CUSUM) ---")
        Xsig, Ysig = load_data(S)

        sims = []
        for sim_idx, col in enumerate(Xsig.columns):
            x = Xsig[col].to_numpy(np.float32, copy=False)
            y = Ysig[col].to_numpy(np.int32,  copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{sim_idx}", sim_idx=sim_idx))
        if not sims:
            print("No complete sims; skip.")
            continue

        train_sims, val_sims, test_sims_final = split_60_20_20(sims, rng)
        print(f"  Splits: {len(train_sims)} train, {len(val_sims)} val, {len(test_sims_final)} test")

        # ---- pick k by validation: tune contamination per k, keep best (k, c) ----
        best = None  # (combined_score, k, c, scores_concat, val_lengths, Yval_concat)
        for k in K_GRID:
            val_scores_concat = []
            val_lengths = []
            Yval_concat = []
            for d in val_sims:
                cu = CUSUMDetector(k=k)
                cu.fit_baseline(d["x"], d["y"])
                scores_full = cu.decision_function_series(d["x"])
                if len(scores_full) < VALID_DAYS:
                    continue
                tail_scores = scores_full[-VALID_DAYS:]
                val_scores_concat.extend(tail_scores)
                val_lengths.append(VALID_DAYS)
                Yval_concat.extend(d["y"][-VALID_DAYS:])

            val_scores_concat = np.asarray(val_scores_concat, dtype=float)
            Yval_concat = np.asarray(Yval_concat, dtype=int)

            if not len(val_scores_concat):
                continue

            c_best = tune_contamination_threshold(
                val_sims, val_lengths, val_scores_concat,
                spec_target=SPEC_TARGET,
                w_sens=W_SENS, w_spec=W_SPEC
            )
            thr_val = np.percentile(val_scores_concat, c_best * 100)
            yhat_v = (val_scores_concat <= thr_val).astype(int)

            # combined val score for k selection (R-comparator)
            O_full_val = np.stack([d["y"] for d in val_sims], axis=1)
            A_list, ofs = [], 0
            for L in val_lengths:
                A_list.append(yhat_v[ofs:ofs+L])
                ofs += L
            A_val = np.column_stack(A_list)
            sp_v = compute_specificity_R(A_val, O_full_val, IDX_RANGE)
            s_v  = compute_sensitivity_R(A_val, O_full_val)
            combined = (W_SENS * (s_v if not np.isnan(s_v) else 0.0)
                        + W_SPEC * (sp_v if not np.isnan(sp_v) else 0.0))

            if best is None or combined > best[0]:
                best = (combined, k, c_best, val_scores_concat, val_lengths, Yval_concat, thr_val)

        if best is None:
            print("  No usable validation sims; skip.")
            continue

        _, k_best, c_best, val_scores_concat, val_lengths, Yval_concat, thr_val = best
        print(f"  Selected k={k_best}, contamination={c_best}")

        # ---- per-sim val rows ----
        yhat_v = (val_scores_concat <= thr_val).astype(int)
        splits_yhat = split_by_lengths(yhat_v, val_lengths)
        splits_y    = split_by_lengths(Yval_concat, val_lengths)
        for d, yh, y in zip(val_sims, splits_yhat, splits_y):
            if len(yh) == 0:
                continue
            s_i, sp_i = sens_spec(y, yh)
            pod_i = pod_anyhit(yh, y)
            tim_i = timeliness_single(yh, y)
            rows_val.append(dict(
                split="val", model="cusum", signal=S, sim=d["sim"],
                k=k_best,
                contamination=c_best, thr=thr_val,
                sens=s_i, spec=sp_i, pod=pod_i, timeliness=tim_i,
                n_points=len(yh)
            ))

        # ---- TEST scoring on ABS window ----
        per_sim_scores = []
        per_sim_labels = []
        for d in test_sims_final:
            cu = CUSUMDetector(k=k_best)
            cu.fit_baseline(d["x"], d["y"])
            scores_full = cu.decision_function_series(d["x"])
            if len(scores_full) < ABS_END:
                continue
            win_scores = scores_full[ABS_START:ABS_END]
            y_win = d["y"][ABS_START:ABS_END].astype(int)
            per_sim_scores.append(win_scores)
            per_sim_labels.append(y_win)

        if not per_sim_scores:
            print("  No test sims with full ABS window.")
            continue

        scores_concat = np.concatenate(per_sim_scores)
        thr_test = thr_val
        yhat_concat = (scores_concat <= thr_test).astype(int)

        ofs = 0
        A_list = []
        for y_win in per_sim_labels:
            A_list.append(yhat_concat[ofs:ofs+WIN_LEN])
            ofs += WIN_LEN
        A = np.column_stack(A_list)
        O = np.column_stack(per_sim_labels)

        import os as _os
        cache_dir = f"score_cache/cusum_{MAG_TAG}"
        _os.makedirs(cache_dir, exist_ok=True)
        alarm_filename = f"{cache_dir}/cusum_alarms_signal_{S}.csv"
        pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(alarm_filename, index=False)
        outbreak_filename = f"{cache_dir}/cusum_outbreaks_signal_{S}.csv"
        pd.DataFrame(O, columns=[f"sim_{i}" for i in range(O.shape[1])]).to_csv(outbreak_filename, index=False)
        # raw scores too — needed later for the stacked meta-learner
        score_filename = f"{cache_dir}/cusum_scores_signal_{S}.csv"
        score_mat = np.column_stack(per_sim_scores)
        pd.DataFrame(score_mat, columns=[f"sim_{i}" for i in range(score_mat.shape[1])]).to_csv(score_filename, index=False)
        # Backward-compat: also write flat path for small magnitude (used by ensemble scripts)
        if MAG_TAG == "small":
            pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(f"cusum_alarms_signal_{S}.csv", index=False)
            pd.DataFrame(score_mat, columns=[f"sim_{i}" for i in range(score_mat.shape[1])]).to_csv(f"cusum_scores_signal_{S}.csv", index=False)
        print(f"Saved alarms: {alarm_filename}")
        print(f"Saved outbreaks: {outbreak_filename}")
        print(f"Saved scores: {score_filename}")

        O_full_list = [d["y"] for d in test_sims_final]
        O_full = np.stack(O_full_list, axis=1)

        fpr_R  = compute_fpr_R(A, O_full, IDX_RANGE)
        spec_R = compute_specificity_R(A, O_full, IDX_RANGE)
        sens_R = compute_sensitivity_R(A, O_full)
        pod_R  = compute_pod_R(A, O_full)
        tim_R  = compute_timeliness_R(A, O_full)

        sens0 = compute_sensitivity(A, O)
        spec0 = compute_specificity(A, O)
        pod0  = compute_pod(A, O)
        tim0  = compute_timeliness(A, O)

        print(f"R-COMPARATOR -> Sens={sens_R:.3f}, Spec={spec_R:.3f}, "
              f"FPR={fpr_R:.3f}, POD={pod_R:.3f}, Tim={tim_R:.3f}")
        print(f"ORIGINAL     -> Sens={sens0:.3f}, Spec={spec0:.3f}, "
              f"POD={pod0:.3f}, Tim={tim0:.3f}")

        for d, yh, ytrue in zip(test_sims_final, A_list, per_sim_labels):
            s_i, sp_i = sens_spec(ytrue, yh)
            pod_i = pod_anyhit(yh, ytrue)
            tim_i = timeliness_single(yh, ytrue)
            rows_test.append(dict(
                split="test", model="cusum", signal=S, sim=d["sim"],
                k=k_best,
                contamination=c_best, thr=thr_test,
                sens=s_i, spec=sp_i, pod=pod_i, timeliness=tim_i,
                n_points=len(yh)
            ))

        summary_all[S] = dict(
            sensitivity=sens_R, specificity=spec_R, fpr=fpr_R, pod=pod_R, timeliness=tim_R,
            contamination=c_best,
            k=k_best
        )

    if summary_all:
        df = pd.DataFrame.from_dict(summary_all, orient="index")
        df.index.name = "signal"
        print("\n=== SUMMARY (CUSUM, all signals) ===")
        print(df)
        print("\nMeans:\n", df.mean(numeric_only=True))
        out_path = f"results/CUSUM_per_sig_big_{MAG_TAG}.csv"
        df.to_csv(out_path)
        print(f"Wrote {out_path}")

    if rows_val:
        dfv = pd.DataFrame(rows_val)
        dfv.sort_values(["signal", "sim", "split"], inplace=True)
        dfv.to_csv("CUSUM_per_sim_val.csv", index=False)
        print("Saved: CUSUM_per_sim_val.csv")

    if rows_test:
        dft = pd.DataFrame(rows_test)
        dft.sort_values(["signal", "sim", "split"], inplace=True)
        dft.to_csv("CUSUM_per_sim_test.csv", index=False)
        print("Saved: CUSUM_per_sim_test.csv")
