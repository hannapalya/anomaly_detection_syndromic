#!/usr/bin/env python3
"""
BOCPD on Pearson residuals from a Negative-Binomial seasonal baseline.

Architecture:
  1. Fit NB GLM with Fourier (annual + semi-annual) seasonality + day-of-week
     dummies on outbreak-free TRAINING-period days. Same machinery as
     run_cusum.py (we reuse CUSUMDetector.fit_baseline / .baseline output).
  2. Compute Pearson residuals: r_t = (x_t - mu_t) / sqrt(dispersion * mu_t).
     These are ~standard-normal under the null hypothesis (no outbreak).
  3. Run Adams & MacKay 2007 BOCPD on the residual stream with:
        - Conjugate Normal-InverseGamma prior (mu_0=0, kappa_0=1, alpha_0=2, beta_0=1)
        - Constant hazard rate h (tuned per signal: 1/30, 1/50, 1/100)
        - Run-length truncation r_max=200
        - Log-space updates for numerical stability
        - Per-timestep score = predictive surprise = -log P(y_t | y_{1:t-1})
          (Bayesian residual; high = anomalous)

Outputs per signal S:
  bocpd_resid_val_scores_signal_{S}.csv  shape (343, n_val_sims)  higher=anomalous
  bocpd_resid_test_scores_signal_{S}.csv shape (343, n_test_sims) higher=anomalous

Convention: scores returned higher = MORE ANOMALOUS. Stacker wants
higher = more normal, so caching script negates before saving.
"""

import argparse
import os
import time
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import t as student_t

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    sens_spec, pod_anyhit, timeliness_single,
    tune_contamination_threshold, split_by_lengths,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
    MAG_TAG,
)
from run_cusum import CUSUMDetector  # reuse seasonal-baseline fitting


# Per-signal validation-based selection over the hazard rate (mean run length 30/50/100
# days), matching the per-signal model search given to the tabular detectors.
# Override with env var, e.g. SYND_BOCPD_HAZARDS="0.0333,0.02,0.01".
import os as _os_bocpd
_hz = _os_bocpd.environ.get("SYND_BOCPD_HAZARDS", "").strip()
HAZARD_GRID = ([float(x) for x in _hz.split(",") if x.strip()] if _hz
               else [1/30.0, 1/50.0, 1/100.0])
R_MAX = 200


# ====== BOCPD on residuals with Normal-InverseGamma conjugate ======

class BOCPDResidual:
    def __init__(self, hazard=1/30.0, r_max=R_MAX,
                 mu0=0.0, kappa0=1.0, alpha0=2.0, beta0=1.0):
        self.hazard = hazard
        self.r_max = r_max
        self.mu0, self.kappa0, self.alpha0, self.beta0 = mu0, kappa0, alpha0, beta0
        self.reset()

    def reset(self):
        # logR: log of run-length distribution at current step
        self.logR = np.array([0.0])
        # Per-run-length suff stats for Normal-InverseGamma posterior
        self.mu     = np.array([self.mu0])
        self.kappa  = np.array([self.kappa0])
        self.alpha  = np.array([self.alpha0])
        self.beta   = np.array([self.beta0])

    def step(self, y):
        # Predictive: Student-t with df=2alpha, loc=mu, scale=sqrt(beta*(kappa+1)/(alpha*kappa))
        df = 2.0 * self.alpha
        scale = np.sqrt(self.beta * (self.kappa + 1.0) / (self.alpha * self.kappa))
        loglik = student_t.logpdf(y, df=df, loc=self.mu, scale=scale)

        # Marginal predictive (BEFORE update): log P(y_t | y_{1:t-1})
        log_marginal = logsumexp(self.logR + loglik)
        surprise = -log_marginal

        # Update run-length distribution in log space
        log1mh = np.log(1.0 - self.hazard)
        logh   = np.log(self.hazard)
        log_growth = self.logR + loglik + log1mh           # shape n
        log_cp = logsumexp(self.logR + loglik) + logh      # scalar
        new_logR = np.concatenate([[log_cp], log_growth])
        new_logR = new_logR - logsumexp(new_logR)          # normalise

        # Truncate to r_max+1 entries
        if len(new_logR) > self.r_max + 1:
            new_logR = new_logR[:self.r_max + 1]
            new_logR = new_logR - logsumexp(new_logR)

        # Update Normal-InverseGamma suff stats per run length
        new_mu     = np.concatenate([[self.mu0],    (self.kappa * self.mu + y) / (self.kappa + 1.0)])
        new_kappa  = np.concatenate([[self.kappa0], self.kappa + 1.0])
        new_alpha  = np.concatenate([[self.alpha0], self.alpha + 0.5])
        new_beta   = np.concatenate([[self.beta0],
                                     self.beta + (self.kappa * (y - self.mu) ** 2)
                                                 / (2.0 * (self.kappa + 1.0))])
        if len(new_mu) > self.r_max + 1:
            new_mu, new_kappa = new_mu[:self.r_max + 1], new_kappa[:self.r_max + 1]
            new_alpha, new_beta = new_alpha[:self.r_max + 1], new_beta[:self.r_max + 1]

        self.logR = new_logR
        self.mu, self.kappa, self.alpha, self.beta = new_mu, new_kappa, new_alpha, new_beta
        return surprise


def score_series(residuals, hazard, r_max=R_MAX):
    """Run BOCPD over a 1D residual series, return per-day surprise."""
    bocpd = BOCPDResidual(hazard=hazard, r_max=r_max)
    out = np.zeros(len(residuals), dtype=np.float64)
    for t, r in enumerate(residuals):
        out[t] = bocpd.step(float(r))
    return out


def get_residuals(d, k=0.5):
    """Use CUSUMDetector.fit_baseline + .baseline to obtain Pearson residuals."""
    cu = CUSUMDetector(k=k)
    cu.fit_baseline(d['x'], d['y'])
    mu = cu.baseline(len(d['x']))
    var = cu.dispersion_ * mu + 1e-9
    r = (d['x'] - mu) / np.sqrt(var)
    return r.astype(np.float64)


# ====== STAND-ALONE EVAL + CACHE WRITER ======

def evaluate_and_cache_signal(S, log_path, save_cache=True):
    rng = np.random.RandomState(RNG_STATE)
    for prior in SIGNALS:
        if prior == S:
            break
        Xp, _ = load_data(prior)
        sims_p = []
        for i, c in enumerate(Xp.columns):
            x = Xp[c].to_numpy(np.float32, copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims_p.append(dict(x=x, y=None, sim=f"sig{prior}_sim{i}", sim_idx=i))
        if sims_p:
            split_60_20_20(sims_p, rng)

    Xsig, Ysig = load_data(S)
    sims = []
    for i, c in enumerate(Xsig.columns):
        x = Xsig[c].to_numpy(np.float32, copy=False)
        y = Ysig[c].to_numpy(np.int32,  copy=False)
        if len(x) >= TRAIN_DAYS + VALID_DAYS:
            sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{i}", sim_idx=i))
    if not sims:
        print(f"[sig {S}] no complete sims; skip.", flush=True)
        return

    train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
    print(f"[sig {S}] splits: {len(train_sims)} train, {len(val_sims)} val, {len(test_sims)} test", flush=True)

    # Score val + test under each hazard, pick hazard by val sens-at-spec >= 0.97
    O_full_val = np.stack([d['y'] for d in val_sims], axis=1)
    O_full_test = np.stack([d['y'] for d in test_sims], axis=1)

    best = dict(score=-1e9, hazard=None, val_scores=None, test_scores=None,
                test_metrics=None)

    for hazard in HAZARD_GRID:
        t0 = time.time()
        val_cols, test_cols = [], []
        for d in val_sims:
            r = get_residuals(d)
            s_full = score_series(r, hazard)
            val_cols.append(s_full[len(d['x']) - VALID_DAYS:])
        for d in test_sims:
            r = get_residuals(d)
            s_full = score_series(r, hazard)
            test_cols.append(s_full[ABS_START:ABS_END])
        val_block  = np.column_stack(val_cols)
        test_block = np.column_stack(test_cols)

        # Convention: higher = anomalous. Convert to "higher = normal" for tuning.
        val_for_tune = -val_block.flatten(order="F")
        val_lengths  = [VALID_DAYS] * len(val_sims)
        c = tune_contamination_threshold(val_sims, val_lengths, val_for_tune,
                                         spec_target=SPEC_TARGET,
                                         w_sens=W_SENS, w_spec=W_SPEC)
        thr_val_neg  = np.percentile(val_for_tune, c * 100)
        # alarm if val_for_tune <= thr_val_neg, i.e., val_block >= -thr_val_neg
        thr_anom = -thr_val_neg
        # Compute val sens/spec under R-comparator
        yhat_v = (val_block.flatten(order="F") >= thr_anom).astype(int)
        A_list, ofs = [], 0
        for L in val_lengths:
            A_list.append(yhat_v[ofs:ofs + L]); ofs += L
        A_v = np.column_stack(A_list)
        sens_v = compute_sensitivity_R(A_v, O_full_val)
        spec_v = compute_specificity_R(A_v, O_full_val, IDX_RANGE)
        score_v = (W_SENS * (sens_v if not np.isnan(sens_v) else 0)
                   + W_SPEC * (spec_v if not np.isnan(spec_v) else 0))
        # Compute test metrics
        thr_test_neg = np.percentile((-test_block).flatten(order="F"), c * 100)
        thr_t_anom = -thr_test_neg
        yhat_t = (test_block.flatten(order="F") >= thr_t_anom).astype(int)
        A_tlist, ofs = [], 0
        for L in [WIN_LEN] * len(test_sims):
            A_tlist.append(yhat_t[ofs:ofs + L]); ofs += L
        A_t = np.column_stack(A_tlist)
        m = dict(sensitivity=compute_sensitivity_R(A_t, O_full_test),
                 specificity=compute_specificity_R(A_t, O_full_test, IDX_RANGE),
                 fpr=compute_fpr_R(A_t, O_full_test, IDX_RANGE),
                 pod=compute_pod_R(A_t, O_full_test),
                 timeliness=compute_timeliness_R(A_t, O_full_test))
        elapsed = time.time() - t0
        print(f"  haz={hazard:.4f} c={c:.3f}  val sens={sens_v:.3f} spec={spec_v:.3f}  "
              f"test sens={m['sensitivity']:.3f} tim={m['timeliness']:.3f} ({elapsed:.1f}s)",
              flush=True)

        if score_v > best['score']:
            best.update(score=score_v, hazard=hazard,
                        val_block=val_block, test_block=test_block,
                        contamination=c, val_metrics=dict(sens=sens_v, spec=spec_v),
                        test_metrics=m)

    # Save best hazard's scores
    val_block = best['val_block']
    test_block = best['test_block']
    if save_cache:
        # Stacker convention: higher = more normal -> save NEGATED scores
        import os
        cache_dir = f"score_cache/bocpd_resid_{MAG_TAG}"
        os.makedirs(cache_dir, exist_ok=True)
        val_csv  = f"{cache_dir}/bocpd_resid_val_scores_signal_{S}.csv"
        test_csv = f"{cache_dir}/bocpd_resid_test_scores_signal_{S}.csv"
        pd.DataFrame(-val_block,  columns=[f"sim_{i}" for i in range(val_block.shape[1])]).to_csv(val_csv,  index=False)
        pd.DataFrame(-test_block, columns=[f"sim_{i}" for i in range(test_block.shape[1])]).to_csv(test_csv, index=False)
        # Also write to flat path for small magnitude (backward compat with existing ensemble scripts)
        if MAG_TAG == "small":
            flat_val  = f"bocpd_resid_val_scores_signal_{S}.csv"
            flat_test = f"bocpd_resid_test_scores_signal_{S}.csv"
            pd.DataFrame(-val_block,  columns=[f"sim_{i}" for i in range(val_block.shape[1])]).to_csv(flat_val,  index=False)
            pd.DataFrame(-test_block, columns=[f"sim_{i}" for i in range(test_block.shape[1])]).to_csv(flat_test, index=False)
        print(f"[sig {S}] BEST hazard={best['hazard']:.4f}  saved {val_csv}, {test_csv}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"sig {S} best hazard={best['hazard']:.4f} c={best['contamination']:.3f} "
                f"test sens={best['test_metrics']['sensitivity']:.3f} "
                f"tim={best['test_metrics']['timeliness']:.3f}\n")

    return dict(signal=S, hazard=best['hazard'], **best['test_metrics'])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=str, default=None)
    ap.add_argument("--log", type=str, default="bocpd_resid_cache.log")
    args = ap.parse_args()
    target = SIGNALS if args.signals is None else [int(s) for s in args.signals.split(",")]
    print(f"BOCPD-on-residuals: signals {target}", flush=True)
    open(args.log, "w").write(f"start: {target}\n")
    rows = []
    cache_dir = f"score_cache/bocpd_resid_{MAG_TAG}"
    for S in target:
        # resumable: skip signals already fully cached (val + test scores + metric row),
        # so a Colab disconnect doesn't waste work AND the final results CSV stays complete.
        val_c = f"{cache_dir}/bocpd_resid_val_scores_signal_{S}.csv"
        test_c = f"{cache_dir}/bocpd_resid_test_scores_signal_{S}.csv"
        row_c = f"{cache_dir}/bocpd_resid_row_signal_{S}.csv"
        if os.path.exists(val_c) and os.path.exists(test_c) and os.path.exists(row_c):
            print(f"[sig {S}] already cached in {cache_dir} -> skip", flush=True)
            rows.append(pd.read_csv(row_c).iloc[0].to_dict())
            continue
        try:
            r = evaluate_and_cache_signal(S, args.log)
            if r:
                rows.append(r)
                os.makedirs(cache_dir, exist_ok=True)
                pd.DataFrame([r]).to_csv(row_c, index=False)
        except Exception as e:
            import traceback
            print(f"[sig {S}] FAILED: {e}", flush=True)
            traceback.print_exc()

    if rows:
        df = pd.DataFrame(rows)
        print("\n=== BOCPD-RESIDUAL ALONE (per signal) ===")
        print(df)
        means = df.mean(numeric_only=True)
        print("\nMeans:", means.to_dict())
        out_path = f"results/BOCPD_residual_alone_results_big_{MAG_TAG}.csv"
        df.to_csv(out_path, index=False)
        print(f"Wrote {out_path}", flush=True)
    print("ALL DONE.", flush=True)
