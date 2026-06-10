# run_lof.py
#!/usr/bin/env python3
"""
Local Outlier Factor (LOF) anomaly detector on tabular features.

- 60/20/20 split of simulations per signal (fixed RNG seed).
- Train LOF on first 6 years (TRAIN_DAYS) of train sims.
- Build validation features on last 49 weeks of val sims.
- Use R-comparator-based tuning of contamination (via decision score percentile).
- Test on absolute window [ABS_START:ABS_END) = [2205:2548) across test sims.
- Saves:
    - lof_alarms_signal_{S}.csv
    - lof_outbreaks_signal_{S}.csv
    - LOF_Tuned_all_days_per_sig.csv
    - LOF_per_sim_val.csv
    - LOF_per_sim_test.csv
"""

import numpy as np
import pandas as pd

from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import check_is_fitted

from anom_common import (
    load_data,
    split_60_20_20,
    create_features,
    sens_spec,
    compute_sensitivity,
    compute_specificity,
    compute_pod,
    compute_timeliness,
    compute_fpr_R,
    compute_specificity_R,
    compute_sensitivity_R,
    compute_pod_R,
    compute_timeliness_R,
    split_by_lengths,
    pod_anyhit,
    timeliness_single,
    tune_contamination_threshold,
    SIGNALS,
    TRAIN_DAYS,
    VALID_DAYS,
    RNG_STATE,
    ABS_START,
    ABS_END,
    WIN_LEN,
    IDX_RANGE,
    DAYS,
    YEARS,
    MAG_TAG,
)


class LOFNovelty:
    """
    LOF with novelty detection.

    decision_function(X) = score_samples(X)
    (higher = more normal).
    """

    def __init__(self, n_neighbors=20, metric="minkowski", n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self._lof = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self._lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination="auto",  # we handle contamination via tuning
            novelty=True,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self._lof.fit(X)
        return self

    def decision_function(self, X):
        check_is_fitted(self._lof)
        X = np.asarray(X, dtype=float)
        scores = self._lof.score_samples(X)
        return scores  # higher = more normal


if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)

    summary_all = {}
    rows_val, rows_test = [], []

    # (WINDOW_SIZE, N_NEIGHBORS)
    HP_GRID = [
        (7,  10),
        (7,  20),
        (14, 20),
        (21, 30),
    ]

    # Optional signal subset (e.g. to complete an interrupted run): SYND_LOF_SIGNALS="15,16".
    import os as _os0
    _subset = _os0.environ.get("SYND_LOF_SIGNALS", "").strip()
    SIGNALS_TO_RUN = [int(s) for s in _subset.split(",") if s.strip()] if _subset else list(SIGNALS)
    print(f"Running LOF on signals: {SIGNALS_TO_RUN}")

    for S in SIGNALS_TO_RUN:
        print(f"\n--- Signal {S} (LOF) ---")
        Xsig, Ysig = load_data(S)

        sims = []
        for sim_idx, col in enumerate(Xsig.columns):
            x = Xsig[col].to_numpy(np.float32, copy=False)
            y = Ysig[col].to_numpy(np.int32,  copy=False)
            # need at least TRAIN_DAYS + VALID_DAYS total length
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(x=x, y=y, sim=f"sig{S}_sim{sim_idx}", sim_idx=sim_idx))

        if not sims:
            print("No complete sims; skip.")
            continue

        train_sims, val_sims, test_sims_all = split_60_20_20(sims, rng)
        print(f"  Splits: {len(train_sims)} train, {len(val_sims)} val, {len(test_sims_all)} test")

        def build_train_matrix(window_size):
            XtrL = []
            for d in train_sims:
                feats = create_features(d["x"][:TRAIN_DAYS], window_size)
                if len(feats):
                    XtrL.append(feats)
            return np.concatenate(XtrL) if XtrL else np.empty((0, 20), np.float32)

        def build_val_tail(window_size):
            """
            Build validation windows with (window_size-1) days of context so that
            the first prediction aligns with the FIRST day of the 49-week tail.

            Returns:
                Xv, Yv, lengths, used_val_sims
            where used_val_sims is the subset of val_sims that contributed data.
            """
            XvL, YvL, lengths = [], [], []
            used_val_sims = []
            for d in val_sims:
                x = d["x"]
                y = d["y"]
                tail_start = len(x) - VALID_DAYS
                ctx_start  = tail_start - (window_size - 1)
                if ctx_start < 0:
                    continue
                x_ctx_tail = x[ctx_start:tail_start + VALID_DAYS]
                y_ctx_tail = y[ctx_start:tail_start + VALID_DAYS]
                feats = create_features(x_ctx_tail, window_size)
                if len(feats) == VALID_DAYS:
                    y_al = y_ctx_tail[window_size - 1:window_size - 1 + VALID_DAYS].astype(int)
                    XvL.append(feats)
                    YvL.append(y_al)
                    lengths.append(len(y_al))
                    used_val_sims.append(d)
            Xv = np.concatenate(XvL) if XvL else np.empty((0, 20), np.float32)
            Yv = np.concatenate(YvL) if YvL else np.empty((0,), np.int32)
            return Xv, Yv, lengths, used_val_sims

        best = dict(
            score=-1.0,
            params=None,
            contamination=None,
            val_lengths=None,
            val_scores=None,
            Yval=None,
            thr_val=None,
            val_sims_used=None,
        )

        # ---- hyperparam search ----
        for (WINDOW_SIZE, N_NEI) in HP_GRID:
            Xtr = build_train_matrix(WINDOW_SIZE)
            if not len(Xtr):
                continue

            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)

            Xval, Yval, val_lengths, val_sims_used = build_val_tail(WINDOW_SIZE)
            Xval_s = scaler.transform(Xval) if len(Xval) else Xval

            lof = LOFNovelty(n_neighbors=N_NEI, n_jobs=-1).fit(Xtr_s)

            if len(Xval_s):
                val_scores = lof.decision_function(Xval_s)  # higher = more normal
                c_best = tune_contamination_threshold(
                    val_sims_used, val_lengths, val_scores
                )
                thr = np.percentile(val_scores, c_best * 100)
                yhat = (val_scores <= thr).astype(int)

                # R-comparator metrics for validation
                O_full_val = np.stack([d["y"] for d in val_sims_used], axis=1)
                A_list, offset = [], 0
                for L in val_lengths:
                    if L > 0:
                        A_list.append(yhat[offset:offset+L])
                        offset += L
                if A_list and len(A_list) == len(val_sims_used):
                    A = np.column_stack(A_list)
                    s  = compute_sensitivity_R(A, O_full_val)
                    sp = compute_specificity_R(A, O_full_val, IDX_RANGE)
                    # simple score: require specificity target (enforced inside tuner already)
                    score = (2.0 * s + 3.0 * sp) if (not np.isnan(sp)) else -1.0
                else:
                    score = -1.0

                if score > best["score"]:
                    best.update(
                        score=score,
                        params=dict(
                            WINDOW_SIZE=WINDOW_SIZE,
                            N_NEI=N_NEI,
                            scaler=scaler,
                            model=lof,
                        ),
                        contamination=c_best,
                        val_lengths=val_lengths,
                        val_scores=val_scores,
                        Yval=Yval,
                        thr_val=thr,
                        val_sims_used=val_sims_used,
                    )
            else:
                # no validation features for this HP; keep only if nothing else found
                if best["params"] is None:
                    best.update(
                        score=0.0,
                        params=dict(
                            WINDOW_SIZE=WINDOW_SIZE,
                            N_NEI=N_NEI,
                            scaler=scaler,
                            model=lof,
                        ),
                        contamination=0.02,
                        val_lengths=[],
                        val_scores=np.array([]),
                        Yval=np.array([]),
                        thr_val=np.nan,
                        val_sims_used=[],
                    )

        if best["params"] is None:
            print("  No training/validation features; skipping signal.")
            continue

        P = best["params"]
        WINDOW_SIZE = P["WINDOW_SIZE"]
        print(
            f"  Best HP → window={WINDOW_SIZE}, n_neighbors={P['N_NEI']}, "
            f"contamination≈{best['contamination']:.3f}"
        )

        # ---- per-sim validation metrics ----
        if len(best["val_scores"]):
            yhat_v = (best["val_scores"] <= best["thr_val"]).astype(int)
            splits_yhat = split_by_lengths(yhat_v, best["val_lengths"])
            splits_y    = split_by_lengths(best["Yval"],  best["val_lengths"])
            for d, yh, y in zip(best["val_sims_used"], splits_yhat, splits_y):
                if len(yh) == 0:
                    continue
                s_i, sp_i = sens_spec(y, yh)
                pod_i = pod_anyhit(yh, y)
                tim_i = timeliness_single(yh, y)
                rows_val.append(dict(
                    split="val", model="lof", signal=S, sim=d["sim"],
                    window=WINDOW_SIZE,
                    n_neighbors=P["N_NEI"],
                    contamination=best["contamination"], thr=best["thr_val"],
                    sens=s_i, spec=sp_i, pod=pod_i, timeliness=tim_i,
                    n_points=len(yh),
                ))

        # ---- refit LOF on all TRAIN data with best params ----
        Xtr = build_train_matrix(WINDOW_SIZE)
        scaler = P["scaler"]
        Xtr_s = scaler.transform(Xtr)

        lof_tuned = LOFNovelty(
            n_neighbors=P["N_NEI"],
            n_jobs=-1,
        ).fit(Xtr_s)
        print(f"  LOF trained on first 6 years (tabular features).")

        # ---- TEST: build features for ABS window ----
        per_sim_preds, per_sim_labels = [], []
        Xte_concat, Yte_concat = [], []
        test_sims_used = []

        for d in test_sims_all:
            x = d["x"]
            y = d["y"]

            ctx_lo = ABS_START - (WINDOW_SIZE - 1)
            if ctx_lo < 0 or ABS_END > len(x):
                continue

            x_ctx = x[ctx_lo:ABS_END]
            feats = create_features(x_ctx, WINDOW_SIZE)
            if len(feats) != WIN_LEN:
                continue

            y_win = y[ABS_START:ABS_END].astype(int)

            Xte_concat.append(feats)
            Yte_concat.append(y_win)
            test_sims_used.append(d)

        Xte = np.concatenate(Xte_concat) if Xte_concat else np.empty((0, 20), np.float32)
        Yte = np.concatenate(Yte_concat) if Yte_concat else np.empty((0,), np.int32)

        if len(Xte) == 0:
            print("  Testing: 0 feature vectors; skipping metrics.")
            continue

        print(
            f"  Testing window {ABS_START}:{ABS_END-1}: {len(Xte)} vectors, "
            f"{Yte.sum()} outbreaks ({100 * Yte.mean():.1f}%)"
        )

        Xte_s = scaler.transform(Xte)
        test_scores = lof_tuned.decision_function(Xte_s)
        thr_test = np.percentile(test_scores, best["contamination"] * 100)
        yhat_concat = (test_scores <= thr_test).astype(int)

        ofs = 0
        per_sim_preds.clear()
        per_sim_labels.clear()
        for y_win in Yte_concat:
            per_sim_preds.append(yhat_concat[ofs:ofs+WIN_LEN])
            ofs += WIN_LEN
            per_sim_labels.append(y_win)

        A = np.stack(per_sim_preds, axis=1)
        O = np.stack(per_sim_labels, axis=1)

        # save A/O + scores
        import os as _os
        cache_dir = f"score_cache/lof_{MAG_TAG}"
        _os.makedirs(cache_dir, exist_ok=True)
        alarm_filename = f"{cache_dir}/lof_alarms_signal_{S}.csv"
        pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
            alarm_filename, index=False
        )
        outbreak_filename = f"{cache_dir}/lof_outbreaks_signal_{S}.csv"
        pd.DataFrame(O, columns=[f"sim_{i}" for i in range(O.shape[1])]).to_csv(
            outbreak_filename, index=False
        )

        # Continuous scores
        test_score_mat = test_scores.reshape(len(test_sims_used), WIN_LEN).T
        pd.DataFrame(test_score_mat,
                     columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
                     ).to_csv(f"{cache_dir}/lof_test_scores_signal_{S}.csv", index=False)
        val_score_mat_cols = []
        _ofs = 0
        for _L in best["val_lengths"]:
            val_score_mat_cols.append(best["val_scores"][_ofs:_ofs + _L])
            _ofs += _L
        val_score_mat = np.column_stack(val_score_mat_cols)
        pd.DataFrame(val_score_mat,
                     columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
                     ).to_csv(f"{cache_dir}/lof_val_scores_signal_{S}.csv", index=False)
        if MAG_TAG == 'small':
            pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
                f"lof_alarms_signal_{S}.csv", index=False)
            pd.DataFrame(test_score_mat,
                         columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
                         ).to_csv(f"lof_test_scores_signal_{S}.csv", index=False)
            pd.DataFrame(val_score_mat,
                         columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
                         ).to_csv(f"lof_val_scores_signal_{S}.csv", index=False)
        print(f"Saved alarms: {alarm_filename}")
        print(f"Saved outbreaks: {outbreak_filename}")
        print(f"Saved scores: {cache_dir}/lof_{{val,test}}_scores_signal_{S}.csv")

        # R-comparator uses only sims that actually contributed predictions
        O_full_list = [d["y"] for d in test_sims_used]
        O_full = np.stack(O_full_list, axis=1)

        fpr_R  = compute_fpr_R(A, O_full, IDX_RANGE)
        spec_R = compute_specificity_R(A, O_full, IDX_RANGE)
        sens_R = compute_sensitivity_R(A, O_full)
        pod_R  = compute_pod_R(A, O_full)
        tim_R  = compute_timeliness_R(A, O_full, days=DAYS, years=YEARS)
        print(
            f"R-COMPARATOR → Sens={sens_R:.3f}, Spec={spec_R:.3f}, "
            f"FPR={fpr_R:.3f}, POD={pod_R:.3f}, Tim={tim_R:.3f}"
        )

        sens0 = compute_sensitivity(A, O)
        spec0 = compute_specificity(A, O)
        pod0  = compute_pod(A, O)
        tim0  = compute_timeliness(A, O)
        print(
            f"ORIGINAL     → Sens={sens0:.3f}, Spec={spec0:.3f}, "
            f"POD={pod0:.3f}, Tim={tim0:.3f}"
        )

        # per-sim TEST rows
        for d, yh, ytrue in zip(test_sims_used, per_sim_preds, per_sim_labels):
            s_i, sp_i = sens_spec(ytrue, yh)
            pod_i = pod_anyhit(yh, ytrue)
            tim_i = timeliness_single(yh, ytrue)
            rows_test.append(dict(
                split="test", model="lof", signal=S, sim=d["sim"],
                window=WINDOW_SIZE,
                n_neighbors=P["N_NEI"],
                contamination=best["contamination"], thr=thr_test,
                sens=s_i, spec=sp_i, pod=pod_i, timeliness=tim_i,
                n_points=len(yh),
            ))

        summary_all[S] = dict(
            sensitivity=sens_R,
            specificity=spec_R,
            fpr=fpr_R,
            pod=pod_R,
            timeliness=tim_R,
            contamination=best["contamination"],
            win=WINDOW_SIZE,
            n_neighbors=P["N_NEI"],
            sensitivity_orig=sens0,
            specificity_orig=spec0,
            pod_orig=pod0,
            timeliness_orig=tim0,
        )

    # ===== SUMMARY / CSVs =====
    if summary_all:
        df = pd.DataFrame.from_dict(summary_all, orient="index")
        df.index.name = "signal"
        print("\n=== SUMMARY (LOF, all signals) ===")
        print(df)
        print("\nMeans:\n", df.mean(numeric_only=True))
        import os as _os
        _os.makedirs("results", exist_ok=True)
        out_path = f"results/LOF_Tuned_per_sig_big_{MAG_TAG}.csv"
        # If completing a subset run, MERGE with the existing summary rather than overwrite.
        if _subset and _os.path.exists(out_path):
            prev = pd.read_csv(out_path)
            prev.columns = [c.strip() for c in prev.columns]
            sig_col = prev.columns[0]
            prev = prev.set_index(sig_col)
            prev.index = prev.index.astype(int)
            prev.index.name = "signal"
            prev = prev.drop(index=[s for s in df.index if s in prev.index], errors="ignore")
            df = pd.concat([prev, df]).sort_index()
            print(f"Merged subset {SIGNALS_TO_RUN} into existing summary -> {len(df)} signals total")
        df.to_csv(out_path)
        print(f"Wrote {out_path}")
        df.to_csv("LOF_Tuned_all_days_per_sig.csv")

        combined_summary = {}
        for signal in summary_all.keys():
            combined_summary[f"Signal_{signal}"] = {
                "Sensitivity_All": summary_all[signal]["sensitivity"],
                "Specificity_All": summary_all[signal]["specificity"],
                "POD_All": summary_all[signal]["pod"],
                "Timeliness_All": summary_all[signal]["timeliness"],
                "Contamination_All": summary_all[signal]["contamination"],
                "Win": summary_all[signal]["win"],
                "n_neighbors": summary_all[signal]["n_neighbors"],
            }
        pd.DataFrame.from_dict(combined_summary, orient="index").to_csv(
            "LOF_Tuned_results_per_sig.csv"
        )

    if rows_val:
        dfv = pd.DataFrame(rows_val)
        dfv.sort_values(["signal", "sim", "split"], inplace=True)
        dfv.to_csv("LOF_per_sim_val.csv", index=False)
        print("Saved: LOF_per_sim_val.csv")

    if rows_test:
        dft = pd.DataFrame(rows_test)
        dft.sort_values(["signal", "sim", "split"], inplace=True)
        dft.to_csv("LOF_per_sim_test.csv", index=False)
        print("Saved: LOF_per_sim_test.csv")
