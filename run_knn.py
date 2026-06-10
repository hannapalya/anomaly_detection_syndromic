# run_knn.py
#!/usr/bin/env python3
"""
kNN-based anomaly detector on tabular features.

- 60/20/20 split of simulations per signal (fixed RNG seed).
- Train kNN (NearestNeighbors) on first 6 years (TRAIN_DAYS) of train sims.
- Anomaly score = distance to k-th nearest neighbor; decision score = -distance
  (higher = more normal).
- Validation on last 49 weeks using R-comparator-based contamination tuning.
- Test on absolute window [ABS_START:ABS_END) = [2205:2548).
- Saves:
    - knn_alarms_signal_{S}.csv
    - knn_outbreaks_signal_{S}.csv
    - KNN_Tuned_all_days_per_sig.csv
    - KNN_per_sim_val.csv
    - KNN_per_sim_test.csv
"""

import numpy as np
import pandas as pd

from sklearn.neighbors import NearestNeighbors
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


class KNNAnomalyDetector:
    """
    kNN anomaly detector.

    decision_function(X) = -distance_to_kth_neighbor
    (higher = more normal).
    """

    def __init__(self, n_neighbors=20, metric="minkowski", n_jobs=-1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self._nn = None

    def fit(self, X, y=None):
        X = np.asarray(X, dtype=float)
        self._nn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self._nn.fit(X)
        return self

    def _kth_distances(self, X):
        distances, _ = self._nn.kneighbors(X, n_neighbors=self.n_neighbors)
        kth = distances[:, -1]
        return kth

    def decision_function(self, X):
        check_is_fitted(self._nn)
        X = np.asarray(X, dtype=float)
        kth = self._kth_distances(X)
        return -kth  # higher = more normal


if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)

    summary_all = {}
    rows_val, rows_test = [], []

    # (WINDOW_SIZE, K_NEIGH)
    HP_GRID = [
        (7,  5),
        (7,  15),
        (14, 10),
        (21, 10),
    ]

    for S in SIGNALS:
        print(f"\n--- Signal {S} (kNN) ---")
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
        for (WINDOW_SIZE, K_NEI) in HP_GRID:
            Xtr = build_train_matrix(WINDOW_SIZE)
            if not len(Xtr):
                continue

            scaler = StandardScaler().fit(Xtr)
            Xtr_s = scaler.transform(Xtr)

            Xval, Yval, val_lengths, val_sims_used = build_val_tail(WINDOW_SIZE)
            Xval_s = scaler.transform(Xval) if len(Xval) else Xval

            knn = KNNAnomalyDetector(n_neighbors=K_NEI, n_jobs=-1).fit(Xtr_s)

            if len(Xval_s):
                val_scores = knn.decision_function(Xval_s)  # higher = more normal
                c_best = tune_contamination_threshold(
                    val_sims_used, val_lengths, val_scores
                )
                thr = np.percentile(val_scores, c_best * 100)
                yhat = (val_scores <= thr).astype(int)

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
                    score = (2.0 * s + 3.0 * sp) if (not np.isnan(sp)) else -1.0
                else:
                    score = -1.0

                if score > best["score"]:
                    best.update(
                        score=score,
                        params=dict(
                            WINDOW_SIZE=WINDOW_SIZE,
                            K_NEI=K_NEI,
                            scaler=scaler,
                            model=knn,
                        ),
                        contamination=c_best,
                        val_lengths=val_lengths,
                        val_scores=val_scores,
                        Yval=Yval,
                        thr_val=thr,
                        val_sims_used=val_sims_used,
                    )
            else:
                if best["params"] is None:
                    best.update(
                        score=0.0,
                        params=dict(
                            WINDOW_SIZE=WINDOW_SIZE,
                            K_NEI=K_NEI,
                            scaler=scaler,
                            model=knn,
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
            f"  Best HP → window={WINDOW_SIZE}, k={P['K_NEI']}, "
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
                    split="val", model="knn", signal=S, sim=d["sim"],
                    window=WINDOW_SIZE,
                    k=P["K_NEI"],
                    contamination=best["contamination"], thr=best["thr_val"],
                    sens=s_i, spec=sp_i, pod=pod_i, timeliness=tim_i,
                    n_points=len(yh),
                ))

        # ---- refit kNN on all TRAIN data ----
        Xtr = build_train_matrix(WINDOW_SIZE)
        scaler = P["scaler"]
        Xtr_s = scaler.transform(Xtr)

        knn_tuned = KNNAnomalyDetector(
            n_neighbors=P["K_NEI"],
            n_jobs=-1,
        ).fit(Xtr_s)
        print(f"  kNN trained on first 6 years (tabular features).")

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
        test_scores = knn_tuned.decision_function(Xte_s)
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

        import os as _os
        cache_dir = f"score_cache/knn_{MAG_TAG}"
        _os.makedirs(cache_dir, exist_ok=True)
        alarm_filename = f"{cache_dir}/knn_alarms_signal_{S}.csv"
        pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
            alarm_filename, index=False
        )
        outbreak_filename = f"{cache_dir}/knn_outbreaks_signal_{S}.csv"
        pd.DataFrame(O, columns=[f"sim_{i}" for i in range(O.shape[1])]).to_csv(
            outbreak_filename, index=False
        )

        # Cache continuous scores (higher = more normal, matching stacker convention)
        test_score_mat = test_scores.reshape(len(test_sims_used), WIN_LEN).T
        pd.DataFrame(test_score_mat,
                     columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
                     ).to_csv(f"{cache_dir}/knn_test_scores_signal_{S}.csv", index=False)
        # Val scores from the tuned model (best["val_scores"]); split per sim.
        val_score_mat_cols = []
        _ofs = 0
        for _L in best["val_lengths"]:
            val_score_mat_cols.append(best["val_scores"][_ofs:_ofs + _L])
            _ofs += _L
        val_score_mat = np.column_stack(val_score_mat_cols)
        pd.DataFrame(val_score_mat,
                     columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
                     ).to_csv(f"{cache_dir}/knn_val_scores_signal_{S}.csv", index=False)
        # Backward-compat flat path for small magnitude
        if MAG_TAG == "small":
            pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
                f"knn_alarms_signal_{S}.csv", index=False)
            pd.DataFrame(test_score_mat,
                         columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
                         ).to_csv(f"knn_test_scores_signal_{S}.csv", index=False)
            pd.DataFrame(val_score_mat,
                         columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
                         ).to_csv(f"knn_val_scores_signal_{S}.csv", index=False)

        print(f"Saved alarms: {alarm_filename}")
        print(f"Saved outbreaks: {outbreak_filename}")
        print(f"Saved scores: {cache_dir}/knn_{{val,test}}_scores_signal_{S}.csv")

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

        for d, yh, ytrue in zip(test_sims_used, per_sim_preds, per_sim_labels):
            s_i, sp_i = sens_spec(ytrue, yh)
            pod_i = pod_anyhit(yh, ytrue)
            tim_i = timeliness_single(yh, ytrue)
            rows_test.append(dict(
                split="test", model="knn", signal=S, sim=d["sim"],
                window=WINDOW_SIZE,
                k=P["K_NEI"],
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
            k=P["K_NEI"],
            sensitivity_orig=sens0,
            specificity_orig=spec0,
            pod_orig=pod0,
            timeliness_orig=tim0,
        )

    if summary_all:
        df = pd.DataFrame.from_dict(summary_all, orient="index")
        df.index.name = "signal"
        print("\n=== SUMMARY (kNN, all signals) ===")
        print(df)
        print("\nMeans:\n", df.mean(numeric_only=True))
        import os as _os
        _os.makedirs("results", exist_ok=True)
        out_path = f"results/KNN_Tuned_per_sig_big_{MAG_TAG}.csv"
        df.to_csv(out_path)
        print(f"Wrote {out_path}")
        df.to_csv("KNN_Tuned_all_days_per_sig.csv")

        combined_summary = {}
        for signal in summary_all.keys():
            combined_summary[f"Signal_{signal}"] = {
                "Sensitivity_All": summary_all[signal]["sensitivity"],
                "Specificity_All": summary_all[signal]["specificity"],
                "POD_All": summary_all[signal]["pod"],
                "Timeliness_All": summary_all[signal]["timeliness"],
                "Contamination_All": summary_all[signal]["contamination"],
                "Win": summary_all[signal]["win"],
                "k": summary_all[signal]["k"],
            }
        pd.DataFrame.from_dict(combined_summary, orient="index").to_csv(
            "KNN_Tuned_results_per_sig.csv"
        )

    if rows_val:
        dfv = pd.DataFrame(rows_val)
        dfv.sort_values(["signal", "sim", "split"], inplace=True)
        dfv.to_csv("KNN_per_sim_val.csv", index=False)
        print("Saved: KNN_per_sim_val.csv")

    if rows_test:
        dft = pd.DataFrame(rows_test)
        dft.sort_values(["signal", "sim", "split"], inplace=True)
        dft.to_csv("KNN_per_sim_test.csv", index=False)
        print("Saved: KNN_per_sim_test.csv")
