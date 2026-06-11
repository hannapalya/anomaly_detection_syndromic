#!/usr/bin/env python3
"""
Residual-feature variants for the four tabular detectors.

For each simulation, a Negative-Binomial seasonal+DOW baseline is fitted on the
first six years. The standard 20-dimensional sliding-window feature vector is
then constructed from Pearson residuals rather than raw counts. The detector
families, hyperparameter grids, validation calibration, and test evaluation
match the raw-count tabular runners.

Outputs:
  results/IF_residual_per_sig_big_{MAG}.csv
  results/KNN_residual_per_sig_big_{MAG}.csv
  results/LOF_residual_per_sig_big_{MAG}.csv
  results/OCSVM_residual_per_sig_big_{MAG}.csv
"""
import os
import argparse
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor, NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
from sklearn.utils.validation import check_is_fitted

from anom_common import (
    load_data, split_60_20_20, create_features, fit_nb_baseline,
    pearson_residuals, compute_sensitivity_R, compute_specificity_R,
    compute_fpr_R, compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold, TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END,
    WIN_LEN, IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
    MAG_TAG,
)


METHOD_ORDER = ("IF", "KNN", "LOF", "OCSVM")

IF_GRID = [
    dict(window=7, n_estimators=200, max_samples=0.7, max_features=0.6),
    dict(window=7, n_estimators=500, max_samples=0.9, max_features=0.8),
    dict(window=14, n_estimators=200, max_samples=0.9, max_features=0.6),
    dict(window=14, n_estimators=500, max_samples=0.7, max_features=0.8),
    dict(window=21, n_estimators=200, max_samples=0.7, max_features=0.8),
    dict(window=21, n_estimators=500, max_samples=0.9, max_features=0.6),
]

KNN_GRID = [
    dict(window=7, k=5),
    dict(window=7, k=15),
    dict(window=14, k=10),
    dict(window=21, k=10),
]

LOF_GRID = [
    dict(window=7, n_neighbors=10),
    dict(window=7, n_neighbors=20),
    dict(window=14, n_neighbors=20),
    dict(window=21, n_neighbors=30),
]

OCSVM_GRID = [
    dict(window=7, gamma="scale", nu=0.03),
    dict(window=7, gamma="scale", nu=0.05),
    dict(window=14, gamma="scale", nu=0.03),
    dict(window=14, gamma="scale", nu=0.05),
    dict(window=21, gamma="scale", nu=0.05),
]


class KNNAnomalyDetector:
    def __init__(self, n_neighbors=20, metric="minkowski", n_jobs=1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self._nn = None

    def fit(self, X, y=None):
        self._nn = NearestNeighbors(
            n_neighbors=self.n_neighbors,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self._nn.fit(np.asarray(X, dtype=float))
        return self

    def decision_function(self, X):
        check_is_fitted(self._nn)
        distances, _ = self._nn.kneighbors(
            np.asarray(X, dtype=float), n_neighbors=self.n_neighbors
        )
        return -distances[:, -1]


class LOFNovelty:
    def __init__(self, n_neighbors=20, metric="minkowski", n_jobs=1):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.n_jobs = n_jobs
        self._lof = None

    def fit(self, X, y=None):
        self._lof = LocalOutlierFactor(
            n_neighbors=self.n_neighbors,
            contamination="auto",
            novelty=True,
            metric=self.metric,
            n_jobs=self.n_jobs,
        )
        self._lof.fit(np.asarray(X, dtype=float))
        return self

    def decision_function(self, X):
        check_is_fitted(self._lof)
        return self._lof.score_samples(np.asarray(X, dtype=float))


def method_grid(method):
    return {
        "IF": IF_GRID,
        "KNN": KNN_GRID,
        "LOF": LOF_GRID,
        "OCSVM": OCSVM_GRID,
    }[method]


def residual_series(d):
    if "_resid_full" not in d:
        x = np.asarray(d["x"], dtype=np.float64)
        params, dispersion = fit_nb_baseline(x[:TRAIN_DAYS])
        d["_resid_full"] = pearson_residuals(x, params, dispersion, start=0)
    return d["_resid_full"]


def build_features_for_sim(d, window_size):
    x = np.asarray(d["x"], dtype=np.float64)
    y = np.asarray(d["y"], dtype=np.int32)
    resid_full = residual_series(d)

    train_feats = create_features(resid_full[:TRAIN_DAYS], window_size)

    val_ctx_start = len(x) - VALID_DAYS - (window_size - 1)
    if val_ctx_start < 0:
        val_feats, val_labels = None, None
    else:
        val_resid = resid_full[val_ctx_start:len(x)]
        val_feats = create_features(val_resid, window_size)
        val_labels = y[len(x) - VALID_DAYS:len(x)]

    test_ctx_start = ABS_START - (window_size - 1)
    if test_ctx_start < 0 or ABS_END > len(x):
        test_feats, test_labels = None, None
    else:
        test_resid = resid_full[test_ctx_start:ABS_END]
        test_feats = create_features(test_resid, window_size)
        test_labels = y[ABS_START:ABS_END]

    return train_feats, val_feats, test_feats, val_labels, test_labels


def build_feature_blocks(train_sims, val_sims, test_sims, window_size):
    train_blocks = []
    for d in train_sims:
        feats, _, _, _, _ = build_features_for_sim(d, window_size)
        if feats is not None and len(feats):
            train_blocks.append(feats)

    val_blocks, val_labels, val_used = [], [], []
    for d in val_sims:
        _, feats, _, labels, _ = build_features_for_sim(d, window_size)
        if feats is not None and labels is not None and len(feats) == len(labels):
            val_blocks.append(feats)
            val_labels.append(labels)
            val_used.append(d)

    test_blocks, test_labels, test_used = [], [], []
    for d in test_sims:
        _, _, feats, _, labels = build_features_for_sim(d, window_size)
        if feats is not None and labels is not None and len(feats) == len(labels):
            test_blocks.append(feats)
            test_labels.append(labels)
            test_used.append(d)

    return dict(
        Xtr=np.concatenate(train_blocks, axis=0) if train_blocks else np.empty((0, 20), np.float32),
        Xval=np.concatenate(val_blocks, axis=0) if val_blocks else np.empty((0, 20), np.float32),
        Xte=np.concatenate(test_blocks, axis=0) if test_blocks else np.empty((0, 20), np.float32),
        val_lengths=[len(v) for v in val_blocks],
        test_lengths=[len(t) for t in test_blocks],
        val_labels=val_labels,
        test_labels=test_labels,
        val_sims=val_used,
        test_sims=test_used,
    )


def fit_model(method, params, Xtr):
    fit_X = Xtr
    if method == "OCSVM" and len(fit_X) > 30000:
        idx = np.random.RandomState(RNG_STATE).choice(len(fit_X), 30000, replace=False)
        fit_X = fit_X[idx]

    scaler = StandardScaler().fit(fit_X)
    fit_X_scaled = scaler.transform(fit_X)

    if method == "IF":
        model = IsolationForest(
            n_estimators=params["n_estimators"],
            contamination=0.05,
            max_samples=params["max_samples"],
            max_features=params["max_features"],
            random_state=RNG_STATE,
            n_jobs=1,
            bootstrap=False,
        ).fit(fit_X_scaled)
    elif method == "KNN":
        model = KNNAnomalyDetector(n_neighbors=params["k"], n_jobs=1).fit(fit_X_scaled)
    elif method == "LOF":
        model = LOFNovelty(n_neighbors=params["n_neighbors"], n_jobs=1).fit(fit_X_scaled)
    elif method == "OCSVM":
        model = OneClassSVM(
            kernel="rbf",
            gamma=params["gamma"],
            nu=params["nu"],
            cache_size=500,
        ).fit(fit_X_scaled)
    else:
        raise ValueError(f"Unknown method: {method}")

    return model, scaler


def score_model(model, scaler, X):
    if not len(X):
        return np.array([], dtype=float)
    return model.decision_function(scaler.transform(X))


def split_scores(scores, lengths):
    cols = []
    offset = 0
    for length in lengths:
        cols.append(scores[offset:offset + length])
        offset += length
    return np.column_stack(cols) if cols else np.empty((0, 0), dtype=float)


def validation_objective(val_sims, val_lengths, val_scores, contamination):
    threshold = np.percentile(val_scores, contamination * 100)
    yhat = (val_scores <= threshold).astype(int)
    alarm_matrix = split_scores(yhat, val_lengths).astype(int)
    if alarm_matrix.size == 0 or alarm_matrix.shape[1] != len(val_sims):
        return -1.0, threshold, np.nan, np.nan

    O_full = np.stack([d["y"] for d in val_sims], axis=1)
    sensitivity = compute_sensitivity_R(alarm_matrix, O_full)
    specificity = compute_specificity_R(alarm_matrix, O_full, IDX_RANGE)
    if np.isnan(specificity):
        score = -1.0
    elif specificity >= SPEC_TARGET:
        score = W_SENS * (0.0 if np.isnan(sensitivity) else sensitivity) + W_SPEC * specificity
    else:
        score = specificity
    return score, threshold, sensitivity, specificity


def evaluate_signal(signal, methods=METHOD_ORDER):
    rng = np.random.RandomState(RNG_STATE)
    for prior in SIGNALS:
        if prior == signal:
            break
        Xp, _ = load_data(prior)
        prior_sims = []
        for i, column in enumerate(Xp.columns):
            x = Xp[column].to_numpy(np.float32, copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                prior_sims.append(dict(x=x, y=None, sim_idx=i))
        if prior_sims:
            split_60_20_20(prior_sims, rng)

    Xsig, Ysig = load_data(signal)
    sims = []
    for i, column in enumerate(Xsig.columns):
        x = Xsig[column].to_numpy(np.float32, copy=False)
        y = Ysig[column].to_numpy(np.int32, copy=False)
        if len(x) >= TRAIN_DAYS + VALID_DAYS:
            sims.append(dict(x=x, y=y, sim=f"sig{signal}_sim{i}", sim_idx=i))

    train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
    feature_cache = {}
    results_out = {}

    for method in methods:
        best = None
        for params in method_grid(method):
            window_size = params["window"]
            if window_size not in feature_cache:
                feature_cache[window_size] = build_feature_blocks(
                    train_sims, val_sims, test_sims, window_size
                )
            blocks = feature_cache[window_size]
            if not len(blocks["Xtr"]) or not len(blocks["Xval"]) or not len(blocks["Xte"]):
                continue
            try:
                model, scaler = fit_model(method, params, blocks["Xtr"])
                val_scores = score_model(model, scaler, blocks["Xval"])
            except Exception as exc:
                print(f"  {method} params={params} failed: {exc}", flush=True)
                continue

            contamination = tune_contamination_threshold(
                blocks["val_sims"],
                blocks["val_lengths"],
                val_scores,
                spec_target=SPEC_TARGET,
                w_sens=W_SENS,
                w_spec=W_SPEC,
            )
            score, threshold, sensitivity, specificity = validation_objective(
                blocks["val_sims"], blocks["val_lengths"], val_scores, contamination
            )
            if best is None or score > best["score"]:
                best = dict(
                    score=score,
                    params=params,
                    model=model,
                    scaler=scaler,
                    blocks=blocks,
                    val_scores=val_scores,
                    contamination=contamination,
                    threshold=threshold,
                    val_sensitivity=sensitivity,
                    val_specificity=specificity,
                )

        if best is None:
            print(f"  {method}-residual: no usable config", flush=True)
            continue

        blocks = best["blocks"]
        test_scores = score_model(best["model"], best["scaler"], blocks["Xte"])
        yhat_test = (test_scores <= best["threshold"]).astype(int)
        A = split_scores(yhat_test, blocks["test_lengths"]).astype(int)
        O = np.stack(blocks["test_labels"], axis=1)
        O_full = np.stack([d["y"] for d in blocks["test_sims"]], axis=1)

        metrics = dict(
            sensitivity=compute_sensitivity_R(A, O_full),
            specificity=compute_specificity_R(A, O_full, IDX_RANGE),
            fpr=compute_fpr_R(A, O_full, IDX_RANGE),
            pod=compute_pod_R(A, O_full),
            timeliness=compute_timeliness_R(A, O_full),
            contamination=best["contamination"],
            threshold=best["threshold"],
            val_sensitivity=best["val_sensitivity"],
            val_specificity=best["val_specificity"],
            **best["params"],
        )
        results_out[method] = metrics

        method_key = method.lower()
        cache_dir = f"score_cache/{method_key}_residual_{MAG_TAG}"
        os.makedirs(cache_dir, exist_ok=True)

        val_score_mat = split_scores(best["val_scores"], blocks["val_lengths"])
        test_score_mat = split_scores(test_scores, blocks["test_lengths"])
        pd.DataFrame(
            val_score_mat, columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
        ).to_csv(f"{cache_dir}/{method_key}_residual_val_scores_signal_{signal}.csv", index=False)
        pd.DataFrame(
            test_score_mat, columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
        ).to_csv(f"{cache_dir}/{method_key}_residual_test_scores_signal_{signal}.csv", index=False)
        pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
            f"{cache_dir}/{method_key}_residual_alarms_signal_{signal}.csv", index=False
        )
        pd.DataFrame(O, columns=[f"sim_{i}" for i in range(O.shape[1])]).to_csv(
            f"{cache_dir}/{method_key}_residual_outbreaks_signal_{signal}.csv", index=False
        )

        if MAG_TAG == "small":
            pd.DataFrame(
                val_score_mat, columns=[f"sim_{i}" for i in range(val_score_mat.shape[1])]
            ).to_csv(f"{method_key}_residual_val_scores_signal_{signal}.csv", index=False)
            pd.DataFrame(
                test_score_mat, columns=[f"sim_{i}" for i in range(test_score_mat.shape[1])]
            ).to_csv(f"{method_key}_residual_test_scores_signal_{signal}.csv", index=False)
            pd.DataFrame(A, columns=[f"sim_{i}" for i in range(A.shape[1])]).to_csv(
                f"{method_key}_residual_alarms_signal_{signal}.csv", index=False
            )
            pd.DataFrame(O, columns=[f"sim_{i}" for i in range(O.shape[1])]).to_csv(
                f"{method_key}_residual_outbreaks_signal_{signal}.csv", index=False
            )

    return results_out


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--signals", type=str, default=None,
                        help="Comma-separated signal numbers. Default: all signals.")
    parser.add_argument("--methods", type=str, default=None,
                        help="Comma-separated methods from IF,KNN,LOF,OCSVM. Default: all.")
    parser.add_argument("--merge", action="store_true",
                        help="Merge subset results into existing per-signal CSVs.")
    args = parser.parse_args()

    target_signals = SIGNALS if args.signals is None else [
        int(s.strip()) for s in args.signals.split(",") if s.strip()
    ]
    target_methods = METHOD_ORDER if args.methods is None else tuple(
        m.strip().upper() for m in args.methods.split(",") if m.strip()
    )
    invalid_methods = sorted(set(target_methods) - set(METHOD_ORDER))
    if invalid_methods:
        raise ValueError(f"Unknown residual methods: {invalid_methods}")
    merge_outputs = args.merge or set(target_signals) != set(SIGNALS)

    print(f"Residual-feature ML runner — magnitude={MAG_TAG}")
    print(f"  Signals: {target_signals}")
    print(f"  Methods: {', '.join(target_methods)}")
    print("=" * 80)

    rows = {method: [] for method in target_methods}
    for signal in target_signals:
        print(f"[sig {signal}] processing...", flush=True)
        try:
            result = evaluate_signal(signal, methods=target_methods)
        except Exception as exc:
            import traceback
            print(f"[sig {signal}] FAILED: {exc}", flush=True)
            traceback.print_exc()
            continue
        for method, metrics in result.items():
            print(
                f"  {method}-residual: sens={metrics['sensitivity']:.3f} "
                f"spec={metrics['specificity']:.3f} pod={metrics['pod']:.3f} "
                f"tim={metrics['timeliness']:.3f}",
                flush=True,
            )
            rows[method].append(dict(signal=signal, **metrics))

    os.makedirs("results", exist_ok=True)
    for method in target_methods:
        if not rows[method]:
            continue
        df = pd.DataFrame(rows[method]).set_index("signal")
        out_path = f"results/{method}_residual_per_sig_big_{MAG_TAG}.csv"
        if merge_outputs and os.path.exists(out_path):
            previous = pd.read_csv(out_path)
            signal_col = previous.columns[0]
            previous = previous.set_index(signal_col)
            previous.index = previous.index.astype(int)
            previous = previous.drop(index=df.index, errors="ignore")
            df = pd.concat([previous, df]).sort_index()
        print(f"\n=== {method}-RESIDUAL (per signal) ===")
        print(df)
        print("Means:\n", df.mean(numeric_only=True))
        df.to_csv(out_path)
        print(f"Wrote {out_path}")
