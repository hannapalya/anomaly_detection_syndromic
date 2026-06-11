#!/usr/bin/env python3
"""Chunked Colab runner for methodology-correction reruns."""
import argparse
import os
import subprocess
import sys
import tarfile
from pathlib import Path


RUNNERS = {
    "if": ["python", "IsolationForest.py"],
    "knn": ["python", "run_knn.py"],
    "lof": ["python", "run_lof.py"],
    "ocsvm": ["python", "run_ocsvm.py"],
    "nbhmm": ["python", "run_nbhmm.py"],
    "cusum": ["python", "run_cusum.py"],
    "bocpd": ["python", "run_bocpd_residual.py"],
    "ratechange": ["python", "run_ratechange_residual.py"],
    "residual": ["python", "run_residual_ml.py"],
    "vae": ["python", "run_vae_count.py"],
    "ensemble": ["python", "run_unsup_or_ensemble.py"],
}


def split_csv(value):
    return [part.strip() for part in value.split(",") if part.strip()]


def expand_signals(value):
    if not value:
        return None
    signals = []
    for part in split_csv(value):
        if "-" in part:
            lo, hi = [int(x) for x in part.split("-", 1)]
            signals.extend(range(lo, hi + 1))
        else:
            signals.append(int(part))
    return ",".join(str(s) for s in signals)


def run(cmd, env):
    print("\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True, env=env)


def pack_outputs(repo_root, out_path):
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    include_dirs = ["results", "score_cache", "cloud_out"]
    include_globs = [
        "*_alarms_signal_*.csv",
        "*_outbreaks_signal_*.csv",
        "*_scores_signal_*.csv",
        "*_per_sim_*.csv",
        "*_per_sig*.csv",
    ]
    with tarfile.open(out_path, "w:gz") as tar:
        for dirname in include_dirs:
            path = repo_root / dirname
            if path.exists():
                tar.add(path, arcname=dirname)
        for pattern in include_globs:
            for path in repo_root.glob(pattern):
                if path.is_file():
                    tar.add(path, arcname=path.name)
    print(f"\nPacked outputs -> {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--magnitudes", default="small",
                        help="Comma-separated subset of small,medium,large.")
    parser.add_argument("--methods", default="residual",
                        help=f"Comma-separated methods: {','.join(RUNNERS)}.")
    parser.add_argument("--data-root", default="/content/syndromic",
                        help="Directory containing big_signal_datasets_<mag> folders.")
    parser.add_argument("--residual-signals", default=None,
                        help="For residual runs only, e.g. 1-4 or 1,2,3.")
    parser.add_argument("--residual-methods", default=None,
                        help="For residual runs only, e.g. IF,KNN or LOF,OCSVM.")
    parser.add_argument("--aggregate", action="store_true",
                        help="Run collect_all_results.py for each magnitude after methods.")
    parser.add_argument("--pack", default="/content/drive/MyDrive/syndromic_results/methodology_chunk.tgz",
                        help="Output tarball path. Use --pack '' to skip packing.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    methods = split_csv(args.methods.lower())
    unknown = sorted(set(methods) - set(RUNNERS))
    if unknown:
        raise SystemExit(f"Unknown methods: {unknown}")

    residual_signals = expand_signals(args.residual_signals)
    magnitudes = split_csv(args.magnitudes.lower())

    for mag in magnitudes:
        data_dir = Path(args.data_root) / f"big_signal_datasets_{mag}"
        if not data_dir.is_dir():
            raise SystemExit(f"Missing data directory: {data_dir}")
        env = os.environ.copy()
        env["SYND_DATA_DIR"] = str(data_dir)
        env.setdefault("PYTHONUNBUFFERED", "1")
        print(f"\n=== magnitude={mag} data={data_dir} ===", flush=True)

        for method in methods:
            cmd = list(RUNNERS[method])
            if method == "residual":
                if residual_signals:
                    cmd.extend(["--signals", residual_signals, "--merge"])
                if args.residual_methods:
                    cmd.extend(["--methods", args.residual_methods])
            run(cmd, env)

        if args.aggregate:
            run(["python", "collect_all_results.py", "--magnitude", mag], env)

    if args.pack:
        pack_outputs(repo_root, args.pack)


if __name__ == "__main__":
    sys.exit(main())
