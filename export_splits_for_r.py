#!/usr/bin/env python3
"""Export the Python-pipeline splits (per-signal sim_idx lists) for R consumption.

Outputs:
  splits_for_r.json : {"signal_S": {"train": [...], "val": [...], "test": [...]}, ...}
"""

import json
import numpy as np
from anom_common import load_data, split_60_20_20, TRAIN_DAYS, VALID_DAYS, RNG_STATE, SIGNALS


if __name__ == "__main__":
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)
    out = {}
    for S in SIGNALS:
        Xsig, Ysig = load_data(S)
        sims = []
        for sim_idx, col in enumerate(Xsig.columns):
            x = Xsig[col].to_numpy(np.float32, copy=False)
            if len(x) >= TRAIN_DAYS + VALID_DAYS:
                sims.append(dict(sim_idx=int(sim_idx), col=str(col)))
        # split_60_20_20 just needs len(sims); only uses .sim_idx for tracking
        train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
        out[f"signal_{S}"] = dict(
            train=[d['sim_idx'] for d in train_sims],
            val=[d['sim_idx']   for d in val_sims],
            test=[d['sim_idx']  for d in test_sims],
        )
        print(f"sig {S}: train={len(train_sims)} val={len(val_sims)} test={len(test_sims)}")
    with open("splits_for_r.json", "w") as f:
        json.dump(out, f)
    print("Saved splits_for_r.json")
