#!/usr/bin/env python3
"""
PyTorch+MPS port of the production LSTM-AE architecture from LSTM_adapter.py.

Architecture mirrors the Keras model exactly:
  Encoder: BiLSTM(128, return_sequences=True)  -> BN -> Dropout(0.2)
           BiLSTM(64,  return_sequences=True)  -> BN -> Dropout(0.2)
           BiLSTM(32,  return_sequences=False) -> BN -> Dropout(0.3)
           Dense(4, linear)                   -> latent
  Decoder: RepeatVector(14)
           BiLSTM(32,  return_sequences=True)  -> BN -> Dropout(0.2)
           BiLSTM(64,  return_sequences=True)  -> BN -> Dropout(0.2)
           BiLSTM(128, return_sequences=True)  -> BN -> Dropout(0.2)
           TimeDistributed(Dense(1, linear))

Training: Huber loss, Adam(1e-3), batch 256, max 40 epochs,
          early stopping with patience=7 starting after min_epochs=20,
          restoring best weights by val_loss.

Output per signal S:
  lstm_ae_val_scores_signal_{S}.csv   shape (VALID_DAYS, n_val_sims)
  lstm_ae_test_scores_signal_{S}.csv  shape (WIN_LEN,    n_test_sims)
"""

import argparse
import copy
import os
import time
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from anom_common import (
    load_data, split_60_20_20,
    TRAIN_DAYS, VALID_DAYS, ABS_END, WIN_LEN,
    RNG_STATE, SIGNALS,
    fit_nb_baseline, pearson_residuals,
)


# Per-signal model-HP search reads these from env vars (defaults reproduce the
# original single-config run). LSTM_WIDTH scales the BiLSTM hidden sizes (128/64/32)
# uniformly; LSTM_OUT_DIR isolates each config so a downstream selection step can
# pick the best per signal by the same validation criterion used for the tabular methods.
WINDOW       = 14
LATENT       = int(os.environ.get("LSTM_LATENT", 4))
WIDTH        = float(os.environ.get("LSTM_WIDTH", 1.0))
EPOCHS_MAX   = int(os.environ.get("LSTM_EPOCHS", 40))
OUT_DIR      = os.environ.get("LSTM_OUT_DIR", ".")
# Residual-feature variant: feed standardised Pearson residuals (from the NB
# seasonal baseline fitted on each sim's training period) instead of raw counts,
# so the autoencoder never sees the annual cycle. Same architecture/loss.
RESIDUAL     = os.environ.get("LSTM_RESIDUAL", "0") == "1"
MIN_EPOCHS   = min(20, EPOCHS_MAX)
PATIENCE     = 7
BATCH_SIZE   = 256
LEARNING_RATE = 1e-3
TAIL_DAYS    = VALID_DAYS  # 343

DEVICE = torch.device("mps" if torch.backends.mps.is_available()
                      else ("cuda" if torch.cuda.is_available() else "cpu"))
print(f"Device: {DEVICE}")


class BiLSTMBlock(nn.Module):
    """Bidirectional LSTM + BN + Dropout. Matches Keras Bidirectional + BN + Dropout."""
    def __init__(self, in_dim, hidden, dropout=0.2, return_sequences=True):
        super().__init__()
        self.lstm = nn.LSTM(in_dim, hidden, batch_first=True, bidirectional=True)
        self.bn = nn.BatchNorm1d(hidden * 2)
        self.do = nn.Dropout(dropout)
        self.return_sequences = return_sequences

    def forward(self, x):
        out, (h, _) = self.lstm(x)
        if not self.return_sequences:
            # Concat forward last and backward last (Keras Bidirectional return_sequences=False)
            out = torch.cat([h[0], h[1]], dim=-1)              # [B, 2*hidden]
            out = self.bn(out)
            return self.do(out)
        # Apply BN per-channel: [B, L, C] -> [B, C, L] -> bn -> [B, L, C]
        out = self.bn(out.permute(0, 2, 1)).permute(0, 2, 1)
        return self.do(out)


class LSTMAEProd(nn.Module):
    def __init__(self, win=WINDOW, latent=LATENT, width=WIDTH):
        super().__init__()
        self.win = win
        # base hidden sizes (128/64/32) scaled uniformly by `width`; bidirectional
        # blocks output 2*hidden, so all inter-layer dims scale consistently.
        h1, h2, h3 = int(round(128 * width)), int(round(64 * width)), int(round(32 * width))
        self.enc1 = BiLSTMBlock(1,      h1, dropout=0.2, return_sequences=True)
        self.enc2 = BiLSTMBlock(2 * h1, h2, dropout=0.2, return_sequences=True)
        self.enc3 = BiLSTMBlock(2 * h2, h3, dropout=0.3, return_sequences=False)
        self.bottleneck = nn.Linear(2 * h3, latent)
        self.dec1 = BiLSTMBlock(latent, h3, dropout=0.2, return_sequences=True)
        self.dec2 = BiLSTMBlock(2 * h3, h2, dropout=0.2, return_sequences=True)
        self.dec3 = BiLSTMBlock(2 * h2, h1, dropout=0.2, return_sequences=True)
        self.proj = nn.Linear(2 * h1, 1)

    def forward(self, x):                     # x: [B, 14, 1]
        x = self.enc1(x)                      # [B, 14, 256]
        x = self.enc2(x)                      # [B, 14, 128]
        x = self.enc3(x)                      # [B, 64]
        z = self.bottleneck(x)                # [B, 4]
        rep = z.unsqueeze(1).repeat(1, self.win, 1)  # [B, 14, 4]
        x = self.dec1(rep)                    # [B, 14, 64]
        x = self.dec2(x)                      # [B, 14, 128]
        x = self.dec3(x)                      # [B, 14, 256]
        return self.proj(x)                   # [B, 14, 1]


def _make_windows(series, win=WINDOW, stride=1):
    s = np.asarray(series, dtype=np.float32)
    n = len(s)
    if n < win:
        return np.empty((0, win), dtype=np.float32)
    starts = range(0, n - win + 1, stride)
    return np.stack([s[i:i + win] for i in starts])


def stack_train_windows(train_sims, train_days):
    parts = [_make_windows(d['x'][:train_days]) for d in train_sims]
    parts = [p for p in parts if len(p)]
    return np.concatenate(parts) if parts else np.empty((0, WINDOW), np.float32)


def build_tail_windows(sims, tail_days):
    parts = []
    per_sim_len = []
    for d in sims:
        wins = _make_windows(d['x'][-tail_days:])
        per_sim_len.append(len(wins))
        if len(wins):
            parts.append(wins)
    if not parts:
        return np.empty((0, WINDOW), np.float32), per_sim_len
    return np.concatenate(parts), per_sim_len


def train_lstm_ae(Xtr, Xv):
    """Train with Huber loss, early stopping with restore best weights."""
    torch.manual_seed(RNG_STATE)
    np.random.seed(RNG_STATE)

    Xtr_t = torch.from_numpy(Xtr).unsqueeze(-1)
    train_loader = DataLoader(TensorDataset(Xtr_t), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    if len(Xv):
        Xv_t = torch.from_numpy(Xv).unsqueeze(-1)
        val_loader = DataLoader(TensorDataset(Xv_t), batch_size=BATCH_SIZE, shuffle=False)
    else:
        # 90/10 split of training as validation
        split_idx = int(0.9 * len(Xtr_t))
        perm = torch.randperm(len(Xtr_t))
        Xv_t = Xtr_t[perm[split_idx:]]
        Xtr_t2 = Xtr_t[perm[:split_idx]]
        train_loader = DataLoader(TensorDataset(Xtr_t2), batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
        val_loader = DataLoader(TensorDataset(Xv_t), batch_size=BATCH_SIZE, shuffle=False)

    model = LSTMAEProd().to(DEVICE)
    optim = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    huber = nn.HuberLoss(delta=1.0)

    best_val = float("inf")
    best_state = None
    epochs_since_best = 0

    for epoch in range(EPOCHS_MAX):
        t0 = time.time()
        model.train()
        train_loss = 0.0; ntrain = 0
        for (xb,) in train_loader:
            xb = xb.to(DEVICE)
            optim.zero_grad()
            out = model(xb)
            loss = huber(out, xb)
            loss.backward()
            optim.step()
            train_loss += float(loss.detach().cpu()); ntrain += 1
        train_loss /= max(ntrain, 1)

        model.eval()
        val_loss = 0.0; nval = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(DEVICE)
                out = model(xb)
                val_loss += float(huber(out, xb).detach().cpu()); nval += 1
        val_loss /= max(nval, 1)
        elapsed = time.time() - t0
        print(f"  epoch {epoch+1:2d}/{EPOCHS_MAX}  train={train_loss:.4f}  val={val_loss:.4f}  ({elapsed:.1f}s)",
              flush=True)

        if val_loss < best_val - 1e-6:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            epochs_since_best = 0
        else:
            epochs_since_best += 1

        if (epoch + 1) >= MIN_EPOCHS and epochs_since_best >= PATIENCE:
            print(f"  early stop at epoch {epoch+1} (best val={best_val:.4f})", flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)
    return model


@torch.no_grad()
def score_per_window_mse(model, X, batch=512):
    model.eval()
    if len(X) == 0:
        return np.array([], dtype=np.float32)
    Xt = torch.from_numpy(X).unsqueeze(-1)
    out = []
    for i in range(0, len(Xt), batch):
        x = Xt[i:i + batch].to(DEVICE)
        rec = model(x)
        out.append(((x - rec) ** 2).mean(dim=(1, 2)).cpu().numpy())
    return np.concatenate(out).astype(np.float32)


def per_sim_split(flat, n_sims, per_len):
    return flat.reshape(n_sims, per_len).T  # [per_len, n_sims]


def pad_lead(score_block, total_len, valid_len):
    pad_len = total_len - valid_len
    if pad_len <= 0:
        return score_block
    out = np.full((total_len, score_block.shape[1]), np.nan, dtype=np.float32)
    out[pad_len:, :] = score_block
    out[:pad_len, :] = score_block[0:1, :]
    return out


def cache_signal(S, log_path):
    rng = np.random.RandomState(RNG_STATE)
    # Walk SIGNALS to consume rng deterministically (matches run_stacked_meta.py)
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
        print(f"[sig {S}] no complete sims; skip.")
        return

    train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
    print(f"[sig {S}] splits: {len(train_sims)} train, {len(val_sims)} val, {len(test_sims)} test", flush=True)

    if RESIDUAL:
        # Replace each sim's count series with Pearson residuals from an NB seasonal
        # baseline fitted on that sim's own training period (outbreak-free first 6 yr).
        # All downstream windowing then operates on the residual stream.
        for d in (*train_sims, *val_sims, *test_sims):
            params, disp = fit_nb_baseline(d['x'][:TRAIN_DAYS])
            r = pearson_residuals(d['x'], params, disp)
            d['x'] = np.asarray(r, dtype=np.float32)
        print(f"[sig {S}] residual-feature mode: inputs replaced by Pearson residuals", flush=True)

    Xtr = stack_train_windows(train_sims, TRAIN_DAYS)
    if Xtr.size == 0:
        print(f"[sig {S}] no train windows; skip.")
        return
    mu = float(Xtr.mean()); sd = float(Xtr.std()) or 1.0
    Xtr = ((Xtr - mu) / sd).astype(np.float32)
    print(f"[sig {S}] train windows: {Xtr.shape}; mu={mu:.3f} sd={sd:.3f}", flush=True)

    Xv, _ = build_tail_windows(val_sims, TAIL_DAYS)
    if Xv.size:
        Xv = ((Xv - mu) / sd).astype(np.float32)
    Xte, te_lens = build_tail_windows(test_sims, TAIL_DAYS)
    if Xte.size:
        Xte = ((Xte - mu) / sd).astype(np.float32)

    t0 = time.time()
    model = train_lstm_ae(Xtr, Xv)
    print(f"[sig {S}] training done in {(time.time()-t0)/60:.1f} min", flush=True)

    val_scores = score_per_window_mse(model, Xv)
    test_scores = score_per_window_mse(model, Xte)

    valid_per_sim = TAIL_DAYS - WINDOW + 1   # 330
    val_block = per_sim_split(val_scores, len(val_sims), valid_per_sim)
    val_full = pad_lead(val_block, TAIL_DAYS, valid_per_sim)

    # Test scores split per sim using te_lens (each should equal valid_per_sim)
    expected = sum(te_lens)
    if expected != len(test_scores):
        raise RuntimeError(f"[sig {S}] test scores {len(test_scores)} != expected {expected}")
    cols = []
    ofs = 0
    for L in te_lens:
        cols.append(test_scores[ofs:ofs + L])
        ofs += L
    test_arr = np.column_stack(cols)
    test_full = pad_lead(test_arr, WIN_LEN, valid_per_sim)

    os.makedirs(OUT_DIR, exist_ok=True)
    val_csv  = os.path.join(OUT_DIR, f"lstm_ae_val_scores_signal_{S}.csv")
    test_csv = os.path.join(OUT_DIR, f"lstm_ae_test_scores_signal_{S}.csv")
    pd.DataFrame(val_full,  columns=[f"sim_{i}" for i in range(val_full.shape[1])]).to_csv(val_csv,  index=False)
    pd.DataFrame(test_full, columns=[f"sim_{i}" for i in range(test_full.shape[1])]).to_csv(test_csv, index=False)
    print(f"[sig {S}] saved {val_csv} {val_full.shape}, {test_csv} {test_full.shape}", flush=True)
    with open(log_path, "a") as f:
        f.write(f"sig {S} val {val_full.shape} test {test_full.shape}\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=str, default=None,
                    help="Comma-separated signals to cache (default: all)")
    ap.add_argument("--log", type=str, default="lstm_ae_prod_cache.log")
    args = ap.parse_args()

    target = SIGNALS if args.signals is None else [int(s) for s in args.signals.split(",")]
    print(f"Caching production-architecture LSTM-AE scores for signals: {target}", flush=True)
    open(args.log, "w").write(f"start: {target}\n")

    for S in target:
        # resumable: skip signals already cached (so a Colab disconnect doesn't waste work)
        vcsv = os.path.join(OUT_DIR, f"lstm_ae_val_scores_signal_{S}.csv")
        tcsv = os.path.join(OUT_DIR, f"lstm_ae_test_scores_signal_{S}.csv")
        if os.path.exists(vcsv) and os.path.exists(tcsv):
            print(f"[sig {S}] already cached in {OUT_DIR} -> skip", flush=True)
            continue
        try:
            cache_signal(S, args.log)
        except Exception as e:
            import traceback
            msg = f"[sig {S}] FAILED: {type(e).__name__}: {e}"
            print(msg, flush=True)
            traceback.print_exc()
            with open(args.log, "a") as f:
                f.write(msg + "\n")

    print("ALL DONE.", flush=True)
