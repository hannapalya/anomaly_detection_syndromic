#!/usr/bin/env python3
"""
Variational Autoencoder (VAE) with Negative-Binomial observation model for
unsupervised count time-series anomaly detection.

Key choices:
  - Observation model: Negative Binomial (counts are overdispersed).
    Decoder outputs (mu, alpha) per-timestep via softplus.
    NB parameterised as torch.distributions.NegativeBinomial(total_count=alpha,
    probs=alpha/(alpha+mu)).
  - Architecture: Conv1D encoder/decoder, window 14 days, latent dim 8.
  - Loss = NB-NLL (reconstruction) + beta * KL(q(z|x) || N(0,I)).
  - Trained UNSUPERVISED on training-period count windows.
  - Per-day score = NB-NLL of count_t under decoded distribution at the last
    timestep of the window ending at t. Higher = more anomalous.

Pipeline mirrors run_bocpd_residual.py:
  - Walk SIGNALS to consume RNG matching run_stacked_meta.py.
  - Per signal: split into train/val/test; train on training-period of train
    sims; score val tails (last VALID_DAYS) and test ABS window per sim.
  - Tune contamination via anom_common.tune_contamination_threshold.
  - Save CSVs (higher = more normal -> save NEGATED NLL):
      vae_val_scores_signal_{S}.csv      shape (343, n_val_sims)
      vae_test_scores_signal_{S}.csv     shape (343, n_test_sims)
  - Save results/VAE_per_sig_big_medium.csv with per-signal R-comparator metrics.
"""

import argparse
import os
import time
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from anom_common import (
    load_data, split_60_20_20,
    compute_sensitivity_R, compute_specificity_R, compute_fpr_R,
    compute_pod_R, compute_timeliness_R,
    tune_contamination_threshold,
    TRAIN_DAYS, VALID_DAYS, ABS_START, ABS_END, WIN_LEN,
    IDX_RANGE, RNG_STATE, SIGNALS, SPEC_TARGET, W_SENS, W_SPEC,
)


# ===== CONFIG =====
# Per-signal model-HP search reads these from env vars (defaults reproduce the
# original single-config run). VAE_OUT_DIR lets multiple configs write to
# separate, config-tagged directories so a downstream selection step can pick
# the best per signal by the same validation criterion used for the tabular methods.
WINDOW    = 14
LATENT    = int(os.environ.get("VAE_LATENT", 8))
HIDDEN    = int(os.environ.get("VAE_HIDDEN", 32))
EPOCHS    = int(os.environ.get("VAE_EPOCHS", 6))
OUT_DIR   = os.environ.get("VAE_OUT_DIR", ".")
BATCH     = 256
LR        = 1e-3
BETA      = 1.0
TRAIN_STRIDE = 2     # training window stride (dense=1 was slow)
NB_MU_MIN, NB_MU_MAX = 1e-3, 1e6
NB_AL_MIN, NB_AL_MAX = 1e-3, 1e6


def pick_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


DEVICE = pick_device()
print(f"VAE device: {DEVICE}", flush=True)


# ===== MODEL =====

class VAEDetector(nn.Module):
    """Conv1D VAE with Negative-Binomial decoder for count windows."""

    def __init__(self, win=WINDOW, latent=LATENT, hidden=HIDDEN):
        super().__init__()
        self.win = win
        self.latent = latent
        self.hidden = hidden

        # Encoder: 1 -> 32 -> 32 (preserves length via padding=1, kernel=3)
        self.enc_conv1 = nn.Conv1d(1, hidden, kernel_size=3, padding=1)
        self.enc_conv2 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.enc_fc    = nn.Linear(hidden * win, 2 * latent)

        # Decoder: latent -> 32*win -> reshape -> conv -> conv -> 2 outputs
        self.dec_fc    = nn.Linear(latent, hidden * win)
        self.dec_conv1 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.dec_conv2 = nn.Conv1d(hidden, 2, kernel_size=3, padding=1)  # log_mu_raw, log_alpha_raw

    def encode(self, x):                          # x: [B, win, 1] (z-scored)
        h = x.permute(0, 2, 1)                    # [B, 1, win]
        h = F.relu(self.enc_conv1(h))
        h = F.relu(self.enc_conv2(h))
        h = h.flatten(1)
        h = self.enc_fc(h)
        mu, logvar = h.chunk(2, dim=-1)
        # clamp for numerical stability
        logvar = torch.clamp(logvar, -8.0, 8.0)
        return mu, logvar

    def reparam(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.dec_fc(z)
        h = h.view(-1, self.hidden, self.win)
        h = F.relu(self.dec_conv1(h))
        out = self.dec_conv2(h)                   # [B, 2, win]
        # softplus for positivity then clamp
        mu_y    = F.softplus(out[:, 0, :]) + NB_MU_MIN
        alpha_y = F.softplus(out[:, 1, :]) + NB_AL_MIN
        mu_y    = torch.clamp(mu_y,    NB_MU_MIN, NB_MU_MAX)
        alpha_y = torch.clamp(alpha_y, NB_AL_MIN, NB_AL_MAX)
        return mu_y, alpha_y                      # [B, win], [B, win]

    def forward(self, x):                         # x_norm: [B, win, 1]
        mu_z, logvar_z = self.encode(x)
        z = self.reparam(mu_z, logvar_z)
        mu_y, alpha_y = self.decode(z)
        return mu_y, alpha_y, mu_z, logvar_z


# ===== NB NLL =====

def nb_nll(y, mu, alpha, eps=1e-8):
    """
    Negative-binomial NLL for overdispersed counts.

    Using the parameterisation:
        p(y; mu, alpha) where alpha = total_count, p = alpha/(alpha+mu)
        Mean = mu, Variance = mu + mu^2 / alpha
    Returns per-element NLL, same shape as y.
    """
    mu = mu.clamp(min=NB_MU_MIN, max=NB_MU_MAX)
    alpha = alpha.clamp(min=NB_AL_MIN, max=NB_AL_MAX)
    # log P(y) = lgamma(y+alpha) - lgamma(y+1) - lgamma(alpha)
    #           + alpha*log(alpha/(alpha+mu)) + y*log(mu/(alpha+mu))
    log_unnorm = (
        torch.lgamma(y + alpha)
        - torch.lgamma(y + 1.0)
        - torch.lgamma(alpha)
        + alpha * (torch.log(alpha + eps) - torch.log(alpha + mu + eps))
        + y * (torch.log(mu + eps) - torch.log(alpha + mu + eps))
    )
    return -log_unnorm


def kl_standard_normal(mu, logvar):
    """KL(N(mu, sigma^2) || N(0, I)) summed over latent dims."""
    return -0.5 * torch.sum(1.0 + logvar - mu.pow(2) - logvar.exp(), dim=-1)


# ===== DATA =====

def make_windows_counts(series, win=WINDOW, stride=1):
    s = np.asarray(series, dtype=np.float32)
    n = len(s)
    if n < win:
        return np.empty((0, win), dtype=np.float32)
    starts = range(0, n - win + 1, stride)
    return np.stack([s[i:i + win] for i in starts])


def stack_train_windows(train_sims, train_days, stride=TRAIN_STRIDE):
    parts = []
    for d in train_sims:
        w = make_windows_counts(d['x'][:train_days], stride=stride)
        if len(w):
            parts.append(w)
    if not parts:
        return np.empty((0, WINDOW), dtype=np.float32)
    return np.concatenate(parts)


def windows_for_range(series, start_day, end_day):
    """Return a window for each day t in [start_day, end_day) ending at t.
    Window covers indices [t - WINDOW + 1, t + 1).
    Pads at the start by clamping (uses first available window if t < WINDOW-1)."""
    s = np.asarray(series, dtype=np.float32)
    n = len(s)
    out = np.zeros((end_day - start_day, WINDOW), dtype=np.float32)
    for k, t in enumerate(range(start_day, end_day)):
        a = t - WINDOW + 1
        if a < 0:
            # pad-front with the first value
            pad = -a
            chunk = np.concatenate([np.full(pad, s[0], dtype=np.float32),
                                    s[:t + 1].astype(np.float32)])
        else:
            chunk = s[a:t + 1].astype(np.float32)
        out[k] = chunk
    return out


# ===== TRAINING =====

def train_vae(model, X_norm_windows, X_count_windows, epochs=EPOCHS, batch=BATCH,
              lr=LR, beta=BETA, device=DEVICE, mask_tail=3):
    """Train VAE on count windows with NB observation model.

    During training we randomly mask the last `mask_tail` days of the encoder
    input on each batch (carry-forward fill from day -(mask_tail+1)). This
    forces the encoder to learn to predict the masked recent days from older
    context, mirroring how the model is used at scoring time. KL warmup over
    the first 3 epochs prevents posterior collapse.
    """
    torch.manual_seed(RNG_STATE)
    np.random.seed(RNG_STATE)

    Xn = torch.from_numpy(X_norm_windows).unsqueeze(-1)   # [N, win, 1]
    Xc = torch.from_numpy(X_count_windows)                # [N, win]
    ds = TensorDataset(Xn, Xc)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=True)

    opt = torch.optim.Adam(model.parameters(), lr=lr)
    model.to(device)
    history = []

    for epoch in range(epochs):
        # Linear KL warmup over first 3 epochs from 0 -> beta to avoid posterior collapse
        beta_eff = beta * min(1.0, (epoch + 1) / 3.0)
        model.train()
        t0 = time.time()
        tot_loss = tot_rec = tot_kl = 0.0
        nb = 0
        for (xn, xc) in loader:
            xn = xn.to(device)
            xc = xc.to(device)
            # Carry-forward mask of last `mask_tail` days
            if mask_tail and mask_tail > 0 and mask_tail < xn.shape[1]:
                ref = xn[:, -(mask_tail + 1):-mask_tail, :]
                xn = xn.clone()
                xn[:, -mask_tail:, :] = ref.expand(-1, mask_tail, -1)
            opt.zero_grad()
            mu_y, alpha_y, mu_z, logvar_z = model(xn)
            # Weight reconstruction toward the masked tail (where the prediction
            # task is non-trivial) but still include the rest at lower weight.
            full_nll = nb_nll(xc, mu_y, alpha_y)              # [B, win]
            if mask_tail and mask_tail > 0:
                w = torch.ones(xn.shape[1], device=device)
                w[-mask_tail:] = 3.0   # emphasize masked positions
                rec = (full_nll * w).sum(dim=-1).mean()
            else:
                rec = full_nll.sum(dim=-1).mean()
            kl  = kl_standard_normal(mu_z, logvar_z).mean()
            loss = rec + beta_eff * kl
            if not torch.isfinite(loss):
                continue
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot_loss += float(loss.detach().cpu()); tot_rec += float(rec.detach().cpu())
            tot_kl   += float(kl.detach().cpu()); nb += 1
        nb = max(nb, 1)
        history.append((tot_loss / nb, tot_rec / nb, tot_kl / nb))
        print(f"    epoch {epoch+1}/{epochs} beta={beta_eff:.2f}  loss={tot_loss/nb:.3f} "
              f"rec={tot_rec/nb:.3f} kl={tot_kl/nb:.3f}  ({time.time()-t0:.1f}s)", flush=True)

    return model, history


# ===== SCORING =====

@torch.no_grad()
def score_windows_last_step(model, X_norm_windows, X_count_windows, device=DEVICE,
                            n_samples=4, batch=512, tail_k=3, mask_tail=True):
    """Per-window anomaly score from VAE.

    Aggregation: average NB-NLL of the LAST `tail_k` days under the decoded
    distribution. To make the score genuinely predictive (not auto-encodes the
    current day's outbreak signal), we MASK the last `tail_k` days of the
    encoder input so the encoder only sees historical context.

    Use posterior mean of z + a few MC samples averaged.
    Returns array [N], higher = more anomalous.
    """
    model.eval()
    if len(X_norm_windows) == 0:
        return np.array([], dtype=np.float32)
    Xn = torch.from_numpy(X_norm_windows).unsqueeze(-1).clone()
    Xc = torch.from_numpy(X_count_windows)
    if mask_tail and tail_k > 0:
        # Replace the last tail_k positions with the value at position -(tail_k+1)
        # (a benign "carry-forward" mask so model sees plausible recent context).
        # If tail_k == win, fall back to zero-fill.
        if tail_k < Xn.shape[1]:
            ref = Xn[:, -(tail_k + 1):-tail_k, :]                      # [N, 1, 1]
            Xn[:, -tail_k:, :] = ref.expand(-1, tail_k, -1)
        else:
            Xn[:, -tail_k:, :] = 0.0
    out = np.empty(len(Xn), dtype=np.float32)
    for i in range(0, len(Xn), batch):
        xn = Xn[i:i + batch].to(device)
        xc = Xc[i:i + batch].to(device)
        mu_z, logvar_z = model.encode(xn)
        nlls = []
        # one pass with mean z (no noise)
        mu_y, alpha_y = model.decode(mu_z)
        nll_t = nb_nll(xc[:, -tail_k:], mu_y[:, -tail_k:], alpha_y[:, -tail_k:]).mean(dim=-1)
        nlls.append(nll_t.unsqueeze(0))
        # plus MC samples for posterior expectation
        for _ in range(n_samples):
            z = model.reparam(mu_z, logvar_z)
            mu_y, alpha_y = model.decode(z)
            nll_t = nb_nll(xc[:, -tail_k:], mu_y[:, -tail_k:], alpha_y[:, -tail_k:]).mean(dim=-1)
            nlls.append(nll_t.unsqueeze(0))
        nll_avg = torch.cat(nlls, dim=0).mean(dim=0)
        out[i:i + batch] = nll_avg.detach().cpu().numpy()
    return out


# ===== PER-SIGNAL =====

def evaluate_signal(S, log_path):
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
        print(f"[sig {S}] no complete sims; skip.", flush=True)
        return None

    train_sims, val_sims, test_sims = split_60_20_20(sims, rng)
    print(f"[sig {S}] splits: {len(train_sims)} train, {len(val_sims)} val, {len(test_sims)} test",
          flush=True)

    # ---- Build training windows from training-period only ----
    Xtr_count = stack_train_windows(train_sims, TRAIN_DAYS, stride=TRAIN_STRIDE)
    if len(Xtr_count) == 0:
        print(f"[sig {S}] no training windows; skip.", flush=True)
        return None

    # Normalisation: log1p + z-score using training statistics
    Xtr_log = np.log1p(Xtr_count.astype(np.float32))
    mu_log = float(Xtr_log.mean()); sd_log = float(Xtr_log.std()) or 1.0
    Xtr_norm = ((Xtr_log - mu_log) / sd_log).astype(np.float32)
    print(f"[sig {S}] train windows: {Xtr_count.shape}; log1p mu={mu_log:.3f} sd={sd_log:.3f}",
          flush=True)

    # ---- Train VAE ----
    model = VAEDetector().to(DEVICE)
    t0 = time.time()
    try:
        model, hist = train_vae(model, Xtr_norm, Xtr_count, epochs=EPOCHS,
                                batch=BATCH, lr=LR, beta=BETA, device=DEVICE)
    except RuntimeError as e:
        print(f"[sig {S}] WARNING: training raised on {DEVICE}: {e}; retrying on CPU", flush=True)
        model = VAEDetector().to(torch.device("cpu"))
        model, hist = train_vae(model, Xtr_norm, Xtr_count, epochs=EPOCHS,
                                batch=BATCH, lr=LR, beta=BETA, device=torch.device("cpu"))
    print(f"[sig {S}] training done in {(time.time()-t0)/60:.1f} min", flush=True)

    use_device = next(model.parameters()).device

    def score_block(sim_list, abs_start, abs_end):
        """Build per-day windows ending at each day in [abs_start, abs_end);
        return array [days, n_sims] of NLL (higher = anomalous)."""
        cols = []
        for d in sim_list:
            wcount = windows_for_range(d['x'], abs_start, abs_end)  # [days, win]
            wnorm  = ((np.log1p(wcount) - mu_log) / sd_log).astype(np.float32)
            scores = score_windows_last_step(model, wnorm, wcount.astype(np.float32),
                                             device=use_device)
            cols.append(scores)
        return np.column_stack(cols) if cols else np.zeros((abs_end - abs_start, 0), dtype=np.float32)

    # ---- VAL: last VALID_DAYS of each sim ----
    val_lengths = []
    val_blocks = []
    for d in val_sims:
        a = len(d['x']) - VALID_DAYS
        b = len(d['x'])
        val_lengths.append(b - a)
    # Score val
    val_cols = []
    for d in val_sims:
        a = len(d['x']) - VALID_DAYS
        b = len(d['x'])
        wcount = windows_for_range(d['x'], a, b)
        wnorm  = ((np.log1p(wcount) - mu_log) / sd_log).astype(np.float32)
        scores = score_windows_last_step(model, wnorm, wcount.astype(np.float32),
                                         device=use_device)
        val_cols.append(scores)
    val_block = np.column_stack(val_cols) if val_cols else np.zeros((VALID_DAYS, 0), dtype=np.float32)

    # ---- TEST: ABS window [ABS_START, ABS_END) ----
    test_block = score_block(test_sims, ABS_START, ABS_END)
    test_lengths = [WIN_LEN] * len(test_sims)

    # ---- Sanitize NaN/Inf ----
    val_block = np.nan_to_num(val_block, nan=0.0, posinf=1e6, neginf=-1e6)
    test_block = np.nan_to_num(test_block, nan=0.0, posinf=1e6, neginf=-1e6)

    # ---- Tune contamination on val (decision = NEGATED NLL: higher = more normal) ----
    val_decision = -val_block.flatten(order="F")
    c = tune_contamination_threshold(val_sims, val_lengths, val_decision,
                                     spec_target=SPEC_TARGET,
                                     w_sens=W_SENS, w_spec=W_SPEC)
    # Compute val sens/spec
    O_full_val = np.stack([d['y'] for d in val_sims], axis=1)
    thr_val_dec = np.percentile(val_decision, c * 100)
    yhat_v = (val_decision <= thr_val_dec).astype(int)
    A_list, ofs = [], 0
    for L in val_lengths:
        A_list.append(yhat_v[ofs:ofs + L]); ofs += L
    A_v = np.column_stack(A_list)
    sens_v = compute_sensitivity_R(A_v, O_full_val)
    spec_v = compute_specificity_R(A_v, O_full_val, IDX_RANGE)

    # ---- Test alarms via validation-selected numeric decision threshold ----
    test_decision = -test_block.flatten(order="F")
    thr_test_dec = thr_val_dec
    yhat_t = (test_decision <= thr_test_dec).astype(int)
    A_tlist, ofs = [], 0
    for L in test_lengths:
        A_tlist.append(yhat_t[ofs:ofs + L]); ofs += L
    A_t = np.column_stack(A_tlist)

    O_full_test = np.stack([d['y'] for d in test_sims], axis=1)
    metrics = dict(
        signal=S,
        sensitivity=compute_sensitivity_R(A_t, O_full_test),
        specificity=compute_specificity_R(A_t, O_full_test, IDX_RANGE),
        fpr=compute_fpr_R(A_t, O_full_test, IDX_RANGE),
        pod=compute_pod_R(A_t, O_full_test),
        timeliness=compute_timeliness_R(A_t, O_full_test),
        contamination=c,
    )
    print(f"[sig {S}] c={c:.3f}  val sens={sens_v:.3f} spec={spec_v:.3f}  "
          f"TEST sens={metrics['sensitivity']:.3f} spec={metrics['specificity']:.3f} "
          f"pod={metrics['pod']:.3f} tim={metrics['timeliness']:.3f} fpr={metrics['fpr']:.3f}",
          flush=True)

    # ---- Save score caches: convention higher = more normal -> save NEGATED NLL ----
    os.makedirs(OUT_DIR, exist_ok=True)
    val_csv  = os.path.join(OUT_DIR, f"vae_val_scores_signal_{S}.csv")
    test_csv = os.path.join(OUT_DIR, f"vae_test_scores_signal_{S}.csv")
    pd.DataFrame(-val_block,  columns=[f"sim_{i}" for i in range(val_block.shape[1])]).to_csv(val_csv,  index=False)
    pd.DataFrame(-test_block, columns=[f"sim_{i}" for i in range(test_block.shape[1])]).to_csv(test_csv, index=False)
    print(f"[sig {S}] saved {val_csv} {val_block.shape}, {test_csv} {test_block.shape}", flush=True)

    with open(log_path, "a") as f:
        f.write(f"sig {S} c={c:.3f} sens={metrics['sensitivity']:.3f} "
                f"tim={metrics['timeliness']:.3f}\n")
    return metrics


# ===== MAIN =====

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--signals", type=str, default=None)
    ap.add_argument("--log", type=str, default="vae_count_cache.log")
    args = ap.parse_args()

    target = SIGNALS if args.signals is None else [int(s) for s in args.signals.split(",")]
    print(f"VAE-NB count detector: signals {target}", flush=True)
    open(args.log, "w").write(f"start: {target}\n")

    rows = []
    for S in target:
        # resumable: skip signals already cached (so a Colab disconnect doesn't waste work)
        vcsv = os.path.join(OUT_DIR, f"vae_val_scores_signal_{S}.csv")
        tcsv = os.path.join(OUT_DIR, f"vae_test_scores_signal_{S}.csv")
        if os.path.exists(vcsv) and os.path.exists(tcsv):
            print(f"[sig {S}] already cached in {OUT_DIR} -> skip", flush=True)
            continue
        try:
            r = evaluate_signal(S, args.log)
            if r:
                rows.append(r)
        except Exception as e:
            import traceback
            print(f"[sig {S}] FAILED: {e}", flush=True)
            traceback.print_exc()

    if rows:
        df = pd.DataFrame(rows)
        print("\n=== VAE-NB ALONE (per signal) ===")
        print(df)
        means = df.mean(numeric_only=True)
        print("\nMeans:", means.to_dict())
        os.makedirs("results", exist_ok=True)
        # Derive magnitude tag from the data dir so a single-magnitude run can't
        # clobber another magnitude's file (e.g. big_signal_datasets_small -> small).
        _dd = os.environ.get("SYND_DATA_DIR", "big_signal_datasets_medium")
        _mag = _dd.split("_")[-1] if _dd.split("_")[-1] in ("small", "medium", "large") else "medium"
        _outp = f"results/VAE_per_sig_big_{_mag}.csv"
        df.to_csv(_outp, index=False)
        print(f"Saved {_outp}")
    print("ALL DONE.", flush=True)
