#!/usr/bin/env python3
"""
Figures for the unsupervised-methods chapter, built on Tufte's principles
(The Visual Display of Quantitative Information, 2nd ed., ch. 4-8):

  - Maximise the data-ink ratio; erase non-data-ink (no boxes, no gridlines,
    no redundant ticks).
  - Range-frame: axis spines span only the observed data range (ch. 6).
  - Dot-dash / rug: the frame carries the marginal distribution where useful.
  - Direct labelling instead of legends.
  - Slopegraph for the magnitude-scaling comparison (Tufte's signature form).
  - Multifunctioning elements: every mark earns its ink.

Outputs (figs/tufte_*.pdf and .png):
  1. tufte_slopegraph_magnitude   -- sensitivity scaling small->medium->large
  2. tufte_persignal_dotplot      -- per-signal sensitivity, all methods, sorted
  3. tufte_duration               -- detection rate vs outbreak duration + rug
  4. tufte_headline_ci            -- ensemble vs single, bootstrap 95% CIs
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

FIGS = "figs"
os.makedirs(FIGS, exist_ok=True)

# ---- Tufte-ish global style: serif type, hairline ink, no top/right spines ----
plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.edgecolor": "#222222",
    "axes.linewidth": 0.6,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.color": "#222222",
    "ytick.color": "#222222",
    "text.color": "#222222",
    "axes.labelcolor": "#222222",
    "figure.dpi": 150,
})
INK = "#222222"
FAINT = "#999999"


def range_frame(ax, xdata=None, ydata=None):
    """Trim spines to the data range (Tufte range-frame)."""
    if xdata is not None:
        ax.spines["bottom"].set_bounds(min(xdata), max(xdata))
    if ydata is not None:
        ax.spines["left"].set_bounds(min(ydata), max(ydata))


# =====================================================================
# FIGURE 1 — SLOPEGRAPH: sensitivity scaling across outbreak magnitudes
# =====================================================================
def _slopegraph(data, ylim, title, outname, value_fmt="{:.2f}"):
    cols = [0, 1, 2]
    labels = ["Small", "Medium", "Large"]
    fig, ax = plt.subplots(figsize=(7.5, 8.4))
    for name, ys in data.items():
        ax.plot(cols, ys, "-", color=INK, lw=0.8, alpha=0.85)
        ax.plot(cols, ys, "o", color=INK, ms=2.5)
    def place_labels(side_x, idx, ha, name_first, anchor_x):
        # bidirectional collision fix: nudge up, then if top is exceeded, slide cluster down
        items = sorted(data.items(), key=lambda kv: kv[1][idx])
        min_gap = (ylim[1] - ylim[0]) * 0.026
        top_pad = (ylim[1] - ylim[0]) * 0.01
        positions = []
        last_y = -1e9
        for name, ys in items:
            yy = max(ys[idx], last_y + min_gap)
            last_y = yy
            positions.append([name, ys[idx], yy])
        overflow = positions[-1][2] - (ylim[1] - top_pad)
        if overflow > 0:
            for i in range(len(positions) - 1, -1, -1):
                positions[i][2] -= overflow
                if i > 0 and positions[i][2] - positions[i-1][2] >= min_gap:
                    break
        for name, true_y, yy in positions:
            v = value_fmt.format(true_y)
            txt = f"{name}  {v}" if name_first else f"{v}  {name}"
            ax.text(side_x, yy, txt, ha=ha, va="center", fontsize=7.5, color=INK)
            # Leader line from label to true dot position; always draw so the
            # eye can follow even tiny displacements
            leader_start_x = side_x + (0.06 if ha == "right" else -0.06)
            ax.plot([leader_start_x, anchor_x], [yy, true_y],
                    color=INK, lw=0.5, alpha=0.55, solid_capstyle="round")
    place_labels(-0.25, 0, "right", name_first=True,  anchor_x=-0.04)
    place_labels(2.25,  2, "left",  name_first=False, anchor_x=2.04)
    ax.set_xlim(-1.55, 3.55)
    ax.set_ylim(*ylim)
    for x, lab in zip(cols, labels):
        ax.text(x, ylim[1] + (ylim[1]-ylim[0])*0.04, lab, ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color=INK)
    ax.axis("off")
    fig.suptitle(title, fontsize=10, y=0.985)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/{outname}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  {outname}")


def fig_slopegraph():
    # Cross-magnitude mean sensitivities (per-signal means; matches Table 2)
    data = {
        "R-KNN":      [0.720, 0.812, 0.871],
        "R-IF":       [0.650, 0.787, 0.869],
        "KNN":        [0.615, 0.750, 0.854],
        "R-OCSVM":    [0.612, 0.754, 0.847],
        "R-LOF":      [0.626, 0.768, 0.847],
        "IF":         [0.525, 0.682, 0.828],
        "LOF":        [0.561, 0.703, 0.822],
        "LSTM-AE":    [0.570, 0.711, 0.816],
        "OCSVM":      [0.464, 0.617, 0.787],
        "OR-ens (LSTM-AE+VAE)": [0.569, 0.687, 0.773],
        "Farrington": [0.440, 0.556, 0.639],
        "VAE":        [0.424, 0.516, 0.611],
        "R-CUSUM":     [0.404, 0.460, 0.393],
        "CUSUM":       [0.243, 0.307, 0.315],
        "R-RateChange":[0.234, 0.272, 0.294],
        "RateChange":  [0.188, 0.218, 0.249],
        "R-BOCPD":     [0.175, 0.212, 0.291],
        "BOCPD":       [0.133, 0.154, 0.202],
    }
    _slopegraph(
        data, ylim=(0.10, 0.90),
        title="Sensitivity across outbreak magnitudes",
        outname="tufte_slopegraph_magnitude")


def fig_slopegraph_psd():
    # Cross-magnitude pooled PSD-within-5d (matches Tables 11, 11M, 11L)
    data = {
        "R-RateChange":[0.91, 0.97,  0.99],
        "OR-ens (LSTM-AE+VAE)": [0.84, 0.93, 0.989],
        "R-KNN":      [0.84,  0.94,  0.984],
        "R-IF":       [0.84,  0.93,  0.981],
        "R-BOCPD":    [0.83,  0.92,  0.98],
        "RateChange": [0.80,  0.89,  0.96],
        "KNN":        [0.78,  0.87,  0.97],
        "R-OCSVM":    [0.78,  0.90,  0.979],
        "IF":         [0.78,  0.90,  0.985],
        "VAE":        [0.77,  0.87,  0.94],
        "BOCPD":      [0.76,  0.86,  0.95],
        "R-LOF":      [0.72,  0.87,  0.978],
        "OCSVM":      [0.68,  0.82,  0.98],
        "LSTM-AE":    [0.65,  0.77,  0.92],
        "LOF":        [0.64,  0.84,  0.95],
        "Farrington": [0.62,  0.78,  0.94],
        "R-CUSUM":    [0.48,  0.51,  0.30],
        "CUSUM":      [0.28,  0.30,  0.25],
    }
    _slopegraph(
        data, ylim=(0.20, 1.02),
        title="PSD$_5$ across outbreak magnitudes",
        outname="tufte_slopegraph_psd_magnitude")


# =====================================================================
# FIGURE 2 — PER-SIGNAL DOT PLOT (corrected small-magnitude data)
# =====================================================================
def fig_persignal_dotplot():
    import sys; sys.path.insert(0, ".")
    os.environ["SYND_DATA_DIR"] = "big_signal_datasets_small"
    from anom_common import (load_data, split_60_20_20, compute_sensitivity_R,
                             TRAIN_DAYS, VALID_DAYS, RNG_STATE, SIGNALS)
    NAMES = {1:"Diarrhoea",2:"Arthropod bites",3:"Cardiac",4:"ICU cardiac",5:"Allergic rhinitis",
             6:"Heat stroke",7:"Herpes zoster",8:"Insect bite",9:"Pertussis",10:"Pneumonia",
             11:"Rubella",12:"Upper resp.",13:"Bronchitis",14:"Hepatitis",15:"ILI",16:"UTI"}
    np.random.seed(RNG_STATE); rng = np.random.RandomState(RNG_STATE)
    farr = {}
    for S in SIGNALS:
        Xs, Ys = load_data(S)
        sims = [dict(x=Xs[c].to_numpy(np.float32), y=Ys[c].to_numpy(np.int32), sim_idx=i)
                for i, c in enumerate(Xs.columns) if len(Xs[c]) >= TRAIN_DAYS + VALID_DAYS]
        _, _, te = split_60_20_20(sims, rng)
        O = np.stack([d["y"] for d in te], axis=1)
        A = pd.read_csv(f"farrington_custom_alarms_signal_{S}.csv").to_numpy(dtype=int)
        if A.shape[0] != 343: A = A[-343:]
        farr[S] = compute_sensitivity_R(A, O)

    def lp(f):
        if not os.path.exists(f): return {}
        d = pd.read_csv(f); d.columns = [c.strip() for c in d.columns]
        if "signal" not in d.columns: d = d.rename(columns={d.columns[0]: "signal"})
        d["signal"] = d["signal"].astype(int); return dict(zip(d["signal"], d["sensitivity"]))
    cols = {"Farrington": farr,
            "IF": lp("results/IsolationForest_Tuned_per_sig_big_small.csv"),
            "KNN": lp("results/KNN_Tuned_per_sig_big_small.csv"),
            "OCSVM": lp("results/OCSVM_Tuned_per_sig_big_small.csv"),
            "LOF": lp("results/LOF_Tuned_per_sig_big_small.csv"),
            "LSTM-AE": lp("results/LSTM_AE_per_sig_big_small.csv"),
            "NB-HMM": lp("results/NBHMM_per_sig_big_small.csv"),
            "VAE": lp("results/VAE_per_sig_big_small.csv"),
            "CUSUM": lp("results/CUSUM_per_sig_big_small.csv")}
    methods = list(cols.keys())
    mean9 = {S: np.nanmean([cols[m].get(S, np.nan) for m in methods]) for S in SIGNALS}
    order = sorted(SIGNALS, key=lambda s: mean9[s])  # hardest at bottom

    all_vals = [cols[m].get(S, np.nan) for m in methods for S in SIGNALS]
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    fig, ax = plt.subplots(figsize=(7.0, 8.0))
    for yi, S in enumerate(order):
        vals = {m: cols[m].get(S, np.nan) for m in methods}
        present = {m: v for m, v in vals.items() if not np.isnan(v)}
        if not present:
            continue
        bestm = max(present, key=present.get)
        # faint connecting line across the range of method values (multifunctioning: shows spread)
        ax.plot([min(present.values()), max(present.values())], [yi, yi],
                "-", color="#dddddd", lw=0.8, zorder=1)
        for m, v in present.items():
            if m == bestm:
                ax.plot(v, yi, "o", color=INK, ms=5, zorder=3)
                ax.text(v + 0.012, yi, bestm, va="center", ha="left",
                        fontsize=6.8, color=INK, zorder=4)
            else:
                ax.plot(v, yi, "o", color=FAINT, ms=2.6, mfc="white",
                        mec=FAINT, mew=0.7, zorder=2)
        # signal name at left
        ax.text(vmin - 0.02, yi, NAMES[S], va="center", ha="right", fontsize=8, color=INK)
        # mean tick (small vertical dash = the cross-method mean, a data measure)
        ax.plot([mean9[S], mean9[S]], [yi - 0.22, yi + 0.22], "-", color=INK, lw=0.9, zorder=3)

    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(vmin - 0.30, vmax + 0.13)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    xticks = [0.0, 0.2, 0.4, 0.6, 0.8]
    xticks = [t for t in xticks if vmin - 0.01 <= t <= vmax + 0.01]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.1f}" for t in xticks])
    range_frame(ax, xdata=[vmin, vmax])
    ax.set_xlabel("Sensitivity at small outbreak magnitude")
    ax.set_title("Per-signal sensitivity, sorted by difficulty\n"
                 "(filled dot = best detector for that signal; "
                 "open dots = the other eight; vertical dash = cross-method mean)",
                 fontsize=10, pad=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/tufte_persignal_dotplot.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  tufte_persignal_dotplot")


# =====================================================================
# FIGURE 3 — DETECTION RATE vs OUTBREAK DURATION (range-frame + rug)
# =====================================================================
def fig_duration():
    df = pd.read_csv("results/event_level_diagnostics.csv")
    df["lenbin"] = pd.cut(df["length"], bins=[0, 1, 3, 7, 14, 1000],
                          labels=["1", "2-3", "4-7", "8-14", "15+"])
    bins = ["1", "2-3", "4-7", "8-14"]
    xpos = [1, 2.5, 5.5, 11]  # approx midpoints (days) for honest horizontal spacing
    methods = ["Farrington", "KNN", "IF", "LSTM-AE", "NB-HMM", "CUSUM"]
    series = {}
    for m in methods:
        col = f"det_{m}"
        if col not in df.columns:
            continue
        series[m] = [df[df.lenbin == b][col].mean() for b in bins]

    fig, ax = plt.subplots(figsize=(6.8, 5.2))
    for m, ys in series.items():
        ax.plot(xpos, ys, "-o", color=INK, lw=0.8, ms=2.5)
        ax.text(xpos[-1] + 0.25, ys[-1], m, va="center", ha="left", fontsize=8, color=INK)
    # rug along the bottom: distribution of outbreak durations (multifunctioning frame).
    # Stacked below the axis: 'N d' tick labels, then % share, then caption, then xlabel.
    for L, xp in zip(bins, xpos):
        frac = (df.lenbin == L).mean()
        ax.text(xp, -0.105, f"{frac*100:.0f}%", ha="center", va="top",
                fontsize=7, color=FAINT)
    ax.text(xpos[0], -0.165, "share of all outbreaks of this duration",
            ha="left", va="top", fontsize=7, color=FAINT)

    ax.set_xlim(0.4, 13.6)
    ax.set_ylim(0, 1.0)
    ax.set_xticks(xpos)
    ax.set_xticklabels([f"{b} d" for b in bins])
    yt = [0, 0.25, 0.5, 0.75, 1.0]
    ax.set_yticks(yt)
    ax.set_yticklabels([f"{t:.2f}" for t in yt])
    range_frame(ax, xdata=[xpos[0], xpos[-1]], ydata=[0, 1.0])
    ax.set_xlabel("Outbreak duration", labelpad=40)
    ax.set_ylabel("Probability of detection")
    ax.set_title("Outbreak duration is the primary determinant of detectability\n"
                 "(small magnitude; each line a detector, labelled at right)",
                 fontsize=10, pad=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/tufte_duration.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  tufte_duration")


# =====================================================================
# FIGURE 4 — HEADLINE COMPARISON with bootstrap 95% CI (range-frame)
# =====================================================================
def fig_headline_ci():
    # Paired cluster-bootstrap summary values (B=10000); keep in sync with
    # results/headline_bootstrap_per_signal.csv when regenerating this figure.
    rows = [
        ("Ensemble\n(Farr+IF+NB-HMM)", 0.725, 0.647, 0.795),
        ("KNN (best single)", 0.615, 0.506, 0.722),
        ("Isolation Forest", 0.523, 0.409, 0.643),
    ]
    # per-signal strips behind each row (dot-dash flavour: show the raw 16 points)
    persig = None
    p = "results/headline_bootstrap_per_signal.csv"
    if os.path.exists(p):
        persig = pd.read_csv(p)
    keymap = {"Ensemble\n(Farr+IF+NB-HMM)": "ensemble", "KNN (best single)": "KNN",
              "Isolation Forest": "IF"}

    fig, ax = plt.subplots(figsize=(6.8, 4.0))
    ys = [i * 1.0 for i in range(len(rows))][::-1]
    # x-range driven by the raw per-signal data so ticks are shown honestly, uncropped
    pt_min, pt_max = 1.0, 0.0
    if persig is not None:
        for k in keymap.values():
            if k in persig.columns:
                pt_min = min(pt_min, float(np.nanmin(persig[k].values)))
                pt_max = max(pt_max, float(np.nanmax(persig[k].values)))
    xlo = min(pt_min, min(r[2] for r in rows)) - 0.03
    xhi = max(pt_max, max(r[3] for r in rows)) + 0.03
    for (label, pe, lo, hi), y in zip(rows, ys):
        # raw per-signal points as a faint rug just above the CI (shows the data behind the mean)
        if persig is not None and keymap[label] in persig.columns:
            pts = persig[keymap[label]].values
            ax.plot(pts, np.full_like(pts, y) + 0.16, "|", color="#cfcfcf", ms=7, mew=0.8)
        # 95% CI as a thin line; point estimate as a dot
        ax.plot([lo, hi], [y, y], "-", color=INK, lw=1.0)
        ax.plot(pe, y, "o", color=INK, ms=5)
        # method label sits ABOVE the left end of its row (no collision with ticks/labels)
        ax.text(xlo, y + 0.30, label, va="bottom", ha="left", fontsize=8.4, color=INK)
        ax.text(pe, y - 0.20, f"{pe:.3f}  [{lo:.2f}, {hi:.2f}]", va="top", ha="center",
                fontsize=7, color=FAINT)
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_xlim(xlo - 0.01, xhi + 0.01)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    xt = [t for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if xlo <= t <= xhi]
    ax.set_xticks(xt); ax.set_xticklabels([f"{t:.1f}" for t in xt])
    range_frame(ax, xdata=[xlo, xhi])
    ax.set_xlabel("Sensitivity (small magnitude) with 95% paired-bootstrap interval")
    ax.set_title("The ensemble's advantage over the best single detector is reliable\n"
                 "(dot = mean over 16 signals; line = 95% CI; faint ticks = the 16 per-signal values)",
                 fontsize=10, pad=12)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/tufte_headline_ci.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  tufte_headline_ci")


# =====================================================================
# FIGURE 5 — EARLY-DETECTION SCORECARD (small multiples; everything at once)
# =====================================================================
def fig_scorecard():
    """One aligned dot-plot panel per metric, sharing method rows.
    Presents the full early-incident-detection operating picture:
      detection breadth (sensitivity), per-event catch rate (POD),
      earliness (timeliness), and false-alarm cost (specificity).
    """
    df = pd.read_csv("results/ALL_METHODS_unified_big_small.csv")
    df = df[df["status"] == "ok"].copy()
    for c in ("sensitivity", "specificity", "pod", "timeliness", "fpr"):
        df[c] = df[c].astype(float)

    # Display name + type. Exclude supervised stackers (companion work) and the
    # all-6 OR-vote (mixes the Farrington alarms that carried the stale-data risk).
    DISP = {
        "OR-vote (if+lstm+nbhmm+noufaily)": ("OR-ensemble (4)", "ens"),
        "OR-vote (if+lstm+nbhmm)":          ("OR-ensemble (3)", "ens"),
        "KNN (per-sig tuned)":              ("KNN", "sin"),
        "LOF (per-sig tuned)":              ("LOF", "sin"),
        "LSTM-AE (production)":             ("LSTM-AE", "sin"),
        "IsolationForest (tuned)":          ("Isolation Forest", "sin"),
        "NB-HMM":                           ("NB-HMM", "sin"),
        "OCSVM (per-sig tuned)":            ("OCSVM", "sin"),
        "Farrington (custom α=0.01)":       ("Farrington", "sin"),
        "VAE (NegBin)":                     ("VAE", "sin"),
        "CUSUM (NB seasonal)":              ("CUSUM", "sin"),
        "Noufaily-quantile":                ("Noufaily", "sin"),
        "RateChange-residual":              ("RateChange", "sin"),
        "BOCPD-residual":                   ("BOCPD", "sin"),
    }
    df = df[df["method"].isin(DISP)].copy()
    df["disp"] = df["method"].map(lambda m: DISP[m][0])
    df["kind"] = df["method"].map(lambda m: DISP[m][1])
    df = df.sort_values("sensitivity", ascending=True).reset_index(drop=True)  # best at top
    ys = list(range(len(df)))

    # Earliness = 1 - timeliness  (higher = detected earlier in the outbreak window),
    # so every panel reads the same way: right = better, no reversed axis.
    df["earliness"] = 1.0 - df["timeliness"]

    # Panels: (column, title, sub-label). All normal-axis, higher = better.
    panels = [
        ("sensitivity", "Sensitivity", "outbreak days flagged"),
        ("pod",         "Per-event detection", "outbreaks caught at all"),
        ("earliness",   "Earliness", "1 − relative detection position"),
        ("specificity", "Specificity", "1 − false-alarm rate"),
    ]
    fig, axes = plt.subplots(1, 4, figsize=(11.5, 6.6), sharey=True,
                             gridspec_kw={"wspace": 0.12})

    for ax, (col, title, sub) in zip(axes, panels):
        vals = df[col].values
        vmin, vmax = float(vals.min()), float(vals.max())
        rng = (vmax - vmin) + 1e-6
        pad = rng * 0.10
        off = rng * 0.04
        for y, v, k in zip(ys, vals, df["kind"]):
            if k == "ens":
                ax.plot(v, y, "D", color=INK, ms=5.5, zorder=3)   # diamond = ensemble
            else:
                ax.plot(v, y, "o", color=INK, ms=4.0, zorder=3)   # circle = single
            # value label always to the RIGHT of the dot -> never collides with names
            ax.text(v + off, y, f"{v:.2f}", va="center", ha="left",
                    fontsize=6.3, color=FAINT, zorder=2)

        # left gutter keeps the smallest dot off the method names; right pad holds
        # the value label of the best (right-most) dot
        ax.set_xlim(vmin - rng * 0.22, vmax + rng * 0.42)

        # specificity panel: mark the 0.95 operating-point target
        if col == "specificity":
            ax.axvline(0.95, color=FAINT, lw=0.7, ls=(0, (3, 2)), zorder=1)
            ax.text(0.95, -0.7, "0.95 target", rotation=90,
                    va="bottom", ha="right", fontsize=6.5, color=FAINT)

        ax.set_ylim(-0.8, len(df) - 0.2)
        ax.spines["left"].set_visible(False)
        ax.set_yticks([])   # drop integer row-index labels (we draw method names ourselves)
        ticks = np.round(np.linspace(vmin, vmax, 3), 2)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{t:.2f}" for t in ticks], fontsize=7.5)
        range_frame(ax, xdata=[vmin, vmax])
        ax.set_title(title, fontsize=9.5, pad=16, fontweight="bold")
        ax.text(0.5, 1.005, sub, transform=ax.transAxes, ha="center", va="bottom",
                fontsize=7, color=FAINT)

    # method names + ensemble emphasis on the leftmost panel
    axL = axes[0]
    for y, name, k in zip(ys, df["disp"], df["kind"]):
        axL.text(-0.02, y, name, transform=axL.get_yaxis_transform(),
                 va="center", ha="right",
                 fontsize=8, fontweight=("bold" if k == "ens" else "normal"),
                 color=INK)

    fig.suptitle("Early-detection scorecard at the operating point (small outbreaks, specificity-matched tuning)\n"
                 "diamonds = OR-vote ensembles, circles = single detectors; rows sorted by sensitivity; in every panel right = better",
                 fontsize=10.5, y=0.99)
    fig.subplots_adjust(left=0.125, right=0.985, top=0.88, bottom=0.07)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/tufte_early_detection_scorecard.{ext}", bbox_inches="tight")
    plt.close(fig)
    print("  tufte_early_detection_scorecard")


# =====================================================================
# FIGURE 6 — TIMELY DETECTION: PSD(d) curves (detect within first d days)
# =====================================================================
def fig_timely_detection(mag="small"):
    """Probability of Successful Detection within d days of onset, PSD(d), plus the
    specificity each catch rate costs. `mag` in {'small','medium','large'} selects
    the magnitude (reads the matching results CSVs and writes a tagged figure).

    Per-sim, single-outbreak definition (one injected outbreak per simulation;
    onset = first injected day; delay = first alarm-on-an-outbreak-day - onset).
    PSD(d) rises to POD as d grows.
    """
    suf = "" if mag == "small" else f"_{mag}"
    cur = pd.read_csv(f"results/timely_detection_curve{suf}.csv")
    cur = cur[cur["day"] <= 14]
    days = sorted(cur["day"].unique())
    summ = pd.read_csv(f"results/timely_detection_summary{suf}.csv")
    POD = dict(zip(summ.method, summ.POD))
    SPEC = dict(zip(summ.method, summ.specificity))

    # ink, labelled focus set; everything else drawn faint for context
    FOCUS = ["OR-ensemble (LSTM-AE+VAE)", "R-KNN", "R-IF", "RateChange", "KNN",
             "IF", "BOCPD", "Farrington", "CUSUM"]
    SHORT = {"OR-ensemble (LSTM-AE+VAE)": "OR-ens (LSTM-AE+VAE)",
             "RateChange": "R-RateChange",
             "BOCPD":      "R-BOCPD",
             "CUSUM":      "R-CUSUM"}
    all_methods = [m for m in cur["method"].unique() if m in SPEC]

    # two panels share the probability (y) axis: left = PSD(d) curve, right = the
    # specificity (false-alarm cost) of reaching that catch rate.
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(9.8, 5.4), sharey=True,
                                   gridspec_kw=dict(width_ratios=[2.7, 1.2], wspace=0.05))

    # ---- left: PSD(d) curves ----
    for m in all_methods:
        sub = cur[cur.method == m].sort_values("day")
        axL.step(sub["day"], sub["psd_all"], where="post", color="#dddddd", lw=0.7, zorder=1)
    for m in FOCUS:
        sub = cur[cur.method == m].sort_values("day")
        if sub.empty:
            continue
        axL.step(sub["day"], sub["psd_all"], where="post", color=INK, lw=1.0, zorder=3)
        axL.plot(sub["day"], sub["psd_all"], "o", color=INK, ms=2.2, zorder=3)
    axL.axvline(5, color=FAINT, lw=0.7, ls=(0, (3, 2)), zorder=2)
    axL.text(5 - 0.15, 0.02, "first 5 days", rotation=90, va="bottom", ha="right",
             fontsize=7.5, color=FAINT)
    axL.set_xlim(days[0] - 0.1, days[-1] + 0.3)
    axL.set_ylim(0, 1.0)
    axL.set_xticks([1, 3, 5, 7, 9, 11, 13])
    axL.set_xticklabels(["1", "3", "5", "7", "9", "11", "13"], fontsize=8)
    yt = [0, 0.25, 0.5, 0.75, 1.0]
    axL.set_yticks(yt); axL.set_yticklabels([f"{t:.2f}" for t in yt])
    range_frame(axL, xdata=[days[0], days[-1]], ydata=[0, 1.0])
    axL.set_xlabel("Days since outbreak onset (day 1 = onset)")
    axL.set_ylabel("probability the outbreak is detected  (PSD$(d)$, $\\to$ POD)")

    # ---- right: specificity at the operating point, plotted at y = POD ----
    smin = min(SPEC[m] for m in all_methods); smax = max(SPEC[m] for m in all_methods)
    for m in all_methods:                                   # faint: all methods
        axR.plot(SPEC[m], POD[m], "o", color="#cccccc", ms=3.0, zorder=2)
    focus_pts = [(SHORT.get(m, m), POD[m], SPEC[m]) for m in FOCUS if m in SPEC]
    for name, pod, sp in focus_pts:
        axR.plot(sp, pod, "o", color=INK, ms=4.5, zorder=3)
    # de-collide labels, keeping them within [0,1]: nudge up from the bottom, but if
    # that overflows the top (methods bunched near POD=1, as at medium/large), nudge
    # down from the top instead. Faint leader lines tie each label to its dot.
    focus_pts.sort(key=lambda t: t[1])
    ys = [p[1] for p in focus_pts]; gap = 0.045; cap = 1.0
    up = []; last = -1.0
    for y in ys:
        yy = max(y, last + gap); last = yy; up.append(yy)
    if up[-1] > cap:
        labpos = [0.0] * len(ys); last = cap + gap
        for i in range(len(ys) - 1, -1, -1):
            last = min(ys[i], last - gap); labpos[i] = last
    else:
        labpos = up
    xlab = smax + (smax - smin) * 0.55
    for (name, pod, sp), ly in zip(focus_pts, labpos):
        axR.plot([sp, xlab - (smax - smin) * 0.05], [pod, ly], color="#cccccc", lw=0.5, zorder=1)
        axR.text(xlab, ly, name, va="center", ha="left", fontsize=8, color=INK)
    axR.set_xlim(smin - (smax - smin) * 0.15, smax + (smax - smin) * 0.45)
    axR.spines["left"].set_visible(False)
    axR.tick_params(axis="y", length=0)
    xt = [round(t, 3) for t in (0.95, 0.96, 0.97, 0.98) if smin - 1e-9 <= t <= smax + 1e-9]
    axR.set_xticks(xt); axR.set_xticklabels([f"{t:.2f}" for t in xt], fontsize=8)
    range_frame(axR, xdata=[smin, smax])
    axR.set_xlabel("Specificity")

    fig.suptitle(f"PSD($d$) at {mag} outbreak magnitude",
                 fontsize=11, y=0.99)
    fig.subplots_adjust(top=0.86, bottom=0.10, left=0.085, right=0.83)
    for ext in ("pdf", "png"):
        fig.savefig(f"{FIGS}/tufte_timely_detection{suf}.{ext}", bbox_inches="tight")
    plt.close(fig)
    print(f"  tufte_timely_detection{suf}")


if __name__ == "__main__":
    print("Building Tufte-style figures:")
    fig_slopegraph()
    fig_slopegraph_psd()
    fig_persignal_dotplot()
    fig_headline_ci()
    fig_scorecard()
    fig_timely_detection("small")
    for _mag in ("medium", "large"):
        if os.path.exists(f"results/timely_detection_curve_{_mag}.csv"):
            fig_timely_detection(_mag)
    # fig_duration() retired: with the correct single-outbreak definition, outbreak
    # duration is ~uniform (median 16 d; 96% are >= 8 d), so the duration-vs-detection
    # premise ("40% are 1-day") was a fragmentation artifact.
    print("Done -> figs/tufte_*.{pdf,png}")
