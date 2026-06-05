#!/usr/bin/env python3
"""
Vivid, projection-friendly variants of the Tufte figures, for the SLIDES only.
Design brief (chosen by the author): "vivid on cream" + "one bold accent per
figure" — keep the cream paper and serif, render context series in muted grey,
and spotlight each figure's focal series in one vivid crimson, with thicker
lines and larger type so it reads from the back of a room.

The formal paper keeps its austere figs/tufte_*.{pdf,png}; these write to
figs/slide_*.{png,pdf} and are referenced only by the deck.

Run from repo root:  /Users/u5585063/miniconda3/bin/python slides/make_slide_figures.py
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

FIGS = "figs"
os.makedirs(FIGS, exist_ok=True)

PAPER   = "#FCFCFB"   # near-white stock — figure facecolor, matches the slides
INK     = "#1E1E1E"
GREY    = "#BFBDB8"   # muted context grey
GREYLN  = "#D2D0CB"   # context connecting lines
CRIMSON = "#CC2936"   # the one vivid accent — the focal series of each figure
TEAL    = "#117A78"   # tabular family
AMBER   = "#B8740F"   # deep-learning family
SLATE   = "#3D5A80"   # statistical family
FAMCOL  = {"ensemble": CRIMSON, "tabular": TEAL, "deep": AMBER, "statistical": SLATE}

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 12,
    "axes.edgecolor": INK,
    "axes.linewidth": 0.8,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "xtick.color": INK,
    "ytick.color": INK,
    "text.color": INK,
    "axes.labelcolor": INK,
    "figure.dpi": 150,
    "savefig.facecolor": PAPER,
    "figure.facecolor": PAPER,
    "axes.facecolor": PAPER,
})


def range_frame(ax, xdata=None, ydata=None):
    if xdata is not None:
        ax.spines["bottom"].set_bounds(min(xdata), max(xdata))
    if ydata is not None:
        ax.spines["left"].set_bounds(min(ydata), max(ydata))


def save(fig, name):
    for ext in ("png", "pdf"):
        fig.savefig(f"{FIGS}/{name}.{ext}", bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    print("  ", name)


# =====================================================================
# 1 — SLOPEGRAPH: sensitivity scaling, KNN spotlighted
# =====================================================================
def slopegraph():
    data = {
        "KNN":              [0.615, 0.750, 0.854],
        "Isolation Forest": [0.523, 0.678, 0.827],
        "LSTM-AE":          [0.570, 0.711, 0.816],
        "OCSVM":            [0.464, 0.617, 0.787],
        "NB-HMM":           [0.522, 0.633, 0.731],
        "Farrington":       [0.440, 0.556, 0.639],
        "VAE":              [0.424, 0.516, 0.611],
        "Noufaily":         [0.382, 0.457, 0.529],
        "CUSUM":            [0.404, 0.459, 0.393],
        "RateChange":       [0.234, 0.272, 0.294],
    }
    HERO = "KNN"
    cols = [0, 1, 2]
    labels = ["small", "medium", "large"]

    fig, ax = plt.subplots(figsize=(7.2, 5.6))
    # context lines first — labelled only at the right end (left ends crowd at low values)
    right_labels = []   # (yval, text, color, fontsize, weight)
    for name, ys in data.items():
        if name == HERO:
            continue
        ax.plot(cols, ys, "-", color=GREY, lw=1.1, zorder=2)
        ax.plot(cols, ys, "o", color=GREY, ms=3, zorder=2)
        right_labels.append((ys[2], f"{name} {ys[2]:.2f}", GREY, 8.5, "normal"))
    # hero on top, bold crimson
    ys = data[HERO]
    ax.plot(cols, ys, "-", color=CRIMSON, lw=3.0, zorder=4)
    ax.plot(cols, ys, "o", color=CRIMSON, ms=7, zorder=5)
    ax.text(-0.05, ys[0], f"{ys[0]:.2f} {HERO}", ha="right", va="center",
            fontsize=11, color=CRIMSON, fontweight="bold")
    right_labels.append((ys[2], f"{HERO} {ys[2]:.2f}", CRIMSON, 11, "bold"))
    # de-collide the right-end labels top-to-bottom (the strong-scaling cluster crowds)
    right_labels.sort(key=lambda r: -r[0])
    placed = []
    MINGAP = 0.032
    for yval, txt, color, fs, weight in right_labels:
        yy = yval
        for pp in placed:
            if pp - yy < MINGAP:
                yy = pp - MINGAP
        placed.append(yy)
        ax.text(2.05, yy, txt, ha="left", va="center", fontsize=fs, color=color, fontweight=weight)

    ax.set_xlim(-1.25, 3.2)
    ax.set_ylim(0.18, 0.90)
    for x, lab in zip(cols, labels):
        ax.text(x, 0.895, lab, ha="center", va="bottom", fontsize=11,
                fontweight="bold", color=INK)
    ax.axis("off")
    save(fig, "slide_slopegraph")


# =====================================================================
# 2 — PER-SIGNAL DOT PLOT: best-per-signal dots spotlighted in crimson
# =====================================================================
def persignal():
    import sys
    sys.path.insert(0, ".")
    os.environ["SYND_DATA_DIR"] = "big_signal_datasets_small"
    from anom_common import (load_data, split_60_20_20, compute_sensitivity_R,
                             TRAIN_DAYS, VALID_DAYS, RNG_STATE, SIGNALS)
    NAMES = {1: "Diarrhoea", 2: "Arthropod bites", 3: "Cardiac", 4: "ICU cardiac",
             5: "Allergic rhinitis", 6: "Heat stroke", 7: "Herpes zoster", 8: "Insect bite",
             9: "Pertussis", 10: "Pneumonia", 11: "Rubella", 12: "Upper resp.",
             13: "Bronchitis", 14: "Hepatitis", 15: "ILI", 16: "UTI"}
    np.random.seed(RNG_STATE)
    rng = np.random.RandomState(RNG_STATE)
    farr = {}
    for S in SIGNALS:
        Xs, Ys = load_data(S)
        sims = [dict(x=Xs[c].to_numpy(np.float32), y=Ys[c].to_numpy(np.int32), sim_idx=i)
                for i, c in enumerate(Xs.columns) if len(Xs[c]) >= TRAIN_DAYS + VALID_DAYS]
        _, _, te = split_60_20_20(sims, rng)
        O = np.stack([d["y"] for d in te], axis=1)
        A = pd.read_csv(f"farrington_custom_alarms_signal_{S}.csv").to_numpy(dtype=int)
        if A.shape[0] != 343:
            A = A[-343:]
        farr[S] = compute_sensitivity_R(A, O)

    def lp(f):
        if not os.path.exists(f):
            return {}
        d = pd.read_csv(f)
        d.columns = [c.strip() for c in d.columns]
        if "signal" not in d.columns:
            d = d.rename(columns={d.columns[0]: "signal"})
        d["signal"] = d["signal"].astype(int)
        return dict(zip(d["signal"], d["sensitivity"]))

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
    order = sorted(SIGNALS, key=lambda s: mean9[s])
    all_vals = [cols[m].get(S, np.nan) for m in methods for S in SIGNALS]
    vmin, vmax = np.nanmin(all_vals), np.nanmax(all_vals)

    fig, ax = plt.subplots(figsize=(7.4, 6.6))
    for yi, S in enumerate(order):
        vals = {m: cols[m].get(S, np.nan) for m in methods}
        present = {m: v for m, v in vals.items() if not np.isnan(v)}
        if not present:
            continue
        bestm = max(present, key=present.get)
        ax.plot([min(present.values()), max(present.values())], [yi, yi],
                "-", color=GREYLN, lw=1.0, zorder=1)
        for m, v in present.items():
            if m == bestm:
                ax.plot(v, yi, "o", color=CRIMSON, ms=7.5, zorder=4)
                ax.text(v + 0.014, yi, bestm, va="center", ha="left",
                        fontsize=8.2, color=CRIMSON, fontweight="bold", zorder=5)
            else:
                ax.plot(v, yi, "o", color=GREY, ms=3.0, mfc=PAPER,
                        mec=GREY, mew=1.0, zorder=2)
        ax.text(vmin - 0.02, yi, NAMES[S], va="center", ha="right", fontsize=9.5, color=INK)
        ax.plot([mean9[S], mean9[S]], [yi - 0.24, yi + 0.24], "-", color=INK, lw=1.1, zorder=3)

    ax.set_ylim(-0.8, len(order) - 0.2)
    ax.set_xlim(vmin - 0.32, vmax + 0.14)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    xticks = [t for t in [0.0, 0.2, 0.4, 0.6, 0.8] if vmin - 0.01 <= t <= vmax + 0.01]
    ax.set_xticks(xticks)
    ax.set_xticklabels([f"{t:.1f}" for t in xticks])
    range_frame(ax, xdata=[vmin, vmax])
    ax.set_xlabel("sensitivity at small outbreak magnitude", fontsize=11)
    save(fig, "slide_persignal")


# =====================================================================
# 3 — HEADLINE CI: the ensemble row spotlighted in crimson
# =====================================================================
def headline_ci():
    rows = [
        ("Ensemble (Farr+IF+NB-HMM)", 0.725, 0.647, 0.795, True),
        ("KNN (best single)", 0.615, 0.506, 0.722, False),
        ("Isolation Forest", 0.523, 0.409, 0.643, False),
    ]
    persig = None
    p = "results/headline_bootstrap_per_signal.csv"
    if os.path.exists(p):
        persig = pd.read_csv(p)
    keymap = {"Ensemble (Farr+IF+NB-HMM)": "ensemble", "KNN (best single)": "KNN",
              "Isolation Forest": "IF"}

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    ys = [i * 1.0 for i in range(len(rows))][::-1]
    pt_min, pt_max = 1.0, 0.0
    if persig is not None:
        for k in keymap.values():
            if k in persig.columns:
                pt_min = min(pt_min, float(np.nanmin(persig[k].values)))
                pt_max = max(pt_max, float(np.nanmax(persig[k].values)))
    xlo = min(pt_min, min(r[2] for r in rows)) - 0.03
    xhi = max(pt_max, max(r[3] for r in rows)) + 0.03
    for (label, pe, lo, hi, hero), y in zip(rows, ys):
        col = CRIMSON if hero else GREY
        if persig is not None and keymap[label] in persig.columns:
            pts = persig[keymap[label]].values
            ax.plot(pts, np.full_like(pts, y) + 0.16, "|",
                    color=(CRIMSON if hero else "#d8d2c7"), ms=8, mew=1.0,
                    alpha=0.5 if hero else 1.0)
        ax.plot([lo, hi], [y, y], "-", color=col, lw=2.6 if hero else 1.2, zorder=3)
        ax.plot(pe, y, "o", color=col, ms=8 if hero else 5, zorder=4)
        ax.text(xlo, y + 0.30, label, va="bottom", ha="left",
                fontsize=10 if hero else 9, color=col, fontweight="bold" if hero else "normal")
        ax.text(pe, y - 0.22, f"{pe:.3f}  [{lo:.2f}, {hi:.2f}]", va="top", ha="center",
                fontsize=8.5, color=col if hero else GREY)
    ax.set_ylim(-0.6, len(rows) - 0.05)
    ax.set_xlim(xlo - 0.01, xhi + 0.01)
    ax.set_yticks([])
    ax.spines["left"].set_visible(False)
    xt = [t for t in [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9] if xlo <= t <= xhi]
    ax.set_xticks(xt)
    ax.set_xticklabels([f"{t:.1f}" for t in xt])
    range_frame(ax, xdata=[xlo, xhi])
    ax.set_xlabel("sensitivity (small magnitude) with 95% paired-bootstrap interval", fontsize=10.5)
    save(fig, "slide_headline_ci")


# =====================================================================
# 4 — PSD at small magnitude: OR-ensemble spotlighted, with fill
# =====================================================================
def psd_small():
    df = pd.read_csv("results/timely_detection_curve.csv")
    df = df[df["method"] != "method"]
    df["day"] = df["day"].astype(int)
    df["psd_all"] = df["psd_all"].astype(float)
    dmax = df["day"].max()
    HERO = "OR-ensemble (IF+LSTM-AE+NB-HMM)"
    context = ["RateChange", "KNN", "IF", "BOCPD", "NB-HMM", "Farrington", "LSTM-AE", "LOF", "OCSVM", "VAE", "CUSUM"]
    labelers = {"RateChange": "RateChange", "KNN": "KNN", "Farrington": "Farrington", "CUSUM": "CUSUM"}

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    label_pts = []   # (yend, text, color, fontsize, weight)
    for name in context:
        sub = df[df["method"] == name].sort_values("day")
        if sub.empty:
            continue
        ax.plot(sub["day"], sub["psd_all"], "-", color=GREY, lw=1.2, zorder=2)
        if name in labelers:
            label_pts.append((sub["psd_all"].values[-1], labelers[name], "#8a857b", 9, "normal"))
    # hero
    sub = df[df["method"] == HERO].sort_values("day")
    x, y = sub["day"].values, sub["psd_all"].values
    ax.fill_between(x, 0, y, color=CRIMSON, alpha=0.08, zorder=1)
    ax.plot(x, y, "-", color=CRIMSON, lw=3.2, zorder=5)
    ax.plot(x, y, "o", color=CRIMSON, ms=4, zorder=6)
    label_pts.append((y[-1], "OR-ensemble", CRIMSON, 10.5, "bold"))
    # de-collide the right-end labels (top to bottom)
    label_pts.sort(key=lambda r: -r[0])
    placed = []
    for yend, txt, color, fs, weight in label_pts:
        yy = yend
        for pp in placed:
            if pp - yy < 0.05:
                yy = pp - 0.05
        placed.append(yy)
        ax.text(dmax + 0.3, yy, txt, va="center", ha="left",
                fontsize=fs, color=color, fontweight=weight)

    ax.axvline(5, color="#bdb7ac", lw=0.7, ls=(0, (3, 3)), zorder=1)
    ax.text(5.25, 0.03, "first 5 days", rotation=90, va="bottom", ha="left",
            fontsize=8, color="#8a857b")
    ax.spines["bottom"].set_bounds(1, dmax)
    ax.spines["left"].set_bounds(0, 1)
    ax.set_xlim(1, dmax + 4.5)
    ax.set_ylim(-0.02, 1.04)
    ax.set_xticks([1, 5, 10, 15, dmax])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_xlabel("days since outbreak onset", fontsize=11)
    ax.set_ylabel("PSD(d) = Pr(detected within d days)", fontsize=11)
    save(fig, "slide_psd_small")


# =====================================================================
# 5 — PSD small multiples across magnitudes: CUSUM spotlighted in crimson
# =====================================================================
def psd_smallmultiples():
    SERIES = [
        ("OR-ens.",    "OR-ensemble (IF+LSTM-AE+NB-HMM)"),
        ("RateChange", "RateChange"),
        ("KNN",        "KNN"),
        ("Farrington", "Farrington"),
    ]
    PANELS = [("small", "results/timely_detection_curve.csv"),
              ("medium", "results/timely_detection_curve_medium.csv"),
              ("large", "results/timely_detection_curve_large.csv")]

    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.6), sharey=True)
    for ax, (mag, path) in zip(axes, PANELS):
        df = pd.read_csv(path)
        df = df[df["method"] != "method"]
        df["day"] = df["day"].astype(int)
        df["psd_all"] = df["psd_all"].astype(float)
        dmax = df["day"].max()
        ends = []
        for label, name in SERIES:
            sub = df[df["method"] == name].sort_values("day")
            if sub.empty:
                continue
            ax.plot(sub["day"], sub["psd_all"], "-", color=GREY, lw=1.3, zorder=2)
            ends.append([label, GREY, sub["psd_all"].values[-1], "normal"])
        # CUSUM hero
        sub = df[df["method"] == "CUSUM"].sort_values("day")
        x, y = sub["day"].values, sub["psd_all"].values
        ax.fill_between(x, 0, y, color=CRIMSON, alpha=0.07, zorder=1)
        ax.plot(x, y, "-", color=CRIMSON, lw=3.0, zorder=4)
        ends.append(["CUSUM", CRIMSON, y[-1], "bold"])

        ax.axvline(5, color="#cdc7bc", lw=0.6, ls=(0, (3, 3)), zorder=1)
        ax.spines["bottom"].set_bounds(1, dmax)
        ax.spines["left"].set_bounds(0, 1)
        ax.set_xlim(1, dmax)
        ax.set_ylim(-0.02, 1.04)
        ax.set_xticks([1, 5, 10, dmax])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
        if mag != "small":
            ax.tick_params(labelleft=False)

        ends.sort(key=lambda r: r[2])
        placed = []
        for label, color, yend, weight in ends:
            yy = yend
            for pp in placed:
                if abs(yy - pp) < 0.07:
                    yy = pp + 0.07
            placed.append(yy)
            ax.text(dmax + 0.6, yy, label, va="center", ha="left", fontsize=8.5,
                    color=color if color == CRIMSON else "#8a857b",
                    fontweight=weight, clip_on=False)
        ax.set_title(f"{mag} magnitude", fontsize=12, pad=8)
        ax.set_xlabel("days since onset", fontsize=10)
    axes[0].set_ylabel("PSD(d)", fontsize=11)
    fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.16, wspace=0.46)
    save(fig, "slide_psd_smallmultiples")


# =====================================================================
# 6 — TITLE BACKDROP: a faint field of every detector's PSD curve, the
#     ensemble drawn in pale crimson. Data shown large as quiet art.
# =====================================================================
def titlecurves():
    df = pd.read_csv("results/timely_detection_curve.csv")
    df = df[df["method"] != "method"]
    df["day"] = df["day"].astype(int)
    df["psd_all"] = df["psd_all"].astype(float)
    HERO = "OR-ensemble (IF+LSTM-AE+NB-HMM)"

    fig, ax = plt.subplots(figsize=(13.3, 4.2))
    for name, sub in df.groupby("method"):
        sub = sub.sort_values("day")
        if name == HERO:
            continue
        ax.plot(sub["day"], sub["psd_all"], "-", color="#E4E2DD", lw=1.6, zorder=1)
    sub = df[df["method"] == HERO].sort_values("day")
    ax.fill_between(sub["day"], 0, sub["psd_all"], color=CRIMSON, alpha=0.05, zorder=1)
    ax.plot(sub["day"], sub["psd_all"], "-", color=CRIMSON, lw=2.2, alpha=0.32, zorder=2)
    ax.set_xlim(1, sub["day"].max())
    ax.set_ylim(-0.02, 1.05)
    ax.axis("off")
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    save(fig, "slide_titlecurves")


# =====================================================================
# 7 — SPARKLINE SUPERTABLE (Tufte's data-words integration): one row per
#     detector, with the within-table numbers AND a word-sized PSD(d)
#     sparkline coloured by family. Medium magnitude, sorted by sensitivity.
# =====================================================================
def sparktable():
    # small-magnitude sensitivity / specificity / family per method
    META = [
        ("OR-ensemble", "OR-ensemble (IF+LSTM-AE+NB-HMM)", 0.774, 0.946, "ensemble"),
        ("KNN",         "KNN",        0.615, 0.973, "tabular"),
        ("LSTM-AE",     "LSTM-AE",    0.570, 0.975, "deep"),
        ("LOF",         "LOF",        0.565, 0.971, "tabular"),
        ("Isolation Forest", "IF",   0.523, 0.974, "tabular"),
        ("NB-HMM",      "NB-HMM",     0.522, 0.977, "statistical"),
        ("OCSVM",       "OCSVM",      0.464, 0.973, "tabular"),
        ("Farrington",  "Farrington", 0.440, 0.980, "statistical"),
        ("VAE",         "VAE",        0.424, 0.972, "deep"),
        ("CUSUM",       "CUSUM",      0.404, 0.975, "statistical"),
        ("RateChange",  "RateChange", 0.234, 0.973, "statistical"),
        ("BOCPD",       "BOCPD",      0.175, 0.973, "statistical"),
    ]
    curve = pd.read_csv("results/timely_detection_curve.csv")
    curve = curve[curve["method"] != "method"].copy()
    curve["day"] = curve["day"].astype(int)
    curve["psd_all"] = curve["psd_all"].astype(float)
    dmax = curve["day"].max()

    n = len(META)
    fig, ax = plt.subplots(figsize=(9.6, 5.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(-1.35, n - 0.4)
    ax.invert_yaxis()
    ax.axis("off")

    # column anchors (figure-fraction within the axes 0..1)
    X_DOT, X_NAME = 0.005, 0.03
    X_SENS, X_SPEC = 0.37, 0.45
    X_SP0, X_SP1 = 0.51, 0.80
    X_P5 = 0.895
    X_POD = 0.985
    DSPARK = 10                              # show the first 10 days (the rise); full curve runs to 21

    def spark_y(row_y, psd):
        return row_y - (psd - 0.5) * 0.74   # inverted axis: higher psd -> visually higher

    # header
    hy = -1.12
    ax.text(X_NAME, hy, "detector", ha="left", va="center", fontsize=10.5, color=INK, style="italic")
    ax.text(X_SENS, hy, "Sens", ha="right", va="center", fontsize=10.5, color=INK, style="italic")
    ax.text(X_SPEC, hy, "Spec", ha="right", va="center", fontsize=10.5, color=INK, style="italic")
    ax.text((X_SP0 + X_SP1) / 2, hy, "PSD(d): chance caught within d days  →", ha="center", va="center",
            fontsize=9.8, color=INK, style="italic")
    ax.text(X_P5, hy, "5 d", ha="right", va="center", fontsize=10.5, color=INK, style="italic")
    ax.text(X_POD, hy, "POD", ha="right", va="center", fontsize=10.5, color=INK, style="italic")
    ax.plot([0, 1], [-0.55, -0.55], "-", color=INK, lw=0.8)

    for i, (disp, key, sens, spec, fam) in enumerate(META):
        col = FAMCOL[fam]
        sub = curve[curve["method"] == key].sort_values("day")
        # family colour dot + name
        ax.plot(X_DOT, i, "o", color=col, ms=7, zorder=3, clip_on=False)
        ax.text(X_NAME, i, disp, ha="left", va="center", fontsize=10.5,
                color=INK, fontweight="bold" if fam == "ensemble" else "normal")
        ax.text(X_SENS, i, f"{sens:.2f}", ha="right", va="center", fontsize=10.5, color=INK)
        ax.text(X_SPEC, i, f"{spec:.2f}", ha="right", va="center", fontsize=10.5, color="#777")
        # sparkline — first DSPARK days only, so the early rise (where curves differ) fills the width
        if not sub.empty:
            spk = sub[sub["day"] <= DSPARK]
            xx = X_SP0 + (spk["day"].values - 1) / (DSPARK - 1) * (X_SP1 - X_SP0)
            yy = spark_y(i, spk["psd_all"].values)
            ax.plot(xx, yy, "-", color=col, lw=1.9, zorder=2, clip_on=False)
            # within-5-day marker
            d5 = sub[sub["day"] == 5]
            if not d5.empty:
                v5 = d5["psd_all"].values[0]
                x5 = X_SP0 + (5 - 1) / (DSPARK - 1) * (X_SP1 - X_SP0)
                ax.plot(x5, spark_y(i, v5), "o", color=col, ms=3.6, zorder=3, clip_on=False)
                ax.text(X_P5, i, f"{v5:.2f}", ha="right", va="center", fontsize=10.5, color=INK)
            # endpoint marker (BE sparkline convention) + the full POD number
            ax.plot(xx[-1], yy[-1], "o", color=col, ms=3.6, zorder=4, clip_on=False)
            pod = sub["psd_all"].values[-1]
            ax.text(X_POD, i, f"{pod:.2f}", ha="right", va="center", fontsize=10.5, color=INK)

    ax.plot([0, 1], [n - 0.55, n - 0.55], "-", color=INK, lw=0.8)
    # Documentation (Beautiful Evidence, Principle 5): sources, scale, n.
    ax.text(0.0, n - 0.18,
            "small magnitude · 16 signals × 100 test sims · n = 1,598 outbreaks · "
            "sparkline = first 10 days (dot = day 5); full catch rate in the POD column · Noufaily et al. (2019) generator",
            ha="left", va="top", fontsize=7.6, color="#9A958B", style="italic")
    fig.subplots_adjust(left=0.01, right=0.99, top=0.99, bottom=0.04)
    save(fig, "slide_headline_sparktable")


# =====================================================================
# 8 — DOT-DASH-PLOT of the operating space (Beautiful Evidence): a
#     sensitivity-vs-specificity scatter of all 12 detectors whose AXES
#     are the marginal distributions (dot-dash range frame). Multivariate,
#     comparative, family-coloured, ensemble spotlighted.
# =====================================================================
def operating_space():
    # (display, sens, spec, family)  — small magnitude
    D = [
        ("OR-ensemble", 0.774, 0.946, "ensemble"),
        ("KNN",         0.615, 0.973, "tabular"),
        ("LSTM-AE",     0.570, 0.975, "deep"),
        ("LOF",         0.565, 0.971, "tabular"),
        ("IF",          0.523, 0.974, "tabular"),
        ("NB-HMM",      0.522, 0.977, "statistical"),
        ("OCSVM",       0.464, 0.973, "tabular"),
        ("Farrington",  0.440, 0.980, "statistical"),
        ("VAE",         0.424, 0.972, "deep"),
        ("CUSUM",       0.404, 0.975, "statistical"),
        ("RateChange",  0.234, 0.973, "statistical"),
        ("BOCPD",       0.175, 0.973, "statistical"),
    ]
    LABEL = {"OR-ensemble", "KNN", "Farrington", "VAE", "CUSUM", "RateChange", "BOCPD", "NB-HMM"}
    xmin, xmax = 0.10, 0.86
    ymin, ymax = 0.918, 0.986

    fig, ax = plt.subplots(figsize=(8.2, 5.0))
    # points
    for disp, se, sp, fam in D:
        col = FAMCOL[fam]
        hero = (fam == "ensemble")
        ax.plot(se, sp, "o", color=col, ms=10 if hero else 7,
                mec=PAPER, mew=1.0, zorder=4)
        if disp in LABEL:
            below = disp in ("RateChange", "VAE")
            ax.text(se, sp + (-0.0024 if below else 0.0022), disp, ha="center",
                    va="top" if below else "bottom",
                    fontsize=8.5, color=col, fontweight="bold" if hero else "normal",
                    zorder=5)
    # dot-dash marginal distributions ON the frame
    for disp, se, sp, fam in D:
        col = FAMCOL[fam]
        ax.plot([se, se], [ymin, ymin + 0.0035], "-", color=col, lw=1.3, clip_on=False, zorder=3)   # x-rug
        ax.plot([xmin, xmin + 0.012], [sp, sp], "-", color=col, lw=1.3, clip_on=False, zorder=3)     # y-rug

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(ymin, ymax)
    ax.spines["bottom"].set_bounds(min(d[1] for d in D), max(d[1] for d in D))
    ax.spines["left"].set_bounds(min(d[2] for d in D), max(d[2] for d in D))
    ax.set_xticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticks([0.93, 0.95, 0.97])
    ax.set_yticklabels(["0.93", "0.95", "0.97"])
    ax.set_xlabel("sensitivity  (more outbreak days caught  →)", fontsize=11)
    ax.set_ylabel("specificity  (fewer false alarms  ↑)", fontsize=11)
    # tiny direct colour key (no legend box)
    key = [("shallow ML", TEAL), ("deep ML", AMBER), ("statistical", SLATE), ("ensemble", CRIMSON)]
    for i, (lab, col) in enumerate(key):
        ax.text(0.62, ymin + 0.004 + i * 0.0052, "■ " + lab, transform=ax.transData,
                fontsize=8.5, color=col, ha="left", va="bottom")
    ax.text(0.0, -0.13,
            "small magnitude · 16 signals · threshold tuned on validation to spec ≥ 0.97 · "
            "axis ticks are the marginal distributions (dot-dash range frame)",
            transform=ax.transAxes, ha="left", va="top", fontsize=7.6, color="#9A958B", style="italic")
    fig.subplots_adjust(left=0.10, right=0.98, top=0.97, bottom=0.16)
    save(fig, "slide_operating_space")


if __name__ == "__main__":
    print("Building vivid slide figures:")
    slopegraph()
    persignal()
    headline_ci()
    psd_small()
    psd_smallmultiples()
    titlecurves()
    sparktable()
    operating_space()
    print("Done -> figs/slide_*.{png,pdf}")
