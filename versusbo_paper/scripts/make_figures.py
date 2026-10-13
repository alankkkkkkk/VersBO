"""Data figures for the manuscript (reads shared/generated/*).

Usage: python3 make_figures.py {times|libertine}
Writes vector PDFs to shared/figures/<font>/ so each template gets plots
whose lettering matches its body font.
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.lines import Line2D
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
GEN = ROOT / "shared" / "generated"
FONT = sys.argv[1] if len(sys.argv) > 1 else "times"
OUT = ROOT / "shared" / "figures" / FONT
OUT.mkdir(parents=True, exist_ok=True)

# TrueType copies made by otf2ttf.py (Type 42 embedding, no Type 3 fonts).
d = str(Path(__file__).resolve().parent / "fonts") + "/"
if FONT == "times":
    files = ["texgyretermes-regular.ttf", "texgyretermes-italic.ttf", "texgyretermes-bold.ttf",
             "texgyretermes-bolditalic.ttf"]
    family = "TermesTT"
else:
    files = ["LibertinusSerif-Regular.ttf", "LibertinusSerif-Italic.ttf", "LibertinusSerif-Bold.ttf",
             "LibertinusSerif-BoldItalic.ttf"]
    family = "LibertinusTT"
math = "custom"
for f in files:
    p = Path(d) / f
    if p.exists():
        fm.fontManager.addfont(str(p))

plt.rcParams.update({
    "font.family": family,
    "mathtext.fontset": math,
    "mathtext.rm": family, "mathtext.it": f"{family}:italic", "mathtext.bf": f"{family}:bold",
    "font.size": 7.5,
    "axes.titlesize": 8,
    "axes.labelsize": 7.5,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 7,
    "axes.linewidth": 0.6,
    "axes.edgecolor": "#3d3d3a",
    "axes.labelcolor": "#1f1f1d",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "pdf.fonttype": 42,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.015,
})

# Validated categorical palette (dataviz validator: all checks pass, light surface)
RED, BLUE, YELLOW, AQUA, VIOLET = "#d6393b", "#2a78d6", "#eda100", "#1baf7a", "#4a3aa7"
INK, MUTED, GRID = "#1f1f1d", "#6b6a66", "#e6e4df"

R = json.loads((GEN / "results.json").read_text())
NAME = json.loads((ROOT / "shared" / "method_name.json").read_text())["display"]
curves = np.load(GEN / "curves.npz")
lam = np.load(GEN / "lambda.npz")
TASKS = R["tasks"]


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# ------------------------------------------------ Fig. 1b: comparisons vs ratings
def fig1b():
    llm = R["rq0_llm"]
    names = ["Gemma\n12B", "Qwen\n35B", "gpt-oss\n120b", "DeepSeek\nV4", "GPT-5"]
    pair = [x[1] for x in llm]
    rate = [x[2] for x in llm]
    fig, ax = plt.subplots(figsize=(1.98, 1.62))
    x = np.arange(len(llm))
    w = 0.36
    ax.bar(x - w / 2 - 0.01, rate, w, color=BLUE, label="Rating (1–10)", zorder=3,
           edgecolor="white", linewidth=0.6)
    ax.bar(x + w / 2 + 0.01, pair, w, color=RED, label="Pairwise comparison", zorder=3,
           edgecolor="white", linewidth=0.6)
    for xi, (a, b) in enumerate(zip(rate, pair)):
        ax.text(xi + w / 2 + 0.01, b + 0.012, f"{b:.2f}".lstrip("0"), ha="center", va="bottom",
                fontsize=5.8, color=INK)
        ax.text(xi - w / 2 - 0.01, a + 0.012, f"{a:.2f}".lstrip("0"), ha="center", va="bottom",
                fontsize=5.8, color=MUTED)
    ax.axhline(0.5, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=2)
    ax.set_ylim(0.45, 0.95)
    ax.set_yticks([0.5, 0.6, 0.7, 0.8, 0.9])
    ax.set_xticks(x)
    ax.set_xticklabels(names, fontsize=5.6, linespacing=0.95)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_ylabel("Agreement with descriptor")
    grid(ax)
    ax.legend(loc="lower left", bbox_to_anchor=(-0.02, 0.985), frameon=False, handlelength=0.9,
              handletextpad=0.4, borderaxespad=0.0, ncol=2, fontsize=6.2, columnspacing=0.9)
    fig.savefig(OUT / "fig1b_accuracy.pdf")
    plt.close(fig)


# ---------------------------------- Fig. 1c: right-oracle vs wrong-oracle regret
def fig1c():
    rb = R["robust"]
    pts = [
        ("lvgp", "LVGP", rb["lvgp"][0], rb["lvgp"][1], VIOLET, "s"),
        ("onehot", "GP, one-hot", R["onehot_groupA"], R["onehot_groupA"], VIOLET, "s"),
        ("llmgo", "LLM-GO", rb["llmgo"][0], rb["llmgo"][1], BLUE, "o"),
        ("boicl", "BO-ICL", rb["boicl"][0], rb["boicl"][1], BLUE, "o"),
        ("lgbo", "LGBO", rb["lgbo"][0], rb["lgbo"][1], BLUE, "o"),
        ("llmpref", "LLM-Pref", rb["llmpref"][0], rb["llmpref"][1], BLUE, "o"),
        ("caliper", "CALIPER", rb["caliper"][0], rb["caliper"][1], BLUE, "o"),
        ("tacit_fixed", NAME + r", $\lambda$ fixed", rb["tacit_fixed"][0], rb["tacit_fixed"][1], RED, "D"),
        ("tacit", NAME, rb["tacit"][0], rb["tacit"][1], RED, "*"),
    ]
    off = {"lvgp": (0.8, -2.4, "left"), "onehot": (0.0, -2.6, "center"), "llmgo": (0.9, 0.4, "left"),
           "boicl": (0.9, -0.2, "left"), "lgbo": (-0.9, -1.4, "right"), "llmpref": (0.9, -0.3, "left"),
           "caliper": (0.9, -0.8, "left"), "tacit_fixed": (0.9, -0.7, "left"), "tacit": (0.9, -3.3, "left")}
    fig, ax = plt.subplots(figsize=(1.98, 1.62))
    lv = rb["lvgp"][1]
    ax.axhspan(lv, 46, color="#f3f1ec", zorder=0, lw=0)
    ax.axhline(lv, color=MUTED, lw=0.6, ls=(0, (2, 2)), zorder=1)
    ax.text(34.6, 45.2, "shaded: worse than\nignoring the LLM", fontsize=5.6, color=MUTED, ha="right",
            va="top", style="italic", linespacing=1.0)
    ax.plot([8, 36], [8, 36], color=GRID, lw=0.8, zorder=1)
    for key, lab, xv, yv, col, mk in pts:
        filled = key != "tacit_fixed"
        ax.scatter(xv, yv, s=46 if mk == "*" else 15, marker=mk, color=col if filled else "white",
                   edgecolor=col, linewidth=0.9, zorder=4)
        dx, dy, ha = off[key]
        ax.text(xv + dx, yv + dy, lab, fontsize=6.0, ha=ha, va="bottom",
                color=INK if key in ("tacit", "tacit_fixed") else "#3d3d3a",
                fontweight="bold" if key == "tacit" else "normal")
    ax.annotate("", xy=(rb["tacit"][0] + 0.3, rb["tacit"][1] + 0.9),
                xytext=(rb["tacit_fixed"][0] + 0.3, rb["tacit_fixed"][1] - 1.2),
                arrowprops=dict(arrowstyle="-|>", color=RED, lw=0.7, mutation_scale=6,
                                shrinkA=0, shrinkB=0, ls=(0, (3, 1.5))))
    ax.text(rb["tacit"][0] - 0.75, 33.0, r"learned $\lambda$", fontsize=5.8, color=RED, ha="center",
            rotation=90, va="center")
    ax.set_xlim(8, 35)
    ax.set_ylim(20, 46)
    ax.set_xticks([10, 15, 20, 25, 30, 35])
    ax.set_yticks([20, 25, 30, 35, 40, 45])
    ax.set_xlabel("Regret, LLM right (original)")
    ax.set_ylabel("Regret, LLM wrong")
    fig.savefig(OUT / "fig1c_tradeoff.pdf")
    plt.close(fig)


# --------------------------------------------- Fig. 3: curves and reliabilities
SHOW = [("lvgp", "LVGP", VIOLET, (0, (3, 1.6))), ("gollum", "GOLLuM", AQUA, "-"),
        ("llmgo", "LLM-GO", YELLOW, "-"), ("lgbo", "LGBO", BLUE, "-"), ("tacit", NAME, RED, "-")]


def curve_panel(ax, task, ylabel=False, title=None):
    t = np.arange(41)
    for key, lab, col, ls in SHOW:
        c = curves[f"{task}__{key}"]
        mu = c.mean(0)
        se = c.std(0, ddof=1) / np.sqrt(c.shape[0])
        lw = 1.5 if key == "tacit" else 1.0
        ax.fill_between(t, np.maximum(mu - se, 0.3), mu + se, color=col, alpha=0.14, lw=0,
                        zorder=2 if key != "tacit" else 3)
        ax.plot(t, mu, color=col, lw=lw, ls=ls, zorder=4 if key == "tacit" else 3, label=lab)
    ax.set_yscale("log")
    ax.set_xlim(0, 40)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_title(title or task, pad=2.5, loc="left", fontsize=7.6)
    if ylabel:
        ax.set_ylabel("Normalized regret")
    ax.set_xlabel("Iteration", labelpad=1)
    grid(ax, "both")


def set_log_ticks(ax, ticks):
    ax.set_yticks(ticks)
    ax.set_yticklabels([str(v) for v in ticks])
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())


def lambda_panel(ax):
    t = lam["t"]
    for k in lam.files:
        if not k.startswith("orig__"):
            continue
        name = k[len("orig__"):]
        arr = lam[k]
        med = np.full(arr.shape[1], np.nan)
        ok = ~np.all(np.isnan(arr), axis=0)
        med[ok] = np.nanmedian(arr[:, ok], axis=0)
        disc = "discovered" in name
        ax.plot(t, med, color=RED, lw=1.0, ls=(0, (1, 1.2)) if disc else "-", alpha=0.95, zorder=3)
    for k in lam.files:
        if not k.startswith("cf__"):
            continue
        med = np.median(lam[k], axis=0)
        ax.plot(t, med, color="#8d8b86", lw=1.0, zorder=2)
    ax.axhline(0.2, color=MUTED, lw=0.5, ls=(0, (2, 2)))
    ax.set_xlim(0, 40)
    ax.set_ylim(0, 3.0)
    ax.set_xticks([0, 10, 20, 30, 40])
    ax.set_yticks([0, 1, 2, 3])
    ax.set_xlabel("Iteration", labelpad=1)
    ax.set_ylabel(r"Posterior mean $\lambda_k$", labelpad=1)
    ax.set_title(r"(e) DA: reliability $\lambda_k$", pad=2.5, loc="left", fontsize=7.6)
    ax.text(39.5, 2.78, "original oracle", color=RED, fontsize=6.0, ha="right", va="bottom")
    ax.text(39.5, 0.30, "counterfactual", color="#6b6a66", fontsize=6.0, ha="right", va="bottom")
    ax.annotate("new axis", xy=(20.6, 1.02), xytext=(26.0, 1.27), fontsize=5.6, color=RED,
                va="center", ha="left",
                arrowprops=dict(arrowstyle="-", color=RED, lw=0.5, shrinkA=0.5, shrinkB=0.5))
    grid(ax, "both")


def fig3():
    fig, axes = plt.subplots(1, 5, figsize=(6.3, 1.55), gridspec_kw={"wspace": 0.34})
    spec = [("DA", [5, 10, 20, 50, 100]), ("NSA", [10, 20, 50, 100]), ("ChanLam", [20, 50, 100]),
            ("NiSuz", [10, 20, 50, 100])]
    for i, (task, ticks) in enumerate(spec):
        curve_panel(axes[i], task, ylabel=(i == 0), title=f"({'abcd'[i]}) {task}")
        set_log_ticks(axes[i], ticks)
    lambda_panel(axes[4])
    handles = [Line2D([], [], color=c, lw=1.5 if k == "tacit" else 1.0, ls=ls, label=lab)
               for k, lab, c, ls in SHOW[::-1]]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.41, 1.10),
               handlelength=1.8, columnspacing=1.4)
    fig.savefig(OUT / "fig3_curves.pdf")
    plt.close(fig)


def fig_all():
    fig, axes = plt.subplots(3, 3, figsize=(6.3, 4.6), gridspec_kw={"wspace": 0.28, "hspace": 0.62})
    ticks = {"DA": [5, 10, 20, 50, 100], "DeoxyF": [10, 20, 50, 100], "NSA": [10, 20, 50, 100],
             "Boryl": [10, 20, 50, 100], "Amide": [10, 20, 50, 100], "RedAm": [10, 20, 50, 100],
             "SuzC": [1, 10, 100], "ChanLam": [20, 50, 100], "NiSuz": [10, 20, 50, 100]}
    for i, task in enumerate(TASKS):
        ax = axes[i // 3, i % 3]
        curve_panel(ax, task, ylabel=(i % 3 == 0))
        set_log_ticks(ax, ticks[task])
    handles = [Line2D([], [], color=c, lw=1.5 if k == "tacit" else 1.0, ls=ls, label=lab)
               for k, lab, c, ls in SHOW[::-1]]
    fig.legend(handles=handles, loc="upper center", ncol=5, frameon=False, bbox_to_anchor=(0.5, 0.965),
               handlelength=1.8, columnspacing=1.4)
    fig.savefig(OUT / "figA_curves_all.pdf")
    plt.close(fig)


fig1b()
fig1c()
fig3()
fig_all()
print("wrote", sorted(p.name for p in OUT.iterdir()))
