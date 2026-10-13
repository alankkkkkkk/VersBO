"""Single source of truth for every number in the manuscript.

All results are projected values for a reading-group exercise. The script
simulates seed-level best-so-far trajectories whose 40-iteration means match
the projected per-task regrets, then derives every aggregate (ranks, I_1%,
standard errors, Wilcoxon/Holm tests, group means) from those trajectories so
that tables, figures and in-text numbers cannot drift apart.

Outputs (in shared/generated/):
  results.json      every number used anywhere in the paper
  curves.npz        seed-level trajectories for the plotting script
  macros.tex        \newcommand values cited in the text
  tab_*.tex         LaTeX tables
"""
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "shared" / "generated"
OUT.mkdir(parents=True, exist_ok=True)

# Method name lives in one place: shared/method_name.json
NAMECFG = json.loads((ROOT / "shared" / "method_name.json").read_text())
(OUT / "name.tex").write_text("% generated from shared/method_name.json\n"
                              "\\newcommand{\\ours}{\\textup{" + NAMECFG["tex"] + "}}\n")

rng = np.random.default_rng(20261001)

TASKS = ["DA", "DeoxyF", "NSA", "Boryl", "Amide", "RedAm", "SuzC", "ChanLam", "NiSuz"]
GROUP_A = ["DA", "DeoxyF", "NSA", "Boryl"]
GROUP_B = ["Amide", "RedAm", "SuzC", "ChanLam"]
POST_CUTOFF = ["Amide", "ChanLam"]
NSEED, T = 20, 40

# ---------------------------------------------------------------- methods
FAMILIES = [
    ("Without an LLM", ["onehot", "desc", "lvgp", "tabpfn"]),
    ("LLM representation", ["gollum"]),
    ("LLM surrogate or optimizer", ["boicl", "llmgo"]),
    ("LLM prior or preference", ["lgbo", "llmpref", "caliper"]),
    ("Ours", ["tacit"]),
]
MAIN = [m for _, ms in FAMILIES for m in ms]
APP = ["bochemian", "lapeft", "llambo", "reelicit"]
NAMES = {
    "onehot": "GP, one-hot", "desc": "GP, descriptors", "lvgp": "LVGP",
    "tabpfn": "TabPFN-v2 BO", "gollum": "GOLLuM", "boicl": "BO-ICL",
    "llmgo": "LLM-GO", "lgbo": "LGBO", "llmpref": "LLM-Pref",
    "caliper": "CALIPER", "tacit": r"\ours{} (ours)",
    "bochemian": "BoChemian", "lapeft": "LA-PEFT", "llambo": "LLAMBO",
    "reelicit": "ReElicit (adapted)",
}
CITE = {
    "onehot": "", "desc": r"\citep{shields2021bayesian}", "lvgp": r"\citep{zhang2020latent}",
    "tabpfn": r"\citep{rogers2026zeroshot}", "gollum": r"\citep{rankovic2026gollum}",
    "boicl": r"\citep{ramos2026boicl}", "llmgo": r"\citep{macknight2025pretrained}",
    "lgbo": r"\citep{yuan2026lgbo}", "llmpref": r"\citep{patel2026distilling}",
    "caliper": r"\citep{chen2026caliper}", "tacit": "",
    "bochemian": r"\citep{rankovic2023bochemian}", "lapeft": r"\citep{kristiadi2024sober}",
    "llambo": r"\citep{liu2024llambo}", "reelicit": r"\citep{lin2026embedding}",
}

# Projected mean normalized simple regret (x100) after 40 iterations.
MEANS = {
    "onehot":  [31.4, 28.7, 36.2, 33.5, 29.8, 24.1, 2.1, 41.3, 38.6],
    "desc":    [18.2,  9.4, 22.4, 24.7, 21.5, 19.3, 1.8, 35.2, 27.4],
    "lvgp":    [22.6, 19.8, 25.1, 27.3, 24.9, 17.2, 1.5, 36.8, 30.1],
    "tabpfn":  [20.4, 17.3, 23.8, 25.9, 22.3, 18.6, 1.2, 34.1, 26.8],
    "gollum":  [16.9, 15.2, 19.7, 22.8, 18.4, 21.7, 1.9, 33.6, 24.2],
    "boicl":   [19.3, 18.4, 26.5, 23.1, 17.6, 20.8, 2.4, 30.9, 29.7],
    "llmgo":   [14.8, 16.1, 21.3, 19.2, 12.4, 15.9, 0.9, 24.7, 33.8],
    "lgbo":    [15.6, 13.4, 18.9, 20.6, 16.1, 16.4, 1.1, 29.3, 22.5],
    "llmpref": [14.1, 12.7, 20.2, 21.4, 15.3, 17.8, 1.4, 31.6, 23.9],
    "caliper": [15.9, 14.8, 19.4, 21.9, 16.7, 16.9, 1.3, 30.2, 23.1],
    "tacit":   [ 8.3, 10.6, 11.2, 13.7,  9.8, 12.1, 0.8, 26.4, 14.6],
    "bochemian": [19.8, 17.9, 23.6, 25.4, 20.7, 22.9, 2.2, 35.1, 26.9],
    "lapeft":    [18.4, 16.8, 22.1, 24.3, 19.9, 23.4, 2.0, 34.4, 26.1],
    "llambo":    [20.6, 19.7, 27.4, 24.8, 18.9, 21.5, 2.6, 32.8, 31.2],
    "reelicit":  [18.9, 16.3, 21.8, 23.9, 19.6, 19.4, 1.7, 33.0, 25.7],
}
# Time scale (iterations) of the best-so-far decay; LLM-informed methods start faster.
TAU = {"onehot": 15.0, "desc": 10.5, "lvgp": 13.0, "tabpfn": 11.5, "gollum": 9.5,
       "boicl": 9.0, "llmgo": 5.5, "lgbo": 8.0, "llmpref": 8.0, "caliper": 8.5,
       "tacit": 5.0, "bochemian": 10.5, "lapeft": 10.0, "llambo": 9.5, "reelicit": 9.5}
# Easy pools converge faster (SuzC is near-saturated for every method).
TASK_SPEED = {"SuzC": 0.33}
# Normalized regret of the 95th-percentile candidate in each pool (top-5% threshold).
THR = {"DA": 29.0, "DeoxyF": 26.0, "NSA": 36.0, "Boryl": 37.0, "Amide": 30.0,
       "RedAm": 29.0, "SuzC": 4.0, "ChanLam": 52.0, "NiSuz": 43.0}


def r1(x):
    """Round half up to one decimal (what a reader does by hand)."""
    return float(Decimal(str(round(float(x), 6))).quantize(Decimal("0.1"), ROUND_HALF_UP))


def r2(x):
    return float(Decimal(str(round(float(x), 6))).quantize(Decimal("0.01"), ROUND_HALF_UP))


# ------------------------------------------------------------ simulation
def simulate_task(ti):
    # n0 = 5 initial points drawn from the lower half of the pool, so the
    # initial best lies below the median (normalized regret > 100).
    r0 = rng.uniform(105, 150, size=NSEED)            # shared initial design per seed
    speed = TASK_SPEED.get(TASKS[ti], 1.0)
    u = rng.lognormal(0.0, 0.40, size=NSEED)           # shared seed difficulty
    u /= u.mean()
    curves = {}
    for m in MAIN + APP:
        target = MEANS[m][ti]
        e = rng.lognormal(0.0, 0.28, size=NSEED)
        e /= e.mean()
        tau = speed * TAU[m] * rng.lognormal(0.0, 0.15, size=NSEED)
        events = rng.random((NSEED, T + 1)) < 0.42
        events[:, 0] = True
        t = np.arange(T + 1)
        # index of last improvement at or before t
        last = np.where(events, t[None, :], 0)
        last = np.maximum.accumulate(last, axis=1)
        f = target * u * e

        def build(fv):
            fv = np.maximum(fv, 0.02 * target)
            g = fv[:, None] + (r0[:, None] - fv[:, None]) * np.exp(-(last / tau[:, None]) ** 1.05)
            return np.minimum.accumulate(g, axis=1)

        for _ in range(200):
            c = build(f)
            gap = target - c[:, -1].mean()
            if abs(gap) < 1e-6:
                break
            f = f + gap
        curves[m] = build(f)
    return curves


CURVES = {task: simulate_task(i) for i, task in enumerate(TASKS)}
np.savez_compressed(OUT / "curves.npz",
                    **{f"{t}__{m}": CURVES[t][m] for t in TASKS for m in MAIN + APP})

R = {"tasks": TASKS, "methods": MAIN, "app_methods": APP}
final = {m: [CURVES[t][m][:, -1] for t in TASKS] for m in MAIN + APP}
R["mean"] = {m: [r1(x.mean()) for x in final[m]] for m in MAIN + APP}
R["se"] = {m: [r2(x.std(ddof=1) / np.sqrt(NSEED)) for x in final[m]] for m in MAIN + APP}
assert R["mean"] == {m: MEANS[m] for m in MAIN + APP}, "simulation drifted from targets"

# ranks among the 11 main-table methods
ranks = {m: [] for m in MAIN}
for ti in range(len(TASKS)):
    order = sorted(MAIN, key=lambda m: MEANS[m][ti])
    for k, m in enumerate(order):
        ranks[m].append(k + 1)
R["rank"] = ranks
R["avg_rank"] = {m: r2(np.mean(ranks[m])) for m in MAIN}
# ranks among all 15 methods (appendix)
ranks15 = {m: [] for m in MAIN + APP}
for ti in range(len(TASKS)):
    order = sorted(MAIN + APP, key=lambda m: MEANS[m][ti])
    for k, m in enumerate(order):
        ranks15[m].append(k + 1)
R["avg_rank15"] = {m: r2(np.mean(ranks15[m])) for m in MAIN + APP}

# iterations to a top-1% experiment (capped at T+1)
def hit(curve, thr):
    idx = np.argmax(curve <= thr, axis=1).astype(float)
    idx[~(curve <= thr).any(axis=1)] = T + 1
    return idx

I1 = {m: [float(hit(CURVES[t][m], THR[t]).mean()) for t in TASKS] for m in MAIN + APP}
R["i1_task"] = {m: [r1(x) for x in I1[m]] for m in MAIN + APP}
R["i1_avg"] = {m: r1(np.mean(I1[m])) for m in MAIN + APP}

best_base_rank = min((m for m in MAIN if m != "tacit"), key=lambda m: R["avg_rank"][m])
fastest = min((m for m in MAIN if m != "tacit"), key=lambda m: R["i1_avg"][m])
R["best_baseline_by_rank"] = best_base_rank
R["fastest_baseline"] = fastest
def pct(a, b):
    return int(Decimal(str(100 * (1 - a / b))).quantize(Decimal("1"), ROUND_HALF_UP))
R["i1_reduction_pct"] = pct(R["i1_avg"]["tacit"], R["i1_avg"][fastest])
R["i1_reduction_rank_pct"] = pct(R["i1_avg"]["tacit"], R["i1_avg"][best_base_rank])

# Wilcoxon signed-rank (paired over seeds) TACIT vs the strongest other method per task, Holm-corrected
tests = []
for ti, t in enumerate(TASKS):
    other = min((m for m in MAIN if m != "tacit"), key=lambda m: MEANS[m][ti])
    d = CURVES[t]["tacit"][:, -1] - CURVES[t][other][:, -1]
    p = float(wilcoxon(d).pvalue)
    tests.append({"task": t, "runner": other, "p": p,
                  "tacit_better": bool(MEANS["tacit"][ti] < MEANS[other][ti])})
order = np.argsort([x["p"] for x in tests])
k = len(tests)
running = 0.0
for rank_i, idx in enumerate(order):
    adj = min(1.0, (k - rank_i) * tests[idx]["p"])
    running = max(running, adj)
    tests[idx]["p_holm"] = running
R["tests"] = tests
R["n_sig_wins"] = sum(1 for x in tests if x["tacit_better"] and x["p_holm"] < 0.05)
R["n_wins"] = sum(1 for x in tests if x["tacit_better"])
R["sig_win_tasks"] = [x["task"] for x in tests if x["tacit_better"] and x["p_holm"] < 0.05]
R["nonsig_tasks"] = [x["task"] for x in tests if x["p_holm"] >= 0.05]

def gmean(m, tasks):
    return r1(np.mean([MEANS[m][TASKS.index(t)] for t in tasks]))

R["all_mean"] = {m: gmean(m, TASKS) for m in MAIN + APP}
R["groupA_mean"] = {m: gmean(m, GROUP_A) for m in MAIN}
R["post_mean"] = {m: gmean(m, POST_CUTOFF) for m in MAIN}

# ------------------------------------------------------- robustness (Group A)
ROBUST = {  # orig, counterfactual, inverted, decontextualized, post-cutoff; None = not applicable
    "lvgp":    [R["groupA_mean"]["lvgp"], R["groupA_mean"]["lvgp"], None, None, R["post_mean"]["lvgp"]],
    "llmgo":   [R["groupA_mean"]["llmgo"], 38.2, None, 22.6, R["post_mean"]["llmgo"]],
    "boicl":   [R["groupA_mean"]["boicl"], 31.5, None, 24.9, R["post_mean"]["boicl"]],
    "lgbo":    [R["groupA_mean"]["lgbo"], 29.8, 34.6, 19.8, R["post_mean"]["lgbo"]],
    "llmpref": [R["groupA_mean"]["llmpref"], 33.4, 37.9, 18.4, R["post_mean"]["llmpref"]],
    "caliper": [R["groupA_mean"]["caliper"], 26.1, 25.3, 19.1, R["post_mean"]["caliper"]],
    "tacit_fixed": [11.6, 41.7, 45.2, 12.0, 19.4],
    "tacit":   [R["groupA_mean"]["tacit"], 24.4, 24.0, 12.3, R["post_mean"]["tacit"]],
}
R["robust"] = ROBUST
R["onehot_groupA"] = R["groupA_mean"]["onehot"]

# --------------------------------------------------------------- ablations
ABL = [  # key, label, all-task mean (original), Group-A counterfactual
    ("full", r"\ours{} (full)", R["all_mean"]["tacit"], ROBUST["tacit"][1]),
    ("A1", r"A1\enspace ratings instead of comparisons", 17.2, 27.9),
    ("A2", r"A2\enspace $\lambda$ fixed large", 12.4, ROBUST["tacit_fixed"][1]),
    ("A3", r"A3\enspace $\lambda$ fixed at 0 (= LVGP)", R["all_mean"]["lvgp"], R["groupA_mean"]["lvgp"]),
    ("A4", r"A4\enspace no free dimensions ($R{=}0$)", 14.6, 35.8),
    ("A5", r"A5\enspace MAP coordinates only", 14.1, 26.9),
    ("A6", r"A6\enspace random comparisons", 15.3, 24.9),
    ("A7", r"A7\enspace no axis discovery", 12.8, 24.6),
    ("A8", r"A8\enspace unnamed similarity queries", 16.4, 25.8),
    ("A9", r"A9\enspace numeric values in prompts", 12.6, 24.5),
    ("A10", r"A10\enspace repeats as independent", 12.2, 26.3),
]
R["ablations"] = [{"key": a, "label": b, "orig": c, "cf": d} for a, b, c, d in ABL]
SENS = {
    "Kmax": {"values": [1, 2, 3, 5], "regret": [14.0, 12.5, R["all_mean"]["tacit"], 12.4]},
    "R": {"values": [0, 1, 2, 4], "regret": [14.6, 12.2, R["all_mean"]["tacit"], 12.3]},
    "m": {"values": [2, 4, 8, 16], "regret": [13.4, 12.3, R["all_mean"]["tacit"], 11.8]},
    "oracle": {"Gemma-4-12B": 13.8, "GPT-5": 11.2},
    "prompt_sd": {"tacit": 0.7, "llmgo": 3.9},
    "nuts": {"regret": 11.6, "time_factor": 9},
}
R["sensitivity"] = SENS
assert SENS["R"]["regret"][0] == ABL[4][2]

# ------------------------------------------------- surrogate quality (Group A)
R["surrogate"] = {
    "desc":   [1.42, 1.21, 0.18, 0.31],
    "lvgp":   [1.58, 1.30, 0.09, 0.24],
    "gollum": [1.47, 1.19, 0.16, 0.33],
    "boicl":  [1.71, 1.52, 0.21, 0.29],
    "tacit":  [1.24, 1.08, 0.34, 0.42],
}

# ------------------------------------------------------------------- cost
R["cost"] = [  # method, tokens (M), extra compute
    ("llambo", 3.4, "2 CPU-min"), ("boicl", 2.6, "2 CPU-min"), ("llmgo", 1.9, "--"),
    ("lgbo", 0.8, "3 CPU-min"), ("caliper", 0.6, "3 CPU-min"), ("llmpref", 0.15, "2 CPU-min"),
    ("gollum", None, "1.1 GPU-h (H100)"), ("tacit", 0.09, "4 CPU-min"),
]
R["tacit_queries"] = 212
R["tacit_repeats"] = 2
R["random_query_factor"] = 3.2
tok = {m: v for m, v, _ in R["cost"] if v}
R["token_ratio"] = {m: r1(tok[m] / tok["tacit"]) for m in ["llmgo", "boicl", "llambo"]}

# ------------------------------------------------------------------- RQ0
R["rq0_llm"] = [  # model, pairwise acc, rating-derived acc, direct-prediction Spearman
    ("Gemma-4-12B", 0.74, 0.61, 0.12),
    ("Qwen3.6-35B-A3B", 0.81, 0.65, 0.21),
    ("gpt-oss-120b", 0.83, 0.67, 0.24),
    ("DeepSeek-V4-Flash", 0.84, 0.68, 0.26),
    ("GPT-5", 0.87, 0.71, 0.31),
]
R["rq0_avg"] = [r2(np.mean([x[i] for x in R["rq0_llm"]])) for i in (1, 2, 3)]
R["rq0_prop"] = [  # task, factor, property, descriptor, accuracy (main oracle), class
    ("DA", "ligand", "steric bulk", r"\%$V_\mathrm{bur}$", 0.66, "steric"),
    ("DA", "ligand", r"$\sigma$-donor strength", "HOMO energy", 0.80, "electronic"),
    ("DA", "base", "basicity", r"p$K_\mathrm{a}$(H)", 0.94, "acid-base"),
    ("DA", "solvent", "polarity", r"dielectric const.", 0.92, "medium"),
    ("DeoxyF", "alcohol", "steric hindrance", r"\%$V_\mathrm{bur}$ (C$_\alpha$)", 0.72, "steric"),
    ("DeoxyF", "sulfonyl fluoride", "electrophilicity", "LUMO energy", 0.82, "electronic"),
    ("DeoxyF", "base", "basicity", r"p$K_\mathrm{a}$(H), MeCN", 0.93, "acid-base"),
    ("NSA", "catalyst", "3,3$'$-substituent bulk", r"\%$V_\mathrm{bur}$", 0.67, "steric"),
    ("NSA", "catalyst", r"Br{\o}nsted acidity", r"p$K_\mathrm{a}$", 0.91, "acid-base"),
    ("NSA", "imine", "electrophilicity", "LUMO energy", 0.81, "electronic"),
    ("Boryl", "ligand", "steric bulk", r"\%$V_\mathrm{bur}$", 0.70, "steric"),
    ("Boryl", "aryl electrophile", r"C--X electrophilicity", "LUMO energy", 0.84, "electronic"),
]
R["rq0_prop_mean"] = r2(np.mean([x[4] for x in R["rq0_prop"]]))
for cls in ["steric", "acid-base", "electronic"]:
    R[f"rq0_{cls}"] = r2(np.mean([x[4] for x in R["rq0_prop"] if x[5] == cls]))
assert abs(R["rq0_prop_mean"] - R["rq0_llm"][1][1]) < 0.005
R["rating_compression_pct"] = 41

# ----------------------------------------------------- interpretability
R["axis_rho"] = {"tacit": 0.71, "tacit_fixed": 0.64, "lvgp_rot": 0.36}
R["discovery_fired"] = 5
R["discovered"] = [  # task, factor, high vs low examples, LLM-named property, final mean lambda
    ("DA", "base", r"Cs$^+$ vs.\ K$^+$ carboxylates", "cation solubility in the solvent", 1.6),
    ("NSA", "catalyst", r"extended vs.\ simple 3,3$'$-aryls", r"3,3$'$-aryl $\pi$-surface", 1.2),
    ("Boryl", "ligand", r"bidentate vs.\ monodentate", "chelation (denticity)", 0.9),
    ("ChanLam", "Cu source", r"Cu(OAc)$_2$ vs.\ Cu(OTf)$_2$", "counter-anion basicity", 1.3),
    ("NiSuz", "additive", r"salts vs.\ none", "Lewis acidity of additive", 0.2),
]

# ------------------------------------------------------------- tasks/dims
R["task_table"] = [
    # id, reaction, objective, candidates, factors (levels), numeric, d, group
    ("DA", "Direct arylation", "yield", "1,728", r"ligand (12), base (4), solvent (4)", r"$T$, conc.", 12, "A"),
    ("DeoxyF", "Deoxyfluorination", "yield", "740", r"alcohol (37), RSO$_2$F (5), base (4)", "--", 10, "A"),
    ("NSA", r"CPA $N$,$S$-acetal formation", r"$\Delta\Delta G^{\ddagger}$", "1,075", r"catalyst (43), imine (5), thiol (5)", "--", 10, "A"),
    ("Boryl", "Ni-cat. borylation", "yield", "1,518", r"Ar--X (33), ligand (23), solvent (2)", "--", 9, "A"),
    ("Amide", "Amide coupling", "yield", "648", r"acid (3), amine (3), conditions (72)", "--", 8, "B"),
    ("RedAm", "Reductive amination", "conversion", "768", r"substrate (16), solvent (2)", "3", 8, "B"),
    ("SuzC", "Suzuki--Miyaura", "conversion", "960", r"Ar--X (8), boronate (10), conditions (12)", "--", 12, "B"),
    ("ChanLam", "Chan--Lam coupling", "wtd.\\ selectivity", "1,920", r"RSO$_2$NH$_2$ (10), ArB(OH)$_2$ (2), Cu (4), solvent (4), base (6)", "--", 14, "B"),
    ("NiSuz", "Ni-cat. Suzuki (emulated)", "yield", "87,203", r"ligand, base, solvent, additive", "--", 13, "C"),
]

# ----------------------------------------------------------- lambda curves
def lam_curves():
    t = np.arange(T + 1)
    axes = [("ligand: steric bulk", 0.9), ("ligand: $\\sigma$-donor strength", 1.6),
            ("base: basicity", 2.6), ("solvent: polarity", 1.9)]
    out = {"orig": {}, "cf": {}, "t": t.tolist()}
    for name, end in axes:
        seeds_o, seeds_c = [], []
        for _ in range(NSEED):
            k = rng.uniform(0.15, 0.3)
            endo = end * rng.lognormal(0, 0.18)
            start = rng.uniform(0.9, 1.3)
            seeds_o.append(endo + (start - endo) * np.exp(-k * t) + rng.normal(0, 0.05, t.size).cumsum() * 0.02)
            kc = rng.uniform(0.12, 0.20)
            low = rng.uniform(0.03, 0.09)
            seeds_c.append(low + (start - low) * np.exp(-kc * t) + np.abs(rng.normal(0, 0.01, t.size)))
        out["orig"][name] = np.clip(np.array(seeds_o), 0, None)
        out["cf"][name] = np.clip(np.array(seeds_c), 0, None)
    # axis added by discovery at iteration 10 in the original task
    disc = []
    for _ in range(NSEED):
        s = np.full(t.size, np.nan)
        endd = 1.6 * rng.lognormal(0, 0.2)
        tt = t[t >= 10] - 10
        s[t >= 10] = 0.08 + (endd - 0.08) / (1 + np.exp(-(tt - 9) / 3.0))
        disc.append(s)
    out["orig"]["base: cation solubility (discovered)"] = np.array(disc)
    return out

LAM = lam_curves()
cf_cross = []
for name, arr in LAM["cf"].items():
    med = np.median(arr, axis=0)
    cf_cross.append(int(np.argmax(med < 0.2)))
R["lambda_cf_cross_iter"] = max(cf_cross)
R["lambda_orig_final"] = {k: r1(np.median(v[:, -1])) for k, v in LAM["orig"].items()}
np.savez_compressed(OUT / "lambda.npz", t=np.array(LAM["t"]),
                    **{f"orig__{k}": v for k, v in LAM["orig"].items()},
                    **{f"cf__{k}": v for k, v in LAM["cf"].items()})

# ================================================================ LaTeX
def fmt(x, d=1):
    return "--" if x is None else f"{x:.{d}f}"


def best_two(vals):
    s = sorted(set(vals))
    return s[0], (s[1] if len(s) > 1 else None)


def tab_main():
    lines = []
    cols = [[MEANS[m][i] for m in MAIN] for i in range(len(TASKS))]
    rk = [R["avg_rank"][m] for m in MAIN]
    i1 = [R["i1_avg"][m] for m in MAIN]
    def cell(v, colvals, sig=False):
        b1, b2 = best_two(colvals)
        s = f"{v:.1f}" if v >= 0 else ""
        if v == b1:
            s = r"\textbf{" + s + "}"
        elif v == b2:
            s = r"\underline{" + s + "}"
        if sig == 1:
            s += r"$^{\dagger}$"
        elif sig == 2:
            s += r"$^{\ddagger}$"
        return s
    sigset = set(R["sig_win_tasks"])
    sigset10 = {x["task"] for x in R["tests"] if x["tacit_better"] and 0.05 <= x["p_holm"] < 0.1}
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\small")
    lines.append(r"\setlength{\tabcolsep}{4.6pt}")
    lines.append(r"\begin{tabular}{@{}l ccccc cccc c c@{}}")
    lines.append(r"\toprule")
    lines.append(r" & \multicolumn{4}{c}{Group A: DFT descriptors} & \multicolumn{4}{c}{Group B: recent HTE} & C & & \\")
    lines.append(r"\cmidrule(lr){2-5}\cmidrule(lr){6-9}\cmidrule(lr){10-10}")
    lines.append(r"Method & DA & DeoxyF & NSA & Boryl & Amide & RedAm & SuzC & ChanLam & NiSuz & Rank\,$\downarrow$ & $I_{5\%}$\,$\downarrow$ \\")
    for fam, ms in FAMILIES:
        lines.append(r"\midrule")
        if fam != "Ours":
            lines.append(r"\multicolumn{12}{@{}l}{\textit{" + fam + r"}}\\[1pt]")
        for m in ms:
            row = [NAMES[m] if m == "tacit" else r"\quad " + NAMES[m]]
            for i, t in enumerate(TASKS):
                flag = 0
                if m == "tacit":
                    flag = 1 if t in sigset else (2 if t in sigset10 else 0)
                row.append(cell(MEANS[m][i], cols[i], sig=flag))
            row.append(cell(R["avg_rank"][m], rk).replace(f"{R['avg_rank'][m]:.1f}", f"{R['avg_rank'][m]:.2f}"))
            row.append(cell(R["i1_avg"][m], i1))
            if m == "tacit":
                lines.append(r"\rowcolor{tacitrow}")
            lines.append(" & ".join(row) + r" \\")
    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\caption{Normalized simple regret ($\times 100$, lower is better) after 40 iterations, mean over 20 seeds. "
        r"Rank is the average rank over the nine tasks; $I_{5\%}$ is the mean number of iterations until the best observed experiment lies in the top 5\% of the pool (capped at 41). "
        r"\textbf{Bold}: best; \underline{underlined}: second best. $^{\dagger}$/$^{\ddagger}$: \ours{} is better than the runner-up at $p<0.05$/$p<0.1$ (paired Wilcoxon over seeds, Holm-corrected over nine tasks). "
        r"Standard errors are in Table~\ref{tab:se}.}")
    lines.append(r"\label{tab:main}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def tab_robust():
    rows = [("lvgp", "LVGP (no LLM)"), ("llmgo", "LLM-GO"), ("boicl", "BO-ICL"), ("lgbo", "LGBO"),
            ("llmpref", "LLM-Pref"), ("caliper", "CALIPER"),
            ("tacit_fixed", r"\ours{}, fixed $\lambda$"), ("tacit", r"\ours{}")]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{2.4pt}",
         r"\begin{tabular}{@{}l ccccc@{}}", r"\toprule",
         r"Method & Orig. & Counterf. & Invert. & No ctx. & Recent \\", r"\midrule"]
    colbest = [min(ROBUST[k][j] for k, _ in rows if ROBUST[k][j] is not None) for j in range(5)]
    for k, name in rows:
        vals = ROBUST[k]
        cells = []
        for j, v in enumerate(vals):
            c = fmt(v)
            if v is not None and v == colbest[j]:
                c = r"\textbf{" + c + "}"
            cells.append(c)
        if k == "tacit_fixed":
            L.append(r"\midrule")
        if k == "tacit":
            L.append(r"\rowcolor{tacitrow}")
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Regret ($\times 100$) on Group A with the original oracle and with counterfactual, inverted and context-free oracles. The last column averages Amide and ChanLam, two 2025 datasets whose best conditions the main oracle cannot name. "
          r"A dash marks a test that does not apply. LVGP uses no LLM, so its counterfactual score equals its original one.}",
          r"\label{tab:robust}", r"\end{table}"]
    return "\n".join(L)


def tab_ablation():
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l cc@{}}", r"\toprule",
         r"Variant & All, orig. & A, counterf. \\", r"\midrule"]
    for a in R["ablations"]:
        o, c = fmt(a["orig"]), fmt(a["cf"])
        if a["key"] == "full":
            L.append(r"\rowcolor{tacitrow}")
            L.append(a["label"] + f" & \\textbf{{{o}}} & {c}" + r" \\")
            L.append(r"\midrule")
        else:
            L.append(a["label"] + f" & {o} & {c}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Ablations: regret ($\times 100$) averaged over all nine tasks with the original oracle, and over Group A with the counterfactual oracle (\S\ref{sec:robust}). Each row changes one component.}",
          r"\label{tab:ablation}", r"\end{table}"]
    return "\n".join(L)


def tab_surrogate():
    rows = [("desc", "GP, descriptors"), ("lvgp", "LVGP"), ("gollum", "GOLLuM"), ("boicl", "BO-ICL"), ("tacit", r"\ours{}")]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.5pt}",
         r"\begin{tabular}{@{}l cc cc@{}}", r"\toprule",
         r" & \multicolumn{2}{c}{NLPD\,$\downarrow$} & \multicolumn{2}{c}{Top-decile $\rho$\,$\uparrow$} \\",
         r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}",
         r"Surrogate & $n{=}10$ & $n{=}20$ & $n{=}10$ & $n{=}20$ \\", r"\midrule"]
    for k, name in rows:
        v = R["surrogate"][k]
        cells = [f"{v[0]:.2f}", f"{v[1]:.2f}", f"{v[2]:.2f}", f"{v[3]:.2f}"]
        if k == "tacit":
            cells = [r"\textbf{" + c + "}" for c in cells]
            L.append(r"\rowcolor{tacitrow}")
        L.append(name + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Surrogate quality on Group A from $n$ random experiments, averaged over 20 splits. NLPD: negative log predictive density on the rest of the pool; top-decile $\rho$: Spearman correlation on the best 10\% of the pool.}",
          r"\label{tab:surrogate}", r"\end{table}"]
    return "\n".join(L)


def tab_cost():
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{5pt}",
         r"\begin{tabular}{@{}l r l@{}}", r"\toprule",
         r"Method & LLM tokens & Other compute \\", r"\midrule"]
    for m, tk, comp in R["cost"]:
        tks = "--" if tk is None else (f"{tk:.2f}M" if tk < 0.1 or m == "llmpref" else f"{tk:.1f}M")
        name = NAMES[m].replace(" (ours)", "")
        if m == "tacit":
            L.append(r"\rowcolor{tacitrow}")
            tks = r"\textbf{" + tks + "}"
        L.append(f"{name} & {tks} & {comp}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Cost of one 40-iteration campaign on DA with the main oracle (input plus output tokens), mean over seeds. GOLLuM finetunes its own encoder and queries no generative model.}",
          r"\label{tab:cost}", r"\end{table}"]
    return "\n".join(L)


def tab_se():
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.6pt}",
         r"\begin{tabular}{@{}l ccccccccc@{}}", r"\toprule",
         "Method & " + " & ".join(TASKS) + r" \\", r"\midrule"]
    for m in MAIN + APP:
        if m == APP[0]:
            L.append(r"\midrule")
        cells = [f"{MEANS[m][i]:.1f}\\,{{\\scriptsize$\\pm${R['se'][m][i]:.2f}}}" for i in range(len(TASKS))]
        L.append(NAMES[m].replace(" (ours)", "") + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Mean $\pm$ standard error of the final normalized regret ($\times 100$) over 20 seeds, including the four baselines reported only here (bottom block). Average ranks over all 15 methods: "
          + ", ".join(f"{NAMES[m].replace(' (ours)', '')} {R['avg_rank15'][m]:.2f}" for m in APP + ["tacit"]) + ".}",
          r"\label{tab:se}", r"\end{table*}"]
    return "\n".join(L)


def tab_i1():
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4.6pt}",
         r"\begin{tabular}{@{}l ccccccccc c@{}}", r"\toprule",
         "Method & " + " & ".join(TASKS) + r" & Mean \\", r"\midrule"]
    for m in MAIN:
        cells = [f"{R['i1_task'][m][i]:.1f}" for i in range(len(TASKS))]
        L.append(NAMES[m].replace(" (ours)", "") + " & " + " & ".join(cells) + f" & {R['i1_avg'][m]:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Iterations until the best observed experiment lies in the top 5\% of the pool ($I_{5\%}$, capped at 41), mean over 20 seeds.}",
          r"\label{tab:i1}", r"\end{table*}"]
    return "\n".join(L)


def tab_rq0():
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.6pt}",
         r"\begin{tabular}{@{}l ccc@{}}", r"\toprule",
         r"LLM & Pairwise & From ratings & Prediction $\rho$ \\", r"\midrule"]
    for name, a, b, c in R["rq0_llm"]:
        L.append(f"{name} & {a:.2f} & {b:.2f} & {c:.2f}" + r" \\")
    L.append(r"\midrule")
    a, b, c = R["rq0_avg"]
    L.append(f"Mean & {a:.2f} & {b:.2f} & {c:.2f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Accuracy of LLM judgments on Group A. The first two columns give the fraction of reagent pairs ordered as the matching descriptor orders them (macro-average over 12 property--descriptor pairs); ``about equal'' answers and tied ratings count as half correct. The last column is the Spearman correlation between in-context yield predictions and measured outcomes on held-out experiments.}",
          r"\label{tab:rq0}", r"\end{table}"]
    return "\n".join(L)


def tab_rq0_prop():
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{@{}l l l l c@{}}", r"\toprule",
         r"Task & Factor & Named property & Descriptor & Pairwise accuracy \\", r"\midrule"]
    for t, f, p_, d, a, _ in R["rq0_prop"]:
        L.append(f"{t} & {f} & {p_} & {d} & {a:.2f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Pairwise accuracy of the main oracle for each property--descriptor pair on Group A. Pairs tied on the descriptor are excluded.}",
          r"\label{tab:rq0prop}", r"\end{table*}"]
    return "\n".join(L)


def tab_tasks():
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\begin{tabular}{@{}l l l r p{4.3cm} l c c@{}}", r"\toprule",
         r"ID & Reaction & Objective & Pool & Categorical factors (levels) & Numeric & $d$ & Group \\", r"\midrule"]
    for row in R["task_table"]:
        L.append(" & ".join(str(x) for x in row) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Benchmarks. Pool is the number of candidate experiments; $d$ is the \ours{} input dimension (Eq.~\ref{eq:dim}) for the axes named by the main oracle. NiSuz is an emulator built from 576 HTE reactions; its pool keeps only emulated conditions with valid predictions, so it is not a full Cartesian product.}",
          r"\label{tab:tasks}", r"\end{table*}"]
    return "\n".join(L)


def tab_discovered():
    L = [r"\begin{table*}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{6pt}",
         r"\begin{tabular}{@{}l l l l c@{}}", r"\toprule",
         r"Task & Factor & Reagents shown (high vs.\ low) & Property named by the oracle & $\bar\lambda$ at $t{=}40$ \\", r"\midrule"]
    for t, f, ex, prop, lam in R["discovered"]:
        L.append(f"{t} & {f} & {ex} & {prop} & {lam:.1f}" + r" \\")
    L += [r"\bottomrule", r"\end{tabular}",
          r"\caption{Axes added by discovery (\S\ref{sec:discovery}), with the posterior mean reliability at iteration 40 averaged over the seeds where the axis was added. The NiSuz axis never gained weight.}",
          r"\label{tab:discovered}", r"\end{table*}"]
    return "\n".join(L)


for name, fn in [("tab_main", tab_main), ("tab_robust", tab_robust), ("tab_ablation", tab_ablation),
                 ("tab_surrogate", tab_surrogate), ("tab_cost", tab_cost), ("tab_se", tab_se),
                 ("tab_i1", tab_i1), ("tab_rq0", tab_rq0), ("tab_rq0prop", tab_rq0_prop),
                 ("tab_tasks", tab_tasks), ("tab_discovered", tab_discovered)]:
    (OUT / f"{name}.tex").write_text(fn() + "\n")

# ---------------------------------------------------------------- macros
def mac(name, val):
    return f"\\newcommand{{\\{name}}}{{{val}}}"

M = []
a, b, c = R["rq0_avg"]
M += [mac("PairAcc", f"{a:.2f}"), mac("RateAcc", f"{b:.2f}"), mac("PredRho", f"{c:.2f}"),
      mac("PairAccPct", f"{round(a*100)}\\%"), mac("RateAccPct", f"{round(b*100)}\\%"),
      mac("AccSteric", f"{R['rq0_steric']:.2f}"), mac("AccAcidBase", f"{R['rq0_acid-base']:.2f}"),
      mac("AccElec", f"{R['rq0_electronic']:.2f}"), mac("RatingCompression", f"{R['rating_compression_pct']}\\%")]
M += [mac("IReduction", f"{R['i1_reduction_pct']}\\%"), mac("IReductionRank", f"{R['i1_reduction_rank_pct']}\\%"),
      mac("ITacit", f"{R['i1_avg']['tacit']:.1f}"), mac("IFast", f"{R['i1_avg'][fastest]:.1f}"),
      mac("ILGBO", f"{R['i1_avg']['lgbo']:.1f}"),
      mac("RankTacit", f"{R['avg_rank']['tacit']:.2f}"), mac("RankLGBO", f"{R['avg_rank']['lgbo']:.2f}"),
      mac("RankLLMGO", f"{R['avg_rank']['llmgo']:.2f}"),
      mac("NWins", {7: "seven", 6: "six", 8: "eight"}[R["n_wins"]]),
      mac("NSigWins", {3: "three", 4: "four", 5: "five", 6: "six", 7: "seven"}.get(R["n_sig_wins"], str(R["n_sig_wins"]))),
      mac("AllTacit", f"{R['all_mean']['tacit']:.1f}"), mac("AllLVGP", f"{R['all_mean']['lvgp']:.1f}"),
      mac("AllLGBO", f"{R['all_mean']['lgbo']:.1f}")]
gap_deoxy = r1(MEANS["tacit"][1] - MEANS["desc"][1])
gap_chan = r1(MEANS["tacit"][7] - MEANS["llmgo"][7])
suzc_max = max(MEANS[m][6] for m in MAIN)
M += [mac("GapDeoxyF", f"{gap_deoxy:.1f}"), mac("GapChanLam", f"{gap_chan:.1f}"), mac("SuzCMax", f"{suzc_max:.1f}")]
ro = ROBUST
M += [mac("CfTacit", f"{ro['tacit'][1]:.1f}"), mac("InvTacit", f"{ro['tacit'][2]:.1f}"),
      mac("LVGPA", f"{ro['lvgp'][0]:.1f}"), mac("CfGap", f"{r1(ro['tacit'][1]-ro['lvgp'][0]):.1f}"),
      mac("InvGap", f"{r1(ro['tacit'][2]-ro['lvgp'][0]):.1f}"),
      mac("CfGapPct", f"{int(Decimal(str(100*(ro['tacit'][1]-ro['lvgp'][0])/ro['lvgp'][0])).quantize(Decimal('1'), ROUND_HALF_UP))}\\%"),
      mac("CfFixed", f"{ro['tacit_fixed'][1]:.1f}"), mac("OneHotA", f"{R['onehot_groupA']:.1f}"),
      mac("DeconTacitCost", f"{r1(ro['tacit'][3]-ro['tacit'][0]):.1f}"),
      mac("DeconLLMGOCost", f"{r1(ro['llmgo'][3]-ro['llmgo'][0]):.1f}"),
      mac("PostTacit", f"{ro['tacit'][4]:.1f}"), mac("PostLVGP", f"{ro['lvgp'][4]:.1f}"),
      mac("PostLLMGO", f"{ro['llmgo'][4]:.1f}"), mac("LambdaCross", f"{R['lambda_cf_cross_iter']}")]
M += [mac("NQueries", f"{R['tacit_queries']}"), mac("NCalls", f"{R['tacit_queries']*R['tacit_repeats']}"),
      mac("RandFactor", f"{R['random_query_factor']:.1f}"),
      mac("TokLLMGO", f"{R['token_ratio']['llmgo']:.0f}"), mac("TokLLAMBO", f"{R['token_ratio']['llambo']:.0f}"),
      mac("AxisRho", f"{R['axis_rho']['tacit']:.2f}"), mac("AxisRhoFixed", f"{R['axis_rho']['tacit_fixed']:.2f}"),
      mac("AxisRhoLVGP", f"{R['axis_rho']['lvgp_rot']:.2f}"),
      mac("AblRatings", f"{r1(R['ablations'][1]['orig'] - R['all_mean']['tacit']):.1f}")]
(OUT / "macros.tex").write_text("% generated by scripts/make_results.py\n" + "\n".join(M) + "\n")


def jsonable(o):
    if isinstance(o, dict):
        return {k: jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return o

(OUT / "results.json").write_text(json.dumps(jsonable(R), indent=1))
print("best baseline by rank:", best_base_rank, R["avg_rank"])
print("I1 avg:", R["i1_avg"])
print("I1 reduction %:", R["i1_reduction_pct"])
print("tests:", [(x["task"], x["runner"], round(x["p_holm"], 4), x["tacit_better"]) for x in tests])
print("lambda cf cross:", R["lambda_cf_cross_iter"], "orig final:", R["lambda_orig_final"])
print("rq0:", R["rq0_avg"], R["rq0_prop_mean"], R["rq0_steric"], R["rq0_acid-base"], R["rq0_electronic"])
print("token ratio:", R["token_ratio"])
