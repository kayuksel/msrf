"""Regenerate the three README figures from the released per-dataset results.
  python3 make_figures.py            # writes ../plots/fig_*.png
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")
PLOTS = os.path.join(HERE, "..", "plots")
os.makedirs(PLOTS, exist_ok=True)

uni = [json.loads(l) for l in open(os.path.join(RESULTS, "unified_protocol_results.jsonl"))]
comp = [json.loads(l) for l in open(os.path.join(RESULTS, "complementarity_results.jsonl"))]

# (label, results key, dims at length 256, transform ms/series by run_runtime.py, single core)
METHODS = [
    ("MultiRocket", "MultiRocket", 49728, 6.3),
    ("MSRF*C", "MSRF*C2272", 2272, 0.28),
    ("MiniRocket", "MiniRocket", 9996, 0.40),
    ("Hydra", "Hydra", 5120, 5.2),
    ("MSRF*", "MSRF*760", 760, 0.19),
    ("MSRF+", "MSRF+519", 519, 0.11),
    ("Rocket-10k", "Rocket10k-ppvmax", 10000, 5.0),
    ("MSRF-1410", "MSRF-1410", 1410, 1.3),
    ("QUANT", "QUANT", 2253, 0.20),
    ("catch22", "catch22", 22, 0.31),
]
KEYS = [k for _, k, _, _ in METHODS]
cc = [r for r in uni if all(k in r for k in KEYS)]
print(f"complete-case n={len(cc)}")
acc = {k: float(np.mean([r[k] for r in cc])) for k in KEYS}
for lbl, k, d, _ in METHODS:
    print(f"{lbl:12s} dims={d:6d} acc={acc[k]:.4f}")

# ---- fig 1: accuracy vs dimension (Pareto frontier) ----
fig, ax = plt.subplots(figsize=(7.2, 4.6))
pts = [(d, acc[k], lbl, ms) for lbl, k, d, ms in METHODS]
ours = {"MSRF*", "MSRF+", "MSRF-1410", "MSRF*C"}
# (x multiplier, y offset, ha) per label, tuned to avoid collisions
OFF = {"MultiRocket": (0.95, 0.004, "right"), "MiniRocket": (1.12, -0.009, "left"),
       "Hydra": (1.15, -0.008, "left"), "MSRF*": (1.18, 0.002, "left"),
       "MSRF+": (0.85, -0.020, "right"), "Rocket-10k": (1.15, -0.004, "left"),
       "MSRF-1410": (0.85, -0.018, "right"), "QUANT": (1.15, -0.010, "left"),
       "catch22": (1.25, 0.004, "left"), "MSRF*C": (0.80, 0.005, "right")}
for d, a, lbl, ms in pts:
    c = "#c62828" if lbl in ours else "#37474f"
    ax.scatter(d, a, s=70, color=c, zorder=3)
    note = f"{lbl}" + (f"\n{ms} ms/series" if ms else "")
    mx, dy, ha = OFF.get(lbl, (1.12, 0.006, "left"))
    ax.annotate(note, (d, a), xytext=(d * mx, a + dy), fontsize=8,
                color=c, ha=ha, va="bottom" if dy > 0 else "top")
front = []
for d, a, lbl, _ in sorted(pts):
    if not front or a > front[-1][1]:
        front.append((d, a))
ax.step([d for d, _ in front], [a for _, a in front], where="post",
        color="#c62828", alpha=0.35, lw=2, zorder=1)
ax.set_xscale("log")
ax.set_xlabel("feature dimensions (log scale)")
ax.set_ylabel(f"mean accuracy ({len(cc)} UCR datasets, one protocol)")
ax.grid(alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_pareto.png"), dpi=160)

# ---- fig 2: combination gains ----
BASES = ["MiniRocket", "Rocket10k-ppvmax", "QUANT", "Hydra", "MultiRocket"]
ADDS = [("+NC410", "#90a4ae"), ("+MSRF+519", "#ef9a9a"), ("+MSRF*760", "#c62828")]
fig, ax = plt.subplots(figsize=(7.2, 4.2))
w = 0.26
for j, (add, color) in enumerate(ADDS):
    xs, ys, ps = [], [], []
    for i, base in enumerate(BASES):
        pairs = [(r[base], r[base + add]) for r in comp if base in r and base + add in r]
        if not pairs:
            continue
        d = np.array([b - a for a, b in pairs])
        xs.append(i + (j - 1) * w)
        ys.append(d.mean())
        dd = d[d != 0]
        ps.append(wilcoxon(dd).pvalue if len(dd) >= 5 else np.nan)
    ax.bar(xs, ys, width=w, color=color, label=add.replace("+NC410", "+non-conv-410 (submission)"))
    for x, y, p in zip(xs, ys, ps):
        if np.isfinite(p) and p < 0.05:
            ax.annotate(f"p={p:.1g}", (x, y), ha="center", fontsize=7,
                        xytext=(x, y + 0.0006), fontweight="bold")
ax.axhline(0, color="k", lw=0.8)
ax.set_xticks(range(len(BASES)))
ax.set_xticklabels([b.replace("Rocket10k-ppvmax", "Rocket-10k\n(ppv+max)") for b in BASES])
ax.set_ylabel("mean accuracy gain when appended\n(52-dataset subset)")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_complementarity.png"), dpi=160)

# ---- fig 3: budget scaling + the submission's own saturation trend ----
fig, (a1, a2) = plt.subplots(1, 2, figsize=(8.6, 3.6))
dims = [705, 1410, 2820]            # 0.5x / 1x / 2x, all spaces scaled together (52-dataset subset)
accs = [0.8507, 0.8554, 0.8615]     # measured, otzC ablation (a)
a1.plot(dims, accs, "o-", color="#c62828")
for d, a in zip(dims, accs):
    a1.annotate(f"{a:.4f}", (d, a), xytext=(d, a + 0.001), ha="center", fontsize=8)
a1.set_xscale("log")
a1.set_xticks(dims)
a1.set_xticklabels(dims)
a1.set_xlabel("MSRF total dimensions")
a1.set_ylabel("mean accuracy (52-dataset subset)")
a1.set_title("budget scaling: ±0.005 per halving/doubling", fontsize=9)
a1.grid(alpha=0.25)
budgets = ["10k", "20k", "60k"]     # submission Sec. 5.3: +MSRF wins/ties/losses vs conv budget
wins, losses = [67, 43, 29], [14, 18, 13]
x = np.arange(3)
a2.bar(x - 0.17, wins, 0.34, color="#c62828", label="wins")
a2.bar(x + 0.17, losses, 0.34, color="#90a4ae", label="losses")
a2.set_xticks(x)
a2.set_xticklabels(budgets)
a2.set_xlabel("convolutional feature budget")
a2.set_ylabel("datasets improved by +MSRF (submission)")
a2.set_title("complementarity fades as budget grows\n(submission Sec. 5.3)", fontsize=9)
a2.legend(fontsize=8)
a2.grid(axis="y", alpha=0.25)
fig.tight_layout()
fig.savefig(os.path.join(PLOTS, "fig_scaling.png"), dpi=160)
print("figures written to", PLOTS)
