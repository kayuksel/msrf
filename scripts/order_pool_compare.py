"""Coarse vs fine patch grid for the order-sensitive pools, against the released MSRF+519.

The coarse-grid ablation loses accuracy at n_patch<=8 and wins at n_patch 9-24, which says the
threshold is a property of the patch grid rather than of the data. This compares the two grids
on identical base columns, per n_patch stratum.
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
ARMS = ["contrast", "earliness", "cusum", "all3"]


def load(f):
    return {json.loads(l)["dataset"]: json.loads(l) for l in open(os.path.join(R, f))}


co, fi = load("order_pool_results.jsonl"), load("order_pool_fine_results.jsonl")
ds = sorted(set(co) & set(fi))
base = np.array([co[d]["base"] for d in ds])
npat = np.array([co[d]["n_patch"] for d in ds])
G = {"coarse": {k: np.array([co[d][k] for d in ds]) for k in ARMS + ["order3"]},
     "fine": {k: np.array([fi[d][k] for d in ds]) for k in ARMS + ["order3"]}}

print(f"=== {len(ds)} UCR datasets | order pools on the released vs a >=24-patch grid ===")
print(f"    base (MSRF+519) = {base.mean():.4f}\n")
print(f"  {'arm':<10} {'grid':<7} {'mean':<8} {'delta':<9} {'W/L/T':<14} p")
for k in ARMS + ["order3"]:
    for g in ("coarse", "fine"):
        x = G[g][k]
        d = x - base
        w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
        p = wilcoxon(x, base).pvalue if w + l else 1.0
        print(f"  {k:<10} {g:<7} {x.mean():.4f}   {d.mean():+.4f}   "
              f"{w:3d}/{l:3d}/{t:3d}     p={p:.3g}")
    print()

print(f"{'stratum':<18} {'n':>4}  " + "  ".join(f"{a[:8]:>17}" for a in ARMS))
print(f"{'':<18} {'':>4}  " + "  ".join(f"{'coarse':>8}{'fine':>9}" for _ in ARMS))
for lo, hi, nm in [(0, 2, "n_patch<=2"), (3, 8, "n_patch 3-8"),
                   (9, 24, "n_patch 9-24"), (25, 10**9, "n_patch>=25")]:
    m = (npat >= lo) & (npat <= hi)
    if not m.any():
        continue
    print(f"{nm:<18} {m.sum():>4}  " + "  ".join(
        f"{(G['coarse'][a][m] - base[m]).mean():>+8.4f}{(G['fine'][a][m] - base[m]).mean():>+9.4f}"
        for a in ARMS))

print("\nbest configuration:")
cand = [(f"{a} ({g} grid)", G[g][a]) for a in ARMS for g in ("coarse", "fine")]
for nm, x in sorted(cand, key=lambda kv: -(kv[1] - base).mean())[:4]:
    d = x - base
    print(f"  {nm:<26} {x.mean():.4f}  {d.mean():+.4f}  p={wilcoxon(x, base).pvalue:.3g}")
print("\nreference: MSRF+519 0.8240 (519d)  MSRF*760 0.8319 (760d)  "
      "MSRF*C2272 0.8624  MultiRocket 0.8670")
