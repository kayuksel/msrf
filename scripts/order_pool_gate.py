"""When is it safe to spend dimensions on order-sensitive pooling?

The coarse-grid ablation has two clean failure modes, and both are visible WITHOUT test labels:
  1. too few patches   -- nothing to pool over (n_patch small)
  2. too few examples  -- +173..519 dims on a small, many-class train set overfits
                          (the Pig* datasets: n_train=104, C=52, delta ~ -0.15)
This scores both predictors against the realised delta and reports the accuracy of an a-priori
gate that spends the extra dimensions only where both conditions hold.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
grid = sys.argv[1] if len(sys.argv) > 1 else "coarse"
f = "order_pool_results.jsonl" if grid == "coarse" else "order_pool_fine_results.jsonl"
rows = sorted({json.loads(l)["dataset"]: json.loads(l) for l in open(os.path.join(R, f))}.values(),
              key=lambda r: r["dataset"])
base = np.array([r["base"] for r in rows])
npat = np.array([r["n_patch"] for r in rows])
ntr = np.array([r["n_train"] for r in rows])
C = np.array([r["C"] for r in rows])
per_class = ntr / C

print(f"=== {grid} grid, {len(rows)} datasets ===\n")
for arm, extra in [("contrast", 173), ("all3", 519)]:
    d = np.array([r[arm] for r in rows]) - base
    print(f"{arm} (+{extra} dims):  overall {d.mean():+.4f}")
    for nm, v in [("n_patch", npat), ("n_train", ntr), ("n_train/C", per_class),
                  ("n_train/extra_dims", ntr / extra)]:
        rho, p = spearmanr(v, d)
        print(f"    spearman({nm:<18}, delta) rho={rho:+.3f} p={p:.3g}")
    # a-priori gate: enough patches to see order AND enough examples to afford the dims
    for min_pat, min_pc in [(9, 0), (9, 10), (9, 20), (16, 10)]:
        g = (npat >= min_pat) & (per_class >= min_pc)
        gated = np.where(g, base + d, base)
        dd = gated - base
        p = wilcoxon(gated, base).pvalue if (dd != 0).any() else 1.0
        print(f"    gate n_patch>={min_pat:2d}, n_train/C>={min_pc:2d}: fires on {g.sum():3d}"
              f"  mean {gated.mean():.4f}  delta {dd.mean():+.4f}  p={p:.3g}")
    print()
