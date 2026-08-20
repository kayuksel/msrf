"""Summarise order_pool_results.jsonl.

Reports each arm's paired delta vs the released MSRF+519 base, and then tests the MECHANISM:
the order-sensitive pools can only carry information when there is order to see, i.e. when the
patch sequence is long. If the gain is real it must concentrate at large n_patch and vanish at
n_patch<=2 (T<=~96), where the blocks are constant or near-degenerate. A flat profile in
n_patch would mean any aggregate gain is noise, not the permutation-invariance mechanism.
"""
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr, wilcoxon

P = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results",
                 "order_pool_results.jsonl")
rows = sorted({json.loads(l)["dataset"]: json.loads(l) for l in open(P)}.values(),
              key=lambda r: r["dataset"])
ARMS = ["base", "contrast", "earliness", "cusum", "all3", "order3"]
A = {k: np.array([r[k] for r in rows]) for k in ARMS}
npat = np.array([r["n_patch"] for r in rows])
n = len(rows)


def paired(x, b, label):
    d = x - b
    w, l, t = int((d > 0).sum()), int((d < 0).sum()), int((d == 0).sum())
    p = wilcoxon(x, b).pvalue if w + l else 1.0
    print(f"  {label:<12} mean {x.mean():.4f}  delta {d.mean():+.4f}  "
          f"W/L/T {w:3d}/{l:3d}/{t:3d}  p={p:.3g}")


print(f"=== {n} UCR datasets, frozen MSRF+ features, order-sensitive pooling ===\n")
print(f"  {'arm':<12} {'mean':<7}  {'vs base':<15} {'W/L/T':<16} p")
print(f"  {'base':<12} {A['base'].mean():.4f}  (released MSRF+519)")
for k in ARMS[1:]:
    paired(A[k], A["base"], k)

print("\nmechanism -- gain must concentrate where the patch sequence is long:")
for k in ["contrast", "earliness", "cusum", "all3"]:
    d = A[k] - A["base"]
    rho, pr = spearmanr(npat, d)
    print(f"  {k:<12} spearman(n_patch, delta) rho={rho:+.3f} p={pr:.3g}")

print(f"\n{'stratum':<22} {'n':>4} " + " ".join(f"{k:>10}" for k in ARMS[1:5]))
for lo, hi, nm in [(0, 2, "n_patch<=2"), (3, 8, "n_patch 3-8"),
                   (9, 24, "n_patch 9-24"), (25, 10**9, "n_patch>=25")]:
    m = (npat >= lo) & (npat <= hi)
    if not m.any():
        continue
    print(f"{nm:<22} {m.sum():>4} " + " ".join(
        f"{(A[k][m] - A['base'][m]).mean():>+10.4f}" for k in ARMS[1:5]))

best = max(ARMS[1:5], key=lambda k: (A[k] - A["base"])[npat >= 9].mean())
m = npat >= 9
d = (A[best] - A["base"])[m]
p = wilcoxon(A[best][m], A["base"][m]).pvalue
print(f"\nbest arm on the n_patch>=9 subset ({m.sum()} datasets): {best} "
      f"{d.mean():+.4f}  p={p:.3g}")
print(f"published reference points: MSRF+519 0.8240  MSRF*760 0.8319  "
      f"MSRF*C2272 0.8624  MultiRocket 0.8670")
