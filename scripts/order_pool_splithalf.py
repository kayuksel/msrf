"""Split-half validation of the a-priori gate.

The gate (n_patch >= A, n_train/C >= B) has two thresholds. Choosing them on all 113 datasets
and reporting the resulting delta is optimistic. Here A,B are selected on a random half of the
archive and the delta is measured on the held-out half, over many splits. The reported number is
the out-of-sample gain; the in-sample one is printed alongside to show the selection bias.
"""
import json
import os
import sys

import numpy as np

R = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
grid = sys.argv[1] if len(sys.argv) > 1 else "coarse"
f = "order_pool_results.jsonl" if grid == "coarse" else "order_pool_fine_results.jsonl"
rows = sorted({json.loads(l)["dataset"]: json.loads(l) for l in open(os.path.join(R, f))}.values(),
              key=lambda r: r["dataset"])
base = np.array([r["base"] for r in rows])
npat = np.array([r["n_patch"] for r in rows])
pc = np.array([r["n_train"] for r in rows]) / np.array([r["C"] for r in rows])
GRID = [(a, b) for a in (0, 4, 9, 16, 25) for b in (0, 5, 10, 20, 40)]
n = len(rows)

print(f"=== split-half validation of the gate, {grid} grid, {n} datasets, 400 splits ===\n")
print(f"  {'arm':<10} {'in-sample':>10} {'out-of-sample':>15} {'ungated':>10}   modal gate")
for arm in ("contrast", "all3"):
    d = np.array([r[arm] for r in rows]) - base
    ins, oos, picks = [], [], {}
    for s in range(400):
        rng = np.random.RandomState(s)
        p = rng.permutation(n)
        A, B = p[: n // 2], p[n // 2:]
        best = max(GRID, key=lambda g: np.where((npat[A] >= g[0]) & (pc[A] >= g[1]), d[A], 0).mean())
        ins.append(np.where((npat[A] >= best[0]) & (pc[A] >= best[1]), d[A], 0).mean())
        oos.append(np.where((npat[B] >= best[0]) & (pc[B] >= best[1]), d[B], 0).mean())
        picks[best] = picks.get(best, 0) + 1
    modal = max(picks, key=picks.get)
    print(f"  {arm:<10} {np.mean(ins):>+10.4f} {np.mean(oos):>+15.4f} {d.mean():>+10.4f}   "
          f"n_patch>={modal[0]}, n_train/C>={modal[1]} ({100*picks[modal]/400:.0f}% of splits)")
print("\n  out-of-sample is the number to quote: thresholds fit on one half, delta measured on the other.")
