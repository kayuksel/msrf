"""Paired Wilcoxon signed-rank tests + win/loss counts over the released per-dataset results.
Reproduces rebuttal Tables R2 (method comparisons) and R3 (combination gains).
  python3 run_significance.py
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def rows(fn):
    return [json.loads(l) for l in open(os.path.join(RESULTS, fn))]


def test(rs, a, b):
    d = np.array([r[a] - r[b] for r in rs if a in r and b in r])
    dd = d[d != 0]
    p = wilcoxon(dd).pvalue if len(dd) >= 5 else float("nan")
    print(f"{a:24s} vs {b:24s}: mean {d.mean():+.4f}  p={p:.3g}  "
          f"W/T/L={(d > 1e-9).sum()}/{(np.abs(d) <= 1e-9).sum()}/{(d < -1e-9).sum()}  n={len(d)}")


print("== Table R2: unified protocol (full archive) ==")
uni = rows("unified_protocol_results.jsonl")
for a, b in [("MSRF-1410", "catch22"), ("MSRF-1410", "MiniRocket"),
             ("MSRF+519", "MiniRocket"), ("MSRF*760", "MSRF+519"),
             ("MSRF*760", "MiniRocket"), ("MSRF*760", "QUANT"),
             ("MSRF*760", "Hydra"), ("MiniRocket", "Rocket10k-ppvmax"),
             ("MSRF*C2272", "Hydra"), ("MSRF*C2272", "MiniRocket"),
             ("MSRF*C2272", "MiniRocket-1512"), ("MSRF*C2272", "QUANT"),
             ("MSRF*C2272", "MultiRocket")]:
    test(uni, a, b)

print("\n== Table R3: combination gains (52-dataset subset) ==")
comp = rows("complementarity_results.jsonl")
for base in ("MiniRocket", "Rocket10k-ppvmax", "QUANT", "Hydra", "MultiRocket"):
    for add in ("+NC410", "+MSRF+519", "+MSRF*760"):
        if any(base + add in r for r in comp):
            test(comp, base + add, base)
