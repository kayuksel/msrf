"""The adaptation ladder: fixed pooling vs a label-free rule vs per-dataset CV vs a test oracle.

Reproduces the four numbers quoted in the paper from the shipped result files, so none of them
depends on an ad-hoc inline computation. Reads:
  order_pool_fine_results.jsonl  (per-dataset accuracy of each pooling arm, and n_patch/n_train/C)
  pool_select_results.jsonl      (inner-CV score of each arm on the TRAINING split, plus test acc)

  fixed        one pooling choice for every dataset
  gate         apply the blocks iff (n_patch >= P) and (n_train/C >= Q); both quantities are known
               without any labels. P,Q are scanned here over a small grid -- that scan is in-sample,
               and the split-half estimate of its optimism is reported alongside.
  cv_select    the arm with the best stratified k-fold accuracy on the training split. CV cannot
               separate the arms on ~20% of datasets, so the tie-break is resolved randomly and the
               reported value is the mean +- sd over many draws rather than one lucky convention.
  oracle       per-dataset best on TEST -- an upper bound, not a method. Reported over the same
               arm set the selector chose from; an oracle over more arms is not its ceiling.

  python3 order_pool_ladder.py
"""
import json
import os

import numpy as np
from scipy.stats import wilcoxon

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
ARMS = ("base", "contrast", "earliness", "cusum", "all3")
SEEDS = 2000
GRID_P = (2, 3, 4)
GRID_Q = (4, 5, 6, 8, 10)


def load(f):
    return {json.loads(l)["dataset"]: json.loads(l) for l in open(os.path.join(RES, f))}


def main():
    fine, sel = load("order_pool_fine_results.jsonl"), load("pool_select_results.jsonl")
    ds = sorted(set(fine) & set(sel))
    base = np.array([fine[d]["base"] for d in ds])
    all3 = np.array([fine[d]["all3"] for d in ds])
    npat = np.array([fine[d]["n_patch"] for d in ds])
    perc = np.array([fine[d]["n_train"] / fine[d]["C"] for d in ds])
    print(f"n = {len(ds)}   base (519-d) = {base.mean():.4f}\n")
    print(f"  fixed  (all3 everywhere)        {all3.mean():.4f}   {all3.mean()-base.mean():+.4f}"
          f"   p={wilcoxon(all3, base).pvalue:.2g}")

    best = max(((np.where((npat >= P) & (perc >= Q), all3, base).mean(), P, Q)
                for P in GRID_P for Q in GRID_Q), key=lambda t: t[0])
    _, P, Q = best
    gate = np.where((npat >= P) & (perc >= Q), all3, base)
    print(f"  gate   (n_patch>={P}, n/C>={Q})       {gate.mean():.4f}   {gate.mean()-base.mean():+.4f}"
          f"   p={wilcoxon(gate, base).pvalue:.2g}   [thresholds scanned in-sample]")

    rng = np.random.RandomState(0)
    cvb = np.array([sel[d]["cv_base"] for d in ds], float)
    cva = np.array([sel[d]["cv_all3"] for d in ds], float)
    teb = np.array([sel[d]["te_base"] for d in ds])
    tea = np.array([sel[d]["te_all3"] for d in ds])
    tie = np.isclose(cvb, cva) | ~(np.isfinite(cvb) & np.isfinite(cva))
    dec = np.where(tie, np.nan, np.where(cva > cvb, 1.0, 0.0))
    draws = np.array([np.where(np.isnan(dec), rng.randint(0, 2, len(ds)), dec) for _ in range(SEEDS)])
    accs = np.where(draws == 1, tea, teb).mean(1)
    print(f"  cv     (train-split, {SEEDS} tie-breaks) {accs.mean():.4f} +- {accs.std():.4f}"
          f"   {accs.mean()-base.mean():+.4f}   [CV cannot separate on {int(tie.sum())}/{len(ds)}]")

    # the oracle must be over the SAME arm set the selector chose from, or the comparison is unfair
    orc2 = np.maximum(teb, tea)
    orc5 = np.array([max(sel[d][f"te_{a}"] for a in ARMS) for d in ds])
    print(f"  oracle (TEST, same 2 arms)      {orc2.mean():.4f}   {orc2.mean()-base.mean():+.4f}"
          f"   [upper bound for the cv row above]")
    print(f"  oracle (TEST, all {len(ARMS)} arms)      {orc5.mean():.4f}   {orc5.mean()-base.mean():+.4f}"
          f"   [upper bound for a 5-way selector]")

    # split-half optimism of the gate's in-sample threshold scan
    opt = []
    for s in range(400):
        r = np.random.RandomState(s)
        idx = r.permutation(len(ds))
        a, b = idx[: len(ds) // 2], idx[len(ds) // 2:]
        bp = max(((np.where((npat[a] >= P2) & (perc[a] >= Q2), all3[a], base[a]).mean(), P2, Q2)
                  for P2 in GRID_P for Q2 in GRID_Q), key=lambda t: t[0])
        _, P2, Q2 = bp
        g = np.where((npat[b] >= P2) & (perc[b] >= Q2), all3[b], base[b])
        opt.append(g.mean() - base[b].mean())
    print(f"\n  gate, split-half out-of-sample  {np.mean(opt):+.4f}"
          f"   (in-sample {gate.mean()-base.mean():+.4f}; optimism {gate.mean()-base.mean()-np.mean(opt):+.4f})")


if __name__ == "__main__":
    main()
