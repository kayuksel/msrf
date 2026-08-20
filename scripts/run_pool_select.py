"""Is per-dataset pooling selection worth it? The legitimate version.

The test-set oracle over the 5 pooling arms is +0.0248 -- 2.5x the best fixed arm. But an oracle
is not a method. The realisable rule selects the arm by cross-validation on the TRAINING split
only (exactly what RidgeClassifierCV already does for alpha) and is then scored once on test.
If the realised gain falls short of the fixed arm, the oracle gap was selection noise and
per-dataset pooling buys nothing while costing the paper its universality claim.

  fixed_all3   the best single arm, same pooling for every dataset
  cv_select    arm chosen per dataset by stratified k-fold CV on train, then scored on test
  oracle       per-dataset best on TEST -- upper bound, not a method

  python3 run_pool_select.py --cache ~/Desktop/worldmodel/ucr_cache
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

from sklearn.model_selection import StratifiedKFold

import ucr_io
from order_pool import BLOCKS, FineOrderPoolEnc, cols
from run_order_pool import ARMS
from run_rfm_head import load_ts

RESULTS = "pool_select_results.jsonl"


def cv_acc(Z, y, k=5, seed=0):
    """Inner CV accuracy on the training split, using the SAME head as the outer protocol."""
    _, cnt = np.unique(y, return_counts=True)
    k = int(min(k, cnt.min()))
    if k < 2:
        return np.nan
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    return float(np.mean([ucr_io.clf(Z[a], y[a], Z[b], y[b]) for a, b in skf.split(Z, y)]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    ap.add_argument("--folds", type=int, default=5)
    a = ap.parse_args()

    path = os.path.join(ucr_io.RESULTS, RESULTS)
    done = {json.loads(l)["dataset"] for l in open(path)} if os.path.exists(path) else set()
    enc = FineOrderPoolEnc()
    nc = enc.n_cols
    for name in ucr_io.dataset_names():
        if name in done:
            continue
        try:
            Xtr, ytr, Xte, yte = load_ts(name, a.cache)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        t0 = time.time()
        Etr, Ete = enc.transform(Xtr), enc.transform(Xte)
        row = dict(n_train=int(Xtr.shape[0]), C=int(len(np.unique(ytr))))
        for nm, blocks in ARMS:
            c = cols(nc, *blocks)
            row[f"cv_{nm}"] = cv_acc(Etr[:, c], ytr, a.folds)
            row[f"te_{nm}"] = ucr_io.clf(Etr[:, c], ytr, Ete[:, c], yte)
        names = [nm for nm, _ in ARMS]
        cvs = np.array([row[f"cv_{n}"] for n in names], dtype=float)
        pick = names[int(np.nanargmax(cvs))] if np.isfinite(cvs).any() else "all3"
        row["pick"] = pick
        row["cv_select"] = row[f"te_{pick}"]
        row["oracle"] = max(row[f"te_{n}"] for n in names)
        row["sec"] = round(time.time() - t0, 1)
        ucr_io.update_results(name, row, results_file=RESULTS)
        print(f"{name:28s} pick={pick:10s} cv_select {row['cv_select']:.4f}  "
              f"fixed_all3 {row['te_all3']:.4f}  oracle {row['oracle']:.4f}  ({row['sec']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
