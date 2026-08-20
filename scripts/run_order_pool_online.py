"""Ablation: is the frozen encoder's ORDER-BLIND pooling costing accuracy?

MSRF+519 pools per-patch features with mean||std||max -- all permutation-invariant in the
patch axis, so the embedding cannot see temporal rearrangement above the patch scale. This
script appends three zero-parameter order-sensitive pools (see order_pool.py) and prices each
one separately on the archive under the repo's single unchanged protocol (official split,
StandardScaler, RidgeClassifierCV over 13 alphas).

One encode pass per dataset serves every arm: columns 0:519 of OrderPoolEnc are byte-identical
to the released encoder, and each arm is a column slice.

  base       519   mean||std||max                     (the released MSRF+519)
  +contrast  692   base + first-half/second-half contrast
  +earliness 692   base + arrival time of the extreme
  +cusum     692   base + CUSUM excursion range
  +all3     1038   base + all three
  order3     519   the three blocks ALONE (no base) -- do they carry standalone signal?

  python3 run_order_pool.py --cache ~/Desktop/worldmodel/ucr_cache

Writes/updates order_pool_online_results.jsonl. Incremental and resumable.
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

import ucr_io
from msrfc.msenc_encoder_np import patchify
from order_pool_online import OnlineOrderPoolEnc  # noqa
from order_pool import BLOCKS, cols
from run_rfm_head import load_ts

RESULTS = "order_pool_online_results.jsonl"
ARMS = [("base", ()), ("contrast", ("contrast",)), ("earliness", ("earliness",)),
        ("cusum", ("cusum",)), ("all3", BLOCKS)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    a = ap.parse_args()

    path = os.path.join(ucr_io.RESULTS, RESULTS)
    done = {json.loads(l)["dataset"] for l in open(path)} if os.path.exists(path) else set()
    enc = OnlineOrderPoolEnc()
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
        sec = time.time() - t0
        npat = len(patchify((Xtr[0] - Xtr[0].mean()) / (Xtr[0].std() + 1e-8)))

        row = dict(n_train=int(Xtr.shape[0]), T=int(Xtr.shape[1]), C=int(len(np.unique(ytr))),
                   n_patch=int(npat), sec_enc=round(sec, 2))
        for nm, blocks in ARMS:
            c = cols(nc, *blocks)
            row[nm] = ucr_io.clf(Etr[:, c], ytr, Ete[:, c], yte)
        c = cols(nc, *BLOCKS, base=False)
        row["order3"] = ucr_io.clf(Etr[:, c], ytr, Ete[:, c], yte)

        ucr_io.update_results(name, row, results_file=RESULTS)
        print(f"{name:28s} N={row['n_train']:5d} T={row['T']:5d} np={npat:3d} | "
              + "  ".join(f"{nm} {row[nm]:.3f}" for nm, _ in ARMS)
              + f"  | order3 {row['order3']:.3f}  ({sec:.1f}s)", flush=True)


if __name__ == "__main__":
    main()
