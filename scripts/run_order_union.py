"""Does order-sensitive pooling lift the HEADLINE classifier, or only the encoder?

run_order_pool_fine.py improves the encoder component (MSRF+519 -> 0.8342). The paper's headline
is MSRF*C2272 = MiniRocket-1512 (+) MSRF*760, at 0.8624. MiniRocket pools its convolution
responses with PPV over the whole series, which is ALSO permutation-invariant in time -- so the
order blocks may be complementary to it rather than redundant. Untested until now, and it is the
only thing that decides whether the paper's headline number moves.

  mrfc        2272   MiniRocket-1512 (+) [Z519 (+) battery241]      (the published headline)
  mrfc_order  2791   the above (+) the 519 order-block columns

Identical protocol; the MiniRocket seed and kernel count are the published ones.

  python3 run_order_union.py --cache ~/Desktop/worldmodel/ucr_cache
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

from sklearn.preprocessing import StandardScaler

import ucr_io
from msrfc.msrf_battery import battery, make_banks
from order_pool import BLOCKS, FineOrderPoolEnc, cols
from run_rfm_head import load_ts

RESULTS = "order_union_results.jsonl"


def star(Etr, Ete):
    """msrf_star's transform, but on features already computed (no second encode pass)."""
    sc = StandardScaler().fit(Etr)
    Ztr, Zte = sc.transform(Etr), sc.transform(Ete)
    bk = make_banks(Ztr.shape[1])
    return (np.concatenate([Ztr, battery(Ztr, bk)], 1),
            np.concatenate([Zte, battery(Zte, bk)], 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    ap.add_argument("--n-kernels", type=int, default=1512)
    a = ap.parse_args()
    from aeon.transformations.collection.convolution_based import MiniRocket

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
        b = cols(nc)                                   # base 519
        o = cols(nc, *BLOCKS, base=False)              # order blocks 519
        Str, Ste = star(Etr[:, b], Ete[:, b])          # -> 760 each

        mr = MiniRocket(n_kernels=a.n_kernels, random_state=0)
        Mtr = np.nan_to_num(np.asarray(mr.fit_transform(Xtr[:, None, :])))
        Mte = np.nan_to_num(np.asarray(mr.transform(Xte[:, None, :])))
        sec = time.time() - t0

        row = dict(n_train=int(Xtr.shape[0]), T=int(Xtr.shape[1]),
                   C=int(len(np.unique(ytr))), sec=round(sec, 1))
        row["mrfc"] = ucr_io.clf(np.hstack([Mtr, Str]), ytr, np.hstack([Mte, Ste]), yte)
        row["mrfc_order"] = ucr_io.clf(np.hstack([Mtr, Str, Etr[:, o]]), ytr,
                                       np.hstack([Mte, Ste, Ete[:, o]]), yte)
        ucr_io.update_results(name, row, results_file=RESULTS)
        print(f"{name:28s} N={row['n_train']:5d} | mrfc {row['mrfc']:.4f}  "
              f"mrfc_order {row['mrfc_order']:.4f}  ({row['mrfc_order']-row['mrfc']:+.4f})  "
              f"{sec:.0f}s", flush=True)


if __name__ == "__main__":
    main()
