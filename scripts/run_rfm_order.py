"""Do order pooling and the nonlinear head close the SAME gap?

Two independent findings sit on the identical frozen 519-d features and are the same size:
a leaf-RFM readout buys +0.0125 (paper's ablation), and order-sensitive pooling buys +0.0098.
Both could be recovering the same missing information -- temporal arrangement, which a
permutation-invariant pool destroys and which a nonlinear head can partly reconstruct from
interactions among the surviving symmetric statistics. Or they could be independent.

The question is decidable and the answer changes what the paper may claim:

  rfm@1038 ~= rfm@519    -> SAME gap. Most of what the nonlinear head bought was order
                            information, and a linear head recovers it once pooling stops
                            discarding it. "Linearity costs 0.013" becomes
                            "a pooling defect cost 0.013; linearity costs what is left."
  rfm@1038 ~= 519+0.0125 -> INDEPENDENT. Order pooling and nonlinearity are additive, no
                            ratio may be quoted, and the linearity price stands as published.

Four readouts, identical splits, identical scaling, identical support caps:

  ridge_519 / rfm_519     the paper's ablation, recomputed here for a paired comparison
  ridge_1038 / rfm_1038   the same two heads on the order-augmented (fine-grid) features

  python3 run_rfm_order.py --cache ~/Desktop/worldmodel/ucr_cache
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
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # must precede the numba import
from sklearn.preprocessing import StandardScaler

import ucr_io
from order_pool import FineOrderPoolEnc, cols
from run_rfm_head import acc_ridge, acc_rfm, load_ts

RESULTS = "rfm_order_results.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    ap.add_argument("--support-cap", type=int, default=1024)
    a = ap.parse_args()

    path = os.path.join(ucr_io.RESULTS, RESULTS)
    done = {json.loads(l)["dataset"] for l in open(path)} if os.path.exists(path) else set()
    enc = FineOrderPoolEnc()
    nc = enc.n_cols
    slices = {"519": cols(nc), "1038": cols(nc, "contrast", "earliness", "cusum")}
    for name in ucr_io.dataset_names():
        if name in done:
            continue
        try:
            Xtr, ytr, Xte, yte = load_ts(name, a.cache)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        C = int(len(np.unique(ytr)))
        Etr, Ete = enc.transform(Xtr), enc.transform(Xte)
        row = dict(n_train=int(Xtr.shape[0]), C=C)
        for tag, c in slices.items():
            # scaler fit on train only, per slice -- same as the paper's ablation
            sc = StandardScaler().fit(Etr[:, c])
            Ztr, Zte = sc.transform(Etr[:, c]), sc.transform(Ete[:, c])
            row[f"ridge_{tag}"] = acc_ridge(Ztr, ytr, Zte, yte)
            t0 = time.time()
            row[f"rfm_{tag}"], info = acc_rfm(Ztr, ytr, Zte, yte, C, a.support_cap)
            row[f"sec_{tag}"] = round(time.time() - t0, 1)
            row[f"capped_{tag}"] = info["capped"]
        ucr_io.update_results(name, row, results_file=RESULTS)
        print(f"{name:28s} N={row['n_train']:5d} C={C:2d} | ridge {row['ridge_519']:.3f}"
              f"->{row['ridge_1038']:.3f}  rfm {row['rfm_519']:.3f}->{row['rfm_1038']:.3f}  "
              f"({row['sec_519']+row['sec_1038']:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
