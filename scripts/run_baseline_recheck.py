"""Re-measure MSRF+519 and MSRF*760 in THIS environment, so order-pooling comparisons are within-run.

The published per-dataset column (results/unified_protocol_results.jsonl, key MSRF+519) was produced
on a different library stack. The encoder is deterministic and its features are bit-identical here,
but RidgeClassifierCV's alpha selection is not stable across sklearn versions, so 24 of 113 datasets
land on a different alpha and the archive mean re-measures at 0.8244 instead of 0.8240.

Every order-pooling delta is paired within a single run, so those are unaffected. But the Pareto
claim -- "the CUSUM block reaches X at 692 dims, better than the 760-d composed encoder at 0.8319"
-- compares a number measured HERE against a number measured THERE, which is not sound. This script
measures the 760-d encoder here, so the claim can be made within-run or dropped.

  python3 run_baseline_recheck.py --cache ~/Desktop/worldmodel/ucr_cache
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
from sklearn.preprocessing import StandardScaler

import ucr_io
from msrfc import MultiSpaceEncCore, battery, make_banks
from run_rfm_head import load_ts

RESULTS = "baseline_recheck_results.jsonl"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    a = ap.parse_args()
    path = os.path.join(ucr_io.RESULTS, RESULTS)
    done = {json.loads(l)["dataset"] for l in open(path)} if os.path.exists(path) else set()
    enc = MultiSpaceEncCore()
    banks = None
    for name in ucr_io.dataset_names():
        if name in done:
            continue
        try:
            Xtr, ytr, Xte, yte = load_ts(name, a.cache)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        Etr, Ete = enc.transform(Xtr), enc.transform(Xte)
        if banks is None:
            banks = make_banks(Etr.shape[1])
        sc = StandardScaler().fit(Etr)
        Ztr, Zte = sc.transform(Etr), sc.transform(Ete)
        Str = np.concatenate([Etr, battery(Ztr, banks)], 1)      # 519 + 241 = 760
        Ste = np.concatenate([Ete, battery(Zte, banks)], 1)
        row = {"msrf519": ucr_io.clf(Etr, ytr, Ete, yte),
               "msrf760": ucr_io.clf(Str, ytr, Ste, yte),
               "d519": int(Etr.shape[1]), "d760": int(Str.shape[1])}
        ucr_io.update_results(name, row, results_file=RESULTS)
        print(f"{name:28s} 519 {row['msrf519']:.4f}  760 {row['msrf760']:.4f} (d={row['d760']})", flush=True)


if __name__ == "__main__":
    main()
