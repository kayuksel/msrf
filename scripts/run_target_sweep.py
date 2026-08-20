"""Is the >=24-patch resampling target a tuned constant or an insensitive one?

The order-pooling gain is reported on a grid that resamples short series up to `target` patches.
That constant was chosen by reading the n_patch stratum table on the same 113 datasets the gain is
reported on, so it carries selection bias of unknown size. If accuracy is flat in `target` over a
wide range the constant is not doing the work and the gain is real; if it peaks sharply at 24 the
headline is partly a fitted hyperparameter and must be reported as such.

Sweeps target over a grid, recording the all3 arm only (`base` is target-independent by
construction -- it always uses the released coarse grid -- and is recorded once as a check).

  python3 run_target_sweep.py --cache ~/Desktop/worldmodel/ucr_cache
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

import ucr_io
from order_pool import BLOCKS, FineOrderPoolEnc, cols
from run_rfm_head import load_ts

RESULTS = "target_sweep_results.jsonl"
TARGETS = (6, 8, 12, 16, 24, 32, 48)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    a = ap.parse_args()

    path = os.path.join(ucr_io.RESULTS, RESULTS)
    done = {json.loads(l)["dataset"] for l in open(path)} if os.path.exists(path) else set()
    encs = {t: FineOrderPoolEnc(target=t) for t in TARGETS}
    nc = next(iter(encs.values())).n_cols
    c_all3, c_base = cols(nc, *BLOCKS), cols(nc)
    for name in ucr_io.dataset_names():
        if name in done:
            continue
        try:
            Xtr, ytr, Xte, yte = load_ts(name, a.cache)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        row, base = {}, None
        for t in TARGETS:
            Etr, Ete = encs[t].transform(Xtr), encs[t].transform(Xte)
            row[f"all3_t{t}"] = ucr_io.clf(Etr[:, c_all3], ytr, Ete[:, c_all3], yte)
            b = ucr_io.clf(Etr[:, c_base], ytr, Ete[:, c_base], yte)
            if base is None:
                row["base"] = base = b
            elif abs(b - base) > 1e-12:                       # must be target-independent
                row["base_drift"] = True
        row["T"] = int(Xtr.shape[1])
        ucr_io.update_results(name, row, results_file=RESULTS)
        print(f"{name:28s} T={row['T']:5d} base {row['base']:.3f} | "
              + "  ".join(f"t{t}={row[f'all3_t{t}']:.3f}" for t in TARGETS), flush=True)


if __name__ == "__main__":
    main()
