"""MSRF-family columns of the unified protocol: MSRF-1410 (submission config), MSRF+ (519 dims)
and MSRF* (760 dims), all under the identical head (see ucr_io.py). Writes/updates per-dataset
rows in results/unified_protocol_results.jsonl. Incremental and resumable.

  python3 run_unified_protocol.py --cache ./ucr_cache
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
sys.path.insert(0, os.path.join(HERE, "..", ".."))  # repo root: the original msrf package
from sklearn.preprocessing import StandardScaler

import ucr_io
from msenc_encoder_np import MultiSpaceEncCore
from msrf_battery import make_banks, battery
from msrf import MSRFTransform


def msrf1410_feats(Xtr, Xte):
    """The submission's configuration: TRF200 + GRF(5x256-moment) + SRF200 + CRF500(ppv,max)."""
    T = Xtr.shape[1]
    tf = lambda: MSRFTransform(T=T)
    return (tf().transform_numpy(Xtr.astype(np.float32)),
            tf().transform_numpy(Xte.astype(np.float32)))


def msrfplus_feats(Xtr, Xte):
    """MSRF+: windowed multi-space encoder, 173 features x mean/std/max pooling = 519 dims."""
    enc = MultiSpaceEncCore()
    return enc.transform(Xtr), enc.transform(Xte)


def msrfstar_feats(Xtr, Xte):
    """MSRF*: the 519-dim embedding + a 241-dim second-level battery of multi-space random
    features computed FROM the embedding (fixed integer seeds, no data-dependent selection)."""
    Etr, Ete = msrfplus_feats(Xtr, Xte)
    sc = StandardScaler().fit(Etr)
    Ztr, Zte = sc.transform(Etr), sc.transform(Ete)
    B = make_banks(Ztr.shape[1])
    return (np.concatenate([Ztr, battery(Ztr, B)], 1),
            np.concatenate([Zte, battery(Zte, B)], 1))


METHODS = {
    "MSRF-1410": msrf1410_feats,
    "MSRF+519": msrfplus_feats,
    "MSRF*760": msrfstar_feats,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    ap.add_argument("--methods", default=",".join(METHODS))
    a = ap.parse_args()
    todo = [m for m in a.methods.split(",") if m in METHODS]
    for i, name in enumerate(ucr_io.dataset_names()):
        try:
            Xtr, ytr, Xte, yte = ucr_io.load(name, a.cache)
        except Exception as e:
            print(f"[{i}] {name} SKIP: {e}", flush=True)
            continue
        rec = {}
        for m in todo:
            try:
                Ftr, Fte = METHODS[m](Xtr, Xte)
                rec[m] = ucr_io.clf(Ftr, ytr, Fte, yte)
            except Exception as e:
                rec[m + "_err"] = str(e)[:80]
        ucr_io.update_results(name, rec)
        print(f"[{i}] {name} " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
