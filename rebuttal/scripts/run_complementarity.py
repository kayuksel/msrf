"""Combination experiment (Table R3): for each base method X, compare X vs X + appended feature
blocks — the submission's 410 non-convolutional MSRF features, MSRF+ (519), and MSRF* (760) —
on the fixed 52-dataset subset (datasets_subset52.txt), identical head throughout. Writes
per-dataset rows to results/complementarity_results.jsonl.

  python3 run_complementarity.py --cache ./ucr_cache
Note (macOS): OMP_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue (see run_sota_baselines.py).
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
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from sklearn.preprocessing import StandardScaler

import ucr_io
from msenc_encoder_np import MultiSpaceEncCore
from msrf_battery import make_banks, battery
from msrf import MSRFTransform
from run_sota_baselines import aeon_method, rocket10k_internal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    a = ap.parse_args()
    from aeon.transformations.collection.convolution_based import (
        MiniRocket, MultiRocket, HydraTransformer)
    from aeon.transformations.collection.interval_based import QUANTTransformer
    BASES = {
        "MiniRocket": aeon_method(MiniRocket, random_state=0),
        "Rocket10k-ppvmax": rocket10k_internal,
        "Hydra": aeon_method(HydraTransformer, random_state=0),
        "QUANT": aeon_method(QUANTTransformer),
        "MultiRocket": aeon_method(MultiRocket),
    }
    names = [l.strip() for l in open(os.path.join(HERE, "datasets_subset52.txt")) if l.strip()]
    enc = MultiSpaceEncCore()
    for i, name in enumerate(names):
        try:
            Xtr, ytr, Xte, yte = ucr_io.load(name, a.cache)
        except Exception as e:
            print(f"[{i}] {name} SKIP: {e}", flush=True)
            continue
        T = Xtr.shape[1]
        # appended blocks: NC410 (submission's non-convolutional spaces), MSRF+519, MSRF*760
        nc = lambda X: MSRFTransform(T=T, n_trf=200, n_grf_projections=5, n_srf=200,
                                     n_crf_kernels=0).transform_numpy(X.astype(np.float32))
        NCtr, NCte = nc(Xtr), nc(Xte)
        Etr, Ete = enc.transform(Xtr), enc.transform(Xte)
        sc = StandardScaler().fit(Etr)
        Ztr, Zte = sc.transform(Etr), sc.transform(Ete)
        B = make_banks(Ztr.shape[1])
        Str = np.concatenate([Ztr, battery(Ztr, B)], 1)
        Ste = np.concatenate([Zte, battery(Zte, B)], 1)
        rec = {}
        for base, feats in BASES.items():
            try:
                Ftr, Fte = feats(Xtr, Xte)
                Ftr, Fte = np.nan_to_num(Ftr), np.nan_to_num(Fte)
                rec[base] = ucr_io.clf(Ftr, ytr, Fte, yte)
                for tag, (Atr, Ate) in (("+NC410", (NCtr, NCte)),
                                        ("+MSRF+519", (Ztr, Zte)),
                                        ("+MSRF*760", (Str, Ste))):
                    rec[base + tag] = ucr_io.clf(np.concatenate([Ftr, Atr], 1), ytr,
                                                 np.concatenate([Fte, Ate], 1), yte)
            except Exception as e:
                rec[base + "_err"] = str(e)[:80]
        ucr_io.update_results(name, rec, "complementarity_results.jsonl")
        print(f"[{i}] {name} " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
