"""Reviewer-requested baselines under the identical protocol: canonical MiniRocket [1],
MultiRocket [2], Hydra [3], QUANT [4], catch22 — all via their aeon implementations — plus the
submission's internal ROCKET-style 10k configuration (5,000 random kernels x ppv/max), labeled
as such. Writes/updates per-dataset rows in results/unified_protocol_results.jsonl.

  python3 run_sota_baselines.py --cache ./ucr_cache
Note (macOS): run with OMP_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue to avoid a
torch(Hydra)+numba(MultiRocket/QUANT) threading deadlock in one process.
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
import ucr_io
from msrf import MSRFTransform


def aeon_method(cls, **kw):
    def feats(Xtr, Xte):
        tf = cls(**kw)
        return (np.asarray(tf.fit_transform(Xtr[:, None, :])),
                np.asarray(tf.transform(Xte[:, None, :])))
    return feats


def rocket10k_internal(Xtr, Xte):
    """The submission's GPU convolutional baseline (random Gaussian kernels, random dilations,
    ppv+max): previously mislabeled 'equivalent to MiniRocket'; kept for the consistency check."""
    T = Xtr.shape[1]
    kw = dict(n_trf=0, n_grf_projections=0, n_srf=0, n_crf_kernels=5000,
              crf_pool_ops=["ppv", "max"])
    return (MSRFTransform(T=T, **kw).transform_numpy(Xtr.astype(np.float32)),
            MSRFTransform(T=T, **kw).transform_numpy(Xte.astype(np.float32)))


def methods():
    from aeon.transformations.collection.convolution_based import (
        MiniRocket, MultiRocket, HydraTransformer)
    from aeon.transformations.collection.interval_based import QUANTTransformer
    from aeon.transformations.collection.feature_based import Catch22
    return {
        "MiniRocket": aeon_method(MiniRocket, random_state=0),
        "MultiRocket": aeon_method(MultiRocket),
        "Hydra": aeon_method(HydraTransformer, random_state=0),
        "QUANT": aeon_method(QUANTTransformer),
        "catch22": aeon_method(Catch22),
        "Rocket10k-ppvmax": rocket10k_internal,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    M = methods()
    ap.add_argument("--methods", default=",".join(M))
    a = ap.parse_args()
    todo = [m for m in a.methods.split(",") if m in M]
    for i, name in enumerate(ucr_io.dataset_names()):
        try:
            Xtr, ytr, Xte, yte = ucr_io.load(name, a.cache)
        except Exception as e:
            print(f"[{i}] {name} SKIP: {e}", flush=True)
            continue
        rec = {}
        for m in todo:
            try:
                Ftr, Fte = M[m](Xtr, Xte)
                rec[m] = ucr_io.clf(Ftr, ytr, Fte, yte)
            except Exception as e:
                rec[m + "_err"] = str(e)[:80]
        ucr_io.update_results(name, rec)
        print(f"[{i}] {name} " + json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
