"""What does the order-pooling repair cost on one CPU core?

Mirrors rebuttal_sweeps/ondevice_profile.py (same lengths, same best-of-N timing, same synthetic
batch) so the numbers drop straight into the published on-device table instead of being a
differently-measured set. Threads are pinned to 1 because the claim is single-core.

The cost has two separable parts and the arms separate them:

  enc519        the deployed encoder -- reference row
  order_coarse  blocks on the RELEASED patch grid: isolates the cost of the three blocks alone
  order_fine    blocks on the >=24-patch grid: adds the cost of resampling short series, which is
                where nearly all the overhead lives (need = 64 + 23*16 = 432 samples, so L=128 and
                L=256 are upsampled and L>=512 is not -- the overhead should fall with L)
  order_online  the one-pass O(1)-state blocks: should cost no more than order_fine

Run on an otherwise idle machine; concurrent jobs invalidate every number here.
  python3 profile_order.py
"""
import gc
import json
import os
import sys
import time

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMBA_NUM_THREADS"] = "1"
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from msrfc import MultiSpaceEncCore
from order_pool import FineOrderPoolEnc, FusedOrderPoolEnc, OrderPoolEnc
from order_pool_online import OnlineOrderPoolEnc

LENGTHS = (128, 256, 512, 1024)
N = 200
OUT = os.path.join(HERE, "..", "results", "profile_order.json")


def best_of(fn, reps=5):
    b = float("inf")
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


ARMS = [("enc519", MultiSpaceEncCore(), 519),
        ("order_coarse", OrderPoolEnc(), 1038),
        ("order_fine", FineOrderPoolEnc(), 1038),
        ("order_fused", FusedOrderPoolEnc(), 1038),
        ("order_online", OnlineOrderPoolEnc(), 1038)]

res = {"lengths": list(LENGTHS), "n_series": N, "ms_per_series": {}, "dims": {}}
for L in LENGTHS:
    X = np.random.RandomState(0).randn(N, L)
    for nm, enc, d in ARMS:
        enc.transform(X[:8])                                   # jit warm-up
        ms = 1000 * best_of(lambda e=enc: e.transform(X)) / N
        res["ms_per_series"].setdefault(nm, {})[L] = ms
        res["dims"][nm] = d
    ref = res["ms_per_series"]["enc519"][L]
    print(f"L={L:5d}  " + "  ".join(
        f"{nm}={res['ms_per_series'][nm][L]:.3f}ms({res['ms_per_series'][nm][L]/ref:.2f}x)"
        for nm, _, _ in ARMS), flush=True)

json.dump(res, open(OUT, "w"), indent=1)
print("wrote", os.path.relpath(OUT))
