"""Single-core CPU throughput (warmed, best-of-5) + head-fitting cost, identical protocol for
every method (rebuttal efficiency table).   OMP_NUM_THREADS=1 python3 run_runtime.py"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from msenc_encoder_np import MultiSpaceEncCore
from msrf_battery import make_banks, battery
from msrf import MSRFTransform

X = np.random.RandomState(0).randn(200, 256)


def t(fn, reps=5):
    best = 1e9
    for _ in range(reps):
        t0 = time.perf_counter(); fn(); best = min(best, time.perf_counter() - t0)
    return 1000 * best / len(X)


enc = MultiSpaceEncCore()
enc.transform(X[:8])  # warm up numba
E = enc.transform(X)
B = make_banks(E.shape[1])
res = {}
res["MSRF+ (519d)"] = t(lambda: enc.transform(X))
res["MSRF* (760d)"] = t(lambda: battery(StandardScaler().fit_transform(enc.transform(X)), B))

tf1410 = MSRFTransform(T=256); tf1410.transform_numpy(X[:8].astype(np.float32))
res["MSRF-1410"] = t(lambda: tf1410.transform_numpy(X.astype(np.float32)), reps=3)
kw = dict(n_trf=0, n_grf_projections=0, n_srf=0, n_crf_kernels=5000, crf_pool_ops=["ppv", "max"])
tfr = MSRFTransform(T=256, **kw); tfr.transform_numpy(X[:8].astype(np.float32))
res["Rocket10k-ppvmax"] = t(lambda: tfr.transform_numpy(X.astype(np.float32)), reps=3)

try:
    from aeon.transformations.collection.convolution_based import (
        MiniRocket, MultiRocket, HydraTransformer)
    from aeon.transformations.collection.interval_based import QUANTTransformer
    from aeon.transformations.collection.feature_based import Catch22
    for nm, tf in (("MiniRocket-1512", MiniRocket(n_kernels=1512, random_state=0)),
                   ("MiniRocket (9996d)", MiniRocket(random_state=0)),
                   ("Hydra (5120d)", HydraTransformer(random_state=0)),
                   ("MultiRocket (49728d)", MultiRocket()),
                   ("QUANT (2253d)", QUANTTransformer()),
                   ("catch22 (22d)", Catch22())):
        tf.fit(X[:8, None, :]); tf.transform(X[:8, None, :])  # warm up
        res[nm] = t(lambda tf=tf: tf.transform(X[:, None, :]), reps=3)
    res["MSRF*C (2272d)"] = res["MiniRocket-1512"] + res["MSRF* (760d)"]
except ImportError:
    print("install aeon for baseline timings")

for k, v in res.items():
    print(f"{k:24s} {v:6.2f} ms/series")

# head-fitting cost vs feature dimension (n=500 train instances)
y = np.random.RandomState(1).randint(0, 5, 500)
for d in (760, 2272, 5120, 50000):
    F = np.random.RandomState(2).randn(500, d).astype(np.float32)
    t0 = time.perf_counter()
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit(F, y)
    print(f"RidgeClassifierCV fit, n=500 d={d}: {time.perf_counter()-t0:.2f} s")
