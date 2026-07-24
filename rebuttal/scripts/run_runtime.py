"""Single-core CPU throughput + head-fitting cost, identical machine/protocol for every method
(rebuttal efficiency table).   OMP_NUM_THREADS=1 python3 run_runtime.py"""
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from msenc_encoder_np import MultiSpaceEncCore
from msrf_battery import make_banks, battery

X = np.random.RandomState(0).randn(200, 256)

enc = MultiSpaceEncCore()
enc.transform(X[:8])  # warm up numba
t0 = time.perf_counter(); E = enc.transform(X); dt = time.perf_counter() - t0
print(f"MSRF+ (519d): {1000*dt/len(X):.2f} ms/series")
B = make_banks(E.shape[1])
Z = StandardScaler().fit_transform(E)
t0 = time.perf_counter(); battery(Z, B); dt2 = time.perf_counter() - t0
print(f"MSRF* (760d): {1000*(dt+dt2)/len(X):.2f} ms/series")

try:
    from aeon.transformations.collection.convolution_based import (
        MiniRocket, MultiRocket, HydraTransformer)
    from aeon.transformations.collection.interval_based import QUANTTransformer
    for nm, tf in (("MiniRocket", MiniRocket(random_state=0)), ("MultiRocket", MultiRocket()),
                   ("Hydra", HydraTransformer(random_state=0)), ("QUANT", QUANTTransformer())):
        tf.fit(X[:8, None, :])
        t0 = time.perf_counter(); tf.transform(X[:, None, :]); dt = time.perf_counter() - t0
        print(f"{nm}: {1000*dt/len(X):.2f} ms/series")
except ImportError:
    print("install aeon for baseline timings")

# head-fitting cost vs feature dimension (n=500 train instances)
y = np.random.RandomState(1).randint(0, 5, 500)
for d in (760, 6000, 50000):
    F = np.random.RandomState(2).randn(500, d).astype(np.float32)
    t0 = time.perf_counter()
    RidgeClassifierCV(alphas=np.logspace(-3, 3, 13)).fit(F, y)
    print(f"RidgeClassifierCV fit, n=500 d={d}: {time.perf_counter()-t0:.2f} s")
