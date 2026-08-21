"""Per-window and batch-amortised latency, with thread pools pinned or at the runtime default.

Why this exists. The published on-device table reports ms/window for the frozen encoder and for
fitted baselines under one caption ("one ARM CPU core ... no threading"). Two facts make that caption
wrong, and both change the comparison rather than merely rescaling it:

  1. Nothing pinned the thread pools. `msenc_numba` is `@njit(parallel=True)` with `prange` over the
     BATCH, so a 200-window batch is spread across cores. The encoder rows are therefore
     multi-threaded throughput, not single-core latency.
  2. aeon self-pins. MiniRocket/MultiRocket call `set_num_threads(n_jobs)` and Hydra calls
     `torch.set_num_threads(...)`, both defaulting to 1 regardless of environment. The baseline rows
     were already single-threaded.

So the encoder rows move when you pin and the baseline rows do not, which is why the fix is not a
uniform rescale: any speed-up ratio taken from that table compares N threads against 1.

A third issue is independent of threading. Because the `prange` is over the batch, dividing a
200-window batch by 200 reports throughput, not the latency of classifying one window. On-device the
batch is 1. This script therefore reports BOTH:

  per_window   B=1, the number a real-time budget must be built from
  batch        B=200 amortised, comparable with the published table

Run it twice and it merges both conditions into one JSON:

  python3 profile_pinned.py --threads 1     # pinned: the honest single-core figure
  python3 profile_pinned.py --threads 0     # runtime default: reproduces the published row

Nothing is overwritten between runs; results land under a per-condition key so the two can be diffed.
Thread limits must be set before numba/torch import, so they are applied from argv at the top of the
file rather than in main().
"""
import argparse
import json
import os
import sys

_ap = argparse.ArgumentParser(add_help=False)
_ap.add_argument("--threads", type=int, default=1,
                 help="1 = pin every pool to one thread; 0 = leave the runtime default alone")
_known, _ = _ap.parse_known_args()
THREADS = _known.threads
if THREADS >= 1:                      # must precede numba, torch and BLAS imports
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "NUMBA_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = str(THREADS)
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import gc
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))

from sklearn.preprocessing import StandardScaler

from msrfc import MultiSpaceEncCore, battery, make_banks

LENGTHS = (128, 256, 512, 1024)
BATCH = 200
REPS = 5
OUT = os.path.join(HERE, "..", "results", "profile_pinned.json")


def best_of(fn, reps=REPS):
    """Minimum wall time over `reps` warmed repetitions -- the least noisy estimator of latency."""
    b = float("inf")
    for _ in range(reps):
        gc.collect()
        t0 = time.perf_counter()
        fn()
        b = min(b, time.perf_counter() - t0)
    return b


def encoder_arms():
    """The three encoder variants of the published table, as (name, callable, dims)."""
    enc = MultiSpaceEncCore()
    banks = {}

    def f519(X):
        enc.transform(X)

    def f760(X):
        E = enc.transform(X)
        if "b" not in banks:
            banks["b"] = make_banks(E.shape[1])
        battery(StandardScaler().fit_transform(E), banks["b"])

    arms = [("TSWM_emb_519", f519, 519), ("TSWM_760", f760, 760)]
    try:
        from aeon.transformations.collection.convolution_based import MiniRocket
        def f2272(X):
            mr = MiniRocket(n_kernels=1512, random_state=0)
            np.asarray(mr.fit_transform(X))
            f760(X)
        arms.append(("TSWM_MR_2272", f2272, 2272))
    except Exception as e:                       # aeon absent: encoder rows still profile
        print(f"note: MiniRocket unavailable, skipping the 2,272-d arm ({e})", flush=True)
    return arms


def baseline_arms():
    """Fitted baselines. NOTE: aeon pins its own thread count (n_jobs=1 by default) and Hydra pins
    torch, so these rows are single-threaded under BOTH conditions. That asymmetry is the point."""
    try:
        from aeon.transformations.collection.convolution_based import (
            HydraTransformer, MiniRocket, MultiRocket)
        from aeon.transformations.collection.feature_based import Catch22
        from aeon.transformations.collection.interval_based import QUANTTransformer
    except Exception as e:
        print(f"note: aeon unavailable, no baseline rows ({e})", flush=True)
        return []

    def wrap(cls, **kw):
        def g(X):
            np.asarray(cls(**kw).fit_transform(X))
        return g

    return [("MiniRocket_1512", wrap(MiniRocket, n_kernels=1512, random_state=0), 1512),
            ("MiniRocket_9996", wrap(MiniRocket, random_state=0), 9996),
            ("MultiRocket", wrap(MultiRocket), 49728),
            ("Hydra", wrap(HydraTransformer), 5120),
            ("QUANT", wrap(QUANTTransformer), 2253),
            ("catch22", wrap(Catch22), 22)]


def main():
    ap = argparse.ArgumentParser(parents=[_ap])
    ap.parse_args()
    key = "pinned_1" if THREADS >= 1 else "runtime_default"
    res = json.load(open(OUT)) if os.path.exists(OUT) else {}
    res.setdefault(key, {"threads_env": THREADS, "batch": BATCH, "reps": REPS,
                         "per_window": {}, "batch_amortised": {}, "dims": {}})
    slot = res[key]
    arms = encoder_arms() + baseline_arms()

    for L in LENGTHS:
        rng = np.random.RandomState(0)
        Xb = rng.randn(BATCH, L)
        X1 = Xb[:1]
        for name, fn, d in arms:
            fn(Xb[:8])                                        # jit / import warm-up
            per = 1000 * best_of(lambda: fn(X1))               # B=1: real-time latency
            bat = 1000 * best_of(lambda: fn(Xb)) / BATCH       # B=200 amortised: throughput
            slot["per_window"].setdefault(name, {})[str(L)] = round(per, 4)
            slot["batch_amortised"].setdefault(name, {})[str(L)] = round(bat, 4)
            slot["dims"][name] = d
            print(f"  L={L:5d}  {name:16s} per-window {per:8.3f} ms   batch/200 {bat:8.3f} ms",
                  flush=True)

    json.dump(res, open(OUT, "w"), indent=1)
    print("\nwrote", os.path.relpath(OUT))

    if len(res) == 2:                                          # both conditions present: diff them
        a, b = res.get("pinned_1"), res.get("runtime_default")
        print("\n=== pinned vs runtime default (batch-amortised, the published condition) ===")
        for name in a["batch_amortised"]:
            if name not in b["batch_amortised"]:
                continue
            row = "  ".join(
                f"L{L}:{a['batch_amortised'][name][str(L)]/b['batch_amortised'][name][str(L)]:5.2f}x"
                for L in LENGTHS)
            print(f"  {name:16s} {row}")
        print("\n  A ratio near 1.00 means the row was already single-threaded (expected for the")
        print("  aeon baselines). Ratios well above 1.00 mark rows whose published figure depended")
        print("  on multi-threading, and those are the ones the paper must restate.")


if __name__ == "__main__":
    main()
