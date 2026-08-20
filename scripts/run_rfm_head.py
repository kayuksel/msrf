"""Ceiling ablation: what does the LINEAR readout cost?

Every other script here reads the frozen features with the same RidgeClassifierCV head, because a
linear head is what makes the deployed model auditable (a linear attribution is an identity, not an
approximation) and personalisable by an exact rank-one update. This script prices that choice: it
swaps in a leaf-RFM head (xRFM, Beaglehole et al., ICLR 2026 -- kernel ridge with iterated AGOP
feature reweighting; see rfm_head.py) on the SAME frozen MSRF+ (519-d) features.

Three readouts per dataset, identical features, identical scaling, identical splits:

  ridge      the deployed readout (RidgeClassifierCV, 13 log-spaced alphas)
  ridge_oof  CONTROL -- the same stratified 3-fold OOF selection machinery and the same
             task-aligned metric as the RFM head, but NO feature learning. Without it a win
             cannot be attributed to AGOP rather than to per-dataset tuning capacity.
  rfm        the leaf-RFM head, selected over config x AGOP iteration x lambda

  python3 run_rfm_head.py --cache ./ucr_cache

Writes/updates rfm_head_results.jsonl. Incremental and resumable.
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
sys.path.insert(0, os.path.join(HERE, ".."))  # repo root
# msrfc is numba-jitted and this is the only script here that also needs torch; numba's default
# OpenMP threading layer and torch's runtime segfault in one process on macOS. Must precede the
# numba import.
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")
import torch
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

import ucr_io
from msrfc import MultiSpaceEncCore
from rfm_head import ALPHAS, RFMHead, _clf_oof_metric, _torch_ridge, make_folds

RESULTS = "rfm_head_results.jsonl"


def load_ts(name, cache):
    """Dep-free reader for the cached .ts splits. Deliberately NOT ucr_io.load: that pulls aeon,
    whose OpenMP runtime segfaults when it shares a process with torch on macOS, and this is the
    only script here that needs torch. Same files, same official split, same label coding."""
    def split(tag):
        path = os.path.join(cache, name, f"{name}_{tag}.ts")
        rows, started = [], False
        for raw in open(path):
            line = raw.strip()
            if not started:
                started = line.lower() == "@data"
                continue
            if not line or line.startswith(("#", "@")) or ":" not in line:
                continue
            parts = line.split(":")
            rows.append(([float("nan") if t.strip() in ("?", "", "NaN") else float(t)
                          for t in parts[0].split(",")], parts[-1].strip()))
        return np.array([r[0] for r in rows], dtype=np.float64), [r[1] for r in rows]

    Xtr, ytr = split("TRAIN")
    Xte, yte = split("TEST")
    m = {c: i for i, c in enumerate(sorted(set(ytr)))}
    return Xtr, np.array([m[l] for l in ytr]), Xte, np.array([m.get(l, -1) for l in yte])


def acc_ridge(Ztr, ytr, Zte, yte):
    return float((RidgeClassifierCV(alphas=ALPHAS).fit(Ztr, ytr).predict(Zte) == yte).mean())


def acc_ridge_oof(Ztr, ytr, Zte, yte, C):
    """The control: ridge whose alpha is chosen on the same OOF folds, minimising the same
    task-aligned metric (binary 1-AUC / multiclass temperature-tuned log loss) the RFM head
    selects on. Isolates selection capacity from feature learning."""
    Phi_s = torch.as_tensor(Ztr, dtype=torch.float32)
    Phi_q = torch.as_tensor(Zte, dtype=torch.float32)
    Y = torch.nn.functional.one_hot(torch.as_tensor(ytr).long(), C).float()
    folds = make_folds(len(ytr), ytr, 3, 0)
    best_lam, best_E = 1.0, np.inf
    for lam in ALPHAS:
        S = np.full((len(ytr), C), np.nan)
        ok = np.zeros(len(ytr), dtype=bool)
        for fit_idx, val_idx in folds:
            if len(np.unique(ytr[fit_idx])) < C:
                continue
            Wt = _torch_ridge(Phi_s[fit_idx], Y[fit_idx], float(lam))
            S[val_idx] = (Phi_s[val_idx] @ Wt).numpy()
            ok[val_idx] = True
        if ok.sum() < 2 or len(np.unique(ytr[ok])) < 2:
            continue
        E = _clf_oof_metric(ytr[ok], S[ok], C)
        if E < best_E:
            best_lam, best_E = float(lam), E
    Wt = _torch_ridge(Phi_s, Y, best_lam)
    return float(((Phi_q @ Wt).numpy().argmax(1) == yte).mean()), best_lam


def acc_rfm(Ztr, ytr, Zte, yte, C, max_n=1024):
    """Kernel support is capped (stratified, seed 0): RFM costs O(n^2)-O(n^3) per config x fold x
    iteration. Only 3 of the 113 datasets exceed the cap; each row records whether it bit."""
    capped = len(Ztr) > max_n
    if capped:
        rng = np.random.RandomState(0)
        per = max(1, max_n // C)
        sub = np.concatenate([rng.choice(np.where(ytr == c)[0],
                                         min(int((ytr == c).sum()), per), replace=False)
                              for c in range(C)])
        if len(sub) > max_n:
            sub = rng.choice(sub, max_n, replace=False)
        Ztr, ytr = Ztr[sub], ytr[sub]
    Sq, info = RFMHead().fit_predict(Ztr, ytr, Zte, is_reg=False, n_classes=C)
    info["capped"], info["n_support"] = bool(capped), int(len(Ztr))
    return float((Sq.argmax(1) == yte).mean()), info


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    ap.add_argument("--support-cap", type=int, default=1024)
    a = ap.parse_args()

    path = os.path.join(ucr_io.RESULTS, RESULTS)
    done = {json.loads(l)["dataset"] for l in open(path)} if os.path.exists(path) else set()
    enc = MultiSpaceEncCore()
    for name in ucr_io.dataset_names():
        if name in done:
            continue
        try:
            Xtr, ytr, Xte, yte = load_ts(name, a.cache)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        C = len(np.unique(ytr))
        sc = StandardScaler().fit(enc.transform(Xtr))
        Ztr, Zte = sc.transform(enc.transform(Xtr)), sc.transform(enc.transform(Xte))

        a_ridge = acc_ridge(Ztr, ytr, Zte, yte)
        a_oof, lam = acc_ridge_oof(Ztr, ytr, Zte, yte, C)
        t0 = time.time()
        a_rfm, info = acc_rfm(Ztr, ytr, Zte, yte, C, a.support_cap)
        sec = time.time() - t0

        ucr_io.update_results(name, dict(
            n_train=int(Xtr.shape[0]), C=int(C), ridge=a_ridge, ridge_oof=a_oof, rfm=a_rfm,
            oof_lam=lam, rfm_cfg=info["config"], rfm_iters=info["t"], rfm_lam=info["lam"],
            capped=info["capped"], n_support=info["n_support"], sec_rfm=round(sec, 1)),
            results_file=RESULTS)
        print(f"{name:28s} N={Xtr.shape[0]:5d} C={C:2d} | ridge {a_ridge:.3f}  "
              f"oof {a_oof:.3f}  rfm {a_rfm:.3f}  ({sec:.0f}s)", flush=True)


if __name__ == "__main__":
    main()
