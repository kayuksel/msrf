"""Saturation analyses behind the effective-rank answer (reviewer THZ8):
(a) participation-ratio effective rank of the test-set embedding spectrum, conv-10k alone vs
    conv-10k + the submission's 410 non-convolutional features;
(b) out-of-sample subspace novelty: ridge-predict the appended block from the conv block on train,
    report unexplained variance on test, against a spanned-by-construction control (random linear
    combinations of the conv features) which sets the noise floor.
Runs on the 12 smallest datasets of the 52-dataset subset (size-sorted, deterministic).

  python3 run_saturation.py --cache ./ucr_cache
"""
import argparse
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler

import ucr_io
from msrf import MSRFTransform


def pr_rank(F):
    """Participation ratio of the covariance spectrum: (sum lambda)^2 / sum lambda^2."""
    lam = np.linalg.svd(F - F.mean(0), compute_uv=False) ** 2
    return float(lam.sum() ** 2 / (lam ** 2).sum())


def unexplained(Ctr, Cte, Btr, Bte, alpha=1.0):
    """Fit conv->block on train; 1 - variance-weighted R^2 on test."""
    r = Ridge(alpha=alpha).fit(Ctr, Btr)
    resid = Bte - r.predict(Cte)
    return float((resid ** 2).sum() / ((Bte - Btr.mean(0)) ** 2).sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="./ucr_cache")
    ap.add_argument("--n", type=int, default=12)
    a = ap.parse_args()
    names = [l.strip() for l in open(os.path.join(HERE, "datasets_subset52.txt")) if l.strip()]
    sized = []
    for name in names:
        try:
            Xtr, ytr, Xte, yte = ucr_io.load(name, a.cache)
            sized.append((Xtr.shape[0] * Xtr.shape[1], name))
        except Exception:
            pass
    picked = [n for _, n in sorted(sized)[:a.n]]
    print("datasets:", ", ".join(picked))
    prs_c, prs_cn, novel_nc, novel_ctl, prs_nc = [], [], [], [], []
    for name in picked:
        Xtr, ytr, Xte, yte = ucr_io.load(name, a.cache)
        T = Xtr.shape[1]
        conv = dict(n_trf=0, n_grf_projections=0, n_srf=0, n_crf_kernels=5000,
                    crf_pool_ops=["ppv", "max"])
        nonc = dict(n_trf=200, n_grf_projections=5, n_srf=200, n_crf_kernels=0)
        mk = lambda kw, X: MSRFTransform(T=T, **kw).transform_numpy(X.astype(np.float32))
        Ctr, Cte = mk(conv, Xtr), mk(conv, Xte)
        Ntr, Nte = mk(nonc, Xtr), mk(nonc, Xte)
        sc = StandardScaler().fit(Ctr); Ctr, Cte = sc.transform(Ctr), sc.transform(Cte)
        sn = StandardScaler().fit(Ntr); Ntr, Nte = sn.transform(Ntr), sn.transform(Nte)
        # spanned control: fixed random rotation of the conv block, same width as the nc block
        W = np.random.RandomState(0).randn(Ctr.shape[1], Ntr.shape[1]) / np.sqrt(Ctr.shape[1])
        Gtr, Gte = Ctr @ W, Cte @ W
        sg = StandardScaler().fit(Gtr); Gtr, Gte = sg.transform(Gtr), sg.transform(Gte)
        prs_c.append(pr_rank(Cte))
        prs_cn.append(pr_rank(np.concatenate([Cte, Nte], 1)))
        prs_nc.append(pr_rank(Nte))
        novel_nc.append(unexplained(Ctr, Cte, Ntr, Nte))
        novel_ctl.append(unexplained(Ctr, Cte, Gtr, Gte))
        print(f"{name:24s} PR(conv)={prs_c[-1]:6.1f} PR(conv+nc)={prs_cn[-1]:6.1f} "
              f"novelty(nc)={novel_nc[-1]:.3f} novelty(spanned-ctl)={novel_ctl[-1]:.3f}", flush=True)
    print(f"\nRESULT mean PR: conv-10k {np.mean(prs_c):.1f} -> +nc410 {np.mean(prs_cn):.1f}")
    print(f"RESULT mean OOS unexplained variance: nc410 {np.mean(novel_nc):.3f} "
          f"vs spanned control {np.mean(novel_ctl):.3f}")
    print(f"RESULT mean internal PR of the nc410 block alone: {np.mean(prs_nc):.1f} / 410")


if __name__ == "__main__":
    main()
