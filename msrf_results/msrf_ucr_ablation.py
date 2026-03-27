#!/usr/bin/env python3
"""
MSRF UCR Ablation Study (GPU)
─────────────────────────────
Evaluates Multi-Space Random Features (TRF, GRF, SRF, CRF) on the full
UCR Time Series Classification Archive.  All feature extraction runs on GPU.
Classification via LightGBM (GPU).

Usage:
    python msrf_ucr_ablation.py --quick                    # ~24 small datasets
    python msrf_ucr_ablation.py --all                      # full 112 UCR datasets
    python msrf_ucr_ablation.py --datasets ArrowHead Beef  # specific datasets
    python msrf_ucr_ablation.py --resume                   # skip already-done
    python msrf_ucr_ablation.py --aggregate-only           # just print tables + CD diagram
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SEED = 42
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction: Temporal Random Features (TRF)  — GPU
# ─────────────────────────────────────────────────────────────────────────────

class TRFExtractor(nn.Module):
    """Gaussian-windowed positional features."""

    def __init__(self, T: int, K: int = 200, seed: int = SEED):
        super().__init__()
        rng = np.random.RandomState(seed)
        mu = rng.uniform(0, T, size=K).astype(np.float32)
        sigma = np.exp(rng.uniform(
            np.log(max(1, T * 0.01)), np.log(max(2, T / 2)), size=K
        )).astype(np.float32)
        self.register_buffer("mu", torch.from_numpy(mu))
        self.register_buffer("sigma", torch.from_numpy(sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        T = x.shape[1]
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        w = torch.exp(-((t[None, :] - self.mu[:, None]) ** 2) / (2 * self.sigma[:, None] ** 2))
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        return x @ w.T


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction: Global Random Features (GRF) — moment-pooled RKS, GPU
# ─────────────────────────────────────────────────────────────────────────────

class GRFExtractor(nn.Module):
    """RKS cos(Wx+b) with mean/std pooling."""

    def __init__(self, T: int, n_projections: int = 5, D: int = 256, seed: int = SEED):
        super().__init__()
        rng = np.random.RandomState(seed)
        Ws, bs = [], []
        for _ in range(n_projections):
            Ws.append(torch.from_numpy((rng.randn(T, D) / np.sqrt(T)).astype(np.float32)))
            bs.append(torch.from_numpy(rng.uniform(0, 2 * np.pi, size=D).astype(np.float32)))
        self.register_buffer("Ws", torch.stack(Ws))
        self.register_buffer("bs", torch.stack(bs))
        self.n_projections = n_projections

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feats = []
        for i in range(self.n_projections):
            z = torch.cos(x @ self.Ws[i] + self.bs[i])
            feats.append(z.mean(dim=1, keepdim=True))
            feats.append(z.std(dim=1, keepdim=True))
        return torch.cat(feats, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction: Statistical Random Features (SRF) — GPU
# ─────────────────────────────────────────────────────────────────────────────

def _compute_statistics_gpu(x: torch.Tensor) -> torch.Tensor:
    """x: (N, T) -> (N, 12)"""
    eps = 1e-12
    N, T = x.shape
    mean = x.mean(dim=1)
    std = x.std(dim=1).clamp(min=eps)
    centered = x - mean.unsqueeze(1)
    skew = (centered ** 3).mean(dim=1) / (std ** 3 + eps)
    kurt = (centered ** 4).mean(dim=1) / (std ** 4 + eps) - 3.0
    sorted_x = x.sort(dim=1).values
    q25 = sorted_x[:, max(0, int(T * 0.25))]
    q50 = sorted_x[:, max(0, int(T * 0.50))]
    q75 = sorted_x[:, max(0, int(T * 0.75))]
    iqr = q75 - q25
    dx = x[:, 1:] - x[:, :-1]
    mean_abs_diff = dx.abs().mean(dim=1)
    if T > 1:
        xp = x[:, :-1] - mean.unsqueeze(1)
        xn = x[:, 1:] - mean.unsqueeze(1)
        denom = ((xp ** 2).sum(1).sqrt() * (xn ** 2).sum(1).sqrt()).clamp(min=eps)
        lag1 = (xp * xn).sum(1) / denom
    else:
        lag1 = torch.zeros(N, device=x.device)
    return torch.stack([mean, std, skew, kurt, q25, q50, q75, iqr, lag1,
                        mean_abs_diff, x.min(1).values, x.max(1).values], dim=1)


class SRFExtractor(nn.Module):
    """Two-layer random projection of summary statistics."""

    def __init__(self, K: int = 100, d: int = 8, seed: int = SEED, M: int = 12):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.register_buffer("Ws", torch.from_numpy((rng.randn(K, d, M) / np.sqrt(M)).astype(np.float32)))
        self.register_buffer("bs", torch.from_numpy(rng.randn(K, d).astype(np.float32)))
        self.register_buffer("us", torch.from_numpy((rng.randn(K, d) / np.sqrt(d)).astype(np.float32)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        stats = _compute_statistics_gpu(x)
        h = torch.einsum("nm,kdm->nkd", stats, self.Ws) + self.bs.unsqueeze(0)
        h = F.relu(h)
        return (h * self.us.unsqueeze(0)).sum(dim=2)


# ─────────────────────────────────────────────────────────────────────────────
# Feature Extraction: CRF — MiniRocketPlus from adia/deep_model.py
# Configurable transforms × pool_ops for different feature levels
# ─────────────────────────────────────────────────────────────────────────────

class MiniRocketPlus(nn.Module):
    """GPU ROCKET with configurable transforms and pooling operations.
    Feature count = num_kernels × len(transforms) × len(pool_ops).
    """

    def __init__(
        self,
        input_length: int,
        num_kernels: int = 10000,
        kernel_size: int = 9,
        transforms: Optional[List[str]] = None,
        pool_ops: Optional[List[str]] = None,
        seed: int = SEED,
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.transforms = transforms or ["identity"]
        self.pool_ops = pool_ops or ["ppv"]
        self.eps = 1e-8

        torch.manual_seed(seed)
        self.register_buffer("base_w", torch.randn(num_kernels, kernel_size))
        self.register_buffer(
            "dilations",
            torch.randint(1, max(2, input_length // 2), (num_kernels,))
        )
        km = torch.ones(num_kernels, 1, kernel_size)
        self.register_buffer("kmasks", km)

    def _hilbert(self, x: torch.Tensor) -> torch.Tensor:
        Xf = torch.fft.fft(x.float(), dim=-1)
        N = x.size(-1)
        h = torch.zeros(N, device=x.device, dtype=Xf.dtype)
        if N % 2 == 0:
            h[0] = 1; h[N // 2] = 1; h[1:N // 2] = 2
        else:
            h[0] = 1; h[1:(N + 1) // 2] = 2
        return torch.fft.ifft(Xf * h, dim=-1).abs().to(x.dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T) -> (B, n_kernels * n_transforms * n_pool_ops)"""
        B, T = x.shape
        x_3d = x.unsqueeze(1)
        mask = torch.ones(B, 1, T, device=x.device)
        feats = []

        for tr in self.transforms:
            if tr == "identity":
                rep = x_3d
            elif tr == "diff":
                rep = F.pad(x_3d[..., 1:] - x_3d[..., :-1], (0, 1))
            elif tr == "hilbert":
                rep = self._hilbert(x_3d)
            else:
                rep = x_3d

            for di in torch.unique(self.dilations):
                di_int = int(di.item())
                idxs = (self.dilations == di).nonzero(as_tuple=True)[0]
                w = self.base_w[idxs].unsqueeze(1)
                km = self.kmasks[idxs]
                pad = di_int * (self.kernel_size // 2)

                out = F.conv1d(rep, w, padding=pad, dilation=di_int)
                valid = F.conv1d(mask, km, padding=pad, dilation=di_int).clamp(min=1)
                pos = out > 0
                cnt_pos = pos.sum(dim=2).float()
                L = out.size(2)

                for op in self.pool_ops:
                    if op == "ppv":
                        feat = cnt_pos / valid.sum(dim=2).float()
                    elif op == "max":
                        feat = out.max(dim=2).values
                    elif op == "mpv":
                        Wmax = self.base_w[idxs].clamp(min=0).sum(dim=1).view(1, -1)
                        feat = out.clamp(min=0).sum(dim=2) / (Wmax + self.eps)
                    elif op == "mipv":
                        idxr = torch.arange(L, device=out.device).float()
                        feat = (idxr * pos.float()).sum(dim=2) / cnt_pos.clamp(min=1) / L
                    elif op == "zcr":
                        feat = ((out.sign()[..., :-1] * out.sign()[..., 1:]) < 0).sum(dim=2).float() / L
                    elif op == "stretch":
                        idxr = torch.arange(L, device=out.device).float()
                        pi = pos.float() * idxr.view(1, 1, -1)
                        pi[~pos] = 0.0
                        diff = pi[..., 1:] - pi[..., :-1]
                        feat = diff.max(dim=2).values / L
                    else:
                        continue
                    feats.append(feat.clamp(min=self.eps, max=1e6))

        return torch.cat(feats, dim=1)


# ─────────────────────────────────────────────────────────────────────────────
# MSRF: combined extraction with ablation configs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class MSRFConfig:
    name: str
    trf: int = 0
    grf: int = 0
    grf_dim: int = 256
    srf: int = 0
    # CRF / MiniRocketPlus params
    crf_kernels: int = 0
    crf_transforms: List[str] = field(default_factory=lambda: ["identity"])
    crf_pool_ops: List[str] = field(default_factory=lambda: ["ppv"])
    crf_kernel_size: int = 9
    minirocket_aeon: bool = False


ABLATION_CONFIGS = [
    # Individual spaces
    MSRFConfig(name="TRF_only",        trf=200),
    MSRFConfig(name="GRF_only",        grf=5, grf_dim=256),
    MSRFConfig(name="SRF_only",        srf=200),
    MSRFConfig(name="CRF_only",        crf_kernels=500, crf_pool_ops=["ppv", "max"]),

    # Pairwise combinations
    MSRFConfig(name="TRF+GRF",         trf=200, grf=5),
    MSRFConfig(name="TRF+SRF",         trf=200, srf=200),
    MSRFConfig(name="TRF+CRF",         trf=200, crf_kernels=500, crf_pool_ops=["ppv", "max"]),
    MSRFConfig(name="GRF+SRF",         grf=5, srf=200),
    MSRFConfig(name="GRF+CRF",         grf=5, crf_kernels=500, crf_pool_ops=["ppv", "max"]),
    MSRFConfig(name="SRF+CRF",         srf=200, crf_kernels=500, crf_pool_ops=["ppv", "max"]),

    # Triple
    MSRFConfig(name="TRF+GRF+SRF",     trf=200, grf=5, srf=200),

    # Full MSRF
    MSRFConfig(name="MSRF_full",        trf=200, grf=5, srf=200,
               crf_kernels=500, crf_pool_ops=["ppv", "max"]),
    MSRFConfig(name="MSRF_small",       trf=50, grf=3, srf=50,
               crf_kernels=100, crf_pool_ops=["ppv"]),

    # Equal-budget pure conv baseline (same ~1400 dims as MSRF_full)
    MSRFConfig(name="CRF_700",          crf_kernels=700, crf_pool_ops=["ppv", "max"]),

    # ROCKET baselines — using MiniRocketPlus with different feature levels
    MSRFConfig(name="Rocket_ppv",       crf_kernels=10000, crf_pool_ops=["ppv"]),
    MSRFConfig(name="Rocket_ppv+mpv",   crf_kernels=10000, crf_pool_ops=["ppv", "mpv"]),
    MSRFConfig(name="Rocket_full",      crf_kernels=10000,
               crf_transforms=["identity", "diff"],
               crf_pool_ops=["ppv", "mpv", "mipv"]),
    MSRFConfig(name="MiniRocket_aeon",  minirocket_aeon=True),

    # Augmented ROCKET — ROCKET + our non-conv features (Comparison B: complementarity)
    MSRFConfig(name="Rocket_ppv+MSRF",  trf=200, grf=5, srf=200,
               crf_kernels=10000, crf_pool_ops=["ppv"]),
    MSRFConfig(name="Rocket_pm+MSRF",   trf=200, grf=5, srf=200,
               crf_kernels=10000, crf_pool_ops=["ppv", "mpv"]),
    MSRFConfig(name="Rocket_full+MSRF", trf=200, grf=5, srf=200,
               crf_kernels=10000,
               crf_transforms=["identity", "diff"],
               crf_pool_ops=["ppv", "mpv", "mipv"]),
]


@torch.no_grad()
def extract_features_gpu(
    x_train_np: np.ndarray,
    x_test_np: np.ndarray,
    cfg: MSRFConfig,
    device: torch.device,
    batch_size: int = 4096,
) -> Tuple[np.ndarray, np.ndarray]:
    """Extract features on GPU, return numpy arrays."""

    if cfg.minirocket_aeon:
        from aeon.transformations.collection.convolution_based import MiniRocket
        mr = MiniRocket(n_kernels=10_000, random_state=SEED)
        tr3d = x_train_np[:, np.newaxis, :].astype(np.float64)
        te3d = x_test_np[:, np.newaxis, :].astype(np.float64)
        mr.fit(tr3d)
        return mr.transform(tr3d).astype(np.float32), mr.transform(te3d).astype(np.float32)

    T = x_train_np.shape[1]
    extractors = []
    if cfg.trf > 0:
        extractors.append(TRFExtractor(T, K=cfg.trf, seed=SEED).to(device))
    if cfg.grf > 0:
        extractors.append(GRFExtractor(T, n_projections=cfg.grf, D=cfg.grf_dim, seed=SEED + 100).to(device))
    if cfg.srf > 0:
        extractors.append(SRFExtractor(K=cfg.srf, seed=SEED + 200).to(device))
    if cfg.crf_kernels > 0:
        extractors.append(MiniRocketPlus(
            input_length=T,
            num_kernels=cfg.crf_kernels,
            kernel_size=cfg.crf_kernel_size,
            transforms=cfg.crf_transforms,
            pool_ops=cfg.crf_pool_ops,
            seed=SEED + 300,
        ).to(device))

    def _extract(x_np):
        x_t = torch.from_numpy(x_np.astype(np.float32)).to(device)
        N = x_t.shape[0]
        all_parts = [[] for _ in extractors]
        for start in range(0, N, batch_size):
            batch = x_t[start:start + batch_size]
            for i, ext in enumerate(extractors):
                all_parts[i].append(ext(batch).cpu())
        parts = [torch.cat(p, dim=0).numpy() for p in all_parts]
        return np.hstack(parts) if parts else np.empty((N, 0), dtype=np.float32)

    return _extract(x_train_np), _extract(x_test_np)


# ─────────────────────────────────────────────────────────────────────────────
# Classification
# ─────────────────────────────────────────────────────────────────────────────

def classify_ridge(
    F_train: np.ndarray, y_train: np.ndarray,
    F_test: np.ndarray, y_test: np.ndarray,
    **kwargs,
) -> Dict[str, Any]:
    """RidgeClassifierCV — the standard ROCKET-paper classifier. Near-instant."""
    from sklearn.linear_model import RidgeClassifierCV

    scaler = StandardScaler()
    Xtr = np.nan_to_num(scaler.fit_transform(F_train), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(scaler.transform(F_test), nan=0.0, posinf=0.0, neginf=0.0)

    clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
    clf.fit(Xtr, y_train)
    y_pred = clf.predict(Xte)
    acc = float((y_pred == y_test).mean())
    return {"accuracy": acc, "n_features": F_train.shape[1], "alpha": float(clf.alpha_)}


def classify_lgbm(
    F_train: np.ndarray, y_train: np.ndarray,
    F_test: np.ndarray, y_test: np.ndarray,
    n_classes: int = 2, use_gpu: bool = True,
) -> Dict[str, Any]:
    """LightGBM with GPU support."""
    import lightgbm as lgb

    scaler = StandardScaler()
    Xtr = np.nan_to_num(scaler.fit_transform(F_train), nan=0.0, posinf=0.0, neginf=0.0)
    Xte = np.nan_to_num(scaler.transform(F_test), nan=0.0, posinf=0.0, neginf=0.0)

    params = {
        "objective": "multiclass" if n_classes > 2 else "binary",
        "num_class": n_classes if n_classes > 2 else 1,
        "metric": "multi_logloss" if n_classes > 2 else "binary_logloss",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": max(1, len(y_train) // 50),
        "subsample": 0.8,
        "colsample_bytree": min(1.0, 500 / max(1, Xtr.shape[1])),
        "reg_alpha": 0.1, "reg_lambda": 1.0,
        "n_jobs": -1, "verbose": -1, "seed": SEED,
    }
    if use_gpu and torch.cuda.is_available():
        params["device"] = "gpu"
        params["gpu_use_dp"] = False

    dtrain = lgb.Dataset(Xtr, label=y_train)
    dval = lgb.Dataset(Xte, label=y_test, reference=dtrain)
    model = lgb.train(params, dtrain, num_boost_round=300,
                      valid_sets=[dval],
                      callbacks=[lgb.early_stopping(20, verbose=False), lgb.log_evaluation(0)])

    if n_classes > 2:
        y_proba = model.predict(Xte)
        y_pred = y_proba.argmax(axis=1)
    else:
        y_proba = model.predict(Xte)
        y_pred = (y_proba > 0.5).astype(int)

    return {"accuracy": float((y_pred == y_test).mean()),
            "n_features": F_train.shape[1], "best_iteration": model.best_iteration}


# ─────────────────────────────────────────────────────────────────────────────
# Dataset loading
# ─────────────────────────────────────────────────────────────────────────────

def load_ucr_dataset(name: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    from aeon.datasets import load_classification
    from sklearn.preprocessing import LabelEncoder

    x_train, y_train = load_classification(name, split="train")
    x_test, y_test = load_classification(name, split="test")

    if x_train.ndim == 3 and x_train.shape[1] == 1:
        x_train = x_train.squeeze(1)
        x_test = x_test.squeeze(1)

    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_test = le.transform(y_test)

    x_train = np.nan_to_num(x_train, nan=0.0).astype(np.float64)
    x_test = np.nan_to_num(x_test, nan=0.0).astype(np.float64)
    return x_train, y_train, x_test, y_test


def get_dataset_list(args) -> List[str]:
    if args.datasets:
        return args.datasets
    from aeon.datasets.tsc_datasets import univariate_equal_length
    names = sorted(univariate_equal_length)
    if args.quick:
        small = [n for n in names if n in {
            "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "Car",
            "CBF", "Coffee", "DiatomSizeReduction", "ECGFiveDays",
            "FaceFour", "GunPoint", "ItalyPowerDemand", "Lightning7",
            "MedicalImages", "MoteStrain", "OliveOil", "Plane",
            "SmoothSubspace", "SwedishLeaf", "Symbols",
            "SyntheticControl", "Trace", "TwoLeadECG", "Wafer",
        }]
        return small if small else names[:20]
    return names


# ─────────────────────────────────────────────────────────────────────────────
# Critical Difference Diagram (Demšar / Nemenyi)
# ─────────────────────────────────────────────────────────────────────────────

def _compute_ranks(accs, datasets, configs):
    n_ds, n_cfg = len(datasets), len(configs)
    rank_matrix = np.full((n_ds, n_cfg), np.nan)
    for i, ds in enumerate(datasets):
        vals = np.array([accs[ds].get(cfg, np.nan) for cfg in configs])
        valid = ~np.isnan(vals)
        if valid.sum() < 2:
            continue
        order = np.argsort(-vals[valid])
        r = np.empty_like(order, dtype=float)
        r[order] = np.arange(1, valid.sum() + 1, dtype=float)
        for v in np.unique(vals[valid]):
            tied = np.where(vals[valid] == v)[0]
            if len(tied) > 1:
                r[tied] = r[tied].mean()
        rank_matrix[i, valid] = r
    return rank_matrix


def _nemenyi_cd(k: int, n: int, alpha: float = 0.05) -> float:
    from scipy.stats import studentized_range
    q_alpha = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q_alpha * np.sqrt(k * (k + 1) / (6.0 * n))


def plot_critical_difference(
    avg_ranks: Dict[str, float],
    n_datasets: int,
    cd: float,
    output_path: Path,
    title: str = "Critical Difference Diagram",
    highlight: Optional[List[str]] = None,
):
    """Publication-quality CD diagram à la Demšar (2006).

    Standard layout: rank axis on top, best (rank 1) on left.
    Methods split left/right; each gets its own row with a horizontal
    connector to its rank tick. Clique bars below the axis.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import rcParams
    rcParams["font.family"] = "sans-serif"

    names = sorted(avg_ranks, key=lambda k_: avg_ranks[k_])
    ranks = np.array([avg_ranks[n] for n in names])
    k = len(names)
    highlight = set(highlight or [])

    # --- clique detection (groups not significantly different) ---
    cliques = []
    for i in range(k):
        for j in range(i + 1, k):
            if ranks[j] - ranks[i] < cd:
                merged = False
                for c in cliques:
                    if i in c and j not in c and all(ranks[j] - ranks[m] < cd for m in c):
                        c.add(j); merged = True; break
                    elif j in c and i not in c and all(ranks[m] - ranks[i] < cd for m in c):
                        c.add(i); merged = True; break
                    elif i in c and j in c:
                        merged = True; break
                if not merged:
                    cliques.append({i, j})
    changed = True
    while changed:
        changed = False
        new_cliques, used = [], set()
        for ci, c1 in enumerate(cliques):
            if ci in used: continue
            cur = set(c1)
            for cj, c2 in enumerate(cliques):
                if cj <= ci or cj in used: continue
                union = cur | c2
                si = sorted(union)
                if all(ranks[b] - ranks[a] < cd for a in si for b in si if b > a):
                    cur = union; used.add(cj); changed = True
            new_cliques.append(cur); used.add(ci)
        cliques = new_cliques
    cliques = [c for i, c in enumerate(cliques)
               if not any(c < o for j, o in enumerate(cliques) if j != i)]

    # --- figure geometry ---
    row_h = 0.32
    n_left = (k + 1) // 2
    n_right = k - n_left
    body_h = max(n_left, n_right) * row_h
    clique_h = len(cliques) * 0.18
    top_margin = 1.3
    fig_h = top_margin + body_h + clique_h + 0.1
    fig_w = max(7.0, 3.0 + k * 0.45)
    text_margin = max(len(n) for n in names) * 0.085 + 0.3

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.set_axis_off()
    fig.patch.set_facecolor("white")

    ax_left = text_margin
    ax_right = fig_w - text_margin
    ax_span = ax_right - ax_left

    def rank_to_x(r):
        return ax_left + (r - 1) / max(1, k - 1) * ax_span

    ax.set_xlim(0, fig_w)
    ax.set_ylim(-(body_h + clique_h + 0.2), top_margin)

    # --- rank axis ---
    axis_y = 0.0
    ax.plot([ax_left, ax_right], [axis_y, axis_y], "k-", lw=1.8, clip_on=False)
    for r in range(1, k + 1):
        x = rank_to_x(r)
        ax.plot([x, x], [axis_y - 0.06, axis_y + 0.06], "k-", lw=1.2, clip_on=False)
    for r in range(1, k + 1, max(1, k // 8)):
        ax.text(rank_to_x(r), axis_y + 0.12, str(r), ha="center", va="bottom",
                fontsize=9, fontweight="bold")
    if k > 1 and k % max(1, k // 8) != 1:
        ax.text(rank_to_x(k), axis_y + 0.12, str(k), ha="center", va="bottom",
                fontsize=9, fontweight="bold")

    # --- CD bar ---
    cd_y = axis_y + 0.45
    cd_x1, cd_x2 = rank_to_x(1), rank_to_x(1 + cd)
    ax.plot([cd_x1, cd_x2], [cd_y, cd_y], "k-", lw=2.5, clip_on=False)
    for cx in [cd_x1, cd_x2]:
        ax.plot([cx, cx], [cd_y - 0.05, cd_y + 0.05], "k-", lw=2.5, clip_on=False)
    ax.text((cd_x1 + cd_x2) / 2, cd_y + 0.08, f"CD = {cd:.2f}",
            ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- title ---
    ax.text(fig_w / 2, cd_y + 0.4, title, ha="center", va="bottom",
            fontsize=12, fontweight="bold")

    # --- split methods into left (best) and right (worst) ---
    left_items = [(i, names[i], ranks[i]) for i in range(n_left)]
    right_items = [(i, names[i], ranks[i]) for i in range(n_left, k)]

    def draw_method(idx, name, rank, slot, side):
        y = -(0.3 + slot * row_h)
        rx = rank_to_x(rank)
        is_hl = name in highlight
        color = "#c0392b" if is_hl else "#222222"
        weight = "bold" if is_hl else "normal"
        fsize = 9.5 if is_hl else 9

        ax.plot(rx, axis_y, "ko", ms=3.5, clip_on=False, zorder=5)
        if side == "left":
            ax.plot([rx, rx], [axis_y, y], "-", color="#888", lw=0.6, clip_on=False)
            ax.plot([ax_left - 0.15, rx], [y, y], "-", color="#888", lw=0.6, clip_on=False)
            ax.plot(ax_left - 0.15, y, "s", color=color, ms=3, clip_on=False)
            ax.text(ax_left - 0.3, y, name, ha="right", va="center",
                    fontsize=fsize, fontweight=weight, color=color)
        else:
            ax.plot([rx, rx], [axis_y, y], "-", color="#888", lw=0.6, clip_on=False)
            ax.plot([rx, ax_right + 0.15], [y, y], "-", color="#888", lw=0.6, clip_on=False)
            ax.plot(ax_right + 0.15, y, "s", color=color, ms=3, clip_on=False)
            ax.text(ax_right + 0.3, y, name, ha="left", va="center",
                    fontsize=fsize, fontweight=weight, color=color)

    for slot, (idx, name, rank) in enumerate(left_items):
        draw_method(idx, name, rank, slot, "left")
    for slot, (idx, name, rank) in enumerate(right_items):
        draw_method(idx, name, rank, slot, "right")

    # --- clique bars ---
    clique_base = -(0.3 + max(n_left, n_right) * row_h + 0.05)
    clique_colors = ["#2c3e50", "#7f8c8d", "#34495e", "#95a5a6", "#1a252f"]
    for ci, clique in enumerate(cliques):
        idxs_c = sorted(clique)
        y = clique_base - ci * 0.17
        x1, x2 = rank_to_x(ranks[idxs_c[0]]), rank_to_x(ranks[idxs_c[-1]])
        cc = clique_colors[ci % len(clique_colors)]
        ax.plot([x1, x2], [y, y], "-", color=cc, lw=4.5,
                solid_capstyle="round", clip_on=False)

    plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(output_path, dpi=200, bbox_inches="tight", facecolor="white", pad_inches=0.15)
    plt.close(fig)
    logger.info(f"CD diagram saved to {output_path}")


def _make_cd_for_subset(output_dir, accs, datasets, subset_configs, filename, title, highlight):
    """Helper: compute ranks within a subset of configs and plot CD diagram."""
    from scipy.stats import friedmanchisquare
    complete = [ds for ds in datasets if all(accs[ds].get(c) is not None for c in subset_configs)]
    if len(complete) < 3:
        logger.warning(f"  {filename}: only {len(complete)} complete datasets — need ≥3, skipping.")
        return
    rm = _compute_ranks(accs, complete, subset_configs)
    rm = rm[~np.isnan(rm).any(axis=1)]
    n = rm.shape[0]
    k = len(subset_configs)
    avg = {c: float(rm[:, i].mean()) for i, c in enumerate(subset_configs)}
    try:
        stat, p = friedmanchisquare(*[rm[:, i] for i in range(k)])
        logger.info(f"  {filename}: Friedman χ²={stat:.2f}, p={p:.2e} (n={n}, k={k})")
    except Exception:
        pass
    cd = _nemenyi_cd(k, n, alpha=0.05)
    hl = [c for c in subset_configs if c in highlight]
    for ext in ["pdf", "png"]:
        plot_critical_difference(avg, n, cd,
            output_path=output_dir / f"{filename}.{ext}",
            title=f"{title} ({n} datasets)", highlight=hl)


def generate_cd_diagrams(output_dir, accs, datasets, configs):
    from scipy.stats import friedmanchisquare

    complete_ds = [ds for ds in datasets if all(accs[ds].get(c) is not None for c in configs)]
    if len(complete_ds) < 3:
        logger.warning(f"Only {len(complete_ds)} complete datasets — need ≥3 for CD.")
        return

    n = len(complete_ds)
    k = len(configs)
    rm = _compute_ranks(accs, complete_ds, configs)
    valid = ~np.isnan(rm).any(axis=1)
    rm = rm[valid]
    n = rm.shape[0]
    avg_ranks = {cfg: float(rm[:, i].mean()) for i, cfg in enumerate(configs)}

    try:
        stat, p = friedmanchisquare(*[rm[:, i] for i in range(k)])
        logger.info(f"Friedman χ²={stat:.2f}, p={p:.2e} (n={n}, k={k})")
    except Exception:
        pass

    cd = _nemenyi_cd(k, n, alpha=0.05)
    logger.info(f"Nemenyi CD (α=0.05) = {cd:.3f}")

    hl = set(c for c in configs if "MSRF" in c or "msrf" in c)

    for ext in ["pdf", "png"]:
        plot_critical_difference(
            avg_ranks, n, cd,
            output_path=output_dir / f"cd_diagram_full.{ext}",
            title=f"All Configurations",
            highlight=list(hl),
        )

    # Focused: MSRF vs pure-ROCKET baselines
    key_names = {"MSRF_full", "MSRF_small", "CRF_only", "CRF_700",
                 "Rocket_ppv", "Rocket_ppv+mpv", "Rocket_full", "MiniRocket_aeon"}
    key = [c for c in configs if c in key_names]
    if len(key) >= 3:
        _make_cd_for_subset(output_dir, accs, datasets, key,
            "cd_msrf_vs_rocket", "MSRF vs ROCKET Baselines",
            hl)

    # Comparison A — Feature Efficiency (same ~1400 dims budget)
    eff_names = {"MSRF_full", "CRF_700", "CRF_only", "MSRF_small"}
    eff = [c for c in configs if c in eff_names]
    if len(eff) >= 3:
        _make_cd_for_subset(output_dir, accs, datasets, eff,
            "cd_feature_efficiency", "Feature Efficiency (same dim budget)",
            hl)

    # Comparison B — Complementarity (ROCKET ± our non-conv features)
    comp_pairs = [
        ("Rocket_ppv", "Rocket_ppv+MSRF"),
        ("Rocket_ppv+mpv", "Rocket_pm+MSRF"),
        ("Rocket_full", "Rocket_full+MSRF"),
    ]
    comp = []
    for base, aug in comp_pairs:
        if base in configs: comp.append(base)
        if aug in configs: comp.append(aug)
    if "MSRF_full" in configs:
        comp.append("MSRF_full")
    comp = list(dict.fromkeys(comp))
    if len(comp) >= 3:
        _make_cd_for_subset(output_dir, accs, datasets, comp,
            "cd_complementarity", "ROCKET + MSRF Features (Complementarity)",
            hl)


# ─────────────────────────────────────────────────────────────────────────────
# Results aggregation
# ─────────────────────────────────────────────────────────────────────────────

def aggregate_results(output_dir: Path):
    results_file = output_dir / "all_results.jsonl"
    if not results_file.exists():
        logger.warning("No results file found.")
        return

    records = []
    with open(results_file) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        return

    accs: Dict[str, Dict[str, float]] = defaultdict(dict)
    for r in records:
        if r.get("accuracy") is not None:
            accs[r["dataset"]][r["config"]] = r["accuracy"]

    datasets = sorted(accs.keys())
    configs = sorted({r["config"] for r in records if r.get("accuracy") is not None})

    header = f"{'Dataset':<30s}" + "".join(f"{c:>16s}" for c in configs)
    print("\n" + "=" * len(header))
    print("ACCURACY TABLE")
    print("=" * len(header))
    print(header)
    print("-" * len(header))

    for ds in datasets:
        row = f"{ds:<30s}"
        best_acc = max(accs[ds].values()) if accs[ds] else -1
        for cfg in configs:
            val = accs[ds].get(cfg)
            if val is not None:
                marker = " *" if val == best_acc else "  "
                row += f"{val:>14.4f}{marker}"
            else:
                row += f"{'—':>16s}"
        print(row)

    ranks: Dict[str, List[float]] = defaultdict(list)
    for ds in datasets:
        ds_accs = [(cfg, accs[ds].get(cfg, -1)) for cfg in configs]
        ds_accs_sorted = sorted(ds_accs, key=lambda x: -x[1])
        for rank, (cfg, _) in enumerate(ds_accs_sorted, 1):
            ranks[cfg].append(rank)

    print(f"\n{'='*60}\nAVERAGE RANKS (lower is better)\n{'='*60}")
    rank_summary = sorted([(c, np.mean(r), np.std(r)) for c, r in ranks.items()], key=lambda x: x[1])
    for cfg, mr, sr in rank_summary:
        wins = sum(1 for ds in datasets if accs[ds].get(cfg, -1) == max(accs[ds].values()))
        print(f"  {cfg:<22s}  rank={mr:.2f} ± {sr:.2f}  wins={wins}")

    print(f"\n{'='*60}\nMEAN ACCURACY\n{'='*60}")
    for cfg, _, _ in rank_summary:
        vals = [accs[ds].get(cfg) for ds in datasets if accs[ds].get(cfg) is not None]
        if vals:
            print(f"  {cfg:<22s}  {np.mean(vals):.4f} ± {np.std(vals):.4f}")

    summary = {
        "n_datasets": len(datasets),
        "configs": configs,
        "avg_ranks": {c: float(np.mean(r)) for c, r in ranks.items()},
        "mean_accuracy": {c: float(np.mean([accs[ds].get(c, np.nan) for ds in datasets])) for c in configs},
    }
    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    if len(datasets) >= 3 and len(configs) >= 2:
        try:
            generate_cd_diagrams(output_dir, accs, datasets, configs)
        except Exception as e:
            logger.error(f"CD diagram failed: {e}")
            import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────

def run_single_dataset(
    dataset_name: str,
    configs: List[MSRFConfig],
    output_dir: Path,
    device: torch.device,
    skip_existing: bool = True,
    use_gpu_lgbm: bool = True,
) -> List[Dict[str, Any]]:
    results_file = output_dir / "all_results.jsonl"
    existing = set()
    if skip_existing and results_file.exists():
        with open(results_file) as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    existing.add((r["dataset"], r["config"]))

    try:
        x_train, y_train, x_test, y_test = load_ucr_dataset(dataset_name)
    except Exception as e:
        logger.error(f"  Failed to load {dataset_name}: {e}")
        return []

    n_classes = len(np.unique(y_train))
    logger.info(f"  train={x_train.shape}, test={x_test.shape}, classes={n_classes}, T={x_train.shape[1]}")

    results = []
    for cfg in configs:
        if (dataset_name, cfg.name) in existing:
            logger.info(f"  [{cfg.name}] skipping (done)")
            continue

        t0 = time.time()
        try:
            F_train, F_test = extract_features_gpu(x_train, x_test, cfg, device)
        except Exception as e:
            logger.error(f"  [{cfg.name}] features failed: {e}")
            result = {"dataset": dataset_name, "config": cfg.name, "accuracy": None, "error": str(e)}
            results.append(result)
            with open(results_file, "a") as f:
                f.write(json.dumps(result) + "\n")
            continue
        feat_time = time.time() - t0

        t0 = time.time()
        try:
            if use_gpu_lgbm:
                clf_result = classify_lgbm(F_train, y_train, F_test, y_test,
                                           n_classes=n_classes, use_gpu=True)
            else:
                clf_result = classify_ridge(F_train, y_train, F_test, y_test)
        except Exception as e:
            logger.error(f"  [{cfg.name}] classification failed: {e}")
            clf_result = {"accuracy": None, "error": str(e)}
        clf_time = time.time() - t0

        result = {
            "dataset": dataset_name, "config": cfg.name,
            "n_train": int(x_train.shape[0]), "n_test": int(x_test.shape[0]),
            "T": int(x_train.shape[1]), "n_classes": n_classes,
            "n_features": int(F_train.shape[1]),
            "feat_time_s": round(feat_time, 3), "clf_time_s": round(clf_time, 3),
            **clf_result,
        }
        results.append(result)
        with open(results_file, "a") as f:
            f.write(json.dumps(result) + "\n")

        acc_str = f"{result['accuracy']:.4f}" if result.get("accuracy") else "FAIL"
        logger.info(f"  [{cfg.name}] dims={F_train.shape[1]} acc={acc_str} feat={feat_time:.2f}s clf={clf_time:.2f}s")

    return results


def main():
    parser = argparse.ArgumentParser(description="MSRF UCR Ablation Study (GPU)")
    parser.add_argument("--datasets", nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--output-dir", default="msrf_results")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--configs", nargs="+", default=None)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lgbm", action="store_true", help="Use LightGBM (GPU) instead of RidgeCV")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.aggregate_only:
        aggregate_results(output_dir)
        return

    logger.info(f"Device: {DEVICE}")
    dataset_names = get_dataset_list(args)
    logger.info(f"Datasets: {len(dataset_names)}")

    configs = ABLATION_CONFIGS
    if args.configs:
        cfg_set = set(args.configs)
        configs = [c for c in ABLATION_CONFIGS if c.name in cfg_set]
        if not configs:
            logger.error(f"Available: {[c.name for c in ABLATION_CONFIGS]}")
            sys.exit(1)

    logger.info(f"Configs: {[c.name for c in configs]}")

    for i, ds in enumerate(dataset_names):
        logger.info(f"\n{'='*60}\nDataset {i+1}/{len(dataset_names)}: {ds}\n{'='*60}")
        run_single_dataset(ds, configs, output_dir, DEVICE,
                           skip_existing=args.resume, use_gpu_lgbm=args.lgbm)

    logger.info(f"\n{'='*60}\nAggregating...\n{'='*60}")
    aggregate_results(output_dir)


if __name__ == "__main__":
    main()
