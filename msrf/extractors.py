"""
Multi-Space Random Feature extractors for time series classification.

All extractors are torch.nn.Module subclasses designed for GPU execution.
Random parameters are sampled once at construction and frozen (no training).
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class TRFExtractor(nn.Module):
    """Temporal Random Features — Gaussian-windowed positional aggregation.

    Each feature computes a weighted average of the input signal using a
    Gaussian window centered at a random position with random width.

    Parameters
    ----------
    T : int
        Length of the input time series.
    K : int
        Number of temporal features to generate.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, T: int, K: int = 200, seed: int = 42):
        super().__init__()
        rng = np.random.RandomState(seed)
        mu = rng.uniform(0, T, size=K).astype(np.float32)
        sigma = np.exp(
            rng.uniform(np.log(max(1, T * 0.01)), np.log(max(2, T / 2)), size=K)
        ).astype(np.float32)
        self.register_buffer("mu", torch.from_numpy(mu))
        self.register_buffer("sigma", torch.from_numpy(sigma))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (N, T)

        Returns
        -------
        Tensor of shape (N, K)
        """
        T = x.shape[1]
        t = torch.arange(T, device=x.device, dtype=torch.float32)
        w = torch.exp(
            -((t[None, :] - self.mu[:, None]) ** 2) / (2 * self.sigma[:, None] ** 2)
        )
        w = w / (w.sum(dim=1, keepdim=True) + 1e-12)
        return x @ w.T


class GRFExtractor(nn.Module):
    """Global Random Features — moment-pooled Random Kitchen Sinks.

    Projects the full time series through random Fourier features cos(Wx + b),
    then extracts mean and standard deviation of the embedding as two scalar
    features per projection matrix.

    Parameters
    ----------
    T : int
        Length of the input time series.
    n_projections : int
        Number of independent projection matrices.
    D : int
        Embedding dimension per projection.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(self, T: int, n_projections: int = 5, D: int = 256, seed: int = 42):
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
        """
        Parameters
        ----------
        x : Tensor of shape (N, T)

        Returns
        -------
        Tensor of shape (N, 2 * n_projections) — mean and std per projection.
        """
        feats = []
        for i in range(self.n_projections):
            z = torch.cos(x @ self.Ws[i] + self.bs[i])
            feats.append(z.mean(dim=1, keepdim=True))
            feats.append(z.std(dim=1, keepdim=True))
        return torch.cat(feats, dim=1)


def compute_statistics(x: torch.Tensor) -> torch.Tensor:
    """Compute summary statistics for SRF input.

    Parameters
    ----------
    x : Tensor of shape (N, T)

    Returns
    -------
    Tensor of shape (N, 12) — mean, std, skew, kurtosis, q25, q50, q75,
    IQR, lag-1 autocorrelation, mean absolute difference, min, max.
    """
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
    return torch.stack(
        [mean, std, skew, kurt, q25, q50, q75, iqr, lag1, mean_abs_diff,
         x.min(1).values, x.max(1).values],
        dim=1,
    )


class SRFExtractor(nn.Module):
    """Statistical Random Features — two-layer random projection of summary statistics.

    Parameters
    ----------
    K : int
        Number of statistical features.
    d : int
        Hidden dimension of the two-layer random network.
    seed : int
        Random seed for reproducibility.
    M : int
        Number of input summary statistics (default 12 from ``compute_statistics``).
    """

    def __init__(self, K: int = 200, d: int = 8, seed: int = 42, M: int = 12):
        super().__init__()
        rng = np.random.RandomState(seed)
        self.register_buffer(
            "Ws", torch.from_numpy((rng.randn(K, d, M) / np.sqrt(M)).astype(np.float32))
        )
        self.register_buffer(
            "bs", torch.from_numpy(rng.randn(K, d).astype(np.float32))
        )
        self.register_buffer(
            "us", torch.from_numpy((rng.randn(K, d) / np.sqrt(d)).astype(np.float32))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (N, T) — raw time series.

        Returns
        -------
        Tensor of shape (N, K)
        """
        stats = compute_statistics(x)
        h = torch.einsum("nm,kdm->nkd", stats, self.Ws) + self.bs.unsqueeze(0)
        h = F.relu(h)
        return (h * self.us.unsqueeze(0)).sum(dim=2)


class CRFExtractor(nn.Module):
    """Convolutional Random Features — MiniRocket-style GPU implementation.

    Supports configurable transforms (identity, diff) and pooling operations
    (ppv, max, mpv, mipv) for different feature levels.

    Parameters
    ----------
    input_length : int
        Length of the input time series.
    num_kernels : int
        Number of random convolutional kernels.
    kernel_size : int
        Length of each kernel.
    transforms : list of str
        Input transforms to apply before convolution.
    pool_ops : list of str
        Pooling operations to extract per kernel.
    seed : int
        Random seed for reproducibility.
    """

    def __init__(
        self,
        input_length: int,
        num_kernels: int = 500,
        kernel_size: int = 9,
        transforms: Optional[List[str]] = None,
        pool_ops: Optional[List[str]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.num_kernels = num_kernels
        self.kernel_size = kernel_size
        self.transforms = transforms or ["identity"]
        self.pool_ops = pool_ops or ["ppv", "max"]
        self.eps = 1e-8

        torch.manual_seed(seed)
        self.register_buffer("base_w", torch.randn(num_kernels, kernel_size))
        self.register_buffer(
            "dilations",
            torch.randint(1, max(2, input_length // 2), (num_kernels,)),
        )
        self.register_buffer("kmasks", torch.ones(num_kernels, 1, kernel_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (N, T)

        Returns
        -------
        Tensor of shape (N, num_kernels * len(transforms) * len(pool_ops))
        """
        B, T = x.shape
        x_3d = x.unsqueeze(1)
        mask = torch.ones(B, 1, T, device=x.device)
        feats = []

        for tr in self.transforms:
            if tr == "diff":
                rep = F.pad(x_3d[..., 1:] - x_3d[..., :-1], (0, 1))
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
                    else:
                        continue
                    feats.append(feat.clamp(min=self.eps, max=1e6))

        return torch.cat(feats, dim=1)
