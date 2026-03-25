"""
MSRF transform and classification pipeline.

Provides ``MSRFTransform`` for feature extraction and a convenience function
``msrf_classify`` that runs the full pipeline (extract → scale → Ridge).
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from .extractors import CRFExtractor, GRFExtractor, SRFExtractor, TRFExtractor


class MSRFTransform(nn.Module):
    """Full MSRF feature extraction pipeline.

    Combines Temporal, Global, Statistical, and Convolutional Random Features
    into a single feature vector per time series.

    Parameters
    ----------
    T : int
        Length of the input time series.
    n_trf : int
        Number of temporal random features.
    n_grf_projections : int
        Number of GRF projection matrices (yields 2 features each).
    grf_dim : int
        Embedding dimension for each GRF projection.
    n_srf : int
        Number of statistical random features.
    n_crf_kernels : int
        Number of convolutional kernels.
    crf_transforms : list of str
        Transforms for CRF (e.g., ["identity"], ["identity", "diff"]).
    crf_pool_ops : list of str
        Pooling operations for CRF (e.g., ["ppv", "max"]).
    seed : int
        Base random seed (each extractor offsets from this).
    """

    def __init__(
        self,
        T: int,
        n_trf: int = 200,
        n_grf_projections: int = 5,
        grf_dim: int = 256,
        n_srf: int = 200,
        n_crf_kernels: int = 500,
        crf_transforms: Optional[List[str]] = None,
        crf_pool_ops: Optional[List[str]] = None,
        seed: int = 42,
    ):
        super().__init__()
        self.extractors = nn.ModuleList()

        if n_trf > 0:
            self.extractors.append(TRFExtractor(T, K=n_trf, seed=seed))
        if n_grf_projections > 0:
            self.extractors.append(
                GRFExtractor(T, n_projections=n_grf_projections, D=grf_dim, seed=seed + 100)
            )
        if n_srf > 0:
            self.extractors.append(SRFExtractor(K=n_srf, seed=seed + 200))
        if n_crf_kernels > 0:
            self.extractors.append(
                CRFExtractor(
                    input_length=T,
                    num_kernels=n_crf_kernels,
                    transforms=crf_transforms or ["identity"],
                    pool_ops=crf_pool_ops or ["ppv", "max"],
                    seed=seed + 300,
                )
            )

    @property
    def n_features(self) -> str:
        return "call forward() to determine exact count (depends on CRF config)"

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Parameters
        ----------
        x : Tensor of shape (N, T)

        Returns
        -------
        Tensor of shape (N, K) where K is the total feature count.
        """
        parts = [ext(x) for ext in self.extractors]
        return torch.cat(parts, dim=1) if parts else x.new_empty(x.shape[0], 0)

    @torch.no_grad()
    def transform_numpy(
        self,
        x: np.ndarray,
        device: Optional[torch.device] = None,
        batch_size: int = 4096,
    ) -> np.ndarray:
        """Extract features from a numpy array, handling batching and device transfer.

        Parameters
        ----------
        x : ndarray of shape (N, T)
        device : torch device (default: cuda if available, else cpu)
        batch_size : int

        Returns
        -------
        ndarray of shape (N, K)
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)
        self.eval()

        x_t = torch.from_numpy(x.astype(np.float32))
        parts = []
        for start in range(0, x_t.shape[0], batch_size):
            batch = x_t[start : start + batch_size].to(device)
            parts.append(self.forward(batch).cpu())
        return torch.cat(parts, dim=0).numpy()


def msrf_classify(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_trf: int = 200,
    n_grf_projections: int = 5,
    grf_dim: int = 256,
    n_srf: int = 200,
    n_crf_kernels: int = 500,
    crf_transforms: Optional[List[str]] = None,
    crf_pool_ops: Optional[List[str]] = None,
    seed: int = 42,
    alphas: Optional[np.ndarray] = None,
    device: Optional[torch.device] = None,
) -> dict:
    """Run the full MSRF pipeline: extract features → standardize → Ridge classify.

    Parameters
    ----------
    X_train, X_test : ndarray of shape (N, T)
        Raw time series.
    y_train, y_test : ndarray of shape (N,)
        Class labels.
    n_trf, n_grf_projections, grf_dim, n_srf, n_crf_kernels :
        Feature counts (see ``MSRFTransform``).
    crf_transforms, crf_pool_ops :
        CRF configuration.
    seed : int
        Random seed.
    alphas : ndarray, optional
        Regularization values for RidgeClassifierCV.
    device : torch.device, optional

    Returns
    -------
    dict with keys: accuracy, n_features, alpha, y_pred
    """
    T = X_train.shape[1]
    transform = MSRFTransform(
        T=T,
        n_trf=n_trf,
        n_grf_projections=n_grf_projections,
        grf_dim=grf_dim,
        n_srf=n_srf,
        n_crf_kernels=n_crf_kernels,
        crf_transforms=crf_transforms,
        crf_pool_ops=crf_pool_ops,
        seed=seed,
    )

    F_train = transform.transform_numpy(X_train, device=device)
    F_test = transform.transform_numpy(X_test, device=device)

    scaler = StandardScaler()
    F_train = np.nan_to_num(scaler.fit_transform(F_train), nan=0.0, posinf=0.0, neginf=0.0)
    F_test = np.nan_to_num(scaler.transform(F_test), nan=0.0, posinf=0.0, neginf=0.0)

    if alphas is None:
        alphas = np.logspace(-4, 4, 20)

    clf = RidgeClassifierCV(alphas=alphas)
    clf.fit(F_train, y_train)
    y_pred = clf.predict(F_test)
    acc = float((y_pred == y_test).mean())

    return {
        "accuracy": acc,
        "n_features": F_train.shape[1],
        "alpha": float(clf.alpha_),
        "y_pred": y_pred,
    }
