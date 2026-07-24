"""Torch-free the multi-space encoder encoder: patchify + per-patch phi + pooling + multivariate channel pooling.

Mirrors the reference implementation exactly, but with the numpy phi (no torch). Two pooling modes:
  * 'full'   -> mean||std||max over all patches of all channels = 3*173 = 519 dims (universal default)
  * 'pruned' -> selective per-family pooling (POOL_MAP) = 211 dims (compact, in-distribution-specialized)
"""
import os
import numpy as np
from msenc_np import WMTransformNP, L

_BANKS = None      # None -> embedded frozen banks (_banks_data.py); or pass an npz path/dict

# Discovered per-family pooling map (52-UCR leave-one-out prune over (family x pool) blocks); 519 -> 211.
POOL_MAP = {
    "srf_mlp": ["max"], "spectral": ["max"], "trf_gausswin": ["mean"],
    "crf_ppv": ["mean", "std"], "crf_max": ["mean"], "morphology_updown": ["std"],
    "perm_entropy": ["max"], "curvature": ["mean"], "conv_position": ["max"],
    "ar_residual": ["mean", "max"], "ricker_wavelet": ["mean"], "hydra_compete": ["mean", "std"],
}
_OPS = ("mean", "std", "max")


def patchify(v, stride=16, resample_short=True):
    v = np.asarray(v, float)
    if not np.isfinite(v).all():
        v = np.nan_to_num(v)
    if len(v) < L:
        if resample_short and len(v) >= 2:
            v = np.interp(np.linspace(0, len(v) - 1, L), np.arange(len(v)), v)
        else:
            v = np.pad(v, (L - len(v), 0), mode="edge")
    st = list(range(0, len(v) - L + 1, stride))
    if st[-1] != len(v) - L:
        st.append(len(v) - L)
    return np.stack([v[s:s + L] for s in st])


class MultiSpaceEncCore:
    """Frozen closed-form universal time-series encoder (torch-free). transform(X) -> (N, D)."""

    def __init__(self, banks=_BANKS, pooling="full", backend="numba"):
        if banks is None:
            from _banks_data import load_banks
            banks = load_banks()
        elif isinstance(banks, str):
            banks = dict(np.load(banks))
        if backend == "numba":
            from msenc_numba import WMTransformNB
            self.tf = WMTransformNB(banks)
        else:
            self.tf = WMTransformNP(banks)
        self.backend = backend
        self.families = list(WMTransformNP.FAMILIES)
        self.n_cols = self.tf.n_cols
        self.pooling = pooling

    def _pool(self, p):
        """Pool per-patch features (n_patches, 173) -> instance embedding. std uses ddof=0 (matches WMEncoder)."""
        if self.pooling == "full":
            return np.concatenate([p.mean(0), p.std(0), p.max(0)])
        cols, idx = [], 0
        for nm, w in self.families:
            seg = p[:, idx:idx + w]; idx += w
            keep = POOL_MAP.get(nm, [])
            if "mean" in keep: cols.append(seg.mean(0))
            if "std" in keep:  cols.append(seg.std(0))
            if "max" in keep:  cols.append(seg.max(0))
        return np.concatenate(cols)

    def _instance(self, channels):
        pats = [self.tf.phi(patchify((np.asarray(ch, float) - np.asarray(ch, float).mean())
                                     / (np.asarray(ch, float).std() + 1e-8))) for ch in channels]
        return self._pool(np.concatenate(pats, 0))

    def transform(self, X):
        """X: list of instances (each a list/2D-array of channels) or a 2-D (N, T) array.
        Batches all patches of all instances into ONE phi call (numba prange), then pools per instance."""
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = [[row] for row in X]
        all_pats, bounds = [], []
        for inst in X:
            inst = inst if isinstance(inst, (list, tuple)) else list(inst) if isinstance(inst, np.ndarray) else [inst]
            cnt = 0
            for ch in inst:
                ch = np.asarray(ch, float)
                z = (ch - ch.mean()) / (ch.std() + 1e-8)     # numpy ddof=0, matches WMEncoder
                P = patchify(z); all_pats.append(P); cnt += len(P)
            bounds.append(cnt)
        feats = self.tf.phi(np.concatenate(all_pats, 0))      # single batched phi over all patches
        out, i = [], 0
        for cnt in bounds:
            out.append(self._pool(feats[i:i + cnt])); i += cnt
        return np.asarray(out)

    @property
    def output_dim(self):
        if self.pooling == "full":
            return 3 * self.n_cols
        return sum(w * sum(o in POOL_MAP.get(nm, []) for o in _OPS) for nm, w in self.families)
