"""Second-level multi-space random-feature battery over an embedding (import-safe: no side effects).
Fixed integer seeds and nonlinearities; banks sized to the input dimension D.
"""
import numpy as np
import torch


def make_banks(D):
    g = lambda s: torch.Generator().manual_seed(s)
    B = {
        "A209": torch.randn(8, D, generator=g(209)),
        "W321": torch.randn(D, 8, generator=g(321)), "W322": torch.randn(D, 8, generator=g(322)),
        "W501": torch.randn(D, 16, generator=g(501)), "W701": torch.randn(D, 16, generator=g(701)),
        "W801": torch.randn(D, 16, generator=g(801)), "W951": torch.randn(D, 16, generator=g(951)),
        "W1901": torch.randn(D, 16, generator=g(1901)),
        "W201": torch.randn(D, 24, generator=g(201)), "b202": torch.rand(24, generator=g(202)) * 6.2831853,
        "W4011": torch.randn(D, 16, generator=g(4011)), "b4012": torch.rand(16, generator=g(4012)) * 6.2831853,
        "W5141": torch.randn(D, 20, generator=g(5141)),
        "W6041": torch.randn(D, 12, generator=g(6041)), "W6042": torch.randn(D, 12, generator=g(6042)),
        "A6091": torch.randn(12, D, generator=g(6091)),
        "W7011": torch.randn(D, 24, generator=g(7011)),
    }
    return {k: v.double().numpy() for k, v in B.items()}


def battery(Z, B):
    """Z: (n, D) train-standardized embedding -> (n, 241) nonlinear random features."""
    D = Z.shape[1]; s = np.sqrt(D)
    ssq = np.sign(Z) * np.sqrt(np.abs(Z))
    slg = np.sign(Z) * np.log1p(np.abs(Z))
    r = Z.argsort(1).argsort(1).astype(float) / (D - 1)
    xr = (r - 0.5) * np.sqrt(12.0)
    feats = [
        np.tanh(Z @ B["W501"] / s),
        np.tanh(ssq @ B["W801"] / s),
        np.tanh(slg @ B["W701"] / s),
        np.tanh((ssq * xr) @ B["W1901"] / s),
        np.cos(Z @ B["W201"] / s + B["b202"]),
        np.cos(xr @ B["W4011"] / (3 * s) + B["b4012"]),
        np.cos(xr @ B["W6041"] / s),
        np.cos(xr @ B["W6042"] / (3 * s)),
        np.maximum(xr @ B["W321"] / s, 0.0),
        (xr @ B["W322"] / s) ** 2,
        np.maximum(xr @ B["W7011"] / s, 0.0),
        1.0 / (1.0 + np.exp(-8.0 * (xr @ B["W951"] / s))),
        np.log1p(np.abs(Z @ B["W5141"] / s)),
        np.exp(-((Z[:, None, :] - B["A209"][None]) ** 2).mean(2)),
        np.exp(-((Z[:, None, :] - B["A6091"][None]) ** 2).mean(2)),
        np.exp(-(xr[:, :, None] - np.linspace(-2, 2, 8)) ** 2).mean(1),
        np.quantile(xr, np.linspace(0, 1, 9), axis=1).T,
    ]
    return np.concatenate(feats, 1)
