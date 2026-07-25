"""MSRF*C — Multi-Space Random Features, revised configurations.

- MultiSpaceEncCore : the universal multi-space encoder (MSRF+, 519 dims; zero fitted parameters)
- make_banks/battery: the second-level multi-space battery (MSRF* = 519 + 241 = 760 dims)
- MSRFC             : the headline classifier — compact canonical MiniRocket dictionary
                      (1,512 kernels, via aeon) + MSRF* embedding = 2,272 dims, ridge head.
"""
import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

from .msenc_encoder_np import MultiSpaceEncCore
from .msrf_battery import make_banks, battery

__all__ = ["MultiSpaceEncCore", "make_banks", "battery", "MSRFC", "msrf_star"]


def msrf_star(X, enc=None, scaler=None, banks=None):
    """MSRF* features (760 dims) for X: (n, length). Returns (F, scaler, banks) so a train-fit
    scaler can be reused on test data."""
    enc = enc or MultiSpaceEncCore()
    E = enc.transform(X)
    if scaler is None:
        scaler = StandardScaler().fit(E)
    Z = scaler.transform(E)
    if banks is None:
        banks = make_banks(Z.shape[1])
    return np.concatenate([Z, battery(Z, banks)], 1), scaler, banks


class MSRFC:
    """MiniRocket-1512 (aeon) + MSRF* (760d) -> 2,272 dims, RidgeClassifierCV head.

    >>> clf = MSRFC().fit(X_train, y_train)     # X: (n_series, length)
    >>> acc = clf.score(X_test, y_test)
    """

    def __init__(self, n_kernels=1512, alphas=None, random_state=0):
        from aeon.transformations.collection.convolution_based import MiniRocket
        self._mr = MiniRocket(n_kernels=n_kernels, random_state=random_state)
        self._enc = MultiSpaceEncCore()
        self._alphas = np.logspace(-3, 3, 13) if alphas is None else alphas
        self._scaler = self._banks = self._head_scaler = self._head = None

    def transform(self, X, train=False):
        X = np.asarray(X, dtype=np.float64)
        M = np.nan_to_num(np.asarray(
            self._mr.fit_transform(X[:, None, :]) if train else self._mr.transform(X[:, None, :])))
        if train:
            S, self._scaler, self._banks = msrf_star(X, self._enc)
        else:
            S, _, _ = msrf_star(X, self._enc, self._scaler, self._banks)
        return np.concatenate([M, S], 1)

    def fit(self, X, y):
        F = self.transform(X, train=True)
        self._head_scaler = StandardScaler().fit(F)
        self._head = RidgeClassifierCV(alphas=self._alphas).fit(self._head_scaler.transform(F), y)
        return self

    def predict(self, X):
        return self._head.predict(self._head_scaler.transform(self.transform(X)))

    def score(self, X, y):
        return float((self.predict(X) == np.asarray(y)).mean())
