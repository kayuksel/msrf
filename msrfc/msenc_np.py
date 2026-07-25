"""Torch-free NumPy port of the multi-space patch encoder phi (the 19 closed-form families).

This is a faithful, dependency-free (numpy-only) reimplementation of the reference implementation.
It is written to be *numerically identical* to the torch version (see contrib/test_parity.py), which
matters because the published results are tied to the specific seeded banks of the discovered
reference implementation. The banks are therefore frozen constants (extracted once from the torch seeds) rather than
regenerated from a numpy RNG.

torch-semantics matched explicitly:
  * std / var use the UNBIASED (ddof=1) estimator (torch default), not numpy's ddof=0.
  * median uses torch's LOWER median (sorted[(n-1)//2]), not numpy's averaged median.
  * quantile uses linear interpolation (both libraries' default).
  * conv1d is cross-correlation with 'same'-length padding, matching torch.nn.functional.conv1d.
  * rfft/fft via numpy.fft (bit-compatible with torch.fft for these sizes).

The vectorized-numpy form here is the reference; contrib/msenc_numba.py adds the njit kernels.
"""
import math
import numpy as np

L = 64
_EPS = 1e-8


# --------------------------------------------------------------------------------------- helpers
def _std(x, axis):                       # torch default: unbiased (ddof=1)
    return np.std(x, axis=axis, ddof=1)


def _var(x, axis):
    return np.var(x, axis=axis, ddof=1)


def _lower_median(x, axis):              # torch.median returns the lower of the two middles
    n = x.shape[axis]
    return np.partition(x, (n - 1) // 2, axis=axis).take((n - 1) // 2, axis=axis)


def _conv1d(w, W, dilation, padding):
    """Cross-correlation matching F.conv1d(w[:,None,:], W, padding=padding, dilation=dilation).
    w: (B, L). W: (Cout, 1, k). returns (B, Cout, out_len)."""
    B, Lw = w.shape
    Cout, _, k = W.shape
    xp = np.pad(w, ((0, 0), (padding, padding)))
    out_len = xp.shape[1] - dilation * (k - 1)
    out = np.zeros((B, Cout, out_len), dtype=w.dtype)
    for j in range(k):
        seg = xp[:, j * dilation: j * dilation + out_len]          # (B, out_len)
        out += W[None, :, 0, j, None] * seg[:, None, :]            # (1,Cout,1)*(B,1,out_len)
    return out


class WMTransformNP:
    """Frozen multi-space patch encoder phi: (B, L) -> (B, 173), pure numpy over frozen seeded banks."""

    FAMILIES = [
        ("stats", 12), ("srf_mlp", 12), ("autocorr", 2), ("spectral", 2), ("turning", 2),
        ("trf_gausswin", 12), ("crf_ppv", 32), ("hilbert_env", 2), ("crf_max", 32),
        ("morphology_updown", 4), ("fftbands", 6), ("perm_entropy", 3), ("curvature", 3),
        ("conv_position", 3), ("ar_residual", 4), ("acf_first_min", 3), ("histogram_mode", 3),
        ("ricker_wavelet", 4), ("hydra_compete", 32),
    ]

    def __init__(self, banks, L=L):
        self.L = L
        # frozen seeded random-projection banks (numpy float64)
        self.trf_mu = np.asarray(banks["trf_mu"], float)        # (12,)
        self.trf_sig = np.asarray(banks["trf_sig"], float)      # (12,)
        self.srf_W = np.asarray(banks["srf_W"], float)          # (12, 6, 12)
        self.srf_b = np.asarray(banks["srf_b"], float)          # (12, 6)
        self.srf_u = np.asarray(banks["srf_u"], float)          # (12, 6)
        self.crf_w = np.asarray(banks["crf_w"], float)          # (16, 1, 9)
        self.hydra_w = np.asarray(banks["hydra_w"], float)      # (2, 4, 9) per-kernel normalized

    # ----- family 0: stats (12) -----
    def stats(self, w):
        m = w.mean(1)
        s = np.maximum(_std(w, 1), _EPS)
        c = w - w.mean(1, keepdims=True)
        q25 = np.quantile(w, 0.25, axis=1)
        q50 = np.quantile(w, 0.50, axis=1)
        q75 = np.quantile(w, 0.75, axis=1)
        ac1 = (c[:, :-1] * c[:, 1:]).sum(1) / (
            np.sqrt((c[:, :-1] ** 2).sum(1)) * np.sqrt((c[:, 1:] ** 2).sum(1)) + _EPS)
        return np.stack([
            m, s,
            (c ** 3).mean(1) / (s ** 3 + _EPS),
            (c ** 4).mean(1) / (s ** 4 + _EPS) - 3.0,
            q25, q50, q75, q75 - q25,
            ac1,
            np.abs(w[:, 1:] - w[:, :-1]).mean(1), w.min(1), w.max(1),
        ], axis=1)

    # ----- family 1: random-MLP projection of stats (12) -----
    def srf_mlp(self, w, st):
        h = np.maximum(np.einsum("nm,kdm->nkd", st, self.srf_W) + self.srf_b[None], 0.0)
        return (h * self.srf_u[None]).sum(2)

    # ----- family 2: autocorr lags 1,2 (2) -----
    def autocorr(self, w):
        c = w - w.mean(1, keepdims=True)
        return np.stack([
            (c[:, :-1] * c[:, 1:]).sum(1) / ((c[:, :-1] ** 2).sum(1) + _EPS),
            (c[:, :-2] * c[:, 2:]).sum(1) / ((c[:, :-2] ** 2).sum(1) + _EPS),
        ], axis=1)

    # ----- family 3: spectral centroid + entropy (2) -----
    def spectral(self, w):
        p = np.abs(np.fft.rfft(w, axis=1))[:, 1:]
        p = p / (p.sum(1, keepdims=True) + _EPS)
        f = np.arange(p.shape[1], dtype=float)
        return np.stack([(f[None, :] * p).sum(1), -(p * np.log(p + 1e-12)).sum(1)], axis=1)

    # ----- family 4: turning-point + frac-positive-diff (2) -----
    def turning(self, w):
        dw = w[:, 1:] - w[:, :-1]
        return np.stack([(dw[:, 1:] * dw[:, :-1] < 0).mean(1), (dw > 0).mean(1)], axis=1)

    # ----- family 5: Gaussian-window weighted means at 12 centers (12) -----
    def trf_gausswin(self, w):
        t = np.arange(w.shape[1], dtype=float) / w.shape[1]
        win = np.exp(-((t[None, :] - self.trf_mu[:, None]) ** 2) / (2 * self.trf_sig[:, None] ** 2 + _EPS))
        win = win / (win.sum(1, keepdims=True) + _EPS)
        return np.einsum("bl,ml->bm", w, win)               # == w @ win.T (einsum avoids numpy-2.x matmul FPE flag)

    # ----- family 6: conv PPV at dilations 2,4 (32) -----
    def crf_ppv(self, w):
        kw = self.crf_w.shape[2]
        o1 = _conv1d(w, self.crf_w, dilation=2, padding=2 * (kw // 2))
        o2 = _conv1d(w, self.crf_w, dilation=4, padding=4 * (kw // 2))
        return np.concatenate([(o1 > 0).mean(2), (o2 > 0).mean(2)], axis=1)

    # ----- family 7: Hilbert envelope stats (2) -----
    def hilbert_env(self, w):
        n = w.shape[1]
        k = np.arange(n, dtype=float)
        h = np.where(k == 0, 1.0, np.where(k < n / 2, 2.0, np.where(k == n // 2, 1.0, 0.0)))
        env = np.abs(np.fft.ifft(np.fft.fft(w, axis=1) * h[None, :], axis=1))
        return np.stack([_std(env, 1) / (env.mean(1) + _EPS),
                         np.abs(env[:, 1:] - env[:, :-1]).mean(1) / (env.mean(1) + _EPS)], axis=1)

    # ----- family 8: conv max-pool at dilations 2,4 (32) -----
    def crf_max(self, w):
        kw = self.crf_w.shape[2]
        return np.concatenate([
            _conv1d(w, self.crf_w, dilation=d, padding=d * (kw // 2)).max(2) for d in (2, 4)], axis=1)

    # ----- family 9: up/down morphology asymmetry (4) -----
    def morphology_updown(self, w):
        dw = w[:, 1:] - w[:, :-1]
        p = np.clip(dw, 0, None)
        nn = np.clip(-dw, 0, None)
        return np.stack([
            p.max(1), nn.max(1),
            np.log((p.max(1) + 1e-6) / (nn.max(1) + 1e-6)),
            np.log(((np.clip(dw, 0, None) ** 2).sum(1) + 1e-6) / ((np.clip(dw, None, 0) ** 2).sum(1) + 1e-6)),
        ], axis=1)

    # ----- family 10: binned FFT log band power, 6 bands (6) -----
    def fftbands(self, w):
        p = (np.abs(np.fft.rfft(w, axis=1))[:, 1:]) ** 2
        pn = p / (p.sum(1, keepdims=True) + _EPS)
        bands = [pn[:, i:i + 6].sum(1) for i in range(0, pn.shape[1], 6)]
        return np.log(np.clip(np.stack(bands, axis=1), 1e-8, None))

    # ----- family 11: permutation entropy + Hjorth mobility/complexity (3) -----
    def perm_entropy(self, w):
        code = ((w[:, :-2] < w[:, 1:-1]).astype(np.int64) * 4
                + (w[:, 1:-1] < w[:, 2:]).astype(np.int64) * 2
                + (w[:, :-2] < w[:, 2:]).astype(np.int64))
        H = np.stack([(code == kk).mean(1) for kk in range(8)], axis=1)
        d1 = w[:, 1:] - w[:, :-1]
        d2 = w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]
        mob = np.sqrt((_var(d1, 1) + _EPS) / (_var(w, 1) + _EPS))
        comp = np.sqrt((_var(d2, 1) + _EPS) / (_var(d1, 1) + _EPS)) / (mob + _EPS)
        return np.stack([-(H * np.log(H + 1e-12)).sum(1) / math.log(6.0), mob, comp], axis=1)

    # ----- family 12: curvature via 2nd difference (3) -----
    def curvature(self, w):
        c = w[:, 2:] - 2 * w[:, 1:-1] + w[:, :-2]
        return np.stack([np.abs(c).mean(1), np.abs(c).max(1),
                         np.log(((np.clip(c, 0, None) ** 2).sum(1) + 1e-6)
                                / ((np.clip(c, None, 0) ** 2).sum(1) + 1e-6))], axis=1)

    # ----- family 13: conv max-response position summary (3) -----
    def conv_position(self, w):
        kw = self.crf_w.shape[2]
        pos = _conv1d(w, self.crf_w, dilation=2, padding=2 * (kw // 2)).argmax(2).astype(float) / w.shape[1]
        return np.stack([pos.mean(1), _std(pos, 1), pos.max(1) - pos.min(1)], axis=1)

    # ----- family 14: AR(2)+AR(3) ridge residual energy + coeff norm (4) -----
    def ar_residual(self, w):
        c = w - w.mean(1, keepdims=True)

        def fit(X, y):
            XT = np.transpose(X, (0, 2, 1))
            A = XT @ X + 1e-3 * np.eye(X.shape[2])
            beta = np.linalg.solve(A, XT @ y[:, :, None])[:, :, 0]
            resid = _var(y - (X @ beta[:, :, None])[:, :, 0], 1)
            return np.stack([np.log(resid + _EPS), np.log(np.maximum((beta ** 2).sum(1), _EPS))], axis=1)

        ar2 = fit(np.stack([c[:, 1:-1], c[:, :-2]], axis=2), c[:, 2:])
        ar3 = fit(np.stack([c[:, 2:-1], c[:, 1:-2], c[:, :-3]], axis=2), c[:, 3:])
        return np.concatenate([ar2, ar3], axis=1)

    # ----- family 15: ACF first-min / first-zero (3) -----
    def acf_first_min(self, w):
        c = w - w.mean(1, keepdims=True)
        n = w.shape[1]
        acf = np.stack([(c[:, :n - k] * c[:, k:]).sum(1) / ((c ** 2).sum(1) + _EPS) for k in range(1, 33)], axis=1)
        ismin = (acf[:, 1:-1] < acf[:, :-2]) & (acf[:, 1:-1] <= acf[:, 2:])
        has = ismin.any(1)
        fm = ismin.astype(np.float64).argmax(1)
        first_min = np.where(has, fm.astype(float) + 2.0, float(acf.shape[1])) / n
        first_zero = ((np.cumsum((acf < 0).astype(np.int64), axis=1).clip(max=1)
                       * np.arange(acf.shape[1], 0, -1)[None, :]).argmax(1)).astype(float) / n
        val_after = np.take_along_axis(acf, np.clip(fm + 1, None, acf.shape[1] - 1)[:, None], axis=1)[:, 0]
        return np.stack([first_min, first_zero, val_after], axis=1)

    # ----- family 16: soft histogram mode (3) -----
    def histogram_mode(self, w):
        z = (w - w.mean(1, keepdims=True)) / np.maximum(_std(w, 1)[:, None], 1e-6)
        ctr = np.linspace(-2.5, 2.5, 10)
        e = np.exp(-0.5 * ((ctr[None, None, :] - z[:, :, None]) / (5.0 / 9.0)) ** 2).sum(1)  # (B,10)
        logits = 4.0 * e
        soft = np.exp(logits - logits.max(1, keepdims=True))
        soft = soft / soft.sum(1, keepdims=True)
        m10 = (soft * ctr[None, :]).sum(1)
        return np.stack([m10, m10 - _lower_median(z, 1), (np.abs(z) < 0.5).mean(1)], axis=1)

    # ----- family 17: Ricker-wavelet PPV at 4 scales (4) -----
    def ricker_wavelet(self, w):
        x = np.linspace(-7.0, 7.0, 15)
        ks = []
        for s in (1.5, 2.5, 4.0, 6.0):
            r = (1.0 - (x / s) ** 2) * np.exp(-(x ** 2) / (2 * s * s))
            ks.append((r - r.mean()) / (np.abs(r).sum() + _EPS))
        W = np.stack(ks)[:, None, :]                       # (4,1,15)
        return (_conv1d(w, W, dilation=1, padding=15 // 2) > 0).mean(2)


    # ----- family 18: Hydra-style competing-kernel soft win-counts (32) -----
    def hydra_compete(self, w):
        cols = []
        for x in (w, w[:, 1:] - w[:, :-1]):
            for d in (2, 4):
                for g in range(self.hydra_w.shape[0]):
                    r = _conv1d(x, self.hydra_w[g][:, None, :], dilation=d, padding=d * 4)  # (B, 4, T)
                    arg = r.argmax(1)                                                        # (B, T)
                    win = np.clip(r.max(1), 0.0, None)                                       # (B, T)
                    soft = np.zeros((x.shape[0], 4), dtype=w.dtype)
                    for kk in range(4):
                        soft[:, kk] = (win * (arg == kk)).sum(1)
                    soft = soft / (soft.sum(1, keepdims=True) + 1e-8)
                    cols.append(np.sqrt(soft))
        return np.concatenate(cols, axis=1)

    # ----- assemble phi -----
    def phi(self, w):
        w = np.asarray(w, float)
        st = self.stats(w)
        cols = [
            st, self.srf_mlp(w, st),
            self.autocorr(w), self.spectral(w), self.turning(w),
            self.trf_gausswin(w), self.crf_ppv(w), self.hilbert_env(w), self.crf_max(w),
            self.morphology_updown(w), self.fftbands(w), self.perm_entropy(w), self.curvature(w),
            self.conv_position(w), self.ar_residual(w), self.acf_first_min(w), self.histogram_mode(w),
            self.ricker_wavelet(w), self.hydra_compete(w),
        ]
        cols = [c if c.ndim == 2 else c[:, None] for c in cols]
        return np.concatenate(cols, axis=1)

    __call__ = phi

    @property
    def n_cols(self):
        return sum(c for _, c in self.FAMILIES)
