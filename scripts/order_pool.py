"""Order-sensitive pooling operators for the frozen MSRF+ encoder.

WHY. MultiSpaceEncCore pools the per-patch feature sequence p: (n_patches, 173) with
mean||std||max. All three are symmetric functions of the patch multiset, so the 519-d
embedding is INVARIANT to permuting the patches: the encoder cannot see temporal
rearrangement above the patch scale (L=64, stride 16). The per-patch formulas are
order-sensitive only WITHIN a patch. MultiRocket's edge over MiniRocket comes partly from
its order-sensitive pools (LSPV, MIPV), which is the same gap.

WHAT. Three zero-parameter, order-sensitive pooling operators, adapted from the
change-point/regime genome (pre/post contrast, `zr_*_rmax` envelope, `cs_*_csrange`
CUSUM). Each is deliberately NORMALISED so it carries position information that mean/std/max
cannot already express -- otherwise the attribution is confounded by magnitude:

  contrast   (b-a)/(|a|+|b|+eps) on first-half vs second-half patch means.
             Scale-free and bounded in [-1,1] by the triangle inequality.
  earliness  mean_t(cummax_t |p|) / max_t|p| in (0,1]. ~1 iff the extreme arrives at t=0,
             ~1/n iff it arrives last. Pure arrival-time coordinate; the genome's raw
             `.abs().cummax().mean()` is left un-normalised and is then largely predictable
             from the existing max pool, which would muddy the ablation.
  cusum      (max_t S_t - min_t S_t)/std(p), S_t = cumsum(p - mean(p))/sqrt(t).
             Excursion range of the normalised centred partial sums -- a Kolmogorov-Smirnov /
             CUSUM statistic on the patch sequence. Centring is our change from the genome
             (which does not centre): without it the block is dominated by mean(p) and adds
             little beyond the mean pool. The sqrt(t) scaling is the genome's, and weights
             early excursions more than a Brownian-bridge sqrt(n) would.

n_patches < 2 carries no order information (T <= 64 is interpolated to one patch), so the
blocks return their degenerate constants there and StandardScaler zeroes them out.

CAVEAT. MultiSpaceEncCore.transform concatenates the patches of ALL channels along axis 0
before pooling. For univariate input (UCR: X is (N, T)) that array IS the time-ordered patch
sequence, which is the case these operators are defined for. For multivariate input the
concatenation would make "order" run across channel boundaries; pool per channel first.
"""
import numpy as np

from msrfc.msenc_encoder_np import MultiSpaceEncCore

BLOCKS = ("contrast", "earliness", "cusum")


def order_blocks(p, eps=1e-8):
    """p: (n_patches, n_cols) -> {name: (n_cols,)}. Each block is order-sensitive."""
    n, d = p.shape
    if n < 2:
        return {"contrast": np.zeros(d), "earliness": np.ones(d), "cusum": np.zeros(d)}
    half = n // 2
    a, b = p[:half].mean(0), p[half:].mean(0)
    ap = np.abs(p)
    S = (p - p.mean(0)).cumsum(0) / np.sqrt(np.arange(1, n + 1))[:, None]
    return {
        "contrast": (b - a) / (np.abs(a) + np.abs(b) + eps),
        "earliness": np.maximum.accumulate(ap, 0).mean(0) / (ap.max(0) + eps),
        "cusum": (S.max(0) - S.min(0)) / (p.std(0) + eps),
    }


class OrderPoolEnc(MultiSpaceEncCore):
    """MultiSpaceEncCore with the order-sensitive blocks appended.

    Layout: [mean | std | max | contrast | earliness | cusum], 6 * 173 = 1038 dims.
    Columns 0:519 are byte-identical to the released MSRF+519 encoder, so one encode pass
    serves every ablation arm -- slice with `cols()`.
    """

    def _pool(self, p):
        base = super()._pool(p)
        blk = order_blocks(p)
        return np.concatenate([base] + [blk[b] for b in BLOCKS])

    @property
    def output_dim(self):
        return super().output_dim + len(BLOCKS) * self.n_cols


def cols(n_cols, *blocks, base=True):
    """Column index array for an ablation arm: cols(173, 'cusum') -> base 519 + cusum block."""
    idx = list(range(3 * n_cols)) if base else []
    for b in blocks:
        s = (3 + BLOCKS.index(b)) * n_cols
        idx += list(range(s, s + n_cols))
    return np.asarray(idx, dtype=int)


# ---------------------------------------------------------------------------
# Finer patch grid for the order blocks only.
#
# The ablation shows the order pools LOSE accuracy at n_patch<=8 and win at n_patch 9-24.
# That threshold is a property of the PATCH GRID, not of the data: with L=64 and stride 16 a
# T=96 series yields 3 patches, so a scalar order summary has almost nothing to summarise.
# L is frozen (the banks expect length-64 patches), so the only free knob is the grid: resample
# short series up until `target` patches of length 64 fit at stride 16, then patchify.
# Resampling manufactures no information -- it re-exposes existing temporal structure on a grid
# the pooling layer can actually see. The base 519 columns keep the released coarse grid, so the
# arms stay comparable and columns 0:519 stay byte-identical.
# ---------------------------------------------------------------------------
from msrfc.msenc_encoder_np import L as _L, patchify as _patchify

_TARGET = 24


def fine_patchify(v, target=_TARGET, stride=16):
    v = np.asarray(v, float)
    need = _L + (target - 1) * stride
    if len(v) < need and len(v) >= 2:
        v = np.interp(np.linspace(0, len(v) - 1, need), np.arange(len(v)), v)
    return _patchify(v, stride=stride)


class FineOrderPoolEnc(MultiSpaceEncCore):
    """Base 519 on the released coarse grid + order blocks on a >=`target`-patch grid.

    Same 1038-column layout as OrderPoolEnc, so `cols()` slices identically.
    """

    def __init__(self, *a, target=_TARGET, **kw):
        super().__init__(*a, **kw)
        self.target = target

    def transform(self, X):
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = [[row] for row in X]
        base = super().transform(X)
        pats, bounds = [], []
        for inst in X:
            chans = inst if isinstance(inst, (list, tuple)) else list(inst)
            cnt = 0
            for ch in chans:
                ch = np.asarray(ch, float)
                z = (ch - ch.mean()) / (ch.std() + 1e-8)
                P = fine_patchify(z, self.target)
                pats.append(P)
                cnt += len(P)
            bounds.append(cnt)
        feats = self.tf.phi(np.concatenate(pats, 0))
        out, i = [], 0
        for cnt in bounds:
            blk = order_blocks(feats[i:i + cnt])
            out.append(np.concatenate([blk[b] for b in BLOCKS]))
            i += cnt
        return np.concatenate([base, np.asarray(out)], 1)


class FusedOrderPoolEnc(MultiSpaceEncCore):
    """Byte-identical to FineOrderPoolEnc but without the redundant phi pass.

    FineOrderPoolEnc calls super().transform() for the base 519 and then runs phi AGAIN on the
    fine grid for the order blocks. Whenever the series is long enough that no resampling happens
    (len >= need = 64 + (target-1)*16 = 432), the two grids are IDENTICAL, so that second phi call
    recomputes exactly the same features -- which is why the profile showed ~2x overhead at L=1024
    even though nothing is resampled there.

    Here phi runs once on the fine grid, and the base pooling is taken from those same features for
    every instance whose channels were all long enough. Short instances (where the grids genuinely
    differ) keep the two-pass path, so the base column stays bit-exact in every case.
    """

    def __init__(self, *a, target=_TARGET, **kw):
        super().__init__(*a, **kw)
        self.target = target
        self.need = _L + (target - 1) * 16

    def transform(self, X):
        if isinstance(X, np.ndarray) and X.ndim == 2:
            X = [[row] for row in X]
        fine, fb, short = [], [], []
        for k, inst in enumerate(X):
            chans = inst if isinstance(inst, (list, tuple)) else list(inst)
            cnt, is_short = 0, False
            for ch in chans:
                ch = np.asarray(ch, float)
                z = (ch - ch.mean()) / (ch.std() + 1e-8)
                if len(z) < self.need:
                    is_short = True
                P = fine_patchify(z, self.target)
                fine.append(P)
                cnt += len(P)
            fb.append(cnt)
            if is_short:
                short.append(k)

        feats = self.tf.phi(np.concatenate(fine, 0))          # the ONLY phi call for long series
        base_rows, blk_rows, i = [None] * len(X), [], 0
        for k, cnt in enumerate(fb):
            seg = feats[i:i + cnt]
            i += cnt
            blk = order_blocks(seg)
            blk_rows.append(np.concatenate([blk[b] for b in BLOCKS]))
            if k not in short:
                base_rows[k] = self._pool(seg)                # reuse: grids coincide

        if short:                                            # exact fallback for short series
            sub = super().transform([X[k] for k in short])
            for j, k in enumerate(short):
                base_rows[k] = sub[j]
        return np.concatenate([np.asarray(base_rows), np.asarray(blk_rows)], 1)
