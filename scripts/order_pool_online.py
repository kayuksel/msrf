"""One-pass, O(1)-state variants of the order blocks -- can the repair run in a streaming deployment?

The blocks in order_pool.py are two-pass: `cusum` centres each patch feature on the mean over ALL
patches before accumulating, and `earliness` divides by the max over all patches. A device consuming
a sensor stream patch by patch does not have those totals until the window ends, so as written the
repair costs a second pass and O(n_patches) storage.

Every quantity here is instead a function of PREFIX statistics only, so each is computable
incrementally with O(1) state per feature column:

  running mean   m_t = (1/t) sum_{i<=t} p_i                      (Welford / cumulative)
  running std    s_t from the cumulative sum of squares
  cusum_online   range over t of  S_t = sum_{i<=t}(p_i - m_i)/sqrt(t),  divided by s_n
                 -- classic sequential CUSUM: the deviation is taken against the mean SO FAR,
                    not against a mean that depends on the future
  earliness      mean over t of runmax_t(|p|) / (runmax_n(|p|) + eps). VERIFIED IDENTICAL to the
                 two-pass block: max over all patches IS the final running max, so this one was
                 already streaming and the two-pass framing was simply wrong about it.
  contrast       (mean of 2nd half - mean of 1st half) needs the halfway index, so it is not
                 prefix-only for an unbounded stream -- but on-device the window length is FIXED
                 (L and the patch target are constants), so the halfway point is known in advance
                 and two running sums suffice. Also O(1) state in the deployed setting.

So `cusum` is the only block whose value actually changes when restricted to prefix statistics,
and this file exists to price that one substitution.

If accuracy is preserved the repair is free on-device, which is the claim worth making at an
industrial venue; if it degrades, the two-pass version is the one to report.
"""
import numpy as np

from order_pool import BLOCKS, _L, fine_patchify, order_blocks  # noqa: F401
from msrfc import MultiSpaceEncCore

EPS = 1e-8


def order_blocks_online(p, eps=EPS):
    """p: (n_patches, n_cols) -> {name: (n_cols,)}, using prefix statistics only.

    Returns the same three keys/widths as order_blocks so `cols()` slicing is unchanged.
    """
    n, d = p.shape
    if n < 2:
        return {"contrast": np.zeros(d), "earliness": np.ones(d), "cusum": np.zeros(d)}
    t = np.arange(1, n + 1, dtype=float)[:, None]
    csum = p.cumsum(0)
    m = csum / t                                    # running mean, prefix-only
    var = np.maximum(np.square(p).cumsum(0) / t - m ** 2, 0.0)
    s_n = np.sqrt(var[-1])                          # running std at the last patch
    S = (p - m).cumsum(0) / np.sqrt(t)              # deviation vs the mean SO FAR
    ap = np.abs(p)
    runmax = np.maximum.accumulate(ap, 0)
    half = n // 2
    a, b = p[:half].mean(0), p[half:].mean(0)       # still two-pass; reported as such
    return {
        "contrast": (b - a) / (np.abs(a) + np.abs(b) + eps),
        "earliness": runmax.mean(0) / (runmax[-1] + eps),
        "cusum": (S.max(0) - S.min(0)) / (s_n + eps),
    }


class OnlineOrderPoolEnc(MultiSpaceEncCore):
    """FineOrderPoolEnc with the one-pass blocks. Same 1038-column layout."""

    def __init__(self, *a, target=24, **kw):
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
                z = (ch - ch.mean()) / (ch.std() + EPS)
                P = fine_patchify(z, self.target)
                pats.append(P)
                cnt += len(P)
            bounds.append(cnt)
        feats = self.tf.phi(np.concatenate(pats, 0))
        out, i = [], 0
        for cnt in bounds:
            blk = order_blocks_online(feats[i:i + cnt])
            out.append(np.concatenate([blk[b] for b in BLOCKS]))
            i += cnt
        return np.concatenate([base, np.asarray(out)], 1)
