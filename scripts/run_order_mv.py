"""Per-channel order pooling: the multivariate case is wrong as written.

MultiSpaceEncCore.transform concatenates the patches of EVERY channel along axis 0 and pools the
stack. For mean/std/max that is deliberate and harmless -- they are symmetric, so pooling the union
of channels is exactly the channel-independent pooling the encoder claims. For the order blocks it
is a bug: with C channels of n patches each, `contrast` differences the mean of the first Cn/2 rows
against the last Cn/2, i.e. it compares the FIRST HALF OF THE CHANNELS against the second half, and
`cusum` accumulates across channel boundaries. On univariate data (C=1) the stack IS the patch
sequence, so every result reported elsewhere is unaffected; on multivariate data the blocks measure
channel order, which is arbitrary.

The fix keeps the encoder channel-symmetric: compute the three blocks per channel, then average
over channels. Averaging is the symmetric choice (concatenating would make the width depend on C
and destroy the frozen-encoder property, since UEA channel counts differ per dataset).

  mv_stacked     the blocks as currently written -- channel order leaks in
  mv_perchannel  blocks computed per channel, averaged over channels

Dataset caps (n<=600, T<=1300, channels<=65) are the paper's own multivariate-transfer protocol,
reused unchanged so this table is comparable with the published transfer results rather than being
a differently-scoped set. Per-channel pooling costs one phi call per channel, so the channel cap
also keeps DuckDuckGeese (1,345 channels) from dominating the runtime.

  python3 run_order_mv.py --cache ~/Desktop/worldmodel/rebuttal_sweeps/uea_cache
"""
import argparse
import json
import os
import sys
import time
import warnings

import numpy as np

warnings.filterwarnings("ignore")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import ucr_io
from msrfc import MultiSpaceEncCore
from order_pool import BLOCKS, cols, fine_patchify, order_blocks

RESULTS = "order_mv_results.jsonl"
EPS = 1e-8


def load_mv_ts(name, cache):
    """Multivariate .ts reader: channels are ':'-separated, the label is the last field."""
    def split(tag):
        rows, started = [], False
        for raw in open(os.path.join(cache, name, f"{name}_{tag}.ts")):
            line = raw.strip()
            if not started:
                started = line.lower() == "@data"
                continue
            if not line or line.startswith(("#", "@")) or ":" not in line:
                continue
            parts = line.split(":")
            chans = [[float("nan") if t.strip() in ("?", "", "NaN") else float(t)
                      for t in c.split(",")] for c in parts[:-1]]
            rows.append((chans, parts[-1].strip()))
        T = min(min(len(c) for c in ch) for ch, _ in rows)
        X = np.nan_to_num(np.array([[c[:T] for c in ch] for ch, _ in rows], dtype=np.float64))
        return X, [y for _, y in rows]

    Xtr, ytr = split("TRAIN")
    Xte, yte = split("TEST")
    m = {c: i for i, c in enumerate(sorted(set(ytr)))}
    return Xtr, np.array([m[l] for l in ytr]), Xte, np.array([m.get(l, -1) for l in yte])


class MVOrderPoolEnc(MultiSpaceEncCore):
    """Order blocks either over the channel-concatenated stack (per_channel=False, the bug) or
    per channel then averaged (per_channel=True, the fix). Base 519 columns are identical either
    way -- they come from super().transform()."""

    def __init__(self, *a, target=24, per_channel=True, **kw):
        super().__init__(*a, **kw)
        self.target, self.per_channel = target, per_channel

    def transform(self, X):
        base = super().transform(X)
        out = []
        for inst in X:
            grids = []
            for ch in inst:
                ch = np.asarray(ch, float)
                z = (ch - ch.mean()) / (ch.std() + EPS)
                grids.append(fine_patchify(z, self.target))
            if self.per_channel:
                per = [order_blocks(self.tf.phi(g)) for g in grids]
                blk = {b: np.mean([p[b] for p in per], axis=0) for b in BLOCKS}
            else:
                blk = order_blocks(self.tf.phi(np.concatenate(grids, 0)))
            out.append(np.concatenate([blk[b] for b in BLOCKS]))
        return np.concatenate([base, np.asarray(out)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=os.path.expanduser(
        "~/Desktop/worldmodel/rebuttal_sweeps/uea_cache"))
    ap.add_argument("--max-n", type=int, default=600)
    ap.add_argument("--max-t", type=int, default=1300)
    ap.add_argument("--max-ch", type=int, default=65)
    a = ap.parse_args()
    names = sorted(d for d in os.listdir(a.cache) if os.path.isdir(os.path.join(a.cache, d)))
    path = os.path.join(ucr_io.RESULTS, RESULTS)
    done = {json.loads(l)["dataset"] for l in open(path)} if os.path.exists(path) else set()
    encs = {"stacked": MVOrderPoolEnc(per_channel=False), "perchannel": MVOrderPoolEnc(per_channel=True)}
    nc = encs["stacked"].n_cols
    c_base, c_all3 = cols(nc), cols(nc, *BLOCKS)
    for name in names:
        if name in done:
            continue
        try:
            Xtr, ytr, Xte, yte = load_mv_ts(name, a.cache)
        except Exception as e:
            print(f"skip {name}: {e}", flush=True)
            continue
        n, nch, T = Xtr.shape
        if n > a.max_n or T > a.max_t or nch > a.max_ch:
            print(f"skip {name}: n={n} ch={nch} T={T} outside protocol caps", flush=True)
            continue
        t0 = time.time()
        row = dict(n_train=int(Xtr.shape[0]), C_cls=int(len(np.unique(ytr))),
                   n_ch=int(Xtr.shape[1]), T=int(Xtr.shape[2]))
        for tag, enc in encs.items():
            Etr, Ete = enc.transform(Xtr), enc.transform(Xte)
            if tag == "stacked":
                row["base"] = ucr_io.clf(Etr[:, c_base], ytr, Ete[:, c_base], yte)
            row[tag] = ucr_io.clf(Etr[:, c_all3], ytr, Ete[:, c_all3], yte)
        row["sec"] = round(time.time() - t0, 1)
        ucr_io.update_results(name, row, results_file=RESULTS)
        print(f"{name:28s} ch={row['n_ch']:3d} T={row['T']:5d} | base {row['base']:.3f}  "
              f"stacked {row['stacked']:.3f}  perchannel {row['perchannel']:.3f}  ({row['sec']:.0f}s)",
              flush=True)


if __name__ == "__main__":
    main()
