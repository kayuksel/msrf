# MSRF — Multi-Space Random Features for Time Series Classification

**This repository presents the revised method and its full evaluation.** The headline
configuration, **MSRF\*C**, combines a compact canonical MiniRocket dictionary (1,512 kernels)
with the framework's universal multi-space embedding (760 dims) — 2,272 dimensions total:

- **significantly more accurate than Hydra** (+0.008, Wilcoxon p=0.042) at **44% of its
  dimensions** and **~18× its transform speed** (0.28 vs 5.2 ms/series, single core);
- **statistically tied with full canonical MiniRocket** (+0.003, p=0.17) at **23% of its
  dimensions**;
- second overall on the archive, behind only MultiRocket (which uses 22× the dimensions and
  23× the transform time);
- the multi-space block's contribution is isolated and significant: **+0.008 over the compact
  MiniRocket dictionary alone (p=5.7e-5, W/L 63/24)** — diversity of feature spaces, not more
  kernels, closes the gap.

![Accuracy vs dimension frontier](plots/fig_pareto.png)

## Main result

All 113 equal-length UCR datasets, one protocol for every method: official train/test splits,
standardized features, `RidgeClassifierCV(alphas=logspace(-3,3,13))`, no per-dataset tuning of
any transform; baselines via their aeon implementations. Latency: ms/series to transform 200
series of length 256, single CPU core, warmed, best-of-5 (`scripts/run_runtime.py`).

| method | dims | accuracy | transform ms/series |
|---|---|---|---|
| MultiRocket | 49,728 | 0.8670 | 6.3 |
| **MSRF\*C** | **2,272** | **0.8624** | **0.28** |
| MiniRocket | 9,996 | 0.8591 | 0.40 |
| MiniRocket-1512 (MSRF\*C's conv component alone) | 1,512 | 0.8548 | 0.08 |
| Hydra | 5,120 | 0.8547 | 5.2 |
| **MSRF\*** | **760** | 0.8319 | 0.19 |
| **MSRF+** | **519** | 0.8240 | 0.11 |
| Rocket-10k ppv+max (original internal baseline) | 10,000 | 0.8197 | 5.0 |
| MSRF-1410 (original configuration) | 1,410 | 0.8102 | 1.3 |
| QUANT | 2,253 | 0.8059 | 0.20 |
| catch22 | 22 | 0.6900 | 0.31 |

QUANT note: it is designed for an extra-trees classifier and reports higher accuracy in its own
pipeline; the unified ridge head is what makes transforms comparable here.

## Quickstart

```bash
pip install -e . && pip install aeon
```

```python
from msrfc import MSRFC

clf = MSRFC().fit(X_train, y_train)   # X: (n_series, length) numpy arrays
accuracy = clf.score(X_test, y_test)  # MiniRocket-1512 + MSRF* -> 2,272 dims, ridge head
```

The universal encoder alone (zero fitted parameters, torch-free, numba-accelerated with a
pure-numpy fallback; frozen integer-seeded banks embedded — no data files, no learned weights):

```python
from msrfc import MultiSpaceEncCore
enc = MultiSpaceEncCore()             # MSRF+ (519d)
F = enc.transform(X)                  # (n_series, length) -> (n, 519)
```

## Method

MSRF distributes random features over complementary representation spaces — temporal (TRF),
global (GRF, moment-pooled random Fourier features), statistical (SRF), and convolutional. The
revised configurations refine the framework space by space:

1. **MSRF+ (519 dims).** All spaces computed on sliding windows (length 64, stride 16), each
   feature pooled per series by mean ‖ std ‖ max over its windows — the same features,
   time-localized, with their distribution over the series retained. A deterministic summary
   space (closed-form shape/spectral/autocorrelation descriptors) and a competing-kernel space
   (building on Hydra) join the four random spaces: 173 feature columns × 3 pooling statistics.
2. **MSRF\* (760 dims).** The framework applied to itself: a 241-dim second-level battery of
   multi-space random features (tanh/cos/relu/sigmoid projections, sign-sqrt/log1p compressions,
   rank channels, random landmarks, soft histograms, quantiles) computed *from the embedding*.
3. **MSRF\*C (2,272 dims).** The convolutional space upgraded from random kernels to a compact
   *canonical MiniRocket* dictionary (1,512 kernels, aeon implementation).

**Universality.** MSRF+ and MSRF\* are universal transforms in exactly the sense random
convolutions are: zero learned parameters, bit-reproducible from fixed integer seeds, the
identical bank applied to every dataset. MSRF\*C's multi-space block keeps that property; its
MiniRocket component inherits MiniRocket's per-dataset fitted bias quantiles, and every claim
states that scope.

## Significance (paired Wilcoxon, n=113)

| comparison | mean Δ | p | W/L |
|---|---|---|---|
| MSRF\*C vs Hydra | +0.008 | 0.042 | 52/42 |
| MSRF\*C vs MiniRocket | +0.003 | 0.17 (tied) | 47/41 |
| MSRF\*C vs MiniRocket-1512 (its component) | +0.008 | 5.7e-5 | 63/24 |
| MSRF\*C vs QUANT | +0.057 | 1.4e-11 | 86/16 |
| MSRF\*C vs MultiRocket | −0.005 | 0.028 | 30/63 |
| MSRF\* vs QUANT | +0.026 | 0.004 | 67/41 |
| MSRF\* vs MSRF+ (its base) | +0.008 | 0.002 | 59/44 |
| MSRF\* vs Hydra | −0.023 | 4.3e-5 | 34/74 |
| MiniRocket vs Rocket-10k ppv+max | +0.039 | 1.3e-8 | 80/23 |

The last row quantifies a baseline correction this revision makes: the original internal 10k
baseline ("Rocket_ppv", random Gaussian kernels + ppv/max) had been described as "equivalent to
MiniRocket"; canonical MiniRocket — whose bias quantiles are fitted on each training set — is
+0.039 stronger. All results here use canonical baselines, and the internal bank is relabeled.

## Combination experiments

Appending feature blocks to each base (52-dataset subset, `scripts/datasets_subset52.txt`):

| base | +non-conv-410 (original) | +MSRF+ (519) | +MSRF\* (760) |
|---|---|---|---|
| MiniRocket (0.8848) | −0.001 (n.s.) | +0.001 (n.s.) | +0.001 (n.s.) |
| Rocket-10k ppv+max (0.8587) | +0.002 (n.s.) | **+0.011, p=6e-4** | **+0.011, p=0.007** |
| QUANT (0.8560) | ±0.000 | **+0.012, p=0.004** | **+0.015, p=0.005** |
| Hydra (0.8887) | −0.000 | ±0.000 | +0.000 |
| MultiRocket (0.8901) | +0.000 | −0.002 | −0.002 |

![Combination gains](plots/fig_complementarity.png)

The multi-space block adds significant signal where the base lacks a full-scale fitted
convolutional dictionary (QUANT; plain random banks) and is saturated elsewhere — supporting
saturation analyses in `scripts/run_saturation.py`: appending the original non-convolutional
features to a 10k conv bank adds no participation-ratio effective rank (8.2 → 8.1) and no
out-of-sample subspace novelty beyond a spanned control (0.227 vs 0.401 floor). MSRF\*C uses
this boundary constructively: at a compact budget the fitted dictionary is not saturated, and
the multi-space block is decisively useful.

![Scaling and saturation](plots/fig_scaling.png)

## Ablations

- **MSRF\*C conv-component size** (archive-wide): 756 kernels → 0.8571 (ties Hydra at 30% of its
  dims), 1,512 → 0.8624, 3,024 → 0.8636 — a stable design region, not a tuned point.
- **Fully-universal scaling**: joining the windowed and global configurations with nothing
  fitted (2,170 dims) reaches 0.8481 — within 0.007 of Hydra (n.s.).
- **Budget scaling** (all spaces together, 52-dataset subset): 705d = 0.8507, 1,410d = 0.8554,
  2,820d = 0.8615 — ±0.005 per halving/doubling, no cliff.
- **Classifier head:** logistic regression instead of ridge shifts accuracy ≤0.003 and preserves
  all rankings.
- **Order of operations (MSRF\*):** expanding the full embedding (760d, 0.8319) beats
  pruning-then-expanding (452d, 0.8251).
- **Low-label regime** (5 labels/class, 5 seeds): MiniRocket 0.804 > MSRF+ 0.775 >
  Rocket-10k 0.764 > catch22 0.664 — MSRF+ leads the data-independent transforms.
- **Head cost:** ridge fitting is sub-second at every width (n=500: 0.03–0.04 s at 760–5,120
  dims; 0.17 s at 50k); feature memory at n=1000: 18 MB (MSRF\*C) vs 0.4 GB (MultiRocket).

## Reproduce everything

```bash
pip install numpy scipy scikit-learn numba matplotlib aeon torch
python scripts/download_data.py --out ./ucr_cache           # UCR archive via aeon
python scripts/run_unified_protocol.py --cache ./ucr_cache  # MSRF-1410 / MSRF+ / MSRF* / MSRF*C
python scripts/run_sota_baselines.py --cache ./ucr_cache    # MiniRocket/MultiRocket/Hydra/QUANT/catch22/Rocket-10k
python scripts/run_complementarity.py --cache ./ucr_cache   # combination table
python scripts/run_significance.py                          # all Wilcoxon tables from results/
python scripts/run_saturation.py --cache ./ucr_cache        # saturation analyses
python scripts/run_runtime.py                               # latency + head-cost table
python scripts/make_figures.py                              # regenerates plots/ from results/
```

macOS note: prefix baseline/combination runs with `OMP_NUM_THREADS=1
NUMBA_THREADING_LAYER=workqueue` (torch + numba threading deadlock in one process otherwise).

Per-dataset records behind every table: `results/unified_protocol_results.jsonl` and
`results/complementarity_results.jsonl` (one JSON row per dataset, one key per method).

## Repository layout

- `msrfc/` — the revised implementation (universal encoder, second-level battery, `MSRFC`).
- `msrf/` — the original four-space implementation (kept: it is the MSRF-1410 baseline and the
  non-convolutional block used in the combination experiments).
- `scripts/`, `results/`, `plots/` — the full evaluation above, reproducible end to end.
- `legacy/` — the original submission's paper sources, CD-diagram figures, and results, kept for
  reference; superseded by the evaluation in this README.

## License

MIT
