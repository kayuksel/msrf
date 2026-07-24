# MSRF — Author-response experiments and revised results

This directory contains every experiment referenced in the author response: the improved
configurations (**MSRF+**, **MSRF\***), a unified-protocol comparison against all reviewer-requested
methods (MiniRocket, MultiRocket, Hydra, QUANT), significance tests, combination (complementarity)
experiments, saturation analyses, ablations, and efficiency measurements — with per-dataset records
in `results/` and one-command reproduction scripts in `scripts/`.

**Protocol (identical for every method):** official UCR train/test splits; the feature matrix is
standardized and classified by `RidgeClassifierCV(alphas=logspace(-3,3,13))` with per-dataset alpha
selection; no per-dataset tuning of any transform. Baselines use their aeon implementations.
One harness produced every number below.

---

## 1. Method: from MSRF to MSRF+ and MSRF*

MSRF distributes random features over four complementary spaces — temporal (TRF), global (GRF,
moment-pooled random Fourier features), statistical (SRF), and convolutional (CRF). The improved
configurations keep this framework unchanged and extend it in three ways:

1. **Windowed computation.** All spaces are computed on sliding windows (length 64, stride 16)
   instead of the whole series, and each feature is pooled per series by mean ‖ std ‖ max over its
   windows. This changes *where* features are computed, not *what* they are: every family becomes
   time-localized, and the pooling captures its distribution over the series.
2. **A deterministic summary space** joins the four random spaces: closed-form shape, spectral, and
   autocorrelation descriptors (the class of quantities catch22 curates), computed per window.
3. **A competing-kernel space** (building on Hydra, cited): groups of random kernels where, per
   time step, the best-matching kernel wins; soft win-count histograms (sqrt-compressed) are the
   features.

The result is **MSRF+**: 173 feature columns × 3 pooling statistics = **519 dimensions**.

**MSRF\*** applies the framework to itself: a second-level battery of 241 multi-space random
features — random projections through tanh/cos/relu/sigmoid nonlinearities, sign-sqrt and
sign-log1p compressions, within-vector rank features, random landmark (RBF) distances, soft
histograms, and quantiles — computed *from the 519-dim embedding*. Total: **760 dimensions**.

**Universality.** Exactly like random convolutions, every MSRF variant is a *universal* transform:
zero learned parameters, bit-reproducible from fixed integer seeds, the identical bank applied to
all 107 datasets with no per-dataset hand-crafting or tuning. (Of the compared methods, canonical
MiniRocket is the one that does adapt per dataset — its bias quantiles are fitted on each training
set.)

## 2. Baseline correction

The submission's internal GPU baseline ("Rocket_ppv"/"Rocket_full": 5,000 random Gaussian kernels,
random dilations, ppv/max pooling) was described as "equivalent to MiniRocket". It is not: under
the identical harness, canonical aeon MiniRocket scores **0.8596** vs **0.8200** for the internal
bank (paired Wilcoxon p=1.1e-7, W/L 74/23). The revision relabels the internal baseline
"Rocket-10k (ppv+max)", adopts canonical MiniRocket as the 10k reference throughout, and rebuilds
every affected claim. All tables below already reflect this.

## 3. Main result (Table R1)

105 archive datasets common to all methods. Latency: ms/series to transform 200 series of length
256, single CPU core (`scripts/run_runtime.py`).

| method | dims | accuracy | transform ms/series |
|---|---|---|---|
| MultiRocket | 49,728 | 0.8673 | 13 |
| MiniRocket | 9,996 | 0.8596 | 0.9 |
| Hydra | 5,120 | 0.8546 | 15 |
| **MSRF\* (revision)** | **760** | **0.8339** | **0.5** |
| **MSRF+ (revision)** | **519** | **0.8260** | **0.3** |
| Rocket-10k ppv+max (submission's internal baseline) | 10,000 | 0.8200 | 16 |
| MSRF-1410 (submission) | 1,410 | 0.8106 | 3.5 |
| QUANT | 2,253 | 0.8049 | 0.9 |
| catch22 | 22 | 0.6899 | 121 |

![Accuracy vs dimension frontier](plots/fig_pareto.png)

Reading it plainly: MultiRocket, MiniRocket, and Hydra are more accurate, at 65×, 13×, and 6.7×
the dimensionality. **MSRF\* is the most accurate transform below Hydra's 5,120 dimensions and the
fastest transform we measured**; it significantly outperforms QUANT (+0.029, p=0.002) at one third
the dimensions, and matches the 10k random-kernel bank per-dataset (W/L 52/52) with 13× fewer
dimensions. Note: QUANT is designed for an extra-trees classifier and reports higher accuracy in
its own pipeline; under the unified ridge head used for every method here, it scores 0.8049.

## 4. Significance (Table R2)

Paired Wilcoxon signed-rank over per-dataset accuracies (`scripts/run_significance.py`):

| comparison | mean Δ | p | W/L |
|---|---|---|---|
| MSRF-1410 vs catch22 | +0.120 | 1.6e-14 | 90/13 |
| MSRF-1410 vs MiniRocket | −0.049 | 2.2e-10 | 20/77 |
| MSRF+ vs MiniRocket | −0.034 | 1.1e-7 | 23/71 |
| MSRF\* vs MiniRocket | −0.026 | 5.9e-6 | 27/74 |
| MSRF\* vs Hydra | −0.021 | 1.9e-4 | 33/68 |
| MSRF\* vs QUANT | +0.029 | 0.002 | 63/38 |
| MSRF\* vs MSRF+ (its base) | +0.008 | 0.004 | 56/42 |
| MiniRocket vs Rocket-10k ppv+max | +0.039 | 1.1e-7 | 74/23 |

The last row is the magnitude of the baseline mislabeling that this revision corrects. The MSRF\*
gain over its base replicates archive-wide and involves no data-dependent selection.

## 5. Combination experiments (Table R3)

Appending feature blocks to each base method (52-dataset subset, `scripts/datasets_subset52.txt`;
paired Wilcoxon; `scripts/run_complementarity.py`):

| base | +non-conv-410 (submission) | +MSRF+ (519) | +MSRF\* (760) |
|---|---|---|---|
| MiniRocket (0.8848) | −0.001 (n.s.) | +0.001 (n.s.) | +0.001 (n.s., W/L 20/9) |
| Rocket-10k ppv+max (0.8587) | +0.002 (n.s.) | **+0.011, p=6e-4** (W/L 24/5) | **+0.011, p=0.007** |
| QUANT (0.8560) | ±0.000 | **+0.012, p=0.004** | **+0.015, p=0.005** |
| Hydra (0.8887) | −0.000 | ±0.000 | +0.000 |
| MultiRocket (0.8901) | +0.000 | −0.002 | −0.002 |

![Combination gains](plots/fig_complementarity.png)

The pattern is consistent: the multi-space block adds significant signal to bases without
data-fitted convolutional dictionaries (QUANT; the plain random-kernel bank), and is saturated on
top of transforms that fit their dictionaries per dataset (MiniRocket's quantile biases), contain a
competing-kernel family themselves (Hydra — the same family MSRF+ now includes), or operate at 50k
scale (MultiRocket). This extends the submission's own diminishing-returns finding (Sec. 5.3:
wins fall 67 → 29 as the convolutional budget grows 10k → 60k).

## 6. Saturation analyses (reviewer THZ8's experiment)

`scripts/run_saturation.py`, 12 smallest subset datasets: appending the submission's 410
non-convolutional features to the 10k conv bank (a) does **not** increase participation-ratio
effective rank (8.2 → 8.1; the block is internally low-rank, PR 4.2/410), and (b) carries **no
out-of-sample subspace novelty**: ridge-predicting the block from the conv bank leaves 0.227
unexplained test variance, *below* the 0.401 floor of a spanned-by-construction control. Combined
with the −0.001 accuracy effect on canonical MiniRocket (Table R3), the conclusion is conceded and
stated in the revision: on top of a strong 10k convolutional representation, the submission's
non-convolutional features are largely redundant.

![Scaling and saturation](plots/fig_scaling.png)

## 7. Ablations

- **Budget scaling** (all spaces scaled together, 52-dataset subset): 705d = 0.8507,
  1,410d = 0.8554, 2,820d = 0.8615 — ±0.005 per halving/doubling; the default is a budget choice,
  not a tuned optimum.
- **Classifier head:** logistic regression instead of ridge shifts accuracy ≤0.003 and preserves
  all method rankings.
- **Order of operations (MSRF\*):** expanding the full 519-dim embedding (760d, 0.8339) beats
  pruning-then-expanding (452d, 0.8251): the battery is a nonlinear sketch of the embedding, not a
  substitute for discarded coordinates.
- **Low-label regime** (5 labels/class, 5 seeds, 52-dataset subset): MiniRocket 0.804,
  MSRF+ 0.775, Rocket-10k 0.764, MSRF-1410 0.763, catch22 0.664 — MSRF+ leads the
  data-independent transforms; MiniRocket's fitted biases help even here.
- **Head cost:** RidgeClassifierCV fitting is sub-second at every compared width (n=500: 0.2 s at
  760 dims, 0.9 s at 50k); the feature-matrix memory differs 65× (6 MB vs 0.4 GB at n=1000).

## 8. Reproduce everything

```bash
pip install numpy scipy scikit-learn numba matplotlib aeon torch
python scripts/download_data.py --out ./ucr_cache           # UCR archive via aeon
python scripts/run_unified_protocol.py --cache ./ucr_cache  # MSRF-1410 / MSRF+ / MSRF*
python scripts/run_sota_baselines.py --cache ./ucr_cache    # MiniRocket/MultiRocket/Hydra/QUANT/catch22/Rocket-10k
python scripts/run_complementarity.py --cache ./ucr_cache   # Table R3
python scripts/run_significance.py                          # Tables R2 + R3 stats from results/
python scripts/run_saturation.py --cache ./ucr_cache        # Sec. 6 numbers
python scripts/run_runtime.py                               # latency + head-cost table
python scripts/make_figures.py                              # regenerates plots/ from results/
```

macOS note: prefix the baseline/combination runs with `OMP_NUM_THREADS=1
NUMBA_THREADING_LAYER=workqueue` (torch + numba threading deadlock in one process otherwise).

The improved encoder is self-contained and torch-free (`scripts/msenc_*.py`, numba-accelerated
with a pure-numpy fallback; frozen seeded banks embedded — no data files, no learned weights):

```python
from msenc_encoder_np import MultiSpaceEncCore
enc = MultiSpaceEncCore()          # MSRF+ (519d)
F = enc.transform(X)               # X: (n_series, length) -> (n, 519)
```

Per-dataset records behind every table: `results/unified_protocol_results.jsonl` and
`results/complementarity_results.jsonl` (one JSON row per dataset, one key per method).
