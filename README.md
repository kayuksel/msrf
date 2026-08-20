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
- **The price of the linear head** (113 datasets): swapping the ridge readout for a leaf-RFM head
  (xRFM, ICLR 2026 — kernel ridge with iterated AGOP feature reweighting) on the *same* frozen
  MSRF+ 519 dims gives 0.8240 → 0.8365 (+0.0125, W/L 57/40, p=0.006); +0.032 (p=1.3e-5) on the 43
  datasets with ≥300 training series, −0.013 on the smallest (n<100). A control head with identical
  selection machinery but no feature learning does not explain it (RFM beats that control by
  +0.0177, p=0.0009), so what the linear head forgoes is per-dataset *feature learning*, not tuning
  capacity. The linear head is a deployment choice — exact attribution, rank-one personalisation, no
  support set to store — and this is its measured price. `scripts/run_rfm_head.py`
- **Head cost:** ridge fitting is sub-second at every width (n=500: 0.03–0.04 s at 760–5,120
  dims; 0.17 s at 50k); feature memory at n=1000: 18 MB (MSRF\*C) vs 0.4 GB (MultiRocket).

## Order-sensitive pooling

`MultiSpaceEncCore._pool` is `concat([mean, std, max])` over the patch axis. All three are symmetric
functions of the patch multiset, so **MSRF+519 is provably invariant to any reordering of patches**
above the 64-sample window: it cannot distinguish a stream that rises then falls from one that falls
then rises. Three order-sensitive pooling statistics fix it, each defined from *prefix* statistics
only so it is computable online with O(1) state per feature column (`scripts/order_pool_online.py`):

| pooling | dims | accuracy | vs 519-d | W/L/T | p |
|---|---|---|---|---|---|
| MSRF+519 (this stack) | 519 | 0.8244 | — | — | — |
| + `contrast` (pre/post) | 692 | 0.8304 | +0.0060 | 62/35/16 | 0.0023 |
| + `earliness` (running-max envelope) | 692 | 0.8314 | +0.0070 | 61/33/19 | 5.1e-4 |
| + `cusum` (centred excursion range) | 692 | **0.8348** | +0.0105 | 71/25/17 | 1.1e-5 |
| + all three | 1,038 | **0.8358** | +0.0115 | 69/31/13 | 9.5e-5 |
| order blocks **alone** (amplitude divided out) | 519 | 0.7815 | −0.0428 | 28/77/8 | 5.7e-6 |

- **Pareto point:** `cusum` at 692 dims (0.8348) beats MSRF\*760 at 760 dims (0.8323 on this stack)
  by +0.0025, W/L 63/41, p=0.038 — better accuracy from fewer dimensions, from a pooling change
  alone. Both sides measured in the same run; see the reproducibility note below.
- **Order alone carries most of the signal.** The three blocks with amplitude normalised away reach
  0.7815, which is the point of calling the representation a world model rather than a fingerprint.
- **The gain is a property of the patch grid, not the data.** On the released grid the blocks *cost*
  −0.033 on the 17 datasets short enough to yield ≤2 patches; resampling short series to a ≥24-patch
  grid turns that into +0.006. Accuracy is flat in that target over 8–48 patches (spread 0.0023, every
  target significant), so the constant is not load-bearing. Largest gains at 9–24 patches (+0.026).
- **It does not compound with a convolutional bank.** Added to MSRF\*C2272 the blocks move accuracy
  by +0.0008 (0.8623 → 0.8631): MiniRocket's dilated kernels already carry the order information the
  symbolic pool discarded.
- **Two passes buy nothing.** `contrast` and `earliness` are bit-identical under the prefix-only and
  global-normalised definitions; `cusum` differs by +0.0018 (p=0.43) in favour of prefix-only. The
  repair is therefore free in a streaming deployment.
- **Cost on one core** (`scripts/profile_order.py`, threads pinned, best-of-5, 200 series): the blocks
  themselves are free (1.02× on the released grid). All overhead is the denser grid, and it is a
  resampling cost — `FusedOrderPoolEnc` runs φ once where the grids coincide and costs **1.07×** at
  L=512/1024, bit-identically. Only series shorter than 432 samples pay (2.9–7×).
- **Multivariate** (UEA-17, caps n≤600/T≤1300/ch≤65): pooling the channel-concatenated stack makes
  the blocks *cross-channel* statistics rather than temporal ones. That is not a defect here — stacked
  scores 0.7098 vs base 0.6758 (+0.0340, W/L 10/3, p=0.043), while the channel-symmetric per-channel
  formulation gives only +0.0069 (n.s.). The effect is largest at 2–3 channels, where the halfway
  split lands on a channel boundary and `contrast` becomes a clean channel difference (Libras +0.228,
  UWaveGestureLibrary +0.160). Use per-channel if cross-channel mixing is undesirable.
- **Adapting the pooling per dataset barely pays** (two-pass arms): one fixed choice 0.8342, a
  label-free rule on (patch count, n/C) 0.8382, per-dataset selection by cross-validation on the
  *training* split 0.8388 ± 0.0009 (2,000 random tie-breaks; CV cannot separate the arms on 20% of
  datasets), against a test-set oracle of 0.8449. The label-free rule is within noise of the
  CV-selected one, so the universal encoder loses almost nothing.
- **Interaction with a nonlinear head.** On the order-augmented features the leaf-RFM head's edge over
  ridge shrinks by −0.0051 (p=0.041), so part of what looked like the price of linearity was this
  pooling defect; it does not vanish, and RFM still gains +0.0048 from the order columns itself, so
  the two are complementary. `results/rfm_order_results.jsonl`

### Reproducibility note

The published per-dataset column (`results/unified_protocol_results.jsonl`, key `MSRF+519`) was
produced on an older library stack. The encoder is deterministic and its features are bit-identical
here, but `RidgeClassifierCV`'s alpha selection is not stable across scikit-learn versions: 24 of 113
datasets land on a different alpha, so the archive mean re-measures at 0.8244 rather than 0.8240. The
difference is unbiased (13 datasets higher, 11 lower, Wilcoxon p=0.22, median |Δ| 0.0016 among those
that differ). Every comparison above is a **paired** delta measured within a single run, so this does
not affect any reported gain; `scripts/run_baseline_recheck.py` re-measures both baselines on the
current stack for the Pareto comparison. `results/rfm_head_results.jsonl` was likewise produced on
the older stack (its ridge column matches `MSRF+519` on all 113 datasets); the order-pooling and
RFM-interaction results in `results/rfm_order_results.jsonl` were produced on the current one.

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
python scripts/run_rfm_head.py --cache ./ucr_cache          # linear-head price: ridge vs OOF-control vs leaf-RFM
python scripts/run_order_pool_online.py --cache ./ucr_cache  # order-sensitive pooling (prefix-only blocks)
python scripts/run_order_pool_fine.py --cache ./ucr_cache    # same, global-normalised (two-pass) blocks
python scripts/run_order_pool.py --cache ./ucr_cache         # blocks on the released patch grid
python scripts/run_target_sweep.py --cache ./ucr_cache       # sensitivity to the resampling target
python scripts/run_order_union.py --cache ./ucr_cache        # do the blocks add to MSRF*C2272?
python scripts/run_pool_select.py --cache ./ucr_cache        # per-dataset vs label-free pooling choice
python scripts/run_order_mv.py --cache ./uea_cache           # multivariate: stacked vs per-channel
python scripts/run_rfm_order.py --cache ./ucr_cache          # leaf-RFM on order-augmented features
python scripts/run_baseline_recheck.py --cache ./ucr_cache   # re-measure MSRF+519 / MSRF*760 on this stack
python scripts/profile_order.py                              # encode latency of the repair, one core
python scripts/make_figures.py                              # regenerates plots/ from results/
```

macOS note: prefix baseline/combination runs with `OMP_NUM_THREADS=1
NUMBA_THREADING_LAYER=workqueue` (torch + numba threading deadlock in one process otherwise).

Per-dataset records behind every table: `results/unified_protocol_results.jsonl` and
`results/complementarity_results.jsonl` (one JSON row per dataset, one key per method); `results/rfm_head_results.jsonl` carries the
three readouts per dataset with the RFM head's selected kernel, AGOP iteration count and fit time.

## Repository layout

- `msrfc/` — the revised implementation (universal encoder, second-level battery, `MSRFC`).
- `msrf/` — the original four-space implementation (kept: it is the MSRF-1410 baseline and the
  non-convolutional block used in the combination experiments).
- `scripts/`, `results/`, `plots/` — the full evaluation above, reproducible end to end.
- `legacy/` — the original submission's paper sources, CD-diagram figures, and results, kept for
  reference; superseded by the evaluation in this README.

## License

MIT
