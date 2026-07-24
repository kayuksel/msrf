# MSRF: Multi-Space Random Features for Time Series

MSRF distributes random features across four complementary representation spaces for time series classification:

| Space | Captures | Invariant to | 
|---|---|---|
| **Temporal (TRF)** | Positional signal structure | Signal shape |
| **Global (GRF)** | Sequence-level geometry | Position, local pattern |
| **Statistical (SRF)** | Distributional properties | Temporal ordering |
| **Convolutional (CRF)** | Local pattern occurrence | Absolute position |

By spanning orthogonal axes of variation, MSRF offers the best measured accuracy-per-dimension at compact budgets (below ~5,000 dims) with the fastest transform we measured, and provides complementary signal when combined with quantile-based methods and plain random convolutional banks.

## Key Results (UCR archive, unified RidgeClassifierCV protocol)

See **[`rebuttal/README.md`](rebuttal/README.md)** for the full author-response experiments: improved
configurations (MSRF+ 519d, MSRF* 760d), comparisons against canonical MiniRocket, MultiRocket, Hydra,
and QUANT under one harness, significance tests, combination experiments, and reproduction scripts.

- **Compact frontier**: MSRF* (760 dims) is the most accurate transform below Hydra's 5,120 dims and
  the fastest measured (0.5 ms/series, single core); it significantly outperforms QUANT (+0.029,
  p=0.002) at one-third the dimensions.
- **Combination gains, scoped**: appended MSRF features significantly improve QUANT and plain
  random-kernel banks (+0.011 to +0.015, p<=0.007); on top of data-fitted convolutional dictionaries
  (canonical MiniRocket) or 50k-dim transforms they are saturated — measured and stated.
- **Baseline correction**: the internal 10k baseline previously described as "equivalent to
  MiniRocket" is ROCKET-style (random kernels); canonical MiniRocket is +0.039 stronger (p=1.1e-7).
  All results here use canonical baselines.

## Installation

```bash
pip install -e .
```

Requires PyTorch (CPU or GPU) and scikit-learn. For UCR dataset loading, install `aeon`:

```bash
pip install aeon
```

## Quick Start

```python
import numpy as np
from msrf import msrf_classify

# X_train, X_test: (N, T) numpy arrays
# y_train, y_test: (N,) label arrays
result = msrf_classify(X_train, y_train, X_test, y_test)
print(f"Accuracy: {result['accuracy']:.4f}")
```

### Custom configuration

```python
from msrf import MSRFTransform

T = 128  # series length
transform = MSRFTransform(
    T=T,
    n_trf=200,            # temporal features
    n_grf_projections=5,  # GRF projections (yields 10 features)
    grf_dim=256,           # embedding dim per GRF projection
    n_srf=200,            # statistical features
    n_crf_kernels=500,    # convolutional kernels (yields 1000 features)
    crf_pool_ops=["ppv", "max"],
)

# Extract features (handles batching and GPU transfer)
F_train = transform.transform_numpy(X_train)  # (N, 1410)
F_test = transform.transform_numpy(X_test)

# Use any classifier
from sklearn.linear_model import RidgeClassifierCV
clf = RidgeClassifierCV(alphas=np.logspace(-4, 4, 20))
clf.fit(F_train, y_train)
print(f"Accuracy: {clf.score(F_test, y_test):.4f}")
```

### As a ROCKET augmentation (non-convolutional features only)

```python
from msrf import MSRFTransform

# Extract only TRF + GRF + SRF (410 dims) to concatenate with your ROCKET features
transform = MSRFTransform(T=T, n_trf=200, n_grf_projections=5, n_srf=200, n_crf_kernels=0)
F_msrf = transform.transform_numpy(X_train)  # (N, 410)
F_augmented = np.hstack([your_rocket_features, F_msrf])
```

## Feature Spaces

**Temporal Random Features (TRF)** apply Gaussian windows centered at random positions along the time axis, computing position-weighted averages. Translation-variant by design.

**Global Random Features (GRF)** project the full time series through Random Fourier Features cos(Wx + b), then extract mean and std of the embedding — a novel *moment-pooled* readout that compresses a D-dimensional random Fourier embedding into 2 scalar features per projection.

**Statistical Random Features (SRF)** compute summary statistics (mean, std, skew, kurtosis, quantiles, autocorrelation) and pass them through two-layer random projections to capture nonlinear interactions.

**Convolutional Random Features (CRF)** use ROCKET-style random kernels (random weights and dilations) with configurable transforms and pooling. Note: this is *not* equivalent to canonical MiniRocket, whose fixed kernel dictionary and data-fitted bias quantiles score higher at the same budget — see `rebuttal/README.md` for the measured comparison.

## Citation

```bibtex
@inproceedings{msrf2026,
  title={Multi-Space Random Features for Time Series Classification},
  author={Anonymous},
  booktitle={Advances in Neural Information Processing Systems},
  year={2026}
}
```

## License

MIT
