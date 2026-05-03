# MSRF: Multi-Space Random Features for Time Series

MSRF distributes random features across four complementary representation spaces for time series classification:

| Space | Captures | Invariant to | 
|---|---|---|
| **Temporal (TRF)** | Positional signal structure | Signal shape |
| **Global (GRF)** | Sequence-level geometry | Position, local pattern |
| **Statistical (SRF)** | Distributional properties | Temporal ordering |
| **Convolutional (CRF)** | Local pattern occurrence | Absolute position |

By spanning orthogonal axes of variation,  MSRF achieves competitive accuracy with ROCKET-family methods using far fewer features, and provides genuinely complementary signal when combined with existing convolutional pipelines.

## Key Results (112 UCR datasets, Ridge classifier)

- **MSRF as ROCKET plug-in**: Adding 410 non-convolutional MSRF features (4% overhead) to MiniRocket's 10,000 features improves accuracy on **67/112 datasets** with only 14 losses (83% win rate).
- **Feature efficiency**: MSRF with 1,410 features achieves higher mean accuracy than MiniRocket with 10,000 features (0.808 vs 0.794) — **86% fewer dimensions**.
- **Diversity > scale**: At a matched ~1,400-dim budget, multi-space allocation beats pure convolutional on 59% of datasets.

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

**Convolutional Random Features (CRF)** use MiniRocket-style random kernels with configurable transforms and pooling. Equivalent to MiniRocket when using 10K kernels with PPV pooling.

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
