#!/usr/bin/env python3
"""
Minimal example: run MSRF on a single UCR dataset.

Requirements: pip install aeon  (for loading UCR datasets)
"""

import numpy as np
from aeon.datasets import load_classification

from msrf import MSRFTransform, msrf_classify


def main():
    dataset = "GunPoint"
    print(f"Loading UCR dataset: {dataset}")
    X_train, y_train = load_classification(dataset, split="train")
    X_test, y_test = load_classification(dataset, split="test")

    # aeon returns (N, 1, T) — squeeze the channel dimension
    X_train = X_train.squeeze(1).astype(np.float32)
    X_test = X_test.squeeze(1).astype(np.float32)

    # --- Option 1: Full pipeline (extract + classify in one call) ---
    result = msrf_classify(X_train, y_train, X_test, y_test)
    print(f"\nMSRF_full accuracy on {dataset}: {result['accuracy']:.4f}")
    print(f"  Features: {result['n_features']}, Ridge alpha: {result['alpha']:.4f}")

    # --- Option 2: Use the transform directly for custom classifiers ---
    T = X_train.shape[1]
    transform = MSRFTransform(T=T, n_trf=200, n_grf_projections=5, n_srf=200, n_crf_kernels=500)
    F_train = transform.transform_numpy(X_train)
    F_test = transform.transform_numpy(X_test)
    print(f"\n  Feature matrix shape: {F_train.shape}")

    # --- Option 3: Non-convolutional features only (as ROCKET augmentation) ---
    result_nonconv = msrf_classify(
        X_train, y_train, X_test, y_test,
        n_trf=200, n_grf_projections=5, n_srf=200, n_crf_kernels=0,
    )
    print(f"\n  TRF+GRF+SRF only: {result_nonconv['accuracy']:.4f} ({result_nonconv['n_features']} features)")


if __name__ == "__main__":
    main()
