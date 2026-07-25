from setuptools import setup, find_packages

setup(
    name="msrf",
    version="0.2.0",
    description="Multi-Space Random Features for Time Series Classification (MSRF+/MSRF*/MSRF*C)",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21",
        "torch>=1.10",
        "scikit-learn>=1.0",
    ],
    extras_require={
        "ucr": ["aeon>=0.7"],
        "msrfc": ["aeon>=0.7", "numba"],
    },
)
