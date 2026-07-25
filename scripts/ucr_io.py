"""Shared data loading + evaluation protocol used by every rebuttal script.

Protocol: official UCR train/test splits, StandardScaler on the feature matrix,
RidgeClassifierCV(alphas=logspace(-3,3,13)) with per-dataset alpha selection.
Identical for every method.
"""
import json
import os

import numpy as np
from sklearn.linear_model import RidgeClassifierCV
from sklearn.preprocessing import StandardScaler

ALPHAS = np.logspace(-3, 3, 13)
HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")


def load(name, cache="./ucr_cache"):
    """Official split -> (Xtr, ytr, Xte, yte); X: (n, L) float64, y: int labels."""
    from aeon.datasets import load_classification
    Xtr, ytr = load_classification(name, split="train", extract_path=cache)
    Xte, yte = load_classification(name, split="test", extract_path=cache)
    Xtr = np.asarray(Xtr, dtype=np.float64).squeeze(1)
    Xte = np.asarray(Xte, dtype=np.float64).squeeze(1)
    m = {c: i for i, c in enumerate(sorted(set(ytr)))}
    return (Xtr, np.array([m[l] for l in ytr]),
            Xte, np.array([m.get(l, -1) for l in yte]))


def clf(Ftr, ytr, Fte, yte):
    """The one classification head used everywhere."""
    Ftr, Fte = np.nan_to_num(Ftr), np.nan_to_num(Fte)
    sc = StandardScaler().fit(Ftr)
    c = RidgeClassifierCV(alphas=ALPHAS).fit(sc.transform(Ftr), ytr)
    return float((c.predict(sc.transform(Fte)) == yte).mean())


def dataset_names(results_file="unified_protocol_results.jsonl"):
    """Dataset universe = rows of the released per-dataset results."""
    path = os.path.join(RESULTS, results_file)
    return [json.loads(l)["dataset"] for l in open(path)]


def update_results(name, updates, results_file="unified_protocol_results.jsonl"):
    """Merge computed columns into the per-dataset jsonl (rewrite in place)."""
    path = os.path.join(RESULTS, results_file)
    rows = [json.loads(l) for l in open(path)] if os.path.exists(path) else []
    idx = {r["dataset"]: r for r in rows}
    idx.setdefault(name, {"dataset": name}).update(updates)
    if name not in [r["dataset"] for r in rows]:
        rows.append(idx[name])
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
