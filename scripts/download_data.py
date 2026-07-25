"""Download the UCR univariate archive (112 datasets) via aeon; cache locally for all experiments.
  python3 download_data.py [--out ./ucr_cache]
"""
import argparse, os
from aeon.datasets import load_classification

NAMES112 = None  # resolved from aeon's UCR list at runtime

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="./ucr_cache"); a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    from aeon.datasets.tsc_datasets import univariate as NAMES
    for i, name in enumerate(sorted(NAMES)):
        try:
            load_classification(name, extract_path=a.out)
            print(f"[{i+1}/{len(NAMES)}] {name} ok", flush=True)
        except Exception as e:
            print(f"[{i+1}] {name} SKIP: {e}", flush=True)

if __name__ == "__main__":
    main()
