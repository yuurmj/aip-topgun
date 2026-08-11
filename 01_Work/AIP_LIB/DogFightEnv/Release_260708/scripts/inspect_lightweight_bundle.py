from __future__ import annotations

import argparse
import gzip
import pickle
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="List lightweight bundle weight keys.")
    parser.add_argument("bundle_dir")
    parser.add_argument("--prefix", default="")
    args = parser.parse_args()

    path = Path(args.bundle_dir) / "policy_weights.pkl.gz"
    with gzip.open(path, "rb") as handle:
        weights = pickle.load(handle)
    for key, value in weights.items():
        if key.startswith(args.prefix):
            print(f"{key}: {getattr(value, 'shape', None)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
