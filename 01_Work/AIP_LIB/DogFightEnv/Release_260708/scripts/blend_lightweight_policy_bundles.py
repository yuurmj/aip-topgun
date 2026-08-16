from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Linearly interpolate two compatible lightweight policy bundles."
    )
    parser.add_argument("--base-bundle", required=True)
    parser.add_argument("--tuned-bundle", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--tuned-weight", required=True, type=float)
    return parser.parse_args()


def load_weights(bundle: Path) -> dict:
    with gzip.open(bundle / "policy_weights.pkl.gz", "rb") as handle:
        return pickle.load(handle)


def main() -> None:
    args = parse_args()
    base = Path(args.base_bundle).resolve()
    tuned = Path(args.tuned_bundle).resolve()
    output = Path(args.output_bundle).resolve()
    alpha = float(args.tuned_weight)
    if not 0.0 <= alpha <= 1.0:
        raise ValueError("--tuned-weight must be in [0, 1]")
    if output.exists():
        raise FileExistsError(f"output bundle already exists: {output}")

    base_weights = load_weights(base)
    tuned_weights = load_weights(tuned)
    if base_weights.keys() != tuned_weights.keys():
        raise ValueError("bundle weight keys do not match")

    blended = {}
    for key, base_value in base_weights.items():
        tuned_value = tuned_weights[key]
        if np.shape(base_value) != np.shape(tuned_value):
            raise ValueError(f"shape mismatch for {key}")
        blended[key] = (1.0 - alpha) * base_value + alpha * tuned_value

    metadata = json.loads((base / "metadata.json").read_text(encoding="utf-8"))
    metadata.setdefault("metadata", {})["policy_weight_blend"] = {
        "base_bundle": str(base),
        "tuned_bundle": str(tuned),
        "tuned_weight": alpha,
    }
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as handle:
        pickle.dump(blended, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"blended bundle: {output}")


if __name__ == "__main__":
    main()
