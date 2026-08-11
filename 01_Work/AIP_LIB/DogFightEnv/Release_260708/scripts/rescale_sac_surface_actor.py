from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from pathlib import Path


MEAN_HEAD_WEIGHT = "pi.net.mlp.0.weight"
MEAN_HEAD_BIAS = "pi.net.mlp.0.bias"
SURFACE_ACTIONS = slice(0, 3)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Rescale SAC roll/pitch/rudder actor means so a policy preserves "
            "approximately the same physical surface commands at a new authority."
        )
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--source-scale", required=True, type=float)
    parser.add_argument("--target-scale", required=True, type=float)
    parser.add_argument(
        "--rescale-surface-std",
        action="store_true",
        help="Also reduce roll/pitch/rudder exploration std by the authority ratio.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_bundle).resolve()
    output = Path(args.output_bundle).resolve()
    if not 0.0 < args.source_scale <= 1.0:
        raise ValueError("source-scale must be in (0, 1]")
    if not 0.0 < args.target_scale <= 1.0:
        raise ValueError("target-scale must be in (0, 1]")
    if output.exists():
        raise FileExistsError(f"output bundle already exists: {output}")

    metadata_path = source / "metadata.json"
    weights_path = source / "policy_weights.pkl.gz"
    if not metadata_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"invalid source bundle: {source}")

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    with gzip.open(weights_path, "rb") as handle:
        weights = pickle.load(handle)

    if MEAN_HEAD_WEIGHT not in weights or MEAN_HEAD_BIAS not in weights:
        raise KeyError("bundle does not contain the expected SAC actor mean head")
    head_weight = weights[MEAN_HEAD_WEIGHT].copy()
    head_bias = weights[MEAN_HEAD_BIAS].copy()
    if head_weight.shape[0] != 8 or head_bias.shape[0] != 8:
        raise ValueError(
            "expected a four-action SAC head with 8 mean/log-std outputs, got "
            f"{head_weight.shape} and {head_bias.shape}"
        )

    ratio = args.source_scale / args.target_scale
    head_weight[SURFACE_ACTIONS] *= ratio
    head_bias[SURFACE_ACTIONS] *= ratio
    if args.rescale_surface_std:
        # SAC emits [mean(4), log_std(4)]. Adding log(ratio) multiplies the
        # exploration standard deviation while preserving its state dependence.
        head_bias[4:7] += math.log(ratio)
    weights[MEAN_HEAD_WEIGHT] = head_weight
    weights[MEAN_HEAD_BIAS] = head_bias

    env_config = metadata.get("algorithm_config", {}).get("env_config", {})
    env_config["control_surface_scale"] = args.target_scale
    bundle_metadata = metadata.setdefault("metadata", {})
    bundle_metadata["surface_actor_rescale"] = {
        "source_bundle": str(source),
        "source_scale": args.source_scale,
        "target_scale": args.target_scale,
        "actor_mean_ratio": ratio,
        "surface_actions": ["roll", "pitch", "rudder"],
        "throttle_unchanged": True,
        "surface_std_rescaled": args.rescale_surface_std,
    }

    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as handle:
        pickle.dump(weights, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"source={source}")
    print(f"output={output}")
    print(f"actor_mean_ratio={ratio:.6f}")


if __name__ == "__main__":
    main()
