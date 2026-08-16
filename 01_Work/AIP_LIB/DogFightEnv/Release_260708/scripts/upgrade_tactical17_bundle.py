from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upgrade a tactical16 SAC bundle to periodic-yaw tactical17."
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument(
        "--yaw-cos-sign",
        type=float,
        choices=[-1.0, 1.0],
        default=1.0,
        help=(
            "Sign applied to the copied tactical16 yaw column. Use -1 to "
            "preserve the near-zero-degree branch and +1 for near 360 degrees."
        ),
    )
    return parser.parse_args()


def expand_input_columns(value, *, includes_action: bool, yaw_cos_sign: float = 1.0):
    old_width = 20 if includes_action else 16
    new_width = old_width + 1
    if len(value.shape) != 2 or int(value.shape[1]) != old_width:
        return value

    if hasattr(value, "new_zeros"):
        expanded = value.new_zeros((int(value.shape[0]), new_width))
    else:
        expanded = np.zeros(
            (int(value.shape[0]), new_width),
            dtype=value.dtype,
        )
    expanded[:, 0:2] = value[:, 0:2]
    # tactical16 yaw was one scalar. Preserve the near-zero/360-degree branch
    # by mapping its learned column to cos(yaw); sin(yaw) starts neutral.
    expanded[:, 3] = float(yaw_cos_sign) * value[:, 2]
    expanded[:, 4:17] = value[:, 3:16]
    if includes_action:
        expanded[:, 17:21] = value[:, 16:20]
    return expanded


def main():
    args = parse_args()
    source = Path(args.source_bundle).resolve()
    output = Path(args.output_bundle).resolve()
    if output.exists():
        raise FileExistsError(f"output bundle already exists: {output}")

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    with gzip.open(source / "policy_weights.pkl.gz", "rb") as handle:
        weights = pickle.load(handle)

    changed = []
    upgraded = {}
    for key, value in weights.items():
        shape = getattr(value, "shape", ())
        if len(shape) == 2 and int(shape[1]) in {16, 20}:
            upgraded[key] = expand_input_columns(
                value,
                includes_action=int(shape[1]) == 20,
                yaw_cos_sign=args.yaw_cos_sign,
            )
            changed.append((key, tuple(shape), tuple(upgraded[key].shape)))
        else:
            upgraded[key] = value

    env_config = metadata["algorithm_config"]["env_config"]
    env_config["observation_mode"] = "tactical17"
    details = metadata.setdefault("metadata", {})
    details["obs_mode"] = "tactical17"
    details["observation_mode"] = "tactical17"
    details["observation_size"] = 17
    details["upgraded_from_bundle"] = str(source)
    details["yaw_encoding"] = "sin_cos"
    details["yaw_cos_sign"] = args.yaw_cos_sign

    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as handle:
        pickle.dump(upgraded, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"upgraded bundle: {output}")
    for key, before, after in changed:
        print(f"{key}: {before} -> {after}")


if __name__ == "__main__":
    main()
