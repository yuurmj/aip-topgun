from __future__ import annotations

import argparse
import gzip
import json
import pickle
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(
        description="Upgrade a tactical17 SAC bundle to tactical21."
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--output-bundle", required=True)
    return parser.parse_args()


def expand_input_columns(value, *, includes_action: bool):
    old_width = 21 if includes_action else 17
    new_width = old_width + 4
    if len(value.shape) != 2 or int(value.shape[1]) != old_width:
        return value
    if hasattr(value, "new_zeros"):
        expanded = value.new_zeros((int(value.shape[0]), new_width))
    else:
        expanded = np.zeros((int(value.shape[0]), new_width), dtype=value.dtype)
    expanded[:, :17] = value[:, :17]
    if includes_action:
        expanded[:, 21:25] = value[:, 17:21]
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
    upgraded = {}
    changed = []
    for key, value in weights.items():
        width = int(value.shape[1]) if len(getattr(value, "shape", ())) == 2 else -1
        if width in {17, 21}:
            upgraded[key] = expand_input_columns(
                value,
                includes_action=width == 21,
            )
            changed.append((key, tuple(value.shape), tuple(upgraded[key].shape)))
        else:
            upgraded[key] = value
    env_config = metadata["algorithm_config"]["env_config"]
    env_config["observation_mode"] = "tactical21"
    details = metadata.setdefault("metadata", {})
    details.update({
        "obs_mode": "tactical21",
        "observation_mode": "tactical21",
        "observation_size": 21,
        "upgraded_from_bundle": str(source),
        "tactical21_new_columns_initialized": "zero",
    })
    output.mkdir(parents=True)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    with gzip.open(output / "policy_weights.pkl.gz", "wb") as handle:
        pickle.dump(upgraded, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"upgraded bundle: {output}")
    for key, before, after in changed:
        print(f"{key}: {before} -> {after}")


if __name__ == "__main__":
    main()
