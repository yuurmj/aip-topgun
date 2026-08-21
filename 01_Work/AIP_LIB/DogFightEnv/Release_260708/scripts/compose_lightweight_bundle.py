from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create one self-contained bundle using actor weights from one "
            "bundle and the saved runtime/env contract from another."
        )
    )
    parser.add_argument("--weights-bundle", required=True)
    parser.add_argument("--runtime-bundle", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--model-name", required=True)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def observation_size(metadata: dict) -> int | None:
    value = metadata.get("metadata", {}).get("observation_size")
    return int(value) if value is not None else None


def model_config(metadata: dict) -> dict:
    return dict(metadata.get("metadata", {}).get("model_config", {}))


def main() -> int:
    args = parse_args()
    weights_bundle = Path(args.weights_bundle).resolve()
    runtime_bundle = Path(args.runtime_bundle).resolve()
    output_bundle = Path(args.output_bundle).resolve()
    if output_bundle.exists():
        raise FileExistsError(f"refusing to overwrite output bundle: {output_bundle}")

    weights_path = weights_bundle / "policy_weights.pkl.gz"
    weights_metadata_path = weights_bundle / "metadata.json"
    runtime_metadata_path = runtime_bundle / "metadata.json"
    for path in (weights_path, weights_metadata_path, runtime_metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"required bundle file not found: {path}")

    weights_metadata = json.loads(weights_metadata_path.read_text(encoding="utf-8"))
    runtime_metadata = json.loads(runtime_metadata_path.read_text(encoding="utf-8"))
    if weights_metadata.get("algorithm_class") != runtime_metadata.get("algorithm_class"):
        raise ValueError("weights and runtime bundles use different algorithms")
    if observation_size(weights_metadata) != observation_size(runtime_metadata):
        raise ValueError("weights and runtime bundles use different observation sizes")

    weight_model = model_config(weights_metadata)
    runtime_model = model_config(runtime_metadata)
    architecture_keys = (
        "fcnet_hiddens",
        "fcnet_activation",
        "head_fcnet_hiddens",
        "head_fcnet_activation",
        "use_lstm",
        "lstm_cell_size",
    )
    mismatch = {
        key: (weight_model.get(key), runtime_model.get(key))
        for key in architecture_keys
        if weight_model.get(key) != runtime_model.get(key)
    }
    if mismatch:
        raise ValueError(f"bundle model architectures differ: {mismatch}")

    weight_hash = sha256(weights_path)
    runtime_hash = sha256(runtime_metadata_path)
    payload = runtime_metadata
    metadata = payload.setdefault("metadata", {})
    metadata["model_name"] = args.model_name
    metadata["packaging"] = {
        "kind": "composed lightweight submission bundle",
        "weights_bundle": str(weights_bundle),
        "weights_sha256": weight_hash,
        "runtime_bundle": str(runtime_bundle),
        "runtime_metadata_sha256": runtime_hash,
        "learner_updates": 0,
    }

    output_bundle.mkdir(parents=True)
    shutil.copy2(weights_path, output_bundle / weights_path.name)
    output_metadata_path = output_bundle / "metadata.json"
    output_metadata_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if sha256(output_bundle / weights_path.name) != weight_hash:
        raise RuntimeError("composed bundle weight hash does not match source")

    print(f"bundle={output_bundle}")
    print(f"weights_sha256={weight_hash}")
    print(f"runtime_metadata_sha256={runtime_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
