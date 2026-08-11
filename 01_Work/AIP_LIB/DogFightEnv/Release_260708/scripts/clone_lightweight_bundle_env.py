from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone lightweight policy weights while replacing saved env settings."
    )
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--target-loiter-bank-deg", required=True, type=float)
    parser.add_argument("--target-loiter-pitch-deg", default=0.0, type=float)
    parser.add_argument(
        "--residual-baseline",
        required=True,
        help="Four comma-separated values in RL action space.",
    )
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source = Path(args.source).resolve()
    output = Path(args.output).resolve()
    source_metadata = source / "metadata.json"
    source_weights = source / "policy_weights.pkl.gz"
    if not source_metadata.is_file() or not source_weights.is_file():
        raise FileNotFoundError(f"Incomplete source bundle: {source}")
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite existing output: {output}")

    baseline = [float(value) for value in args.residual_baseline.split(",")]
    if len(baseline) != 4:
        raise ValueError("--residual-baseline must contain four values")

    payload = json.loads(source_metadata.read_text(encoding="utf-8"))
    env_config = payload["algorithm_config"]["env_config"]
    env_config["target_mode"] = "loiter"
    env_config["target_loiter"] = {
        "enabled": True,
        "bank": args.target_loiter_bank_deg,
        "pitch": args.target_loiter_pitch_deg,
    }
    residual = env_config.setdefault("residual_action", {})
    residual["enabled"] = True
    residual["baseline"] = baseline

    metadata = payload.setdefault("metadata", {})
    metadata["model_name"] = args.model_name
    metadata["target_mode"] = "loiter"
    metadata["packaging"] = {
        "kind": "environment-only clone",
        "source_bundle": str(source),
        "source_weights_sha256": sha256(source_weights),
        "learner_updates": 0,
    }

    output.mkdir(parents=True)
    shutil.copy2(source_weights, output / source_weights.name)
    (output / source_metadata.name).write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )

    cloned_hash = sha256(output / source_weights.name)
    if cloned_hash != metadata["packaging"]["source_weights_sha256"]:
        raise RuntimeError("Cloned policy weight hash does not match source")
    print(f"bundle={output}")
    print(f"policy_weights_sha256={cloned_hash}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
