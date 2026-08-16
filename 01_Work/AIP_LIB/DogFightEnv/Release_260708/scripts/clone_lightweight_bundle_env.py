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
    parser.add_argument(
        "--target-mode",
        choices=["fixed", "loiter", "autopilot", "behavior_tree"],
        default="loiter",
    )
    parser.add_argument("--target-control-surface-scale", type=float)
    parser.add_argument("--target-loiter-bank-deg", required=True, type=float)
    parser.add_argument("--target-loiter-pitch-deg", default=0.0, type=float)
    parser.add_argument(
        "--heading-jitter-deg",
        type=float,
        help="Optional ownship heading jitter saved in the cloned env config.",
    )
    parser.add_argument(
        "--residual-baseline",
        required=True,
        help="Four comma-separated values in RL action space.",
    )
    parser.add_argument("--disable-residual-action", action="store_true")
    parser.add_argument("--feedback-observation-index", type=int)
    parser.add_argument(
        "--feedback-action-gains",
        help="Four comma-separated gains for saved observation action feedback.",
    )
    parser.add_argument(
        "--feedback-action-bias",
        default="0,0,0,0",
        help="Four comma-separated fixed action offsets saved with feedback.",
    )
    parser.add_argument(
        "--feedback-branches-json",
        help="Optional ordered JSON branch list for latched observation feedback.",
    )
    parser.add_argument("--feedback-branches-file")
    parser.add_argument("--feedback-branch-observation-index", type=int)
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
    env_config["target_mode"] = args.target_mode
    env_config["target_loiter"] = {
        "enabled": True,
        "bank": args.target_loiter_bank_deg,
        "pitch": args.target_loiter_pitch_deg,
    }
    residual = env_config.setdefault("residual_action", {})
    residual["enabled"] = not args.disable_residual_action
    residual["baseline"] = baseline
    if args.target_control_surface_scale is not None:
        env_config["target_control_surface_scale"] = float(
            args.target_control_surface_scale
        )
    if args.heading_jitter_deg is not None:
        if args.heading_jitter_deg < 0.0:
            raise ValueError("--heading-jitter-deg must be non-negative")
        randomization = env_config.setdefault("ownship_randomization", {})
        randomization["enabled"] = True
        randomization["r_heading"] = args.heading_jitter_deg
    if args.feedback_observation_index is not None:
        feedback_gains = [
            float(value) for value in str(args.feedback_action_gains).split(",")
        ]
        if len(feedback_gains) != 4:
            raise ValueError("--feedback-action-gains must contain four values")
        feedback_bias = [
            float(value) for value in str(args.feedback_action_bias).split(",")
        ]
        if len(feedback_bias) != 4:
            raise ValueError("--feedback-action-bias must contain four values")
        env_config["observation_action_feedback"] = {
            "enabled": True,
            "observation_index": args.feedback_observation_index,
            "action_gains": feedback_gains,
            "action_bias": feedback_bias,
        }
        if args.feedback_branches_json or args.feedback_branches_file:
            branches_payload = (
                Path(args.feedback_branches_file).read_text(encoding="utf-8")
                if args.feedback_branches_file
                else args.feedback_branches_json
            )
            branches = json.loads(branches_payload)
            if not isinstance(branches, list) or not branches:
                raise ValueError("--feedback-branches-json must be a non-empty list")
            env_config["observation_action_feedback"].update({
                "latch_branch": True,
                "branch_observation_index": (
                    args.feedback_branch_observation_index
                    if args.feedback_branch_observation_index is not None
                    else args.feedback_observation_index
                ),
                "branches": branches,
            })

    metadata = payload.setdefault("metadata", {})
    metadata["model_name"] = args.model_name
    metadata["target_mode"] = args.target_mode
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
