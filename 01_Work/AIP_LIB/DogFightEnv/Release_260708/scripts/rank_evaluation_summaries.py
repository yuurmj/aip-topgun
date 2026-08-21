from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rank saved rulebook evaluation summaries."
    )
    parser.add_argument("--root", default=".")
    parser.add_argument("--minimum-episodes", type=int, default=10)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--require-options", action="store_true")
    parser.add_argument("--target-mode")
    parser.add_argument("--unreal-observation-proxy")
    parser.add_argument("--ownship-speed-advantage-mps", type=float)
    parser.add_argument("--provider-action-repeat", type=int)
    parser.add_argument("--fresh-process-only", action="store_true")
    return parser.parse_args()


def describe_conditions(payload: dict) -> dict:
    options = payload.get("evaluation_options")
    verified = isinstance(options, dict)
    options = options if verified else {}
    episodes = payload.get("episodes", [])
    speed_advantage = options.get("ownship_speed_advantage_mps")
    if speed_advantage is None and episodes:
        differences = []
        for episode in episodes:
            own = episode.get("initial_speed_mps")
            target = episode.get("initial_target_speed_mps")
            if own is not None and target is not None:
                differences.append(float(own) - float(target))
        if differences and max(differences) - min(differences) <= 1e-6:
            speed_advantage = sum(differences) / len(differences)
        elif differences:
            speed_advantage = "mixed"
    residual = options.get("saved_guidance_residual_scale")
    return {
        "options_verified": verified,
        "target_mode": options.get("target_mode", payload.get("target_mode")),
        "unreal_observation_proxy": options.get("unreal_observation_proxy"),
        "ownship_speed_advantage_mps": speed_advantage,
        "provider_action_repeat": options.get("provider_action_repeat"),
        "fresh_process_per_seed": bool(payload.get("fresh_process_per_seed", False)),
        "residual_scale": residual,
    }


def condition_matches(args: argparse.Namespace, conditions: dict) -> bool:
    if args.require_options and not conditions["options_verified"]:
        return False
    if args.target_mode and conditions["target_mode"] != args.target_mode:
        return False
    if (
        args.unreal_observation_proxy
        and conditions["unreal_observation_proxy"] != args.unreal_observation_proxy
    ):
        return False
    if args.ownship_speed_advantage_mps is not None:
        actual = conditions["ownship_speed_advantage_mps"]
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), args.ownship_speed_advantage_mps, abs_tol=1e-6
        ):
            return False
    if (
        args.provider_action_repeat is not None
        and conditions["provider_action_repeat"] != args.provider_action_repeat
    ):
        return False
    if args.fresh_process_only and not conditions["fresh_process_per_seed"]:
        return False
    return True


def main() -> int:
    args = parse_args()
    rows = []
    scanned = 0
    for path in Path(args.root).rglob("summary.json"):
        scanned += 1
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
            summary = payload.get("summary", {})
            episodes = int(summary.get("episodes", 0))
            points = summary.get("rulebook_points")
            if episodes < args.minimum_episodes or points is None:
                continue
            points = float(points)
            conditions = describe_conditions(payload)
            if not condition_matches(args, conditions):
                continue
            rows.append(
                {
                    "name": path.parent.name,
                    "episodes": episodes,
                    "points": points,
                    "ppe": points / episodes,
                    "wins": int(summary.get("rulebook_wins", 0)),
                    "losses": int(summary.get("rulebook_losses", 0)),
                    "draws": int(summary.get("rulebook_draws", 0)),
                    "kills": int(summary.get("ownship_kills", 0)),
                    "crashes": int(summary.get("crash", 0)),
                    "damage": int(summary.get("damage_successes", 0)),
                    "bundle": Path(str(payload.get("bundle_dir", ""))).name,
                    "options": "verified" if conditions["options_verified"] else "legacy",
                    "target": conditions["target_mode"],
                    "speed_adv": conditions["ownship_speed_advantage_mps"],
                    "proxy": conditions["unreal_observation_proxy"],
                    "repeat": conditions["provider_action_repeat"],
                    "fresh": conditions["fresh_process_per_seed"],
                    "residual": conditions["residual_scale"],
                    "path": str(path.resolve()),
                }
            )
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            continue

    rows.sort(
        key=lambda row: (
            row["ppe"],
            row["damage"] / row["episodes"],
            row["kills"] / row["episodes"],
            -row["crashes"] / row["episodes"],
        ),
        reverse=True,
    )
    print(f"scanned={scanned} comparable={len(rows)}")
    fields = (
        "ppe",
        "episodes",
        "points",
        "wins",
        "losses",
        "draws",
        "kills",
        "crashes",
        "damage",
        "options",
        "target",
        "speed_adv",
        "proxy",
        "repeat",
        "fresh",
        "residual",
        "name",
        "bundle",
        "path",
    )
    print("\t".join(fields))
    for row in rows[: max(0, args.limit)]:
        print("\t".join(str(row[field]) for field in fields))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
