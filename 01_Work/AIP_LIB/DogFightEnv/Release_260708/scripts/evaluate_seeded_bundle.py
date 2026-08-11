from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from DogFightEnvWrapper import DogFightWrapper
from dogfight.ai.rl_action_provider import (
    AltitudeRecoveryActionProvider,
    DirectSACMLPActionProvider,
    RLActionProvider,
)
from dogfight.ai.rllib_utils import build_algorithm_from_bundle
from dogfight.ai.student_hooks import load_reward_hook
from dogfight.sim.state_schema import StateIndex


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a lightweight RL bundle on fixed episode seeds."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument(
        "--env-config-bundle-dir",
        help=(
            "Optional bundle whose saved env_config is used for evaluation. "
            "The policy weights still come from --bundle-dir."
        ),
    )
    parser.add_argument(
        "--target-mode",
        choices=["fixed", "loiter", "autopilot", "behavior_tree"],
    )
    parser.add_argument("--target-behavior-dll", default="AIP_BASE_target.dll")
    parser.add_argument("--seeds", default="1001,1002,1003,1004,1005,1006,1007,1008,1009,1010")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-engage-time", type=float)
    parser.add_argument("--control-surface-scale", type=float)
    parser.add_argument("--ownship-speed-mps", type=float)
    parser.add_argument("--target-speed-mps", type=float)
    parser.add_argument(
        "--target-speed-range-mps",
        help="Randomized target initial speed as minimum,maximum m/s.",
    )
    parser.add_argument("--initial-distance-m", type=float)
    parser.add_argument(
        "--distance-range-m",
        help="Randomized initial separation as minimum,maximum meters.",
    )
    parser.add_argument("--heading-jitter-deg", type=float)
    parser.add_argument("--position-radius-m", type=float)
    parser.add_argument(
        "--legacy-scenario-indices",
        help="Override ref_old_random scenario indices with a comma-separated list.",
    )
    parser.add_argument(
        "--direct-sac-mlp",
        action="store_true",
        help="Run a standard SAC MLP actor directly without starting Ray.",
    )
    parser.add_argument("--recovery-bundle-dir")
    parser.add_argument("--recovery-trigger-altitude-m", type=float, default=3000.0)
    parser.add_argument("--recovery-release-altitude-m", type=float, default=4500.0)
    parser.add_argument(
        "--provider-primary-bias",
        default="0,0,0,0",
        help="Four-value RL-action bias applied only to the primary policy.",
    )
    parser.add_argument("--disable-ownship-randomization", action="store_true")
    parser.add_argument("--target-loiter-bank-deg", type=float)
    parser.add_argument("--target-loiter-pitch-deg", type=float, default=0.0)
    parser.add_argument("--target-control-surface-scale", type=float)
    parser.add_argument("--residual-surface-scale", type=float)
    parser.add_argument("--residual-throttle-scale", type=float, default=0.05)
    parser.add_argument(
        "--residual-baseline",
        default="0.0,-0.0003,-0.001,-0.04",
        help="Four comma-separated baseline values in RL action space.",
    )
    return parser.parse_args()


def load_bundle_env_config(bundle_dir: Path) -> dict:
    metadata = json.loads((bundle_dir / "metadata.json").read_text(encoding="utf-8"))
    config = dict(metadata.get("algorithm_config", {}).get("env_config", {}))
    if not config:
        raise RuntimeError(f"Bundle has no saved env_config: {bundle_dir}")
    return config


def initial_geometry(env) -> dict:
    ownship = env._ownship_state
    target = env._target_state
    geo = env._geo_info
    return {
        "initial_distance_m": float(geo._get_distance(ownship, target)),
        "initial_ata_deg": abs(
            float(geo._get_antenna_train_angle(ownship, target, True))
        ),
        "initial_aa_deg": abs(float(geo._get_aspect_angle(ownship, target, True))),
        # Geodetic ALT/KCAS are populated after the first native step. NED D and
        # the configured speed are valid immediately after reset.
        "initial_altitude_m": -float(ownship[StateIndex.D]),
        "initial_speed_mps": float(env.config["ownship"][6]),
        "initial_target_speed_mps": float(
            env.info.get("initial_target_speed_mps", env.config["target"][6])
        ),
        "initial_heading_deg": float(ownship[StateIndex.YAW]),
        "initial_n_m": float(ownship[StateIndex.N]),
        "initial_e_m": float(ownship[StateIndex.E]),
        "initial_d_m": float(ownship[StateIndex.D]),
        "legacy_scenario_index": int(
            env.info.get("legacy_scenario_index", -1)
        ),
        "episode_target_mode": str(env.config.get("target_mode", "")),
    }


def ownship_in_wez(env) -> bool:
    distance = float(env._geo_info._get_distance(
        env._ownship_state, env._target_state
    ))
    ata = abs(float(env._geo_info._get_antenna_train_angle(
        env._ownship_state, env._target_state, True
    )))
    wez = env._wez
    return (
        float(wez["min_range_m"]) <= distance <= float(wez["max_range_m"])
        and ata <= float(wez["angle_deg"]) / 2.0
    )


def run_episode(env, seed: int) -> dict:
    env.reset(seed=seed)
    initial = initial_geometry(env)
    started_in_wez = ownship_in_wez(env)
    was_in_wez = started_in_wez
    entered_wez_from_outside = False
    terminated = truncated = False
    min_altitude = float("inf")
    min_distance = float("inf")
    trajectory = hashlib.sha256()
    info = {}
    total_reward = 0.0

    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        total_reward += float(reward)
        is_in_wez = bool(env._in_wez)
        if is_in_wez and not was_in_wez:
            entered_wez_from_outside = True
        was_in_wez = is_in_wez
        state = np.asarray(env._ownship_state, dtype=np.float64)
        altitude = float(state[StateIndex.ALT])
        if altitude > 0.0:
            min_altitude = min(min_altitude, altitude)
        distance = float(env._geo_info._get_distance(
            env._ownship_state, env._target_state
        ))
        min_distance = min(min_distance, distance)
        trajectory.update(state.tobytes())

    steps = int(info.get("ep_step_count", 0))
    max_consecutive = int(info.get("ep_max_consecutive_wez_steps", 0))
    step_seconds = float(env._step_ratio) / float(env._sim_hz)
    final_distance = float(env._geo_info._get_distance(
        env._ownship_state, env._target_state
    ))
    final_ata = abs(float(env._geo_info._get_antenna_train_angle(
        env._ownship_state, env._target_state, True
    )))
    target_health = float(info.get("target_health", 0.0))
    cumulative_target_damage = max(0.0, 1.0 - target_health)
    return {
        "seed": seed,
        **initial,
        "outcome": str(info.get("outcome", "")),
        "end_condition": str(info.get("end_condition", "")),
        "episode_steps": steps,
        "survival_time_s": steps * step_seconds,
        "min_altitude_m": min_altitude if np.isfinite(min_altitude) else 0.0,
        "min_distance_m": min_distance if np.isfinite(min_distance) else 0.0,
        "final_distance_m": final_distance,
        "final_ata_deg": final_ata,
        "ownship_health": float(info.get("ownship_health", 0.0)),
        "target_health": target_health,
        "ownship_damage": float(info.get("ownship_damage", 0.0)),
        "target_damage": float(info.get("target_damage", 0.0)),
        "wez_steps": int(info.get("ep_wez_steps", 0)),
        "max_consecutive_wez_steps": max_consecutive,
        "max_consecutive_wez_s": max_consecutive * step_seconds,
        "wez_1s_achieved": max_consecutive * step_seconds >= 1.0,
        "started_in_wez": started_in_wez,
        "entered_wez_from_outside": entered_wez_from_outside,
        "cumulative_target_damage": cumulative_target_damage,
        "total_reward": total_reward,
        "trajectory_sha256": trajectory.hexdigest(),
    }


def summarize(rows: list[dict]) -> dict:
    count = len(rows)
    outcomes = {name: sum(row["outcome"] == name for row in rows) for name in (
        "win", "loss", "draw", "timeout", "crash"
    )}
    ownship_kills = sum(
        row["end_condition"].strip().lower() == "target destroyed" for row in rows
    )
    target_crash_wins = sum(
        row["outcome"] == "win"
        and row["end_condition"].strip().lower() != "target destroyed"
        for row in rows
    )
    return {
        "episodes": count,
        **outcomes,
        "ownship_kills": ownship_kills,
        "target_crash_wins": target_crash_wins,
        "crash_rate": outcomes["crash"] / count,
        "timeout_rate": outcomes["timeout"] / count,
        "mean_survival_time_s": sum(row["survival_time_s"] for row in rows) / count,
        "mean_min_altitude_m": sum(row["min_altitude_m"] for row in rows) / count,
        "mean_min_distance_m": sum(row["min_distance_m"] for row in rows) / count,
        "best_min_distance_m": min(row["min_distance_m"] for row in rows),
        "min_initial_distance_m": min(row["initial_distance_m"] for row in rows),
        "max_initial_distance_m": max(row["initial_distance_m"] for row in rows),
        "min_initial_target_speed_mps": min(
            row.get("initial_target_speed_mps", 0.0) for row in rows
        ),
        "max_initial_target_speed_mps": max(
            row.get("initial_target_speed_mps", 0.0) for row in rows
        ),
        "wez_1s_successes": sum(bool(row["wez_1s_achieved"]) for row in rows),
        "started_in_wez_count": sum(bool(row["started_in_wez"]) for row in rows),
        "entered_wez_from_outside_count": sum(
            bool(row["entered_wez_from_outside"]) for row in rows
        ),
        "max_consecutive_wez_s": max(row["max_consecutive_wez_s"] for row in rows),
        "damage_successes": sum(row["cumulative_target_damage"] > 0.0 for row in rows),
        "mean_cumulative_target_damage": sum(
            row["cumulative_target_damage"] for row in rows
        ) / count,
        "max_cumulative_target_damage": max(
            row["cumulative_target_damage"] for row in rows
        ),
        "unique_trajectory_hashes": len({row["trajectory_sha256"] for row in rows}),
        "unique_initial_geometries": len({
            (
                round(row["initial_distance_m"], 6),
                round(row["initial_heading_deg"], 6),
                round(row["initial_n_m"], 6),
                round(row["initial_e_m"], 6),
                round(row["initial_d_m"], 6),
                round(row.get("initial_target_speed_mps", 0.0), 6),
            )
            for row in rows
        }),
    }


def main():
    args = parse_args()
    bundle_dir = Path(args.bundle_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    seeds = [int(item.strip()) for item in args.seeds.split(",") if item.strip()]

    env_config_bundle_dir = (
        Path(args.env_config_bundle_dir).resolve()
        if args.env_config_bundle_dir
        else bundle_dir
    )
    env_config = load_bundle_env_config(env_config_bundle_dir)
    if args.target_mode:
        env_config["target_mode"] = args.target_mode
    if args.target_loiter_bank_deg is not None:
        env_config["target_mode"] = "loiter"
        env_config["target_loiter"] = {
            "enabled": True,
            "bank": float(args.target_loiter_bank_deg),
            "pitch": float(args.target_loiter_pitch_deg),
        }
    if args.target_control_surface_scale is not None:
        env_config["target_control_surface_scale"] = float(
            args.target_control_surface_scale
        )
    if args.residual_surface_scale is not None:
        baseline = [
            float(part.strip()) for part in args.residual_baseline.split(",")
        ]
        if len(baseline) != 4:
            raise ValueError("--residual-baseline must contain four values")
        env_config["residual_action"] = {
            "enabled": True,
            "baseline": baseline,
            "scale": [
                float(args.residual_surface_scale),
                float(args.residual_surface_scale),
                float(args.residual_surface_scale),
                float(args.residual_throttle_scale),
            ],
        }
    if args.max_engage_time is not None:
        env_config["max_engage_time"] = args.max_engage_time
    if args.control_surface_scale is not None:
        env_config["control_surface_scale"] = args.control_surface_scale
    if args.ownship_speed_mps is not None:
        env_config["ownship"] = list(env_config["ownship"])
        env_config["ownship"][6] = args.ownship_speed_mps
    if args.target_speed_mps is not None:
        env_config["target"] = list(env_config["target"])
        env_config["target"][6] = args.target_speed_mps
    if args.target_speed_range_mps is not None:
        parts = [
            part.strip() for part in args.target_speed_range_mps.split(",")
        ]
        if len(parts) != 2:
            raise ValueError("--target-speed-range-mps must be minimum,maximum")
        env_config["target_speed_range_mps"] = [
            float(parts[0]),
            float(parts[1]),
        ]
    if args.initial_distance_m is not None:
        env_config["target"] = list(env_config["target"])
        env_config["target"][0] = (
            float(env_config["ownship"][0]) + args.initial_distance_m
        )
    if any(value is not None for value in (
        args.distance_range_m,
        args.heading_jitter_deg,
        args.position_radius_m,
    )):
        randomization = dict(env_config.get("ownship_randomization", {}))
        randomization["enabled"] = True
        if args.distance_range_m is not None:
            parts = [part.strip() for part in args.distance_range_m.split(",")]
            if len(parts) != 2:
                raise ValueError("--distance-range-m must be minimum,maximum")
            randomization["distance_range_m"] = [float(parts[0]), float(parts[1])]
        if args.heading_jitter_deg is not None:
            randomization["r_heading"] = args.heading_jitter_deg
        if args.position_radius_m is not None:
            randomization["radius"] = args.position_radius_m
        env_config["ownship_randomization"] = randomization
    if args.disable_ownship_randomization:
        env_config["ownship_randomization"] = {"enabled": False}
    if args.legacy_scenario_indices is not None:
        indices = [
            int(part.strip())
            for part in args.legacy_scenario_indices.split(",")
            if part.strip()
        ]
        if not indices:
            raise ValueError("--legacy-scenario-indices must not be empty")
        scenario = dict(env_config.get("initial_scenario", {}))
        scenario.update({
            "mode": "ref_old_random",
            "legacy_use_random_scenario": True,
            "legacy_use_first_scenario_only": False,
            "legacy_scenario_indices": indices,
        })
        env_config["initial_scenario"] = scenario
    env_config["target_behavior_dll"] = args.target_behavior_dll

    reward_fn = None
    reward_module = str(env_config.get("reward_module", "")).strip()
    if reward_module:
        reward_fn, defaults = load_reward_hook(reward_module)
        env_config["reward"] = {**defaults, **dict(env_config.get("reward", {}))}

    if args.direct_sac_mlp:
        provider = DirectSACMLPActionProvider(bundle_dir=str(bundle_dir))
        if args.recovery_bundle_dir:
            primary_bias = np.asarray([
                float(part.strip())
                for part in args.provider_primary_bias.split(",")
            ], dtype=np.float32)
            provider = AltitudeRecoveryActionProvider(
                provider,
                DirectSACMLPActionProvider(
                    bundle_dir=str(Path(args.recovery_bundle_dir).resolve())
                ),
                trigger_altitude_m=args.recovery_trigger_altitude_m,
                release_altitude_m=args.recovery_release_altitude_m,
                primary_bias=primary_bias,
            )
    else:
        provider = RLActionProvider(
            bundle_dir=str(bundle_dir),
            algorithm_factory=build_algorithm_from_bundle,
            explore=False,
        )
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=provider,
        reward_fn=reward_fn,
    )
    try:
        rows = []
        for seed in seeds:
            row = run_episode(env, seed)
            rows.append(row)
            print(
                f"seed={seed} outcome={row['outcome']} "
                f"time={row['survival_time_s']:.1f}s "
                f"wez_hold={row['max_consecutive_wez_s']:.1f}s "
                f"cumulative_target_damage={row['cumulative_target_damage']:.4f}"
            )
    finally:
        env.close()

    csv_path = output_dir / "episodes.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "bundle_dir": str(bundle_dir),
        "target_mode": env_config["target_mode"],
        "seeds": seeds,
        "summary": summarize(rows),
        "episodes": rows,
    }
    json_path = output_dir / "summary.json"
    json_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print(f"CSV: {csv_path}")
    print(f"JSON: {json_path}")


if __name__ == "__main__":
    main()
