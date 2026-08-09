# -*- coding: utf-8 -*-
"""Evaluate the v21 iter_0001 PPO checkpoint at 900 m without training.

This script never calls ``algorithm.train()``. It restores the native RLlib
checkpoint, runs deterministic policy inference, saves Tacview CSV replays, and
writes per-episode plus aggregate success metrics.

Default evaluation:
- v21 checkpoint: stage 0 / iter_0001
- 900 m center, left, right geometry
- 10 episodes per geometry (30 total)
- ownship 264 m/s, target 250 m/s
- 8 second engagement
- small seeded attitude-only jitter to avoid identical deterministic runs
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shutil
import sys
from copy import deepcopy
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    text = str(path)
    if text not in sys.path:
        sys.path.insert(0, text)

_pythonpath_entries = [str(ROOT), str(SRC)]
_existing_pythonpath = os.environ.get("PYTHONPATH", "")
if _existing_pythonpath:
    _pythonpath_entries.append(_existing_pythonpath)
os.environ["PYTHONPATH"] = os.pathsep.join(_pythonpath_entries)
os.environ.setdefault("RAY_raylet_start_wait_time_s", "60")


DEFAULT_CHECKPOINT = (
    ROOT
    / "artifacts"
    / "curriculum"
    / "team01"
    / "ppo_mlp_v21_900m_fire_cone_refine"
    / "stage_0_attack_refine_900m_fire_cone_8s"
    / "checkpoints"
    / "iter_0001"
)
DEFAULT_OUTPUT_DIR = (
    ROOT
    / "artifacts"
    / "evaluation"
    / "team01"
    / "ppo_mlp_v21_iter0001_900m_eval_30"
)

TARGET_NORTH_M = 6000.0
TARGET_ALTITUDE_D_M = -7000.0
TARGET_SPEED_MPS = 250.0
OWNSHIP_SPEED_MPS = 264.0

RESULT_FIELDS = [
    "scenario",
    "episode",
    "seed",
    "steps",
    "outcome",
    "end_condition",
    "ownship_health",
    "target_health",
    "target_damage",
    "damage_occurred",
    "crashed",
    "initial_distance_m",
    "min_distance_m",
    "mean_ata_deg",
    "min_ata_deg",
    "firing_half_angle_deg",
    "ata_within_firing_cone_rate",
    "ata_within_5deg_rate",
    "mean_abs_roll_deg",
    "max_abs_roll_deg",
    "mean_abs_pitch_deg",
    "altitude_drop_m",
    "min_altitude_m",
    "total_reward",
    "replay_dir",
    "ownship_log",
    "target_log",
    "summary_json",
]

SUMMARY_FIELDS = [
    "group",
    "episodes",
    "crash_rate",
    "damage_rate",
    "mean_target_damage",
    "mean_ata_deg",
    "min_ata_deg",
    "firing_half_angle_deg",
    "ata_within_firing_cone_rate",
    "ata_within_5deg_rate",
    "mean_abs_roll_deg",
    "max_abs_roll_deg",
    "mean_altitude_drop_m",
    "pass_crash",
    "pass_damage_rate",
    "pass_mean_damage",
    "pass_ata",
    "pass_mean_roll",
    "pass_max_roll",
    "pass_altitude_drop",
    "pass_all",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate v21 iter_0001 at 900 m without PPO updates."
    )
    parser.add_argument(
        "--checkpoint",
        default=str(DEFAULT_CHECKPOINT),
        help="Native RLlib checkpoint directory. Relative paths are resolved from Release root.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Evaluation artifact directory. Relative paths are resolved from Release root.",
    )
    parser.add_argument(
        "--episodes-per-scenario",
        type=int,
        default=10,
        help="Episodes for each of center/left/right.",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=220,
        help="Safety cap for evaluation environment steps.",
    )
    parser.add_argument(
        "--base-seed",
        type=int,
        default=200900,
        help="Base seed for reproducible attitude jitter.",
    )
    parser.add_argument(
        "--exact-repeat",
        action="store_true",
        help="Disable attitude jitter. Deterministic repetitions may then be identical.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete an existing evaluation output directory before running.",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate the checkpoint and required student files without starting Ray.",
    )
    return parser.parse_args()


def _resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


def _validate_checkpoint(checkpoint: Path) -> None:
    required = [
        checkpoint / "rllib_checkpoint.json",
        checkpoint / "algorithm_state.pkl",
        checkpoint / "learner_group",
    ]
    missing = [path for path in required if not path.exists()]
    if missing:
        details = "\n".join(f"  - {path}" for path in missing)
        raise FileNotFoundError(
            "The v21 native checkpoint is incomplete or not found:\n" + details
        )


def _prepare_output_dir(output_dir: Path, overwrite: bool) -> None:
    if output_dir.exists():
        if not overwrite:
            raise RuntimeError(
                f"Evaluation output already exists: {output_dir}\n"
                "Use --overwrite to replace it, or choose another --output-dir."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)


def _make_scenario(
    name: str,
    *,
    east_m: float,
    heading_deg: float,
    exact_repeat: bool,
) -> dict[str, Any]:
    lateral_m = float(east_m)
    forward_m = math.sqrt(900.0 ** 2 - lateral_m ** 2)
    scenario: dict[str, Any] = {
        "name": name,
        "weight": 1.0,
        "ownship": [
            TARGET_NORTH_M + forward_m,
            lateral_m,
            TARGET_ALTITUDE_D_M,
            0.0,
            0.0,
            heading_deg,
            OWNSHIP_SPEED_MPS,
        ],
        "target": [
            TARGET_NORTH_M,
            0.0,
            TARGET_ALTITUDE_D_M,
            0.0,
            0.0,
            180.0,
            TARGET_SPEED_MPS,
        ],
        "target_mode": "autopilot",
        "target_autopilot": {
            "heading_cmd": 180.0,
            "altitude_cmd": TARGET_ALTITUDE_D_M,
            "speed_cmd": TARGET_SPEED_MPS,
        },
    }
    if not exact_repeat:
        # The requested 900 m radial distance and speed remain fixed. Only a mild
        # initial attitude variation is sampled, so repetitions are not exact copies.
        scenario["ownship_randomization"] = {
            "enabled": True,
            "r_roll": 2.0,
            "r_pitch": 1.0,
            "r_heading": 0.25,
        }
    return scenario


def _scenario_definitions(exact_repeat: bool) -> list[dict[str, Any]]:
    return [
        _make_scenario(
            "900m_center",
            east_m=0.0,
            heading_deg=180.0,
            exact_repeat=exact_repeat,
        ),
        _make_scenario(
            "900m_left",
            east_m=-20.0,
            heading_deg=179.5,
            exact_repeat=exact_repeat,
        ),
        _make_scenario(
            "900m_right",
            east_m=20.0,
            heading_deg=180.5,
            exact_repeat=exact_repeat,
        ),
    ]


def _load_runtime_dependencies():
    import ray
    from ray.tune.registry import register_env

    from DogFightEnvWrapper import DogFightWrapper
    from dogfight.ai.policy_probe_logger import _PolicyActorView
    from dogfight.ai.rllib_utils import build_algorithm_config
    from dogfight.ai.student_hooks import load_observation_hook, load_reward_hook

    return {
        "ray": ray,
        "register_env": register_env,
        "DogFightWrapper": DogFightWrapper,
        "PolicyActorView": _PolicyActorView,
        "build_algorithm_config": build_algorithm_config,
        "load_observation_hook": load_observation_hook,
        "load_reward_hook": load_reward_hook,
    }


def _make_env_creator(deps: dict[str, Any]):
    DogFightWrapper = deps["DogFightWrapper"]
    load_observation_hook = deps["load_observation_hook"]
    load_reward_hook = deps["load_reward_hook"]

    def env_creator(env_config):
        cfg = dict(env_config)
        cfg["_runner_index"] = getattr(
            env_config,
            "worker_index",
            cfg.get("_runner_index", "v21_eval"),
        )
        cfg["_env_index"] = getattr(
            env_config,
            "vector_index",
            cfg.get("_env_index", 0),
        )
        reward_fn = None
        observation_hook = None

        reward_module = str(cfg.get("reward_module", "")).strip()
        if reward_module:
            reward_fn, reward_config = load_reward_hook(reward_module)
            cfg.setdefault("reward", deepcopy(reward_config))

        observation_module = str(cfg.get("observation_module", "")).strip()
        if observation_module:
            observation_hook = load_observation_hook(observation_module)
            cfg["observation_mode"] = observation_hook["mode"]
            cfg["observation_module"] = observation_module
            cfg["observation_summary"] = observation_hook["description"]

        env = DogFightWrapper(
            cfg,
            reward_fn=reward_fn,
            observation_fn=(
                observation_hook["build_observation"] if observation_hook else None
            ),
            observation_size=observation_hook["size"] if observation_hook else None,
            observation_low=observation_hook["low"] if observation_hook else None,
            observation_high=observation_hook["high"] if observation_hook else None,
        )
        if reward_module and "reward" in cfg:
            env.config["reward"] = deepcopy(cfg["reward"])
        return env

    return env_creator


def _base_env_config(env_creator) -> dict[str, Any]:
    preview_config = {
        "observation_mode": "tactical16",
        "target_mode": "autopilot",
        "target_behavior_dll": "AIP_BASE_target.dll",
        "ownship_control_mode": "rl",
        "reward_module": "student.my_reward_v21",
    }
    preview = env_creator(preview_config)
    try:
        config = {
            **preview_config,
            "observation_mode": preview.config["observation_mode"],
            "reward": deepcopy(preview.config["reward"]),
            "wez": deepcopy(preview.config["wez"]),
            "step_ratio": 6,
            "max_engage_time": 8.0,
            "min_altitude": 6550.0,
            "episode_step_limit": 8 * 60,
            "target_autopilot": {
                "heading_cmd": 180.0,
                "altitude_cmd": TARGET_ALTITUDE_D_M,
                "speed_cmd": TARGET_SPEED_MPS,
            },
        }
    finally:
        preview.close()
    return config


def _start_ray(ray_module) -> None:
    ray_module.shutdown()
    ray_module.init(
        ignore_reinit_error=True,
        include_dashboard=False,
        runtime_env={
            "env_vars": {
                "PYTHONPATH": os.environ["PYTHONPATH"],
                "RAY_raylet_start_wait_time_s": os.environ[
                    "RAY_raylet_start_wait_time_s"
                ],
            }
        },
    )


def _build_algorithm(
    deps: dict[str, Any],
    env_creator,
    env_config: dict[str, Any],
    checkpoint: Path,
):
    env_name = "dogfight-v21-iter0001-900m-eval"
    deps["register_env"](env_name, env_creator)

    config = deps["build_algorithm_config"](
        algorithm_name="ppo",
        env_name=env_name,
        env_config=env_config,
        args={
            "framework": "torch",
            "num_env_runners": 0,
            "num_envs_per_env_runner": 1,
            "lr": 2.0e-5,
            "gamma": 0.99,
            "train_batch_size": 1024,
            "minibatch_size": 128,
            "gae_lambda": 0.95,
            "clip_param": 0.05,
            "tau": 0.005,
            "target_entropy": "auto",
            "replay_buffer_capacity": None,
            "model_config": {
                "enabled": True,
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                "head_fcnet_hiddens": [],
                "head_fcnet_activation": "relu",
                "vf_share_layers": True,
            },
            "network_spec": None,
            "use_lstm": False,
            "use_lstm_sac": False,
            "use_lstm_prioritized_replay": False,
            "lstm_scope": "actor_only",
            "lstm_cell_size": 64,
            "max_seq_len": 8,
            "debug_io": False,
        },
    )
    algorithm = config.build_algo()
    print(f"[v21-eval] Restoring native checkpoint: {checkpoint}")
    algorithm.restore(str(checkpoint))
    print("[v21-eval] Checkpoint restored. No training will be performed.")
    return algorithm


def _normalize_reset(result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        obs, info = result
        return obs, dict(info)
    return result, {}


def _normalize_step(result: Any) -> tuple[Any, float, bool, bool, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated), bool(truncated), dict(info)
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), False, dict(info)
    raise ValueError(f"Unexpected env.step return: {type(result).__name__}")


def _first_file_with_suffix(directory: Path, suffix: str) -> Path | None:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.name.endswith(suffix):
            return path
    return None


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        return list(csv.DictReader(file))


def _great_circle_bearing_distance(
    lat1_deg: float,
    lon1_deg: float,
    lat2_deg: float,
    lon2_deg: float,
) -> tuple[float, float]:
    radius_m = 6_371_000.0
    lat1 = math.radians(lat1_deg)
    lat2 = math.radians(lat2_deg)
    d_lat = lat2 - lat1
    d_lon = math.radians(lon2_deg - lon1_deg)
    hav = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(d_lon / 2.0) ** 2
    )
    horizontal_distance = 2.0 * radius_m * math.asin(
        min(1.0, math.sqrt(max(0.0, hav)))
    )
    y = math.sin(d_lon) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(d_lon)
    )
    bearing_deg = (math.degrees(math.atan2(y, x)) + 360.0) % 360.0
    return bearing_deg, horizontal_distance


def _antenna_train_angle_deg(
    ownship_row: dict[str, str],
    target_row: dict[str, str],
) -> tuple[float, float]:
    bearing_deg, horizontal_distance = _great_circle_bearing_distance(
        float(ownship_row["Latitude"]),
        float(ownship_row["Longitude"]),
        float(target_row["Latitude"]),
        float(target_row["Longitude"]),
    )
    altitude_delta = float(target_row["Altitude"]) - float(
        ownship_row["Altitude"]
    )
    elevation_rad = math.atan2(altitude_delta, max(horizontal_distance, 1e-9))
    pitch_rad = math.radians(float(ownship_row["Pitch (deg)"]))
    yaw_delta_rad = math.radians(
        ((bearing_deg - float(ownship_row["Yaw (deg)"]) + 180.0) % 360.0)
        - 180.0
    )
    dot = (
        math.cos(pitch_rad) * math.cos(elevation_rad) * math.cos(yaw_delta_rad)
        + math.sin(pitch_rad) * math.sin(elevation_rad)
    )
    ata_deg = math.degrees(math.acos(max(-1.0, min(1.0, dot))))
    distance_3d = math.hypot(horizontal_distance, altitude_delta)
    return ata_deg, distance_3d


def _mean(values: Iterable[float]) -> float:
    values_list = list(values)
    return sum(values_list) / len(values_list) if values_list else 0.0


def _is_crash(end_condition: str) -> bool:
    return end_condition in {
        "ownship altitude below min",
        "FDM Update Fail",
        "Ownship FDM output Fall",
    }


def _analyze_episode(
    *,
    scenario: str,
    episode: int,
    seed: int,
    steps: int,
    total_reward: float,
    info: dict[str, Any],
    replay_dir: Path,
    firing_half_angle_deg: float,
) -> dict[str, Any]:
    ownship_log = _first_file_with_suffix(replay_dir, "_ownship_(F-16)[Blue].csv")
    target_log = _first_file_with_suffix(replay_dir, "_target_(F-16)[Red].csv")
    summary_json = _first_file_with_suffix(replay_dir, "_summary.json")
    if ownship_log is None or target_log is None:
        raise FileNotFoundError(f"Tacview CSV pair not found in {replay_dir}")

    ownship_rows = _read_csv(ownship_log)
    target_rows = _read_csv(target_log)
    paired_count = min(len(ownship_rows), len(target_rows))
    if paired_count <= 0:
        raise RuntimeError(f"Empty Tacview CSV in {replay_dir}")

    ownship_rows = ownship_rows[:paired_count]
    target_rows = target_rows[:paired_count]
    ata_and_distance = [
        _antenna_train_angle_deg(own, target)
        for own, target in zip(ownship_rows, target_rows)
    ]
    ata_values = [item[0] for item in ata_and_distance]
    distance_values = [item[1] for item in ata_and_distance]
    roll_values = [abs(float(row["Roll (deg)"])) for row in ownship_rows]
    pitch_values = [abs(float(row["Pitch (deg)"])) for row in ownship_rows]
    altitude_values = [float(row["Altitude"]) for row in ownship_rows]

    summary: dict[str, Any] = {}
    if summary_json is not None:
        summary = json.loads(summary_json.read_text(encoding="utf-8"))

    end_condition = str(
        summary.get("end_condition", info.get("end_condition", ""))
    )
    outcome = str(summary.get("outcome", info.get("outcome", "")))
    ownship_health = float(
        summary.get("ownship_health", info.get("ownship_health", 1.0))
    )
    target_health = float(
        summary.get("target_health", info.get("target_health", 1.0))
    )
    target_damage = max(0.0, min(1.0, 1.0 - target_health))

    return {
        "scenario": scenario,
        "episode": episode,
        "seed": seed,
        "steps": steps,
        "outcome": outcome,
        "end_condition": end_condition,
        "ownship_health": ownship_health,
        "target_health": target_health,
        "target_damage": target_damage,
        "damage_occurred": target_damage > 1e-9,
        "crashed": _is_crash(end_condition),
        "initial_distance_m": distance_values[0],
        "min_distance_m": min(distance_values),
        "mean_ata_deg": _mean(ata_values),
        "min_ata_deg": min(ata_values),
        "firing_half_angle_deg": firing_half_angle_deg,
        "ata_within_firing_cone_rate": _mean(
            1.0 if value <= firing_half_angle_deg else 0.0
            for value in ata_values
        ),
        "ata_within_5deg_rate": _mean(
            1.0 if value <= 5.0 else 0.0 for value in ata_values
        ),
        "mean_abs_roll_deg": _mean(roll_values),
        "max_abs_roll_deg": max(roll_values),
        "mean_abs_pitch_deg": _mean(pitch_values),
        "altitude_drop_m": max(0.0, altitude_values[0] - min(altitude_values)),
        "min_altitude_m": min(altitude_values),
        "total_reward": total_reward,
        "replay_dir": str(replay_dir),
        "ownship_log": str(ownship_log),
        "target_log": str(target_log),
        "summary_json": str(summary_json) if summary_json else "",
    }


def _run_episode(
    *,
    env_creator,
    actor,
    base_env_config: dict[str, Any],
    scenario: dict[str, Any],
    episode: int,
    seed: int,
    max_steps: int,
    replay_dir: Path,
    firing_half_angle_deg: float,
) -> dict[str, Any]:
    config = deepcopy(base_env_config)
    config["initial_scenario"] = {
        "mode": "scenario_pool",
        "scenarios": [deepcopy(scenario)],
    }
    config["artifacts_dir"] = str(replay_dir)
    config["_runner_index"] = "v21_900m_eval"
    config["_env_index"] = episode
    replay_dir.mkdir(parents=True, exist_ok=True)

    env = env_creator(config)
    try:
        actor.reset_state()
        obs, info = _normalize_reset(env.reset(seed=seed))
        total_reward = 0.0
        terminated = False
        truncated = False
        steps = 0

        for steps in range(1, max_steps + 1):
            action = actor.compute(np.asarray(obs, dtype=np.float32))["action"]
            obs, reward, terminated, truncated, info = _normalize_step(env.step(action))
            total_reward += reward
            if terminated or truncated:
                break

        if not (terminated or truncated):
            info = dict(info)
            info.setdefault("end_condition", "evaluation step limit")
            info.setdefault("outcome", "preview")

        env.make_tacviewLog()
        return _analyze_episode(
            scenario=scenario["name"],
            episode=episode,
            seed=seed,
            steps=steps,
            total_reward=total_reward,
            info=info,
            replay_dir=replay_dir,
            firing_half_angle_deg=firing_half_angle_deg,
        )
    finally:
        env.close()


def _aggregate(group: str, rows: list[dict[str, Any]]) -> dict[str, Any]:
    episodes = len(rows)
    crash_rate = _mean(1.0 if row["crashed"] else 0.0 for row in rows)
    damage_rate = _mean(1.0 if row["damage_occurred"] else 0.0 for row in rows)
    mean_damage = _mean(float(row["target_damage"]) for row in rows)
    mean_ata = _mean(float(row["mean_ata_deg"]) for row in rows)
    min_ata = min(float(row["min_ata_deg"]) for row in rows)
    firing_half_angle_deg = float(rows[0]["firing_half_angle_deg"])
    firing_cone_rate = _mean(
        float(row["ata_within_firing_cone_rate"]) for row in rows
    )
    ata_five = _mean(float(row["ata_within_5deg_rate"]) for row in rows)
    mean_roll = _mean(float(row["mean_abs_roll_deg"]) for row in rows)
    max_roll = max(float(row["max_abs_roll_deg"]) for row in rows)
    mean_alt_drop = _mean(float(row["altitude_drop_m"]) for row in rows)

    pass_crash = crash_rate == 0.0
    pass_damage_rate = damage_rate >= (0.75 if group == "overall" else 0.60)
    pass_mean_damage = mean_damage >= 0.20
    pass_ata = mean_ata <= 5.0
    pass_mean_roll = mean_roll <= 15.0
    pass_max_roll = max_roll < 55.0
    pass_altitude_drop = mean_alt_drop <= 150.0
    pass_all = all(
        [
            pass_crash,
            pass_damage_rate,
            pass_mean_damage,
            pass_ata,
            pass_mean_roll,
            pass_max_roll,
            pass_altitude_drop,
        ]
    )

    return {
        "group": group,
        "episodes": episodes,
        "crash_rate": crash_rate,
        "damage_rate": damage_rate,
        "mean_target_damage": mean_damage,
        "mean_ata_deg": mean_ata,
        "min_ata_deg": min_ata,
        "firing_half_angle_deg": firing_half_angle_deg,
        "ata_within_firing_cone_rate": firing_cone_rate,
        "ata_within_5deg_rate": ata_five,
        "mean_abs_roll_deg": mean_roll,
        "max_abs_roll_deg": max_roll,
        "mean_altitude_drop_m": mean_alt_drop,
        "pass_crash": pass_crash,
        "pass_damage_rate": pass_damage_rate,
        "pass_mean_damage": pass_mean_damage,
        "pass_ata": pass_ata,
        "pass_mean_roll": pass_mean_roll,
        "pass_max_roll": pass_max_roll,
        "pass_altitude_drop": pass_altitude_drop,
        "pass_all": pass_all,
    }


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def _format_percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _print_summary(summary_rows: list[dict[str, Any]]) -> None:
    print("\n" + "=" * 76)
    print("v21 iter_0001 — 900 m fixed evaluation")
    print("=" * 76)
    for row in summary_rows:
        print(
            f"[{row['group']}] episodes={row['episodes']} "
            f"crash={_format_percent(row['crash_rate'])} "
            f"damage_rate={_format_percent(row['damage_rate'])} "
            f"mean_damage={_format_percent(row['mean_target_damage'])} "
            f"ATA={row['mean_ata_deg']:.2f}deg "
            f"fire_cone={_format_percent(row['ata_within_firing_cone_rate'])} "
            f"roll_mean={row['mean_abs_roll_deg']:.2f}deg "
            f"roll_max={row['max_abs_roll_deg']:.2f}deg "
            f"alt_drop={row['mean_altitude_drop_m']:.1f}m "
            f"PASS={row['pass_all']}"
        )
    overall = next(row for row in summary_rows if row["group"] == "overall")
    scenario_pass = all(
        row["pass_all"] for row in summary_rows if row["group"] != "overall"
    )
    final_pass = bool(overall["pass_all"] and scenario_pass)
    print("-" * 76)
    print(f"FINAL PASS: {final_pass}")
    print("925 m progression is allowed only when FINAL PASS is True.")
    print("Firing-cone occupancy is diagnostic and has no pass threshold yet.")
    print("=" * 76)


def main() -> int:
    args = parse_args()
    if args.episodes_per_scenario <= 0:
        raise ValueError("--episodes-per-scenario must be positive")
    if args.max_steps <= 0:
        raise ValueError("--max-steps must be positive")

    checkpoint = _resolve_path(args.checkpoint)
    output_dir = _resolve_path(args.output_dir)
    _validate_checkpoint(checkpoint)
    reward_file = ROOT / "student" / "my_reward_v21.py"
    if not reward_file.exists():
        raise FileNotFoundError(f"Required reward module not found: {reward_file}")
    if args.check_only:
        print("[v21-eval] Static path check PASS")
        print(f"  checkpoint: {checkpoint}")
        print(f"  reward:     {reward_file}")
        print(f"  output:     {output_dir}")
        print("  training updates: 0")
        return 0

    _prepare_output_dir(output_dir, args.overwrite)

    deps = _load_runtime_dependencies()
    env_creator = _make_env_creator(deps)
    ray_module = deps["ray"]
    algorithm = None

    metadata = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "checkpoint": str(checkpoint),
        "output_dir": str(output_dir),
        "episodes_per_scenario": args.episodes_per_scenario,
        "total_episodes": args.episodes_per_scenario * 3,
        "max_steps": args.max_steps,
        "base_seed": args.base_seed,
        "exact_repeat": args.exact_repeat,
        "distance_m": 900.0,
        "ownship_speed_mps": OWNSHIP_SPEED_MPS,
        "target_speed_mps": TARGET_SPEED_MPS,
        "training_updates": 0,
        "diagnostic_metrics": {
            "firing_cone_definition": "abs(ATA) <= wez.angle_deg / 2",
            "firing_cone_pass_threshold": None,
        },
        "success_criteria": {
            "overall_crash_rate_max": 0.0,
            "overall_damage_rate_min": 0.75,
            "per_scenario_damage_rate_min": 0.60,
            "mean_target_damage_min": 0.20,
            "mean_ata_deg_max": 5.0,
            "mean_abs_roll_deg_max": 15.0,
            "max_abs_roll_deg_strictly_below": 55.0,
            "mean_altitude_drop_m_max": 150.0,
        },
    }
    (output_dir / "evaluation_config.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    try:
        _start_ray(ray_module)
        base_env_config = _base_env_config(env_creator)
        build_env_config = deepcopy(base_env_config)
        build_env_config["initial_scenario"] = {
            "mode": "scenario_pool",
            "scenarios": [_scenario_definitions(args.exact_repeat)[0]],
        }
        algorithm = _build_algorithm(
            deps,
            env_creator,
            build_env_config,
            checkpoint,
        )
        actor = deps["PolicyActorView"](algorithm)
        firing_half_angle_deg = float(base_env_config["wez"]["angle_deg"]) / 2.0
        print(
            f"[v21-eval] Actual firing half-angle: {firing_half_angle_deg:.3f} deg"
        )

        result_rows: list[dict[str, Any]] = []
        scenarios = _scenario_definitions(args.exact_repeat)
        for scenario_index, scenario in enumerate(scenarios):
            print(f"\n[v21-eval] Scenario: {scenario['name']}")
            for episode in range(args.episodes_per_scenario):
                seed = args.base_seed + scenario_index * 1000 + episode
                replay_dir = output_dir / "replays" / scenario["name"] / f"episode_{episode:02d}"
                row = _run_episode(
                    env_creator=env_creator,
                    actor=actor,
                    base_env_config=base_env_config,
                    scenario=scenario,
                    episode=episode,
                    seed=seed,
                    max_steps=args.max_steps,
                    replay_dir=replay_dir,
                    firing_half_angle_deg=firing_half_angle_deg,
                )
                result_rows.append(row)
                print(
                    f"  ep={episode:02d} seed={seed} "
                    f"damage={_format_percent(float(row['target_damage']))} "
                    f"ATA={float(row['mean_ata_deg']):.2f}deg "
                    f"fire_cone={_format_percent(float(row['ata_within_firing_cone_rate']))} "
                    f"roll={float(row['mean_abs_roll_deg']):.2f}deg "
                    f"alt_drop={float(row['altitude_drop_m']):.1f}m "
                    f"crash={row['crashed']}"
                )

        _write_csv(output_dir / "evaluation_results.csv", RESULT_FIELDS, result_rows)

        summary_rows = [_aggregate("overall", result_rows)]
        for scenario in scenarios:
            scenario_rows = [
                row for row in result_rows if row["scenario"] == scenario["name"]
            ]
            summary_rows.append(_aggregate(scenario["name"], scenario_rows))

        _write_csv(output_dir / "evaluation_summary.csv", SUMMARY_FIELDS, summary_rows)
        final_pass = bool(
            summary_rows[0]["pass_all"]
            and all(row["pass_all"] for row in summary_rows[1:])
        )
        summary_payload = {
            "metadata": metadata,
            "groups": summary_rows,
            "final_pass": final_pass,
            "next_action": (
                "Proceed to a controlled 925 m experiment from v21 iter_0001."
                if final_pass
                else "Do not proceed to 925 m; inspect failed criteria and re-evaluate."
            ),
        }
        (output_dir / "evaluation_summary.json").write_text(
            json.dumps(summary_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        _print_summary(summary_rows)
        print(f"\nResults: {output_dir / 'evaluation_results.csv'}")
        print(f"Summary: {output_dir / 'evaluation_summary.csv'}")
        print(f"JSON:    {output_dir / 'evaluation_summary.json'}")
        return 0
    finally:
        if algorithm is not None:
            try:
                algorithm.stop()
            except Exception:
                pass
        try:
            ray_module.shutdown()
        except Exception:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
