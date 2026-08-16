from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
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
    BundleMappedDirectSACActionProvider,
    DirectSACMLPActionProvider,
    ObservationFeedbackActionProvider,
    NegativeTargetBankCanonicalActionProvider,
    CanonicalTargetBankActionRestoreProvider,
    RLActionProvider,
    TargetBankActionScaleProvider,
)
from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.hybrid_action_provider import (
    GuidanceResidualActionProvider,
    HeadingThenActionProvider,
    GeometryThenActionProvider,
    ProximitySpeedMatchActionProvider,
    PursuitAutopilotActionProvider,
    PursuitGuidanceActionProvider,
    TargetFormationAutopilotActionProvider,
    TargetTrailAutopilotActionProvider,
    WezThenActionProvider,
    build_pursuit_guidance_provider,
)
from dogfight.ai.rllib_utils import (
    build_algorithm_from_bundle,
    build_sac_lstm_module_from_bundle,
)
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
    parser.add_argument(
        "--fresh-process-per-seed",
        action="store_true",
        help=(
            "Evaluate each seed in a new Python process so native simulator "
            "state cannot leak across episodes."
        ),
    )
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
        "--ownship-speed-advantage-mps",
        type=float,
        help=(
            "Add a fixed ownship-only initial speed advantage to ref_old_random "
            "scenarios."
        ),
    )
    parser.add_argument(
        "--direct-sac-mlp",
        action="store_true",
        help="Run a standard SAC MLP actor directly without starting Ray.",
    )
    parser.add_argument(
        "--direct-sac-lstm",
        action="store_true",
        help="Run a recurrent SAC RLModule directly without starting Ray.",
    )
    parser.add_argument(
        "--canonicalize-negative-target-bank",
        action="store_true",
        help="Mirror negative target-bank geometry/action around the actor.",
    )
    parser.add_argument("--recovery-bundle-dir")
    parser.add_argument("--recovery-trigger-altitude-m", type=float, default=3000.0)
    parser.add_argument("--recovery-release-altitude-m", type=float, default=4500.0)
    parser.add_argument(
        "--provider-primary-bias",
        default="0,0,0,0",
        help="Four-value RL-action bias applied only to the primary policy.",
    )
    parser.add_argument("--mapped-recovery-bundle-dir")
    parser.add_argument(
        "--mapped-recovery-trigger-altitude-m", type=float, default=3000.0
    )
    parser.add_argument(
        "--mapped-recovery-release-altitude-m", type=float, default=4500.0
    )
    parser.add_argument(
        "--mapped-recovery-bias",
        default="0,0,0,0",
        help=(
            "Four-value policy-space bias applied before mapping the independent "
            "recovery bundle to simulator controls."
        ),
    )
    parser.add_argument(
        "--observation-feedback-index",
        type=int,
        help="Observation element used for an optional linear action correction.",
    )
    parser.add_argument(
        "--observation-feedback-gains",
        default="0,0,0,0",
        help="Four action gains multiplied by --observation-feedback-index.",
    )
    parser.add_argument("--switch-to-bt-after-wez-steps", type=int)
    parser.add_argument("--ownship-bt-dll", default="AIP_BASE.dll")
    parser.add_argument(
        "--ownship-bt-only",
        action="store_true",
        help="Use the native ownship behavior tree for the entire episode.",
    )
    parser.add_argument(
        "--pursuit-guidance-only",
        action="store_true",
        help="Use the deterministic flight-safe pursuit curriculum teacher.",
    )
    parser.add_argument("--guidance-heading-to-bank", type=float, default=1.2)
    parser.add_argument("--guidance-max-bank-deg", type=float, default=60.0)
    parser.add_argument("--guidance-roll-gain", type=float, default=1.0 / 45.0)
    parser.add_argument("--guidance-pitch-gain", type=float, default=1.0 / 20.0)
    parser.add_argument("--guidance-capture-pitch-gain", type=float)
    parser.add_argument("--guidance-pitch-command-sign", type=float, default=-1.0)
    parser.add_argument("--guidance-rudder-gain", type=float, default=0.15)
    parser.add_argument("--guidance-lead-time-s", type=float, default=0.0)
    parser.add_argument("--guidance-altitude-guard-m", type=float, default=3000.0)
    parser.add_argument("--guidance-recovery-pitch-deg", type=float, default=15.0)
    parser.add_argument("--guidance-brake-range-m", type=float, default=0.0)
    parser.add_argument("--guidance-near-throttle-action", type=float, default=-1.0)
    parser.add_argument("--guidance-brake-heading-error-deg", type=float, default=0.0)
    parser.add_argument("--guidance-capture-range-m", type=float, default=0.0)
    parser.add_argument("--guidance-capture-heading-error-deg", type=float, default=0.0)
    parser.add_argument("--guidance-capture-bank-scale", type=float, default=1.0)
    parser.add_argument("--guidance-capture-rudder-gain", type=float)
    parser.add_argument("--guidance-capture-heading-rate-to-rudder", type=float, default=0.0)
    parser.add_argument("--guidance-wez-hold-target-heading-blend", type=float, default=0.0)
    parser.add_argument("--guidance-wez-hold-target-pitch-blend", type=float, default=0.0)
    parser.add_argument("--guidance-capture-lead-time-s", type=float)
    parser.add_argument("--guidance-heading-rate-to-bank", type=float, default=0.0)
    parser.add_argument("--guidance-capture-heading-bias-deg", type=float, default=0.0)
    parser.add_argument("--guidance-capture-bias-target-roll-sign", action="store_true")
    parser.add_argument("--guidance-trail-distance-m", type=float, default=0.0)
    parser.add_argument(
        "--guidance-capture-target-bank-blend", type=float, default=0.0
    )
    parser.add_argument("--guidance-capture-heading-rate-to-bank", type=float)
    parser.add_argument("--guidance-capture-roll-gain", type=float)
    parser.add_argument("--guidance-capture-speed-gain", type=float, default=0.0)
    parser.add_argument(
        "--guidance-target-bank-feedforward-range-m", type=float, default=0.0
    )
    parser.add_argument(
        "--guidance-target-bank-feedforward-blend", type=float, default=0.0
    )
    parser.add_argument(
        "--guidance-capture-roll-delta-damping", type=float, default=0.0
    )
    parser.add_argument("--guidance-latch-capture-mode", action="store_true")
    parser.add_argument("--guidance-wez-hold-latch-ata-deg", type=float, default=0.0)
    parser.add_argument(
        "--guidance-wez-hold-target-bank-blend", type=float, default=0.0
    )
    parser.add_argument("--guidance-wez-hold-roll-gain", type=float)
    parser.add_argument(
        "--trace-interval-steps",
        type=int,
        default=0,
        help="Write per-interval guidance geometry diagnostics when positive.",
    )
    parser.add_argument("--pursuit-autopilot-only", action="store_true")
    parser.add_argument("--pursuit-autopilot-lead-time-s", type=float, default=3.0)
    parser.add_argument("--pursuit-autopilot-speed-mps", type=float, default=300.0)
    parser.add_argument("--switch-to-bundle-after-wez-steps", type=int)
    parser.add_argument("--secondary-bundle-dir")
    parser.add_argument("--switch-to-bundle-distance-m", type=float)
    parser.add_argument("--switch-to-bundle-ata-deg", type=float)
    parser.add_argument("--switch-to-formation-autopilot-distance-m", type=float)
    parser.add_argument("--switch-to-formation-autopilot-ata-deg", type=float)
    parser.add_argument("--switch-to-hold-guidance-distance-m", type=float)
    parser.add_argument("--switch-to-hold-guidance-ata-deg", type=float)
    parser.add_argument("--hold-guidance-target-bank-blend", type=float, default=0.8)
    parser.add_argument("--hold-guidance-roll-gain", type=float, default=0.02)
    parser.add_argument("--hold-guidance-roll-delta-damping", type=float, default=0.0)
    parser.add_argument("--hold-guidance-speed-gain", type=float, default=0.02)
    parser.add_argument("--hold-guidance-lead-time-s", type=float, default=0.0)
    parser.add_argument("--hold-guidance-trail-distance-m", type=float, default=0.0)
    parser.add_argument("--hold-guidance-heading-rate-to-bank", type=float, default=0.0)
    parser.add_argument("--hold-guidance-target-bank-multiplier", type=float, default=1.0)
    parser.add_argument("--hold-guidance-wez-bank-multiplier", type=float)
    parser.add_argument("--hold-guidance-wez-latch-ata-deg", type=float, default=0.0)
    parser.add_argument("--hold-guidance-target-heading-blend", type=float, default=0.0)
    parser.add_argument("--hold-guidance-rudder-gain", type=float, default=0.0)
    parser.add_argument("--hold-align-then-capture-heading-deg", type=float)
    parser.add_argument("--hold-align-speed-offset-mps", type=float, default=0.0)
    parser.add_argument("--hold-align-trail-distance-m", type=float, default=0.0)
    parser.add_argument("--hold-align-heading-to-bank", type=float, default=3.0)
    parser.add_argument("--hold-align-bank-gain-baseline-deg", type=float, default=0.0)
    parser.add_argument("--hold-align-bank-to-heading-gain", type=float, default=0.0)
    parser.add_argument("--hold-align-bank-damping-baseline-deg", type=float, default=0.0)
    parser.add_argument("--hold-align-bank-to-roll-damping", type=float, default=0.0)
    parser.add_argument("--hold-align-capture-ata-deg", type=float, default=0.0)
    parser.add_argument("--hold-capture-then-align-ata-deg", type=float)
    parser.add_argument("--hold-align-target-pitch-blend", type=float, default=0.0)
    parser.add_argument("--hold-align-target-heading-blend", type=float, default=1.0)
    parser.add_argument("--hold-align-near-ata-deg", type=float, default=0.0)
    parser.add_argument("--hold-align-near-heading-blend", type=float)
    parser.add_argument("--hold-align-pitch-gain", type=float, default=0.30)
    parser.add_argument("--hold-align-rudder-gain", type=float, default=0.0)
    parser.add_argument("--hold-formation-distance-m", type=float)
    parser.add_argument("--hold-formation-ata-deg", type=float)
    parser.add_argument("--hold-trail-autopilot-distance-m", type=float)
    parser.add_argument("--hold-trail-autopilot-ata-deg", type=float)
    parser.add_argument("--hold-trail-point-distance-m", type=float, default=600.0)
    parser.add_argument("--hold-trail-speed-offset-mps", type=float, default=0.0)
    parser.add_argument("--speed-match-start-distance-m", type=float)
    parser.add_argument("--speed-match-gain", type=float, default=0.02)
    parser.add_argument(
        "--secondary-bundle-bias",
        default="0,0,0,0",
        help="Four-value policy-space bias for the WEZ-latched secondary bundle.",
    )
    parser.add_argument("--secondary-target-bank-reference-deg", type=float)
    parser.add_argument(
        "--secondary-target-bank-roll-exponent", type=float, default=1.0
    )
    parser.add_argument(
        "--secondary-target-bank-rudder-exponent", type=float, default=1.0
    )
    parser.add_argument("--disable-ownship-randomization", action="store_true")
    parser.add_argument("--target-loiter-bank-deg", type=float)
    parser.add_argument("--target-loiter-pitch-deg", type=float, default=0.0)
    parser.add_argument("--target-control-surface-scale", type=float)
    parser.add_argument("--residual-surface-scale", type=float)
    parser.add_argument("--residual-throttle-scale", type=float, default=0.05)
    parser.add_argument(
        "--guidance-residual-scale",
        help="Optional four-value override for saved pursuit-guidance residual scale.",
    )
    parser.add_argument(
        "--disable-saved-guidance-residual",
        action="store_true",
        help="Do not wrap the selected provider in the bundle's saved guidance residual.",
    )
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
    delta = np.asarray(target[:3], dtype=np.float64) - np.asarray(
        ownship[:3], dtype=np.float64
    )
    los_heading = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
    heading_error = (
        los_heading - float(ownship[StateIndex.YAW]) + 180.0
    ) % 360.0 - 180.0
    return {
        "initial_distance_m": float(geo._get_distance(ownship, target)),
        "initial_ata_deg": abs(
            float(geo._get_antenna_train_angle(ownship, target, True))
        ),
        "initial_aa_deg": abs(float(geo._get_aspect_angle(ownship, target, True))),
        # Geodetic ALT/KCAS are populated after the first native step. NED D and
        # the configured speed are valid immediately after reset.
        "initial_altitude_m": -float(ownship[StateIndex.D]),
        "initial_speed_mps": float(env.info.get(
            "initial_ownship_speed_mps", env.config["ownship"][6]
        )),
        "initial_target_speed_mps": float(env.info.get(
            "initial_target_speed_mps", env.config["target"][6]
        )),
        "initial_heading_deg": float(ownship[StateIndex.YAW]),
        "initial_n_m": float(ownship[StateIndex.N]),
        "initial_e_m": float(ownship[StateIndex.E]),
        "initial_d_m": float(ownship[StateIndex.D]),
        "initial_target_n_m": float(target[StateIndex.N]),
        "initial_target_e_m": float(target[StateIndex.E]),
        "initial_target_d_m": float(target[StateIndex.D]),
        "initial_los_heading_deg": los_heading,
        "initial_heading_error_deg": heading_error,
        "initial_ownship_roll_deg": float(ownship[StateIndex.ROLL]),
        "initial_target_roll_deg": float(target[StateIndex.ROLL]),
        "legacy_scenario_index": int(
            env.info.get("legacy_scenario_index", -1)
        ),
        "episode_target_mode": str(env.config.get("target_mode", "")),
        "target_loiter_bank_deg": float(
            env.info.get("target_loiter_bank_deg", 0.0)
        ),
    }


def ownship_in_wez(env) -> bool:
    distance = float(env._geo_info._get_distance(
        env._ownship_state, env._target_state
    ))
    ata = abs(float(env._geo_info._get_antenna_train_angle(
        env._ownship_state, env._target_state, False
    )))
    wez = env._wez
    return (
        float(wez["min_range_m"]) <= distance <= float(wez["max_range_m"])
        and ata <= float(wez["angle_deg"]) / 2.0
    )


def los_attitude_errors(env) -> tuple[float, float]:
    ownship = np.asarray(env._ownship_state, dtype=np.float64)
    target = np.asarray(env._target_state, dtype=np.float64)
    delta = target[:3] - ownship[:3]
    horizontal = max(float(np.hypot(delta[0], delta[1])), 1.0)
    desired_heading = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
    heading_error = (desired_heading - float(ownship[StateIndex.YAW]) + 180.0) % 360.0 - 180.0
    desired_pitch = math.degrees(math.atan2(-float(delta[2]), horizontal))
    pitch_error = (desired_pitch - float(ownship[StateIndex.PITCH]) + 180.0) % 360.0 - 180.0
    return heading_error, pitch_error


def run_episode(env, seed: int, trace_rows: list[dict] | None = None, trace_interval_steps: int = 0) -> dict:
    env.reset(seed=seed)
    initial = initial_geometry(env)
    started_in_wez = ownship_in_wez(env)
    was_in_wez = started_in_wez
    entered_wez_from_outside = False
    first_wez_entry_distance_m = None
    first_wez_entry_ata_deg = None
    first_wez_exit_distance_m = None
    first_wez_exit_ata_deg = None
    first_wez_entry_heading_error_deg = None
    first_wez_entry_pitch_error_deg = None
    first_wez_exit_heading_error_deg = None
    first_wez_exit_pitch_error_deg = None
    terminated = truncated = False
    min_altitude = float("inf")
    min_distance = float("inf")
    min_ata = float("inf")
    min_ata_inside_wez_range = float("inf")
    trajectory = hashlib.sha256()
    info = {}
    total_reward = 0.0

    while not (terminated or truncated):
        _, reward, terminated, truncated, info = env.step(
            np.zeros(4, dtype=np.float32)
        )
        total_reward += float(reward)
        is_in_wez = bool(env._in_wez)
        state = np.asarray(env._ownship_state, dtype=np.float64)
        altitude = float(state[StateIndex.ALT])
        if altitude > 0.0:
            min_altitude = min(min_altitude, altitude)
        distance = float(env._geo_info._get_distance(
            env._ownship_state, env._target_state
        ))
        min_distance = min(min_distance, distance)
        ata = abs(float(env._geo_info._get_antenna_train_angle(
            env._ownship_state, env._target_state, False
        )))
        if is_in_wez and not was_in_wez:
            entered_wez_from_outside = True
            if first_wez_entry_distance_m is None:
                first_wez_entry_distance_m = distance
                first_wez_entry_ata_deg = ata
                (
                    first_wez_entry_heading_error_deg,
                    first_wez_entry_pitch_error_deg,
                ) = los_attitude_errors(env)
        elif was_in_wez and not is_in_wez and first_wez_exit_distance_m is None:
            first_wez_exit_distance_m = distance
            first_wez_exit_ata_deg = ata
            (
                first_wez_exit_heading_error_deg,
                first_wez_exit_pitch_error_deg,
            ) = los_attitude_errors(env)
        was_in_wez = is_in_wez
        min_ata = min(min_ata, ata)
        if float(env._wez["min_range_m"]) <= distance <= float(
            env._wez["max_range_m"]
        ):
            min_ata_inside_wez_range = min(min_ata_inside_wez_range, ata)
        trajectory.update(state.tobytes())
        step_count = int(info.get("ep_step_count", 0))
        if (
            trace_rows is not None
            and trace_interval_steps > 0
            and step_count % trace_interval_steps == 0
        ):
            from student.sac_pursuit_reward_v3 import _body_velocity_to_ned

            target_state = np.asarray(env._target_state, dtype=np.float64)
            own_velocity = _body_velocity_to_ned(state)
            target_velocity = _body_velocity_to_ned(target_state)
            heading_error, pitch_error = los_attitude_errors(env)
            trace_rows.append({
                "seed": seed,
                "step": step_count,
                "time_s": step_count * float(env._step_ratio) / float(env._sim_hz),
                "distance_m": distance,
                "ata_deg": ata,
                "ownship_bank_deg": float(state[StateIndex.ROLL]),
                "target_bank_deg": float(target_state[StateIndex.ROLL]),
                "ownship_pitch_deg": float(state[StateIndex.PITCH]),
                "target_pitch_deg": float(target_state[StateIndex.PITCH]),
                "ownship_speed_mps": float(np.linalg.norm(own_velocity)),
                "target_speed_mps": float(np.linalg.norm(target_velocity)),
                "ownship_heading_deg": float(state[StateIndex.YAW]),
                "target_heading_deg": float(target_state[StateIndex.YAW]),
                "los_heading_error_deg": heading_error,
                "los_pitch_error_deg": pitch_error,
                "altitude_m": altitude,
                "target_altitude_m": float(target_state[StateIndex.ALT]),
                "ownship_n_m": float(state[StateIndex.N]),
                "ownship_e_m": float(state[StateIndex.E]),
                "ownship_d_m": float(state[StateIndex.D]),
                "target_n_m": float(target_state[StateIndex.N]),
                "target_e_m": float(target_state[StateIndex.E]),
                "target_d_m": float(target_state[StateIndex.D]),
                "ownship_u_mps": float(state[StateIndex.U]),
                "ownship_v_mps": float(state[StateIndex.V]),
                "ownship_w_mps": float(state[StateIndex.W]),
                "target_u_mps": float(target_state[StateIndex.U]),
                "target_v_mps": float(target_state[StateIndex.V]),
                "target_w_mps": float(target_state[StateIndex.W]),
            })

    steps = int(info.get("ep_step_count", 0))
    max_consecutive = int(info.get("ep_max_consecutive_wez_steps", 0))
    step_seconds = float(env._step_ratio) / float(env._sim_hz)
    final_distance = float(env._geo_info._get_distance(
        env._ownship_state, env._target_state
    ))
    final_ata = abs(float(env._geo_info._get_antenna_train_angle(
        env._ownship_state, env._target_state, False
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
        "min_ata_deg": min_ata if np.isfinite(min_ata) else 180.0,
        "min_ata_inside_wez_range_deg": (
            min_ata_inside_wez_range
            if np.isfinite(min_ata_inside_wez_range)
            else 180.0
        ),
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
        "first_wez_entry_distance_m": first_wez_entry_distance_m,
        "first_wez_entry_ata_deg": first_wez_entry_ata_deg,
        "first_wez_exit_distance_m": first_wez_exit_distance_m,
        "first_wez_exit_ata_deg": first_wez_exit_ata_deg,
        "first_wez_entry_heading_error_deg": first_wez_entry_heading_error_deg,
        "first_wez_entry_pitch_error_deg": first_wez_entry_pitch_error_deg,
        "first_wez_exit_heading_error_deg": first_wez_exit_heading_error_deg,
        "first_wez_exit_pitch_error_deg": first_wez_exit_pitch_error_deg,
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
        "best_min_ata_deg": min(row.get("min_ata_deg", 180.0) for row in rows),
        "best_min_ata_inside_wez_range_deg": min(
            row.get("min_ata_inside_wez_range_deg", 180.0) for row in rows
        ),
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

    if args.fresh_process_per_seed and len(seeds) > 1:
        child_argv = [
            item for item in sys.argv[1:]
            if item != "--fresh-process-per-seed"
        ]

        def replace_option(argv: list[str], name: str, value: str) -> list[str]:
            result = list(argv)
            index = result.index(name)
            result[index + 1] = value
            return result

        rows = []
        target_mode = None
        for seed in seeds:
            seed_output = output_dir / f"seed_{seed}"
            current = replace_option(child_argv, "--seeds", str(seed))
            current = replace_option(current, "--output-dir", str(seed_output))
            subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), *current],
                cwd=str(ROOT),
                check=True,
            )
            child = json.loads(
                (seed_output / "summary.json").read_text(encoding="utf-8")
            )
            target_mode = child["target_mode"]
            rows.extend(child["episodes"])

        csv_path = output_dir / "episodes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        result = {
            "bundle_dir": str(bundle_dir),
            "target_mode": target_mode,
            "seeds": seeds,
            "fresh_process_per_seed": True,
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
        return

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
    if args.ownship_speed_advantage_mps is not None:
        scenario = dict(env_config.get("initial_scenario", {}))
        randomization = dict(scenario.get("legacy_randomization", {}))
        randomization["ownship_speed_advantage_mps"] = (
            args.ownship_speed_advantage_mps
        )
        scenario["legacy_randomization"] = randomization
        env_config["initial_scenario"] = scenario
    env_config["target_behavior_dll"] = args.target_behavior_dll

    reward_fn = None
    reward_module = str(env_config.get("reward_module", "")).strip()
    if reward_module:
        reward_fn, defaults = load_reward_hook(reward_module)
        env_config["reward"] = {**defaults, **dict(env_config.get("reward", {}))}

    if args.pursuit_autopilot_only:
        provider = PursuitAutopilotActionProvider(
            lead_time_s=args.pursuit_autopilot_lead_time_s,
            speed_mps=args.pursuit_autopilot_speed_mps,
        )
    elif args.pursuit_guidance_only:
        provider = PursuitGuidanceActionProvider(
            max_bank_deg=args.guidance_max_bank_deg,
            heading_to_bank=args.guidance_heading_to_bank,
            roll_gain=args.guidance_roll_gain,
            pitch_gain=args.guidance_pitch_gain,
            capture_pitch_gain=args.guidance_capture_pitch_gain,
            pitch_command_sign=args.guidance_pitch_command_sign,
            rudder_gain=args.guidance_rudder_gain,
            lead_time_s=args.guidance_lead_time_s,
            altitude_guard_m=args.guidance_altitude_guard_m,
            recovery_pitch_deg=args.guidance_recovery_pitch_deg,
            brake_range_m=args.guidance_brake_range_m,
            near_throttle_action=args.guidance_near_throttle_action,
            brake_heading_error_deg=args.guidance_brake_heading_error_deg,
            capture_range_m=args.guidance_capture_range_m,
            capture_heading_error_deg=args.guidance_capture_heading_error_deg,
            capture_bank_scale=args.guidance_capture_bank_scale,
            capture_rudder_gain=args.guidance_capture_rudder_gain,
            capture_heading_rate_to_rudder=args.guidance_capture_heading_rate_to_rudder,
            wez_hold_target_heading_blend=args.guidance_wez_hold_target_heading_blend,
            wez_hold_target_pitch_blend=args.guidance_wez_hold_target_pitch_blend,
            capture_lead_time_s=args.guidance_capture_lead_time_s,
            heading_rate_to_bank=args.guidance_heading_rate_to_bank,
            capture_heading_bias_deg=args.guidance_capture_heading_bias_deg,
            capture_bias_target_roll_sign=args.guidance_capture_bias_target_roll_sign,
            trail_distance_m=args.guidance_trail_distance_m,
            capture_target_bank_blend=args.guidance_capture_target_bank_blend,
            capture_heading_rate_to_bank=args.guidance_capture_heading_rate_to_bank,
            capture_roll_gain=args.guidance_capture_roll_gain,
            capture_speed_gain=args.guidance_capture_speed_gain,
            target_bank_feedforward_range_m=(
                args.guidance_target_bank_feedforward_range_m
            ),
            target_bank_feedforward_blend=(
                args.guidance_target_bank_feedforward_blend
            ),
            capture_roll_delta_damping=(
                args.guidance_capture_roll_delta_damping
            ),
            latch_capture_mode=args.guidance_latch_capture_mode,
            wez_hold_latch_ata_deg=args.guidance_wez_hold_latch_ata_deg,
            wez_hold_target_bank_blend=args.guidance_wez_hold_target_bank_blend,
            wez_hold_roll_gain=args.guidance_wez_hold_roll_gain,
        )
    elif args.ownship_bt_only:
        provider = BTActionProvider(dll_name=args.ownship_bt_dll)
    elif args.direct_sac_mlp:
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
    elif args.direct_sac_lstm:
        provider = RLActionProvider(
            bundle_dir=str(bundle_dir),
            algorithm_factory=build_sac_lstm_module_from_bundle,
            explore=False,
        )
    else:
        provider = RLActionProvider(
            bundle_dir=str(bundle_dir),
            algorithm_factory=build_algorithm_from_bundle,
            explore=False,
        )
    if bool(env_config.get("canonicalize_negative_target_bank", False)):
        provider = CanonicalTargetBankActionRestoreProvider(provider)
    elif args.canonicalize_negative_target_bank:
        provider = NegativeTargetBankCanonicalActionProvider(provider)
    guidance_residual = env_config.get("pursuit_guidance_residual", {})
    if (
        bool(guidance_residual.get("enabled", False))
        and not args.disable_saved_guidance_residual
    ):
        residual_scale = guidance_residual.get(
            "residual_scale", [0.25, 0.25, 0.25, 0.10]
        )
        if args.guidance_residual_scale:
            residual_scale = [
                float(part.strip())
                for part in args.guidance_residual_scale.split(",")
            ]
            if len(residual_scale) != 4:
                raise ValueError("--guidance-residual-scale must contain four values")
        provider = GuidanceResidualActionProvider(
            provider,
            build_pursuit_guidance_provider(guidance_residual),
            residual_scale=residual_scale,
            near_residual_scale=guidance_residual.get("near_residual_scale"),
            near_distance_m=guidance_residual.get("near_distance_m", 0.0),
            near_ata_deg=guidance_residual.get("near_ata_deg", 180.0),
            suppress_opposing_roll_heading_error_deg=guidance_residual.get(
                "suppress_opposing_roll_heading_error_deg", 0.0
            ),
            opposing_roll_guidance_scale=guidance_residual.get(
                "opposing_roll_guidance_scale", 1.0
            ),
        )
    if args.observation_feedback_index is not None:
        feedback_gains = np.asarray([
            float(part.strip())
            for part in args.observation_feedback_gains.split(",")
        ], dtype=np.float32)
        provider = ObservationFeedbackActionProvider(
            provider,
            observation_index=args.observation_feedback_index,
            action_gains=feedback_gains,
        )
    if args.speed_match_start_distance_m is not None:
        provider = ProximitySpeedMatchActionProvider(
            provider,
            start_distance_m=args.speed_match_start_distance_m,
            speed_gain=args.speed_match_gain,
        )
    if args.mapped_recovery_bundle_dir:
        mapped_recovery_bias = np.asarray([
            float(part.strip())
            for part in args.mapped_recovery_bias.split(",")
        ], dtype=np.float32)
        provider = AltitudeRecoveryActionProvider(
            provider,
            BundleMappedDirectSACActionProvider(
                str(Path(args.mapped_recovery_bundle_dir).resolve()),
                policy_bias=mapped_recovery_bias,
            ),
            trigger_altitude_m=args.mapped_recovery_trigger_altitude_m,
            release_altitude_m=args.mapped_recovery_release_altitude_m,
        )
    if args.switch_to_bt_after_wez_steps is not None:
        observation_mode = str(env_config.get("observation_mode", ""))
        wez_index = 15 if observation_mode in {"tactical17", "tactical21", "tactical25"} else 14
        provider = WezThenActionProvider(
            provider,
            BTActionProvider(dll_name=args.ownship_bt_dll),
            wez_observation_index=wez_index,
            required_steps=args.switch_to_bt_after_wez_steps,
        )

    def build_secondary_bundle_provider():
        secondary_bundle_bias = np.asarray([
            float(part.strip())
            for part in args.secondary_bundle_bias.split(",")
        ], dtype=np.float32)
        secondary = BundleMappedDirectSACActionProvider(
            args.secondary_bundle_dir,
            policy_bias=secondary_bundle_bias,
        )
        if args.secondary_target_bank_reference_deg is not None:
            secondary = TargetBankActionScaleProvider(
                secondary,
                reference_bank_deg=args.secondary_target_bank_reference_deg,
                roll_exponent=args.secondary_target_bank_roll_exponent,
                rudder_exponent=args.secondary_target_bank_rudder_exponent,
            )
        return secondary

    if args.switch_to_bundle_after_wez_steps is not None:
        if not args.secondary_bundle_dir:
            raise ValueError(
                "--secondary-bundle-dir is required with "
                "--switch-to-bundle-after-wez-steps"
            )
        observation_mode = str(env_config.get("observation_mode", ""))
        wez_index = 15 if observation_mode in {"tactical17", "tactical21", "tactical25"} else 14
        provider = WezThenActionProvider(
            provider,
            build_secondary_bundle_provider(),
            wez_observation_index=wez_index,
            required_steps=args.switch_to_bundle_after_wez_steps,
        )
    if args.switch_to_bundle_distance_m is not None:
        if not args.secondary_bundle_dir or args.switch_to_bundle_ata_deg is None:
            raise ValueError(
                "--secondary-bundle-dir and --switch-to-bundle-ata-deg are "
                "required with --switch-to-bundle-distance-m"
            )
        provider = GeometryThenActionProvider(
            provider,
            build_secondary_bundle_provider(),
            max_distance_m=args.switch_to_bundle_distance_m,
            max_ata_deg=args.switch_to_bundle_ata_deg,
        )
    if args.switch_to_formation_autopilot_distance_m is not None:
        if args.switch_to_formation_autopilot_ata_deg is None:
            raise ValueError(
                "--switch-to-formation-autopilot-ata-deg is required with "
                "--switch-to-formation-autopilot-distance-m"
            )
        provider = GeometryThenActionProvider(
            provider,
            TargetFormationAutopilotActionProvider(),
            max_distance_m=args.switch_to_formation_autopilot_distance_m,
            max_ata_deg=args.switch_to_formation_autopilot_ata_deg,
        )
    if args.switch_to_hold_guidance_distance_m is not None:
        if args.switch_to_hold_guidance_ata_deg is None:
            raise ValueError(
                "--switch-to-hold-guidance-ata-deg is required with "
                "--switch-to-hold-guidance-distance-m"
            )
        capture_provider = PursuitGuidanceActionProvider(
                max_bank_deg=80.0,
                heading_to_bank=2.5,
                roll_gain=args.hold_guidance_roll_gain,
                pitch_gain=0.30,
                capture_pitch_gain=0.30,
                pitch_command_sign=-1.0,
                rudder_gain=args.hold_guidance_rudder_gain,
                capture_rudder_gain=args.hold_guidance_rudder_gain,
                lead_time_s=args.hold_guidance_lead_time_s,
                trail_distance_m=args.hold_guidance_trail_distance_m,
                capture_range_m=100000.0,
                capture_heading_error_deg=180.0,
                capture_bank_scale=0.30,
                capture_heading_rate_to_bank=(
                    args.hold_guidance_heading_rate_to_bank
                ),
                capture_target_bank_blend=args.hold_guidance_target_bank_blend,
                capture_target_bank_multiplier=(
                    args.hold_guidance_target_bank_multiplier
                ),
                capture_target_heading_blend=(
                    args.hold_guidance_target_heading_blend
                ),
                wez_hold_latch_ata_deg=args.hold_guidance_wez_latch_ata_deg,
                wez_hold_target_bank_blend=args.hold_guidance_target_bank_blend,
                wez_hold_target_bank_multiplier=(
                    args.hold_guidance_wez_bank_multiplier
                ),
                capture_roll_gain=args.hold_guidance_roll_gain,
                capture_roll_delta_damping=(
                    args.hold_guidance_roll_delta_damping
                ),
                capture_speed_gain=args.hold_guidance_speed_gain,
            )
        hold_provider = capture_provider
        if args.hold_align_then_capture_heading_deg is not None:
            alignment_provider = PursuitGuidanceActionProvider(
                max_bank_deg=80.0,
                heading_to_bank=args.hold_align_heading_to_bank,
                target_bank_heading_gain_baseline_deg=(
                    args.hold_align_bank_gain_baseline_deg
                ),
                target_bank_to_heading_gain=(
                    args.hold_align_bank_to_heading_gain
                ),
                target_bank_roll_damping_baseline_deg=(
                    args.hold_align_bank_damping_baseline_deg
                ),
                target_bank_to_roll_damping=(
                    args.hold_align_bank_to_roll_damping
                ),
                roll_gain=args.hold_guidance_roll_gain,
                pitch_gain=args.hold_align_pitch_gain,
                capture_pitch_gain=args.hold_align_pitch_gain,
                pitch_command_sign=-1.0,
                rudder_gain=args.hold_align_rudder_gain,
                capture_rudder_gain=args.hold_align_rudder_gain,
                trail_distance_m=args.hold_align_trail_distance_m,
                lead_time_s=0.0,
                capture_range_m=100000.0,
                capture_heading_error_deg=180.0,
                capture_bank_scale=1.0,
                capture_target_heading_blend=(
                    args.hold_align_target_heading_blend
                ),
                capture_near_ata_deg=args.hold_align_near_ata_deg,
                capture_near_target_heading_blend=(
                    args.hold_align_near_heading_blend
                ),
                capture_target_pitch_blend=(
                    args.hold_align_target_pitch_blend
                ),
                capture_target_bank_blend=0.0,
                capture_roll_gain=args.hold_guidance_roll_gain,
                capture_roll_delta_damping=(
                    args.hold_guidance_roll_delta_damping
                ),
                capture_speed_gain=args.hold_guidance_speed_gain,
                capture_speed_offset_mps=args.hold_align_speed_offset_mps,
                wez_hold_latch_ata_deg=args.hold_guidance_wez_latch_ata_deg,
                wez_hold_target_bank_blend=(
                    args.hold_guidance_target_bank_blend
                ),
                wez_hold_target_bank_multiplier=(
                    args.hold_guidance_wez_bank_multiplier
                ),
                wez_hold_target_heading_blend=(
                    args.hold_guidance_target_heading_blend
                ),
                wez_hold_target_pitch_blend=(
                    args.hold_align_target_pitch_blend
                ),
            )
            hold_provider = HeadingThenActionProvider(
                alignment_provider,
                capture_provider,
                max_heading_difference_deg=(
                    args.hold_align_then_capture_heading_deg
                ),
                max_capture_ata_deg=args.hold_align_capture_ata_deg,
            )
            if args.hold_capture_then_align_ata_deg is not None:
                hold_provider = HeadingThenActionProvider(
                    capture_provider,
                    alignment_provider,
                    max_heading_difference_deg=-1.0,
                    max_capture_ata_deg=(
                        args.hold_capture_then_align_ata_deg
                    ),
                )
        if args.hold_formation_distance_m is not None:
            if args.hold_formation_ata_deg is None:
                raise ValueError(
                    "--hold-formation-ata-deg is required with "
                    "--hold-formation-distance-m"
                )
            hold_provider = GeometryThenActionProvider(
                hold_provider,
                TargetFormationAutopilotActionProvider(),
                max_distance_m=args.hold_formation_distance_m,
                max_ata_deg=args.hold_formation_ata_deg,
            )
        if args.hold_trail_autopilot_distance_m is not None:
            if args.hold_trail_autopilot_ata_deg is None:
                raise ValueError(
                    "--hold-trail-autopilot-ata-deg is required with "
                    "--hold-trail-autopilot-distance-m"
                )
            hold_provider = GeometryThenActionProvider(
                hold_provider,
                TargetTrailAutopilotActionProvider(
                    trail_distance_m=args.hold_trail_point_distance_m,
                    speed_offset_mps=args.hold_trail_speed_offset_mps,
                ),
                max_distance_m=args.hold_trail_autopilot_distance_m,
                max_ata_deg=args.hold_trail_autopilot_ata_deg,
            )
        provider = GeometryThenActionProvider(
            provider,
            hold_provider,
            max_distance_m=args.switch_to_hold_guidance_distance_m,
            max_ata_deg=args.switch_to_hold_guidance_ata_deg,
        )
    env = DogFightWrapper(
        env_config=env_config,
        ownship_action_provider=provider,
        reward_fn=reward_fn,
    )
    try:
        rows = []
        trace_rows = []
        for seed in seeds:
            row = run_episode(
                env,
                seed,
                trace_rows=trace_rows,
                trace_interval_steps=args.trace_interval_steps,
            )
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
    if trace_rows:
        trace_path = output_dir / "trace.csv"
        with trace_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(trace_rows[0]))
            writer.writeheader()
            writer.writerows(trace_rows)

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
