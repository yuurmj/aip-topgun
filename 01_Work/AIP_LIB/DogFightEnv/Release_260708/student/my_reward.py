# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    # Existing v7 attack/reward settings
    "survival_bonus": 0.0,
    "damage_scale": 30.0,
    "pursuit_half_angle_deg": 60.0,
    "step_penalty": -0.01,
    "pursuit_scale": 0.08,
    "pursuit_angle_deg": 60.0,
    "pursuit_range_m": 6000.0,
    "attack_min_range_m": 500.0,
    "attack_max_range_m": 920.0,
    "attack_ata_deg": 15.0,
    "attack_position_bonus": 0.10,
    "wez_hold_bonus": 0.15,
    "target_damage_scale": 30.0,
    "ownship_damage_scale": 25.0,

    # v8 altitude stability settings
    # A small reward inside the desired altitude band.  It is deliberately
    # smaller than the maximum pursuit reward so that attack is still useful.
    "safe_altitude_min_m": 5000.0,
    "safe_altitude_max_m": 8000.0,
    "safe_altitude_bonus": 0.01,

    # Warning starts at 5000 m and grows linearly to the full penalty at
    # 3000 m.  Below 3000 m the per-step penalty is capped at 0.50.
    "low_altitude_warning_m": 5000.0,
    "low_altitude_m": 3000.0,
    "low_altitude_penalty": 0.50,

    # StateIndex has no vertical-speed entry in this release.  Pitch is the
    # earliest verified stateless signal available to compute_reward().
    # Negative pitch is nose-down in the supplied simulator and replay logs.
    "dive_pitch_start_deg": -10.0,
    "dive_pitch_full_deg": -45.0,
    "dive_pitch_penalty": 0.10,

    "bad_close_range_m": 300.0,
    "bad_close_ata_deg": 30.0,
    "bad_close_penalty": 0.20,

    # Existing terminal settings, except the requested stronger crash penalty
    "win_reward": 150.0,
    "loss_reward": -150.0,
    "crash_penalty": -200.0,
    "guard_fail_penalty": -50.0,
    "target_crash_reward": -100.0,
    "timeout_penalty": -50.0,
    "draw_reward": -10.0,
}


def _linear_factor(value: float, start: float, full: float) -> float:
    """Return a clipped 0..1 factor from start toward full."""
    span = full - start
    if span == 0.0:
        return 1.0 if value == full else 0.0
    return max(0.0, min(1.0, (value - start) / span))


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    """Compute the v8 reward while preserving the v7 component contract."""

    distance_m = float(geo_info._get_distance(ownship_state, target_state))
    ata_deg = abs(
        float(
            geo_info._get_antenna_train_angle(
                ownship_state,
                target_state,
                False,
            )
        )
    )

    step_reward = float(reward_config.get("step_penalty", -0.01))

    pursuit_angle_deg = float(reward_config.get("pursuit_angle_deg", 60.0))
    pursuit_range_m = float(reward_config.get("pursuit_range_m", 6000.0))
    ata_factor = (
        max(0.0, 1.0 - ata_deg / pursuit_angle_deg)
        if pursuit_angle_deg > 0.0
        else 0.0
    )
    range_factor = (
        max(0.0, 1.0 - distance_m / pursuit_range_m)
        if pursuit_range_m > 0.0
        else 0.0
    )
    pursuit_reward = (
        float(reward_config.get("pursuit_scale", 0.08))
        * ata_factor
        * range_factor
    )

    in_attack_position = (
        ata_deg <= float(reward_config.get("attack_ata_deg", 15.0))
        and float(reward_config.get("attack_min_range_m", 500.0))
        <= distance_m
        <= float(reward_config.get("attack_max_range_m", 920.0))
    )
    if in_attack_position:
        pursuit_reward += float(
            reward_config.get("attack_position_bonus", 0.10)
        )
    if float(target_damage) > 0.0:
        pursuit_reward += float(reward_config.get("wez_hold_bonus", 0.15))

    damage_reward = (
        float(reward_config.get("target_damage_scale", 30.0))
        * float(target_damage)
        - float(reward_config.get("ownship_damage_scale", 25.0))
        * float(ownship_damage)
    )

    altitude_m = float(ownship_state[StateIndex.ALT])
    pitch_deg = float(ownship_state[StateIndex.PITCH])
    safety_reward = 0.0

    safe_min_m = float(reward_config.get("safe_altitude_min_m", 5000.0))
    safe_max_m = float(reward_config.get("safe_altitude_max_m", 8000.0))
    if safe_min_m <= altitude_m <= safe_max_m:
        safety_reward += float(reward_config.get("safe_altitude_bonus", 0.01))

    warning_m = float(reward_config.get("low_altitude_warning_m", 5000.0))
    low_m = float(reward_config.get("low_altitude_m", 3000.0))
    if altitude_m < warning_m:
        if warning_m > low_m:
            low_factor = min(1.0, (warning_m - altitude_m) / (warning_m - low_m))
        else:
            low_factor = 1.0
        safety_reward -= (
            float(reward_config.get("low_altitude_penalty", 0.50))
            * max(0.0, low_factor)
        )

    dive_start_deg = float(reward_config.get("dive_pitch_start_deg", -10.0))
    dive_full_deg = float(reward_config.get("dive_pitch_full_deg", -45.0))
    if pitch_deg < dive_start_deg:
        dive_factor = _linear_factor(pitch_deg, dive_start_deg, dive_full_deg)
        safety_reward -= (
            float(reward_config.get("dive_pitch_penalty", 0.10))
            * dive_factor
        )

    close_range_penalty = 0.0
    if (
        distance_m < float(reward_config.get("bad_close_range_m", 300.0))
        and ata_deg > float(reward_config.get("bad_close_ata_deg", 30.0))
    ):
        close_range_penalty = -float(
            reward_config.get("bad_close_penalty", 0.20)
        )

    terminal_reward = 0.0
    if terminated:
        if end_condition == "target destroyed":
            terminal_reward = float(reward_config.get("win_reward", 150.0))
        elif end_condition == "target altitude below min":
            terminal_reward = float(
                reward_config.get("target_crash_reward", -100.0)
            )
        elif end_condition == "ownship destroyed":
            terminal_reward = float(reward_config.get("loss_reward", -150.0))
        elif end_condition in (
            "ownship altitude below min",
            "FDM Update Fail",
            "Ownship FDM output Fall",
        ):
            terminal_reward = float(reward_config.get("crash_penalty", -200.0))
        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(
                reward_config.get("guard_fail_penalty", -50.0)
            )
        else:
            terminal_reward = float(reward_config.get("draw_reward", -10.0))
    elif truncated:
        terminal_reward = float(reward_config.get("timeout_penalty", -50.0))

    # Keep exactly the existing four keys for training_log.csv compatibility.
    components = {
        "pursuit": float(pursuit_reward),
        "damage": float(damage_reward),
        "safety": float(safety_reward),
        "survival": 0.0,
    }

    total_reward = (
        step_reward
        + pursuit_reward
        + damage_reward
        + safety_reward
        + close_range_penalty
        + terminal_reward
    )
    return float(total_reward), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
