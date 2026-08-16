# -*- coding: utf-8 -*-
"""Second flight-safety attempt with early level-flight learning signals."""
from __future__ import annotations

from student.sac_flight_safety_reward import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    _clip,
    compute_reward as compute_base_reward,
)
from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "altitude_hold_target_m": 7000.0,
    "altitude_hold_range_m": 3000.0,
    "altitude_hold_scale": 0.05,
    "level_pitch_scale": 0.03,
    "level_roll_scale": 0.01,
}


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
    total, components = compute_base_reward(
        ownship_state,
        target_state,
        ownship_damage,
        target_damage,
        geo_info,
        wez_config,
        reward_config,
        terminated,
        truncated,
        end_condition,
    )

    altitude = float(ownship_state[StateIndex.ALT])
    pitch_deg = float(ownship_state[StateIndex.PITCH])
    roll_deg = float(ownship_state[StateIndex.ROLL])

    altitude_error = abs(
        altitude - float(reward_config["altitude_hold_target_m"])
    )
    altitude_factor = _clip(
        altitude_error / max(float(reward_config["altitude_hold_range_m"]), 1.0)
    )
    altitude_hold = -float(reward_config["altitude_hold_scale"]) * altitude_factor**2
    level_pitch = -float(reward_config["level_pitch_scale"]) * _clip(
        abs(pitch_deg) / 45.0
    ) ** 2
    level_roll = -float(reward_config["level_roll_scale"]) * _clip(
        abs(roll_deg) / 60.0
    ) ** 2

    components["altitude_hold"] = altitude_hold
    components["level_pitch"] = level_pitch
    components["level_roll"] = level_roll
    return float(total + altitude_hold + level_pitch + level_roll), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
