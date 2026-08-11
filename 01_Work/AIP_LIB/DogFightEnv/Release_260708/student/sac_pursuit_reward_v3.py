# -*- coding: utf-8 -*-
"""Flight-safe pursuit reward with signed closure and geometry signals."""
from __future__ import annotations

import math
import numpy as np

from dogfight.sim.state_schema import StateIndex
from student.sac_flight_safety_reward import _clip
from student.sac_flight_safety_reward_v2 import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    compute_reward as compute_base_reward,
)


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "pursuit_scale": 0.0,
    "closure_scale": 0.30,
    "closure_reference_mps": 300.0,
    "alignment_scale": 0.15,
    "range_scale": 0.15,
    "range_reference_m": 8000.0,
}


def _signed_clip(value: float) -> float:
    return max(-1.0, min(1.0, float(value)))


def _body_velocity_to_ned(state) -> np.ndarray:
    """Rotate body-axis u/v/w velocity into the NED frame."""
    roll, pitch, yaw = (
        math.radians(float(state[StateIndex.ROLL])),
        math.radians(float(state[StateIndex.PITCH])),
        math.radians(float(state[StateIndex.YAW])),
    )
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation = np.asarray(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=np.float64,
    )
    body_velocity = np.asarray(
        [
            state[StateIndex.U],
            state[StateIndex.V],
            state[StateIndex.W],
        ],
        dtype=np.float64,
    )
    return rotation @ body_velocity


def _closure_rate_mps(ownship_state, target_state) -> float:
    relative_position = np.asarray(target_state[:3], dtype=np.float64) - np.asarray(
        ownship_state[:3], dtype=np.float64
    )
    distance = float(np.linalg.norm(relative_position))
    if distance <= 1.0e-6:
        return 0.0
    relative_velocity = _body_velocity_to_ned(target_state) - _body_velocity_to_ned(
        ownship_state
    )
    range_rate = float(np.dot(relative_position, relative_velocity) / distance)
    return -range_rate


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

    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(float(geo_info._get_antenna_train_angle(
        ownship_state, target_state, False
    )))
    warning = float(reward_config["altitude_warning_m"])
    floor = float(reward_config["altitude_floor_m"])
    altitude = float(ownship_state[StateIndex.ALT])
    risk = _clip((warning - altitude) / max(warning - floor, 1.0))
    safe_gate = 1.0 - risk

    closure = _closure_rate_mps(ownship_state, target_state)
    closure_reward = float(reward_config["closure_scale"]) * _signed_clip(
        closure / max(float(reward_config["closure_reference_mps"]), 1.0)
    ) * safe_gate
    alignment_reward = float(reward_config["alignment_scale"]) * _signed_clip(
        1.0 - ata / 90.0
    ) * safe_gate
    range_reward = float(reward_config["range_scale"]) * _signed_clip(
        1.0 - 2.0 * distance / max(float(reward_config["range_reference_m"]), 1.0)
    ) * safe_gate

    old_pursuit = float(components.get("pursuit", 0.0))
    pursuit = closure_reward + alignment_reward + range_reward
    components["pursuit"] = pursuit
    components["closure"] = closure_reward
    components["alignment"] = alignment_reward
    components["range"] = range_reward
    return float(total - old_pursuit + pursuit), components


__all__ = [
    "MY_REWARD_CONFIG",
    "_body_velocity_to_ned",
    "_closure_rate_mps",
    "compute_reward",
]
