# -*- coding: utf-8 -*-
"""Moving-target reward with explicit close-range braking and floor penalty."""
from __future__ import annotations

from student.sac_flight_safety_reward import _clip
from student.sac_moving_target_reward_v5 import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    compute_reward as compute_base_reward,
)
from student.sac_pursuit_reward_v3 import _closure_rate_mps


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "close_range_brake_scale": 1.0,
    "inside_min_range_scale": 2.0,
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

    distance = float(geo_info._get_distance(ownship_state, target_state))
    min_range = max(float(wez_config.get("min_range_m", 152.4)), 1.0)
    max_range = max(float(wez_config.get("max_range_m", 914.4)), min_range + 1.0)
    closure = _closure_rate_mps(ownship_state, target_state)
    closure_reference = max(float(reward_config["closure_reference_mps"]), 1.0)

    # Once inside the WEZ maximum range, reduce the incentive to keep closing.
    # The penalty grows toward the inner boundary and is zero while opening.
    proximity = _clip((max_range - distance) / (max_range - min_range))
    positive_closure = _clip(max(closure, 0.0) / closure_reference)
    close_range_brake = -float(reward_config["close_range_brake_scale"]) * (
        proximity * positive_closure
    )

    # Crossing the inner WEZ boundary is a distinct failure mode from poor ATA.
    inside_min_range = -float(reward_config["inside_min_range_scale"]) * _clip(
        (min_range - distance) / min_range
    )

    components["close_range_brake"] = close_range_brake
    components["inside_min_range"] = inside_min_range
    return float(total + close_range_brake + inside_min_range), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
