# -*- coding: utf-8 -*-
"""Strict-WEZ hold reward layered on flight-safe pursuit shaping."""
from __future__ import annotations

from student.sac_flight_safety_reward import _clip
from student.sac_pursuit_reward_v3 import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    compute_reward as compute_base_reward,
)


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "wez_hold_scale": 2.0,
    "fine_alignment_scale": 0.50,
    "fine_alignment_reference_deg": 10.0,
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
    ata = abs(float(geo_info._get_antenna_train_angle(
        ownship_state, target_state, False
    )))
    fine_alignment = -float(reward_config["fine_alignment_scale"]) * _clip(
        ata / max(float(reward_config["fine_alignment_reference_deg"]), 1.0)
    ) ** 2
    wez_hold = float(reward_config["wez_hold_scale"]) if target_damage > 0.0 else 0.0
    components["fine_alignment"] = fine_alignment
    components["wez_hold"] = wez_hold
    return float(total + fine_alignment + wez_hold), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
