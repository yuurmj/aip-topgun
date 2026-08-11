# -*- coding: utf-8 -*-
"""Moving-target alignment reward layered on the strict-WEZ safety reward."""
from __future__ import annotations

from student.sac_flight_safety_reward import _clip
from student.sac_wez_hold_reward_v4 import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    compute_reward as compute_base_reward,
)


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "moving_alignment_scale": 1.0,
    "moving_alignment_reference_deg": 5.0,
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
    reference = max(float(reward_config["moving_alignment_reference_deg"]), 0.1)
    # +scale at perfect alignment, 0 at half-reference, -scale at and beyond
    # the reference angle. This gives SAC a dense signal before strict WEZ.
    moving_alignment = float(reward_config["moving_alignment_scale"]) * (
        1.0 - 2.0 * _clip(ata / reference)
    )
    components["moving_alignment"] = moving_alignment
    return float(total + moving_alignment), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
