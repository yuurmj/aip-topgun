# -*- coding: utf-8 -*-
"""Offensive reward with an explicit deep, sustained firing solution.

V9 improved approach geometry but often skimmed the outer WEZ for less than
one second.  This layer makes the useful firing region unambiguous: low ATA,
mid-WEZ range, and a target that cannot point back at the ownship.
"""
from __future__ import annotations

from student.sac_flight_safety_reward import _clip
from student.sac_offensive_reward_v9 import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    compute_reward as compute_base_reward,
)


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "fire_control_scale": 35.0,
    "fire_control_ata_deg": 8.0,
    "fire_control_center_range_m": 560.0,
    "fire_control_half_width_m": 355.0,
    "fire_control_target_blind_deg": 45.0,
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
    own_ata = abs(float(geo_info._get_antenna_train_angle(
        ownship_state, target_state, False
    )))
    target_ata = abs(float(geo_info._get_antenna_train_angle(
        target_state, ownship_state, False
    )))
    ata_gate = _clip(
        1.0 - own_ata / max(float(reward_config["fire_control_ata_deg"]), 0.1)
    )
    range_gate = _clip(
        1.0
        - abs(distance - float(reward_config["fire_control_center_range_m"]))
        / max(float(reward_config["fire_control_half_width_m"]), 1.0)
    )
    blind_gate = _clip(
        target_ata / max(float(reward_config["fire_control_target_blind_deg"]), 1.0)
    )
    fire_control = (
        float(reward_config["fire_control_scale"])
        * ata_gate
        * range_gate
        * blind_gate
    )
    components["fire_control"] = fire_control
    return float(total + fire_control), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
