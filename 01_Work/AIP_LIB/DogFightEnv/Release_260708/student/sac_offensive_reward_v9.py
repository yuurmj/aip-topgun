# -*- coding: utf-8 -*-
"""Offensive dogfight reward that prefers one-sided firing geometry.

The previous reward could score highly by briefly entering the WEZ during a
mutual head-on pass.  This variant rewards damage and angular *advantage*,
adds a dense one-sided attack-window signal, and penalizes being threatened or
disengaging.  Flight-safety terms from v8 remain active.
"""
from __future__ import annotations

from student.sac_flight_safety_reward import _clip
from student.sac_cutoff_headon_reward_v8 import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    compute_reward as compute_base_reward,
)


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "attack_window_scale": 12.0,
    "attack_window_ata_deg": 30.0,
    "attack_window_range_m": 2500.0,
    "threat_penalty_scale": 12.0,
    "threat_ata_deg": 30.0,
    "disengagement_penalty_scale": 3.0,
    "disengagement_range_m": 5000.0,
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

    attack_angle = _clip(
        1.0 - own_ata / max(float(reward_config["attack_window_ata_deg"]), 1.0)
    )
    attack_range = _clip(
        1.0
        - distance / max(float(reward_config["attack_window_range_m"]), 1.0)
    )
    target_blind = _clip(target_ata / 90.0)
    attack_window = (
        float(reward_config["attack_window_scale"])
        * attack_angle
        * attack_range
        * target_blind
    )

    target_threat = _clip(
        1.0
        - target_ata / max(float(reward_config["threat_ata_deg"]), 1.0)
    )
    own_exposed = _clip(own_ata / 90.0)
    threat_range = _clip(
        1.0 - distance / max(float(wez_config.get("max_range_m", 914.4)), 1.0)
    )
    threat = -float(reward_config["threat_penalty_scale"]) * (
        target_threat * own_exposed * threat_range
    )

    disengagement = -float(
        reward_config["disengagement_penalty_scale"]
    ) * _clip(
        (distance - float(reward_config["disengagement_range_m"]))
        / max(float(reward_config["disengagement_range_m"]), 1.0)
    )

    components["attack_window"] = attack_window
    components["threat"] = threat
    components["disengagement"] = disengagement
    return float(total + attack_window + threat + disengagement), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
