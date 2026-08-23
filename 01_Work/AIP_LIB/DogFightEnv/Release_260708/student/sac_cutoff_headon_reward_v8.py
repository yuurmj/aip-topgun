# -*- coding: utf-8 -*-
"""Reward for the Viewer cutoff model's close head-on opening.

The earlier SAC reward strongly rewarded any ownship WEZ occupancy.  In a
head-on merge that also rewards a mutual firing solution, which produced the
observed near-equal damage followed by weak re-engagement.  This layer keeps
the established pursuit shaping but explicitly prefers damage advantage and
one-sided angular advantage.
"""
from __future__ import annotations

from student.sac_flight_safety_reward import _clip
from student.sac_wez_sync_reward_v7 import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    compute_reward as compute_base_reward,
)


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "target_damage_bonus_scale": 60.0,
    "ownship_damage_penalty_scale": 160.0,
    "mutual_damage_penalty": 20.0,
    "angular_advantage_scale": 4.0,
    "angular_advantage_reference_deg": 90.0,
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

    own_ata = abs(float(geo_info._get_antenna_train_angle(
        ownship_state, target_state, False
    )))
    target_ata = abs(float(geo_info._get_antenna_train_angle(
        target_state, ownship_state, False
    )))
    reference = max(
        float(reward_config["angular_advantage_reference_deg"]), 1.0
    )
    advantage = float(reward_config["angular_advantage_scale"]) * max(
        -1.0,
        min(1.0, (target_ata - own_ata) / reference),
    )
    target_bonus = float(
        reward_config["target_damage_bonus_scale"]
    ) * float(target_damage)
    ownship_penalty = -float(
        reward_config["ownship_damage_penalty_scale"]
    ) * float(ownship_damage)
    mutual_penalty = (
        -float(reward_config["mutual_damage_penalty"])
        if ownship_damage > 0.0 and target_damage > 0.0
        else 0.0
    )

    components["angular_advantage"] = advantage
    components["target_damage_bonus"] = target_bonus
    components["ownship_damage_penalty"] = ownship_penalty
    components["mutual_damage_penalty"] = mutual_penalty
    return float(
        total + advantage + target_bonus + ownship_penalty + mutual_penalty
    ), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
