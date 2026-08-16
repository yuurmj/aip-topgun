# -*- coding: utf-8 -*-
"""Strict-WEZ reward with near-target turn, heading, and speed synchronization."""
from __future__ import annotations

import numpy as np

from dogfight.sim.state_schema import StateIndex
from student.sac_flight_safety_reward import _clip
from student.sac_wez_range_reward_v6 import (
    MY_REWARD_CONFIG as BASE_REWARD_CONFIG,
    compute_reward as compute_base_reward,
)


MY_REWARD_CONFIG = {
    **BASE_REWARD_CONFIG,
    "sync_range_m": 2000.0,
    "sync_ata_gate_deg": 30.0,
    "bank_match_scale": 1.0,
    "bank_match_reference_deg": 30.0,
    "heading_match_scale": 1.0,
    "heading_match_reference_deg": 45.0,
    "speed_match_scale": 1.0,
    "speed_match_reference_mps": 100.0,
}


def _wrapped_degrees(value: float) -> float:
    return (float(value) + 180.0) % 360.0 - 180.0


def _signed_match(error: float, reference: float) -> float:
    """Return +1 at zero error, 0 at half reference, and -1 at reference."""
    return 1.0 - 2.0 * _clip(abs(float(error)) / max(float(reference), 1.0e-6))


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
    max_wez_range = float(wez_config.get("max_range_m", 914.4))
    sync_range = max(float(reward_config["sync_range_m"]), max_wez_range + 1.0)
    range_gate = _clip((sync_range - distance) / (sync_range - max_wez_range))
    ata_gate = _clip(
        1.0 - ata / max(float(reward_config["sync_ata_gate_deg"]), 0.1)
    )
    gate = range_gate * ata_gate

    bank_error = _wrapped_degrees(
        float(ownship_state[StateIndex.ROLL])
        - float(target_state[StateIndex.ROLL])
    )
    heading_error = _wrapped_degrees(
        float(ownship_state[StateIndex.YAW])
        - float(target_state[StateIndex.YAW])
    )
    ownship_speed = float(np.linalg.norm(
        ownship_state[StateIndex.U : StateIndex.W + 1]
    ))
    target_speed = float(np.linalg.norm(
        target_state[StateIndex.U : StateIndex.W + 1]
    ))
    speed_error = ownship_speed - target_speed

    bank_match = gate * float(reward_config["bank_match_scale"]) * _signed_match(
        bank_error, reward_config["bank_match_reference_deg"]
    )
    heading_match = gate * float(
        reward_config["heading_match_scale"]
    ) * _signed_match(
        heading_error, reward_config["heading_match_reference_deg"]
    )
    speed_match = gate * float(reward_config["speed_match_scale"]) * _signed_match(
        speed_error, reward_config["speed_match_reference_mps"]
    )

    components["bank_match"] = bank_match
    components["heading_match"] = heading_match
    components["speed_match"] = speed_match
    return float(total + bank_match + heading_match + speed_match), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
