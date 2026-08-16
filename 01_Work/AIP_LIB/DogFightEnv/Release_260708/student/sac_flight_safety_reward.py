# -*- coding: utf-8 -*-
"""SAC flight-safety reward used by the team01 v2 survival experiment."""
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
    "step_penalty": -0.01,
    "survival_bonus": 0.05,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "ownship_crash_reward": -120.0,
    "target_crash_reward": 100.0,
    "timeout_reward": -30.0,
    "draw_reward": -10.0,
    "guard_fail_penalty": -100.0,
    "damage_scale": 0.0,
    "altitude_warning_m": 2000.0,
    "altitude_floor_m": 300.0,
    "altitude_risk_scale": 0.10,
    "dive_penalty_scale": 0.05,
    "dive_reference_deg": 45.0,
    "excess_bank_deg": 60.0,
    "excess_bank_scale": 0.02,
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
}


def _clip(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, float(value)))


def _terminal_reward(
    ownship_state,
    target_state,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> float:
    if not (terminated or truncated):
        return 0.0

    condition = str(end_condition or "").strip().lower()
    if condition in {
        "ownship altitude below min",
        "ownship fdm output fall",
        "fdm update fail",
    }:
        return float(reward_config["ownship_crash_reward"])
    if condition in {"target altitude below min", "target fdm output fall"}:
        return float(reward_config["target_crash_reward"])
    if condition in {"max time out", "episode step limit"}:
        return float(reward_config["timeout_reward"])
    if condition == "two circle headon guard fail":
        return float(reward_config["guard_fail_penalty"])

    ownship_health = float(ownship_state[StateIndex.HEALTH])
    target_health = float(target_state[StateIndex.HEALTH])
    if target_health <= 0.0 < ownship_health or condition == "target destroyed":
        return float(reward_config["win_reward"])
    if ownship_health <= 0.0 < target_health or condition == "ownship destroyed":
        return float(reward_config["loss_reward"])
    return float(reward_config["draw_reward"])


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
    """Return flight-safety reward and independently logged components."""
    altitude = float(ownship_state[StateIndex.ALT])
    pitch_deg = float(ownship_state[StateIndex.PITCH])
    roll_deg = float(ownship_state[StateIndex.ROLL])

    warning = float(reward_config["altitude_warning_m"])
    floor = float(reward_config["altitude_floor_m"])
    risk = _clip((warning - altitude) / max(warning - floor, 1.0))

    altitude_reward = -float(reward_config["altitude_risk_scale"]) * risk**2
    dive_factor = _clip(
        -pitch_deg / max(float(reward_config["dive_reference_deg"]), 1.0)
    )
    dive_reward = -float(reward_config["dive_penalty_scale"]) * risk * dive_factor

    bank_start = float(reward_config["excess_bank_deg"])
    bank_factor = _clip((abs(roll_deg) - bank_start) / max(180.0 - bank_start, 1.0))
    bank_reward = -float(reward_config["excess_bank_scale"]) * risk * bank_factor

    pursuit_reward = 0.0
    pursuit_scale = float(reward_config.get("pursuit_scale", 0.0))
    if pursuit_scale:
        distance = float(geo_info._get_distance(ownship_state, target_state))
        ata = abs(
            float(
                geo_info._get_antenna_train_angle(
                    ownship_state, target_state, False
                )
            )
        )
        half_angle = max(float(reward_config["pursuit_half_angle_deg"]), 1.0)
        pursuit_range = max(float(reward_config["pursuit_range_m"]), 1.0)
        ata_factor = _clip(1.0 - ata / half_angle)
        range_factor = _clip(1.0 - distance / pursuit_range)
        safe_gate = 1.0 - risk
        pursuit_reward = pursuit_scale * ata_factor * range_factor * safe_gate

    components = {
        "step": float(reward_config["step_penalty"]),
        "survival": float(reward_config.get("survival_bonus", 0.0)),
        "altitude": altitude_reward,
        "dive": dive_reward,
        "bank": bank_reward,
        "pursuit": pursuit_reward,
        "damage": float(reward_config["damage_scale"])
        * (float(target_damage) - float(ownship_damage)),
        "terminal": _terminal_reward(
            ownship_state,
            target_state,
            reward_config,
            terminated,
            truncated,
            end_condition,
        ),
    }
    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
