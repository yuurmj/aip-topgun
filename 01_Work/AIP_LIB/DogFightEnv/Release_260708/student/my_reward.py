# -*- coding: utf-8 -*-
"""PPO v3 reward for stable WEZ entry and hold.

Required contract:
  - MY_REWARD_CONFIG must be a dict.
  - compute_reward(...) returns (total_reward: float, components: dict).
  - Component names are logged as ep_reward_<name>.
"""
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
    # Basic flight reward
    "step_penalty": -0.01,
    "low_altitude_threshold_m": 600.0,
    "low_altitude_penalty": 0.20,

    # Broad approach signal (weaker than v1)
    "pursuit_scale": 0.02,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 8000.0,

    # Soft attack geometry around the real WEZ
    "attack_geometry_scale": 0.30,
    "soft_ata_limit_deg": 30.0,
    "soft_aa_limit_deg": 45.0,
    "attack_range_min_m": 152.4,
    "attack_range_best_m": 533.4,
    "attack_range_max_m": 1000.0,

    # ATA curriculum bonuses, applied only within 1500 m
    "ata_stage_max_range_m": 1500.0,
    "ata_under_10_bonus": 0.03,
    "ata_under_5_bonus": 0.08,
    "ata_under_2_bonus": 0.15,
    "ata_under_1_bonus": 0.25,

    # Real WEZ entry/hold bonus per RL step
    "wez_bonus": 1.00,

    # Penalize passing too close while no longer pointing at the target
    "too_close_distance_m": 300.0,
    "too_close_ata_deg": 10.0,
    "too_close_penalty": 0.40,

    # Distance-state penalty. This is not a true distance-progress reward.
    "escape_start_m": 3000.0,
    "escape_full_m": 8000.0,
    "escape_penalty_scale": 0.02,

    # Combat and terminal rewards
    "damage_scale": 20.0,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "crash_penalty": -100.0,
    "timeout_penalty": -30.0,
    "draw_reward": -20.0,
    "guard_fail_penalty": -50.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _triangular_range_score(
    distance: float,
    minimum: float,
    best: float,
    maximum: float,
) -> float:
    """Return 0 outside [minimum, maximum] and 1 at best."""
    if distance < minimum or distance > maximum:
        return 0.0
    if distance <= best:
        return _clamp01((distance - minimum) / max(best - minimum, 1.0))
    return _clamp01((maximum - distance) / max(maximum - best, 1.0))


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
    """Compute PPO v3 reward for approach, WEZ entry, and WEZ hold."""
    distance = float(geo_info._get_distance(ownship_state, target_state))
    ata = abs(
        float(
            geo_info._get_antenna_train_angle(
                ownship_state,
                target_state,
                False,
            )
        )
    )
    aa = abs(
        float(
            geo_info._get_aspect_angle(
                ownship_state,
                target_state,
                False,
            )
        )
    )

    components: dict[str, float] = {}

    # 1. Time efficiency
    components["step"] = float(reward_config.get("step_penalty", -0.01))

    # 2. Low-altitude safety
    altitude = float(ownship_state[StateIndex.ALT])
    altitude_threshold = float(
        reward_config.get("low_altitude_threshold_m", 600.0)
    )
    components["safety"] = (
        -float(reward_config.get("low_altitude_penalty", 0.20))
        if altitude < altitude_threshold
        else 0.0
    )

    # 3. Broad pursuit signal: point toward and approach the target.
    pursuit_ata_limit = float(
        reward_config.get("pursuit_half_angle_deg", 30.0)
    )
    pursuit_range = float(reward_config.get("pursuit_range_m", 8000.0))
    pursuit_ata_score = _clamp01(1.0 - ata / max(pursuit_ata_limit, 1.0))
    pursuit_range_score = _clamp01(1.0 - distance / max(pursuit_range, 1.0))
    components["pursuit"] = (
        float(reward_config.get("pursuit_scale", 0.02))
        * pursuit_ata_score
        * pursuit_range_score
    )

    # 4. Soft attack geometry.
    range_score = _triangular_range_score(
        distance,
        float(reward_config.get("attack_range_min_m", 152.4)),
        float(reward_config.get("attack_range_best_m", 533.4)),
        float(reward_config.get("attack_range_max_m", 1000.0)),
    )
    ata_score = _clamp01(
        1.0 - ata / max(float(reward_config.get("soft_ata_limit_deg", 30.0)), 1.0)
    )

    # In this environment, aspect angle is best near 0 degrees when ownship
    # is behind the target.
    tail_error = aa
    tail_score = _clamp01(
        1.0
        - tail_error
        / max(float(reward_config.get("soft_aa_limit_deg", 45.0)), 1.0)
    )
    components["attack_geometry"] = (
        float(reward_config.get("attack_geometry_scale", 0.30))
        * range_score
        * ata_score
        * tail_score
    )

    # 5. Staged ATA bonuses near attack range.
    ata_stage = 0.0
    if distance <= float(reward_config.get("ata_stage_max_range_m", 1500.0)):
        if ata <= 1.0:
            ata_stage = float(reward_config.get("ata_under_1_bonus", 0.25))
        elif ata <= 2.0:
            ata_stage = float(reward_config.get("ata_under_2_bonus", 0.15))
        elif ata <= 5.0:
            ata_stage = float(reward_config.get("ata_under_5_bonus", 0.08))
        elif ata <= 10.0:
            ata_stage = float(reward_config.get("ata_under_10_bonus", 0.03))
    components["ata_stage"] = ata_stage

    # 6. Actual geometric WEZ entry and per-step hold reward.
    wez_half_angle = float(wez_config["angle_deg"]) / 2.0
    in_wez = (
        float(wez_config["min_range_m"])
        <= distance
        <= float(wez_config["max_range_m"])
        and ata <= wez_half_angle
    )
    components["wez"] = (
        float(reward_config.get("wez_bonus", 1.00)) if in_wez else 0.0
    )

    # 7. Overshoot / too-close penalty.
    wez_min_range = float(wez_config["min_range_m"])
    too_close_distance = float(
        reward_config.get("too_close_distance_m", 300.0)
    )
    too_close_ata = float(
        reward_config.get("too_close_ata_deg", 10.0)
    )
    too_close_penalty = float(
        reward_config.get("too_close_penalty", 0.40)
    )

    if distance < wez_min_range:
        # Never reward diving inside the minimum legal WEZ distance.
        components["too_close"] = -too_close_penalty
    elif distance < too_close_distance and ata > too_close_ata:
        # Penalize the beginning of an overshoot while aim is already lost.
        components["too_close"] = -(too_close_penalty * 0.5)
    else:
        components["too_close"] = 0.0

    # 8. Penalize remaining far away. Previous-step distance is unavailable here.
    escape_start = float(reward_config.get("escape_start_m", 3000.0))
    escape_full = float(reward_config.get("escape_full_m", 8000.0))
    escape_score = _clamp01(
        (distance - escape_start) / max(escape_full - escape_start, 1.0)
    )
    components["escape"] = (
        -float(reward_config.get("escape_penalty_scale", 0.02))
        * escape_score
    )

    # 9. Damage differential.
    components["damage"] = float(reward_config.get("damage_scale", 20.0)) * (
        float(target_damage) - float(ownship_damage)
    )

    # 10. Terminal reward. Both terminated and truncated are handled.
    terminal_reward = 0.0
    if terminated or truncated:
        ownship_health = float(ownship_state[StateIndex.HEALTH])
        target_health = float(target_state[StateIndex.HEALTH])

        if truncated or end_condition in {"max time out", "episode step limit"}:
            terminal_reward = float(
                reward_config.get("timeout_penalty", -30.0)
            )
        elif end_condition in {"ownship altitude below min", "FDM Update Fail"}:
            terminal_reward = float(reward_config.get("crash_penalty", -100.0))
        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(
                reward_config.get("guard_fail_penalty", -50.0)
            )
        elif target_health <= 0.0 < ownship_health:
            terminal_reward = float(reward_config.get("win_reward", 100.0))
        elif ownship_health <= 0.0 < target_health:
            terminal_reward = float(reward_config.get("loss_reward", -100.0))
        else:
            terminal_reward = float(reward_config.get("draw_reward", -20.0))
    components["terminal"] = terminal_reward

    return float(sum(components.values())), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
