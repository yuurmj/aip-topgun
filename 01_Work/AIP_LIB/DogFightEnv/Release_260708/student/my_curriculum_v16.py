# -*- coding: utf-8 -*-
"""PPO v16 curriculum: retain attack examples in every entry stage."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.curriculum import CurriculumStage


def _autopilot_scenario(
    name: str,
    distance_m: float,
    ownship_speed_mps: float,
    *,
    weight: float = 1.0,
    target_speed_mps: float = 250.0,
    ownship_east_m: float = 0.0,
    ownship_heading_deg: float = 180.0,
    target_heading_deg: float = 180.0,
) -> dict:
    target_north_m = 6000.0
    return {
        "name": name,
        "weight": float(weight),
        "ownship": [
            target_north_m + float(distance_m),
            float(ownship_east_m),
            -7000.0,
            0.0,
            0.0,
            float(ownship_heading_deg),
            float(ownship_speed_mps),
        ],
        "target": [
            target_north_m,
            0.0,
            -7000.0,
            0.0,
            0.0,
            float(target_heading_deg),
            float(target_speed_mps),
        ],
        "target_mode": "autopilot",
        "target_autopilot": {
            "heading_cmd": float(target_heading_deg),
            "altitude_cmd": -7000.0,
            "speed_cmd": float(target_speed_mps),
        },
    }


def _hold_scenarios(weight: float) -> list[dict]:
    return [
        _autopilot_scenario(
            "v16_hold_650m",
            650.0,
            250.0,
            weight=weight,
        ),
        _autopilot_scenario(
            "v16_hold_750m",
            750.0,
            250.0,
            weight=weight,
        ),
        _autopilot_scenario(
            "v16_hold_850m",
            850.0,
            250.0,
            weight=weight,
        ),
    ]


def _near_entry_scenarios(weight: float) -> list[dict]:
    return [
        _autopilot_scenario(
            "v16_near_950m",
            950.0,
            275.0,
            weight=weight,
        ),
        _autopilot_scenario(
            "v16_near_1050m_left",
            1050.0,
            275.0,
            weight=weight,
            ownship_east_m=-40.0,
            ownship_heading_deg=179.0,
        ),
        _autopilot_scenario(
            "v16_near_1150m_right",
            1150.0,
            275.0,
            weight=weight,
            ownship_east_m=40.0,
            ownship_heading_deg=181.0,
        ),
    ]


def _full_entry_scenarios(weight: float) -> list[dict]:
    return [
        _autopilot_scenario(
            "v16_full_1100m",
            1100.0,
            285.0,
            weight=weight,
        ),
        _autopilot_scenario(
            "v16_full_1200m_left",
            1200.0,
            285.0,
            weight=weight,
            ownship_east_m=-50.0,
            ownship_heading_deg=179.0,
        ),
        _autopilot_scenario(
            "v16_full_1300m_right",
            1300.0,
            285.0,
            weight=weight,
            ownship_east_m=50.0,
            ownship_heading_deg=181.0,
        ),
    ]


ATTACK_MEMORY_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": _hold_scenarios(weight=1.0),
}

MIXED_NEAR_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": (
        _hold_scenarios(weight=2.0)
        + _near_entry_scenarios(weight=1.0)
    ),
}

MIXED_ENTRY_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": (
        _hold_scenarios(weight=1.5)
        + _near_entry_scenarios(weight=1.0)
        + _full_entry_scenarios(weight=0.8)
    ),
}

FULL_ENTRY_WITH_REHEARSAL_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": (
        _hold_scenarios(weight=0.8)
        + _near_entry_scenarios(weight=0.8)
        + _full_entry_scenarios(weight=2.0)
    ),
}


COMMON_STABILITY_OVERRIDES = {
    "stable_roll_target_deg": 12.0,
    "stable_roll_recovery_deg": 60.0,
    "stable_pitch_target_deg": 4.0,
    "stable_pitch_recovery_deg": 20.0,
    "target_altitude_m": 7000.0,
    "target_altitude_tolerance_m": 100.0,
    "target_altitude_recovery_m": 1000.0,
    "stable_wez_roll_weight": 0.25,
    "stable_wez_pitch_weight": 0.35,
    "stable_wez_altitude_weight": 0.40,
    "roll_penalty_start_deg": 22.0,
    "roll_penalty_full_deg": 70.0,
    "inverted_roll_start_deg": 70.0,
    "inverted_roll_full_deg": 145.0,
    "level_pitch_start_deg": 7.0,
    "level_pitch_full_deg": 28.0,
    "dive_pitch_start_deg": -3.0,
    "dive_pitch_full_deg": -18.0,
    "level_flight_bonus": 0.0,
}


def _attack_overrides(
    *,
    target_damage_scale: float,
    damage_step_bonus: float,
    no_fire_penalty: float,
    entry_corridor_bonus: float,
) -> dict:
    return {
        "attack_shaping_enabled": 1.0,
        "alignment_scale": 0.03,
        "pre_wez_outer_margin_m": 1400.0,
        "coarse_aim_angle_deg": 30.0,
        "coarse_aim_bonus": 0.05,
        "fine_aim_angle_deg": 10.0,
        "fine_aim_bonus": 0.10,
        "entry_corridor_angle_deg": 18.0,
        "entry_corridor_bonus": entry_corridor_bonus,
        "wez_hold_bonus": 0.08,
        "stable_wez_bonus": 0.25,
        "wez_unstable_penalty": 0.45,
        "attack_track_angle_deg": 8.0,
        "attack_track_bonus": 0.55,
        "attack_ready_angle_deg": 5.0,
        "attack_ready_bonus": 0.65,
        "centerline_angle_deg": 3.0,
        "centerline_bonus": 0.80,
        "wez_miss_angle_deg": 25.0,
        "wez_miss_penalty": 1.10,
        "no_fire_in_range_penalty": no_fire_penalty,
        "no_fire_full_angle_deg": 45.0,
        "damage_step_bonus": damage_step_bonus,
        "weapon_quality_bonus": 0.80,
        "target_damage_scale": target_damage_scale,
        "ownship_damage_scale": 40.0,
        "preferred_range_min_m": 350.0,
        "preferred_range_peak_m": 550.0,
        "preferred_range_max_m": 820.0,
        "preferred_range_bonus": 0.35,
        "too_close_range_m": 260.0,
        "too_close_penalty": 0.70,
        "close_misaligned_range_m": 450.0,
        "close_misaligned_angle_deg": 8.0,
        "close_misaligned_penalty": 1.20,
        "win_reward": 150.0,
        "loss_reward": -180.0,
        "crash_penalty": -320.0,
        "target_crash_reward": 0.0,
        "timeout_penalty": 0.0,
        "draw_reward": 0.0,
    }


def _base_stage_rewards() -> dict:
    return {
        **COMMON_STABILITY_OVERRIDES,
        "survival_bonus": 0.01,
        "step_penalty": -0.01,
        "roll_stability_bonus": 0.18,
        "pitch_stability_bonus": 0.28,
        "altitude_stability_bonus": 0.24,
        "altitude_center_penalty": 0.38,
        "safe_altitude_bonus": 0.02,
        "altitude_band_penalty": 0.50,
        "low_altitude_penalty": 0.90,
        "roll_penalty": 0.75,
        "inverted_roll_penalty": 1.20,
        "level_pitch_penalty": 0.55,
        "dive_pitch_penalty": 0.85,
        "safety_reward_min": -3.00,
    }


def _env_overrides(max_engage_time: float, scenario: dict) -> dict:
    return {
        "step_ratio": 6,
        "max_engage_time": float(max_engage_time),
        "target_autopilot": {
            "heading_cmd": 180.0,
            "altitude_cmd": -7000.0,
            "speed_cmd": 250.0,
        },
        "initial_scenario": scenario,
    }


def get_stages() -> list[CurriculumStage]:
    """Return a 40-iteration attack-preserving entry curriculum."""
    return [
        CurriculumStage(
            index=0,
            name="attack_memory_rehearsal_8s",
            description=(
                "Start from the v15 attack-capable bundle and reinforce "
                "650-850 m firing before any entry fine-tuning."
            ),
            target_mode="autopilot",
            episode_step_limit=480,
            max_iterations=6,
            checkpoint_interval=3,
            reward_overrides={
                **_base_stage_rewards(),
                **_attack_overrides(
                    target_damage_scale=100.0,
                    damage_step_bonus=1.20,
                    no_fire_penalty=0.70,
                    entry_corridor_bonus=0.03,
                ),
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides=_env_overrides(
                8.0,
                ATTACK_MEMORY_SCENARIO,
            ),
        ),
        CurriculumStage(
            index=1,
            name="mixed_hold_near_entry_12s",
            description=(
                "Mix easy firing episodes with 950-1,150 m entry episodes "
                "so attack is rehearsed while entry is learned."
            ),
            target_mode="autopilot",
            episode_step_limit=720,
            max_iterations=10,
            checkpoint_interval=5,
            reward_overrides={
                **_base_stage_rewards(),
                **_attack_overrides(
                    target_damage_scale=110.0,
                    damage_step_bonus=1.30,
                    no_fire_penalty=0.80,
                    entry_corridor_bonus=0.05,
                ),
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides=_env_overrides(
                12.0,
                MIXED_NEAR_SCENARIO,
            ),
        ),
        CurriculumStage(
            index=2,
            name="mixed_full_entry_attack_15s",
            description=(
                "Mix hold, near-entry, and full-entry episodes while making "
                "actual firing more valuable than WEZ entry alone."
            ),
            target_mode="autopilot",
            episode_step_limit=900,
            max_iterations=12,
            checkpoint_interval=4,
            reward_overrides={
                **_base_stage_rewards(),
                **_attack_overrides(
                    target_damage_scale=125.0,
                    damage_step_bonus=1.45,
                    no_fire_penalty=0.90,
                    entry_corridor_bonus=0.06,
                ),
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides=_env_overrides(
                15.0,
                MIXED_ENTRY_SCENARIO,
            ),
        ),
        CurriculumStage(
            index=3,
            name="full_entry_attack_with_rehearsal_15s",
            description=(
                "Emphasize 1,100-1,300 m entry but retain hold and near-entry "
                "rehearsal to prevent catastrophic forgetting of attack."
            ),
            target_mode="autopilot",
            episode_step_limit=900,
            max_iterations=12,
            checkpoint_interval=4,
            reward_overrides={
                **_base_stage_rewards(),
                **_attack_overrides(
                    target_damage_scale=140.0,
                    damage_step_bonus=1.60,
                    no_fire_penalty=1.00,
                    entry_corridor_bonus=0.07,
                ),
                "attack_track_bonus": 0.65,
                "attack_ready_bonus": 0.75,
                "centerline_bonus": 0.95,
                "wez_miss_penalty": 1.25,
                "close_misaligned_penalty": 1.35,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides=_env_overrides(
                15.0,
                FULL_ENTRY_WITH_REHEARSAL_SCENARIO,
            ),
        ),
    ]


__all__ = [
    "ATTACK_MEMORY_SCENARIO",
    "MIXED_NEAR_SCENARIO",
    "MIXED_ENTRY_SCENARIO",
    "FULL_ENTRY_WITH_REHEARSAL_SCENARIO",
    "COMMON_STABILITY_OVERRIDES",
    "get_stages",
]
