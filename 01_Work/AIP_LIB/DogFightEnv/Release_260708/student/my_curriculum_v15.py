# -*- coding: utf-8 -*-
"""PPO v15 refinement curriculum built on the v14 Stage-2 bundle."""
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
    target_speed_mps: float = 250.0,
    heading_deg: float = 180.0,
) -> dict:
    """Build a same-heading pursuit scenario at 7,000 m altitude."""
    target_north_m = 6000.0
    return {
        "name": name,
        "weight": 1.0,
        "ownship": [
            target_north_m + distance_m,
            0.0,
            -7000.0,
            0.0,
            0.0,
            heading_deg,
            ownship_speed_mps,
        ],
        "target": [
            target_north_m,
            0.0,
            -7000.0,
            0.0,
            0.0,
            heading_deg,
            target_speed_mps,
        ],
        "target_mode": "autopilot",
        "target_autopilot": {
            "heading_cmd": heading_deg,
            "altitude_cmd": -7000.0,
            "speed_cmd": target_speed_mps,
        },
    }


ALTITUDE_RECOVERY_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        _autopilot_scenario(
            name="v15_altitude_recovery_far_target",
            distance_m=9000.0,
            ownship_speed_mps=250.0,
            target_speed_mps=250.0,
            heading_deg=0.0,
        )
    ],
}


WEZ_HOLD_ATTACK_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        _autopilot_scenario("v15_wez_hold_650m", 650.0, 250.0),
        _autopilot_scenario("v15_wez_hold_750m", 750.0, 250.0),
        _autopilot_scenario("v15_wez_hold_850m", 850.0, 250.0),
    ],
}


NEAR_ENTRY_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        _autopilot_scenario("v15_near_entry_950m", 950.0, 280.0),
        _autopilot_scenario("v15_near_entry_1050m", 1050.0, 280.0),
        _autopilot_scenario("v15_near_entry_1150m", 1150.0, 280.0),
    ],
}


FULL_ENTRY_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        _autopilot_scenario("v15_full_entry_1100m", 1100.0, 290.0),
        _autopilot_scenario("v15_full_entry_1200m", 1200.0, 290.0),
        _autopilot_scenario("v15_full_entry_1300m", 1300.0, 290.0),
    ],
}


COMMON_STABILITY_OVERRIDES = {
    "stable_roll_target_deg": 12.0,
    "stable_roll_recovery_deg": 60.0,
    "stable_pitch_target_deg": 4.0,
    "stable_pitch_recovery_deg": 20.0,
    "target_altitude_m": 7000.0,
    "target_altitude_tolerance_m": 100.0,
    "target_altitude_recovery_m": 1000.0,
    "stable_wez_roll_weight": 0.30,
    "stable_wez_pitch_weight": 0.35,
    "stable_wez_altitude_weight": 0.35,
    "roll_penalty_start_deg": 25.0,
    "roll_penalty_full_deg": 75.0,
    "inverted_roll_start_deg": 75.0,
    "inverted_roll_full_deg": 150.0,
    "level_pitch_start_deg": 8.0,
    "level_pitch_full_deg": 30.0,
    "dive_pitch_start_deg": -4.0,
    "dive_pitch_full_deg": -20.0,
    "level_flight_bonus": 0.0,
}


def _attack_rewards(
    *,
    entry_corridor_bonus: float,
    target_damage_scale: float,
) -> dict:
    return {
        "attack_shaping_enabled": 1.0,
        "alignment_scale": 0.07,
        "pre_wez_outer_margin_m": 1400.0,
        "coarse_aim_angle_deg": 30.0,
        "coarse_aim_bonus": 0.08,
        "fine_aim_angle_deg": 10.0,
        "fine_aim_bonus": 0.14,
        "entry_corridor_angle_deg": 20.0,
        "entry_corridor_bonus": entry_corridor_bonus,
        "wez_hold_bonus": 0.25,
        "stable_wez_bonus": 0.60,
        "wez_unstable_penalty": 0.45,
        "attack_ready_angle_deg": 5.0,
        "attack_ready_bonus": 0.25,
        "wez_miss_penalty": 0.70,
        "target_damage_scale": target_damage_scale,
        "ownship_damage_scale": 30.0,
        "win_reward": 120.0,
        "loss_reward": -170.0,
        "crash_penalty": -300.0,
        "target_crash_reward": 0.0,
        "timeout_penalty": 0.0,
        "draw_reward": 0.0,
    }


def get_stages() -> list[CurriculumStage]:
    """Return the 40-iteration v15 refinement curriculum."""
    return [
        CurriculumStage(
            index=0,
            name="altitude_and_attitude_recovery_8s",
            description=(
                "Repair v14 altitude drift and prevent large roll excursions "
                "before returning to attack training."
            ),
            target_mode="autopilot",
            episode_step_limit=480,
            max_iterations=8,
            checkpoint_interval=4,
            reward_overrides={
                **COMMON_STABILITY_OVERRIDES,
                "attack_shaping_enabled": 0.0,
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "roll_stability_bonus": 0.25,
                "pitch_stability_bonus": 0.35,
                "altitude_stability_bonus": 0.30,
                "altitude_center_penalty": 0.50,
                "safe_altitude_bonus": 0.02,
                "altitude_band_penalty": 0.55,
                "low_altitude_penalty": 1.00,
                "roll_penalty": 0.80,
                "inverted_roll_penalty": 1.20,
                "level_pitch_penalty": 0.60,
                "dive_pitch_penalty": 0.90,
                "safety_reward_min": -3.00,
                "target_damage_scale": 0.0,
                "ownship_damage_scale": 0.0,
                "win_reward": 0.0,
                "loss_reward": -170.0,
                "target_crash_reward": 0.0,
                "timeout_penalty": 0.0,
                "draw_reward": 0.0,
                "crash_penalty": -300.0,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 8.0,
                "target_autopilot": {
                    "heading_cmd": 0.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": ALTITUDE_RECOVERY_SCENARIO,
            },
        ),
        CurriculumStage(
            index=1,
            name="stable_wez_hold_and_attack_8s",
            description=(
                "Consolidate stable WEZ hold and automatic attack at "
                "650-850 m before entry training."
            ),
            target_mode="autopilot",
            episode_step_limit=480,
            max_iterations=8,
            checkpoint_interval=4,
            reward_overrides={
                **COMMON_STABILITY_OVERRIDES,
                **_attack_rewards(
                    entry_corridor_bonus=0.05,
                    target_damage_scale=35.0,
                ),
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "roll_stability_bonus": 0.20,
                "pitch_stability_bonus": 0.28,
                "altitude_stability_bonus": 0.22,
                "altitude_center_penalty": 0.35,
                "roll_penalty": 0.65,
                "inverted_roll_penalty": 1.00,
                "level_pitch_penalty": 0.50,
                "dive_pitch_penalty": 0.75,
                "safety_reward_min": -2.50,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 8.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": WEZ_HOLD_ATTACK_SCENARIO,
            },
        ),
        CurriculumStage(
            index=2,
            name="near_wez_entry_attack_12s",
            description=(
                "Learn smooth entry from 950-1,150 m, then hold a stable "
                "attack-ready WEZ position."
            ),
            target_mode="autopilot",
            episode_step_limit=720,
            max_iterations=12,
            checkpoint_interval=4,
            reward_overrides={
                **COMMON_STABILITY_OVERRIDES,
                **_attack_rewards(
                    entry_corridor_bonus=0.14,
                    target_damage_scale=35.0,
                ),
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "roll_stability_bonus": 0.18,
                "pitch_stability_bonus": 0.25,
                "altitude_stability_bonus": 0.20,
                "altitude_center_penalty": 0.32,
                "roll_penalty": 0.65,
                "inverted_roll_penalty": 1.00,
                "level_pitch_penalty": 0.50,
                "dive_pitch_penalty": 0.75,
                "safety_reward_min": -2.50,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 12.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": NEAR_ENTRY_SCENARIO,
            },
        ),
        CurriculumStage(
            index=3,
            name="full_wez_entry_hold_attack_15s",
            description=(
                "Start 1,100-1,300 m away, enter WEZ, maintain altitude and "
                "attitude, and convert the hold into target damage."
            ),
            target_mode="autopilot",
            episode_step_limit=900,
            max_iterations=12,
            checkpoint_interval=4,
            reward_overrides={
                **COMMON_STABILITY_OVERRIDES,
                **_attack_rewards(
                    entry_corridor_bonus=0.16,
                    target_damage_scale=40.0,
                ),
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "roll_stability_bonus": 0.18,
                "pitch_stability_bonus": 0.25,
                "altitude_stability_bonus": 0.20,
                "altitude_center_penalty": 0.35,
                "stable_wez_bonus": 0.65,
                "attack_ready_bonus": 0.30,
                "roll_penalty": 0.70,
                "inverted_roll_penalty": 1.10,
                "level_pitch_penalty": 0.55,
                "dive_pitch_penalty": 0.80,
                "safety_reward_min": -2.75,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 15.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": FULL_ENTRY_SCENARIO,
            },
        ),
    ]


__all__ = [
    "ALTITUDE_RECOVERY_SCENARIO",
    "WEZ_HOLD_ATTACK_SCENARIO",
    "NEAR_ENTRY_SCENARIO",
    "FULL_ENTRY_SCENARIO",
    "COMMON_STABILITY_OVERRIDES",
    "get_stages",
]
