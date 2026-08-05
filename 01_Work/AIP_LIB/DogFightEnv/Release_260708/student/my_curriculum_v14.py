# -*- coding: utf-8 -*-
"""Fixed 10 + 10 + 10 iteration curriculum for PPO v14."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.curriculum import CurriculumStage


SHORT_LEVEL_FLIGHT_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "v14_short_level_flight_9000m_separation",
            "weight": 1.0,
            "ownship": [
                1000.0,
                0.0,
                -7000.0,
                0.0,
                0.0,
                0.0,
                250.0,
            ],
            "target": [
                10000.0,
                0.0,
                -7000.0,
                0.0,
                0.0,
                0.0,
                250.0,
            ],
            "target_mode": "autopilot",
            "target_autopilot": {
                "heading_cmd": 0.0,
                "altitude_cmd": -7000.0,
                "speed_cmd": 250.0,
            },
        }
    ],
}


WEZ_HOLD_700M_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "v14_wez_hold_700m",
            "weight": 1.0,
            "ownship": [
                6700.0,
                0.0,
                -7000.0,
                0.0,
                0.0,
                180.0,
                250.0,
            ],
            "target": [
                6000.0,
                0.0,
                -7000.0,
                0.0,
                0.0,
                180.0,
                250.0,
            ],
            "target_mode": "autopilot",
            "target_autopilot": {
                "heading_cmd": 180.0,
                "altitude_cmd": -7000.0,
                "speed_cmd": 250.0,
            },
        }
    ],
}


def _entry_scenario(distance_m: float) -> dict:
    return {
        "name": f"v14_wez_entry_{int(distance_m)}m",
        "weight": 1.0,
        # Heading 180: target is ahead toward decreasing N. Ownship starts
        # behind it and is 40 m/s faster, creating actual closure.
        "ownship": [
            6000.0 + distance_m,
            0.0,
            -7000.0,
            0.0,
            0.0,
            180.0,
            290.0,
        ],
        "target": [
            6000.0,
            0.0,
            -7000.0,
            0.0,
            0.0,
            180.0,
            250.0,
        ],
        "target_mode": "autopilot",
        "target_autopilot": {
            "heading_cmd": 180.0,
            "altitude_cmd": -7000.0,
            "speed_cmd": 250.0,
        },
    }


WEZ_ENTRY_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        _entry_scenario(1100.0),
        _entry_scenario(1200.0),
        _entry_scenario(1300.0),
    ],
}


COMMON_STABILITY_OVERRIDES = {
    "stable_roll_target_deg": 15.0,
    "stable_roll_recovery_deg": 60.0,
    "stable_pitch_target_deg": 5.0,
    "stable_pitch_recovery_deg": 25.0,
    "roll_penalty_start_deg": 30.0,
    "roll_penalty_full_deg": 90.0,
    "level_pitch_start_deg": 12.0,
    "level_pitch_full_deg": 40.0,
    "dive_pitch_start_deg": -12.0,
    "dive_pitch_full_deg": -40.0,
    "level_flight_bonus": 0.0,
}


def get_stages() -> list[CurriculumStage]:
    """Return the exact 30-iteration v14 curriculum."""
    return [
        CurriculumStage(
            index=0,
            name="short_level_flight_10s",
            description=(
                "Ten-second level-flight episodes using broad, additive "
                "roll, pitch, and altitude stability rewards."
            ),
            target_mode="autopilot",
            episode_step_limit=600,
            max_iterations=10,
            checkpoint_interval=5,
            reward_overrides={
                **COMMON_STABILITY_OVERRIDES,
                "attack_shaping_enabled": 0.0,
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "roll_stability_bonus": 0.25,
                "pitch_stability_bonus": 0.25,
                "altitude_stability_bonus": 0.10,
                "safe_altitude_bonus": 0.05,
                "altitude_band_penalty": 0.35,
                "low_altitude_penalty": 0.70,
                "roll_penalty": 0.25,
                "level_pitch_penalty": 0.25,
                "dive_pitch_penalty": 0.30,
                "safety_reward_min": -1.50,
                "target_damage_scale": 0.0,
                "ownship_damage_scale": 0.0,
                "win_reward": 0.0,
                "target_crash_reward": 0.0,
                "timeout_penalty": 0.0,
                "draw_reward": 0.0,
                "crash_penalty": -250.0,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 10.0,
                "target_autopilot": {
                    "heading_cmd": 0.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": SHORT_LEVEL_FLIGHT_SCENARIO,
            },
        ),
        CurriculumStage(
            index=1,
            name="wez_hold_700m_5s",
            description=(
                "Start inside the real WEZ at 700 m and hold it for a "
                "five-second episode without a timeout penalty."
            ),
            target_mode="autopilot",
            episode_step_limit=300,
            max_iterations=10,
            checkpoint_interval=5,
            reward_overrides={
                **COMMON_STABILITY_OVERRIDES,
                "attack_shaping_enabled": 1.0,
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "roll_stability_bonus": 0.20,
                "pitch_stability_bonus": 0.20,
                "altitude_stability_bonus": 0.08,
                "safe_altitude_bonus": 0.03,
                "altitude_band_penalty": 0.35,
                "low_altitude_penalty": 0.60,
                "roll_penalty": 0.25,
                "level_pitch_penalty": 0.25,
                "dive_pitch_penalty": 0.30,
                "safety_reward_min": -1.50,
                "alignment_scale": 0.05,
                "coarse_aim_bonus": 0.06,
                "fine_aim_bonus": 0.10,
                "wez_hold_bonus": 0.35,
                "stable_wez_bonus": 0.65,
                "wez_unstable_penalty": 0.35,
                "wez_miss_penalty": 0.60,
                "target_damage_scale": 20.0,
                "ownship_damage_scale": 25.0,
                "win_reward": 100.0,
                "target_crash_reward": 0.0,
                "timeout_penalty": 0.0,
                "draw_reward": 0.0,
                "crash_penalty": -250.0,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 5.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": WEZ_HOLD_700M_SCENARIO,
            },
        ),
        CurriculumStage(
            index=2,
            name="wez_entry_and_hold_10s",
            description=(
                "Start 1,100-1,300 m outside/near the WEZ, close with a "
                "faster ownship, enter the WEZ, and then hold it."
            ),
            target_mode="autopilot",
            episode_step_limit=600,
            max_iterations=10,
            checkpoint_interval=5,
            reward_overrides={
                **COMMON_STABILITY_OVERRIDES,
                "attack_shaping_enabled": 1.0,
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "roll_stability_bonus": 0.18,
                "pitch_stability_bonus": 0.18,
                "altitude_stability_bonus": 0.08,
                "safe_altitude_bonus": 0.03,
                "altitude_band_penalty": 0.35,
                "low_altitude_penalty": 0.60,
                "roll_penalty": 0.25,
                "level_pitch_penalty": 0.25,
                "dive_pitch_penalty": 0.30,
                "safety_reward_min": -1.50,
                "alignment_scale": 0.08,
                "pre_wez_outer_margin_m": 1500.0,
                "coarse_aim_bonus": 0.10,
                "fine_aim_bonus": 0.15,
                "wez_hold_bonus": 0.35,
                "stable_wez_bonus": 0.65,
                "wez_unstable_penalty": 0.35,
                "wez_miss_penalty": 0.60,
                "target_damage_scale": 20.0,
                "ownship_damage_scale": 25.0,
                "win_reward": 100.0,
                "target_crash_reward": 0.0,
                # A successful entry-and-hold episode is expected to time out.
                "timeout_penalty": 0.0,
                "draw_reward": 0.0,
                "crash_penalty": -250.0,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 10.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": WEZ_ENTRY_SCENARIO,
            },
        ),
    ]


__all__ = [
    "SHORT_LEVEL_FLIGHT_SCENARIO",
    "WEZ_HOLD_700M_SCENARIO",
    "WEZ_ENTRY_SCENARIO",
    "COMMON_STABILITY_OVERRIDES",
    "get_stages",
]
