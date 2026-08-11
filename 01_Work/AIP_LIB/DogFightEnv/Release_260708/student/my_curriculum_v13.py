# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.curriculum import CurriculumStage


LEVEL_FLIGHT_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "v13_level_flight_9000m_separation",
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


STABLE_WEZ_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "v13_stable_wez_700m",
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


def get_stages() -> list[CurriculumStage]:
    """Return the fixed 20+10 iteration v13 curriculum."""

    return [
        CurriculumStage(
            index=0,
            name="strict_level_flight",
            description=(
                "Learn genuinely stable level flight with "
                "strong positive reward only near level attitude."
            ),
            target_mode="autopilot",
            episode_step_limit=3600,
            max_iterations=20,
            checkpoint_interval=5,
            reward_overrides={
                "attack_shaping_enabled": 0.0,
                "survival_bonus": 0.02,
                "step_penalty": -0.01,
                "safe_altitude_bonus": 0.05,
                "level_flight_bonus": 0.50,
                "altitude_band_penalty": 0.60,
                "low_altitude_penalty": 1.20,
                "roll_penalty": 1.20,
                "level_pitch_penalty": 0.80,
                "dive_pitch_penalty": 1.20,
                "safety_reward_min": -2.50,
                "target_damage_scale": 0.0,
                "ownship_damage_scale": 0.0,
                "win_reward": 0.0,
                "target_crash_reward": 0.0,
                "timeout_penalty": 0.0,
                "draw_reward": 0.0,
                "crash_penalty": -250.0,
            },
            randomization={
                "enabled": False,
            },

            # v12 RLlib tactical metrics were n/a.
            # Use an exact fixed iteration count.
            advance_conditions={},
            advance_window=1,

            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 60.0,
                "target_autopilot": {
                    "heading_cmd": 0.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": (
                    LEVEL_FLIGHT_SCENARIO
                ),
            },
        ),
        CurriculumStage(
            index=1,
            name="stable_wez_hold_3s",
            description=(
                "Transfer the level-flight policy and hold "
                "the real WEZ with stable roll and pitch; "
                "target is at least 30 RL steps."
            ),
            target_mode="autopilot",
            episode_step_limit=3600,
            max_iterations=10,
            checkpoint_interval=5,
            reward_overrides={
                "attack_shaping_enabled": 1.0,
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "safe_altitude_bonus": 0.03,
                "level_flight_bonus": 0.15,
                "altitude_band_penalty": 0.50,
                "low_altitude_penalty": 0.80,
                "roll_penalty": 1.00,
                "level_pitch_penalty": 0.60,
                "dive_pitch_penalty": 1.00,
                "safety_reward_min": -2.00,
                "wez_hold_bonus": 1.00,
                "stable_wez_bonus": 1.00,
                "wez_unstable_penalty": 0.80,
                "wez_miss_penalty": 1.20,
                "target_damage_scale": 40.0,
                "ownship_damage_scale": 25.0,
                "win_reward": 150.0,
                "timeout_penalty": -100.0,
                "crash_penalty": -250.0,
            },
            randomization={
                "enabled": False,
            },

            # Success is judged from consecutive replay rows.
            advance_conditions={},
            advance_window=1,

            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 60.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": (
                    STABLE_WEZ_SCENARIO
                ),
            },
        ),
    ]


__all__ = [
    "LEVEL_FLIGHT_SCENARIO",
    "STABLE_WEZ_SCENARIO",
    "get_stages",
]