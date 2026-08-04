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
            "name": "level_flight_9000m_separation",
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


WEZ_HOLD_SCENARIO = {
    "mode": "scenario_pool",
    "scenarios": [
        {
            "name": "wez_hold_700m",
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
    """Return the v12 curriculum, capped at 30 iterations."""

    return [
        CurriculumStage(
            index=0,
            name="level_flight_survival",
            description=(
                "Learn level flight before any pursuit "
                "or damage reward."
            ),
            target_mode="autopilot",
            episode_step_limit=3600,
            max_iterations=20,
            checkpoint_interval=5,
            reward_overrides={
                "attack_shaping_enabled": 0.0,
                "survival_bonus": 0.05,
                "step_penalty": -0.01,
                "safe_altitude_bonus": 0.02,
                "low_altitude_penalty": 1.00,
                "roll_penalty": 0.50,
                "level_pitch_penalty": 0.30,
                "dive_pitch_penalty": 0.50,
                "target_damage_scale": 0.0,
                "ownship_damage_scale": 0.0,
                "win_reward": 0.0,
                "target_crash_reward": 0.0,
                "timeout_penalty": 0.0,
                "draw_reward": 0.0,
                "crash_penalty": -200.0,
            },
            randomization={
                "enabled": False,
            },
            advance_conditions={
                "crash_rate_max": 0.20,
            },
            advance_window=5,
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
            name="wez_hold_1s",
            description=(
                "Transfer the level-flight policy and "
                "hold the real WEZ for at least 1 second; "
                "the target metric is 15 RL steps."
            ),
            target_mode="autopilot",
            episode_step_limit=3600,
            max_iterations=10,
            checkpoint_interval=5,
            reward_overrides={
                "attack_shaping_enabled": 1.0,
                "survival_bonus": 0.01,
                "step_penalty": -0.01,
                "safe_altitude_bonus": 0.01,
                "low_altitude_penalty": 0.50,
                "roll_penalty": 0.30,
                "level_pitch_penalty": 0.20,
                "dive_pitch_penalty": 0.30,
                "wez_hold_bonus": 1.00,
                "wez_miss_penalty": 1.00,
                "target_damage_scale": 40.0,
                "ownship_damage_scale": 25.0,
                "win_reward": 150.0,
                "timeout_penalty": -100.0,
                "crash_penalty": -200.0,
            },
            randomization={
                "enabled": False,
            },
            advance_conditions={
                "ep_wez_steps_min": 15.0,
                "crash_rate_max": 0.20,
            },
            advance_window=5,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 60.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": -7000.0,
                    "speed_cmd": 250.0,
                },
                "initial_scenario": (
                    WEZ_HOLD_SCENARIO
                ),
            },
        ),
    ]


__all__ = [
    "LEVEL_FLIGHT_SCENARIO",
    "WEZ_HOLD_SCENARIO",
    "get_stages",
]