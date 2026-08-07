# -*- coding: utf-8 -*-
"""PPO v19 curriculum: one controlled 875 m -> 900 m experiment.

v18 iter_0003 passed the dedicated 30-episode 875 m evaluation, so v19
changes only the distance curriculum while keeping the validated flight speed,
8 s engagement window, and altitude recovery room. A single stage preserves
the full native PPO checkpoint state for all four fine-tuning iterations.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.curriculum import CurriculumStage


TARGET_NORTH_M = 6000.0
TARGET_ALTITUDE_D_M = -7000.0
TARGET_SPEED_MPS = 250.0
OWNSHIP_SPEED_MPS = 264.0


def _autopilot_scenario(
    name: str,
    distance_m: float,
    *,
    weight: float,
    east_m: float = 0.0,
    heading_deg: float = 180.0,
) -> dict:
    """Create a pursuit scenario with the requested true horizontal distance."""
    lateral_m = float(east_m)
    if abs(lateral_m) > float(distance_m):
        raise ValueError("lateral offset cannot exceed requested initial distance")
    forward_m = math.sqrt(float(distance_m) ** 2 - lateral_m ** 2)

    return {
        "name": name,
        "weight": float(weight),
        "ownship": [
            TARGET_NORTH_M + forward_m,
            lateral_m,
            TARGET_ALTITUDE_D_M,
            0.0,
            0.0,
            float(heading_deg),
            OWNSHIP_SPEED_MPS,
        ],
        "target": [
            TARGET_NORTH_M,
            0.0,
            TARGET_ALTITUDE_D_M,
            0.0,
            0.0,
            180.0,
            TARGET_SPEED_MPS,
        ],
        "target_mode": "autopilot",
        "target_autopilot": {
            "heading_cmd": 180.0,
            "altitude_cmd": TARGET_ALTITUDE_D_M,
            "speed_cmd": TARGET_SPEED_MPS,
        },
    }


def _distance_variants(
    prefix: str,
    distance_m: float,
    *,
    group_weight: float,
) -> list[dict]:
    """Keep the validated center/left/right family at an exact radial distance."""
    return [
        _autopilot_scenario(
            f"{prefix}_{int(distance_m)}m_center",
            distance_m,
            weight=group_weight,
        ),
        _autopilot_scenario(
            f"{prefix}_{int(distance_m)}m_left",
            distance_m,
            weight=group_weight,
            east_m=-20.0,
            heading_deg=179.5,
        ),
        _autopilot_scenario(
            f"{prefix}_{int(distance_m)}m_right",
            distance_m,
            weight=group_weight,
            east_m=20.0,
            heading_deg=180.5,
        ),
    ]


# Both distance groups contain three variants, so normalized probability mass
# is exactly 70% for 875 m rehearsal and 30% for the new 900 m extension.
V19_SCENARIO_POOL = {
    "mode": "scenario_pool",
    "scenarios": (
        _distance_variants("v19_rehearsal", 875.0, group_weight=0.70)
        + _distance_variants("v19_extension", 900.0, group_weight=0.30)
    ),
}


def get_stages() -> list[CurriculumStage]:
    """Return one four-iteration 900 m extension stage."""
    return [
        CurriculumStage(
            index=0,
            name="attack_extension_875_900m_8s",
            description=(
                "Rehearse validated 875 m attacks at 70% and extend only to "
                "900 m at 30%, with checkpoint and replay logging every iteration."
            ),
            target_mode="autopilot",
            episode_step_limit=8 * 60,
            max_iterations=4,
            checkpoint_interval=1,
            reward_overrides={},
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 8.0,
                "min_altitude": 6550.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": TARGET_ALTITUDE_D_M,
                    "speed_cmd": TARGET_SPEED_MPS,
                },
                "initial_scenario": V19_SCENARIO_POOL,
            },
        )
    ]


__all__ = [
    "TARGET_SPEED_MPS",
    "OWNSHIP_SPEED_MPS",
    "V19_SCENARIO_POOL",
    "get_stages",
]
