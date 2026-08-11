# -*- coding: utf-8 -*-
"""PPO v21 curriculum: align 900 m attacks with the real firing cone.

v20 iter_0001 retained excellent flight stability but independent evaluation
showed insufficient right-side damage. Source inspection confirmed that the
environment applies damage only inside wez.angle_deg / 2 (currently 1 degree),
while v20 shaping treated ATA <= 5 degrees as a strong attack posture. v21
keeps the validated 900 m 20/30/50 scenario distribution and changes only the
reward alignment. The run is deliberately limited to one PPO iteration.
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
REFINE_DISTANCE_M = 900.0


def _autopilot_scenario(
    name: str,
    *,
    weight: float,
    east_m: float = 0.0,
    heading_deg: float = 180.0,
) -> dict:
    """Create a 900 m pursuit scenario with exact horizontal separation."""
    lateral_m = float(east_m)
    if abs(lateral_m) > REFINE_DISTANCE_M:
        raise ValueError("lateral offset cannot exceed requested initial distance")

    # Keep true initial horizontal distance exactly 900 m even for +/-20 m
    # lateral variants used by the validated v19 curriculum/evaluator.
    forward_m = math.sqrt(REFINE_DISTANCE_M**2 - lateral_m**2)

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


# Weights sum to 1.0, so the normalized probabilities are exactly:
# center 20%, left 30%, right 50%.
V21_SCENARIO_POOL = {
    "mode": "scenario_pool",
    "scenarios": [
        _autopilot_scenario(
            "v21_firecone_900m_center",
            weight=0.20,
        ),
        _autopilot_scenario(
            "v21_firecone_900m_left",
            weight=0.30,
            east_m=-20.0,
            heading_deg=179.5,
        ),
        _autopilot_scenario(
            "v21_firecone_900m_right",
            weight=0.50,
            east_m=20.0,
            heading_deg=180.5,
        ),
    ],
}


def get_stages() -> list[CurriculumStage]:
    """Return one single-iteration 900 m firing-cone refinement stage."""
    return [
        CurriculumStage(
            index=0,
            name="attack_refine_900m_fire_cone_8s",
            description=(
                "Align 900 m attack shaping with the simulator firing cone while "
                "retaining center/left/right sampling of 20/30/50%."
            ),
            target_mode="autopilot",
            episode_step_limit=8 * 60,
            max_iterations=1,
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
                "initial_scenario": V21_SCENARIO_POOL,
            },
        )
    ]


__all__ = [
    "TARGET_SPEED_MPS",
    "OWNSHIP_SPEED_MPS",
    "REFINE_DISTANCE_M",
    "V21_SCENARIO_POOL",
    "get_stages",
]
