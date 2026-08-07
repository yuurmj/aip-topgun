# -*- coding: utf-8 -*-
"""PPO v18 curriculum: one controlled 850 m -> 875 m experiment.

A single stage is intentional. The current common trainer transfers a
lightweight bundle at a stage boundary, while the requested v17 native
checkpoint contains the PPO value function and optimizer state as well as the
policy. Keeping one stage preserves that full restored training state for all
8 iterations.
"""
from __future__ import annotations

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
    """Create a same-direction pursuit scenario behind the target."""
    return {
        "name": name,
        "weight": float(weight),
        "ownship": [
            TARGET_NORTH_M + float(distance_m),
            float(east_m),
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
    """Use the same three mild geometry variants as the v17 attack anchor."""
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


# Three 850 m scenarios each have weight 0.70 and three 875 m scenarios each
# have weight 0.30. Because both groups contain the same number of variants,
# the normalized group probabilities are exactly 70% and 30%.
V18_SCENARIO_POOL = {
    "mode": "scenario_pool",
    "scenarios": (
        _distance_variants("v18_rehearsal", 850.0, group_weight=0.70)
        + _distance_variants("v18_extension", 875.0, group_weight=0.30)
    ),
}


def get_stages() -> list[CurriculumStage]:
    """Return the single eight-iteration 875 m extension stage."""
    return [
        CurriculumStage(
            index=0,
            name="attack_extension_850_875m_8s",
            description=(
                "Rehearse the v17 850 m attack at 70% and extend only to "
                "875 m at 30%, with full per-iteration checkpointing."
            ),
            target_mode="autopilot",
            episode_step_limit=8 * 60,
            max_iterations=8,
            checkpoint_interval=1,
            reward_overrides={},
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 8.0,
                # Allow recovery below v17's 6,700 m boundary while retaining a
                # hard terminal floor before the aircraft reaches unsafe terrain.
                "min_altitude": 6550.0,
                "target_autopilot": {
                    "heading_cmd": 180.0,
                    "altitude_cmd": TARGET_ALTITUDE_D_M,
                    "speed_cmd": TARGET_SPEED_MPS,
                },
                "initial_scenario": V18_SCENARIO_POOL,
            },
        )
    ]


__all__ = [
    "TARGET_SPEED_MPS",
    "OWNSHIP_SPEED_MPS",
    "V18_SCENARIO_POOL",
    "get_stages",
]
