# -*- coding: utf-8 -*-
"""PPO+MLP v26: direct legacy full-dogfight training.

Only the student curriculum hook is changed. The common curriculum engine and
environment are left untouched. ``ref_old_random`` is the Release-provided
legacy full-dogfight generator. Every episode is sampled directly from the
official legacy scenario set 0-7 with the competition randomization.
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


LEGACY_SCENARIO_INDICES = [0, 1, 2, 3, 4, 5, 6, 7]
MAX_ENGAGE_TIME_S = 300.0
EPISODE_STEP_LIMIT = 18000


V26_INITIAL_SCENARIO = {
    "mode": "ref_old_random",
    "legacy_use_random_scenario": True,
    "legacy_use_first_scenario_only": False,
    "legacy_scenario_indices": LEGACY_SCENARIO_INDICES,
    "legacy_randomization": {
        "aircraft_radius_m": 100.0,
        "roll_deg": 5.0,
        "pitch_deg": 5.0,
        "heading_deg": 5.0,
        "shared_n_m": 4000.0,
        "shared_e_m": 4000.0,
        "shared_d_m": 4000.0,
        "target_distance_n_m": 300.0,
        "speed_mps": 50.0,
        "loiter_bank_deg_range": [40.0, 70.0],
    },
}


def get_stages() -> list[CurriculumStage]:
    return [
        CurriculumStage(
            index=0,
            name="legacy_full_dogfight_0_7_300s",
            description=(
                "Restore v21 PPO+MLP and train directly on all eight official "
                "legacy full-dogfight scenarios for up to 300 seconds."
            ),
            # ref_old_random switches scenarios 5-7 to legacy loiter mode and
            # keeps scenarios 0-4 on the active behavior-tree opponent.
            target_mode="behavior_tree",
            episode_step_limit=EPISODE_STEP_LIMIT,
            max_iterations=100,
            checkpoint_interval=5,
            reward_overrides={
                "combat_reward_scale": 1.00,
                "damage_reward_scale": 1.00,
                "survival_bonus": 0.001,
            },
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=10,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": MAX_ENGAGE_TIME_S,
                "min_altitude": 300.0,
                "initial_scenario": V26_INITIAL_SCENARIO,
                "geometry_guard": {"enabled": False},
            },
        )
    ]


__all__ = [
    "LEGACY_SCENARIO_INDICES",
    "V26_INITIAL_SCENARIO",
    "get_stages",
]
