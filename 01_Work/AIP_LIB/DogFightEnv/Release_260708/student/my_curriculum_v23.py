# -*- coding: utf-8 -*-
"""PPO v23 general-dogfight bridge step 1.

The v22 experiment broadened the state distribution too aggressively and the
policy forgot most of the validated v21 900 m attack skill.  v23 therefore
returns to the validated v21 native checkpoint and exposes only a small amount
of new combat geometry in one stage:

  * 50% validated 900 m autopilot rehearsal
  * 20% 900 m tail chase against the active behavior-tree target
  * 20% alpha 0 deg head-on bridge
  * 10% alpha 20 deg mirrored bridge

No alpha 40/60 scenarios are present.  The stage is limited to one PPO update.
A single stage also preserves the restored native PPO value-function/optimizer
state throughout this experiment.
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


FEET_TO_METER = 0.3048
CENTER_N_M = 3500.0
CENTER_E_M = 0.0
ALTITUDE_D_M = -7000.0
BRIDGE_SPEED_MPS = 280.0
TAIL_OWNSHIP_SPEED_MPS = 264.0
TAIL_TARGET_SPEED_MPS = 250.0
TAIL_DISTANCE_M = 900.0
TURN_DIAMETER_FT = 6000.0
MID_JITTER_FT = 4500.0


def _wrap_heading(value: float) -> float:
    return float(value) % 360.0


def _tail_chase_scenario(
    name: str,
    *,
    weight: float,
    target_mode: str,
    east_m: float = 0.0,
    heading_deg: float = 180.0,
) -> dict:
    """Create an exact-900 m tail-chase scenario."""
    lateral_m = float(east_m)
    forward_m = math.sqrt(TAIL_DISTANCE_M**2 - lateral_m**2)
    scenario = {
        "name": name,
        "weight": float(weight),
        "ownship": [
            6000.0 + forward_m,
            lateral_m,
            ALTITUDE_D_M,
            0.0,
            0.0,
            float(heading_deg),
            TAIL_OWNSHIP_SPEED_MPS,
        ],
        "target": [
            6000.0,
            0.0,
            ALTITUDE_D_M,
            0.0,
            0.0,
            180.0,
            TAIL_TARGET_SPEED_MPS,
        ],
        "target_mode": target_mode,
    }
    if target_mode == "autopilot":
        scenario["target_autopilot"] = {
            "heading_cmd": 180.0,
            "altitude_cmd": ALTITUDE_D_M,
            "speed_cmd": TAIL_TARGET_SPEED_MPS,
        }
    return scenario


def _two_circle_scenario(
    name: str,
    *,
    alpha_deg: float,
    side: float,
    weight: float,
) -> dict:
    """Create a deterministic sample from the competition two-circle geometry."""
    alpha_rad = math.radians(float(alpha_deg))
    separation_m = (
        2.0 * TURN_DIAMETER_FT * math.sin(alpha_rad) + MID_JITTER_FT
    ) * FEET_TO_METER
    half_sep = separation_m / 2.0
    own_heading = _wrap_heading(float(side) * float(alpha_deg))
    target_heading = _wrap_heading(180.0 + float(side) * float(alpha_deg))
    return {
        "name": name,
        "weight": float(weight),
        "ownship": [
            CENTER_N_M - half_sep,
            CENTER_E_M,
            ALTITUDE_D_M,
            0.0,
            0.0,
            own_heading,
            BRIDGE_SPEED_MPS,
        ],
        "target": [
            CENTER_N_M + half_sep,
            CENTER_E_M,
            ALTITUDE_D_M,
            0.0,
            0.0,
            target_heading,
            BRIDGE_SPEED_MPS,
        ],
        "target_mode": "behavior_tree",
    }


# Existing skill = 70%, new bridge states = 30%.
# The v21 20/30/50 center/left/right bias is retained inside both 900 m groups.
V23_GENERAL_SCENARIO_POOL = {
    "mode": "scenario_pool",
    "scenarios": [
        # 50% validated v21 900 m autopilot rehearsal.
        _tail_chase_scenario(
            "v23_rehearsal_900m_center",
            weight=0.100,
            target_mode="autopilot",
        ),
        _tail_chase_scenario(
            "v23_rehearsal_900m_left",
            weight=0.150,
            target_mode="autopilot",
            east_m=-20.0,
            heading_deg=179.5,
        ),
        _tail_chase_scenario(
            "v23_rehearsal_900m_right",
            weight=0.250,
            target_mode="autopilot",
            east_m=20.0,
            heading_deg=180.5,
        ),

        # 20% same close geometry, now against the active competition BT.
        _tail_chase_scenario(
            "v23_bt_tail_900m_center",
            weight=0.040,
            target_mode="behavior_tree",
        ),
        _tail_chase_scenario(
            "v23_bt_tail_900m_left",
            weight=0.060,
            target_mode="behavior_tree",
            east_m=-20.0,
            heading_deg=179.5,
        ),
        _tail_chase_scenario(
            "v23_bt_tail_900m_right",
            weight=0.100,
            target_mode="behavior_tree",
            east_m=20.0,
            heading_deg=180.5,
        ),

        # 20% first head-on bridge.
        _two_circle_scenario(
            "v23_bt_a000",
            alpha_deg=0.0,
            side=1.0,
            weight=0.200,
        ),

        # 10% mild mirrored crossing/aspect bridge.
        _two_circle_scenario(
            "v23_bt_a020_left",
            alpha_deg=20.0,
            side=-1.0,
            weight=0.050,
        ),
        _two_circle_scenario(
            "v23_bt_a020_right",
            alpha_deg=20.0,
            side=1.0,
            weight=0.050,
        ),
    ],
    # v22 used broader perturbations while 75% of samples were new states.
    # Step 1 deliberately keeps perturbations tiny so the experiment changes
    # geometry gradually without overfitting one exact coordinate.
    "ownship_randomization": {
        "enabled": True,
        "n_m": 15.0,
        "e_m": 15.0,
        "d_m": 10.0,
        "r_roll": 1.0,
        "r_pitch": 1.0,
        "r_heading": 1.5,
        "speed_mps": 2.0,
    },
    "target_randomization": {
        "enabled": True,
        "n_m": 15.0,
        "e_m": 15.0,
        "d_m": 10.0,
        "r_roll": 1.0,
        "r_pitch": 1.0,
        "r_heading": 1.5,
        "speed_mps": 2.0,
    },
    "shared_randomization": {
        "enabled": True,
        "n_m": 75.0,
        "e_m": 75.0,
        "d_m": 30.0,
        "speed_mps": 1.0,
    },
}


def get_stages() -> list[CurriculumStage]:
    """Return the single conservative bridge-step stage."""
    return [
        CurriculumStage(
            index=0,
            name="general_bridge_step1_a000_a020_10s",
            description=(
                "Retain v21 900 m attack skill with 70% familiar states while "
                "introducing only BT tail chase plus alpha 0/20 geometry."
            ),
            target_mode="behavior_tree",
            episode_step_limit=10 * 60,
            max_iterations=1,
            checkpoint_interval=1,
            reward_overrides={},
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 10.0,
                # Keep the competition floor so merge/recovery is not cut off by
                # the old 6,550 m tail-chase limit. Reward shaping handles dives.
                "min_altitude": 300.0,
                "initial_scenario": V23_GENERAL_SCENARIO_POOL,
                "geometry_guard": {"enabled": False},
            },
        )
    ]


__all__ = ["V23_GENERAL_SCENARIO_POOL", "get_stages"]
