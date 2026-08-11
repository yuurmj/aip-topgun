# -*- coding: utf-8 -*-
"""PPO v22 general-dogfight bridge curriculum.

This is deliberately a *single-stage mixed scenario pool*.  The common trainer
rebuilds RLlib at stage boundaries and transfers only a lightweight policy
bundle, so using one stage preserves the restored v21 native checkpoint's value
function/optimizer state throughout the short bridge run.

The pool is not meant to teach one specific AIP_BASE_target.dll opening.  It
mixes the validated v21 attack rehearsal with active behavior-tree opponents
across mirrored, official two-circle-inspired aspect geometries.  The geometry
formula is the same as src/dogfight/envs/single_agent_env.py's
_apply_two_circle_headon_initial_scenario, using the official 6000 ft turn
diameter and a 4500 ft midpoint separation jitter.
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
    """Create an exact-900 m tail-chase rehearsal/active-opponent scenario."""
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
    """Create one deterministic sample from the official two-circle geometry."""
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


# Distribution:
#   25% validated v21 autopilot attack rehearsal (prevents attack forgetting)
#   15% active-BT 900 m tail chase (transfer attack skill to an evasive target)
#   20% alpha 0   two-circle-inspired head-on
#   15% alpha 20  mirrored aspect
#   15% alpha 40  mirrored aspect
#   10% alpha 60  mirrored aspect / ~4.5 km separation
# Total = 100%.
V22_GENERAL_SCENARIO_POOL = {
    "mode": "scenario_pool",
    "scenarios": [
        # v21 attack rehearsal: preserve center/left/right 20/30/50 ratio within 25%.
        _tail_chase_scenario(
            "v22_rehearsal_900m_center",
            weight=0.050,
            target_mode="autopilot",
        ),
        _tail_chase_scenario(
            "v22_rehearsal_900m_left",
            weight=0.075,
            target_mode="autopilot",
            east_m=-20.0,
            heading_deg=179.5,
        ),
        _tail_chase_scenario(
            "v22_rehearsal_900m_right",
            weight=0.125,
            target_mode="autopilot",
            east_m=20.0,
            heading_deg=180.5,
        ),

        # Same close-range attack geometry, but now the target is active BT.
        _tail_chase_scenario(
            "v22_bt_tail_900m_center",
            weight=0.050,
            target_mode="behavior_tree",
        ),
        _tail_chase_scenario(
            "v22_bt_tail_900m_left",
            weight=0.050,
            target_mode="behavior_tree",
            east_m=-20.0,
            heading_deg=179.5,
        ),
        _tail_chase_scenario(
            "v22_bt_tail_900m_right",
            weight=0.050,
            target_mode="behavior_tree",
            east_m=20.0,
            heading_deg=180.5,
        ),

        # Official two-circle-inspired geometries. Alpha 0 is symmetric by side,
        # so one scenario is sufficient; nonzero alphas use mirrored headings.
        _two_circle_scenario(
            "v22_bt_a000",
            alpha_deg=0.0,
            side=1.0,
            weight=0.200,
        ),
        _two_circle_scenario(
            "v22_bt_a020_left",
            alpha_deg=20.0,
            side=-1.0,
            weight=0.075,
        ),
        _two_circle_scenario(
            "v22_bt_a020_right",
            alpha_deg=20.0,
            side=1.0,
            weight=0.075,
        ),
        _two_circle_scenario(
            "v22_bt_a040_left",
            alpha_deg=40.0,
            side=-1.0,
            weight=0.075,
        ),
        _two_circle_scenario(
            "v22_bt_a040_right",
            alpha_deg=40.0,
            side=1.0,
            weight=0.075,
        ),
        _two_circle_scenario(
            "v22_bt_a060_left",
            alpha_deg=60.0,
            side=-1.0,
            weight=0.050,
        ),
        _two_circle_scenario(
            "v22_bt_a060_right",
            alpha_deg=60.0,
            side=1.0,
            weight=0.050,
        ),
    ],
    # Small independent perturbations reduce memorization of one BT opening while
    # keeping each geometry recognizable.
    "ownship_randomization": {
        "enabled": True,
        "n_m": 40.0,
        "e_m": 40.0,
        "d_m": 30.0,
        "r_roll": 3.0,
        "r_pitch": 2.0,
        "r_heading": 3.0,
        "speed_mps": 5.0,
    },
    "target_randomization": {
        "enabled": True,
        "n_m": 40.0,
        "e_m": 40.0,
        "d_m": 30.0,
        "r_roll": 3.0,
        "r_pitch": 2.0,
        "r_heading": 3.0,
        "speed_mps": 5.0,
    },
    "shared_randomization": {
        "enabled": True,
        "n_m": 250.0,
        "e_m": 250.0,
        "d_m": 80.0,
        "speed_mps": 3.0,
    },
}


def get_stages() -> list[CurriculumStage]:
    """Return the short mixed general-dogfight bridge stage."""
    return [
        CurriculumStage(
            index=0,
            name="general_dogfight_bridge_mixed_aspect_15s",
            description=(
                "Preserve v21's validated 900 m firing-cone attack while exposing "
                "the policy to active-BT tail chase and mirrored alpha 0/20/40/60 "
                "two-circle-inspired geometries."
            ),
            # Must be BT at stage/base level because the scenario pool contains
            # behavior_tree scenarios; individual rehearsal scenarios may switch
            # episode target_mode back to autopilot.
            target_mode="behavior_tree",
            episode_step_limit=15 * 60,
            max_iterations=2,
            checkpoint_interval=1,
            reward_overrides={},
            randomization={"enabled": False},
            advance_conditions={},
            advance_window=1,
            env_overrides={
                "step_ratio": 6,
                "max_engage_time": 15.0,
                # Competition default is 300 m. Keep it here so merge/recovery is
                # not artificially terminated at v21's 6,550 m tail-chase floor;
                # the v22 reward still strongly discourages low-altitude dives.
                "min_altitude": 300.0,
                "initial_scenario": V22_GENERAL_SCENARIO_POOL,
                "geometry_guard": {
                    "enabled": False,
                },
            },
        )
    ]


__all__ = [
    "V22_GENERAL_SCENARIO_POOL",
    "get_stages",
]
