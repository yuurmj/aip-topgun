# -*- coding: utf-8 -*-
"""PPO v17 curriculum: progress attack distance in 100 m steps.

The exact best v16 policy at total_iter=22 was not saved as a checkpoint.
Therefore v17 starts from the best reliably saved attack-capable bundle
(v15 Stage 1), saves a native checkpoint every iteration, and raises the
starting distance gradually: 900 -> 1000 -> 1100 -> 1200 -> 1300 m.
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


def _autopilot_scenario(
    name: str,
    distance_m: float,
    ownship_speed_mps: float,
    *,
    weight: float = 1.0,
    east_m: float = 0.0,
    heading_deg: float = 180.0,
    target_speed_mps: float = 250.0,
) -> dict:
    """Create a same-direction pursuit scenario behind the target."""
    target_north_m = 6000.0
    return {
        "name": name,
        "weight": float(weight),
        "ownship": [
            target_north_m + float(distance_m),
            float(east_m),
            -7000.0,
            0.0,
            0.0,
            float(heading_deg),
            float(ownship_speed_mps),
        ],
        "target": [
            target_north_m,
            0.0,
            -7000.0,
            0.0,
            0.0,
            180.0,
            float(target_speed_mps),
        ],
        "target_mode": "autopilot",
        "target_autopilot": {
            "heading_cmd": 180.0,
            "altitude_cmd": -7000.0,
            "speed_cmd": float(target_speed_mps),
        },
    }


def _distance_variants(
    prefix: str,
    distance_m: float,
    ownship_speed_mps: float,
    *,
    weight: float,
) -> list[dict]:
    """Three small geometry variants to reduce one-scenario memorization."""
    return [
        _autopilot_scenario(
            f"{prefix}_{int(distance_m)}m_center",
            distance_m,
            ownship_speed_mps,
            weight=weight,
        ),
        _autopilot_scenario(
            f"{prefix}_{int(distance_m)}m_left",
            distance_m,
            ownship_speed_mps,
            weight=weight,
            east_m=-20.0,
            heading_deg=179.5,
        ),
        _autopilot_scenario(
            f"{prefix}_{int(distance_m)}m_right",
            distance_m,
            ownship_speed_mps,
            weight=weight,
            east_m=20.0,
            heading_deg=180.5,
        ),
    ]


def _pool(
    current_distance_m: float,
    current_speed_mps: float,
    *,
    previous_distance_m: float | None = None,
    previous_speed_mps: float | None = None,
) -> dict:
    """Current distance dominates; the immediately previous distance is rehearsed."""
    scenarios = _distance_variants(
        "v17_current",
        current_distance_m,
        current_speed_mps,
        weight=1.0,
    )
    if previous_distance_m is not None and previous_speed_mps is not None:
        scenarios += _distance_variants(
            "v17_previous",
            previous_distance_m,
            previous_speed_mps,
            weight=0.35,
        )
    return {"mode": "scenario_pool", "scenarios": scenarios}


ANCHOR_POOL = {
    "mode": "scenario_pool",
    "scenarios": (
        _distance_variants("v17_anchor", 750.0, 250.0, weight=1.0)
        + _distance_variants("v17_anchor", 800.0, 250.0, weight=1.0)
        + _distance_variants("v17_anchor", 850.0, 250.0, weight=1.0)
    ),
}

STAGE_900_POOL = _pool(
    900.0,
    250.0,
    previous_distance_m=850.0,
    previous_speed_mps=250.0,
)
STAGE_1000_POOL = _pool(
    1000.0,
    265.0,
    previous_distance_m=900.0,
    previous_speed_mps=250.0,
)
STAGE_1100_POOL = _pool(
    1100.0,
    270.0,
    previous_distance_m=1000.0,
    previous_speed_mps=265.0,
)
STAGE_1200_POOL = _pool(
    1200.0,
    275.0,
    previous_distance_m=1100.0,
    previous_speed_mps=270.0,
)
STAGE_1300_POOL = _pool(
    1300.0,
    280.0,
    previous_distance_m=1200.0,
    previous_speed_mps=275.0,
)


COMMON_REWARD_OVERRIDES = {
    "attack_shaping_enabled": 1.0,
    "survival_bonus": 0.01,
    "step_penalty": -0.01,

    # Outside-WEZ shaping is deliberately small. Entry alone must not dominate.
    "alignment_scale": 0.04,
    "pre_wez_outer_margin_m": 700.0,
    "coarse_aim_angle_deg": 30.0,
    "coarse_aim_bonus": 0.03,
    "fine_aim_angle_deg": 10.0,
    "fine_aim_bonus": 0.08,
    "entry_corridor_angle_deg": 12.0,
    "entry_corridor_bonus": 0.03,
    "far_misaligned_angle_deg": 25.0,
    "far_misaligned_full_deg": 70.0,
    "far_misaligned_penalty": 1.10,

    # Attack must remain centred and stable inside the WEZ.
    "wez_hold_bonus": 0.02,
    "stable_wez_bonus": 0.20,
    "wez_unstable_penalty": 0.90,
    "attack_track_angle_deg": 8.0,
    "attack_track_bonus": 0.75,
    "attack_ready_angle_deg": 5.0,
    "attack_ready_bonus": 1.00,
    "centerline_angle_deg": 3.0,
    "centerline_bonus": 1.20,
    "wez_miss_angle_deg": 25.0,
    "wez_miss_penalty": 1.80,
    "no_fire_in_range_penalty": 1.20,
    "no_fire_full_angle_deg": 45.0,
    "damage_step_bonus": 1.60,
    "weapon_quality_bonus": 1.00,
    "target_damage_scale": 150.0,
    "ownship_damage_scale": 40.0,

    # Avoid overshoot after entry.
    "preferred_range_min_m": 400.0,
    "preferred_range_peak_m": 600.0,
    "preferred_range_max_m": 800.0,
    "preferred_range_bonus": 0.50,
    "too_close_range_m": 300.0,
    "too_close_penalty": 1.40,
    "close_misaligned_range_m": 500.0,
    "close_misaligned_angle_deg": 6.0,
    "close_misaligned_penalty": 2.20,

    # Only confirmed StateIndex.ALT/ROLL/PITCH are used.
    "stable_roll_target_deg": 10.0,
    "stable_roll_recovery_deg": 45.0,
    "stable_pitch_target_deg": 3.0,
    "stable_pitch_recovery_deg": 15.0,
    "roll_stability_bonus": 0.25,
    "pitch_stability_bonus": 0.35,
    "altitude_stability_bonus": 0.35,
    "target_altitude_m": 7000.0,
    "target_altitude_tolerance_m": 80.0,
    "target_altitude_recovery_m": 500.0,
    "altitude_center_penalty": 0.80,
    "stable_wez_roll_weight": 0.25,
    "stable_wez_pitch_weight": 0.35,
    "stable_wez_altitude_weight": 0.40,
    "level_flight_bonus": 0.0,

    # Strong safety protection.
    "safe_altitude_min_m": 6700.0,
    "safe_altitude_max_m": 7400.0,
    "safe_altitude_bonus": 0.05,
    "altitude_band_penalty": 1.20,
    "low_altitude_warning_m": 6850.0,
    "low_altitude_m": 6500.0,
    "low_altitude_penalty": 2.00,
    "roll_penalty_start_deg": 15.0,
    "roll_penalty_full_deg": 45.0,
    "roll_penalty": 1.80,
    "inverted_roll_start_deg": 45.0,
    "inverted_roll_full_deg": 80.0,
    "inverted_roll_penalty": 3.50,
    "catastrophic_roll_start_deg": 55.0,
    "catastrophic_roll_full_deg": 90.0,
    "catastrophic_roll_penalty": 5.00,
    "level_pitch_start_deg": 5.0,
    "level_pitch_full_deg": 18.0,
    "level_pitch_penalty": 1.20,
    "dive_pitch_start_deg": -2.0,
    "dive_pitch_full_deg": -12.0,
    "dive_pitch_penalty": 1.80,
    "safety_reward_min": -8.00,

    "win_reward": 180.0,
    "loss_reward": -220.0,
    "crash_penalty": -500.0,
    "target_crash_reward": 0.0,
    "timeout_penalty": 0.0,
    "draw_reward": 0.0,
}


def _stage_rewards(
    *,
    damage_scale: float,
    damage_step_bonus: float,
    no_fire_penalty: float,
) -> dict:
    return {
        **COMMON_REWARD_OVERRIDES,
        "target_damage_scale": float(damage_scale),
        "damage_step_bonus": float(damage_step_bonus),
        "no_fire_in_range_penalty": float(no_fire_penalty),
    }


def _env_overrides(max_engage_time: float, scenario_pool: dict) -> dict:
    return {
        "step_ratio": 6,
        "max_engage_time": float(max_engage_time),
        # Starting at 7,000 m, falling below 6,700 m ends the episode.
        # This implements the requested 300 m altitude-loss hard boundary
        # without modifying train_curriculum.py or environment source files.
        "min_altitude": 6700.0,
        "target_autopilot": {
            "heading_cmd": 180.0,
            "altitude_cmd": -7000.0,
            "speed_cmd": 250.0,
        },
        "initial_scenario": scenario_pool,
    }


def _make_stage(
    *,
    index: int,
    name: str,
    description: str,
    seconds: float,
    iterations: int,
    pool: dict,
    damage_scale: float,
    damage_step_bonus: float,
    no_fire_penalty: float,
) -> CurriculumStage:
    return CurriculumStage(
        index=index,
        name=name,
        description=description,
        target_mode="autopilot",
        episode_step_limit=int(seconds * 60),
        max_iterations=iterations,
        # Save every policy version. v16's best point was lost because the
        # checkpoint interval was four iterations.
        checkpoint_interval=1,
        reward_overrides=_stage_rewards(
            damage_scale=damage_scale,
            damage_step_bonus=damage_step_bonus,
            no_fire_penalty=no_fire_penalty,
        ),
        randomization={"enabled": False},
        # win_rate/WEZ metrics are currently n/a in the common logger, so an
        # automatic success gate would be unreliable. Review replay results at
        # each stage boundary and keep the best per-iteration checkpoint.
        advance_conditions={},
        advance_window=1,
        env_overrides=_env_overrides(seconds, pool),
    )


def get_stages() -> list[CurriculumStage]:
    """Return a 46-iteration progressive attack curriculum."""
    return [
        _make_stage(
            index=0,
            name="attack_anchor_750_850m_8s",
            description="Re-establish stable firing at 750-850 m.",
            seconds=8.0,
            iterations=6,
            pool=ANCHOR_POOL,
            damage_scale=125.0,
            damage_step_bonus=1.40,
            no_fire_penalty=0.90,
        ),
        _make_stage(
            index=1,
            name="edge_attack_900m_10s",
            description="Attack from the outer WEZ edge at 900 m.",
            seconds=10.0,
            iterations=6,
            pool=STAGE_900_POOL,
            damage_scale=135.0,
            damage_step_bonus=1.50,
            no_fire_penalty=1.00,
        ),
        _make_stage(
            index=2,
            name="entry_attack_1000m_10s",
            description="Enter from 1,000 m and keep the target centred.",
            seconds=10.0,
            iterations=8,
            pool=STAGE_1000_POOL,
            damage_scale=145.0,
            damage_step_bonus=1.60,
            no_fire_penalty=1.10,
        ),
        _make_stage(
            index=3,
            name="entry_attack_1100m_12s",
            description="Enter from 1,100 m and transition directly to fire.",
            seconds=12.0,
            iterations=8,
            pool=STAGE_1100_POOL,
            damage_scale=155.0,
            damage_step_bonus=1.70,
            no_fire_penalty=1.20,
        ),
        _make_stage(
            index=4,
            name="entry_attack_1200m_14s",
            description="Enter from 1,200 m without losing altitude or ATA.",
            seconds=14.0,
            iterations=8,
            pool=STAGE_1200_POOL,
            damage_scale=165.0,
            damage_step_bonus=1.80,
            no_fire_penalty=1.30,
        ),
        _make_stage(
            index=5,
            name="entry_attack_1300m_15s",
            description="Final 1,300 m entry, stable WEZ hold, and firing.",
            seconds=15.0,
            iterations=10,
            pool=STAGE_1300_POOL,
            damage_scale=175.0,
            damage_step_bonus=1.90,
            no_fire_penalty=1.40,
        ),
    ]


__all__ = [
    "ANCHOR_POOL",
    "STAGE_900_POOL",
    "STAGE_1000_POOL",
    "STAGE_1100_POOL",
    "STAGE_1200_POOL",
    "STAGE_1300_POOL",
    "COMMON_REWARD_OVERRIDES",
    "get_stages",
]
