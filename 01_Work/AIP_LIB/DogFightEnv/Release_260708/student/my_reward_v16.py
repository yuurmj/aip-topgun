# -*- coding: utf-8 -*-
"""PPO v16 reward: preserve attack while learning WEZ entry.

Flight-stability shaping uses only confirmed StateIndex signals:
StateIndex.ALT, StateIndex.ROLL, and StateIndex.PITCH.
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

from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    # Compatibility keys.
    "survival_bonus": 0.0,
    "damage_scale": 40.0,
    "pursuit_scale": 0.05,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 2500.0,
    "step_penalty": -0.01,

    "attack_shaping_enabled": 1.0,

    # Outside-WEZ approach shaping. These are intentionally weaker than the
    # in-WEZ firing rewards so entry alone cannot dominate learning.
    "alignment_scale": 0.03,
    "pre_wez_outer_margin_m": 1400.0,
    "coarse_aim_angle_deg": 30.0,
    "coarse_aim_bonus": 0.05,
    "fine_aim_angle_deg": 10.0,
    "fine_aim_bonus": 0.10,
    "entry_corridor_angle_deg": 18.0,
    "entry_corridor_bonus": 0.08,

    # WEZ and firing-cone shaping.
    "wez_hold_bonus": 0.08,
    "stable_wez_bonus": 0.25,
    "wez_unstable_penalty": 0.45,
    "attack_track_angle_deg": 8.0,
    "attack_track_bonus": 0.55,
    "attack_ready_angle_deg": 5.0,
    "attack_ready_bonus": 0.65,
    "centerline_angle_deg": 3.0,
    "centerline_bonus": 0.80,
    "wez_miss_angle_deg": 25.0,
    "wez_miss_penalty": 1.10,
    "no_fire_in_range_penalty": 0.85,
    "no_fire_full_angle_deg": 45.0,

    # Direct firing/damage signals. target_damage is nonzero only when the
    # aircraft is inside the WEZ range and its ATA is inside the weapon cone.
    "damage_step_bonus": 1.40,
    "weapon_quality_bonus": 0.80,
    "target_damage_scale": 120.0,
    "ownship_damage_scale": 40.0,

    # Encourage a controllable firing range instead of blindly overshooting.
    "preferred_range_min_m": 350.0,
    "preferred_range_peak_m": 550.0,
    "preferred_range_max_m": 820.0,
    "preferred_range_bonus": 0.35,
    "too_close_range_m": 260.0,
    "too_close_penalty": 0.70,
    "close_misaligned_range_m": 450.0,
    "close_misaligned_angle_deg": 8.0,
    "close_misaligned_penalty": 1.20,

    # Broad continuous roll/pitch recovery scores.
    "stable_roll_target_deg": 12.0,
    "stable_roll_recovery_deg": 60.0,
    "stable_pitch_target_deg": 4.0,
    "stable_pitch_recovery_deg": 20.0,

    # Additive stability rewards.
    "roll_stability_bonus": 0.18,
    "pitch_stability_bonus": 0.28,
    "altitude_stability_bonus": 0.24,

    # Altitude-centred reward.
    "target_altitude_m": 7000.0,
    "target_altitude_tolerance_m": 100.0,
    "target_altitude_recovery_m": 1000.0,
    "altitude_center_penalty": 0.38,

    # Stable-WEZ score weights.
    "stable_wez_roll_weight": 0.25,
    "stable_wez_pitch_weight": 0.35,
    "stable_wez_altitude_weight": 0.40,

    "level_flight_bonus": 0.0,

    # Absolute altitude safety.
    "safe_altitude_min_m": 5000.0,
    "safe_altitude_max_m": 8500.0,
    "safe_altitude_bonus": 0.02,
    "high_altitude_full_m": 10000.0,
    "altitude_band_penalty": 0.50,
    "low_altitude_warning_m": 5800.0,
    "low_altitude_m": 3500.0,
    "low_altitude_penalty": 0.90,

    # Attitude protection.
    "roll_penalty_start_deg": 22.0,
    "roll_penalty_full_deg": 70.0,
    "roll_penalty": 0.75,
    "inverted_roll_start_deg": 70.0,
    "inverted_roll_full_deg": 145.0,
    "inverted_roll_penalty": 1.20,
    "level_pitch_start_deg": 7.0,
    "level_pitch_full_deg": 28.0,
    "level_pitch_penalty": 0.55,
    "dive_pitch_start_deg": -3.0,
    "dive_pitch_full_deg": -18.0,
    "dive_pitch_penalty": 0.85,
    "safety_reward_min": -3.00,

    # Terminal rewards.
    "win_reward": 150.0,
    "loss_reward": -180.0,
    "crash_penalty": -320.0,
    "guard_fail_penalty": -70.0,
    "target_crash_reward": 0.0,
    "timeout_penalty": 0.0,
    "draw_reward": 0.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _linear_factor(value: float, start: float, full: float) -> float:
    span = full - start
    if span == 0.0:
        return 1.0 if value == full else 0.0
    return _clamp01((value - start) / span)


def _outside_band_factor(
    absolute_value: float,
    start_abs: float,
    full_abs: float,
) -> float:
    if absolute_value <= start_abs:
        return 0.0
    if full_abs <= start_abs:
        return 1.0
    return _clamp01(
        (absolute_value - start_abs) / (full_abs - start_abs)
    )


def _broad_stability_score(
    absolute_value: float,
    target_abs: float,
    recovery_abs: float,
) -> float:
    value = max(0.0, float(absolute_value))
    target = max(0.0, float(target_abs))
    recovery = max(target, float(recovery_abs))

    if recovery <= 0.0:
        return 1.0 if value == 0.0 else 0.0
    if target <= 0.0:
        return 1.0 - _clamp01(value / recovery)
    if value <= target:
        return 1.0 - 0.20 * _clamp01(value / target)
    if recovery <= target:
        return 0.0

    progress = _clamp01((value - target) / (recovery - target))
    return 0.80 * (1.0 - progress)


def _centered_altitude_score(
    altitude_m: float,
    target_altitude_m: float,
    tolerance_m: float,
    recovery_m: float,
) -> float:
    error_m = abs(float(altitude_m) - float(target_altitude_m))
    tolerance = max(0.0, float(tolerance_m))
    recovery = max(tolerance, float(recovery_m))

    if error_m <= tolerance:
        return 1.0
    if recovery <= tolerance:
        return 0.0
    return 1.0 - _clamp01(
        (error_m - tolerance) / (recovery - tolerance)
    )


def _weighted_stability_score(
    roll_score: float,
    pitch_score: float,
    altitude_score: float,
    reward_config: dict,
) -> float:
    roll_weight = max(
        0.0, float(reward_config.get("stable_wez_roll_weight", 0.25))
    )
    pitch_weight = max(
        0.0, float(reward_config.get("stable_wez_pitch_weight", 0.35))
    )
    altitude_weight = max(
        0.0,
        float(reward_config.get("stable_wez_altitude_weight", 0.40)),
    )
    weight_sum = roll_weight + pitch_weight + altitude_weight
    if weight_sum <= 0.0:
        return 0.0
    return _clamp01(
        (
            roll_weight * roll_score
            + pitch_weight * pitch_score
            + altitude_weight * altitude_score
        )
        / weight_sum
    )


def _aim_factor(ata_deg: float, full_angle_deg: float) -> float:
    if full_angle_deg <= 0.0:
        return 0.0
    return max(0.0, 1.0 - float(ata_deg) / float(full_angle_deg))


def _range_readiness(
    distance_m: float,
    wez_min_m: float,
    wez_max_m: float,
    outer_margin_m: float,
) -> float:
    if wez_min_m < 0.0 or wez_max_m <= wez_min_m:
        return 0.0
    if distance_m < wez_min_m:
        if wez_min_m == 0.0:
            return 0.0
        return _clamp01(distance_m / wez_min_m)
    if distance_m <= wez_max_m:
        return 1.0
    if outer_margin_m <= 0.0:
        return 0.0
    return _clamp01(
        1.0 - (distance_m - wez_max_m) / outer_margin_m
    )


def _preferred_range_score(
    distance_m: float,
    minimum_m: float,
    peak_m: float,
    maximum_m: float,
) -> float:
    """Triangular score: 0 at min/max and 1 at the preferred peak."""
    d = float(distance_m)
    low = float(minimum_m)
    peak = float(peak_m)
    high = float(maximum_m)

    if not (low < peak < high):
        return 0.0
    if d <= low or d >= high:
        return 0.0
    if d <= peak:
        return _clamp01((d - low) / (peak - low))
    return _clamp01((high - d) / (high - peak))


def _altitude_band_factors(
    altitude_m: float,
    safe_min_m: float,
    safe_max_m: float,
    low_full_m: float,
    high_full_m: float,
) -> tuple[float, float]:
    if safe_min_m <= altitude_m <= safe_max_m:
        return 1.0, 0.0

    if altitude_m < safe_min_m:
        span = safe_min_m - low_full_m
        if span <= 0.0:
            return 0.0, 1.0
        outside = _clamp01((safe_min_m - altitude_m) / span)
        return 1.0 - outside, outside

    span = high_full_m - safe_max_m
    if span <= 0.0:
        return 0.0, 1.0
    outside = _clamp01((altitude_m - safe_max_m) / span)
    return 1.0 - outside, outside


def compute_reward(
    ownship_state,
    target_state,
    ownship_damage: float,
    target_damage: float,
    geo_info,
    wez_config: dict,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> tuple[float, dict]:
    """Compute v16 entry-preserving attack reward."""
    distance_m = float(
        geo_info._get_distance(ownship_state, target_state)
    )
    ata_deg = abs(
        float(
            geo_info._get_antenna_train_angle(
                ownship_state,
                target_state,
                False,
            )
        )
    )

    # Only confirmed stability indices are used.
    altitude_m = float(ownship_state[StateIndex.ALT])
    roll_deg = abs(float(ownship_state[StateIndex.ROLL]))
    pitch_deg = float(ownship_state[StateIndex.PITCH])

    roll_stability = _broad_stability_score(
        roll_deg,
        float(reward_config.get("stable_roll_target_deg", 12.0)),
        float(reward_config.get("stable_roll_recovery_deg", 60.0)),
    )
    pitch_stability = _broad_stability_score(
        abs(pitch_deg),
        float(reward_config.get("stable_pitch_target_deg", 4.0)),
        float(reward_config.get("stable_pitch_recovery_deg", 20.0)),
    )
    altitude_center = _centered_altitude_score(
        altitude_m,
        float(reward_config.get("target_altitude_m", 7000.0)),
        float(reward_config.get("target_altitude_tolerance_m", 100.0)),
        float(reward_config.get("target_altitude_recovery_m", 1000.0)),
    )
    level_stability = _weighted_stability_score(
        roll_stability,
        pitch_stability,
        altitude_center,
        reward_config,
    )

    safe_min_m = float(reward_config.get("safe_altitude_min_m", 5000.0))
    safe_max_m = float(reward_config.get("safe_altitude_max_m", 8500.0))
    low_full_m = float(reward_config.get("low_altitude_m", 3500.0))
    high_full_m = float(
        reward_config.get("high_altitude_full_m", 10000.0)
    )
    _, altitude_outside = _altitude_band_factors(
        altitude_m,
        safe_min_m,
        safe_max_m,
        low_full_m,
        high_full_m,
    )

    survival_reward = float(reward_config.get("survival_bonus", 0.0))
    step_reward = float(reward_config.get("step_penalty", -0.01))
    attack_enabled = (
        float(reward_config.get("attack_shaping_enabled", 1.0)) > 0.0
    )

    pursuit_reward = 0.0
    damage_reward = 0.0
    close_penalty = 0.0

    if attack_enabled:
        wez_min_m = float(wez_config["min_range_m"])
        wez_max_m = float(wez_config["max_range_m"])
        wez_half_angle_deg = float(wez_config["angle_deg"]) / 2.0
        in_wez_range = wez_min_m <= distance_m <= wez_max_m
        firing_now = float(target_damage) > 0.0

        range_readiness = _range_readiness(
            distance_m,
            wez_min_m,
            wez_max_m,
            float(reward_config.get("pre_wez_outer_margin_m", 1400.0)),
        )

        alignment_reward = float(
            reward_config.get("alignment_scale", 0.03)
        ) * math.cos(math.radians(ata_deg))
        coarse_aim_reward = (
            float(reward_config.get("coarse_aim_bonus", 0.05))
            * _aim_factor(
                ata_deg,
                float(reward_config.get("coarse_aim_angle_deg", 30.0)),
            )
            * range_readiness
        )
        fine_aim_reward = (
            float(reward_config.get("fine_aim_bonus", 0.10))
            * _aim_factor(
                ata_deg,
                float(reward_config.get("fine_aim_angle_deg", 10.0)),
            )
            * range_readiness
        )

        entry_corridor_reward = 0.0
        if distance_m > wez_max_m:
            entry_corridor_reward = (
                float(reward_config.get("entry_corridor_bonus", 0.08))
                * range_readiness
                * _aim_factor(
                    ata_deg,
                    float(
                        reward_config.get(
                            "entry_corridor_angle_deg",
                            18.0,
                        )
                    ),
                )
                * level_stability
            )

        wez_hold_reward = 0.0
        stable_wez_reward = 0.0
        track_reward = 0.0
        attack_ready_reward = 0.0
        centerline_reward = 0.0
        preferred_range_reward = 0.0
        weapon_quality_reward = 0.0
        firing_step_reward = 0.0
        no_fire_reward = 0.0
        wez_unstable_reward = 0.0
        wez_miss_reward = 0.0

        if in_wez_range:
            track_score = _aim_factor(
                ata_deg,
                float(reward_config.get("attack_track_angle_deg", 8.0)),
            )
            ready_score = _aim_factor(
                ata_deg,
                float(reward_config.get("attack_ready_angle_deg", 5.0)),
            )
            centerline_score = _aim_factor(
                ata_deg,
                float(reward_config.get("centerline_angle_deg", 3.0)),
            )

            track_reward = (
                float(reward_config.get("attack_track_bonus", 0.55))
                * track_score
                * level_stability
            )
            attack_ready_reward = (
                float(reward_config.get("attack_ready_bonus", 0.65))
                * ready_score
                * level_stability
            )
            centerline_reward = (
                float(reward_config.get("centerline_bonus", 0.80))
                * centerline_score
                * level_stability
            )

            preferred_range_reward = (
                float(reward_config.get("preferred_range_bonus", 0.35))
                * _preferred_range_score(
                    distance_m,
                    float(
                        reward_config.get("preferred_range_min_m", 350.0)
                    ),
                    float(
                        reward_config.get("preferred_range_peak_m", 550.0)
                    ),
                    float(
                        reward_config.get("preferred_range_max_m", 820.0)
                    ),
                )
                * max(track_score, ready_score)
                * level_stability
            )

            if firing_now:
                wez_hold_reward = float(
                    reward_config.get("wez_hold_bonus", 0.08)
                )
                stable_wez_reward = float(
                    reward_config.get("stable_wez_bonus", 0.25)
                ) * level_stability
                firing_step_reward = float(
                    reward_config.get("damage_step_bonus", 1.40)
                )

                range_damage_score = 0.0
                range_span = wez_max_m - wez_min_m
                if range_span > 0.0:
                    range_damage_score = _clamp01(
                        (wez_max_m - distance_m) / range_span
                    )
                cone_score = _aim_factor(
                    ata_deg,
                    max(wez_half_angle_deg, 1.0),
                )
                weapon_quality_reward = (
                    float(
                        reward_config.get("weapon_quality_bonus", 0.80)
                    )
                    * (0.30 + 0.70 * range_damage_score)
                    * cone_score
                    * level_stability
                )
                wez_unstable_reward = -float(
                    reward_config.get("wez_unstable_penalty", 0.45)
                ) * (1.0 - level_stability)
            else:
                no_fire_angle = float(
                    reward_config.get("no_fire_full_angle_deg", 45.0)
                )
                angle_factor = (
                    _clamp01(ata_deg / no_fire_angle)
                    if no_fire_angle > 0.0
                    else 1.0
                )
                no_fire_reward = -float(
                    reward_config.get("no_fire_in_range_penalty", 0.85)
                ) * (0.25 + 0.75 * angle_factor)

                miss_span = float(
                    reward_config.get("wez_miss_angle_deg", 25.0)
                )
                if ata_deg > wez_half_angle_deg:
                    miss_factor = (
                        _clamp01(
                            (ata_deg - wez_half_angle_deg) / miss_span
                        )
                        if miss_span > 0.0
                        else 1.0
                    )
                    wez_miss_reward = -float(
                        reward_config.get("wez_miss_penalty", 1.10)
                    ) * miss_factor

        if distance_m < float(
            reward_config.get("too_close_range_m", 260.0)
        ):
            close_penalty -= float(
                reward_config.get("too_close_penalty", 0.70)
            )

        if (
            distance_m
            < float(
                reward_config.get("close_misaligned_range_m", 450.0)
            )
            and ata_deg
            > float(
                reward_config.get("close_misaligned_angle_deg", 8.0)
            )
        ):
            close_penalty -= float(
                reward_config.get("close_misaligned_penalty", 1.20)
            )

        pursuit_reward = (
            alignment_reward
            + coarse_aim_reward
            + fine_aim_reward
            + entry_corridor_reward
            + wez_hold_reward
            + stable_wez_reward
            + track_reward
            + attack_ready_reward
            + centerline_reward
            + preferred_range_reward
            + weapon_quality_reward
            + firing_step_reward
            + no_fire_reward
            + wez_unstable_reward
            + wez_miss_reward
        )

        damage_reward = (
            float(reward_config.get("target_damage_scale", 120.0))
            * float(target_damage)
            - float(reward_config.get("ownship_damage_scale", 40.0))
            * float(ownship_damage)
        )

    # Additive flight stability.
    safety_reward = (
        float(reward_config.get("roll_stability_bonus", 0.18))
        * roll_stability
        + float(reward_config.get("pitch_stability_bonus", 0.28))
        * pitch_stability
        + float(reward_config.get("altitude_stability_bonus", 0.24))
        * altitude_center
    )
    safety_reward -= float(
        reward_config.get("altitude_center_penalty", 0.38)
    ) * (1.0 - altitude_center)

    if safe_min_m <= altitude_m <= safe_max_m:
        safety_reward += float(
            reward_config.get("safe_altitude_bonus", 0.02)
        )

    safety_reward += float(
        reward_config.get("level_flight_bonus", 0.0)
    ) * level_stability
    safety_reward -= float(
        reward_config.get("altitude_band_penalty", 0.50)
    ) * altitude_outside

    warning_m = float(
        reward_config.get("low_altitude_warning_m", 5800.0)
    )
    if altitude_m < warning_m:
        if warning_m > low_full_m:
            low_factor = _clamp01(
                (warning_m - altitude_m) / (warning_m - low_full_m)
            )
        else:
            low_factor = 1.0
        safety_reward -= float(
            reward_config.get("low_altitude_penalty", 0.90)
        ) * low_factor

    roll_factor = _outside_band_factor(
        roll_deg,
        float(reward_config.get("roll_penalty_start_deg", 22.0)),
        float(reward_config.get("roll_penalty_full_deg", 70.0)),
    )
    safety_reward -= float(
        reward_config.get("roll_penalty", 0.75)
    ) * roll_factor

    inverted_factor = _outside_band_factor(
        roll_deg,
        float(reward_config.get("inverted_roll_start_deg", 70.0)),
        float(reward_config.get("inverted_roll_full_deg", 145.0)),
    )
    safety_reward -= float(
        reward_config.get("inverted_roll_penalty", 1.20)
    ) * inverted_factor

    pitch_factor = _outside_band_factor(
        abs(pitch_deg),
        float(reward_config.get("level_pitch_start_deg", 7.0)),
        float(reward_config.get("level_pitch_full_deg", 28.0)),
    )
    safety_reward -= float(
        reward_config.get("level_pitch_penalty", 0.55)
    ) * pitch_factor

    dive_start_deg = float(
        reward_config.get("dive_pitch_start_deg", -3.0)
    )
    dive_full_deg = float(
        reward_config.get("dive_pitch_full_deg", -18.0)
    )
    if pitch_deg < dive_start_deg:
        safety_reward -= float(
            reward_config.get("dive_pitch_penalty", 0.85)
        ) * _linear_factor(
            pitch_deg,
            dive_start_deg,
            dive_full_deg,
        )

    safety_reward = max(
        float(reward_config.get("safety_reward_min", -3.00)),
        safety_reward,
    )

    terminal_reward = 0.0
    if terminated:
        if end_condition == "target destroyed":
            terminal_reward = float(
                reward_config.get("win_reward", 150.0)
            )
        elif end_condition == "target altitude below min":
            terminal_reward = float(
                reward_config.get("target_crash_reward", 0.0)
            )
        elif end_condition == "ownship destroyed":
            terminal_reward = float(
                reward_config.get("loss_reward", -180.0)
            )
        elif end_condition in (
            "ownship altitude below min",
            "FDM Update Fail",
            "Ownship FDM output Fall",
        ):
            terminal_reward = float(
                reward_config.get("crash_penalty", -320.0)
            )
        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(
                reward_config.get("guard_fail_penalty", -70.0)
            )
        else:
            terminal_reward = float(
                reward_config.get("draw_reward", 0.0)
            )
    elif truncated:
        terminal_reward = float(
            reward_config.get("timeout_penalty", 0.0)
        )

    components = {
        "pursuit": float(pursuit_reward + close_penalty),
        "damage": float(damage_reward),
        "safety": float(safety_reward),
        "survival": float(survival_reward),
    }

    total_reward = (
        survival_reward
        + step_reward
        + pursuit_reward
        + damage_reward
        + safety_reward
        + close_penalty
        + terminal_reward
    )
    return float(total_reward), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
