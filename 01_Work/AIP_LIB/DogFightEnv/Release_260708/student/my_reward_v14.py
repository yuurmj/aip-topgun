# -*- coding: utf-8 -*-
"""PPO v14 reward: broad additive flight stability + WEZ entry/hold shaping.

Only confirmed StateIndex signals are used for flight stability:
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
    # Compatibility keys preserved for training records and older configs.
    "survival_bonus": 0.0,
    "damage_scale": 40.0,
    "pursuit_scale": 0.05,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 2500.0,
    "step_penalty": -0.01,

    # Stage switch.
    "attack_shaping_enabled": 1.0,

    # Target alignment and pre-WEZ gradients.
    "alignment_scale": 0.05,
    "pre_wez_outer_margin_m": 1500.0,
    "coarse_aim_angle_deg": 30.0,
    "coarse_aim_bonus": 0.08,
    "fine_aim_angle_deg": 10.0,
    "fine_aim_bonus": 0.12,

    # Real WEZ and stable-WEZ shaping.
    "wez_hold_bonus": 0.35,
    "stable_wez_bonus": 0.65,
    "wez_unstable_penalty": 0.35,
    "wez_miss_angle_deg": 10.0,
    "wez_miss_penalty": 0.60,
    "target_damage_scale": 20.0,
    "ownship_damage_scale": 25.0,

    # Broad, continuous stability scores.
    # Score remains positive outside 15 deg roll / 5 deg pitch and falls
    # continuously to zero at the recovery boundary.
    "stable_roll_target_deg": 15.0,
    "stable_roll_recovery_deg": 60.0,
    "stable_pitch_target_deg": 5.0,
    "stable_pitch_recovery_deg": 25.0,

    # Additive rewards: these are added independently, never multiplied.
    "roll_stability_bonus": 0.20,
    "pitch_stability_bonus": 0.20,
    "altitude_stability_bonus": 0.08,

    # Used only to normalize the stable-WEZ quality score.
    "stable_wez_roll_weight": 0.40,
    "stable_wez_pitch_weight": 0.40,
    "stable_wez_altitude_weight": 0.20,

    # Optional compatibility bonus for old YAML files. v14 stages keep it 0.
    "level_flight_bonus": 0.0,

    # Altitude stability.
    "safe_altitude_min_m": 5000.0,
    "safe_altitude_max_m": 8000.0,
    "safe_altitude_bonus": 0.03,
    "high_altitude_full_m": 10000.0,
    "altitude_band_penalty": 0.35,
    "low_altitude_warning_m": 5500.0,
    "low_altitude_m": 3500.0,
    "low_altitude_penalty": 0.60,

    # Wide, weak attitude penalties. Positive recovery gradients above are
    # the primary signal; penalties only become strong for large deviations.
    "roll_penalty_start_deg": 30.0,
    "roll_penalty_full_deg": 90.0,
    "roll_penalty": 0.25,
    "level_pitch_start_deg": 12.0,
    "level_pitch_full_deg": 40.0,
    "level_pitch_penalty": 0.25,
    "dive_pitch_start_deg": -12.0,
    "dive_pitch_full_deg": -40.0,
    "dive_pitch_penalty": 0.30,
    "safety_reward_min": -1.50,

    # Bad close pass.
    "bad_close_range_m": 300.0,
    "bad_close_ata_deg": 30.0,
    "bad_close_penalty": 0.30,

    # Terminal rewards.
    "win_reward": 100.0,
    "loss_reward": -150.0,
    "crash_penalty": -250.0,
    "guard_fail_penalty": -50.0,
    "target_crash_reward": 0.0,
    "timeout_penalty": 0.0,
    "draw_reward": 0.0,
}


def _clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


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
    """Return a continuous [0, 1] score with a broad recovery region.

    0 -> 1.0
    target_abs -> 0.8
    recovery_abs -> 0.0

    This avoids the v13 behavior where the score became zero immediately at
    15 deg roll or 5 deg pitch.
    """
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

    recovery_progress = _clamp01(
        (value - target) / (recovery - target)
    )
    return 0.80 * (1.0 - recovery_progress)


def _weighted_stability_score(
    roll_score: float,
    pitch_score: float,
    altitude_score: float,
    reward_config: dict,
) -> float:
    """Combine independent stability scores additively and normalize."""
    roll_weight = max(
        0.0,
        float(reward_config.get("stable_wez_roll_weight", 0.40)),
    )
    pitch_weight = max(
        0.0,
        float(reward_config.get("stable_wez_pitch_weight", 0.40)),
    )
    altitude_weight = max(
        0.0,
        float(reward_config.get("stable_wez_altitude_weight", 0.20)),
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
    return max(0.0, 1.0 - ata_deg / full_angle_deg)


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


def _altitude_factors(
    altitude_m: float,
    safe_min_m: float,
    safe_max_m: float,
    low_full_m: float,
    high_full_m: float,
) -> tuple[float, float]:
    """Return altitude stability and outside-band penalty factors."""
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
    """Compute v14 broad additive stability and WEZ reward."""
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

    # Only confirmed indices are used for stability shaping.
    altitude_m = float(ownship_state[StateIndex.ALT])
    roll_deg = abs(float(ownship_state[StateIndex.ROLL]))
    pitch_deg = float(ownship_state[StateIndex.PITCH])

    roll_stability = _broad_stability_score(
        roll_deg,
        float(reward_config.get("stable_roll_target_deg", 15.0)),
        float(reward_config.get("stable_roll_recovery_deg", 60.0)),
    )
    pitch_stability = _broad_stability_score(
        abs(pitch_deg),
        float(reward_config.get("stable_pitch_target_deg", 5.0)),
        float(reward_config.get("stable_pitch_recovery_deg", 25.0)),
    )

    safe_min_m = float(reward_config.get("safe_altitude_min_m", 5000.0))
    safe_max_m = float(reward_config.get("safe_altitude_max_m", 8000.0))
    low_full_m = float(reward_config.get("low_altitude_m", 3500.0))
    high_full_m = float(reward_config.get("high_altitude_full_m", 10000.0))

    altitude_stability, altitude_outside = _altitude_factors(
        altitude_m=altitude_m,
        safe_min_m=safe_min_m,
        safe_max_m=safe_max_m,
        low_full_m=low_full_m,
        high_full_m=high_full_m,
    )

    # Additive, normalized quality score used only inside stable-WEZ shaping.
    level_stability = _weighted_stability_score(
        roll_score=roll_stability,
        pitch_score=pitch_stability,
        altitude_score=altitude_stability,
        reward_config=reward_config,
    )

    survival_reward = float(reward_config.get("survival_bonus", 0.0))
    step_reward = float(reward_config.get("step_penalty", -0.01))
    attack_enabled = (
        float(reward_config.get("attack_shaping_enabled", 1.0)) > 0.0
    )

    pursuit_reward = 0.0
    damage_reward = 0.0

    if attack_enabled:
        alignment_reward = float(
            reward_config.get("alignment_scale", 0.05)
        ) * math.cos(math.radians(ata_deg))

        wez_min_m = float(wez_config["min_range_m"])
        wez_max_m = float(wez_config["max_range_m"])
        wez_half_angle_deg = float(wez_config["angle_deg"]) / 2.0

        range_readiness = _range_readiness(
            distance_m=distance_m,
            wez_min_m=wez_min_m,
            wez_max_m=wez_max_m,
            outer_margin_m=float(
                reward_config.get("pre_wez_outer_margin_m", 1500.0)
            ),
        )

        coarse_aim_reward = (
            float(reward_config.get("coarse_aim_bonus", 0.08))
            * _aim_factor(
                ata_deg,
                float(reward_config.get("coarse_aim_angle_deg", 30.0)),
            )
            * range_readiness
        )
        fine_aim_reward = (
            float(reward_config.get("fine_aim_bonus", 0.12))
            * _aim_factor(
                ata_deg,
                float(reward_config.get("fine_aim_angle_deg", 10.0)),
            )
            * range_readiness
        )

        in_wez_range = wez_min_m <= distance_m <= wez_max_m
        in_wez = in_wez_range and ata_deg <= wez_half_angle_deg

        direct_wez_reward = 0.0
        stable_wez_reward = 0.0
        wez_unstable_reward = 0.0
        wez_miss_reward = 0.0

        if in_wez:
            # Direct WEZ bonus is intentionally modest. The stable quality
            # bonus remains additive and is not a roll*pitch product.
            direct_wez_reward = float(
                reward_config.get("wez_hold_bonus", 0.35)
            )
            stable_wez_reward = float(
                reward_config.get("stable_wez_bonus", 0.65)
            ) * level_stability
            wez_unstable_reward = -float(
                reward_config.get("wez_unstable_penalty", 0.35)
            ) * (1.0 - level_stability)

        elif in_wez_range:
            miss_angle_deg = float(
                reward_config.get("wez_miss_angle_deg", 10.0)
            )
            if miss_angle_deg > 0.0:
                miss_factor = _clamp01(
                    (ata_deg - wez_half_angle_deg) / miss_angle_deg
                )
            else:
                miss_factor = 1.0
            wez_miss_reward = -float(
                reward_config.get("wez_miss_penalty", 0.60)
            ) * miss_factor

        pursuit_reward = (
            alignment_reward
            + coarse_aim_reward
            + fine_aim_reward
            + direct_wez_reward
            + stable_wez_reward
            + wez_unstable_reward
            + wez_miss_reward
        )

        damage_reward = (
            float(reward_config.get("target_damage_scale", 20.0))
            * float(target_damage)
            - float(reward_config.get("ownship_damage_scale", 25.0))
            * float(ownship_damage)
        )

    # Independent additive flight-stability rewards.
    safety_reward = (
        float(reward_config.get("roll_stability_bonus", 0.20))
        * roll_stability
        + float(reward_config.get("pitch_stability_bonus", 0.20))
        * pitch_stability
        + float(reward_config.get("altitude_stability_bonus", 0.08))
        * altitude_stability
    )

    if safe_min_m <= altitude_m <= safe_max_m:
        safety_reward += float(
            reward_config.get("safe_altitude_bonus", 0.03)
        )

    # Kept for backward compatibility, but v14 curriculum sets it to zero.
    safety_reward += float(
        reward_config.get("level_flight_bonus", 0.0)
    ) * level_stability

    safety_reward -= float(
        reward_config.get("altitude_band_penalty", 0.35)
    ) * altitude_outside

    warning_m = float(
        reward_config.get("low_altitude_warning_m", 5500.0)
    )
    if altitude_m < warning_m:
        if warning_m > low_full_m:
            low_factor = _clamp01(
                (warning_m - altitude_m) / (warning_m - low_full_m)
            )
        else:
            low_factor = 1.0
        safety_reward -= float(
            reward_config.get("low_altitude_penalty", 0.60)
        ) * low_factor

    roll_factor = _outside_band_factor(
        absolute_value=roll_deg,
        start_abs=float(
            reward_config.get("roll_penalty_start_deg", 30.0)
        ),
        full_abs=float(
            reward_config.get("roll_penalty_full_deg", 90.0)
        ),
    )
    safety_reward -= float(
        reward_config.get("roll_penalty", 0.25)
    ) * roll_factor

    pitch_factor = _outside_band_factor(
        absolute_value=abs(pitch_deg),
        start_abs=float(
            reward_config.get("level_pitch_start_deg", 12.0)
        ),
        full_abs=float(
            reward_config.get("level_pitch_full_deg", 40.0)
        ),
    )
    safety_reward -= float(
        reward_config.get("level_pitch_penalty", 0.25)
    ) * pitch_factor

    dive_start_deg = float(
        reward_config.get("dive_pitch_start_deg", -12.0)
    )
    dive_full_deg = float(
        reward_config.get("dive_pitch_full_deg", -40.0)
    )
    if pitch_deg < dive_start_deg:
        safety_reward -= float(
            reward_config.get("dive_pitch_penalty", 0.30)
        ) * _linear_factor(
            value=pitch_deg,
            start=dive_start_deg,
            full=dive_full_deg,
        )

    safety_reward = max(
        float(reward_config.get("safety_reward_min", -1.50)),
        safety_reward,
    )

    close_range_penalty = 0.0
    if (
        attack_enabled
        and distance_m
        < float(reward_config.get("bad_close_range_m", 300.0))
        and ata_deg
        > float(reward_config.get("bad_close_ata_deg", 30.0))
    ):
        close_range_penalty = -float(
            reward_config.get("bad_close_penalty", 0.30)
        )

    terminal_reward = 0.0
    if terminated:
        if end_condition == "target destroyed":
            terminal_reward = float(
                reward_config.get("win_reward", 100.0)
            )
        elif end_condition == "target altitude below min":
            terminal_reward = float(
                reward_config.get("target_crash_reward", 0.0)
            )
        elif end_condition == "ownship destroyed":
            terminal_reward = float(
                reward_config.get("loss_reward", -150.0)
            )
        elif end_condition in (
            "ownship altitude below min",
            "FDM Update Fail",
            "Ownship FDM output Fall",
        ):
            terminal_reward = float(
                reward_config.get("crash_penalty", -250.0)
            )
        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(
                reward_config.get("guard_fail_penalty", -50.0)
            )
        else:
            terminal_reward = float(
                reward_config.get("draw_reward", 0.0)
            )
    elif truncated:
        terminal_reward = float(
            reward_config.get("timeout_penalty", 0.0)
        )

    # Keep component keys unchanged so existing RLlib logging continues to work.
    components = {
        "pursuit": float(pursuit_reward),
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
        + close_range_penalty
        + terminal_reward
    )
    return float(total_reward), components


__all__ = [
    "MY_REWARD_CONFIG",
    "compute_reward",
]
