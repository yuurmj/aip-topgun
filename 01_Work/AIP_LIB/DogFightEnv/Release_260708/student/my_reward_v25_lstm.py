# -*- coding: utf-8 -*-
"""PPO-LSTM v25 hybrid reward for attack preservation + dogfight reacquisition.

The v23 curriculum showed that a fully permissive dogfight reward improved
merge survival/reacquisition but weakened the validated v21 900 m precision
attack.  v24 therefore switches reward shaping by *current geometry*:

ATTACK MODE
    Close range, small ATA, and low relative range-rate (or confirmed damage).
    Use v21-like precision roll/pitch/aim shaping so a gun solution is stable.

DOGFIGHT MODE
    Head-on/high-closure, crossing, and rear-hemisphere situations.
    Permit large combat bank/pitch and reward ATA reacquisition; penalize only
    genuinely unsafe low-altitude dives.

Only confirmed StateIndex values ALT, ROLL, and PITCH are read.  Distance and
ATA use existing GeometryInfo helpers, and previous-step values are stored per
environment in reward_config.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex


_RUNTIME_STATE_KEY = "__v25_lstm_hybrid_previous_step"


MY_REWARD_CONFIG = {
    "step_penalty": -0.002,
    "survival_bonus": 0.0,

    # Combat outcome remains dominant in both modes.
    "target_damage_scale": 350.0,
    "ownship_damage_scale": 300.0,
    "firing_confirmation_bonus": 0.03,

    # Mode gate. At step_ratio=6 a 900 m 264:250 tail chase changes range only
    # ~1-20 m per policy step in observed v23 replay, while alpha-0 head-on merge states changed range by more than 30 m per policy step.
    "attack_mode_distance_m": 1200.0,
    "attack_mode_ata_deg": 15.0,
    "attack_mode_max_abs_range_delta_m": 20.0,

    # ATTACK MODE: v21-like precision weapon shaping.
    "attack_firing_cone_bonus_max": 0.20,
    "attack_firing_cone_edge_bonus": 0.14,
    "attack_near_cone_outer_deg": 3.0,
    "attack_near_cone_outer_bonus": 0.02,
    "attack_aim_fade_deg": 5.0,
    "attack_stable_firing_cone_bonus": 0.08,
    "attack_ata_improvement_scale": 0.020,
    "attack_ata_improvement_cap": 0.08,
    "attack_ata_worsening_scale": 0.025,
    "attack_ata_worsening_cap": 0.10,
    "attack_wez_misaligned_start_deg": 10.0,
    "attack_wez_misaligned_penalty": 0.05,
    "attack_approach_progress_m": 2.0,
    "attack_good_approach_bonus": 0.04,
    "attack_bad_approach_penalty": 0.06,

    # Precision attitude is enforced only while an actual low-closure gun
    # solution exists. Scale is softer than v21 to avoid freezing combat turns.
    "attack_stable_roll_deg": 15.0,
    "attack_strong_roll_deg": 45.0,
    "attack_very_strong_roll_deg": 55.0,
    "attack_stable_pitch_deg": 5.0,
    "attack_strong_pitch_deg": 15.0,
    "attack_roll_recovery_scale": 0.012,
    "attack_roll_recovery_cap": 0.06,
    "attack_roll_worsening_scale": 0.015,
    "attack_roll_worsening_cap": 0.08,
    "attack_pitch_recovery_scale": 0.012,
    "attack_pitch_recovery_cap": 0.05,
    "attack_pitch_worsening_scale": 0.015,
    "attack_pitch_worsening_cap": 0.06,
    "attack_roll_15_45_penalty": 0.18,
    "attack_roll_45_55_penalty": 0.22,
    "attack_roll_over_55_penalty": 0.45,
    "attack_pitch_absolute_penalty": 0.15,
    "attack_pitch_extreme_penalty": 0.20,

    # Preserve the validated high-altitude 900 m attack without trying to force
    # a dogfight that has already descended to 4-5 km back to exactly 7 km.
    "attack_altitude_reference_m": 7000.0,
    "attack_altitude_shaping_min_m": 6000.0,
    "attack_altitude_recovery_scale": 0.010,
    "attack_altitude_recovery_cap": 0.05,
    "attack_descent_penalty_scale": 0.015,
    "attack_descent_penalty_cap": 0.07,

    # DOGFIGHT MODE: v23-style permissive maneuver/reacquisition shaping.
    "dogfight_firing_cone_bonus_max": 0.18,
    "dogfight_firing_cone_edge_bonus": 0.12,
    "dogfight_near_cone_outer_deg": 3.0,
    "dogfight_near_cone_outer_bonus": 0.015,
    "dogfight_aim_fade_deg": 8.0,
    "dogfight_stable_firing_cone_bonus": 0.06,
    "dogfight_ata_improvement_scale": 0.012,
    "dogfight_ata_improvement_cap": 0.07,
    "rear_hemisphere_deg": 90.0,
    "rear_ata_multiplier": 1.8,
    "rear_ata_cap": 0.12,
    "dogfight_ata_worsening_scale": 0.010,
    "dogfight_ata_worsening_cap": 0.06,
    "dogfight_wez_misaligned_start_deg": 25.0,
    "dogfight_wez_misaligned_penalty": 0.03,
    "dogfight_approach_progress_m": 10.0,
    "dogfight_approach_ata_limit_deg": 20.0,
    "dogfight_good_approach_bonus": 0.02,
    "dogfight_bad_approach_penalty": 0.025,

    # General low-altitude safety shared by both modes. Large roll/pitch at safe
    # altitude is not punished unless ATTACK MODE specifically asks for a shot.
    "descent_watch_altitude_m": 5000.0,
    "low_altitude_warning_m": 3500.0,
    "low_altitude_critical_m": 1500.0,
    "low_altitude_warning_penalty": 0.12,
    "low_altitude_critical_penalty": 0.70,
    "low_altitude_recovery_scale": 0.010,
    "low_altitude_recovery_cap": 0.06,
    "low_altitude_descent_scale": 0.006,
    "low_altitude_descent_cap": 0.06,
    "dive_pitch_start_deg": -15.0,
    "dive_pitch_full_deg": -45.0,
    "dive_descent_penalty": 0.30,
    "extreme_roll_deg": 110.0,
    "extreme_roll_descent_penalty": 0.10,
    "safety_reward_min": -1.5,

    # Terminal outcomes.
    "win_reward": 200.0,
    "loss_reward": -250.0,
    "crash_penalty": -300.0,
    "guard_fail_penalty": -80.0,
    "target_crash_reward": 0.0,
    "timeout_penalty": 0.0,
    "draw_reward": 0.0,
}


def _clamp(value: float, low: float, high: float) -> float:
    return max(float(low), min(float(high), float(value)))


def _clamp01(value: float) -> float:
    return _clamp(value, 0.0, 1.0)


def _linear_progress(value: float, start: float, full: float) -> float:
    span = float(full) - float(start)
    if span == 0.0:
        return 1.0 if float(value) == float(full) else 0.0
    return _clamp01((float(value) - float(start)) / span)


def _precision_stability_score(
    abs_roll_deg: float,
    abs_pitch_deg: float,
    altitude_m: float,
    reward_config: dict,
) -> float:
    roll_limit = float(reward_config.get("attack_stable_roll_deg", 15.0))
    pitch_limit = float(reward_config.get("attack_stable_pitch_deg", 5.0))
    roll_score = 1.0 - _clamp01(
        max(0.0, abs_roll_deg - roll_limit) / max(1.0, 55.0 - roll_limit)
    )
    pitch_score = 1.0 - _clamp01(
        max(0.0, abs_pitch_deg - pitch_limit) / max(1.0, 15.0 - pitch_limit)
    )
    # At lower dogfight altitude do not demand a return to 7 km just to shoot.
    if altitude_m >= float(reward_config.get("attack_altitude_shaping_min_m", 6000.0)):
        target_alt = float(reward_config.get("attack_altitude_reference_m", 7000.0))
        altitude_score = 1.0 - _clamp01(abs(altitude_m - target_alt) / 500.0)
    else:
        altitude_score = 1.0
    return _clamp01(0.45 * roll_score + 0.35 * pitch_score + 0.20 * altitude_score)


def _combat_stability_score(
    abs_roll_deg: float,
    abs_pitch_deg: float,
    altitude_m: float,
) -> float:
    roll_score = 1.0 - _clamp01(max(0.0, abs_roll_deg - 90.0) / 90.0)
    pitch_score = 1.0 - _clamp01(max(0.0, abs_pitch_deg - 40.0) / 50.0)
    altitude_score = _clamp01((altitude_m - 800.0) / 2200.0)
    return _clamp01(0.35 * roll_score + 0.35 * pitch_score + 0.30 * altitude_score)


def _cone_posture_reward(
    ata_deg: float,
    firing_half_angle_deg: float,
    stability: float,
    *,
    max_bonus: float,
    edge_bonus: float,
    near_outer_deg: float,
    near_outer_bonus: float,
    fade_deg: float,
    stability_floor: float,
) -> float:
    cone_deg = max(0.1, float(firing_half_angle_deg))
    near_outer_deg = max(cone_deg, float(near_outer_deg))
    fade_deg = max(near_outer_deg, float(fade_deg))

    if ata_deg <= cone_deg:
        quality = 1.0 - _clamp01(ata_deg / cone_deg)
        raw = edge_bonus + (max_bonus - edge_bonus) * quality
    elif ata_deg <= near_outer_deg:
        progress = _clamp01((ata_deg - cone_deg) / max(1e-6, near_outer_deg - cone_deg))
        raw = edge_bonus + (near_outer_bonus - edge_bonus) * progress
    elif ata_deg < fade_deg:
        progress = _clamp01((ata_deg - near_outer_deg) / max(1e-6, fade_deg - near_outer_deg))
        raw = near_outer_bonus * (1.0 - progress)
    else:
        raw = 0.0
    return raw * (float(stability_floor) + (1.0 - float(stability_floor)) * stability)


def _load_previous_state(
    reward_config: dict,
    *,
    distance_m: float,
    altitude_m: float,
) -> dict | None:
    previous = reward_config.get(_RUNTIME_STATE_KEY)
    if not isinstance(previous, dict):
        return None
    if abs(distance_m - float(previous.get("distance_m", distance_m))) > 1500.0:
        return None
    if abs(altitude_m - float(previous.get("altitude_m", altitude_m))) > 800.0:
        return None
    return previous


def _store_current_state(
    reward_config: dict,
    *,
    distance_m: float,
    ata_deg: float,
    altitude_m: float,
    abs_roll_deg: float,
    abs_pitch_deg: float,
    episode_ended: bool,
) -> None:
    if episode_ended:
        reward_config.pop(_RUNTIME_STATE_KEY, None)
        return
    reward_config[_RUNTIME_STATE_KEY] = {
        "distance_m": float(distance_m),
        "ata_deg": float(ata_deg),
        "altitude_m": float(altitude_m),
        "abs_roll_deg": float(abs_roll_deg),
        "abs_pitch_deg": float(abs_pitch_deg),
    }


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
    """Compute v25 PPO-LSTM reward with geometry-conditioned attack/dogfight modes."""
    distance_m = float(geo_info._get_distance(ownship_state, target_state))
    ata_deg = abs(
        float(geo_info._get_antenna_train_angle(ownship_state, target_state, False))
    )

    altitude_m = float(ownship_state[StateIndex.ALT])
    abs_roll_deg = abs(float(ownship_state[StateIndex.ROLL]))
    pitch_deg = float(ownship_state[StateIndex.PITCH])
    abs_pitch_deg = abs(pitch_deg)

    previous = _load_previous_state(
        reward_config,
        distance_m=distance_m,
        altitude_m=altitude_m,
    )

    alt_delta = 0.0
    roll_recovery_delta = 0.0
    pitch_recovery_delta = 0.0
    ata_improvement_delta = 0.0
    distance_progress_delta = 0.0
    previous_ata = ata_deg
    if previous is not None:
        previous_ata = float(previous["ata_deg"])
        alt_delta = altitude_m - float(previous["altitude_m"])
        roll_recovery_delta = float(previous["abs_roll_deg"]) - abs_roll_deg
        pitch_recovery_delta = float(previous["abs_pitch_deg"]) - abs_pitch_deg
        ata_improvement_delta = previous_ata - ata_deg
        distance_progress_delta = float(previous["distance_m"]) - distance_m

    # A close, aligned state is treated as a precision attack only when range is
    # changing slowly. This prevents a 280:280 head-on pass from activating the
    # v21-like roll/pitch constraints immediately before the merge.
    attack_mode = bool(
        float(target_damage) > 0.0
        or (
            previous is not None
            and distance_m <= float(reward_config.get("attack_mode_distance_m", 1200.0))
            and ata_deg <= float(reward_config.get("attack_mode_ata_deg", 15.0))
            and abs(distance_progress_delta)
            <= float(reward_config.get("attack_mode_max_abs_range_delta_m", 8.0))
        )
    )

    # ------------------------------------------------------------------
    # Priority 1: deal damage and avoid taking damage.
    # ------------------------------------------------------------------
    damage_reward = (
        float(reward_config.get("target_damage_scale", 350.0)) * float(target_damage)
        - float(reward_config.get("ownship_damage_scale", 300.0)) * float(ownship_damage)
    )
    if float(target_damage) > 0.0:
        damage_reward += float(reward_config.get("firing_confirmation_bonus", 0.03))

    firing_half_angle_deg = max(0.1, float(wez_config["angle_deg"]) / 2.0)
    wez_min_m = float(wez_config["min_range_m"])
    wez_max_m = float(wez_config["max_range_m"])
    in_wez_range = wez_min_m <= distance_m <= wez_max_m

    # ------------------------------------------------------------------
    # Pursuit/weapon shaping: precision in ATTACK MODE, maneuver freedom in
    # DOGFIGHT MODE.
    # ------------------------------------------------------------------
    if attack_mode:
        stability = _precision_stability_score(
            abs_roll_deg, abs_pitch_deg, altitude_m, reward_config
        )
        aim_reward = _cone_posture_reward(
            ata_deg,
            firing_half_angle_deg,
            stability,
            max_bonus=float(reward_config.get("attack_firing_cone_bonus_max", 0.20)),
            edge_bonus=float(reward_config.get("attack_firing_cone_edge_bonus", 0.14)),
            near_outer_deg=float(reward_config.get("attack_near_cone_outer_deg", 3.0)),
            near_outer_bonus=float(reward_config.get("attack_near_cone_outer_bonus", 0.02)),
            fade_deg=float(reward_config.get("attack_aim_fade_deg", 5.0)),
            stability_floor=0.50,
        )

        ata_change_reward = 0.0
        if previous is not None:
            if ata_improvement_delta > 0.0:
                ata_change_reward = min(
                    float(reward_config.get("attack_ata_improvement_cap", 0.08)),
                    ata_improvement_delta
                    * float(reward_config.get("attack_ata_improvement_scale", 0.020)),
                )
            elif ata_improvement_delta < 0.0:
                ata_change_reward = -min(
                    float(reward_config.get("attack_ata_worsening_cap", 0.10)),
                    -ata_improvement_delta
                    * float(reward_config.get("attack_ata_worsening_scale", 0.025)),
                )

        wez_reward = 0.0
        if in_wez_range and ata_deg <= firing_half_angle_deg:
            aim_quality = 0.60 + 0.40 * (1.0 - _clamp01(ata_deg / firing_half_angle_deg))
            wez_reward = (
                float(reward_config.get("attack_stable_firing_cone_bonus", 0.08))
                * aim_quality
                * stability
            )
        elif in_wez_range and ata_deg > float(
            reward_config.get("attack_wez_misaligned_start_deg", 10.0)
        ):
            start = float(reward_config.get("attack_wez_misaligned_start_deg", 10.0))
            wez_reward = -float(
                reward_config.get("attack_wez_misaligned_penalty", 0.05)
            ) * _clamp01((ata_deg - start) / max(1.0, 30.0 - start))

        approach_reward = 0.0
        if previous is not None and distance_progress_delta > 0.0:
            progress_scale = _clamp01(
                distance_progress_delta
                / max(0.1, float(reward_config.get("attack_approach_progress_m", 2.0)))
            )
            stable_approach = (
                ata_improvement_delta > 0.0
                and abs_roll_deg <= float(reward_config.get("attack_stable_roll_deg", 15.0))
                and abs_pitch_deg <= float(reward_config.get("attack_stable_pitch_deg", 5.0))
                and alt_delta >= -0.5
            )
            if stable_approach:
                approach_reward = float(
                    reward_config.get("attack_good_approach_bonus", 0.04)
                ) * progress_scale
            elif ata_improvement_delta < 0.0:
                approach_reward = -float(
                    reward_config.get("attack_bad_approach_penalty", 0.06)
                ) * progress_scale
    else:
        stability = _combat_stability_score(abs_roll_deg, abs_pitch_deg, altitude_m)
        aim_reward = _cone_posture_reward(
            ata_deg,
            firing_half_angle_deg,
            stability,
            max_bonus=float(reward_config.get("dogfight_firing_cone_bonus_max", 0.18)),
            edge_bonus=float(reward_config.get("dogfight_firing_cone_edge_bonus", 0.12)),
            near_outer_deg=float(reward_config.get("dogfight_near_cone_outer_deg", 3.0)),
            near_outer_bonus=float(reward_config.get("dogfight_near_cone_outer_bonus", 0.015)),
            fade_deg=float(reward_config.get("dogfight_aim_fade_deg", 8.0)),
            stability_floor=0.75,
        )

        ata_change_reward = 0.0
        if previous is not None:
            if ata_improvement_delta > 0.0:
                scale = float(reward_config.get("dogfight_ata_improvement_scale", 0.012))
                cap = float(reward_config.get("dogfight_ata_improvement_cap", 0.07))
                if previous_ata >= float(reward_config.get("rear_hemisphere_deg", 90.0)):
                    scale *= float(reward_config.get("rear_ata_multiplier", 1.8))
                    cap = float(reward_config.get("rear_ata_cap", 0.12))
                ata_change_reward = min(cap, ata_improvement_delta * scale)
            elif ata_improvement_delta < 0.0:
                ata_change_reward = -min(
                    float(reward_config.get("dogfight_ata_worsening_cap", 0.06)),
                    -ata_improvement_delta
                    * float(reward_config.get("dogfight_ata_worsening_scale", 0.010)),
                )

        wez_reward = 0.0
        if in_wez_range and ata_deg <= firing_half_angle_deg:
            aim_quality = 0.60 + 0.40 * (1.0 - _clamp01(ata_deg / firing_half_angle_deg))
            wez_reward = (
                float(reward_config.get("dogfight_stable_firing_cone_bonus", 0.06))
                * aim_quality
                * (0.75 + 0.25 * stability)
            )
        elif in_wez_range and ata_deg > float(
            reward_config.get("dogfight_wez_misaligned_start_deg", 25.0)
        ):
            start = float(reward_config.get("dogfight_wez_misaligned_start_deg", 25.0))
            wez_reward = -float(
                reward_config.get("dogfight_wez_misaligned_penalty", 0.03)
            ) * _clamp01((ata_deg - start) / max(1.0, 90.0 - start))

        approach_reward = 0.0
        if previous is not None and distance_progress_delta > 0.0:
            progress_scale = _clamp01(
                distance_progress_delta
                / max(0.1, float(reward_config.get("dogfight_approach_progress_m", 10.0)))
            )
            useful_ata = ata_deg <= float(
                reward_config.get("dogfight_approach_ata_limit_deg", 20.0)
            )
            if useful_ata and ata_improvement_delta > 0.0:
                approach_reward = float(
                    reward_config.get("dogfight_good_approach_bonus", 0.02)
                ) * progress_scale
            elif useful_ata and ata_improvement_delta < 0.0:
                approach_reward = -float(
                    reward_config.get("dogfight_bad_approach_penalty", 0.025)
                ) * progress_scale

    pursuit_reward = aim_reward + ata_change_reward + wez_reward + approach_reward

    # ------------------------------------------------------------------
    # Shared general-dogfight low-altitude safety.
    # ------------------------------------------------------------------
    safety_reward = 0.0
    warning_alt = float(reward_config.get("low_altitude_warning_m", 3500.0))
    critical_alt = float(reward_config.get("low_altitude_critical_m", 1500.0))
    watch_alt = float(reward_config.get("descent_watch_altitude_m", 5000.0))

    if altitude_m < warning_alt:
        if altitude_m >= critical_alt:
            safety_reward -= float(
                reward_config.get("low_altitude_warning_penalty", 0.12)
            ) * _linear_progress(altitude_m, warning_alt, critical_alt)
        else:
            safety_reward -= float(reward_config.get("low_altitude_warning_penalty", 0.12))
            safety_reward -= float(
                reward_config.get("low_altitude_critical_penalty", 0.70)
            ) * _clamp01((critical_alt - altitude_m) / max(1.0, critical_alt - 300.0))

    if previous is not None and altitude_m < watch_alt:
        if alt_delta < 0.0:
            safety_reward -= min(
                float(reward_config.get("low_altitude_descent_cap", 0.06)),
                -alt_delta * float(reward_config.get("low_altitude_descent_scale", 0.006)),
            )
        elif alt_delta > 0.0 and altitude_m < warning_alt:
            safety_reward += min(
                float(reward_config.get("low_altitude_recovery_cap", 0.06)),
                alt_delta * float(reward_config.get("low_altitude_recovery_scale", 0.010)),
            )

    if previous is not None and altitude_m < watch_alt and alt_delta < 0.0:
        dive_start = float(reward_config.get("dive_pitch_start_deg", -15.0))
        dive_full = float(reward_config.get("dive_pitch_full_deg", -45.0))
        if pitch_deg < dive_start:
            dive_factor = _linear_progress(pitch_deg, dive_start, dive_full)
            descent_factor = _clamp01(-alt_delta / 4.0)
            safety_reward -= float(
                reward_config.get("dive_descent_penalty", 0.30)
            ) * dive_factor * (0.40 + 0.60 * descent_factor)

        if abs_roll_deg > float(reward_config.get("extreme_roll_deg", 110.0)):
            roll_factor = _clamp01(
                (abs_roll_deg - float(reward_config.get("extreme_roll_deg", 110.0))) / 70.0
            )
            safety_reward -= float(
                reward_config.get("extreme_roll_descent_penalty", 0.10)
            ) * roll_factor

    # ATTACK MODE adds local v21-like precision attitude shaping only while a
    # gun solution is plausible. This is the key v24 hybrid change.
    if attack_mode and previous is not None:
        if roll_recovery_delta > 0.0:
            safety_reward += min(
                float(reward_config.get("attack_roll_recovery_cap", 0.06)),
                roll_recovery_delta
                * float(reward_config.get("attack_roll_recovery_scale", 0.012)),
            )
        elif roll_recovery_delta < 0.0:
            safety_reward -= min(
                float(reward_config.get("attack_roll_worsening_cap", 0.08)),
                -roll_recovery_delta
                * float(reward_config.get("attack_roll_worsening_scale", 0.015)),
            )

        if pitch_recovery_delta > 0.0:
            safety_reward += min(
                float(reward_config.get("attack_pitch_recovery_cap", 0.05)),
                pitch_recovery_delta
                * float(reward_config.get("attack_pitch_recovery_scale", 0.012)),
            )
        elif pitch_recovery_delta < 0.0:
            safety_reward -= min(
                float(reward_config.get("attack_pitch_worsening_cap", 0.06)),
                -pitch_recovery_delta
                * float(reward_config.get("attack_pitch_worsening_scale", 0.015)),
            )

        stable_roll = float(reward_config.get("attack_stable_roll_deg", 15.0))
        strong_roll = float(reward_config.get("attack_strong_roll_deg", 45.0))
        very_strong_roll = float(reward_config.get("attack_very_strong_roll_deg", 55.0))
        if stable_roll < abs_roll_deg <= strong_roll:
            safety_reward -= float(
                reward_config.get("attack_roll_15_45_penalty", 0.18)
            ) * _linear_progress(abs_roll_deg, stable_roll, strong_roll)
        elif strong_roll < abs_roll_deg <= very_strong_roll:
            safety_reward -= float(reward_config.get("attack_roll_15_45_penalty", 0.18))
            safety_reward -= float(
                reward_config.get("attack_roll_45_55_penalty", 0.22)
            ) * _linear_progress(abs_roll_deg, strong_roll, very_strong_roll)
        elif abs_roll_deg > very_strong_roll:
            safety_reward -= float(reward_config.get("attack_roll_15_45_penalty", 0.18))
            safety_reward -= float(reward_config.get("attack_roll_45_55_penalty", 0.22))
            safety_reward -= float(
                reward_config.get("attack_roll_over_55_penalty", 0.45)
            ) * (0.50 + 0.50 * _clamp01((abs_roll_deg - very_strong_roll) / 35.0))

        stable_pitch = float(reward_config.get("attack_stable_pitch_deg", 5.0))
        strong_pitch = float(reward_config.get("attack_strong_pitch_deg", 15.0))
        if stable_pitch < abs_pitch_deg <= strong_pitch:
            safety_reward -= float(
                reward_config.get("attack_pitch_absolute_penalty", 0.15)
            ) * _linear_progress(abs_pitch_deg, stable_pitch, strong_pitch)
        elif abs_pitch_deg > strong_pitch:
            safety_reward -= float(reward_config.get("attack_pitch_absolute_penalty", 0.15))
            safety_reward -= float(
                reward_config.get("attack_pitch_extreme_penalty", 0.20)
            ) * _clamp01((abs_pitch_deg - strong_pitch) / 20.0)

        min_attack_alt = float(reward_config.get("attack_altitude_shaping_min_m", 6000.0))
        if altitude_m >= min_attack_alt:
            target_alt = float(reward_config.get("attack_altitude_reference_m", 7000.0))
            previous_altitude = float(previous["altitude_m"])
            prev_error = abs(previous_altitude - target_alt)
            curr_error = abs(altitude_m - target_alt)
            if alt_delta < 0.0:
                safety_reward -= min(
                    float(reward_config.get("attack_descent_penalty_cap", 0.07)),
                    -alt_delta
                    * float(reward_config.get("attack_descent_penalty_scale", 0.015)),
                )
            if curr_error < prev_error:
                safety_reward += min(
                    float(reward_config.get("attack_altitude_recovery_cap", 0.05)),
                    (prev_error - curr_error)
                    * float(reward_config.get("attack_altitude_recovery_scale", 0.010)),
                )

    safety_reward = max(float(reward_config.get("safety_reward_min", -1.5)), safety_reward)

    # ------------------------------------------------------------------
    # Terminal outcomes.
    # ------------------------------------------------------------------
    terminal_reward = 0.0
    if terminated:
        if end_condition == "target destroyed":
            terminal_reward = float(reward_config.get("win_reward", 200.0))
        elif end_condition == "target altitude below min":
            terminal_reward = float(reward_config.get("target_crash_reward", 0.0))
        elif end_condition == "ownship destroyed":
            terminal_reward = float(reward_config.get("loss_reward", -250.0))
        elif end_condition in (
            "ownship altitude below min",
            "FDM Update Fail",
            "Ownship FDM output Fall",
        ):
            terminal_reward = float(reward_config.get("crash_penalty", -300.0))
        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(reward_config.get("guard_fail_penalty", -80.0))
        else:
            terminal_reward = float(reward_config.get("draw_reward", 0.0))
    elif truncated:
        terminal_reward = float(reward_config.get("timeout_penalty", 0.0))

    step_reward = float(reward_config.get("step_penalty", -0.002))
    survival_reward = float(reward_config.get("survival_bonus", 0.0))
    total_reward = (
        step_reward
        + survival_reward
        + damage_reward
        + pursuit_reward
        + safety_reward
        + terminal_reward
    )

    components = {
        "pursuit": float(pursuit_reward),
        "damage": float(damage_reward),
        "safety": float(safety_reward),
        "survival": float(survival_reward),
    }

    _store_current_state(
        reward_config,
        distance_m=distance_m,
        ata_deg=ata_deg,
        altitude_m=altitude_m,
        abs_roll_deg=abs_roll_deg,
        abs_pitch_deg=abs_pitch_deg,
        episode_ended=bool(terminated or truncated),
    )
    return float(total_reward), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
