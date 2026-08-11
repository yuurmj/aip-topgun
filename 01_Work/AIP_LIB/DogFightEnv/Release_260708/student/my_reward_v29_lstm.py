# -*- coding: utf-8 -*-
"""PPO-LSTM v29 reward: global soft-roll cap + cone-focused precision.

This refinement restarts from the best v26 checkpoint.  It keeps maneuver
freedom below moderate bank angles, softly suppresses excessive bank everywhere,
and rewards *progress toward* the real firing cone instead of rewarding a policy
for merely sitting around ATA 3-5 deg.

Only the confirmed StateIndex values ALT, ROLL, and PITCH are read directly.
Distance and ATA come from the environment's GeometryInfo methods.
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


_RUNTIME_STATE_KEY = "__v29_lstm_previous_step"


MY_REWARD_CONFIG = {
    "step_penalty": -0.005,
    "survival_bonus": 0.0,

    # Compatibility keys are required by describe_reward() when the training
    # record is written.  The custom reward itself uses the directional scales.
    "damage_scale": 350.0,
    "target_damage_scale": 350.0,
    "ownship_damage_scale": 120.0,
    "firing_confirmation_bonus": 0.03,

    # Real firing cone shaping.  Outside the cone, static reward is deliberately
    # weak; directional improvement toward 1 deg does most of the work.
    "firing_cone_bonus_max": 0.26,
    "firing_cone_edge_bonus": 0.18,
    "near_cone_outer_deg": 3.0,
    "near_cone_outer_bonus": 0.008,
    "aim_fade_deg": 5.0,

    # Directional ATA shaping.
    "ata_improvement_scale_far": 0.018,
    "ata_improvement_scale_near": 0.045,
    "ata_improvement_cap": 0.12,
    "ata_worsening_scale_far": 0.022,
    "ata_worsening_scale_near": 0.050,
    "ata_worsening_cap": 0.14,
    "precision_outer_deg": 5.0,
    "precision_progress_bonus_scale": 0.14,
    "precision_progress_penalty_scale": 0.16,
    "milestone_3deg_bonus": 0.015,
    "milestone_2deg_bonus": 0.025,

    # Crossing/holding the true weapon solution matters most.
    "cone_entry_bonus": 0.08,
    "cone_hold_bonus": 0.12,
    "cone_exit_penalty": 0.18,
    "stable_firing_cone_bonus": 0.10,
    "stable_cone_posture_bonus": 0.05,
    "wez_misaligned_penalty": 0.05,
    "wez_misaligned_start_deg": 10.0,

    # Conditional approach only; distance reduction by itself is not rewarded.
    "approach_progress_m": 2.0,
    "good_approach_bonus": 0.035,
    "bad_approach_penalty": 0.055,

    # Altitude protection inherited from the successful 900 m attack branch.
    "target_altitude_m": 7000.0,
    "altitude_tolerance_m": 30.0,
    "altitude_recovery_scale": 0.015,
    "altitude_recovery_cap": 0.08,
    "descent_penalty_scale": 0.030,
    "descent_penalty_cap": 0.12,
    "low_altitude_warning_m": 6950.0,
    "low_altitude_strong_m": 6900.0,
    "episode_min_altitude_m": 6550.0,
    "low_altitude_penalty": 1.00,
    "low_altitude_warning_penalty": 0.05,
    "low_altitude_strong_penalty": 1.00,

    # Global SOFT roll cap: free <=45 deg, mild 45-60, clear suppression >60.
    # This avoids both v27 (too stable/off-cone) and v28 (80-100 deg bank).
    "global_roll_free_deg": 45.0,
    "global_roll_soft_deg": 60.0,
    "global_roll_hard_deg": 90.0,
    "global_roll_soft_penalty": 0.10,
    "global_roll_hard_penalty": 0.30,
    "global_roll_extreme_penalty": 0.45,

    # As ATA approaches the cone, allowable bank shrinks smoothly from 45 deg
    # toward 20 deg.  This is a penalty-only gate; there is no reward for merely
    # being flat at ATA 3-5 deg.
    "near_roll_limit_outer_deg": 45.0,
    "near_roll_limit_cone_deg": 20.0,
    "near_roll_penalty_max": 0.18,
    "stable_roll_deg": 15.0,

    # Pitch is treated similarly but more gently.
    "global_pitch_free_deg": 15.0,
    "global_pitch_penalty_max": 0.18,
    "near_pitch_limit_outer_deg": 15.0,
    "near_pitch_limit_cone_deg": 7.0,
    "near_pitch_penalty_max": 0.12,
    "stable_pitch_deg": 5.0,
    "dive_pitch_start_deg": -2.0,
    "dive_pitch_full_deg": -10.0,
    "dive_descent_penalty": 0.35,

    "safety_reward_min": -3.0,

    "win_reward": 150.0,
    "loss_reward": -200.0,
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


def _stability_score(abs_roll_deg: float, abs_pitch_deg: float, altitude_m: float, cfg: dict) -> float:
    roll_limit = float(cfg.get("stable_roll_deg", 15.0))
    pitch_limit = float(cfg.get("stable_pitch_deg", 5.0))
    target_altitude = float(cfg.get("target_altitude_m", 7000.0))
    roll_score = 1.0 - _clamp01(max(0.0, abs_roll_deg - roll_limit) / max(1.0, 55.0 - roll_limit))
    pitch_score = 1.0 - _clamp01(max(0.0, abs_pitch_deg - pitch_limit) / max(1.0, 15.0 - pitch_limit))
    altitude_score = 1.0 - _clamp01(abs(altitude_m - target_altitude) / 350.0)
    return _clamp01(0.40 * roll_score + 0.30 * pitch_score + 0.30 * altitude_score)


def _attack_posture_reward(ata_deg: float, stability: float, cone_deg: float, cfg: dict) -> float:
    cone_deg = max(0.1, float(cone_deg))
    max_bonus = float(cfg.get("firing_cone_bonus_max", 0.26))
    edge_bonus = float(cfg.get("firing_cone_edge_bonus", 0.18))
    near_outer = max(cone_deg, float(cfg.get("near_cone_outer_deg", 3.0)))
    near_bonus = float(cfg.get("near_cone_outer_bonus", 0.008))
    fade_deg = max(near_outer, float(cfg.get("aim_fade_deg", 5.0)))

    if ata_deg <= cone_deg:
        quality = 1.0 - _clamp01(ata_deg / cone_deg)
        raw = edge_bonus + (max_bonus - edge_bonus) * quality
        # Posture matters only after a real gun solution exists.
        return raw * (0.80 + 0.20 * stability)
    if ata_deg <= near_outer:
        p = _clamp01((ata_deg - cone_deg) / max(1e-6, near_outer - cone_deg))
        return edge_bonus + (near_bonus - edge_bonus) * p
    if ata_deg < fade_deg:
        p = _clamp01((ata_deg - near_outer) / max(1e-6, fade_deg - near_outer))
        return near_bonus * (1.0 - p)
    return 0.0


def _precision_potential(ata_deg: float, cone_deg: float, outer_deg: float) -> float:
    if ata_deg <= cone_deg:
        return 1.0
    if ata_deg >= outer_deg:
        return 0.0
    return _clamp01((outer_deg - ata_deg) / max(1e-6, outer_deg - cone_deg))


def _global_roll_penalty(abs_roll_deg: float, cfg: dict) -> float:
    free = float(cfg.get("global_roll_free_deg", 45.0))
    soft = float(cfg.get("global_roll_soft_deg", 60.0))
    hard = float(cfg.get("global_roll_hard_deg", 90.0))
    p_soft = float(cfg.get("global_roll_soft_penalty", 0.10))
    p_hard = float(cfg.get("global_roll_hard_penalty", 0.30))
    p_extreme = float(cfg.get("global_roll_extreme_penalty", 0.45))

    if abs_roll_deg <= free:
        return 0.0
    if abs_roll_deg <= soft:
        return p_soft * _linear_progress(abs_roll_deg, free, soft)
    if abs_roll_deg <= hard:
        return p_soft + p_hard * _linear_progress(abs_roll_deg, soft, hard)
    return p_soft + p_hard + p_extreme * _clamp01((abs_roll_deg - hard) / 30.0)


def _load_previous_state(cfg: dict, *, distance_m: float, altitude_m: float) -> dict | None:
    previous = cfg.get(_RUNTIME_STATE_KEY)
    if not isinstance(previous, dict):
        return None
    if abs(distance_m - float(previous.get("distance_m", distance_m))) > 200.0:
        return None
    if abs(altitude_m - float(previous.get("altitude_m", altitude_m))) > 300.0:
        return None
    return previous


def _store_current_state(cfg: dict, *, distance_m: float, ata_deg: float, altitude_m: float,
                         abs_roll_deg: float, abs_pitch_deg: float, episode_ended: bool) -> None:
    if episode_ended:
        cfg.pop(_RUNTIME_STATE_KEY, None)
        return
    cfg[_RUNTIME_STATE_KEY] = {
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
    distance_m = float(geo_info._get_distance(ownship_state, target_state))
    ata_deg = abs(float(geo_info._get_antenna_train_angle(ownship_state, target_state, False)))

    altitude_m = float(ownship_state[StateIndex.ALT])
    abs_roll_deg = abs(float(ownship_state[StateIndex.ROLL]))
    pitch_deg = float(ownship_state[StateIndex.PITCH])
    abs_pitch_deg = abs(pitch_deg)

    previous = _load_previous_state(reward_config, distance_m=distance_m, altitude_m=altitude_m)
    alt_delta = 0.0
    ata_delta = 0.0
    distance_progress = 0.0
    if previous is not None:
        alt_delta = altitude_m - float(previous["altitude_m"])
        ata_delta = float(previous["ata_deg"]) - ata_deg
        distance_progress = float(previous["distance_m"]) - distance_m

    stability = _stability_score(abs_roll_deg, abs_pitch_deg, altitude_m, reward_config)
    cone_deg = max(0.1, float(wez_config["angle_deg"]) / 2.0)
    precision_outer = max(cone_deg, float(reward_config.get("precision_outer_deg", 5.0)))
    precision_gate = _precision_potential(ata_deg, cone_deg, precision_outer)

    # 1) Direct combat outcome.
    damage_reward = (
        float(reward_config.get("target_damage_scale", 350.0)) * float(target_damage)
        - float(reward_config.get("ownship_damage_scale", 120.0)) * float(ownship_damage)
    )
    if float(target_damage) > 0.0:
        damage_reward += float(reward_config.get("firing_confirmation_bonus", 0.03))

    # 2) Static aim: intentionally weak outside the real cone.
    aim_reward = _attack_posture_reward(ata_deg, stability, cone_deg, reward_config)

    # 3) Directional precision: reward motion toward 1 deg, not occupancy at 3-5 deg.
    ata_change_reward = 0.0
    if previous is not None:
        prev_ata = float(previous["ata_deg"])
        near = (ata_deg <= precision_outer) or (prev_ata <= precision_outer)
        if ata_delta > 0.0:
            scale = float(reward_config.get("ata_improvement_scale_near" if near else "ata_improvement_scale_far", 0.045 if near else 0.018))
            ata_change_reward += min(float(reward_config.get("ata_improvement_cap", 0.12)), ata_delta * scale)
        elif ata_delta < 0.0:
            scale = float(reward_config.get("ata_worsening_scale_near" if near else "ata_worsening_scale_far", 0.050 if near else 0.022))
            ata_change_reward -= min(float(reward_config.get("ata_worsening_cap", 0.14)), -ata_delta * scale)

        prev_p = _precision_potential(prev_ata, cone_deg, precision_outer)
        curr_p = _precision_potential(ata_deg, cone_deg, precision_outer)
        dp = curr_p - prev_p
        if dp > 0.0:
            ata_change_reward += dp * float(reward_config.get("precision_progress_bonus_scale", 0.14))
        elif dp < 0.0:
            ata_change_reward += dp * float(reward_config.get("precision_progress_penalty_scale", 0.16))

        # Small one-way milestones encourage 5 -> 3 -> 2 -> cone progression.
        if prev_ata > 3.0 >= ata_deg:
            ata_change_reward += float(reward_config.get("milestone_3deg_bonus", 0.015))
        if prev_ata > 2.0 >= ata_deg:
            ata_change_reward += float(reward_config.get("milestone_2deg_bonus", 0.025))
        if prev_ata > cone_deg >= ata_deg:
            ata_change_reward += float(reward_config.get("cone_entry_bonus", 0.08))
        elif prev_ata <= cone_deg < ata_deg:
            ata_change_reward -= float(reward_config.get("cone_exit_penalty", 0.18))

    # 4) Hold a real firing solution.
    wez_min = float(wez_config["min_range_m"])
    wez_max = float(wez_config["max_range_m"])
    in_wez = wez_min <= distance_m <= wez_max
    wez_reward = 0.0
    if in_wez and ata_deg <= cone_deg:
        aim_quality = 0.60 + 0.40 * (1.0 - _clamp01(ata_deg / cone_deg))
        wez_reward += float(reward_config.get("stable_firing_cone_bonus", 0.10)) * aim_quality * (0.45 + 0.55 * stability)
        if previous is not None and float(previous["ata_deg"]) <= cone_deg:
            wez_reward += float(reward_config.get("cone_hold_bonus", 0.12)) * aim_quality * (0.55 + 0.45 * stability)
        if abs_roll_deg <= float(reward_config.get("stable_roll_deg", 15.0)) and abs_pitch_deg <= float(reward_config.get("stable_pitch_deg", 5.0)):
            wez_reward += float(reward_config.get("stable_cone_posture_bonus", 0.05)) * aim_quality
    elif in_wez and ata_deg > float(reward_config.get("wez_misaligned_start_deg", 10.0)):
        wez_reward -= float(reward_config.get("wez_misaligned_penalty", 0.05)) * _clamp01((ata_deg - 10.0) / 20.0)

    # 5) Approach only when geometry is improving.
    approach_reward = 0.0
    if previous is not None and distance_progress > 0.0:
        progress_scale = _clamp01(distance_progress / max(0.1, float(reward_config.get("approach_progress_m", 2.0))))
        if ata_delta > 0.0 and ata_deg <= 10.0 and alt_delta >= -0.5:
            approach_reward += float(reward_config.get("good_approach_bonus", 0.035)) * progress_scale
        elif ata_delta < 0.0 or alt_delta < -1.0:
            approach_reward -= float(reward_config.get("bad_approach_penalty", 0.055)) * progress_scale

    pursuit_reward = aim_reward + ata_change_reward + wez_reward + approach_reward

    # 6) Safety: altitude + global soft roll cap + near-cone posture cap.
    safety_reward = 0.0
    target_altitude = float(reward_config.get("target_altitude_m", 7000.0))
    if previous is not None:
        prev_alt = float(previous["altitude_m"])
        prev_err = abs(prev_alt - target_altitude)
        curr_err = abs(altitude_m - target_altitude)
        if alt_delta < 0.0:
            safety_reward -= min(float(reward_config.get("descent_penalty_cap", 0.12)), -alt_delta * float(reward_config.get("descent_penalty_scale", 0.030)))
        if curr_err < prev_err:
            safety_reward += min(float(reward_config.get("altitude_recovery_cap", 0.08)), (prev_err - curr_err) * float(reward_config.get("altitude_recovery_scale", 0.015)))

    warning_alt = float(reward_config.get("low_altitude_warning_m", 6950.0))
    strong_alt = float(reward_config.get("low_altitude_strong_m", 6900.0))
    minimum_alt = float(reward_config.get("episode_min_altitude_m", 6550.0))
    if strong_alt <= altitude_m < warning_alt:
        safety_reward -= float(reward_config.get("low_altitude_warning_penalty", 0.05)) * _linear_progress(altitude_m, warning_alt, strong_alt)
    elif altitude_m < strong_alt:
        safety_reward -= 0.15 + float(reward_config.get("low_altitude_strong_penalty", 1.00)) * _linear_progress(altitude_m, strong_alt, minimum_alt)

    # Global cap is always active, but leaves useful maneuvering <=45 deg free.
    safety_reward -= _global_roll_penalty(abs_roll_deg, reward_config)

    # Near the gun solution, allowed bank shrinks smoothly 45 -> 20 deg.
    near_roll_outer = float(reward_config.get("near_roll_limit_outer_deg", 45.0))
    near_roll_cone = float(reward_config.get("near_roll_limit_cone_deg", 20.0))
    allowed_roll = near_roll_outer + (near_roll_cone - near_roll_outer) * precision_gate
    if precision_gate > 0.0 and abs_roll_deg > allowed_roll:
        excess = _clamp01((abs_roll_deg - allowed_roll) / max(1.0, 60.0 - allowed_roll))
        safety_reward -= precision_gate * float(reward_config.get("near_roll_penalty_max", 0.18)) * excess

    global_pitch_free = float(reward_config.get("global_pitch_free_deg", 15.0))
    if abs_pitch_deg > global_pitch_free:
        safety_reward -= float(reward_config.get("global_pitch_penalty_max", 0.18)) * _clamp01((abs_pitch_deg - global_pitch_free) / 20.0)

    near_pitch_outer = float(reward_config.get("near_pitch_limit_outer_deg", 15.0))
    near_pitch_cone = float(reward_config.get("near_pitch_limit_cone_deg", 7.0))
    allowed_pitch = near_pitch_outer + (near_pitch_cone - near_pitch_outer) * precision_gate
    if precision_gate > 0.0 and abs_pitch_deg > allowed_pitch:
        excess = _clamp01((abs_pitch_deg - allowed_pitch) / max(1.0, 20.0 - allowed_pitch))
        safety_reward -= precision_gate * float(reward_config.get("near_pitch_penalty_max", 0.12)) * excess

    if previous is not None and pitch_deg < float(reward_config.get("dive_pitch_start_deg", -2.0)) and alt_delta < 0.0:
        dive_factor = _linear_progress(pitch_deg, float(reward_config.get("dive_pitch_start_deg", -2.0)), float(reward_config.get("dive_pitch_full_deg", -10.0)))
        descent_factor = _clamp01(-alt_delta / 2.0)
        safety_reward -= float(reward_config.get("dive_descent_penalty", 0.35)) * dive_factor * (0.50 + 0.50 * descent_factor)

    safety_reward = max(float(reward_config.get("safety_reward_min", -3.0)), safety_reward)

    terminal_reward = 0.0
    if terminated:
        if end_condition == "target destroyed":
            terminal_reward = float(reward_config.get("win_reward", 150.0))
        elif end_condition == "target altitude below min":
            terminal_reward = float(reward_config.get("target_crash_reward", 0.0))
        elif end_condition == "ownship destroyed":
            terminal_reward = float(reward_config.get("loss_reward", -200.0))
        elif end_condition in ("ownship altitude below min", "FDM Update Fail", "Ownship FDM output Fall"):
            terminal_reward = float(reward_config.get("crash_penalty", -300.0))
        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(reward_config.get("guard_fail_penalty", -80.0))
        else:
            terminal_reward = float(reward_config.get("draw_reward", 0.0))
    elif truncated:
        terminal_reward = float(reward_config.get("timeout_penalty", 0.0))

    step_reward = float(reward_config.get("step_penalty", -0.005))
    survival_reward = float(reward_config.get("survival_bonus", 0.0))
    total_reward = step_reward + survival_reward + damage_reward + pursuit_reward + safety_reward + terminal_reward

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
