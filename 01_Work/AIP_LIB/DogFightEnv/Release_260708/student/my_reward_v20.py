# -*- coding: utf-8 -*-
"""PPO v20 reward for the 900 m right-side attack refinement.

Flight-state shaping reads only the confirmed StateIndex values ALT, ROLL,
and PITCH. Distance and ATA are calculated through the environment's existing
GeometryInfo methods. Previous-step values are stored inside each environment's
reward_config dictionary, avoiding process-wide global state.
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


_RUNTIME_STATE_KEY = "__v20_previous_step"


MY_REWARD_CONFIG = {
    # Small time cost. Survival or entry alone should not dominate learning.
    "step_penalty": -0.005,
    "survival_bonus": 0.0,

    # Priority 1: actual damage. At 20% target damage this contributes +60.
    "target_damage_scale": 300.0,
    "ownship_damage_scale": 100.0,
    "firing_confirmation_bonus": 0.03,

    # Priority 2: ATA 3-5 degree attack posture and ATA improvement.
    "attack_posture_bonus_max": 0.20,
    "attack_posture_bonus_at_5deg": 0.12,
    "attack_posture_fade_deg": 10.0,
    "ata_improvement_scale": 0.020,
    "ata_improvement_cap": 0.08,
    "ata_worsening_scale": 0.025,
    "ata_worsening_cap": 0.10,

    # Priority 3: stable WEZ hold.
    "stable_wez_bonus": 0.08,
    "wez_misaligned_penalty": 0.05,
    "wez_misaligned_start_deg": 10.0,

    # Priority 4: conditional approach. Distance decrease alone earns nothing.
    "approach_progress_m": 2.0,
    "good_approach_bonus": 0.04,
    "bad_approach_penalty": 0.06,

    # Altitude trend and recovery toward 7,000 m.
    "target_altitude_m": 7000.0,
    "altitude_tolerance_m": 30.0,
    "altitude_recovery_scale": 0.015,
    "altitude_recovery_cap": 0.08,
    "descent_penalty_scale": 0.030,
    "descent_penalty_cap": 0.12,
    "low_altitude_warning_m": 6950.0,
    "low_altitude_strong_m": 6900.0,
    "episode_min_altitude_m": 6550.0,
    "low_altitude_warning_penalty": 0.05,
    "low_altitude_strong_penalty": 1.00,

    # Roll recovery and absolute roll protection.
    "stable_roll_deg": 15.0,
    "roll_penalty_strong_deg": 45.0,
    "roll_penalty_very_strong_deg": 55.0,
    "roll_recovery_scale": 0.015,
    "roll_recovery_cap": 0.08,
    "roll_worsening_scale": 0.020,
    "roll_worsening_cap": 0.10,
    "roll_15_45_penalty": 0.30,
    "roll_45_55_penalty": 0.40,
    "roll_over_55_penalty": 0.90,

    # Pitch recovery and dive suppression.
    "stable_pitch_deg": 5.0,
    "pitch_penalty_strong_deg": 15.0,
    "pitch_recovery_scale": 0.015,
    "pitch_recovery_cap": 0.06,
    "pitch_worsening_scale": 0.020,
    "pitch_worsening_cap": 0.08,
    "pitch_absolute_penalty": 0.25,
    "pitch_extreme_penalty": 0.35,
    "dive_pitch_start_deg": -2.0,
    "dive_pitch_full_deg": -10.0,
    "dive_descent_penalty": 0.35,

    # Keep a single bad step bounded for PPO value learning.
    "safety_reward_min": -3.0,

    # Terminal outcomes.
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
    """Return 0 at start and 1 at full, supporting either direction."""
    span = float(full) - float(start)
    if span == 0.0:
        return 1.0 if float(value) == float(full) else 0.0
    return _clamp01((float(value) - float(start)) / span)


def _stability_score(
    abs_roll_deg: float,
    abs_pitch_deg: float,
    altitude_m: float,
    reward_config: dict,
) -> float:
    roll_limit = float(reward_config.get("stable_roll_deg", 15.0))
    pitch_limit = float(reward_config.get("stable_pitch_deg", 5.0))
    target_altitude = float(reward_config.get("target_altitude_m", 7000.0))

    roll_score = 1.0 - _clamp01(
        max(0.0, abs_roll_deg - roll_limit) / max(1.0, 55.0 - roll_limit)
    )
    pitch_score = 1.0 - _clamp01(
        max(0.0, abs_pitch_deg - pitch_limit) / max(1.0, 15.0 - pitch_limit)
    )
    altitude_score = 1.0 - _clamp01(abs(altitude_m - target_altitude) / 350.0)
    return _clamp01(0.40 * roll_score + 0.30 * pitch_score + 0.30 * altitude_score)


def _attack_posture_reward(
    ata_deg: float,
    stability: float,
    reward_config: dict,
) -> float:
    max_bonus = float(reward_config.get("attack_posture_bonus_max", 0.20))
    five_bonus = float(reward_config.get("attack_posture_bonus_at_5deg", 0.12))
    fade_deg = max(5.0, float(reward_config.get("attack_posture_fade_deg", 10.0)))

    if ata_deg <= 5.0:
        quality = 1.0 - _clamp01(ata_deg / 5.0)
        raw = five_bonus + (max_bonus - five_bonus) * quality
    elif ata_deg < fade_deg:
        raw = five_bonus * (1.0 - (ata_deg - 5.0) / (fade_deg - 5.0))
    else:
        raw = 0.0
    return raw * (0.50 + 0.50 * stability)


def _load_previous_state(
    reward_config: dict,
    *,
    distance_m: float,
    altitude_m: float,
) -> dict | None:
    previous = reward_config.get(_RUNTIME_STATE_KEY)
    if not isinstance(previous, dict):
        return None

    # A terminal reward normally clears the state. These guards also prevent a
    # rare simulator-failure reset from producing a large false delta.
    if abs(distance_m - float(previous.get("distance_m", distance_m))) > 200.0:
        return None
    if abs(altitude_m - float(previous.get("altitude_m", altitude_m))) > 300.0:
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
    """Compute the v20 900 m attack-refinement reward."""
    distance_m = float(geo_info._get_distance(ownship_state, target_state))
    ata_deg = abs(
        float(
            geo_info._get_antenna_train_angle(
                ownship_state,
                target_state,
                False,
            )
        )
    )

    # Do not introduce unverified state indexes.
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
    if previous is not None:
        alt_delta = altitude_m - float(previous["altitude_m"])
        roll_recovery_delta = float(previous["abs_roll_deg"]) - abs_roll_deg
        pitch_recovery_delta = float(previous["abs_pitch_deg"]) - abs_pitch_deg
        ata_improvement_delta = float(previous["ata_deg"]) - ata_deg
        distance_progress_delta = float(previous["distance_m"]) - distance_m

    stability = _stability_score(
        abs_roll_deg,
        abs_pitch_deg,
        altitude_m,
        reward_config,
    )

    # ------------------------------------------------------------------
    # Priority 1: direct damage.
    # ------------------------------------------------------------------
    damage_reward = (
        float(reward_config.get("target_damage_scale", 300.0))
        * float(target_damage)
        - float(reward_config.get("ownship_damage_scale", 100.0))
        * float(ownship_damage)
    )
    if float(target_damage) > 0.0:
        damage_reward += float(
            reward_config.get("firing_confirmation_bonus", 0.03)
        )

    # ------------------------------------------------------------------
    # Priority 2-4: aim, stable WEZ, and conditional approach.
    # ------------------------------------------------------------------
    aim_reward = _attack_posture_reward(ata_deg, stability, reward_config)

    ata_change_reward = 0.0
    if previous is not None:
        if ata_improvement_delta > 0.0:
            ata_change_reward = min(
                float(reward_config.get("ata_improvement_cap", 0.08)),
                ata_improvement_delta
                * float(reward_config.get("ata_improvement_scale", 0.020)),
            )
        elif ata_improvement_delta < 0.0:
            ata_change_reward = -min(
                float(reward_config.get("ata_worsening_cap", 0.10)),
                -ata_improvement_delta
                * float(reward_config.get("ata_worsening_scale", 0.025)),
            )

    wez_min_m = float(wez_config["min_range_m"])
    wez_max_m = float(wez_config["max_range_m"])
    in_wez_range = wez_min_m <= distance_m <= wez_max_m

    wez_reward = 0.0
    if in_wez_range and ata_deg <= 5.0:
        aim_quality = 0.60 + 0.40 * (1.0 - _clamp01(ata_deg / 5.0))
        wez_reward = (
            float(reward_config.get("stable_wez_bonus", 0.08))
            * aim_quality
            * stability
        )
    elif in_wez_range and ata_deg > float(
        reward_config.get("wez_misaligned_start_deg", 10.0)
    ):
        wez_reward = -float(
            reward_config.get("wez_misaligned_penalty", 0.05)
        ) * _clamp01((ata_deg - 10.0) / 20.0)

    approach_reward = 0.0
    if previous is not None and distance_progress_delta > 0.0:
        progress_scale = _clamp01(
            distance_progress_delta
            / max(0.1, float(reward_config.get("approach_progress_m", 2.0)))
        )
        stable_approach = (
            ata_improvement_delta > 0.0
            and abs_roll_deg <= float(reward_config.get("stable_roll_deg", 15.0))
            and abs_pitch_deg <= float(reward_config.get("stable_pitch_deg", 5.0))
            and altitude_m
            >= float(reward_config.get("low_altitude_strong_m", 6900.0))
            and alt_delta >= -0.5
        )
        bad_approach = (
            ata_improvement_delta < 0.0
            or abs_roll_deg
            > float(reward_config.get("roll_penalty_strong_deg", 45.0))
            or abs_pitch_deg
            > float(reward_config.get("pitch_penalty_strong_deg", 15.0))
            or altitude_m
            < float(reward_config.get("low_altitude_strong_m", 6900.0))
            or alt_delta < -1.0
        )
        if stable_approach:
            approach_reward = float(
                reward_config.get("good_approach_bonus", 0.04)
            ) * progress_scale
        elif bad_approach:
            approach_reward = -float(
                reward_config.get("bad_approach_penalty", 0.06)
            ) * progress_scale

    pursuit_reward = aim_reward + ata_change_reward + wez_reward + approach_reward

    # ------------------------------------------------------------------
    # Altitude, roll, and pitch trend recovery plus absolute safety bands.
    # ------------------------------------------------------------------
    safety_reward = 0.0
    target_altitude = float(reward_config.get("target_altitude_m", 7000.0))

    if previous is not None:
        previous_altitude = float(previous["altitude_m"])
        previous_altitude_error = abs(previous_altitude - target_altitude)
        current_altitude_error = abs(altitude_m - target_altitude)

        if alt_delta < 0.0:
            safety_reward -= min(
                float(reward_config.get("descent_penalty_cap", 0.12)),
                -alt_delta
                * float(reward_config.get("descent_penalty_scale", 0.030)),
            )
        if current_altitude_error < previous_altitude_error:
            safety_reward += min(
                float(reward_config.get("altitude_recovery_cap", 0.08)),
                (previous_altitude_error - current_altitude_error)
                * float(reward_config.get("altitude_recovery_scale", 0.015)),
            )

        if roll_recovery_delta > 0.0:
            safety_reward += min(
                float(reward_config.get("roll_recovery_cap", 0.08)),
                roll_recovery_delta
                * float(reward_config.get("roll_recovery_scale", 0.015)),
            )
        elif roll_recovery_delta < 0.0:
            safety_reward -= min(
                float(reward_config.get("roll_worsening_cap", 0.10)),
                -roll_recovery_delta
                * float(reward_config.get("roll_worsening_scale", 0.020)),
            )

        if pitch_recovery_delta > 0.0:
            safety_reward += min(
                float(reward_config.get("pitch_recovery_cap", 0.06)),
                pitch_recovery_delta
                * float(reward_config.get("pitch_recovery_scale", 0.015)),
            )
        elif pitch_recovery_delta < 0.0:
            safety_reward -= min(
                float(reward_config.get("pitch_worsening_cap", 0.08)),
                -pitch_recovery_delta
                * float(reward_config.get("pitch_worsening_scale", 0.020)),
            )

    warning_altitude = float(
        reward_config.get("low_altitude_warning_m", 6950.0)
    )
    strong_altitude = float(
        reward_config.get("low_altitude_strong_m", 6900.0)
    )
    minimum_altitude = float(
        reward_config.get("episode_min_altitude_m", 6550.0)
    )
    if strong_altitude <= altitude_m < warning_altitude:
        safety_reward -= float(
            reward_config.get("low_altitude_warning_penalty", 0.05)
        ) * _linear_progress(altitude_m, warning_altitude, strong_altitude)
    elif altitude_m < strong_altitude:
        low_factor = _linear_progress(
            altitude_m,
            strong_altitude,
            minimum_altitude,
        )
        safety_reward -= 0.15 + float(
            reward_config.get("low_altitude_strong_penalty", 1.00)
        ) * low_factor

    stable_roll = float(reward_config.get("stable_roll_deg", 15.0))
    strong_roll = float(reward_config.get("roll_penalty_strong_deg", 45.0))
    very_strong_roll = float(
        reward_config.get("roll_penalty_very_strong_deg", 55.0)
    )
    if stable_roll < abs_roll_deg <= strong_roll:
        safety_reward -= float(
            reward_config.get("roll_15_45_penalty", 0.30)
        ) * _linear_progress(abs_roll_deg, stable_roll, strong_roll)
    elif strong_roll < abs_roll_deg <= very_strong_roll:
        safety_reward -= float(
            reward_config.get("roll_15_45_penalty", 0.30)
        )
        safety_reward -= float(
            reward_config.get("roll_45_55_penalty", 0.40)
        ) * _linear_progress(abs_roll_deg, strong_roll, very_strong_roll)
    elif abs_roll_deg > very_strong_roll:
        safety_reward -= float(
            reward_config.get("roll_15_45_penalty", 0.30)
        )
        safety_reward -= float(
            reward_config.get("roll_45_55_penalty", 0.40)
        )
        safety_reward -= float(
            reward_config.get("roll_over_55_penalty", 0.90)
        ) * (0.50 + 0.50 * _clamp01((abs_roll_deg - very_strong_roll) / 35.0))

    stable_pitch = float(reward_config.get("stable_pitch_deg", 5.0))
    strong_pitch = float(
        reward_config.get("pitch_penalty_strong_deg", 15.0)
    )
    if stable_pitch < abs_pitch_deg <= strong_pitch:
        safety_reward -= float(
            reward_config.get("pitch_absolute_penalty", 0.25)
        ) * _linear_progress(abs_pitch_deg, stable_pitch, strong_pitch)
    elif abs_pitch_deg > strong_pitch:
        safety_reward -= float(
            reward_config.get("pitch_absolute_penalty", 0.25)
        )
        safety_reward -= float(
            reward_config.get("pitch_extreme_penalty", 0.35)
        ) * _clamp01((abs_pitch_deg - strong_pitch) / 15.0)

    if previous is not None and pitch_deg < float(
        reward_config.get("dive_pitch_start_deg", -2.0)
    ) and alt_delta < 0.0:
        dive_factor = _linear_progress(
            pitch_deg,
            float(reward_config.get("dive_pitch_start_deg", -2.0)),
            float(reward_config.get("dive_pitch_full_deg", -10.0)),
        )
        descent_factor = _clamp01(-alt_delta / 2.0)
        safety_reward -= float(
            reward_config.get("dive_descent_penalty", 0.35)
        ) * dive_factor * (0.50 + 0.50 * descent_factor)

    safety_reward = max(
        float(reward_config.get("safety_reward_min", -3.0)),
        safety_reward,
    )

    terminal_reward = 0.0
    if terminated:
        if end_condition == "target destroyed":
            terminal_reward = float(reward_config.get("win_reward", 150.0))
        elif end_condition == "target altitude below min":
            terminal_reward = float(
                reward_config.get("target_crash_reward", 0.0)
            )
        elif end_condition == "ownship destroyed":
            terminal_reward = float(reward_config.get("loss_reward", -200.0))
        elif end_condition in (
            "ownship altitude below min",
            "FDM Update Fail",
            "Ownship FDM output Fall",
        ):
            terminal_reward = float(
                reward_config.get("crash_penalty", -300.0)
            )
        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(
                reward_config.get("guard_fail_penalty", -80.0)
            )
        else:
            terminal_reward = float(reward_config.get("draw_reward", 0.0))
    elif truncated:
        terminal_reward = float(
            reward_config.get("timeout_penalty", 0.0)
        )

    step_reward = float(reward_config.get("step_penalty", -0.005))
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
