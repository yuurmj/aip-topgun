# -*- coding: utf-8 -*-
"""PPO v22 reward for a general-dogfight bridge.

The v21 firing-cone logic is retained, but tail-chase-specific flight shaping is
relaxed for maneuvering combat.  This reward uses only the confirmed StateIndex
values ALT, ROLL, and PITCH.  Distance and ATA come from the environment's
existing GeometryInfo methods; previous-step values are kept per environment in
reward_config.

Design priorities:
  1. Damage the opponent and avoid taking damage.
  2. Reacquire/drive ATA down, especially after a merge (rear hemisphere).
  3. Enter and hold the simulator's real firing cone.
  4. Avoid low-altitude uncontrolled dives/crashes without punishing normal
     combat bank/pitch angles at safe altitude.
  5. Keep approach shaping small so simple closure cannot dominate combat.
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


_RUNTIME_STATE_KEY = "__v22_general_previous_step"


MY_REWARD_CONFIG = {
    # Do not let survival/step accumulation dominate combat outcomes.
    "step_penalty": -0.002,
    "survival_bonus": 0.0,

    # Combat outcome is the dominant dense signal.
    "target_damage_scale": 350.0,
    "ownship_damage_scale": 300.0,
    "firing_confirmation_bonus": 0.03,

    # Use the simulator's actual weapon half-angle (wez.angle_deg / 2).
    "firing_cone_bonus_max": 0.18,
    "firing_cone_edge_bonus": 0.12,
    "near_cone_outer_deg": 3.0,
    "near_cone_outer_bonus": 0.015,
    "aim_fade_deg": 8.0,
    "stable_firing_cone_bonus": 0.06,

    # ATA trend works everywhere, not only in WEZ.  Rear-hemisphere recovery
    # gets a little more weight because the official duel exposed a merge /
    # reacquisition failure in v21.
    "ata_improvement_scale": 0.012,
    "ata_improvement_cap": 0.07,
    "rear_hemisphere_deg": 90.0,
    "rear_ata_multiplier": 1.8,
    "rear_ata_cap": 0.12,
    "ata_worsening_scale": 0.010,
    "ata_worsening_cap": 0.06,

    # WEZ misalignment and approach shaping remain deliberately small.
    "wez_misaligned_start_deg": 25.0,
    "wez_misaligned_penalty": 0.03,
    "approach_progress_m": 10.0,
    "approach_ata_limit_deg": 20.0,
    "good_approach_bonus": 0.02,
    "bad_approach_penalty": 0.025,

    # General dogfight safety: do not force the aircraft to remain at 7,000 m.
    # Tactical vertical maneuvering is allowed at safe altitude.  Strong shaping
    # starts only when the aircraft is getting genuinely low.
    "descent_watch_altitude_m": 5000.0,
    "low_altitude_warning_m": 3500.0,
    "low_altitude_critical_m": 1500.0,
    "low_altitude_warning_penalty": 0.12,
    "low_altitude_critical_penalty": 0.70,
    "low_altitude_recovery_scale": 0.010,
    "low_altitude_recovery_cap": 0.06,
    "low_altitude_descent_scale": 0.006,
    "low_altitude_descent_cap": 0.06,

    # Bank/pitch angles themselves are not failures in a dogfight.  Penalize
    # only the v21-style dangerous combination: low altitude + sustained dive,
    # with a small extra term for extreme bank while descending low.
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
    # Do not reward exploiting a specific BT's tendency to hit the floor.
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


def _combat_stability_score(
    abs_roll_deg: float,
    abs_pitch_deg: float,
    altitude_m: float,
) -> float:
    """Soft weapon-quality factor that still permits hard combat maneuvering."""
    # Up to ~90 deg bank and ~40 deg pitch are not treated as inherently bad.
    roll_score = 1.0 - _clamp01(max(0.0, abs_roll_deg - 90.0) / 90.0)
    pitch_score = 1.0 - _clamp01(max(0.0, abs_pitch_deg - 40.0) / 50.0)
    altitude_score = _clamp01((altitude_m - 800.0) / 2200.0)
    return _clamp01(0.35 * roll_score + 0.35 * pitch_score + 0.30 * altitude_score)


def _attack_posture_reward(
    ata_deg: float,
    firing_half_angle_deg: float,
    stability: float,
    reward_config: dict,
) -> float:
    cone_deg = max(0.1, float(firing_half_angle_deg))
    max_bonus = float(reward_config.get("firing_cone_bonus_max", 0.18))
    edge_bonus = float(reward_config.get("firing_cone_edge_bonus", 0.12))
    near_outer_deg = max(
        cone_deg,
        float(reward_config.get("near_cone_outer_deg", 3.0)),
    )
    near_outer_bonus = float(reward_config.get("near_cone_outer_bonus", 0.015))
    fade_deg = max(
        near_outer_deg,
        float(reward_config.get("aim_fade_deg", 8.0)),
    )

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

    # Keep weapon shaping useful during banked turns; stability only modulates it.
    return raw * (0.75 + 0.25 * stability)


def _load_previous_state(
    reward_config: dict,
    *,
    distance_m: float,
    altitude_m: float,
) -> dict | None:
    previous = reward_config.get(_RUNTIME_STATE_KEY)
    if not isinstance(previous, dict):
        return None
    # Reset guards for simulator/episode discontinuities.
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
    episode_ended: bool,
) -> None:
    if episode_ended:
        reward_config.pop(_RUNTIME_STATE_KEY, None)
        return
    reward_config[_RUNTIME_STATE_KEY] = {
        "distance_m": float(distance_m),
        "ata_deg": float(ata_deg),
        "altitude_m": float(altitude_m),
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
    """Compute the v22 general-dogfight bridge reward."""
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

    # Confirmed StateIndex values only.
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
    ata_improvement_delta = 0.0
    distance_progress_delta = 0.0
    previous_ata = ata_deg
    if previous is not None:
        previous_ata = float(previous["ata_deg"])
        alt_delta = altitude_m - float(previous["altitude_m"])
        ata_improvement_delta = previous_ata - ata_deg
        distance_progress_delta = float(previous["distance_m"]) - distance_m

    stability = _combat_stability_score(
        abs_roll_deg,
        abs_pitch_deg,
        altitude_m,
    )

    # ------------------------------------------------------------------
    # Priority 1: deal damage, avoid taking damage.
    # ------------------------------------------------------------------
    damage_reward = (
        float(reward_config.get("target_damage_scale", 350.0)) * float(target_damage)
        - float(reward_config.get("ownship_damage_scale", 300.0)) * float(ownship_damage)
    )
    if float(target_damage) > 0.0:
        damage_reward += float(reward_config.get("firing_confirmation_bonus", 0.03))

    # ------------------------------------------------------------------
    # Priority 2: reacquire and enter the actual firing cone.
    # ------------------------------------------------------------------
    firing_half_angle_deg = max(0.1, float(wez_config["angle_deg"]) / 2.0)
    aim_reward = _attack_posture_reward(
        ata_deg,
        firing_half_angle_deg,
        stability,
        reward_config,
    )

    ata_change_reward = 0.0
    if previous is not None:
        if ata_improvement_delta > 0.0:
            scale = float(reward_config.get("ata_improvement_scale", 0.012))
            cap = float(reward_config.get("ata_improvement_cap", 0.07))
            if previous_ata >= float(reward_config.get("rear_hemisphere_deg", 90.0)):
                scale *= float(reward_config.get("rear_ata_multiplier", 1.8))
                cap = float(reward_config.get("rear_ata_cap", 0.12))
            ata_change_reward = min(cap, ata_improvement_delta * scale)
        elif ata_improvement_delta < 0.0:
            ata_change_reward = -min(
                float(reward_config.get("ata_worsening_cap", 0.06)),
                -ata_improvement_delta
                * float(reward_config.get("ata_worsening_scale", 0.010)),
            )

    wez_min_m = float(wez_config["min_range_m"])
    wez_max_m = float(wez_config["max_range_m"])
    in_wez_range = wez_min_m <= distance_m <= wez_max_m

    wez_reward = 0.0
    if in_wez_range and ata_deg <= firing_half_angle_deg:
        aim_quality = 0.60 + 0.40 * (
            1.0 - _clamp01(ata_deg / firing_half_angle_deg)
        )
        wez_reward = (
            float(reward_config.get("stable_firing_cone_bonus", 0.06))
            * aim_quality
            * (0.75 + 0.25 * stability)
        )
    elif in_wez_range and ata_deg > float(
        reward_config.get("wez_misaligned_start_deg", 25.0)
    ):
        start = float(reward_config.get("wez_misaligned_start_deg", 25.0))
        wez_reward = -float(
            reward_config.get("wez_misaligned_penalty", 0.03)
        ) * _clamp01((ata_deg - start) / max(1.0, 90.0 - start))

    # ------------------------------------------------------------------
    # Priority 3: approach only when the nose geometry is already useful.
    # Pure head-on closure or blind distance reduction earns nothing.
    # ------------------------------------------------------------------
    approach_reward = 0.0
    if previous is not None and distance_progress_delta > 0.0:
        progress_scale = _clamp01(
            distance_progress_delta
            / max(0.1, float(reward_config.get("approach_progress_m", 10.0)))
        )
        useful_ata = ata_deg <= float(
            reward_config.get("approach_ata_limit_deg", 20.0)
        )
        if useful_ata and ata_improvement_delta > 0.0:
            approach_reward = float(
                reward_config.get("good_approach_bonus", 0.02)
            ) * progress_scale
        elif useful_ata and ata_improvement_delta < 0.0:
            approach_reward = -float(
                reward_config.get("bad_approach_penalty", 0.025)
            ) * progress_scale

    pursuit_reward = aim_reward + ata_change_reward + wez_reward + approach_reward

    # ------------------------------------------------------------------
    # General-dogfight safety.  Maneuvering at safe altitude is not punished.
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
            safety_reward -= float(
                reward_config.get("low_altitude_warning_penalty", 0.12)
            )
            safety_reward -= float(
                reward_config.get("low_altitude_critical_penalty", 0.70)
            ) * _clamp01((critical_alt - altitude_m) / max(1.0, critical_alt - 300.0))

    if previous is not None and altitude_m < watch_alt:
        if alt_delta < 0.0:
            safety_reward -= min(
                float(reward_config.get("low_altitude_descent_cap", 0.06)),
                -alt_delta
                * float(reward_config.get("low_altitude_descent_scale", 0.006)),
            )
        elif alt_delta > 0.0 and altitude_m < warning_alt:
            safety_reward += min(
                float(reward_config.get("low_altitude_recovery_cap", 0.06)),
                alt_delta
                * float(reward_config.get("low_altitude_recovery_scale", 0.010)),
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
                (abs_roll_deg - float(reward_config.get("extreme_roll_deg", 110.0)))
                / 70.0
            )
            safety_reward -= float(
                reward_config.get("extreme_roll_descent_penalty", 0.10)
            ) * roll_factor

    safety_reward = max(
        float(reward_config.get("safety_reward_min", -1.5)),
        safety_reward,
    )

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
        episode_ended=bool(terminated or truncated),
    )
    return float(total_reward), components


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
