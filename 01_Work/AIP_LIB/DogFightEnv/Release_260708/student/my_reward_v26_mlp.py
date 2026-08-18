# -*- coding: utf-8 -*-
"""Restore-compatible PPO+MLP v26 reward for legacy full-dogfight training.

The policy input and network are deliberately unchanged from v21
(``tactical16`` and shared ``[256, 256]`` MLP), so the independently validated
v21 native checkpoint can be restored.  This reward adds the missing
full-engagement signals without making blind closure more valuable than
survival or weapon geometry:

* actual damage and the simulator's exact firing cone remain dominant;
* target ATA rewards positional advantage, not just pointing at the target;
* closure is useful when far away but is penalized inside the WEZ when it risks
  an overshoot;
* altitude, time-to-floor, dive, and low-speed risks gate combat shaping before
  the crash terminal is reached.
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


_RUNTIME_STATE_KEY = "__v26_mlp_previous_step"


MY_REWARD_CONFIG = {
    "step_seconds": 0.1,
    "step_penalty": -0.001,
    "survival_bonus": 0.0,
    # Curriculum stages may lower combat shaping without changing the common
    # trainer or duplicating the reward function.
    "combat_reward_scale": 1.0,
    "damage_reward_scale": 1.0,

    # Outcomes and real weapon effect dominate dense shaping.
    "target_damage_scale": 350.0,
    "ownship_damage_scale": 400.0,
    "firing_confirmation_bonus": 0.03,
    "firing_cone_bonus_max": 0.16,
    "firing_cone_edge_bonus": 0.10,
    "near_cone_outer_deg": 3.0,
    "near_cone_outer_bonus": 0.012,
    "aim_fade_deg": 8.0,
    "stable_firing_cone_bonus": 0.05,
    "wez_streak_2_bonus": 0.10,
    "wez_streak_5_bonus": 0.25,
    "wez_streak_10_bonus": 0.50,

    # Reacquisition and positional advantage.
    "ata_improvement_scale": 0.010,
    "ata_improvement_cap": 0.06,
    "ata_worsening_scale": 0.008,
    "ata_worsening_cap": 0.05,
    "positional_advantage_scale": 0.08,
    "rear_aspect_bonus": 0.04,

    # Closure is conditional and bounded. Values are m/s at the 10 Hz RL rate.
    "far_range_m": 1200.0,
    "closure_ata_limit_deg": 45.0,
    "closure_target_mps": 80.0,
    "far_closure_bonus_max": 0.025,
    "wez_high_closure_mps": 60.0,
    "wez_high_closure_penalty": 0.06,
    "overshoot_ata_jump_deg": 25.0,
    "overshoot_penalty": 0.10,

    # Risk starts early enough to teach recovery before the 300 m hard floor.
    "descent_watch_altitude_m": 6000.0,
    "low_altitude_warning_m": 3500.0,
    "low_altitude_critical_m": 1500.0,
    "altitude_warning_penalty": 0.10,
    "altitude_critical_penalty": 0.75,
    "time_to_floor_warning_s": 12.0,
    "time_to_floor_critical_s": 5.0,
    "time_to_floor_penalty": 0.45,
    "descent_penalty_scale": 0.003,
    "descent_penalty_cap": 0.12,
    "recovery_bonus_scale": 0.002,
    "recovery_bonus_cap": 0.05,
    "dive_pitch_start_deg": -15.0,
    "dive_pitch_full_deg": -45.0,
    "dive_penalty": 0.30,
    "extreme_roll_deg": 120.0,
    "extreme_roll_penalty": 0.08,
    "low_speed_warning_kcas": 180.0,
    "low_speed_critical_kcas": 120.0,
    "low_speed_penalty": 0.20,
    "safety_reward_min": -1.75,
    "combat_gate_min": 0.10,

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


def _progress(value: float, start: float, full: float) -> float:
    span = float(full) - float(start)
    if span == 0.0:
        return float(value == full)
    return _clamp01((float(value) - float(start)) / span)


def _attack_posture_reward(
    ata_deg: float,
    firing_half_angle_deg: float,
    reward_config: dict,
) -> float:
    cone = max(0.1, float(firing_half_angle_deg))
    near = max(cone, float(reward_config.get("near_cone_outer_deg", 3.0)))
    fade = max(near, float(reward_config.get("aim_fade_deg", 8.0)))
    max_bonus = float(reward_config.get("firing_cone_bonus_max", 0.16))
    edge_bonus = float(reward_config.get("firing_cone_edge_bonus", 0.10))
    near_bonus = float(reward_config.get("near_cone_outer_bonus", 0.012))

    if ata_deg <= cone:
        quality = 1.0 - _clamp01(ata_deg / cone)
        return edge_bonus + (max_bonus - edge_bonus) * quality
    if ata_deg <= near:
        blend = _clamp01((ata_deg - cone) / max(1e-6, near - cone))
        return edge_bonus + (near_bonus - edge_bonus) * blend
    if ata_deg < fade:
        return near_bonus * (1.0 - _clamp01((ata_deg - near) / (fade - near)))
    return 0.0


def _previous_state(reward_config: dict, sim_time: float) -> dict | None:
    previous = reward_config.get(_RUNTIME_STATE_KEY)
    if not isinstance(previous, dict):
        return None
    previous_time = float(previous.get("sim_time", sim_time))
    if sim_time <= previous_time or sim_time - previous_time > 1.0:
        return None
    return previous


def _store_state(
    reward_config: dict,
    *,
    sim_time: float,
    distance_m: float,
    own_ata_deg: float,
    target_ata_deg: float,
    altitude_m: float,
    wez_streak: int,
    episode_ended: bool,
) -> None:
    if episode_ended:
        reward_config.pop(_RUNTIME_STATE_KEY, None)
        return
    reward_config[_RUNTIME_STATE_KEY] = {
        "sim_time": float(sim_time),
        "distance_m": float(distance_m),
        "own_ata_deg": float(own_ata_deg),
        "target_ata_deg": float(target_ata_deg),
        "altitude_m": float(altitude_m),
        "wez_streak": int(wez_streak),
    }


def _terminal_reward(
    terminated: bool,
    truncated: bool,
    end_condition: str,
    reward_config: dict,
) -> float:
    if terminated:
        mapping = {
            "target destroyed": "win_reward",
            "target altitude below min": "target_crash_reward",
            "ownship destroyed": "loss_reward",
            "ownship altitude below min": "crash_penalty",
            "FDM Update Fail": "crash_penalty",
            "Ownship FDM output Fall": "crash_penalty",
            "two circle headon guard fail": "guard_fail_penalty",
        }
        key = mapping.get(end_condition, "draw_reward")
        defaults = {
            "win_reward": 200.0,
            "target_crash_reward": 0.0,
            "loss_reward": -250.0,
            "crash_penalty": -300.0,
            "guard_fail_penalty": -80.0,
            "draw_reward": 0.0,
        }
        return float(reward_config.get(key, defaults[key]))
    if truncated:
        return float(reward_config.get("timeout_penalty", 0.0))
    return 0.0


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
    """Compute v26 reward and named components for RLlib callbacks."""
    distance_m = float(geo_info._get_distance(ownship_state, target_state))
    own_ata_deg = abs(float(geo_info._get_antenna_train_angle(
        ownship_state, target_state, False
    )))
    target_ata_deg = abs(float(geo_info._get_antenna_train_angle(
        target_state, ownship_state, False
    )))
    altitude_m = float(ownship_state[StateIndex.ALT])
    roll_deg = abs(float(ownship_state[StateIndex.ROLL]))
    pitch_deg = float(ownship_state[StateIndex.PITCH])
    speed_kcas = float(ownship_state[StateIndex.KCAS])
    sim_time = float(ownship_state[StateIndex.SIM_TIME])
    previous = _previous_state(reward_config, sim_time)

    dt = max(1e-3, float(reward_config.get("step_seconds", 0.1)))
    altitude_delta = 0.0
    closure_mps = 0.0
    own_ata_delta = 0.0
    previous_ata = own_ata_deg
    if previous is not None:
        altitude_delta = altitude_m - float(previous["altitude_m"])
        closure_mps = (float(previous["distance_m"]) - distance_m) / dt
        previous_ata = float(previous["own_ata_deg"])
        own_ata_delta = previous_ata - own_ata_deg

    firing_half_angle = max(0.1, float(wez_config["angle_deg"]) / 2.0)
    min_range = float(wez_config["min_range_m"])
    max_range = float(wez_config["max_range_m"])
    in_wez_range = min_range <= distance_m <= max_range
    in_firing_cone = in_wez_range and own_ata_deg <= firing_half_angle
    wez_streak = int(previous.get("wez_streak", 0)) + 1 if (
        previous is not None and in_firing_cone
    ) else int(in_firing_cone)

    damage_reward = (
        float(reward_config.get("target_damage_scale", 350.0)) * float(target_damage)
        - float(reward_config.get("ownship_damage_scale", 400.0)) * float(ownship_damage)
    )
    if target_damage > 0.0:
        damage_reward += float(reward_config.get("firing_confirmation_bonus", 0.03))
    damage_reward *= float(reward_config.get("damage_reward_scale", 1.0))

    aim_reward = _attack_posture_reward(
        own_ata_deg, firing_half_angle, reward_config
    )
    if in_firing_cone:
        aim_reward += float(reward_config.get("stable_firing_cone_bonus", 0.05))
        milestones = {
            2: "wez_streak_2_bonus",
            5: "wez_streak_5_bonus",
            10: "wez_streak_10_bonus",
        }
        if wez_streak in milestones:
            aim_reward += float(reward_config.get(milestones[wez_streak], 0.0))

    ata_trend_reward = 0.0
    if previous is not None:
        if own_ata_delta >= 0.0:
            ata_trend_reward = min(
                float(reward_config.get("ata_improvement_cap", 0.06)),
                own_ata_delta * float(reward_config.get("ata_improvement_scale", 0.010)),
            )
        else:
            ata_trend_reward = -min(
                float(reward_config.get("ata_worsening_cap", 0.05)),
                -own_ata_delta * float(reward_config.get("ata_worsening_scale", 0.008)),
            )

    own_pointing = _clamp01((90.0 - own_ata_deg) / 90.0)
    target_not_pointing = _clamp01(target_ata_deg / 90.0)
    positional_reward = float(
        reward_config.get("positional_advantage_scale", 0.08)
    ) * own_pointing * target_not_pointing
    if own_ata_deg <= 30.0 and target_ata_deg >= 120.0:
        positional_reward += float(reward_config.get("rear_aspect_bonus", 0.04))

    closure_reward = 0.0
    if previous is not None:
        if (
            distance_m > float(reward_config.get("far_range_m", 1200.0))
            and own_ata_deg <= float(reward_config.get("closure_ata_limit_deg", 45.0))
            and closure_mps > 0.0
        ):
            closure_reward += float(
                reward_config.get("far_closure_bonus_max", 0.025)
            ) * _clamp01(
                closure_mps / float(reward_config.get("closure_target_mps", 80.0))
            )
        if in_wez_range and closure_mps > float(
            reward_config.get("wez_high_closure_mps", 60.0)
        ):
            closure_reward -= float(
                reward_config.get("wez_high_closure_penalty", 0.06)
            ) * _clamp01(
                (closure_mps - float(reward_config.get("wez_high_closure_mps", 60.0)))
                / 100.0
            )
        ata_jump = own_ata_deg - previous_ata
        if (
            distance_m < min_range
            or (
                closure_mps > 0.0
                and previous_ata < 15.0
                and ata_jump > float(reward_config.get("overshoot_ata_jump_deg", 25.0))
            )
        ):
            closure_reward -= float(reward_config.get("overshoot_penalty", 0.10))

    # Compute risk and safety shaping before combat gating.
    warning_alt = float(reward_config.get("low_altitude_warning_m", 3500.0))
    critical_alt = float(reward_config.get("low_altitude_critical_m", 1500.0))
    floor_alt = 300.0
    altitude_risk = _progress(altitude_m, warning_alt, critical_alt)
    if altitude_m < critical_alt:
        altitude_risk = max(altitude_risk, _progress(altitude_m, critical_alt, floor_alt))

    vertical_speed_mps = altitude_delta / dt if previous is not None else 0.0
    time_to_floor = float("inf")
    if vertical_speed_mps < -1.0:
        time_to_floor = max(0.0, (altitude_m - floor_alt) / -vertical_speed_mps)
    ttf_warning = float(reward_config.get("time_to_floor_warning_s", 12.0))
    ttf_critical = float(reward_config.get("time_to_floor_critical_s", 5.0))
    time_risk = _progress(time_to_floor, ttf_warning, ttf_critical)
    low_speed_risk = _progress(
        speed_kcas,
        float(reward_config.get("low_speed_warning_kcas", 180.0)),
        float(reward_config.get("low_speed_critical_kcas", 120.0)),
    )
    risk = max(altitude_risk, time_risk, low_speed_risk)
    combat_gate = max(
        float(reward_config.get("combat_gate_min", 0.10)),
        1.0 - 0.90 * risk,
    )

    safety_reward = 0.0
    if altitude_m < warning_alt:
        safety_reward -= float(
            reward_config.get("altitude_warning_penalty", 0.10)
        ) * altitude_risk
        if altitude_m < critical_alt:
            safety_reward -= float(
                reward_config.get("altitude_critical_penalty", 0.75)
            ) * _progress(altitude_m, critical_alt, floor_alt)
    safety_reward -= float(
        reward_config.get("time_to_floor_penalty", 0.45)
    ) * time_risk
    safety_reward -= float(
        reward_config.get("low_speed_penalty", 0.20)
    ) * low_speed_risk

    watch_alt = float(reward_config.get("descent_watch_altitude_m", 6000.0))
    if previous is not None and altitude_m < watch_alt:
        if altitude_delta < 0.0:
            safety_reward -= min(
                float(reward_config.get("descent_penalty_cap", 0.12)),
                -altitude_delta * float(reward_config.get("descent_penalty_scale", 0.003)),
            )
        elif altitude_delta > 0.0 and risk > 0.0:
            safety_reward += min(
                float(reward_config.get("recovery_bonus_cap", 0.05)),
                altitude_delta * float(reward_config.get("recovery_bonus_scale", 0.002)),
            )
        if altitude_delta < 0.0 and pitch_deg < float(
            reward_config.get("dive_pitch_start_deg", -15.0)
        ):
            dive_factor = _progress(
                pitch_deg,
                float(reward_config.get("dive_pitch_start_deg", -15.0)),
                float(reward_config.get("dive_pitch_full_deg", -45.0)),
            )
            safety_reward -= float(reward_config.get("dive_penalty", 0.30)) * dive_factor
        if altitude_delta < 0.0 and roll_deg > float(
            reward_config.get("extreme_roll_deg", 120.0)
        ):
            roll_factor = _clamp01(
                (roll_deg - float(reward_config.get("extreme_roll_deg", 120.0))) / 60.0
            )
            safety_reward -= float(
                reward_config.get("extreme_roll_penalty", 0.08)
            ) * roll_factor

    safety_reward = max(
        float(reward_config.get("safety_reward_min", -1.75)), safety_reward
    )
    pursuit_reward = float(reward_config.get("combat_reward_scale", 1.0)) * combat_gate * (
        aim_reward + ata_trend_reward + positional_reward + closure_reward
    )
    terminal_reward = _terminal_reward(
        terminated, truncated, end_condition, reward_config
    )
    step_reward = float(reward_config.get("step_penalty", -0.001))
    survival_reward = float(reward_config.get("survival_bonus", 0.0))
    total = (
        step_reward
        + survival_reward
        + damage_reward
        + pursuit_reward
        + safety_reward
        + terminal_reward
    )

    _store_state(
        reward_config,
        sim_time=sim_time,
        distance_m=distance_m,
        own_ata_deg=own_ata_deg,
        target_ata_deg=target_ata_deg,
        altitude_m=altitude_m,
        wez_streak=wez_streak,
        episode_ended=bool(terminated or truncated),
    )
    return float(total), {
        "step": float(step_reward),
        "survival": float(survival_reward),
        "pursuit": float(pursuit_reward),
        "damage": float(damage_reward),
        "safety": float(safety_reward),
        "terminal": float(terminal_reward),
    }


__all__ = ["MY_REWARD_CONFIG", "compute_reward"]
