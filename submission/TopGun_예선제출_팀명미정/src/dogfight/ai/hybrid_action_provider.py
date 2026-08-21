from __future__ import annotations

import math
from typing import Callable

import numpy as np

from dogfight.sim.state_schema import StateIndex

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action


def _body_velocity_to_ned(state) -> np.ndarray:
    """Rotate body-axis u/v/w velocity into the NED frame."""

    roll, pitch, yaw = (
        math.radians(float(state[StateIndex.ROLL])),
        math.radians(float(state[StateIndex.PITCH])),
        math.radians(float(state[StateIndex.YAW])),
    )
    sr, cr = math.sin(roll), math.cos(roll)
    sp, cp = math.sin(pitch), math.cos(pitch)
    sy, cy = math.sin(yaw), math.cos(yaw)
    rotation = np.asarray(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=np.float64,
    )
    body_velocity = np.asarray(
        [state[StateIndex.U], state[StateIndex.V], state[StateIndex.W]],
        dtype=np.float64,
    )
    return rotation @ body_velocity


class PursuitGuidanceActionProvider(ActionProvider):
    """Simple flight-safe pure-pursuit controller used as a curriculum teacher."""

    def __init__(
        self,
        *,
        max_bank_deg: float = 60.0,
        heading_to_bank: float = 1.2,
        roll_gain: float = 1.0 / 45.0,
        pitch_gain: float = 1.0 / 20.0,
        rudder_gain: float = 0.15,
        altitude_guard_m: float = 3000.0,
        recovery_pitch_deg: float = 15.0,
        lead_time_s: float = 0.0,
        pitch_command_sign: float = -1.0,
        brake_range_m: float = 0.0,
        near_throttle_action: float = -1.0,
        capture_range_m: float = 0.0,
        capture_heading_error_deg: float = 0.0,
        capture_bank_scale: float = 1.0,
        capture_rudder_gain: float | None = None,
        capture_lead_time_s: float | None = None,
        capture_maneuver_range_m: float = 0.0,
        capture_maneuver_target_pitch_deg: float = 180.0,
        capture_maneuver_target_bank_deg: float = 180.0,
        capture_maneuver_lead_time_s: float | None = None,
        capture_extreme_bank_deg: float = 180.0,
        capture_extreme_bank_lead_time_s: float | None = None,
        capture_slow_target_speed_mps: float = 0.0,
        capture_slow_target_range_m: float = 0.0,
        capture_slow_target_lead_time_s: float | None = None,
        heading_rate_to_bank: float = 0.0,
        capture_heading_bias_deg: float = 0.0,
        capture_bias_target_roll_sign: bool = False,
        trail_distance_m: float = 0.0,
        capture_target_bank_blend: float = 0.0,
        capture_target_bank_blend_start_ata_deg: float = 0.0,
        capture_target_bank_multiplier: float = 1.0,
        capture_heading_rate_to_bank: float | None = None,
        capture_roll_gain: float | None = None,
        latch_capture_mode: bool = False,
        wez_hold_latch_ata_deg: float = 0.0,
        wez_hold_target_bank_blend: float = 0.0,
        wez_hold_target_bank_multiplier: float | None = None,
        wez_hold_roll_gain: float | None = None,
        wez_hold_roll_delta_damping: float | None = None,
        wez_hold_freeze_bank: bool = False,
        wez_hold_pitch_gain: float | None = None,
        wez_hold_pitch_delta_damping: float | None = None,
        brake_heading_error_deg: float = 0.0,
        capture_pitch_gain: float | None = None,
        capture_heading_rate_to_rudder: float = 0.0,
        wez_hold_target_heading_blend: float = 0.0,
        capture_target_heading_blend: float = 0.0,
        capture_near_ata_deg: float = 0.0,
        capture_near_target_heading_blend: float | None = None,
        capture_target_pitch_blend: float = 0.0,
        wez_hold_target_pitch_blend: float = 0.0,
        capture_speed_gain: float = 0.0,
        capture_speed_offset_mps: float = 0.0,
        proximity_speed_match_start_distance_m: float = 0.0,
        proximity_speed_match_gain: float = 0.0,
        proximity_speed_match_min_overspeed_mps: float = 0.0,
        proximity_speed_match_max_ata_deg: float = 180.0,
        proximity_speed_match_latch: bool = False,
        rear_threat_distance_m: float = 0.0,
        rear_threat_ata_deg: float = 180.0,
        rear_threat_roll_gain: float = 0.0,
        rear_threat_bank_pitch_compensation_gain: float = 0.0,
        rear_threat_turn_latch: bool = False,
        rear_threat_turn_release_ata_deg: float = 90.0,
        rear_threat_roll_delta_damping: float = 0.0,
        rear_threat_rudder_gain: float = 0.0,
        target_bank_feedforward_range_m: float = 0.0,
        target_bank_feedforward_blend: float = 0.0,
        capture_roll_delta_damping: float = 0.0,
        bank_pitch_compensation_gain: float = 0.0,
        target_bank_heading_gain_baseline_deg: float = 0.0,
        target_bank_to_heading_gain: float = 0.0,
        target_bank_roll_damping_baseline_deg: float = 0.0,
        target_bank_to_roll_damping: float = 0.0,
        capture_pitch_delta_damping: float = 0.0,
        capture_pitch_error_rate_gain: float = 0.0,
        capture_body_axis_guidance: bool = False,
        capture_body_axis_latch_only: bool = False,
        capture_body_axis_roll_gain: float = 0.0,
        capture_body_axis_pitch_gain: float = 0.0,
        capture_body_axis_roll_rate_gain: float = 0.0,
        capture_body_axis_pitch_rate_gain: float = 0.0,
    ) -> None:
        self.max_bank_deg = float(max_bank_deg)
        self.heading_to_bank = float(heading_to_bank)
        self.roll_gain = float(roll_gain)
        self.pitch_gain = float(pitch_gain)
        self.rudder_gain = float(rudder_gain)
        self.altitude_guard_m = float(altitude_guard_m)
        self.recovery_pitch_deg = float(recovery_pitch_deg)
        self.lead_time_s = float(lead_time_s)
        self.pitch_command_sign = float(pitch_command_sign)
        self.brake_range_m = float(brake_range_m)
        self.near_throttle_action = float(near_throttle_action)
        self.capture_range_m = float(capture_range_m)
        self.capture_heading_error_deg = float(capture_heading_error_deg)
        self.capture_bank_scale = float(capture_bank_scale)
        self.capture_rudder_gain = (
            self.rudder_gain
            if capture_rudder_gain is None
            else float(capture_rudder_gain)
        )
        self.capture_lead_time_s = (
            self.lead_time_s
            if capture_lead_time_s is None
            else float(capture_lead_time_s)
        )
        self.capture_maneuver_range_m = max(
            float(capture_maneuver_range_m), 0.0
        )
        self.capture_maneuver_target_pitch_deg = max(
            float(capture_maneuver_target_pitch_deg), 0.0
        )
        self.capture_maneuver_target_bank_deg = max(
            float(capture_maneuver_target_bank_deg), 0.0
        )
        self.capture_maneuver_lead_time_s = (
            self.capture_lead_time_s
            if capture_maneuver_lead_time_s is None
            else float(capture_maneuver_lead_time_s)
        )
        self.capture_extreme_bank_deg = max(
            float(capture_extreme_bank_deg), 0.0
        )
        self.capture_extreme_bank_lead_time_s = (
            self.capture_maneuver_lead_time_s
            if capture_extreme_bank_lead_time_s is None
            else float(capture_extreme_bank_lead_time_s)
        )
        self.capture_slow_target_speed_mps = max(
            float(capture_slow_target_speed_mps), 0.0
        )
        self.capture_slow_target_range_m = max(
            float(capture_slow_target_range_m), 0.0
        )
        self.capture_slow_target_lead_time_s = (
            self.capture_lead_time_s
            if capture_slow_target_lead_time_s is None
            else float(capture_slow_target_lead_time_s)
        )
        self.heading_rate_to_bank = float(heading_rate_to_bank)
        self.capture_heading_bias_deg = float(capture_heading_bias_deg)
        self.capture_bias_target_roll_sign = bool(capture_bias_target_roll_sign)
        self.trail_distance_m = max(float(trail_distance_m), 0.0)
        self.capture_target_bank_blend = float(np.clip(
            capture_target_bank_blend, -1.0, 1.0
        ))
        self.capture_target_bank_blend_start_ata_deg = max(
            float(capture_target_bank_blend_start_ata_deg), 0.0
        )
        self.capture_target_bank_multiplier = max(
            float(capture_target_bank_multiplier), 0.0
        )
        self.capture_heading_rate_to_bank = (
            self.heading_rate_to_bank
            if capture_heading_rate_to_bank is None
            else float(capture_heading_rate_to_bank)
        )
        self.capture_roll_gain = (
            self.roll_gain
            if capture_roll_gain is None
            else float(capture_roll_gain)
        )
        self.latch_capture_mode = bool(latch_capture_mode)
        self.wez_hold_latch_ata_deg = max(float(wez_hold_latch_ata_deg), 0.0)
        self.wez_hold_target_bank_blend = float(np.clip(
            wez_hold_target_bank_blend, -1.0, 1.0
        ))
        self.wez_hold_target_bank_multiplier = (
            self.capture_target_bank_multiplier
            if wez_hold_target_bank_multiplier is None
            else max(float(wez_hold_target_bank_multiplier), 0.0)
        )
        self.wez_hold_roll_gain = (
            self.capture_roll_gain
            if wez_hold_roll_gain is None
            else float(wez_hold_roll_gain)
        )
        self.wez_hold_roll_delta_damping = (
            max(float(capture_roll_delta_damping), 0.0)
            if wez_hold_roll_delta_damping is None
            else max(float(wez_hold_roll_delta_damping), 0.0)
        )
        self.wez_hold_freeze_bank = bool(wez_hold_freeze_bank)
        self.capture_pitch_gain = (
            self.pitch_gain
            if capture_pitch_gain is None
            else float(capture_pitch_gain)
        )
        self.wez_hold_pitch_gain = (
            self.capture_pitch_gain
            if wez_hold_pitch_gain is None
            else float(wez_hold_pitch_gain)
        )
        self.wez_hold_pitch_delta_damping = (
            max(float(capture_pitch_delta_damping), 0.0)
            if wez_hold_pitch_delta_damping is None
            else max(float(wez_hold_pitch_delta_damping), 0.0)
        )
        self.brake_heading_error_deg = max(float(brake_heading_error_deg), 0.0)
        self.capture_heading_rate_to_rudder = float(
            capture_heading_rate_to_rudder
        )
        self.wez_hold_target_heading_blend = float(np.clip(
            wez_hold_target_heading_blend, 0.0, 1.0
        ))
        self.capture_target_heading_blend = float(np.clip(
            capture_target_heading_blend, 0.0, 1.0
        ))
        self.capture_near_ata_deg = max(float(capture_near_ata_deg), 0.0)
        self.capture_near_target_heading_blend = (
            self.capture_target_heading_blend
            if capture_near_target_heading_blend is None
            else float(np.clip(capture_near_target_heading_blend, 0.0, 1.0))
        )
        self.capture_target_pitch_blend = float(np.clip(
            capture_target_pitch_blend, 0.0, 1.0
        ))
        self.wez_hold_target_pitch_blend = float(np.clip(
            wez_hold_target_pitch_blend, 0.0, 1.0
        ))
        self.capture_speed_gain = max(float(capture_speed_gain), 0.0)
        self.capture_speed_offset_mps = float(capture_speed_offset_mps)
        self.proximity_speed_match_start_distance_m = max(
            float(proximity_speed_match_start_distance_m), 0.0
        )
        self.proximity_speed_match_gain = max(
            float(proximity_speed_match_gain), 0.0
        )
        self.proximity_speed_match_min_overspeed_mps = max(
            float(proximity_speed_match_min_overspeed_mps), 0.0
        )
        self.proximity_speed_match_max_ata_deg = float(np.clip(
            proximity_speed_match_max_ata_deg, 0.0, 180.0
        ))
        self.proximity_speed_match_latch = bool(proximity_speed_match_latch)
        self.rear_threat_distance_m = max(float(rear_threat_distance_m), 0.0)
        self.rear_threat_ata_deg = float(np.clip(rear_threat_ata_deg, 0.0, 180.0))
        self.rear_threat_roll_gain = max(float(rear_threat_roll_gain), 0.0)
        self.rear_threat_bank_pitch_compensation_gain = max(
            float(rear_threat_bank_pitch_compensation_gain), 0.0
        )
        self.rear_threat_turn_latch = bool(rear_threat_turn_latch)
        self.rear_threat_turn_release_ata_deg = float(np.clip(
            rear_threat_turn_release_ata_deg, 0.0, 180.0
        ))
        self.rear_threat_roll_delta_damping = max(
            float(rear_threat_roll_delta_damping), 0.0
        )
        self.rear_threat_rudder_gain = max(float(rear_threat_rudder_gain), 0.0)
        self.target_bank_feedforward_range_m = max(
            float(target_bank_feedforward_range_m), 0.0
        )
        self.target_bank_feedforward_blend = float(np.clip(
            target_bank_feedforward_blend, 0.0, 1.0
        ))
        self.capture_roll_delta_damping = max(
            float(capture_roll_delta_damping), 0.0
        )
        self.bank_pitch_compensation_gain = max(
            float(bank_pitch_compensation_gain), 0.0
        )
        self.target_bank_heading_gain_baseline_deg = max(
            float(target_bank_heading_gain_baseline_deg), 0.0
        )
        self.target_bank_to_heading_gain = float(target_bank_to_heading_gain)
        self.target_bank_roll_damping_baseline_deg = max(
            float(target_bank_roll_damping_baseline_deg), 0.0
        )
        self.target_bank_to_roll_damping = float(
            target_bank_to_roll_damping
        )
        self.capture_pitch_delta_damping = max(
            float(capture_pitch_delta_damping), 0.0
        )
        self.capture_pitch_error_rate_gain = float(capture_pitch_error_rate_gain)
        self.capture_body_axis_guidance = bool(capture_body_axis_guidance)
        self.capture_body_axis_latch_only = bool(capture_body_axis_latch_only)
        self.capture_body_axis_roll_gain = float(capture_body_axis_roll_gain)
        self.capture_body_axis_pitch_gain = float(capture_body_axis_pitch_gain)
        self.capture_body_axis_roll_rate_gain = float(
            capture_body_axis_roll_rate_gain
        )
        self.capture_body_axis_pitch_rate_gain = float(
            capture_body_axis_pitch_rate_gain
        )
        self._previous_heading_error: float | None = None
        self._previous_roll: float | None = None
        self._previous_pitch: float | None = None
        self._previous_pitch_error: float | None = None
        self._previous_body_azimuth: float | None = None
        self._previous_body_elevation: float | None = None
        self._capture_latched = False
        self._maneuver_capture_selected: bool | None = None
        self._extreme_bank_capture_selected = False
        self._slow_target_capture_selected: bool | None = None
        self._wez_hold_latched = False
        self._wez_hold_bank_deg: float | None = None
        self._proximity_speed_match_latched = False
        self._rear_threat_turn_sign = 0.0

    def reset(self, context: ActionContext | None = None) -> None:
        self._previous_heading_error = None
        self._previous_roll = None
        self._previous_pitch = None
        self._previous_pitch_error = None
        self._previous_body_azimuth = None
        self._previous_body_elevation = None
        self._capture_latched = False
        self._maneuver_capture_selected = None
        self._extreme_bank_capture_selected = False
        self._slow_target_capture_selected = None
        self._wez_hold_latched = False
        self._wez_hold_bank_deg = None
        self._proximity_speed_match_latched = False
        self._rear_threat_turn_sign = 0.0

    @staticmethod
    def _wrap_degrees(value: float) -> float:
        return (float(value) + 180.0) % 360.0 - 180.0

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None or context.target_state is None:
            raise ValueError("ownship_state and target_state are required for pursuit guidance")
        ownship = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        raw_delta = target[:3] - ownship[:3]
        distance_m = float(np.linalg.norm(raw_delta))
        delta = raw_delta.copy()
        target_pitch_magnitude = abs(self._wrap_degrees(
            float(target[StateIndex.PITCH])
        ))
        target_bank_magnitude = abs(self._wrap_degrees(
            float(target[StateIndex.ROLL])
        ))
        target_speed_mps = float(np.linalg.norm(target[
            StateIndex.U : StateIndex.W + 1
        ]))
        if (
            self.capture_slow_target_speed_mps > 0.0
            and self._slow_target_capture_selected is None
            and target_speed_mps >= 50.0
        ):
            self._slow_target_capture_selected = (
                target_speed_mps <= self.capture_slow_target_speed_mps
            )
        slow_target_capture_active = (
            self._slow_target_capture_selected is True
            and self.capture_slow_target_range_m > 0.0
            and distance_m <= self.capture_slow_target_range_m
        )
        if (
            self.capture_maneuver_range_m > 0.0
            and distance_m <= self.capture_maneuver_range_m
            and self._maneuver_capture_selected is None
        ):
            maneuver_condition = (
                target_pitch_magnitude >= self.capture_maneuver_target_pitch_deg
                or target_bank_magnitude >= self.capture_maneuver_target_bank_deg
            )
            if maneuver_condition:
                self._maneuver_capture_selected = True
                self._extreme_bank_capture_selected = (
                    target_bank_magnitude >= self.capture_extreme_bank_deg
                )
            elif distance_m <= self.capture_range_m:
                self._maneuver_capture_selected = False
        maneuver_capture_active = (
            self._maneuver_capture_selected is True
            and distance_m <= self.capture_maneuver_range_m
        )
        effective_capture_range_m = (
            self.capture_slow_target_range_m
            if slow_target_capture_active
            else
            self.capture_maneuver_range_m
            if maneuver_capture_active
            else self.capture_range_m
        )
        if self.trail_distance_m > 0.0:
            target_velocity = _body_velocity_to_ned(target)
            horizontal_target_speed = float(np.hypot(
                target_velocity[0], target_velocity[1]
            ))
            if horizontal_target_speed > 1.0:
                delta[:2] -= (
                    self.trail_distance_m
                    * target_velocity[:2]
                    / horizontal_target_speed
                )
        active_lead_time_s = (
            self.capture_slow_target_lead_time_s
            if slow_target_capture_active
            else self.capture_extreme_bank_lead_time_s
            if maneuver_capture_active
            and self._extreme_bank_capture_selected
            else self.capture_maneuver_lead_time_s
            if maneuver_capture_active
            else self.capture_lead_time_s
            if effective_capture_range_m > 0.0
            and distance_m <= effective_capture_range_m
            else self.lead_time_s
        )
        if active_lead_time_s != 0.0:
            relative_velocity = (
                _body_velocity_to_ned(target) - _body_velocity_to_ned(ownship)
            )
            delta = delta + active_lead_time_s * relative_velocity
        horizontal = max(float(np.hypot(delta[0], delta[1])), 1.0)
        desired_heading = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
        heading_error = self._wrap_degrees(
            desired_heading - float(ownship[StateIndex.YAW])
        )
        heading_error_rate = 0.0
        if self._previous_heading_error is not None:
            heading_error_rate = self._wrap_degrees(
                heading_error - self._previous_heading_error
            ) * 60.0
        self._previous_heading_error = heading_error
        capture_condition = (
            effective_capture_range_m > 0.0
            and distance_m <= effective_capture_range_m
            and abs(heading_error) <= self.capture_heading_error_deg
        )
        if self.latch_capture_mode and capture_condition:
            self._capture_latched = True
        capture_mode = capture_condition or (
            self.latch_capture_mode and self._capture_latched
        )
        own_forward = np.asarray([
            math.cos(math.radians(float(ownship[StateIndex.PITCH])))
            * math.cos(math.radians(float(ownship[StateIndex.YAW]))),
            math.cos(math.radians(float(ownship[StateIndex.PITCH])))
            * math.sin(math.radians(float(ownship[StateIndex.YAW]))),
            -math.sin(math.radians(float(ownship[StateIndex.PITCH]))),
        ])
        los_unit = raw_delta / max(distance_m, 1.0)
        roll_rad = math.radians(float(ownship[StateIndex.ROLL]))
        pitch_rad = math.radians(float(ownship[StateIndex.PITCH]))
        yaw_rad = math.radians(float(ownship[StateIndex.YAW]))
        sr, cr = math.sin(roll_rad), math.cos(roll_rad)
        sp, cp = math.sin(pitch_rad), math.cos(pitch_rad)
        sy, cy = math.sin(yaw_rad), math.cos(yaw_rad)
        body_to_ned = np.asarray([
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ], dtype=np.float64)
        body_los = body_to_ned.T @ los_unit
        body_azimuth_deg = math.degrees(math.atan2(
            float(body_los[1]), float(body_los[0])
        ))
        body_elevation_deg = math.degrees(math.atan2(
            -float(body_los[2]),
            max(float(np.hypot(body_los[0], body_los[1])), 1.0e-9),
        ))
        body_azimuth_rate = 0.0
        if self._previous_body_azimuth is not None:
            body_azimuth_rate = self._wrap_degrees(
                body_azimuth_deg - self._previous_body_azimuth
            ) * 60.0
        body_elevation_rate = 0.0
        if self._previous_body_elevation is not None:
            body_elevation_rate = self._wrap_degrees(
                body_elevation_deg - self._previous_body_elevation
            ) * 60.0
        self._previous_body_azimuth = body_azimuth_deg
        self._previous_body_elevation = body_elevation_deg
        ata_deg = math.degrees(math.acos(float(np.clip(
            np.dot(own_forward, los_unit), -1.0, 1.0
        ))))
        if (
            self.wez_hold_latch_ata_deg > 0.0
            and capture_mode
            and ata_deg <= self.wez_hold_latch_ata_deg
        ):
            if not self._wez_hold_latched:
                self._wez_hold_bank_deg = self._wrap_degrees(
                    float(ownship[StateIndex.ROLL])
                )
            self._wez_hold_latched = True
        capture_bias = self.capture_heading_bias_deg
        if self.capture_bias_target_roll_sign:
            target_roll = self._wrap_degrees(float(target[StateIndex.ROLL]))
            if abs(target_roll) > 1.0e-6:
                capture_bias = math.copysign(abs(capture_bias), target_roll)
        control_heading_error = (
            heading_error + capture_bias
            if capture_mode
            else heading_error
        )
        active_target_heading_blend = self.capture_target_heading_blend
        if (
            capture_mode
            and self.capture_near_ata_deg > 0.0
            and ata_deg <= self.capture_near_ata_deg
        ):
            fraction = max(ata_deg / self.capture_near_ata_deg, 0.0)
            active_target_heading_blend = (
                fraction * self.capture_target_heading_blend
                + (1.0 - fraction) * self.capture_near_target_heading_blend
            )
        if capture_mode and active_target_heading_blend > 0.0:
            target_heading_error = self._wrap_degrees(
                float(target[StateIndex.YAW]) - float(ownship[StateIndex.YAW])
            )
            control_heading_error = (
                (1.0 - active_target_heading_blend)
                * control_heading_error
                + active_target_heading_blend * target_heading_error
            )
        if self._wez_hold_latched:
            target_heading_error = self._wrap_degrees(
                float(target[StateIndex.YAW]) - float(ownship[StateIndex.YAW])
            )
            control_heading_error = (
                (1.0 - self.wez_hold_target_heading_blend)
                * control_heading_error
                + self.wez_hold_target_heading_blend * target_heading_error
            )
        active_heading_rate_to_bank = (
            self.capture_heading_rate_to_bank
            if capture_mode
            else self.heading_rate_to_bank
        )
        target_bank_magnitude = abs(self._wrap_degrees(
            float(target[StateIndex.ROLL])
        ))
        active_heading_to_bank = self.heading_to_bank + (
            self.target_bank_to_heading_gain
            * max(
                target_bank_magnitude
                - self.target_bank_heading_gain_baseline_deg,
                0.0,
            )
        )
        active_roll_delta_damping = max(
            self.capture_roll_delta_damping
            + self.target_bank_to_roll_damping
            * max(
                target_bank_magnitude
                - self.target_bank_roll_damping_baseline_deg,
                0.0,
            ),
            0.0,
        )
        if self._wez_hold_latched:
            active_roll_delta_damping = self.wez_hold_roll_delta_damping
        desired_bank = float(np.clip(
            active_heading_to_bank * control_heading_error
            + active_heading_rate_to_bank * heading_error_rate,
            -self.max_bank_deg,
            self.max_bank_deg,
        ))
        if (
            not capture_mode
            and self.target_bank_feedforward_range_m > 0.0
            and distance_m <= self.target_bank_feedforward_range_m
        ):
            target_bank = self._wrap_degrees(float(target[StateIndex.ROLL]))
            desired_bank = (
                (1.0 - self.target_bank_feedforward_blend) * desired_bank
                + self.target_bank_feedforward_blend * target_bank
            )
        if capture_mode:
            desired_bank *= self.capture_bank_scale
            target_bank = self._wrap_degrees(float(target[StateIndex.ROLL]))
            target_bank_blend = (
                self.wez_hold_target_bank_blend
                if self._wez_hold_latched
                else self.capture_target_bank_blend
            )
            if (
                not self._wez_hold_latched
                and self.capture_target_bank_blend_start_ata_deg > 0.0
            ):
                target_bank_blend *= float(np.clip(
                    1.0 - ata_deg / self.capture_target_bank_blend_start_ata_deg,
                    0.0,
                    1.0,
                ))
            target_bank_multiplier = (
                self.wez_hold_target_bank_multiplier
                if self._wez_hold_latched
                else self.capture_target_bank_multiplier
            )
            desired_bank = (
                (1.0 - target_bank_blend) * desired_bank
                + target_bank_blend
                * target_bank_multiplier
                * target_bank
            )
            desired_bank = float(np.clip(
                desired_bank, -self.max_bank_deg, self.max_bank_deg
            ))
        if (
            self._wez_hold_latched
            and self.wez_hold_freeze_bank
            and self._wez_hold_bank_deg is not None
        ):
            desired_bank = self._wez_hold_bank_deg
        rear_threat_trigger = (
            self.rear_threat_distance_m > 0.0
            and distance_m <= self.rear_threat_distance_m
            and ata_deg >= self.rear_threat_ata_deg
            and self.rear_threat_roll_gain > 0.0
        )
        if (
            rear_threat_trigger
            and self.rear_threat_turn_latch
            and self._rear_threat_turn_sign == 0.0
        ):
            self._rear_threat_turn_sign = math.copysign(
                1.0, control_heading_error if control_heading_error != 0.0 else 1.0
            )
        if (
            self._rear_threat_turn_sign != 0.0
            and ata_deg <= self.rear_threat_turn_release_ata_deg
        ):
            self._rear_threat_turn_sign = 0.0
        rear_threat_turn_latched = self._rear_threat_turn_sign != 0.0
        rear_threat_active = rear_threat_trigger or rear_threat_turn_latched
        if rear_threat_turn_latched:
            desired_bank = self._rear_threat_turn_sign * self.max_bank_deg
        active_rudder_gain = (
            self.capture_rudder_gain if capture_mode else self.rudder_gain
        )
        if rear_threat_active:
            active_rudder_gain = self.rear_threat_rudder_gain
        roll_error = self._wrap_degrees(
            desired_bank - float(ownship[StateIndex.ROLL])
        )
        roll_delta = 0.0
        if self._previous_roll is not None:
            roll_delta = self._wrap_degrees(
                float(ownship[StateIndex.ROLL]) - self._previous_roll
            )
        self._previous_roll = float(ownship[StateIndex.ROLL])
        active_roll_gain = (
            self.wez_hold_roll_gain
            if self._wez_hold_latched
            else self.capture_roll_gain if capture_mode else self.roll_gain
        )
        if rear_threat_active:
            active_roll_gain = self.rear_threat_roll_gain
            active_roll_delta_damping = self.rear_threat_roll_delta_damping
        desired_pitch = math.degrees(math.atan2(-float(delta[2]), horizontal))
        active_bank_pitch_compensation_gain = (
            self.rear_threat_bank_pitch_compensation_gain
            if rear_threat_active
            and self.rear_threat_bank_pitch_compensation_gain > 0.0
            else self.bank_pitch_compensation_gain
        )
        desired_pitch += active_bank_pitch_compensation_gain * abs(desired_bank)
        if capture_mode and self.capture_target_pitch_blend > 0.0:
            target_pitch = self._wrap_degrees(float(target[StateIndex.PITCH]))
            desired_pitch = (
                (1.0 - self.capture_target_pitch_blend) * desired_pitch
                + self.capture_target_pitch_blend * target_pitch
            )
        if self._wez_hold_latched:
            target_pitch = self._wrap_degrees(float(target[StateIndex.PITCH]))
            desired_pitch = (
                (1.0 - self.wez_hold_target_pitch_blend) * desired_pitch
                + self.wez_hold_target_pitch_blend * target_pitch
            )
        altitude_m = float(ownship[StateIndex.ALT])
        if altitude_m <= 0.0:
            altitude_m = -float(ownship[StateIndex.D])
        if altitude_m < self.altitude_guard_m:
            desired_pitch = max(desired_pitch, self.recovery_pitch_deg)
        pitch_error = self._wrap_degrees(
            desired_pitch - float(ownship[StateIndex.PITCH])
        )
        pitch_error_rate = 0.0
        if self._previous_pitch_error is not None:
            pitch_error_rate = self._wrap_degrees(
                pitch_error - self._previous_pitch_error
            ) * 60.0
        self._previous_pitch_error = pitch_error
        pitch_delta = 0.0
        if self._previous_pitch is not None:
            pitch_delta = self._wrap_degrees(
                float(ownship[StateIndex.PITCH]) - self._previous_pitch
            )
        self._previous_pitch = float(ownship[StateIndex.PITCH])
        active_pitch_gain = self.capture_pitch_gain if capture_mode else self.pitch_gain
        active_pitch_delta_damping = self.capture_pitch_delta_damping
        if self._wez_hold_latched:
            active_pitch_gain = self.wez_hold_pitch_gain
            active_pitch_delta_damping = self.wez_hold_pitch_delta_damping
        brake_aligned = (
            self.brake_heading_error_deg <= 0.0
            or abs(heading_error) <= self.brake_heading_error_deg
        )
        braking = (
            self.brake_range_m > 0.0
            and distance_m <= self.brake_range_m
            and brake_aligned
        )
        throttle_action = self.near_throttle_action if braking else 1.0
        if capture_mode and self.capture_speed_gain > 0.0:
            ownship_speed = float(np.linalg.norm(ownship[
                StateIndex.U : StateIndex.W + 1
            ]))
            target_speed = float(np.linalg.norm(target[
                StateIndex.U : StateIndex.W + 1
            ]))
            throttle_action = float(np.clip(
                self.capture_speed_gain * (
                    target_speed + self.capture_speed_offset_mps - ownship_speed
                ),
                -1.0,
                1.0,
            ))
        if (
            self.proximity_speed_match_start_distance_m > 0.0
            and distance_m <= self.proximity_speed_match_start_distance_m
            and self.proximity_speed_match_gain > 0.0
        ):
            ownship_speed = float(np.linalg.norm(ownship[
                StateIndex.U : StateIndex.W + 1
            ]))
            target_speed = float(np.linalg.norm(target[
                StateIndex.U : StateIndex.W + 1
            ]))
            overspeed_gate = (
                ownship_speed - target_speed
                >= self.proximity_speed_match_min_overspeed_mps
                and ata_deg <= self.proximity_speed_match_max_ata_deg
            )
            if overspeed_gate and self.proximity_speed_match_latch:
                self._proximity_speed_match_latched = True
            if overspeed_gate or self._proximity_speed_match_latched:
                throttle_action = float(np.clip(
                    self.proximity_speed_match_gain * (
                        target_speed - ownship_speed
                    ),
                    -1.0,
                    1.0,
                ))
        roll_command = active_roll_gain * roll_error - (
            active_roll_delta_damping * roll_delta
            if capture_mode or rear_threat_active
            else 0.0
        )
        pitch_command = active_pitch_gain * pitch_error + (
            self.capture_pitch_error_rate_gain * pitch_error_rate
            if capture_mode
            else 0.0
        ) - (
            active_pitch_delta_damping * pitch_delta
            if capture_mode
            else 0.0
        )
        body_axis_active = (
            capture_mode
            and self.capture_body_axis_guidance
            and (
                not self.capture_body_axis_latch_only
                or self._wez_hold_latched
            )
        )
        if body_axis_active:
            roll_command = (
                self.capture_body_axis_roll_gain * body_azimuth_deg
                + self.capture_body_axis_roll_rate_gain * body_azimuth_rate
                - active_roll_delta_damping * roll_delta
            )
            pitch_command = (
                self.capture_body_axis_pitch_gain * body_elevation_deg
                + self.capture_body_axis_pitch_rate_gain * body_elevation_rate
                - self.capture_pitch_delta_damping * pitch_delta
            )
        action = clip_action(
            [
                roll_command,
                self.pitch_command_sign * pitch_command,
                (
                    active_rudder_gain * control_heading_error
                    + (
                        self.capture_heading_rate_to_rudder * heading_error_rate
                        if capture_mode
                        else 0.0
                    )
                ) / 90.0,
                throttle_action,
            ],
            low=-np.ones(4, dtype=np.float32),
            high=np.ones(4, dtype=np.float32),
        )
        return ActionResult(
            action=action,
            source="rl",
            confidence=0.8,
            info={
                "guidance": "pure_pursuit",
                "active_heading_to_bank": active_heading_to_bank,
                "active_roll_delta_damping": active_roll_delta_damping,
                "heading_error_deg": heading_error,
                "control_heading_error_deg": control_heading_error,
                "capture_heading_bias_deg": capture_bias,
                "desired_bank_deg": desired_bank,
                "heading_error_rate_deg_s": heading_error_rate,
                "heading_rate_to_bank": active_heading_rate_to_bank,
                "roll_gain": active_roll_gain,
                "roll_delta_deg": roll_delta,
                "capture_roll_delta_damping": self.capture_roll_delta_damping,
                "wez_hold_roll_delta_damping": self.wez_hold_roll_delta_damping,
                "capture_pitch_delta_damping": self.capture_pitch_delta_damping,
                "capture_pitch_error_rate_gain": self.capture_pitch_error_rate_gain,
                "pitch_error_rate_deg_s": pitch_error_rate,
                "capture_body_axis_guidance": body_axis_active,
                "body_azimuth_deg": body_azimuth_deg,
                "body_elevation_deg": body_elevation_deg,
                "body_azimuth_rate_deg_s": body_azimuth_rate,
                "body_elevation_rate_deg_s": body_elevation_rate,
                "pitch_delta_deg": pitch_delta,
                "desired_pitch_deg": desired_pitch,
                "pitch_gain": active_pitch_gain,
                "heading_rate_to_rudder": (
                    self.capture_heading_rate_to_rudder if capture_mode else 0.0
                ),
                "lead_time_s": active_lead_time_s,
                "distance_m": distance_m,
                "throttle_action": throttle_action,
                "braking": braking,
                "capture_mode": capture_mode,
                "capture_latched": self._capture_latched,
                "maneuver_capture_active": maneuver_capture_active,
                "effective_capture_range_m": effective_capture_range_m,
                "active_lead_time_s": active_lead_time_s,
                "slow_target_capture_active": slow_target_capture_active,
                "target_speed_mps": target_speed_mps,
                "slow_target_capture_selected": self._slow_target_capture_selected,
                "extreme_bank_capture_active": (
                    maneuver_capture_active
                    and self._extreme_bank_capture_selected
                ),
                "maneuver_capture_selected": self._maneuver_capture_selected,
                "wez_hold_latched": self._wez_hold_latched,
                "wez_hold_freeze_bank": self.wez_hold_freeze_bank,
                "wez_hold_bank_deg": self._wez_hold_bank_deg,
                "wez_hold_pitch_gain": self.wez_hold_pitch_gain,
                "wez_hold_pitch_delta_damping": active_pitch_delta_damping,
                "wez_hold_target_heading_blend": self.wez_hold_target_heading_blend,
                "capture_target_heading_blend": self.capture_target_heading_blend,
                "active_target_heading_blend": active_target_heading_blend,
                "capture_target_pitch_blend": self.capture_target_pitch_blend,
                "wez_hold_target_pitch_blend": self.wez_hold_target_pitch_blend,
                "ata_deg": ata_deg,
                "trail_distance_m": self.trail_distance_m,
                "capture_target_bank_blend": self.capture_target_bank_blend,
                "capture_target_bank_blend_start_ata_deg": (
                    self.capture_target_bank_blend_start_ata_deg
                ),
                "bank_pitch_compensation_gain": self.bank_pitch_compensation_gain,
                "capture_target_bank_multiplier": (
                    self.capture_target_bank_multiplier
                ),
                "capture_speed_gain": self.capture_speed_gain,
                "proximity_speed_match_start_distance_m": (
                    self.proximity_speed_match_start_distance_m
                ),
                "proximity_speed_match_gain": self.proximity_speed_match_gain,
                "proximity_speed_match_min_overspeed_mps": (
                    self.proximity_speed_match_min_overspeed_mps
                ),
                "proximity_speed_match_max_ata_deg": (
                    self.proximity_speed_match_max_ata_deg
                ),
                "proximity_speed_match_latched": (
                    self._proximity_speed_match_latched
                ),
                "rear_threat_active": rear_threat_active,
                "rear_threat_roll_gain": self.rear_threat_roll_gain,
                "rear_threat_bank_pitch_compensation_gain": (
                    self.rear_threat_bank_pitch_compensation_gain
                ),
                "rear_threat_turn_latched": rear_threat_turn_latched,
                "rear_threat_turn_sign": self._rear_threat_turn_sign,
                "rear_threat_roll_delta_damping": (
                    self.rear_threat_roll_delta_damping
                ),
                "rear_threat_rudder_gain": self.rear_threat_rudder_gain,
                "capture_speed_offset_mps": self.capture_speed_offset_mps,
                "target_bank_feedforward_range_m": (
                    self.target_bank_feedforward_range_m
                ),
            },
        )


class PursuitAutopilotActionProvider(ActionProvider):
    """Command the native heading/altitude autopilot toward a lead intercept."""

    def __init__(self, *, lead_time_s: float = 3.0, speed_mps: float = 300.0) -> None:
        self.lead_time_s = float(lead_time_s)
        self.speed_mps = float(speed_mps)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None or context.target_state is None:
            raise ValueError("ownship_state and target_state are required")
        ownship = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        delta = target[:3] - ownship[:3]
        delta += self.lead_time_s * (
            _body_velocity_to_ned(target) - _body_velocity_to_ned(ownship)
        )
        heading_cmd = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
        target_altitude = float(target[StateIndex.ALT])
        if target_altitude <= 0.0:
            target_altitude = -float(target[StateIndex.D])
        # FighterSim.step_autopilot expects NED Down meters, not geodetic
        # altitude. A level 8,000 m command is therefore -8,000 m.
        altitude_cmd = -max(target_altitude, 3000.0)
        return ActionResult(
            action=np.zeros(4, dtype=np.float32),
            source="pursuit_autopilot",
            confidence=0.9,
            info={
                "heading_cmd": heading_cmd,
                "altitude_cmd": altitude_cmd,
                "speed_cmd": self.speed_mps,
                "lead_time_s": self.lead_time_s,
            },
        )


class TargetFormationAutopilotActionProvider(ActionProvider):
    """Use the native autopilot to match the target's instantaneous flight path.

    This provider is intended for close, aligned geometry.  Matching heading,
    altitude, and speed prevents the ownship from immediately flying through a
    turning target after a successful WEZ entry.
    """

    def __init__(self, *, minimum_altitude_m: float = 3000.0) -> None:
        self.minimum_altitude_m = float(minimum_altitude_m)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.target_state is None:
            raise ValueError("target_state is required")
        target = np.asarray(context.target_state, dtype=np.float64)
        target_altitude = float(target[StateIndex.ALT])
        if target_altitude <= 0.0:
            target_altitude = -float(target[StateIndex.D])
        speed_cmd = float(np.linalg.norm(target[
            StateIndex.U : StateIndex.W + 1
        ]))
        return ActionResult(
            action=np.zeros(4, dtype=np.float32),
            source="pursuit_autopilot",
            confidence=0.9,
            info={
                "heading_cmd": float(target[StateIndex.YAW]) % 360.0,
                "altitude_cmd": -max(target_altitude, self.minimum_altitude_m),
                "speed_cmd": max(speed_cmd, 1.0),
                "guidance": "target_formation_autopilot",
            },
        )


class TargetTrailAutopilotActionProvider(ActionProvider):
    """Use native autopilot to pursue a moving point behind the target."""

    def __init__(
        self,
        *,
        trail_distance_m: float = 600.0,
        speed_offset_mps: float = 0.0,
        minimum_altitude_m: float = 3000.0,
    ) -> None:
        self.trail_distance_m = max(float(trail_distance_m), 0.0)
        self.speed_offset_mps = float(speed_offset_mps)
        self.minimum_altitude_m = float(minimum_altitude_m)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None or context.target_state is None:
            raise ValueError("ownship_state and target_state are required")
        ownship = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        yaw = math.radians(float(target[StateIndex.YAW]))
        pitch = math.radians(float(target[StateIndex.PITCH]))
        forward = np.asarray([
            math.cos(pitch) * math.cos(yaw),
            math.cos(pitch) * math.sin(yaw),
            -math.sin(pitch),
        ])
        trail_point = target[:3] - self.trail_distance_m * forward
        delta = trail_point - ownship[:3]
        heading_cmd = math.degrees(math.atan2(float(delta[1]), float(delta[0]))) % 360.0
        target_speed = float(np.linalg.norm(target[
            StateIndex.U : StateIndex.W + 1
        ]))
        target_altitude = max(-float(trail_point[2]), self.minimum_altitude_m)
        return ActionResult(
            action=np.zeros(4, dtype=np.float32),
            source="pursuit_autopilot",
            confidence=0.9,
            info={
                "heading_cmd": heading_cmd,
                "altitude_cmd": -target_altitude,
                "speed_cmd": max(target_speed + self.speed_offset_mps, 1.0),
                "guidance": "target_trail_autopilot",
                "trail_distance_m": self.trail_distance_m,
            },
        )


class GuidanceResidualActionProvider(ActionProvider):
    """Reproduce training-time guidance plus bounded policy residual."""

    def __init__(
        self,
        policy_provider: ActionProvider,
        guidance_provider: PursuitGuidanceActionProvider,
        *,
        residual_scale,
        near_residual_scale=None,
        near_distance_m: float = 0.0,
        near_ata_deg: float = 180.0,
        suppress_opposing_roll_heading_error_deg: float = 0.0,
        opposing_roll_guidance_scale: float = 1.0,
    ) -> None:
        self.policy_provider = policy_provider
        self.guidance_provider = guidance_provider
        self.residual_scale = np.asarray(residual_scale, dtype=np.float32)
        if self.residual_scale.shape != (4,):
            raise ValueError("residual_scale must contain four values")
        self.near_residual_scale = None
        if near_residual_scale is not None:
            self.near_residual_scale = np.asarray(
                near_residual_scale, dtype=np.float32
            )
            if self.near_residual_scale.shape != (4,):
                raise ValueError("near_residual_scale must contain four values")
        self.near_distance_m = max(float(near_distance_m), 0.0)
        self.near_ata_deg = max(float(near_ata_deg), 0.0)
        self.suppress_opposing_roll_heading_error_deg = max(
            float(suppress_opposing_roll_heading_error_deg), 0.0
        )
        self.opposing_roll_guidance_scale = max(
            float(opposing_roll_guidance_scale), 1.0
        )

    def reset(self, context: ActionContext | None = None) -> None:
        self.policy_provider.reset(context)
        self.guidance_provider.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        policy = self.policy_provider.compute_action(context)
        guidance = self.guidance_provider.compute_action(context)
        residual_scale = self.residual_scale
        near_residual_active = False
        if (
            self.near_residual_scale is not None
            and context.ownship_state is not None
            and context.target_state is not None
        ):
            relative_position = np.asarray(
                context.target_state[:3], dtype=np.float64
            ) - np.asarray(context.ownship_state[:3], dtype=np.float64)
            distance = float(np.linalg.norm(relative_position))
            if distance > 1.0e-6:
                pitch = math.radians(float(context.ownship_state[StateIndex.PITCH]))
                yaw = math.radians(float(context.ownship_state[StateIndex.YAW]))
                nose = np.asarray(
                    [math.cos(pitch) * math.cos(yaw),
                     math.cos(pitch) * math.sin(yaw),
                     -math.sin(pitch)],
                    dtype=np.float64,
                )
                cosine = float(np.clip(
                    np.dot(nose, relative_position) / distance, -1.0, 1.0
                ))
                ata_deg = math.degrees(math.acos(cosine))
                near_residual_active = (
                    distance <= self.near_distance_m
                    and ata_deg <= self.near_ata_deg
                )
                if near_residual_active:
                    residual_scale = self.near_residual_scale
        policy_action = np.asarray(policy.action, dtype=np.float32)
        guidance_action = np.asarray(guidance.action, dtype=np.float32)
        opposing_roll_suppressed = False
        signed_heading_error = float(
            guidance.info.get("heading_error_deg", 0.0)
        )
        if context.ownship_state is not None:
            ownship_bank = (
                float(context.ownship_state[StateIndex.ROLL]) + 180.0
            ) % 360.0 - 180.0
            opposing_policy_roll = (
                float(guidance_action[0]) * float(policy_action[0]) < 0.0
            )
            wrong_way_bank = signed_heading_error * ownship_bank < 0.0
            opposing_roll_suppressed = (
                self.suppress_opposing_roll_heading_error_deg > 0.0
                and abs(signed_heading_error)
                >= self.suppress_opposing_roll_heading_error_deg
                and (opposing_policy_roll or wrong_way_bank)
            )
        if opposing_roll_suppressed:
            residual_scale = np.array(residual_scale, copy=True)
            residual_scale[0] = 0.0
            guidance_action = np.array(guidance_action, copy=True)
            guidance_action[0] = float(np.clip(
                guidance_action[0] * self.opposing_roll_guidance_scale,
                -1.0,
                1.0,
            ))
        action = np.clip(
            guidance_action + residual_scale * policy_action,
            -1.0,
            1.0,
        )
        return ActionResult(
            action=action,
            source="rl",
            confidence=min(policy.confidence, guidance.confidence),
            info={
                "guidance_residual": True,
                "residual_scale": residual_scale.tolist(),
                "near_residual_active": near_residual_active,
                "opposing_roll_suppressed": opposing_roll_suppressed,
                "policy": policy.info,
                "guidance": guidance.info,
            },
        )

    def close(self) -> None:
        self.policy_provider.close()
        self.guidance_provider.close()


class TargetAltitudeGuardActionProvider(ActionProvider):
    """Keep a surface-control target from terminating through a terrain dive."""

    def __init__(
        self,
        primary: ActionProvider,
        *,
        trigger_altitude_m: float = 2500.0,
        release_altitude_m: float = 4000.0,
        roll_level_gain: float = 0.02,
        pull_up_action: float = -1.0,
        throttle_action: float = 1.0,
        proportional_pull_up: bool = False,
        minimum_pull_fraction: float = 0.0,
    ) -> None:
        if trigger_altitude_m <= 0.0:
            raise ValueError("trigger_altitude_m must be positive")
        if release_altitude_m <= trigger_altitude_m:
            raise ValueError("release_altitude_m must exceed trigger_altitude_m")
        self.primary = primary
        self.trigger_altitude_m = float(trigger_altitude_m)
        self.release_altitude_m = float(release_altitude_m)
        self.roll_level_gain = max(float(roll_level_gain), 0.0)
        self.pull_up_action = float(np.clip(pull_up_action, -1.0, 1.0))
        self.throttle_action = float(np.clip(throttle_action, -1.0, 1.0))
        self.proportional_pull_up = bool(proportional_pull_up)
        self.minimum_pull_fraction = float(np.clip(
            minimum_pull_fraction, 0.0, 1.0
        ))
        self._recovery_active = False

    def reset(self, context: ActionContext | None = None) -> None:
        self._recovery_active = False
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None:
            raise ValueError("ownship_state is required for target altitude guard")
        state = np.asarray(context.ownship_state, dtype=np.float64)
        altitude_m = float(state[StateIndex.ALT])
        if altitude_m <= 0.0:
            altitude_m = -float(state[StateIndex.D])
        if self._recovery_active:
            self._recovery_active = altitude_m < self.release_altitude_m
        else:
            self._recovery_active = altitude_m <= self.trigger_altitude_m
        if not self._recovery_active:
            result = self.primary.compute_action(context)
            return ActionResult(
                action=result.action,
                source=result.source,
                confidence=result.confidence,
                info={**result.info, "target_altitude_guard": False},
            )
        roll_deg = PursuitGuidanceActionProvider._wrap_degrees(
            float(state[StateIndex.ROLL])
        )
        active_pull_up_action = self.pull_up_action
        if self.proportional_pull_up:
            recovery_fraction = float(np.clip(
                (self.release_altitude_m - altitude_m)
                / (self.release_altitude_m - self.trigger_altitude_m),
                self.minimum_pull_fraction,
                1.0,
            ))
            active_pull_up_action *= recovery_fraction
        action = clip_action([
            -self.roll_level_gain * roll_deg,
            active_pull_up_action,
            0.0,
            self.throttle_action,
        ], low=-np.ones(4, dtype=np.float32), high=np.ones(4, dtype=np.float32))
        return ActionResult(
            action=action,
            source="target_altitude_guard",
            confidence=1.0,
            info={
                "target_altitude_guard": True,
                "target_altitude_m": altitude_m,
                "target_altitude_guard_pull_up_action": active_pull_up_action,
            },
        )

    def close(self) -> None:
        self.primary.close()


class ProximitySpeedMatchActionProvider(ActionProvider):
    """Preserve steering while continuously matching target speed when close."""

    def __init__(
        self,
        provider: ActionProvider,
        *,
        start_distance_m: float,
        speed_gain: float,
    ) -> None:
        self.provider = provider
        self.start_distance_m = float(start_distance_m)
        self.speed_gain = max(float(speed_gain), 0.0)

    def reset(self, context: ActionContext | None = None) -> None:
        self.provider.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.provider.compute_action(context)
        if context.ownship_state is None or context.target_state is None:
            return result
        ownship = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        distance = float(np.linalg.norm(target[:3] - ownship[:3]))
        if distance > self.start_distance_m or result.source != "rl":
            return result
        ownship_speed = float(np.linalg.norm(ownship[
            StateIndex.U : StateIndex.W + 1
        ]))
        target_speed = float(np.linalg.norm(target[
            StateIndex.U : StateIndex.W + 1
        ]))
        action = np.asarray(result.action, dtype=np.float32).copy()
        action[3] = float(np.clip(
            self.speed_gain * (target_speed - ownship_speed), -1.0, 1.0
        ))
        return ActionResult(
            action=action,
            source=result.source,
            confidence=result.confidence,
            info={
                **result.info,
                "proximity_speed_match": True,
                "speed_error_mps": target_speed - ownship_speed,
            },
        )

    def close(self) -> None:
        self.provider.close()


class HybridActionProvider(ActionProvider):
    def __init__(
        self,
        primary_provider: ActionProvider,
        secondary_provider: ActionProvider,
        mode: str = "residual",
        alpha: float = 0.5,
        residual_scale: float = 0.35,
        selector: Callable[[ActionContext, ActionResult, ActionResult], str | bool] | None = None,
        confidence: float = 0.95,
    ):
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.mode = mode
        self.alpha = alpha
        self.residual_scale = residual_scale
        self.selector = selector
        self.confidence = confidence

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary_provider.reset(context)
        self.secondary_provider.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        primary_result = self.primary_provider.compute_action(context)
        secondary_result = self.secondary_provider.compute_action(context)

        if self.mode == "switch":
            decision = self.selector(context, primary_result, secondary_result) if self.selector else "primary"
            use_primary = decision if isinstance(decision, bool) else decision != "secondary"
            chosen = primary_result if use_primary else secondary_result
            return ActionResult(
                action=clip_action(chosen.action),
                source="hybrid",
                confidence=self.confidence,
                info={
                    "mode": self.mode,
                    "selected": chosen.source,
                    "primary": primary_result.info,
                    "secondary": secondary_result.info,
                },
            )

        if self.mode == "blend":
            action = self.alpha * primary_result.action + (1.0 - self.alpha) * secondary_result.action
        else:
            action = secondary_result.action + self.residual_scale * primary_result.action

        return ActionResult(
            action=clip_action(action),
            source="hybrid",
            confidence=self.confidence,
            info={
                "mode": self.mode,
                "alpha": self.alpha,
                "residual_scale": self.residual_scale,
                "primary_source": primary_result.source,
                "secondary_source": secondary_result.source,
            },
        )

    def close(self) -> None:
        self.primary_provider.close()
        self.secondary_provider.close()


class WezThenActionProvider(ActionProvider):
    """Latch from a primary provider to a secondary after a strict WEZ hold."""

    def __init__(
        self,
        primary_provider: ActionProvider,
        secondary_provider: ActionProvider,
        *,
        wez_observation_index: int,
        required_steps: int = 10,
    ) -> None:
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.wez_observation_index = int(wez_observation_index)
        self.required_steps = max(int(required_steps), 1)
        self._consecutive_wez_steps = 0
        self._last_timestep = None
        self._secondary_active = False

    def reset(self, context: ActionContext | None = None) -> None:
        self._consecutive_wez_steps = 0
        self._last_timestep = None
        self._secondary_active = False
        self.primary_provider.reset(context)
        self.secondary_provider.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        timestep = context.info.get("timestep")
        if timestep != self._last_timestep and not self._secondary_active:
            observation = context.observation
            in_wez = (
                observation is not None
                and 0 <= self.wez_observation_index < len(observation)
                and float(observation[self.wez_observation_index]) > 0.0
            )
            self._consecutive_wez_steps = (
                self._consecutive_wez_steps + 1 if in_wez else 0
            )
            self._secondary_active = (
                self._consecutive_wez_steps >= self.required_steps
            )
            self._last_timestep = timestep

        provider = (
            self.secondary_provider
            if self._secondary_active
            else self.primary_provider
        )
        result = provider.compute_action(context)
        return ActionResult(
            action=result.action,
            source=result.source,
            confidence=result.confidence,
            info={
                **result.info,
                "wez_then_secondary": self._secondary_active,
                "consecutive_wez_steps": self._consecutive_wez_steps,
            },
        )

    def close(self) -> None:
        self.primary_provider.close()
        self.secondary_provider.close()


class GeometryThenActionProvider(ActionProvider):
    """Latch to a secondary provider after reaching a close aligned geometry."""

    def __init__(
        self,
        primary_provider: ActionProvider,
        secondary_provider: ActionProvider,
        *,
        max_distance_m: float,
        max_ata_deg: float,
    ) -> None:
        self.primary_provider = primary_provider
        self.secondary_provider = secondary_provider
        self.max_distance_m = float(max_distance_m)
        self.max_ata_deg = float(max_ata_deg)
        self._secondary_active = False

    def reset(self, context: ActionContext | None = None) -> None:
        self._secondary_active = False
        self.primary_provider.reset(context)
        self.secondary_provider.reset(context)

    @staticmethod
    def _ata_deg(context: ActionContext) -> tuple[float, float]:
        ownship = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        delta = target[:3] - ownship[:3]
        distance = float(np.linalg.norm(delta))
        forward = np.asarray([
            math.cos(math.radians(float(ownship[StateIndex.PITCH])))
            * math.cos(math.radians(float(ownship[StateIndex.YAW]))),
            math.cos(math.radians(float(ownship[StateIndex.PITCH])))
            * math.sin(math.radians(float(ownship[StateIndex.YAW]))),
            -math.sin(math.radians(float(ownship[StateIndex.PITCH]))),
        ])
        ata = math.degrees(math.acos(float(np.clip(
            np.dot(forward, delta / max(distance, 1.0)), -1.0, 1.0
        ))))
        return distance, ata

    def compute_action(self, context: ActionContext) -> ActionResult:
        if not self._secondary_active:
            if context.ownship_state is None or context.target_state is None:
                raise ValueError("ownship_state and target_state are required")
            distance, ata = self._ata_deg(context)
            self._secondary_active = (
                distance <= self.max_distance_m and ata <= self.max_ata_deg
            )
        provider = (
            self.secondary_provider if self._secondary_active
            else self.primary_provider
        )
        result = provider.compute_action(context)
        return ActionResult(
            action=result.action,
            source=result.source,
            confidence=result.confidence,
            info={**result.info, "geometry_then_secondary": self._secondary_active},
        )

    def close(self) -> None:
        self.primary_provider.close()
        self.secondary_provider.close()


class HeadingThenActionProvider(ActionProvider):
    """Latch from heading alignment to LOS capture once courses are parallel."""

    def __init__(
        self,
        alignment_provider: ActionProvider,
        capture_provider: ActionProvider,
        *,
        max_heading_difference_deg: float,
        max_capture_ata_deg: float = 0.0,
    ) -> None:
        self.alignment_provider = alignment_provider
        self.capture_provider = capture_provider
        self.max_heading_difference_deg = float(max_heading_difference_deg)
        self.max_capture_ata_deg = max(float(max_capture_ata_deg), 0.0)
        self._capture_active = False

    def reset(self, context: ActionContext | None = None) -> None:
        self._capture_active = False
        self.alignment_provider.reset(context)
        self.capture_provider.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None or context.target_state is None:
            raise ValueError("ownship_state and target_state are required")
        ownship = np.asarray(context.ownship_state, dtype=np.float64)
        target = np.asarray(context.target_state, dtype=np.float64)
        difference = (float(target[StateIndex.YAW]) - float(
            ownship[StateIndex.YAW]
        ) + 180.0) % 360.0 - 180.0
        delta = target[:3] - ownship[:3]
        distance = float(np.linalg.norm(delta))
        forward = np.asarray([
            math.cos(math.radians(float(ownship[StateIndex.PITCH])))
            * math.cos(math.radians(float(ownship[StateIndex.YAW]))),
            math.cos(math.radians(float(ownship[StateIndex.PITCH])))
            * math.sin(math.radians(float(ownship[StateIndex.YAW]))),
            -math.sin(math.radians(float(ownship[StateIndex.PITCH]))),
        ])
        ata = math.degrees(math.acos(float(np.clip(
            np.dot(forward, delta / max(distance, 1.0)), -1.0, 1.0
        ))))
        if (
            abs(difference) <= self.max_heading_difference_deg
            or (
                self.max_capture_ata_deg > 0.0
                and ata <= self.max_capture_ata_deg
            )
        ):
            self._capture_active = True
        provider = (
            self.capture_provider
            if self._capture_active
            else self.alignment_provider
        )
        result = provider.compute_action(context)
        return ActionResult(
            action=result.action,
            source=result.source,
            confidence=result.confidence,
            info={
                **result.info,
                "heading_then_capture": self._capture_active,
                "target_heading_difference_deg": difference,
                "heading_then_ata_deg": ata,
            },
        )

    def close(self) -> None:
        self.alignment_provider.close()
        self.capture_provider.close()


def build_pursuit_guidance_provider(config: dict) -> ActionProvider:
    """Build the same optional two-stage guidance teacher for train and eval."""
    options = dict(config)
    options.pop("enabled", None)
    options.pop("residual_scale", None)
    options.pop("near_residual_scale", None)
    options.pop("near_distance_m", None)
    options.pop("near_ata_deg", None)
    options.pop("suppress_opposing_roll_heading_error_deg", None)
    options.pop("opposing_roll_guidance_scale", None)
    close_config = options.pop("close_guidance", None)
    provider: ActionProvider = PursuitGuidanceActionProvider(**options)
    if close_config is None:
        return provider
    close_options = dict(close_config)
    max_distance_m = float(close_options.pop("max_distance_m"))
    max_ata_deg = float(close_options.pop("max_ata_deg", 180.0))
    return GeometryThenActionProvider(
        provider,
        PursuitGuidanceActionProvider(**close_options),
        max_distance_m=max_distance_m,
        max_ata_deg=max_ata_deg,
    )
