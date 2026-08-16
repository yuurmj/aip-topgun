from __future__ import annotations

import argparse
import gzip
import json
import pickle
import shutil
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clone a lightweight bundle while replacing guidance metadata."
    )
    parser.add_argument("--source-bundle", required=True)
    parser.add_argument("--output-bundle", required=True)
    parser.add_argument("--residual-scale")
    parser.add_argument("--near-residual-scale", default="")
    parser.add_argument("--near-distance-m", type=float, default=0.0)
    parser.add_argument("--near-ata-deg", type=float, default=180.0)
    parser.add_argument("--capture-target-bank-blend", type=float)
    parser.add_argument(
        "--capture-target-bank-blend-start-ata-deg", type=float
    )
    parser.add_argument("--capture-roll-gain", type=float)
    parser.add_argument("--roll-gain", type=float)
    parser.add_argument("--max-bank-deg", type=float)
    parser.add_argument("--heading-to-bank", type=float)
    parser.add_argument("--pitch-gain", type=float)
    parser.add_argument("--capture-pitch-gain", type=float)
    parser.add_argument("--suppress-opposing-roll-heading-error-deg", type=float)
    parser.add_argument("--opposing-roll-guidance-scale", type=float)
    parser.add_argument("--capture-roll-delta-damping", type=float)
    parser.add_argument("--capture-pitch-delta-damping", type=float)
    parser.add_argument("--capture-pitch-error-rate-gain", type=float)
    parser.add_argument("--capture-body-axis-guidance", action="store_true")
    parser.add_argument("--capture-body-axis-latch-only", action="store_true")
    parser.add_argument("--capture-body-axis-roll-gain", type=float)
    parser.add_argument("--capture-body-axis-pitch-gain", type=float)
    parser.add_argument("--capture-body-axis-roll-rate-gain", type=float)
    parser.add_argument("--capture-body-axis-pitch-rate-gain", type=float)
    parser.add_argument("--capture-heading-rate-to-bank", type=float)
    parser.add_argument("--capture-heading-rate-to-rudder", type=float)
    parser.add_argument("--capture-heading-bias-deg", type=float)
    parser.add_argument("--capture-bias-target-roll-sign", action="store_true")
    parser.add_argument("--bank-pitch-compensation-gain", type=float, default=0.0)
    parser.add_argument("--wez-hold-latch-ata-deg", type=float)
    parser.add_argument("--wez-hold-target-bank-blend", type=float)
    parser.add_argument("--wez-hold-roll-gain", type=float)
    parser.add_argument("--wez-hold-roll-delta-damping", type=float)
    parser.add_argument("--wez-hold-freeze-bank", action="store_true")
    parser.add_argument("--wez-hold-pitch-gain", type=float)
    parser.add_argument("--wez-hold-pitch-delta-damping", type=float)
    parser.add_argument("--wez-hold-target-heading-blend", type=float)
    parser.add_argument("--wez-hold-target-pitch-blend", type=float)
    parser.add_argument("--capture-speed-gain", type=float)
    parser.add_argument("--capture-speed-offset-mps", type=float)
    parser.add_argument("--brake-heading-error-deg", type=float)
    parser.add_argument("--brake-range-m", type=float)
    parser.add_argument("--capture-range-m", type=float)
    parser.add_argument("--near-throttle-action", type=float)
    parser.add_argument("--proximity-speed-match-start-distance-m", type=float)
    parser.add_argument("--proximity-speed-match-gain", type=float)
    parser.add_argument("--proximity-speed-match-min-overspeed-mps", type=float)
    parser.add_argument("--proximity-speed-match-max-ata-deg", type=float)
    parser.add_argument("--proximity-speed-match-latch", action="store_true")
    parser.add_argument("--rear-threat-distance-m", type=float)
    parser.add_argument("--rear-threat-ata-deg", type=float)
    parser.add_argument("--rear-threat-roll-gain", type=float)
    parser.add_argument(
        "--rear-threat-bank-pitch-compensation-gain", type=float
    )
    parser.add_argument("--rear-threat-turn-latch", action="store_true")
    parser.add_argument("--rear-threat-turn-release-ata-deg", type=float)
    parser.add_argument("--rear-threat-roll-delta-damping", type=float)
    parser.add_argument("--rear-threat-rudder-gain", type=float)
    parser.add_argument("--trail-distance-m", type=float)
    parser.add_argument("--lead-time-s", type=float)
    parser.add_argument("--capture-target-pitch-blend", type=float)
    parser.add_argument("--capture-lead-time-s", type=float)
    parser.add_argument("--capture-maneuver-range-m", type=float)
    parser.add_argument("--capture-maneuver-target-pitch-deg", type=float)
    parser.add_argument("--capture-maneuver-target-bank-deg", type=float)
    parser.add_argument("--capture-maneuver-lead-time-s", type=float)
    parser.add_argument("--capture-extreme-bank-deg", type=float)
    parser.add_argument("--capture-extreme-bank-lead-time-s", type=float)
    parser.add_argument("--capture-slow-target-speed-mps", type=float)
    parser.add_argument("--capture-slow-target-range-m", type=float)
    parser.add_argument("--capture-slow-target-lead-time-s", type=float)
    parser.add_argument(
        "--canonicalize-negative-target-bank", action="store_true"
    )
    parser.add_argument("--target-bank-feedforward-range-m", type=float, default=0.0)
    parser.add_argument(
        "--target-bank-feedforward-blend", type=float, default=0.0
    )
    parser.add_argument("--wez-angle-deg", type=float)
    parser.add_argument("--target-altitude-guard", action="store_true")
    parser.add_argument(
        "--target-altitude-guard-trigger-m", type=float, default=2500.0
    )
    parser.add_argument(
        "--target-altitude-guard-release-m", type=float, default=4000.0
    )
    parser.add_argument(
        "--target-altitude-guard-roll-level-gain", type=float, default=0.02
    )
    parser.add_argument(
        "--target-altitude-guard-pull-up-action", type=float, default=-1.0
    )
    parser.add_argument(
        "--target-altitude-guard-throttle-action", type=float, default=1.0
    )
    parser.add_argument(
        "--target-altitude-guard-proportional-pull-up", action="store_true"
    )
    parser.add_argument(
        "--target-altitude-guard-minimum-pull-fraction", type=float, default=0.0
    )
    parser.add_argument(
        "--preserve-guidance",
        action="store_true",
        help="Only replace residual scale and WEZ angle; keep guidance gains unchanged.",
    )
    parser.add_argument(
        "--ownship-state",
        help="Optional seven-value N,E,D,roll,pitch,yaw,speed initial state.",
    )
    parser.add_argument(
        "--target-state",
        help="Optional seven-value N,E,D,roll,pitch,yaw,speed initial state.",
    )
    parser.add_argument("--clear-initial-scenario", action="store_true")
    parser.add_argument("--max-engage-time", type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source = Path(args.source_bundle).resolve()
    output = Path(args.output_bundle).resolve()
    if output.exists():
        raise FileExistsError(f"output bundle already exists: {output}")
    residual_scale_text = args.residual_scale
    if residual_scale_text is None and not args.preserve_guidance:
        residual_scale_text = "0.05,0.05,0.05,0.05"
    scales = (
        [float(part.strip()) for part in residual_scale_text.split(",")]
        if residual_scale_text is not None
        else None
    )
    if scales is not None and len(scales) != 4:
        raise ValueError("--residual-scale must contain four values")
    near_scales = None
    if args.near_residual_scale:
        near_scales = [
            float(part.strip()) for part in args.near_residual_scale.split(",")
        ]
        if len(near_scales) != 4:
            raise ValueError("--near-residual-scale must contain four values")

    metadata = json.loads((source / "metadata.json").read_text(encoding="utf-8"))
    env_config = metadata["algorithm_config"]["env_config"]
    guidance = dict(env_config.get("pursuit_guidance_residual", {}))
    guidance["enabled"] = True
    if scales is not None:
        guidance["residual_scale"] = scales
    if near_scales is not None:
        guidance.update({
            "near_residual_scale": near_scales,
            "near_distance_m": args.near_distance_m,
            "near_ata_deg": args.near_ata_deg,
        })
    if not args.preserve_guidance:
        guidance.update(
            {
                "capture_target_bank_blend": (
                    0.50
                    if args.capture_target_bank_blend is None
                    else args.capture_target_bank_blend
                ),
                "capture_target_bank_blend_start_ata_deg": (
                    0.0
                    if args.capture_target_bank_blend_start_ata_deg is None
                    else args.capture_target_bank_blend_start_ata_deg
                ),
                "capture_roll_gain": (
                    0.015 if args.capture_roll_gain is None else args.capture_roll_gain
                ),
                "capture_roll_delta_damping": (
                    0.20
                    if args.capture_roll_delta_damping is None
                    else args.capture_roll_delta_damping
                ),
                "bank_pitch_compensation_gain": args.bank_pitch_compensation_gain,
                "latch_capture_mode": True,
                "wez_hold_latch_ata_deg": (
                    5.0
                    if args.wez_hold_latch_ata_deg is None
                    else args.wez_hold_latch_ata_deg
                ),
                "wez_hold_target_bank_blend": (
                    1.0
                    if args.wez_hold_target_bank_blend is None
                    else args.wez_hold_target_bank_blend
                ),
                "wez_hold_roll_gain": (
                    0.015
                    if args.wez_hold_roll_gain is None
                    else args.wez_hold_roll_gain
                ),
                "wez_hold_target_heading_blend": (
                    0.25
                    if args.wez_hold_target_heading_blend is None
                    else args.wez_hold_target_heading_blend
                ),
                "wez_hold_target_pitch_blend": (
                    0.0
                    if args.wez_hold_target_pitch_blend is None
                    else args.wez_hold_target_pitch_blend
                ),
                "capture_speed_gain": (
                    0.02
                    if args.capture_speed_gain is None
                    else args.capture_speed_gain
                ),
                "target_bank_feedforward_range_m": (
                    args.target_bank_feedforward_range_m
                ),
                "target_bank_feedforward_blend": (
                    args.target_bank_feedforward_blend
                ),
            }
        )
    optional_guidance = {
        "capture_target_bank_blend": (
            args.capture_target_bank_blend if args.preserve_guidance else None
        ),
        "capture_target_bank_blend_start_ata_deg": (
            args.capture_target_bank_blend_start_ata_deg
            if args.preserve_guidance
            else None
        ),
        "capture_speed_gain": (
            args.capture_speed_gain if args.preserve_guidance else None
        ),
        "capture_speed_offset_mps": args.capture_speed_offset_mps,
        "brake_heading_error_deg": args.brake_heading_error_deg,
        "brake_range_m": args.brake_range_m,
        "capture_range_m": args.capture_range_m,
        "near_throttle_action": args.near_throttle_action,
        "proximity_speed_match_start_distance_m": (
            args.proximity_speed_match_start_distance_m
        ),
        "proximity_speed_match_gain": args.proximity_speed_match_gain,
        "proximity_speed_match_min_overspeed_mps": (
            args.proximity_speed_match_min_overspeed_mps
        ),
        "proximity_speed_match_max_ata_deg": (
            args.proximity_speed_match_max_ata_deg
        ),
        "proximity_speed_match_latch": (
            True if args.proximity_speed_match_latch else None
        ),
        "rear_threat_distance_m": args.rear_threat_distance_m,
        "rear_threat_ata_deg": args.rear_threat_ata_deg,
        "rear_threat_roll_gain": args.rear_threat_roll_gain,
        "rear_threat_bank_pitch_compensation_gain": (
            args.rear_threat_bank_pitch_compensation_gain
        ),
        "rear_threat_turn_latch": (
            True if args.rear_threat_turn_latch else None
        ),
        "rear_threat_turn_release_ata_deg": (
            args.rear_threat_turn_release_ata_deg
        ),
        "rear_threat_roll_delta_damping": (
            args.rear_threat_roll_delta_damping
        ),
        "rear_threat_rudder_gain": args.rear_threat_rudder_gain,
        "trail_distance_m": args.trail_distance_m,
        "lead_time_s": args.lead_time_s,
        "capture_target_pitch_blend": args.capture_target_pitch_blend,
        "capture_lead_time_s": args.capture_lead_time_s,
        "capture_maneuver_range_m": args.capture_maneuver_range_m,
        "capture_maneuver_target_pitch_deg": (
            args.capture_maneuver_target_pitch_deg
        ),
        "capture_maneuver_target_bank_deg": (
            args.capture_maneuver_target_bank_deg
        ),
        "capture_maneuver_lead_time_s": args.capture_maneuver_lead_time_s,
        "capture_extreme_bank_deg": args.capture_extreme_bank_deg,
        "capture_extreme_bank_lead_time_s": (
            args.capture_extreme_bank_lead_time_s
        ),
        "capture_slow_target_speed_mps": args.capture_slow_target_speed_mps,
        "capture_slow_target_range_m": args.capture_slow_target_range_m,
        "capture_slow_target_lead_time_s": (
            args.capture_slow_target_lead_time_s
        ),
        "capture_roll_gain": (
            args.capture_roll_gain if args.preserve_guidance else None
        ),
        "roll_gain": args.roll_gain,
        "max_bank_deg": args.max_bank_deg,
        "heading_to_bank": args.heading_to_bank,
        "pitch_gain": args.pitch_gain,
        "capture_pitch_gain": args.capture_pitch_gain,
        "suppress_opposing_roll_heading_error_deg": (
            args.suppress_opposing_roll_heading_error_deg
        ),
        "opposing_roll_guidance_scale": args.opposing_roll_guidance_scale,
        "capture_roll_delta_damping": (
            args.capture_roll_delta_damping if args.preserve_guidance else None
        ),
        "capture_pitch_delta_damping": args.capture_pitch_delta_damping,
        "capture_pitch_error_rate_gain": args.capture_pitch_error_rate_gain,
        "capture_body_axis_guidance": (
            True if args.capture_body_axis_guidance else None
        ),
        "capture_body_axis_latch_only": (
            True if args.capture_body_axis_latch_only else None
        ),
        "capture_body_axis_roll_gain": args.capture_body_axis_roll_gain,
        "capture_body_axis_pitch_gain": args.capture_body_axis_pitch_gain,
        "capture_body_axis_roll_rate_gain": args.capture_body_axis_roll_rate_gain,
        "capture_body_axis_pitch_rate_gain": args.capture_body_axis_pitch_rate_gain,
        "capture_heading_rate_to_bank": args.capture_heading_rate_to_bank,
        "capture_heading_rate_to_rudder": args.capture_heading_rate_to_rudder,
        "capture_heading_bias_deg": args.capture_heading_bias_deg,
        "capture_bias_target_roll_sign": (
            True if args.capture_bias_target_roll_sign else None
        ),
        "wez_hold_latch_ata_deg": (
            args.wez_hold_latch_ata_deg if args.preserve_guidance else None
        ),
        "wez_hold_target_bank_blend": (
            args.wez_hold_target_bank_blend if args.preserve_guidance else None
        ),
        "wez_hold_roll_gain": (
            args.wez_hold_roll_gain if args.preserve_guidance else None
        ),
        "wez_hold_roll_delta_damping": args.wez_hold_roll_delta_damping,
        "wez_hold_freeze_bank": (
            True if args.wez_hold_freeze_bank else None
        ),
        "wez_hold_pitch_gain": args.wez_hold_pitch_gain,
        "wez_hold_pitch_delta_damping": args.wez_hold_pitch_delta_damping,
        "wez_hold_target_heading_blend": (
            args.wez_hold_target_heading_blend if args.preserve_guidance else None
        ),
        "wez_hold_target_pitch_blend": (
            args.wez_hold_target_pitch_blend if args.preserve_guidance else None
        ),
    }
    guidance.update({
        key: value
        for key, value in optional_guidance.items()
        if value is not None
    })
    env_config["pursuit_guidance_residual"] = guidance
    for option_name, config_name in (
        ("ownship_state", "ownship"),
        ("target_state", "target"),
    ):
        state_text = getattr(args, option_name)
        if state_text is not None:
            state_values = [float(part.strip()) for part in state_text.split(",")]
            if len(state_values) != 7:
                option = option_name.replace("_", "-")
                raise ValueError(f"--{option} must contain seven values")
            env_config[config_name] = state_values
    if args.clear_initial_scenario:
        env_config.pop("initial_scenario", None)
        env_config["ownship_randomization"] = {"enabled": False}
    if args.max_engage_time is not None:
        env_config["max_engage_time"] = float(args.max_engage_time)
    if args.canonicalize_negative_target_bank:
        env_config["canonicalize_negative_target_bank"] = True
    if args.wez_angle_deg is not None:
        env_config.setdefault("wez", {})["angle_deg"] = args.wez_angle_deg
    if args.target_altitude_guard:
        if (
            args.target_altitude_guard_release_m
            <= args.target_altitude_guard_trigger_m
        ):
            raise ValueError(
                "target altitude guard release altitude must exceed trigger altitude"
            )
        env_config["target_altitude_guard"] = {
            "enabled": True,
            "trigger_altitude_m": args.target_altitude_guard_trigger_m,
            "release_altitude_m": args.target_altitude_guard_release_m,
            "roll_level_gain": args.target_altitude_guard_roll_level_gain,
            "pull_up_action": args.target_altitude_guard_pull_up_action,
            "throttle_action": args.target_altitude_guard_throttle_action,
            "proportional_pull_up": (
                args.target_altitude_guard_proportional_pull_up
            ),
            "minimum_pull_fraction": (
                args.target_altitude_guard_minimum_pull_fraction
            ),
        }
    details = metadata.setdefault("metadata", {})
    details.update(
        {
            "guidance_config_cloned_from": str(source),
            "guidance_residual_scale": guidance.get("residual_scale"),
            "target_altitude_guard": env_config.get("target_altitude_guard"),
        }
    )

    output.mkdir(parents=True)
    (output / "metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    shutil.copy2(source / "policy_weights.pkl.gz", output / "policy_weights.pkl.gz")

    # Fail early if the copied policy artifact is not readable.
    with gzip.open(output / "policy_weights.pkl.gz", "rb") as handle:
        pickle.load(handle)
    print(f"cloned bundle: {output}")


if __name__ == "__main__":
    main()
