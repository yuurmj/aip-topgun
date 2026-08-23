from __future__ import annotations

import numpy as np

from dogfight.sim.state_schema import StateIndex


TACTICAL_MIRROR_INDICES = {
    16: (0, 2, 7, 11),
    17: (0, 2, 8, 12),
    21: (0, 2, 8, 12, 17, 18),
    25: (0, 2, 8, 12, 17, 18, 21, 22, 23, 24),
}


def mirror_tactical_observation(observation) -> np.ndarray:
    mirrored = np.asarray(observation, dtype=np.float32).copy()
    indices = TACTICAL_MIRROR_INDICES.get(int(mirrored.size))
    if indices is None:
        raise ValueError(
            "tactical mirror supports observation sizes 16/17/21/25, "
            f"got {mirrored.size}"
        )
    mirrored[list(indices)] *= -1.0
    return mirrored


def normalize(value: float, minimum: float, maximum: float) -> float:
    if maximum <= minimum:
        return 0.0
    clipped = float(np.clip(value, minimum, maximum))
    midpoint = (maximum + minimum) / 2.0
    half_range = (maximum - minimum) / 2.0
    return (clipped - midpoint) / half_range


def observation_size(mode: str) -> int:
    if mode == "tactical25":
        return 25
    if mode == "tactical21":
        return 21
    if mode == "tactical17":
        return 17
    if mode == "tactical16":
        return 16
    if mode == "relative14":
        return 14
    return 12  # classic12


def build_observation(mode: str, ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    if mode == "tactical25":
        return _build_tactical25(ownship_state, target_state, geo_info, wez_config)
    if mode == "tactical21":
        return _build_tactical21(ownship_state, target_state, geo_info, wez_config)
    if mode == "tactical17":
        return _build_tactical17(ownship_state, target_state, geo_info, wez_config)
    if mode == "tactical16":
        return _build_tactical16(ownship_state, target_state, geo_info, wez_config)
    if mode == "relative14":
        return _build_relative14(ownship_state, target_state, geo_info)
    return _build_classic12(ownship_state, target_state)


def describe_observation(mode: str) -> dict:
    if mode == "tactical25":
        base = describe_observation("tactical21")
        return {
            "mode": "tactical25",
            "size": 25,
            "features": base["features"] + [
                "ownship_yaw_rate_norm",
                "target_yaw_rate_norm",
                "relative_yaw_rate_norm",
                "los_azimuth_rate_norm",
            ],
            "description": (
                "Tactical21 plus ownship/target yaw rates, relative yaw rate, "
                "and horizontal line-of-sight azimuth rate for moving-target "
                "turn synchronization."
            ),
        }
    if mode == "tactical21":
        base = describe_observation("tactical17")
        return {
            "mode": "tactical21",
            "size": 21,
            "features": base["features"] + [
                "target_roll_norm",
                "relative_heading_sin",
                "relative_heading_cos",
                "relative_speed_norm",
            ],
            "description": (
                "Tactical17 plus target bank, periodic relative heading, and "
                "relative speed for moving-target WEZ hold control."
            ),
        }
    if mode == "tactical17":
        return {
            "mode": "tactical17",
            "size": 17,
            "features": [
                "ownship_roll_norm",
                "ownship_pitch_norm",
                "ownship_yaw_sin",
                "ownship_yaw_cos",
                "ownship_speed_norm",
                "ownship_alt_norm",
                "ownship_health_norm",
                "delta_n_norm",
                "delta_e_norm",
                "delta_d_norm",
                "ata_norm",
                "aa_norm",
                "az_norm",
                "el_norm",
                "target_health_norm",
                "in_wez",
                "pursuit_score_norm",
            ],
            "description": (
                "Periodic-yaw tactical observation. Yaw is encoded as sine and "
                "cosine so 359 and 0 degrees remain adjacent; all features are "
                "bounded to [-1, 1]."
            ),
        }
    if mode == "tactical16":
        return {
            "mode": "tactical16",
            "size": 16,
            "features": [
                "ownship_roll_norm",
                "ownship_pitch_norm",
                "ownship_yaw_norm",
                "ownship_speed_norm",
                "ownship_alt_norm",
                "ownship_health_norm",
                "delta_n_norm",
                "delta_e_norm",
                "delta_d_norm",
                "ata_norm",
                "aa_norm",
                "az_norm",
                "el_norm",
                "target_health_norm",
                "in_wez",
                "pursuit_score_norm",
            ],
            "description": (
                "Full tactical observation: ownship attitude + speed + altitude + health, "
                "relative geometry (ATA, AA, LOS), target health, WEZ flag, pursuit score. "
                "All features normalized to [-1, 1]. Observation space bounds: [-1, 1]."
            ),
        }
    if mode == "relative14":
        return {
            "mode": "relative14",
            "size": 14,
            "features": [
                "delta_n",
                "delta_e",
                "delta_d",
                "ownship_roll_norm",
                "ownship_pitch_norm",
                "ownship_yaw_norm",
                "target_roll_norm",
                "target_pitch_norm",
                "target_yaw_norm",
                "distance_norm",
                "ata_norm",
                "aa_norm",
                "az_norm",
                "el_norm",
            ],
            "description": "Relative geometry observation with normalized attitude and LOS terms.",
        }
    return {
        "mode": "classic12",
        "size": 12,
        "features": [
            "ownship_n",
            "ownship_e",
            "ownship_d",
            "target_n",
            "target_e",
            "target_d",
            "ownship_roll_norm",
            "ownship_pitch_norm",
            "ownship_yaw_norm",
            "target_roll_norm",
            "target_pitch_norm",
            "target_yaw_norm",
        ],
        "description": "Basic position and normalized attitude observation.",
    }


def _build_classic12(ownship_state, target_state) -> np.ndarray:
    observation = np.zeros(12, dtype=np.float32)
    observation[0] = ownship_state[StateIndex.N]
    observation[1] = ownship_state[StateIndex.E]
    observation[2] = ownship_state[StateIndex.D]
    observation[3] = target_state[StateIndex.N]
    observation[4] = target_state[StateIndex.E]
    observation[5] = target_state[StateIndex.D]
    observation[6] = normalize(ownship_state[StateIndex.ROLL], -180.0, 180.0)
    observation[7] = normalize(ownship_state[StateIndex.PITCH], -90.0, 90.0)
    observation[8] = normalize(ownship_state[StateIndex.YAW], 0.0, 360.0)
    observation[9] = normalize(target_state[StateIndex.ROLL], -180.0, 180.0)
    observation[10] = normalize(target_state[StateIndex.PITCH], -90.0, 90.0)
    observation[11] = normalize(target_state[StateIndex.YAW], 0.0, 360.0)
    return observation


def _build_relative14(ownship_state, target_state, geo_info) -> np.ndarray:
    observation = np.zeros(14, dtype=np.float32)
    delta = target_state[:3] - ownship_state[:3]
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)

    observation[0] = normalize(delta[0], -10000.0, 10000.0)
    observation[1] = normalize(delta[1], -10000.0, 10000.0)
    observation[2] = normalize(delta[2], -5000.0, 5000.0)
    observation[3] = normalize(ownship_state[StateIndex.ROLL], -180.0, 180.0)
    observation[4] = normalize(ownship_state[StateIndex.PITCH], -90.0, 90.0)
    observation[5] = normalize(ownship_state[StateIndex.YAW], 0.0, 360.0)
    observation[6] = normalize(target_state[StateIndex.ROLL], -180.0, 180.0)
    observation[7] = normalize(target_state[StateIndex.PITCH], -90.0, 90.0)
    observation[8] = normalize(target_state[StateIndex.YAW], 0.0, 360.0)
    observation[9] = normalize(distance, 0.0, 20000.0)
    observation[10] = normalize(ata, -180.0, 180.0)
    observation[11] = normalize(aa, -180.0, 180.0)
    observation[12] = normalize(az, -180.0, 180.0)
    observation[13] = normalize(el, -90.0, 90.0)
    return observation


def _build_tactical16(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """16-feature tactical observation.

    Index map:
      0-5   ownship: roll, pitch, yaw, speed(KCAS), altitude, health
      6-8   relative position: delta_n, delta_e, delta_d
      9-12  geometry: ATA, AA, LOS_az, LOS_el
      13    target health
      14    in_wez flag  (-1 / +1)
      15    pursuit score (smooth ATA×range gradient, normalized to [-1,1])
    """
    obs = np.zeros(16, dtype=np.float32)

    delta = target_state[:3] - ownship_state[:3]
    distance = geo_info._get_distance(ownship_state, target_state)
    ata = geo_info._get_antenna_train_angle(ownship_state, target_state, False)
    aa = geo_info._get_aspect_angle(ownship_state, target_state, False)
    az, el = geo_info._get_los_angle(ownship_state, target_state)

    # Ownship state
    obs[0] = normalize(float(ownship_state[StateIndex.ROLL]),   -180.0, 180.0)
    obs[1] = normalize(float(ownship_state[StateIndex.PITCH]),   -90.0,  90.0)
    obs[2] = normalize(float(ownship_state[StateIndex.YAW]),       0.0, 360.0)
    obs[3] = normalize(float(ownship_state[StateIndex.KCAS]),      0.0, 600.0)
    obs[4] = normalize(float(ownship_state[StateIndex.ALT]),       0.0, 15000.0)
    obs[5] = normalize(float(ownship_state[StateIndex.HEALTH]),    0.0,  1.0)

    # Relative position
    obs[6] = normalize(float(delta[0]), -15000.0, 15000.0)
    obs[7] = normalize(float(delta[1]), -15000.0, 15000.0)
    obs[8] = normalize(float(delta[2]),  -8000.0,  8000.0)

    # Geometry
    obs[9]  = normalize(float(ata),  -180.0, 180.0)
    obs[10] = normalize(float(aa),   -180.0, 180.0)
    obs[11] = normalize(float(az),   -180.0, 180.0)
    obs[12] = normalize(float(el),    -90.0,  90.0)

    # Target health
    obs[13] = normalize(float(target_state[StateIndex.HEALTH]), 0.0, 1.0)

    # WEZ flag: +1 if ownship is inside weapon engagement zone, -1 otherwise
    if wez_config is not None:
        ata_abs = abs(float(ata))
        in_wez = (
            wez_config["min_range_m"] <= distance <= wez_config["max_range_m"]
            and ata_abs <= wez_config["angle_deg"] / 2.0
        )
        obs[14] = 1.0 if in_wez else -1.0
    else:
        obs[14] = -1.0

    # Pursuit score: smooth ATA×range gradient in [-1, 1]
    # Thresholds are observation-level constants (not tied to reward config)
    ata_factor   = max(0.0, 1.0 - abs(float(ata)) / 30.0)   # full score at ATA=0°
    range_factor = max(0.0, 1.0 - distance / 3000.0)          # full score at distance=0
    pursuit_raw  = ata_factor * range_factor                   # [0, 1]
    obs[15] = 2.0 * pursuit_raw - 1.0                         # → [-1, 1]

    return obs


def _build_tactical17(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """Tactical observation with a continuous two-channel yaw encoding."""
    tactical16 = _build_tactical16(
        ownship_state,
        target_state,
        geo_info,
        wez_config,
    )
    yaw_rad = np.deg2rad(float(ownship_state[StateIndex.YAW]))
    obs = np.empty(17, dtype=np.float32)
    obs[0:2] = tactical16[0:2]
    obs[2] = np.sin(yaw_rad)
    obs[3] = np.cos(yaw_rad)
    obs[4:] = tactical16[3:]
    return obs


def _build_tactical21(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """Tactical17 plus target maneuver and relative-velocity observability."""
    tactical17 = _build_tactical17(
        ownship_state,
        target_state,
        geo_info,
        wez_config,
    )
    obs = np.empty(21, dtype=np.float32)
    obs[:17] = tactical17
    obs[17] = normalize(float(target_state[StateIndex.ROLL]), -180.0, 180.0)
    heading_difference_rad = np.deg2rad(
        (float(target_state[StateIndex.YAW]) - float(ownship_state[StateIndex.YAW]) + 180.0)
        % 360.0
        - 180.0
    )
    obs[18] = np.sin(heading_difference_rad)
    obs[19] = np.cos(heading_difference_rad)
    ownship_speed = float(np.linalg.norm(ownship_state[StateIndex.U : StateIndex.W + 1]))
    target_speed = float(np.linalg.norm(target_state[StateIndex.U : StateIndex.W + 1]))
    obs[20] = normalize(target_speed - ownship_speed, -200.0, 200.0)
    return obs


def _body_velocity_to_ned(state) -> np.ndarray:
    """Rotate body-axis u/v/w velocity into the local NED frame."""
    roll, pitch, yaw = np.deg2rad(
        [
            float(state[StateIndex.ROLL]),
            float(state[StateIndex.PITCH]),
            float(state[StateIndex.YAW]),
        ]
    )
    sr, cr = np.sin(roll), np.cos(roll)
    sp, cp = np.sin(pitch), np.cos(pitch)
    sy, cy = np.sin(yaw), np.cos(yaw)
    rotation = np.asarray(
        [
            [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
            [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
            [-sp, sr * cp, cr * cp],
        ],
        dtype=np.float64,
    )
    return rotation @ np.asarray(
        state[StateIndex.U : StateIndex.W + 1], dtype=np.float64
    )


def _horizontal_los_rate_deg_s(ownship_state, target_state) -> float:
    relative_position = np.asarray(target_state[:2], dtype=np.float64) - np.asarray(
        ownship_state[:2], dtype=np.float64
    )
    horizontal_range_squared = float(np.dot(relative_position, relative_position))
    if horizontal_range_squared <= 1.0e-6:
        return 0.0
    relative_velocity = (
        _body_velocity_to_ned(target_state) - _body_velocity_to_ned(ownship_state)
    )[:2]
    north, east = relative_position
    north_rate, east_rate = relative_velocity
    return float(
        np.degrees((north * east_rate - east * north_rate) / horizontal_range_squared)
    )


def _build_tactical25(ownship_state, target_state, geo_info, wez_config=None) -> np.ndarray:
    """Tactical21 plus angular-rate observability for turn synchronization."""
    tactical21 = _build_tactical21(
        ownship_state,
        target_state,
        geo_info,
        wez_config,
    )
    obs = np.empty(25, dtype=np.float32)
    obs[:21] = tactical21
    ownship_yaw_rate = float(ownship_state[StateIndex.R])
    target_yaw_rate = float(target_state[StateIndex.R])
    obs[21] = normalize(ownship_yaw_rate, -30.0, 30.0)
    obs[22] = normalize(target_yaw_rate, -30.0, 30.0)
    obs[23] = normalize(target_yaw_rate - ownship_yaw_rate, -30.0, 30.0)
    obs[24] = normalize(
        _horizontal_los_rate_deg_s(ownship_state, target_state), -30.0, 30.0
    )
    return obs
