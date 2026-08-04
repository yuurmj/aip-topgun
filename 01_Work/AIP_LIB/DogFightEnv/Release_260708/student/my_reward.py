# -*- coding: utf-8 -*-
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
    # 기존 training record와 호환하기 위한 키
    "survival_bonus": 0.0,
    "damage_scale": 30.0,
    "pursuit_scale": 0.05,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 2500.0,
    "step_penalty": -0.01,

    # 표적 방향 및 WEZ 접근 Reward
    "alignment_scale": 0.05,
    "pre_wez_outer_margin_m": 1500.0,

    "coarse_aim_angle_deg": 30.0,
    "coarse_aim_bonus": 0.08,

    "fine_aim_angle_deg": 10.0,
    "fine_aim_bonus": 0.12,

    # v11 실제 WEZ 유지 및 WEZ 각도 이탈 패널티
    "wez_hold_bonus": 0.75,
    "wez_miss_angle_deg": 10.0,
    "wez_miss_penalty": 0.75,

    # 피해 Reward
    "target_damage_scale": 30.0,
    "ownship_damage_scale": 25.0,

    # 안전 고도
    "safe_altitude_min_m": 5000.0,
    "safe_altitude_max_m": 8000.0,
    "safe_altitude_bonus": 0.01,

    # 저고도 패널티
    "low_altitude_warning_m": 5000.0,
    "low_altitude_m": 3000.0,
    "low_altitude_penalty": 0.50,

    # v11 Roll 안정화
    "roll_penalty_full_deg": 60.0,
    "roll_penalty": 0.30,

    # 수평 Pitch 유지
    "level_pitch_start_deg": 3.0,
    "level_pitch_full_deg": 25.0,
    "level_pitch_penalty": 0.20,

    # 급강하 추가 패널티
    "dive_pitch_start_deg": -3.0,
    "dive_pitch_full_deg": -25.0,
    "dive_pitch_penalty": 0.30,

    # 너무 가까운 거리에서 표적을 놓쳤을 때 패널티
    "bad_close_range_m": 300.0,
    "bad_close_ata_deg": 30.0,
    "bad_close_penalty": 0.20,

    # 종료 Reward
    "win_reward": 150.0,
    "loss_reward": -150.0,
    "crash_penalty": -200.0,
    "guard_fail_penalty": -50.0,
    "target_crash_reward": -100.0,
    "timeout_penalty": -100.0,
    "draw_reward": -10.0,
}


def _clamp01(value: float) -> float:
    """값을 0~1 범위로 제한한다."""
    return max(
        0.0,
        min(
            1.0,
            value,
        ),
    )


def _linear_factor(
    value: float,
    start: float,
    full: float,
) -> float:
    """
    start에서 full 방향으로 증가하는 0~1 계수를 반환한다.

    음수 방향에서도 사용할 수 있다.

    예:
        start = -3
        full = -25

        value = -3  -> 0
        value = -14 -> 약 0.5
        value = -25 -> 1
    """
    span = full - start

    if span == 0.0:
        return 1.0 if value == full else 0.0

    return _clamp01(
        (value - start) / span
    )


def _outside_band_factor(
    absolute_value: float,
    start_abs: float,
    full_abs: float,
) -> float:
    """
    절댓값이 start_abs 이내면 0을 반환한다.

    start_abs를 벗어나면 증가하고 full_abs 이상에서
    1을 반환한다.
    """
    if absolute_value <= start_abs:
        return 0.0

    if full_abs <= start_abs:
        return 1.0

    return _clamp01(
        (
            absolute_value
            - start_abs
        )
        / (
            full_abs
            - start_abs
        )
    )


def _aim_factor(
    ata_deg: float,
    full_angle_deg: float,
) -> float:
    """
    ATA 0도에서 1을 반환하고 제한 각도에서 0을 반환한다.
    """
    if full_angle_deg <= 0.0:
        return 0.0

    return max(
        0.0,
        1.0
        - ata_deg
        / full_angle_deg,
    )


def _range_readiness(
    distance_m: float,
    wez_min_m: float,
    wez_max_m: float,
    outer_margin_m: float,
) -> float:
    """
    실제 WEZ 거리 안에서는 1을 반환한다.

    WEZ 최대 거리보다 먼 경우 outer_margin_m 구간에 걸쳐
    1에서 0으로 감소한다.

    WEZ 최소 거리보다 가까운 경우에도 거리가 0에
    가까워질수록 0으로 감소한다.
    """
    if wez_min_m < 0.0:
        return 0.0

    if wez_max_m <= wez_min_m:
        return 0.0

    if distance_m < wez_min_m:
        if wez_min_m == 0.0:
            return 0.0

        return max(
            0.0,
            distance_m / wez_min_m,
        )

    if distance_m <= wez_max_m:
        return 1.0

    if outer_margin_m <= 0.0:
        return 0.0

    return max(
        0.0,
        1.0
        - (
            distance_m
            - wez_max_m
        )
        / outer_margin_m,
    )


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
    """
    v11 WEZ 유지 및 Roll/Pitch 안정화 Reward.

    training_log.csv 호환을 위해 component는 기존과 동일하게
    pursuit, damage, safety, survival 네 개만 사용한다.
    """

    # ------------------------------------------------------------
    # 1. 거리와 ATA
    # ------------------------------------------------------------
    distance_m = float(
        geo_info._get_distance(
            ownship_state,
            target_state,
        )
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

    step_reward = float(
        reward_config.get(
            "step_penalty",
            -0.01,
        )
    )

    # ------------------------------------------------------------
    # 2. 전체 방향 정렬 Reward
    #
    # ATA 0도   : +alignment_scale
    # ATA 90도  : 0
    # ATA 180도 : -alignment_scale
    # ------------------------------------------------------------
    alignment_scale = float(
        reward_config.get(
            "alignment_scale",
            0.05,
        )
    )

    alignment_reward = (
        alignment_scale
        * math.cos(
            math.radians(ata_deg)
        )
    )

    # ------------------------------------------------------------
    # 3. 실제 환경 WEZ 설정
    #
    # 현재 환경 기본값:
    # 최소 거리 약 152.4m
    # 최대 거리 약 914.4m
    # 전체 각도 2도
    # 실제 판정 반각은 1도
    # ------------------------------------------------------------
    wez_min_m = float(
        wez_config["min_range_m"]
    )

    wez_max_m = float(
        wez_config["max_range_m"]
    )

    wez_half_angle_deg = (
        float(
            wez_config["angle_deg"]
        )
        / 2.0
    )

    range_readiness = _range_readiness(
        distance_m=distance_m,
        wez_min_m=wez_min_m,
        wez_max_m=wez_max_m,
        outer_margin_m=float(
            reward_config.get(
                "pre_wez_outer_margin_m",
                1500.0,
            )
        ),
    )

    # ------------------------------------------------------------
    # 4. 30도 이하 coarse aim
    # ------------------------------------------------------------
    coarse_aim_reward = (
        float(
            reward_config.get(
                "coarse_aim_bonus",
                0.08,
            )
        )
        * _aim_factor(
            ata_deg=ata_deg,
            full_angle_deg=float(
                reward_config.get(
                    "coarse_aim_angle_deg",
                    30.0,
                )
            ),
        )
        * range_readiness
    )

    # ------------------------------------------------------------
    # 5. 10도 이하 fine aim
    # ------------------------------------------------------------
    fine_aim_reward = (
        float(
            reward_config.get(
                "fine_aim_bonus",
                0.12,
            )
        )
        * _aim_factor(
            ata_deg=ata_deg,
            full_angle_deg=float(
                reward_config.get(
                    "fine_aim_angle_deg",
                    10.0,
                )
            ),
        )
        * range_readiness
    )

    # ------------------------------------------------------------
    # 6. 실제 WEZ 유지 및 각도 이탈 패널티
    #
    # 실제 WEZ 안:
    #   +0.75
    #
    # 실제 WEZ 거리 안이지만 ATA가 1도를 초과:
    #   ATA 오차에 따라 최대 -0.75
    # ------------------------------------------------------------
    in_wez_range = (
        wez_min_m
        <= distance_m
        <= wez_max_m
    )

    in_wez = (
        in_wez_range
        and ata_deg
        <= wez_half_angle_deg
    )

    direct_wez_reward = 0.0
    wez_miss_reward = 0.0

    if in_wez:
        direct_wez_reward = float(
            reward_config.get(
                "wez_hold_bonus",
                0.75,
            )
        )

    elif in_wez_range:
        miss_angle_deg = float(
            reward_config.get(
                "wez_miss_angle_deg",
                10.0,
            )
        )

        if miss_angle_deg > 0.0:
            miss_factor = _clamp01(
                (
                    ata_deg
                    - wez_half_angle_deg
                )
                / miss_angle_deg
            )
        else:
            miss_factor = 1.0

        wez_miss_reward = (
            -float(
                reward_config.get(
                    "wez_miss_penalty",
                    0.75,
                )
            )
            * miss_factor
        )

    pursuit_reward = (
        alignment_reward
        + coarse_aim_reward
        + fine_aim_reward
        + direct_wez_reward
        + wez_miss_reward
    )

    # ------------------------------------------------------------
    # 7. 피해 Reward
    # ------------------------------------------------------------
    damage_reward = (
        float(
            reward_config.get(
                "target_damage_scale",
                30.0,
            )
        )
        * float(target_damage)
        - float(
            reward_config.get(
                "ownship_damage_scale",
                25.0,
            )
        )
        * float(ownship_damage)
    )

    # ------------------------------------------------------------
    # 8. 고도, Roll, Pitch
    #
    # 실제 소스에 존재하는 StateIndex만 사용한다.
    # ------------------------------------------------------------
    altitude_m = float(
        ownship_state[StateIndex.ALT]
    )

    roll_deg = abs(
        float(
            ownship_state[StateIndex.ROLL]
        )
    )

    pitch_deg = float(
        ownship_state[StateIndex.PITCH]
    )

    safety_reward = 0.0

    # ------------------------------------------------------------
    # 9. 안전 고도 유지
    # ------------------------------------------------------------
    safe_min_m = float(
        reward_config.get(
            "safe_altitude_min_m",
            5000.0,
        )
    )

    safe_max_m = float(
        reward_config.get(
            "safe_altitude_max_m",
            8000.0,
        )
    )

    if (
        safe_min_m
        <= altitude_m
        <= safe_max_m
    ):
        safety_reward += float(
            reward_config.get(
                "safe_altitude_bonus",
                0.01,
            )
        )

    # ------------------------------------------------------------
    # 10. 저고도 패널티
    # ------------------------------------------------------------
    warning_m = float(
        reward_config.get(
            "low_altitude_warning_m",
            5000.0,
        )
    )

    low_m = float(
        reward_config.get(
            "low_altitude_m",
            3000.0,
        )
    )

    if altitude_m < warning_m:
        if warning_m > low_m:
            low_factor = _clamp01(
                (
                    warning_m
                    - altitude_m
                )
                / (
                    warning_m
                    - low_m
                )
            )
        else:
            low_factor = 1.0

        safety_reward -= (
            float(
                reward_config.get(
                    "low_altitude_penalty",
                    0.50,
                )
            )
            * low_factor
        )

    # ------------------------------------------------------------
    # 11. Roll 안정화 패널티
    #
    # Roll 0도  : 0
    # Roll 30도 : 최대값의 약 50%
    # Roll 60도 : 최대값
    # ------------------------------------------------------------
    roll_full_deg = float(
        reward_config.get(
            "roll_penalty_full_deg",
            60.0,
        )
    )

    if roll_full_deg > 0.0:
        roll_factor = _clamp01(
            roll_deg / roll_full_deg
        )
    else:
        roll_factor = (
            1.0
            if roll_deg > 0.0
            else 0.0
        )

    safety_reward -= (
        float(
            reward_config.get(
                "roll_penalty",
                0.30,
            )
        )
        * roll_factor
    )

    # ------------------------------------------------------------
    # 12. 수평 Pitch 유지 패널티
    #
    # Pitch 절댓값 3도 이내: 0
    # Pitch 절댓값 25도 이상: 최대 -0.20
    # ------------------------------------------------------------
    pitch_factor = _outside_band_factor(
        absolute_value=abs(pitch_deg),
        start_abs=float(
            reward_config.get(
                "level_pitch_start_deg",
                3.0,
            )
        ),
        full_abs=float(
            reward_config.get(
                "level_pitch_full_deg",
                25.0,
            )
        ),
    )

    safety_reward -= (
        float(
            reward_config.get(
                "level_pitch_penalty",
                0.20,
            )
        )
        * pitch_factor
    )

    # ------------------------------------------------------------
    # 13. 급강하 추가 패널티
    #
    # 음수 Pitch 방향에만 추가 적용한다.
    # ------------------------------------------------------------
    dive_start_deg = float(
        reward_config.get(
            "dive_pitch_start_deg",
            -3.0,
        )
    )

    dive_full_deg = float(
        reward_config.get(
            "dive_pitch_full_deg",
            -25.0,
        )
    )

    if pitch_deg < dive_start_deg:
        dive_factor = _linear_factor(
            value=pitch_deg,
            start=dive_start_deg,
            full=dive_full_deg,
        )

        safety_reward -= (
            float(
                reward_config.get(
                    "dive_pitch_penalty",
                    0.30,
                )
            )
            * dive_factor
        )

    # ------------------------------------------------------------
    # 14. 지나치게 가까운 상태에서 표적을 놓친 경우
    # ------------------------------------------------------------
    close_range_penalty = 0.0

    if (
        distance_m
        < float(
            reward_config.get(
                "bad_close_range_m",
                300.0,
            )
        )
        and ata_deg
        > float(
            reward_config.get(
                "bad_close_ata_deg",
                30.0,
            )
        )
    ):
        close_range_penalty = -float(
            reward_config.get(
                "bad_close_penalty",
                0.20,
            )
        )

    # ------------------------------------------------------------
    # 15. 종료 Reward
    # ------------------------------------------------------------
    terminal_reward = 0.0

    if terminated:
        if end_condition == "target destroyed":
            terminal_reward = float(
                reward_config.get(
                    "win_reward",
                    150.0,
                )
            )

        elif end_condition == "target altitude below min":
            terminal_reward = float(
                reward_config.get(
                    "target_crash_reward",
                    -100.0,
                )
            )

        elif end_condition == "ownship destroyed":
            terminal_reward = float(
                reward_config.get(
                    "loss_reward",
                    -150.0,
                )
            )

        elif end_condition in (
            "ownship altitude below min",
            "FDM Update Fail",
            "Ownship FDM output Fall",
        ):
            terminal_reward = float(
                reward_config.get(
                    "crash_penalty",
                    -200.0,
                )
            )

        elif end_condition == "two circle headon guard fail":
            terminal_reward = float(
                reward_config.get(
                    "guard_fail_penalty",
                    -50.0,
                )
            )

        else:
            terminal_reward = float(
                reward_config.get(
                    "draw_reward",
                    -10.0,
                )
            )

    elif truncated:
        terminal_reward = float(
            reward_config.get(
                "timeout_penalty",
                -100.0,
            )
        )

    # ------------------------------------------------------------
    # 16. training_log.csv component
    #
    # 기존 네 개 키를 그대로 유지한다.
    # ------------------------------------------------------------
    components = {
        "pursuit": float(
            pursuit_reward
        ),
        "damage": float(
            damage_reward
        ),
        "safety": float(
            safety_reward
        ),
        "survival": 0.0,
    }

    total_reward = (
        step_reward
        + pursuit_reward
        + damage_reward
        + safety_reward
        + close_range_penalty
        + terminal_reward
    )

    return (
        float(total_reward),
        components,
    )


__all__ = [
    "MY_REWARD_CONFIG",
    "compute_reward",
]