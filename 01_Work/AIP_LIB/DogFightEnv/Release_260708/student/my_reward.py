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
    # 기존 training record 및 reward description 호환용 키
    "survival_bonus": 0.0,
    "damage_scale": 30.0,
    "pursuit_scale": 0.05,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 2500.0,
    "step_penalty": -0.01,

    # v10 WEZ 접근 및 조준 Reward
    "alignment_scale": 0.05,
    "pre_wez_outer_margin_m": 1500.0,

    "coarse_aim_angle_deg": 30.0,
    "coarse_aim_bonus": 0.08,

    "fine_aim_angle_deg": 10.0,
    "fine_aim_bonus": 0.12,

    # 실제 WEZ 내부 유지 보상
    "wez_hold_bonus": 0.50,

    # 피해 Reward
    "target_damage_scale": 30.0,
    "ownship_damage_scale": 25.0,

    # 안전 고도 Reward
    "safe_altitude_min_m": 5000.0,
    "safe_altitude_max_m": 8000.0,
    "safe_altitude_bonus": 0.01,

    # 저고도 패널티
    "low_altitude_warning_m": 5000.0,
    "low_altitude_m": 3000.0,
    "low_altitude_penalty": 0.50,

    # v10 급강하 조기 감지
    "dive_pitch_start_deg": -3.0,
    "dive_pitch_full_deg": -25.0,
    "dive_pitch_penalty": 0.30,

    # 너무 가까운 상태에서 표적을 놓쳤을 때 패널티
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


def _linear_factor(
    value: float,
    start: float,
    full: float,
) -> float:
    """
    start에서 full 방향으로 증가하는 0~1 값을 반환한다.

    예:
        start = -3
        full = -25

        pitch -3도  -> 0
        pitch -14도 -> 약 0.5
        pitch -25도 -> 1
    """
    span = full - start

    if span == 0.0:
        return 1.0 if value == full else 0.0

    return max(
        0.0,
        min(
            1.0,
            (value - start) / span,
        ),
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
        1.0 - ata_deg / full_angle_deg,
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
        - (distance_m - wez_max_m)
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
    v10 WEZ hold 학습용 Reward.

    training_log.csv 호환을 위해 components는 기존과 동일하게
    pursuit, damage, safety, survival 네 항목만 사용한다.
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
    #
    # 표적을 등지는 행동도 음수 신호를 받는다.
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
    # min_range_m = 약 152.4m
    # max_range_m = 약 914.4m
    # angle_deg = 2도
    #
    # 실제 WEZ 판정은 angle_deg / 2이므로 ATA 1도 이하이다.
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
    coarse_aim_angle_deg = float(
        reward_config.get(
            "coarse_aim_angle_deg",
            30.0,
        )
    )

    coarse_aim_bonus = float(
        reward_config.get(
            "coarse_aim_bonus",
            0.08,
        )
    )

    coarse_aim_reward = (
        coarse_aim_bonus
        * _aim_factor(
            ata_deg,
            coarse_aim_angle_deg,
        )
        * range_readiness
    )

    # ------------------------------------------------------------
    # 5. 10도 이하 fine aim
    #
    # ATA 30도 이하 Reward보다 더 좁은 범위에서 작동한다.
    # 실제 WEZ ATA 1도까지 접근하기 위한 중간 단계이다.
    # ------------------------------------------------------------
    fine_aim_angle_deg = float(
        reward_config.get(
            "fine_aim_angle_deg",
            10.0,
        )
    )

    fine_aim_bonus = float(
        reward_config.get(
            "fine_aim_bonus",
            0.12,
        )
    )

    fine_aim_reward = (
        fine_aim_bonus
        * _aim_factor(
            ata_deg,
            fine_aim_angle_deg,
        )
        * range_readiness
    )

    # ------------------------------------------------------------
    # 6. 실제 WEZ 직접 Reward
    #
    # 실제 WEZ 거리와 ATA 조건을 직접 사용한다.
    # target_damage 발생 여부에 의존하지 않는다.
    # ------------------------------------------------------------
    in_wez = (
        wez_min_m
        <= distance_m
        <= wez_max_m
        and ata_deg
        <= wez_half_angle_deg
    )

    if in_wez:
        direct_wez_reward = float(
            reward_config.get(
                "wez_hold_bonus",
                0.50,
            )
        )
    else:
        direct_wez_reward = 0.0

    pursuit_reward = (
        alignment_reward
        + coarse_aim_reward
        + fine_aim_reward
        + direct_wez_reward
    )

    # ------------------------------------------------------------
    # 7. 피해 Reward
    # ------------------------------------------------------------
    target_damage_scale = float(
        reward_config.get(
            "target_damage_scale",
            30.0,
        )
    )

    ownship_damage_scale = float(
        reward_config.get(
            "ownship_damage_scale",
            25.0,
        )
    )

    damage_reward = (
        target_damage_scale
        * float(target_damage)
        - ownship_damage_scale
        * float(ownship_damage)
    )

    # ------------------------------------------------------------
    # 8. 고도 및 pitch
    #
    # StateIndex에 수직속도 항목이 없으므로 검증된 ALT와
    # PITCH만 사용한다.
    # ------------------------------------------------------------
    altitude_m = float(
        ownship_state[StateIndex.ALT]
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

    if safe_min_m <= altitude_m <= safe_max_m:
        safety_reward += float(
            reward_config.get(
                "safe_altitude_bonus",
                0.01,
            )
        )

    # ------------------------------------------------------------
    # 10. 저고도 패널티
    #
    # 5000m 아래부터 증가하고 3000m 이하에서 최대값이 된다.
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
            low_factor = min(
                1.0,
                (
                    warning_m
                    - altitude_m
                )
                / (
                    warning_m
                    - low_m
                ),
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
            * max(
                0.0,
                low_factor,
            )
        )

    # ------------------------------------------------------------
    # 11. 급강하 pitch 패널티
    #
    # -3도부터 시작하고 -25도에서 최대 -0.30이 된다.
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
            pitch_deg,
            dive_start_deg,
            dive_full_deg,
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
    # 12. 너무 가까운 상태에서 표적을 놓친 경우
    # ------------------------------------------------------------
    close_range_penalty = 0.0

    bad_close_range_m = float(
        reward_config.get(
            "bad_close_range_m",
            300.0,
        )
    )

    bad_close_ata_deg = float(
        reward_config.get(
            "bad_close_ata_deg",
            30.0,
        )
    )

    if (
        distance_m < bad_close_range_m
        and ata_deg > bad_close_ata_deg
    ):
        close_range_penalty = -float(
            reward_config.get(
                "bad_close_penalty",
                0.20,
            )
        )

    # ------------------------------------------------------------
    # 13. 종료 Reward
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
    # 14. Component 기록
    #
    # training_log.csv 호환을 위해 기존 네 개 키만 유지한다.
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