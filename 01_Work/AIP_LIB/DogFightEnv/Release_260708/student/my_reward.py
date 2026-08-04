# -*- coding: utf-8 -*-
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    # ============================================================
    # v3 training_record 호환용 설정
    # ============================================================
    "survival_bonus": 0.0,
    "damage_scale": 30.0,
    "pursuit_half_angle_deg": 60.0,

    # ============================================================
    # 시간 보상
    # ============================================================
    "step_penalty": -0.01,

    # ============================================================
    # 상대 추적 보상
    # ============================================================
    # v4: 0.03 → v5: 0.08
    "pursuit_scale": 0.08,
    "pursuit_angle_deg": 60.0,

    # v4: 3000m → v5: 6000m
    "pursuit_range_m": 6000.0,

    # ============================================================
    # 공격 위치 보상
    # ============================================================
    # 약 1,640~3,018ft 거리에서 ATA 15도 이내
    "attack_min_range_m": 500.0,
    "attack_max_range_m": 920.0,
    "attack_ata_deg": 15.0,

    # v4: 0.03 → v5: 0.10
    "attack_position_bonus": 0.10,

    # ============================================================
    # WEZ 및 피해 보상
    # ============================================================
    # v4: 0.05 → v5: 0.15
    "wez_hold_bonus": 0.15,

    # v4: 20.0 → v5: 30.0
    "target_damage_scale": 30.0,
    "ownship_damage_scale": 25.0,

    # ============================================================
    # 안전 관련 보상
    # ============================================================
    "low_altitude_m": 600.0,
    "low_altitude_penalty": 0.20,

    "bad_close_range_m": 300.0,
    "bad_close_ata_deg": 30.0,
    "bad_close_penalty": 0.20,

    # ============================================================
    # 종료 보상
    # ============================================================
    # v4: 100 → v5: 150
    "win_reward": 150.0,

    "loss_reward": -150.0,
    "crash_penalty": -150.0,
    "guard_fail_penalty": -50.0,

    # 상대가 스스로 추락할 때까지 기다리는 행동 방지
    "target_crash_reward": -100.0,

    # 공격하지 않고 시간만 보내는 행동 방지
    "timeout_penalty": -50.0,

    "draw_reward": -10.0,
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
    """
    PPO v5 공격 개선 Reward.

    v3 training_log.csv에서 기록하던 Reward 항목:
        ep_reward_pursuit
        ep_reward_damage
        ep_reward_safety
        ep_reward_survival

    위 네 항목만 components로 반환한다.
    """

    # ============================================================
    # 1. 상대와의 거리 및 ATA 계산
    # ============================================================
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

    # ============================================================
    # 2. 시간 패널티
    # ============================================================
    step_reward = float(
        reward_config.get(
            "step_penalty",
            -0.01,
        )
    )

    # ============================================================
    # 3. 상대 추적 보상
    # ============================================================
    pursuit_angle_deg = float(
        reward_config.get(
            "pursuit_angle_deg",
            60.0,
        )
    )

    pursuit_range_m = float(
        reward_config.get(
            "pursuit_range_m",
            6000.0,
        )
    )

    if pursuit_angle_deg > 0.0:
        ata_factor = max(
            0.0,
            1.0 - ata_deg / pursuit_angle_deg,
        )
    else:
        ata_factor = 0.0

    if pursuit_range_m > 0.0:
        range_factor = max(
            0.0,
            1.0 - distance_m / pursuit_range_m,
        )
    else:
        range_factor = 0.0

    pursuit_reward = (
        float(
            reward_config.get(
                "pursuit_scale",
                0.08,
            )
        )
        * ata_factor
        * range_factor
    )

    # ============================================================
    # 4. 공격하기 좋은 위치 보상
    # ============================================================
    attack_min_range_m = float(
        reward_config.get(
            "attack_min_range_m",
            500.0,
        )
    )

    attack_max_range_m = float(
        reward_config.get(
            "attack_max_range_m",
            920.0,
        )
    )

    attack_ata_deg = float(
        reward_config.get(
            "attack_ata_deg",
            15.0,
        )
    )

    in_attack_position = (
        ata_deg <= attack_ata_deg
        and attack_min_range_m
        <= distance_m
        <= attack_max_range_m
    )

    if in_attack_position:
        pursuit_reward += float(
            reward_config.get(
                "attack_position_bonus",
                0.10,
            )
        )

    # ============================================================
    # 5. WEZ 유지 보상
    # ============================================================
    # 환경에서 target_damage가 0보다 크면,
    # 현재 step에서 아군이 상대를 공격할 수 있는 WEZ 안에 있다는 뜻이다.
    if float(target_damage) > 0.0:
        pursuit_reward += float(
            reward_config.get(
                "wez_hold_bonus",
                0.15,
            )
        )

    # ============================================================
    # 6. 피해량 보상
    # ============================================================
    target_damage_reward = (
        float(
            reward_config.get(
                "target_damage_scale",
                30.0,
            )
        )
        * float(target_damage)
    )

    ownship_damage_penalty = (
        float(
            reward_config.get(
                "ownship_damage_scale",
                25.0,
            )
        )
        * float(ownship_damage)
    )

    damage_reward = (
        target_damage_reward
        - ownship_damage_penalty
    )

    # ============================================================
    # 7. 저고도 안전 패널티
    # ============================================================
    safety_reward = 0.0

    altitude_m = float(
        ownship_state[StateIndex.ALT]
    )

    low_altitude_m = float(
        reward_config.get(
            "low_altitude_m",
            600.0,
        )
    )

    if altitude_m < low_altitude_m:
        safety_reward -= float(
            reward_config.get(
                "low_altitude_penalty",
                0.20,
            )
        )

    # ============================================================
    # 8. 잘못된 근접 패널티
    # ============================================================
    # 상대와 너무 가까운데 ATA가 크면,
    # 공격 자세가 아닌 위험한 근접으로 판단한다.
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

    # ============================================================
    # 9. 종료 보상
    # ============================================================
    terminal_reward = 0.0

    if terminated:
        if end_condition == "target destroyed":
            # 상대를 직접 격추
            terminal_reward = float(
                reward_config.get(
                    "win_reward",
                    150.0,
                )
            )

        elif end_condition == "target altitude below min":
            # 상대가 스스로 추락
            # 공격 성공으로 처리하지 않는다.
            terminal_reward = float(
                reward_config.get(
                    "target_crash_reward",
                    -100.0,
                )
            )

        elif end_condition == "ownship destroyed":
            # 상대에게 격추됨
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
            # 아군 추락 또는 비행 모델 실패
            terminal_reward = float(
                reward_config.get(
                    "crash_penalty",
                    -150.0,
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
        # max time out 또는 episode step limit
        terminal_reward = float(
            reward_config.get(
                "timeout_penalty",
                -50.0,
            )
        )

    # ============================================================
    # 10. v3 로그 형식 유지
    # ============================================================
    components = {
        "pursuit": float(pursuit_reward),
        "damage": float(damage_reward),
        "safety": float(safety_reward),
        "survival": 0.0,
    }

    # step, 근접 패널티, 종료 보상은 실제 학습 Reward에는 포함하지만
    # v3 training_log.csv에 새로운 열로 추가하지 않는다.
    total_reward = (
        step_reward
        + pursuit_reward
        + damage_reward
        + safety_reward
        + close_range_penalty
        + terminal_reward
    )

    return float(total_reward), components


__all__ = [
    "MY_REWARD_CONFIG",
    "compute_reward",
]