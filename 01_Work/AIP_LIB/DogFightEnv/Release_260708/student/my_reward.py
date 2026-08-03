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
    # describe_reward()가 아래 키들을 요구하므로 삭제하면 안 됨
    "survival_bonus": 0.0,
    "damage_scale": 20.0,
    "pursuit_half_angle_deg": 60.0,

    # ============================================================
    # 시간 보상
    # ============================================================
    "step_penalty": -0.01,

    # ============================================================
    # 상대 추적 보상
    # ============================================================
    "pursuit_scale": 0.03,
    "pursuit_angle_deg": 60.0,
    "pursuit_range_m": 3000.0,

    # ============================================================
    # 공격 위치 보상
    # ============================================================
    # 2000ft = 약 609.6m
    # 2500ft = 약 762.0m
    # 3000ft = 약 914.4m
    "attack_min_range_m": 500.0,
    "attack_max_range_m": 920.0,
    "attack_ata_deg": 15.0,
    "attack_position_bonus": 0.03,

    # ============================================================
    # WEZ 및 피해량 보상
    # ============================================================
    "wez_hold_bonus": 0.05,
    "target_damage_scale": 20.0,
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
    "win_reward": 100.0,
    "loss_reward": -150.0,
    "crash_penalty": -150.0,
    "guard_fail_penalty": -50.0,
    "timeout_penalty": -30.0,
    "target_crash_reward": 0.0,
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
    PPO v4 커스텀 Reward.

    v3 training_log.csv에서 기록하던 Reward 항목:
        ep_reward_pursuit
        ep_reward_damage
        ep_reward_safety
        ep_reward_survival

    위 네 항목만 components로 반환한다.
    """

    # ============================================================
    # 1. 현재 거리와 ATA 계산
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
        reward_config.get("step_penalty", -0.01)
    )

    # ============================================================
    # 3. 상대 추적 보상
    # ============================================================
    pursuit_angle_deg = float(
        reward_config.get("pursuit_angle_deg", 60.0)
    )

    pursuit_range_m = float(
        reward_config.get("pursuit_range_m", 3000.0)
    )

    ata_factor = max(
        0.0,
        1.0 - ata_deg / pursuit_angle_deg,
    )

    range_factor = max(
        0.0,
        1.0 - distance_m / pursuit_range_m,
    )

    pursuit_reward = (
        float(reward_config.get("pursuit_scale", 0.03))
        * ata_factor
        * range_factor
    )

    # ============================================================
    # 4. 공격하기 좋은 위치 보상
    # ============================================================
    attack_min_range_m = float(
        reward_config.get("attack_min_range_m", 500.0)
    )

    attack_max_range_m = float(
        reward_config.get("attack_max_range_m", 920.0)
    )

    attack_ata_deg = float(
        reward_config.get("attack_ata_deg", 15.0)
    )

    if (
        ata_deg <= attack_ata_deg
        and attack_min_range_m
        <= distance_m
        <= attack_max_range_m
    ):
        pursuit_reward += float(
            reward_config.get(
                "attack_position_bonus",
                0.03,
            )
        )

    # ============================================================
    # 5. WEZ 진입 보상
    # ============================================================
    # single_agent_env.py에서는 target_damage > 0일 때
    # ownship이 target을 향한 WEZ 안에 들어간 것으로 처리한다.
    if float(target_damage) > 0.0:
        pursuit_reward += float(
            reward_config.get(
                "wez_hold_bonus",
                0.05,
            )
        )

    # ============================================================
    # 6. 피해량 보상
    # ============================================================
    damage_reward = (
        float(
            reward_config.get(
                "target_damage_scale",
                20.0,
            )
        )
        * float(target_damage)
        -
        float(
            reward_config.get(
                "ownship_damage_scale",
                25.0,
            )
        )
        * float(ownship_damage)
    )

    # ============================================================
    # 7. 저고도 안전 패널티
    # ============================================================
    safety_reward = 0.0

    altitude_m = float(
        ownship_state[StateIndex.ALT]
    )

    if altitude_m < float(
        reward_config.get("low_altitude_m", 600.0)
    ):
        safety_reward -= float(
            reward_config.get(
                "low_altitude_penalty",
                0.20,
            )
        )

    # ============================================================
    # 8. 잘못된 근접 패널티
    # ============================================================
    # safety에 합치면 ep_altitude_penalty_steps가 근접 상황까지
    # 저고도 상황으로 잘못 계산하므로 별도로 계산한다.
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

    # ============================================================
    # 9. 종료 보상
    # ============================================================
    terminal_reward = 0.0

    if terminated:
        if end_condition == "target destroyed":
            terminal_reward = float(
                reward_config.get(
                    "win_reward",
                    100.0,
                )
            )

        elif end_condition == "target altitude below min":
            # 상대가 스스로 추락한 경우
            terminal_reward = float(
                reward_config.get(
                    "target_crash_reward",
                    0.0,
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
        terminal_reward = float(
            reward_config.get(
                "timeout_penalty",
                -30.0,
            )
        )

    # ============================================================
    # 10. v3에서 저장하던 Reward 로그 구성
    # ============================================================
    components = {
        "pursuit": float(pursuit_reward),
        "damage": float(damage_reward),
        "safety": float(safety_reward),
        "survival": 0.0,
    }

    # step, close range, terminal 보상은 전체 Reward에는 포함하지만
    # v3 training_log.csv에 별도 열로 추가하지 않는다.
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