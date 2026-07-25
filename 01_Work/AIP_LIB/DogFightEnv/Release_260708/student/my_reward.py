# -*- coding: utf-8 -*-
"""PPO attack-geometry reward for the student dogfight environment.

Goal of this reward:
  1. Keep the aircraft alive.
  2. Point the nose toward the target.
  3. Close the distance without simply flying past the target.
  4. Move toward the target's rear aspect.
  5. Form a stable firing geometry before the exact WEZ is reached.
  6. Apply an explicit penalty when an episode ends by timeout.

Required contract:
  - MY_REWARD_CONFIG must be a dict.
  - compute_reward(...) must return (total_reward: float, components: dict).
  - Each component is logged as ep_reward_<component_name>.
"""
from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

for path in (ROOT, SRC):
    path_text = str(path)

    if path_text not in sys.path:
        sys.path.insert(0, path_text)


from dogfight.sim.state_schema import StateIndex


MY_REWARD_CONFIG = {
    # ============================================================
    # 1. 기본 시간 및 안전 관련 보상
    # ============================================================

    # 한 스텝이 지날 때마다 받는 패널티
    # 공격하지 않고 오래 비행하는 정책을 방지한다.
    "step_penalty": -0.01,

    # 고도 600m 미만으로 내려갔을 때의 패널티
    "low_altitude_penalty": 0.20,

    # ============================================================
    # 2. 상대를 향해 기수를 정렬하고 접근하도록 만드는 보상
    # ============================================================

    # ATA가 작을수록 받는 기수 정렬 보상
    "heading_scale": 0.03,

    # 상대를 바라보면서 가까워졌을 때 받는 추적 보상
    "pursuit_scale": 0.06,

    # 추적 보상이 활성화되는 최대 거리
    # 기존 3000m보다 넓혀서 초기 상태에서도 학습 신호를 받게 한다.
    "pursuit_range_m": 8000.0,

    # ============================================================
    # 3. 상대 후방 및 공격 위치 형성 보상
    # ============================================================

    # 상대 후방에 위치할수록 받는 보상
    "tail_scale": 0.04,

    # 기수, 후방 위치, 공격 거리가 동시에 좋아졌을 때 받는 보상
    "attack_geometry_scale": 0.12,

    # Soft WEZ에서 사용하는 ATA 기준
    # ATA 0도에서 최대, 45도 이상이면 0점
    "soft_ata_limit_deg": 45.0,

    # Soft WEZ에서 사용하는 AA 기준
    # AA 0도는 상대 후방, 90도 이상이면 0점
    "soft_aa_limit_deg": 90.0,

    # 공격 거리 점수가 시작되는 거리
    "attack_range_min_m": 300.0,

    # 공격 거리 점수가 가장 높은 거리
    "attack_range_best_m": 700.0,

    # 공격 거리 점수가 끝나는 거리
    "attack_range_max_m": 1500.0,

    # ============================================================
    # 4. 실제 WEZ 및 지나침 방지
    # ============================================================

    # 실제 WEZ에 진입했을 때 매 스텝 받는 보상
    "wez_bonus": 0.30,

    # 너무 가까이 지나쳤다고 판단할 거리
    "too_close_distance_m": 250.0,

    # 가까운 거리에서 ATA가 이 값보다 크면 지나친 것으로 판단
    "too_close_ata_deg": 45.0,

    # 가까이 지나친 행동에 대한 패널티
    "too_close_penalty": 0.25,

    # ============================================================
    # 5. 피해량 및 종료 보상
    # ============================================================

    # 상대에게 준 피해량과 내가 받은 피해량의 차이
    "damage_scale": 20.0,

    # 승리 보상
    "win_reward": 100.0,

    # 패배 및 추락 보상
    "loss_reward": -100.0,

    # 시간 초과 패널티
    "timeout_penalty": -30.0,

    # 기타 무승부 종료
    "draw_reward": -20.0,

    # Two-circle geometry guard 실패
    "guard_fail_penalty": -50.0,
}


def _clamp01(value: float) -> float:
    """값을 0~1 범위로 제한한다."""
    return max(0.0, min(1.0, float(value)))


def _linear_angle_score(
    angle_deg: float,
    zero_score_angle_deg: float,
) -> float:
    """각도 기반 점수를 계산한다.

    angle_deg가 0도이면 1점이다.
    zero_score_angle_deg 이상이면 0점이다.
    그 사이는 선형적으로 감소한다.
    """
    limit = max(float(zero_score_angle_deg), 1.0e-6)

    return _clamp01(
        1.0 - abs(float(angle_deg)) / limit
    )


def _attack_range_score(
    distance_m: float,
    min_range_m: float,
    best_range_m: float,
    max_range_m: float,
) -> float:
    """공격하기 좋은 거리 점수를 계산한다.

    점수 형태:

        min 거리        best 거리        max 거리
           0  ─────────── 1 ─────────── 0

    너무 멀어도 0점이고 너무 가까워도 0점이다.
    """
    distance = float(distance_m)
    minimum = float(min_range_m)
    best = float(best_range_m)
    maximum = float(max_range_m)

    if not minimum < best < maximum:
        raise ValueError(
            "attack range must satisfy "
            "attack_range_min_m "
            "< attack_range_best_m "
            "< attack_range_max_m"
        )

    # 최소 거리 이하 또는 최대 거리 이상
    if distance <= minimum or distance >= maximum:
        return 0.0

    # 최소 거리에서 최적 거리까지 점수 증가
    if distance <= best:
        return _clamp01(
            (distance - minimum)
            / (best - minimum)
        )

    # 최적 거리에서 최대 거리까지 점수 감소
    return _clamp01(
        (maximum - distance)
        / (maximum - best)
    )


def _terminal_reward(
    ownship_state,
    target_state,
    reward_config: dict,
    terminated: bool,
    truncated: bool,
    end_condition: str,
) -> float:
    """에피소드 종료 시 한 번 적용되는 보상을 계산한다."""
    if not (terminated or truncated):
        return 0.0

    ownship_health = float(
        ownship_state[StateIndex.HEALTH]
    )

    target_health = float(
        target_state[StateIndex.HEALTH]
    )

    # ============================================================
    # 1. 체력을 기준으로 승패 확인
    # ============================================================

    if target_health <= 0.0 < ownship_health:
        return float(
            reward_config.get(
                "win_reward",
                100.0,
            )
        )

    if ownship_health <= 0.0 < target_health:
        return float(
            reward_config.get(
                "loss_reward",
                -100.0,
            )
        )

    # ============================================================
    # 2. 추락 또는 시뮬레이터 오류
    # ============================================================

    if end_condition in {
        "ownship altitude below min",
        "FDM Update Fail",
        "Ownship FDM output Fall",
    }:
        return float(
            reward_config.get(
                "loss_reward",
                -100.0,
            )
        )

    # ============================================================
    # 3. Two-circle guard 실패
    # ============================================================

    if end_condition == "two circle headon guard fail":
        return float(
            reward_config.get(
                "guard_fail_penalty",
                -50.0,
            )
        )

    # ============================================================
    # 4. 시간 초과
    # ============================================================

    if (
        truncated
        or end_condition
        in {
            "max time out",
            "episode step limit",
        }
    ):
        return float(
            reward_config.get(
                "timeout_penalty",
                -30.0,
            )
        )

    # ============================================================
    # 5. 기타 종료
    # ============================================================

    return float(
        reward_config.get(
            "draw_reward",
            -20.0,
        )
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
    """PPO 공격 위치 형성을 위한 보상을 계산한다."""

    # ============================================================
    # 1. 현재 교전 기하 정보 계산
    # ============================================================

    distance = float(
        geo_info._get_distance(
            ownship_state,
            target_state,
        )
    )

    ata = abs(
        float(
            geo_info._get_antenna_train_angle(
                ownship_state,
                target_state,
                False,
            )
        )
    )

    aa = abs(
        float(
            geo_info._get_aspect_angle(
                ownship_state,
                target_state,
                False,
            )
        )
    )

    components: dict[str, float] = {}

    # ============================================================
    # 2. 시간 패널티
    # ============================================================

    components["step"] = float(
        reward_config.get(
            "step_penalty",
            -0.01,
        )
    )

    # ============================================================
    # 3. 기수 정렬 보상
    # ============================================================
    #
    # ATA:
    #   0도   = 기수가 상대를 정확히 향함
    #   90도  = 상대가 기체 옆에 있음
    #   180도 = 상대가 기체 뒤에 있음
    #
    # 기존 보상은 ATA 30도 이상에서 보상이 0이었지만,
    # 여기서는 ATA 180도까지 넓은 기울기를 제공한다.
    #
    # 상대를 지나친 뒤에도 다시 선회할 학습 신호를 주기 위함이다.

    broad_heading_score = _linear_angle_score(
        ata,
        180.0,
    )

    components["heading"] = (
        float(
            reward_config.get(
                "heading_scale",
                0.03,
            )
        )
        * broad_heading_score
    )

    # ============================================================
    # 4. 접근 및 추적 보상
    # ============================================================

    pursuit_range_m = max(
        float(
            reward_config.get(
                "pursuit_range_m",
                8000.0,
            )
        ),
        1.0,
    )

    broad_range_score = _clamp01(
        1.0 - distance / pursuit_range_m
    )

    components["pursuit"] = (
        float(
            reward_config.get(
                "pursuit_scale",
                0.06,
            )
        )
        * broad_heading_score
        * broad_range_score
    )

    # ============================================================
    # 5. 상대 후방 위치 보상
    # ============================================================
    #
    # 이 소스의 GeoMathUtil 기준:
    #
    # AA 0도:
    #   내가 상대 뒤에 있음
    #
    # AA 90도:
    #   내가 상대 옆에 있음
    #
    # AA 180도:
    #   내가 상대 정면에 있음

    broad_tail_score = _linear_angle_score(
        aa,
        180.0,
    )

    components["tail"] = (
        float(
            reward_config.get(
                "tail_scale",
                0.04,
            )
        )
        * broad_heading_score
        * broad_tail_score
        * broad_range_score
    )

    # ============================================================
    # 6. Soft WEZ 공격 위치 보상
    # ============================================================
    #
    # 실제 WEZ는 ATA 1도 이내라서 초반 PPO가 진입하기 어렵다.
    #
    # 그래서 실제 WEZ에 들어가기 전에도 다음 조건이 좋아지면
    # 보상을 주는 Soft WEZ를 사용한다.
    #
    #   - ATA 감소
    #   - AA 감소
    #   - 300~1500m 거리 진입

    soft_ata_score = _linear_angle_score(
        ata,
        float(
            reward_config.get(
                "soft_ata_limit_deg",
                45.0,
            )
        ),
    )

    soft_aa_score = _linear_angle_score(
        aa,
        float(
            reward_config.get(
                "soft_aa_limit_deg",
                90.0,
            )
        ),
    )

    range_score = _attack_range_score(
        distance_m=distance,
        min_range_m=float(
            reward_config.get(
                "attack_range_min_m",
                300.0,
            )
        ),
        best_range_m=float(
            reward_config.get(
                "attack_range_best_m",
                700.0,
            )
        ),
        max_range_m=float(
            reward_config.get(
                "attack_range_max_m",
                1500.0,
            )
        ),
    )

    components["attack_geometry"] = (
        float(
            reward_config.get(
                "attack_geometry_scale",
                0.12,
            )
        )
        * soft_ata_score
        * soft_aa_score
        * range_score
    )

    # ============================================================
    # 7. 실제 WEZ 진입 보상
    # ============================================================
    #
    # single_agent_env.py의 실제 피해 판정과 동일한 조건을 사용한다.

    half_wez_angle_deg = (
        float(wez_config["angle_deg"])
        / 2.0
    )

    in_wez = (
        float(wez_config["min_range_m"])
        <= distance
        <= float(wez_config["max_range_m"])
        and ata <= half_wez_angle_deg
    )

    components["wez"] = (
        float(
            reward_config.get(
                "wez_bonus",
                0.30,
            )
        )
        if in_wez
        else 0.0
    )

    # ============================================================
    # 8. 너무 가까이 지나치는 행동 패널티
    # ============================================================
    #
    # 상대와 250m 이내인데 ATA가 45도보다 크다면,
    # 공격 위치를 만들지 못하고 옆으로 지나친 것으로 판단한다.

    too_close_distance_m = max(
        float(
            reward_config.get(
                "too_close_distance_m",
                250.0,
            )
        ),
        1.0,
    )

    too_close_ata_deg = float(
        reward_config.get(
            "too_close_ata_deg",
            45.0,
        )
    )

    close_pass_score = 0.0

    if (
        distance < too_close_distance_m
        and ata > too_close_ata_deg
    ):
        close_pass_score = _clamp01(
            1.0
            - distance
            / too_close_distance_m
        )

    components["too_close"] = (
        -float(
            reward_config.get(
                "too_close_penalty",
                0.25,
            )
        )
        * close_pass_score
    )

    # ============================================================
    # 9. 피해량 보상
    # ============================================================
    #
    # 상대 피해량이 더 크면 양수,
    # 내 피해량이 더 크면 음수가 된다.

    components["damage"] = (
        float(
            reward_config.get(
                "damage_scale",
                20.0,
            )
        )
        * (
            float(target_damage)
            - float(ownship_damage)
        )
    )

    # ============================================================
    # 10. 저고도 안전 패널티
    # ============================================================

    components["safety"] = 0.0

    if float(
        ownship_state[StateIndex.ALT]
    ) < 600.0:
        components["safety"] = -float(
            reward_config.get(
                "low_altitude_penalty",
                0.20,
            )
        )

    # ============================================================
    # 11. 종료 보상
    # ============================================================

    components["terminal"] = _terminal_reward(
        ownship_state=ownship_state,
        target_state=target_state,
        reward_config=reward_config,
        terminated=terminated,
        truncated=truncated,
        end_condition=end_condition,
    )

    total_reward = float(
        sum(components.values())
    )

    return total_reward, components


__all__ = [
    "MY_REWARD_CONFIG",
    "compute_reward",
]