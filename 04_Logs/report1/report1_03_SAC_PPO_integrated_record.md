# SAC/PPO baseline 통합 기록표

작성일: 2026-07-24

## 요약

| 항목 | SAC baseline | PPO baseline |
|---|---|---|
| 평가 모델 | `test_mlp/sac_mlp_v1/bundle_000015` | `team01/ppo_mlp_v1` 최종 bundle |
| baseline 성격 | 주최측 제공 중간 bundle reference | 팀 학습 완료 technical baseline |
| 학습 근거 | metadata iter 15, 보존 log iter 0–3/405 steps/1 episode | 50 iter/51,200 steps/94 episodes |
| 계획 iteration | 50 | 50 |
| 관측 | tactical16 | tactical16 |
| 행동 | 연속 4차원 | 연속 4차원 |
| 상대 | `AIP_BASE_target.dll` BT | `AIP_BASE_target.dll` BT |
| 평가 횟수 | 10 | 10 |
| 승 | 0 | 0 |
| 패 | 0 | 0 |
| crash | 10 | 0 |
| timeout | 0 | 10 |
| WEZ 진입 | 0 tick | 0 tick |
| 가한 피해 | 0% | 0% |
| 받은 피해 | 0% | 0% |
| 평균 episode 시간 | 45.25초 | 59.98초 |
| 최단 거리 | 23.08 m | 136.78 m |
| 최단 거리 시각 | 8.23초 | 8.27초 |
| 최단 거리 ATA | 95.72° | 109.60° |
| 평균 거리 | 10,331.83 m | 14,411.76 m |
| 최종 거리 | 24,793.48 m | 32,297.99 m |
| 최종 ownship 고도 | 299.67 m | 6,258.69 m |
| 서로 다른 궤적 | 1/10 | 1/10 |
| 대표 실패 | overshoot 후 급강하 추락 | overshoot 후 재교전 실패·이탈 |
| 현재 판정 | 전투 이전에 비행 안정성 실패 | 생존하지만 공격·재획득 실패 |

## 공통 문제

1. 최초 head-on 접근 뒤 merge 처리에 실패한다.
2. 실제 WEZ에 한 틱도 들어가지 못한다.
3. 양측 피해가 0이고 공격 행동이 형성되지 않았다.
4. 고정 초기 조건과 결정론적 추론 때문에 10회가 모두 동일하다.
5. 실제 WEZ보다 넓은 pursuit shaping이 proxy objective가 될 수 있다.
6. 종료 사유와 terminal reward 분류가 일치하지 않는다.

## 알고리즘별 우선순위

| 우선순위 | SAC | PPO |
|---:|---|---|
| 1 | 최종 bundle이 생기도록 전체 학습 | timeout terminal 보상 누락 수정 |
| 2 | crash를 -120으로 명시 | overshoot·이탈·재획득 shaping |
| 3 | 2,000 m부터 안전 shaping | 실제 WEZ 진입·유지 보상 |
| 4 | flight-survival curriculum | progress reward용 이전 geometry 제공 |
| 5 | 생존 후 pursuit/WEZ 학습 | seed/scenario 기반 평가 |

## 다음 평가 프로토콜

1. 알고리즘별 최종 bundle만 사용한다.
2. 동일한 seed 10개를 SAC/PPO에 paired 적용한다.
3. 초기 거리·ATA·AA·고도·속도를 저장한다.
4. 결과에 win/loss/crash/timeout, damage, WEZ step, min/final distance를 남긴다.
5. 궤적 hash가 모두 같은지 품질 검사한다.

## 확정 실험 이름

| 알고리즘 | 다음 실험 이름 | 핵심 변경 |
|---|---|---|
| SAC | `team01/sac_mlp_v2_flight_safety_seeded` | 완전 학습, crash/고도 안전, seeded 평가 |
| PPO | `team01/ppo_mlp_v2_timeout_wez_seeded` | timeout 수정, WEZ/progress, seeded 평가 |
