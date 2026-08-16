# A7 — 15% 조종 권한 SAC 재학습

## 목적

학습 없는 sweep에서 WEZ 1.3초까지 진입했지만 격추하지 못하고 추락한 첫 실패 단계인 조종 권한 15%를 재학습한다.

## 설정

- 실험명: `team01/sac_mlp_v2_wez_a7_authority_015`
- parent: `sac_mlp_v2_wez_a6_qualified_kill`
- control surface: 15%
- 시작거리: 950m
- ownship/target 속도: 350/200m/s
- target: 직선·수평
- reward: `student.sac_wez_hold_reward_v4`, 변경 없음
- 학습: 50 iteration, bundle 매 5 iteration
- randomization: 없음

## 통과 기준

- WEZ 외부에서 시작해 실제 진입
- 연속 WEZ 1초 이상
- target damage 및 자체 사격 격추
- ownship crash 없음

## 상태

50 iteration 학습과 중간 bundle 평가를 완료했다. 최종 bundle 50이 아니라 bundle 15를 채택한다.

## 학습 결과

- 훈련 탐색 episode에서는 연속 WEZ 최고 1.7초까지 나왔지만 완료 episode는 모두 crash였고 win은 없었다.
- 학습량이 늘수록 항상 좋아지지 않았으며, 후반 bundle 대부분은 정렬과 안전성이 퇴행했다.
- 따라서 15·20·25·30·35·40·45·50 bundle을 deterministic seed 1001로 비교했다.

| bundle | 결과 | 연속 WEZ | 누적 피해 | 판단 |
|---:|---|---:|---:|---|
| 15 | win | 3.3초 | 1.0074 | 채택 |
| 20 | timeout | 1.4초 | 0.2079 | 격추 실패 |
| 25 | crash | 0.7초 | 0.0355 | 탈락 |
| 30 | crash | 2.2초 | 0.4619 | 탈락 |
| 35 | crash | 2.2초 | 0.4451 | 탈락 |
| 40 | crash | 1.3초 | 0.1213 | 탈락 |
| 45 | crash | 1.5초 | 0.2162 | 탈락 |
| 50 | crash | 2.2초 | 0.5017 | 탈락 |

## Bundle 15 정식 평가

- seed 1001–1010: win 10/10, crash 0/10.
- started in WEZ 0/10, 외부 WEZ 진입 10/10.
- WEZ 1초 성공 10/10, 연속 WEZ 3.3초.
- 3.5초에 자체 사격으로 target destroyed, 평균 누적 피해 1.0074.
- randomization이 없어 unique trajectory hash는 1개다.

## 최종 판단

- 15% 조종 권한 단계는 통과했다.
- 채택 bundle: `artifacts/models/team01/sac_mlp_v2_wez_a7_authority_015/bundle_000015`.
- 최종 bundle 50은 사용하지 않는다.
- 다음 단계는 A7 bundle 15를 30% 권한으로 학습 없이 먼저 평가하는 것이며, 아직 실행하지 않았다.
