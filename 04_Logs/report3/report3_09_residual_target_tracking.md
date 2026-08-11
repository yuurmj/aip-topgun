# Residual SAC 표적 추적 확대 — A18~A28

## 목표와 판정

- 목표는 초기 기하 A16을 보존하면서 움직이는 표적에 대한 자체 격추 범위를 넓히는 것이다.
- 합격 조건은 seed 1001~1010에서 `ownship_kills 10/10`, 외부 진입 후 연속 WEZ 1초 이상 `10/10`, ownship crash `0/10`이다.
- raw win 중 `target altitude below min`은 격추로 인정하지 않는다.
- 난이도는 shallow loiter 2° → 5° → 10° → loiter 30° 순서로 올렸다.

## 구현 판단

- 기존 안전 정책의 조종을 완전히 덮지 않도록 SAC 출력을 작은 residual action으로 적용했다.
- 움직이는 표적 정렬용 reward `sac_moving_target_reward_v5`를 추가하고 기존 추락·저고도·dive·과도 bank 안전 항목은 유지했다.
- 평가기는 target bank/pitch, target BT surface, residual scale/baseline override와 `ownship_kills`/`target_crash_wins` 분리를 지원하도록 확장했다.
- SAC 기본 learning-start가 1,500 step임을 확인해 실험 YAML에서 `learning_starts`를 명시할 수 있게 연결했다.

## 단계별 결과

| 시도 | 조건 | 핵심 결과 | 판단 |
|---|---|---|---|
| A18 | 고정 기하, bank 2°, residual 10%, 20 iter | geometry 복원 평가에서 격추 9/10, WEZ 10/10, crash 0 | 개선됐지만 미통과 |
| A19 | A18 bundle15에서 초기 기하 범위 복원, bank 2°, 20 iter | bundle20: 격추 10/10, WEZ 10/10, crash 0, 격추 3.3~4.5초 | 채택 |
| A20 | bank 5°, residual 10%, 20 iter | bundle5~15 피해 0.6007, bundle20 0.4514 | 퇴행, 탈락 |
| A21 | bank 5°, residual 20%, actor LR 1e-5, 30 iter | 대표 후보 피해 0.6918 이하, 후반 탐색 중 crash | 탈락 |
| A22 | residual/throttle 무학습 탐색 | 최고 피해 0.7089, WEZ 2.8초 | 격추 미달 |
| A23 | bank 3°, 기본 learning-start, 20 iter | 무학습 피해 0.8452, bundle20 0.8451 | 학습 효과 없음 |
| A24 | bank 3°, learning-start 256, 20 iter | bundle5/10/20 피해 0.7442/0.7983/0.6166 | update 증가 시 퇴행 |
| A25 | A19 가중치, bank 5°, roll baseline +0.02 | 격추 10/10, WEZ 10/10, crash 0, 평균 격추 3.7초 | 채택·self-contained 보존 |
| A26 | A25 가중치, bank 10°, roll baseline +0.04 | 격추 10/10, WEZ 10/10, crash 0, 평균 격추 3.7초 | 현재 최종 채택 |
| A27 | bank 30°, coordinated baseline 및 20 iter | 무학습 최고 피해 0.8097; 학습 bundle20 피해 0.1167 | 학습 후보 탈락 |
| A28 | bank 30°, pitch 미세 보정 −0.002 | 피해 0.8848, WEZ 3.1초, crash 0, 격추 0/1 | 현재 최고이나 미통과 |

## 채택 bundle

- 2° 학습 parent: `artifacts/models/team01/sac_mlp_v2_wez_a19_residual_shallow_geometry/bundle_000020`
- 5° self-contained: `artifacts/models/team01/sac_mlp_v2_wez_a25_shallow5_qualified/bundle_000001`
- 현재 최종 10°: `artifacts/models/team01/sac_mlp_v2_wez_a26_shallow10_qualified/bundle_000001`
- A26은 A25의 정책 가중치를 그대로 복제하고 target bank 10°와 residual baseline `[0.04, -0.0003, -0.001, -0.04]`만 metadata에 내장했다. 가중치 SHA-256은 원본과 같은 `1868773b...160ebf`이며 learner update는 0회다.
- A26 override 없는 seed 1001 재검증은 4.5초 자체 격추, 연속 WEZ 3.4초, crash 0이었다.

## 핵심 판단

- 필수 목표인 외부 WEZ 연속 1초 이상은 모든 정식 단계에서 유지했고, 5°와 10°는 10-seed 자체 격추까지 통과했다.
- 현재 첫 실패 단계는 30° loiter다. 비행 안전 문제는 아니며, 짧은 WEZ 체류 뒤 표적 선회를 계속 따라가지 못하는 추적 문제다.
- 현재 reward에서 SAC update를 늘리면 stochastic 학습 episode와 달리 deterministic mean policy가 반복적으로 퇴행했다. 같은 설정의 장기 학습은 효율적이지 않다.
- 30°에서 static coordinated-turn prior는 피해를 88.5%까지 올렸지만 격추 합격에는 부족했다.
- seed별 독립 평가가 안정적이며, 여러 Ray evaluator를 동시에 띄우면 초기화 경합이 발생했다. 후보 선별은 독립 10초 screen 후 합격 후보만 10-seed 정식 평가한다.

## 다음 실험 계획

- A26을 덮어쓰지 않고 30° 첫 실패점에서 다시 시작한다.
- 다음 학습은 static baseline 추가 탐색보다 표적 bank/LOS rate에 따라 달라지는 state-conditioned residual prior를 우선 구현한다.
- deterministic policy 퇴행을 막기 위해 평가 score를 기준으로 후보를 즉시 보존하고, replay 시작점·actor update 수를 줄인 짧은 iteration만 수행한다.
- 30° 자체 격추 10/10을 통과한 뒤 제한 BT 30%를 재평가하고, 통과 시에만 loiter → 제한 BT → full BT 순서로 진행한다.
- 제한/full BT에서도 공식 성공은 `ownship_kills`만 사용하고 `target_crash_wins`는 별도 실패 항목으로 유지한다.
