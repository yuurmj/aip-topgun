# 다음 실험 방향

## 결론

다음 실험은 full dogfight 재학습이 아니다. A6의 쉬운 격추를 출발점으로 조종 권한, 초기 기하, 표적 난이도를 한 축씩 복원해야 한다. 직전 full 결과의 crash 6/10과 WEZ 1초 0/10은 A6에서 full 환경으로 한 번에 바꾼 분포 차이가 너무 컸다는 뜻이다.

출발 bundle은 full 기준선이 아니라 보존된 `sac_mlp_v2_wez_a6_qualified_kill`을 사용한다. full bundle은 안전성과 WEZ 능력이 퇴행했으므로 다음 parent로 사용하지 않는다.

## 가장 먼저 할 일: 학습 없는 조종 권한 sweep

A6의 기하를 그대로 두고 evaluator의 `--control-surface-scale`만 바꿔 아래 순서로 seed 1001을 평가한다.

`0.05 → 0.15 → 0.30 → 0.60 → 1.00`

- 시작 거리: 915m가 아니라 수치 경계 여유를 둔 950m.
- ownship/target 속도: 350/200m/s.
- target: 직선·수평 비행.
- 엄격 WEZ 정의는 변경하지 않는다.
- 가장 높은 성공 권한을 찾은 뒤 그 후보만 seed 1001–1010으로 확인한다.

full authority에서도 고정 기하 격추가 유지되면 권한 재학습은 생략하고 기하 확장으로 이동한다. 실패하면 마지막 성공 권한의 바로 다음 단계에서 SAC 학습을 시작한다.

## Phase 1: 조종 권한 복원

권장 첫 학습명은 `team01/sac_mlp_v2_wez_a7_authority_005`다.

| 항목 | 시작값 |
|---|---|
| parent | `sac_mlp_v2_wez_a6_qualified_kill` |
| control surface | sweep 결과에 따라 5% 또는 첫 실패 단계 |
| 거리 | 950m |
| 속도 | ownship 350, target 200m/s |
| target | straight/level |
| iterations | 50, bundle 매 5 iteration |
| reward | v4 유지, strict WEZ와 safety 값 변경 금지 |

한 단계가 통과하면 `5% → 15% → 30% → 60% → 100%` 순서로 올린다. 매 단계는 이전 통과 bundle에서 시작하되 replay buffer는 새로 만든다.

## Phase 2: 초기 기하 일반화

100% 조종 권한에서 고정 기하를 통과한 뒤에만 다음 항목을 하나씩 늘린다.

1. 시작 거리: 950m → 1.2km → 1.5km → 3km.
2. heading jitter: ±0° → ±1° → ±3° → ±5°.
3. 위치 jitter: 0m → 50m → 150m → 300m.
4. 속도 차: 150m/s → 100m/s → 50m/s → 혼합 범위.

두 개 이상을 동시에 바꾸지 않는다. 이 단계부터는 seed별 trajectory hash가 여러 개여야 하며, hash 1개인 10회 반복은 일반화 성공으로 세지 않는다.

## Phase 3: 표적 난이도 확장

기하 일반화 후 표적을 다음 순서로 바꾼다.

`straight → shallow loiter(10–15° bank) → loiter(30° bank) → 제한 BT → full BT`

BT로 이동하기 전에는 pursuit 지표를 별도 gate로 둔다.

- 평균 최저거리 1,000m 이하.
- 최종 ATA 중앙값 30도 이하.
- 외부 WEZ 진입 8/10 이상.
- 자체 damage 발생 8/10 이상.

## 공통 통과 기준

| 지표 | 통과 기준 |
|---|---:|
| ownship crash | 1/10 이하 |
| started in WEZ | 0/10 |
| 외부 WEZ 진입 | 8/10 이상 |
| 연속 WEZ 1초 | 8/10 이상 |
| target damage | 8/10 이상 |
| 자체 사격 격추 | acquisition 5/10 이상, full 진입 전 8/10 이상 |

표적 추락으로 얻은 win은 자체 사격 격추에 포함하지 않는다.

## 중단 및 bundle 선택 규칙

- 최종 bundle만 보지 않고 5-iteration 중간 bundle을 seed 1001로 선별한다.
- 통과 후보만 seed 10개로 승격한다.
- 최근 완료 episode 3개가 모두 crash이고 WEZ/damage가 0이면 해당 설정을 조기 종료한다.
- 같은 단계에서 reward scale을 계속 바꾸지 말고 최대 2개 시도 후 이전 통과 단계로 돌아간다.
- 새 단계가 실패해도 survival과 A6 qualified bundle은 덮어쓰지 않는다.

## Reward 변경 우선순위

다음 실험의 첫 시도에서는 reward를 바꾸지 않는다. 환경 난이도와 reward를 동시에 바꾸면 원인 분석이 불가능해진다. 동일 reward로 권한·기하 단계가 두 번 실패했을 때만 아래 순서로 검토한다.

1. fine ATA penalty의 범위와 크기 재조정.
2. positive closure가 끊길 때 즉시 음수 신호 강화.
3. control surface 변화량 또는 과도 action penalty 추가.
4. 안전 고도에서만 pursuit/WEZ shaping 활성화하는 gate 재확인.

## SAC와 PPO 사용 시점

- 위 curriculum 탐색과 연속 조종 학습은 replay를 재사용하는 SAC로 계속한다.
- SAC가 100% 권한, 다양한 기하, loiter 단계까지 통과한 뒤 PPO를 동일 계약의 비교 기준선 또는 full-BT 안정화 후보로 실행한다.
- SAC 가중치를 PPO에 바로 넣는 방식은 구조 호환성이 확인되지 않았으므로 사용하지 않는다. 필요하면 SAC trajectory를 이용한 별도 distillation/behavior cloning 절차를 설계한다.

## 다음 한 번의 권장 작업 범위

다음 세션에서는 먼저 authority sweep을 끝내고, 필요한 경우 A7 한 단계만 학습한다. 한 번에 full BT까지 다시 돌리지 않는다. 목표는 “가장 높은 조종 권한에서 외부 WEZ 1초와 격추를 유지하는 첫 bundle”을 만드는 것이다.
