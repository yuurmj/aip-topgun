# SAC baseline 10회 리플레이 및 문제 분석

작성일: 2026-07-24

## 결론

현재 SAC는 10회 모두 45.25초에 저고도 추락했다. 10회 모두 같은 궤적이고 양측 피해와 WEZ 진입은 0이다.

다만 이 결과를 SAC 알고리즘 자체의 실패로 해석하면 안 된다. 평가한 것은 완전 학습된 최종 모델이 아니라 주최측 제공 artifact에 포함된 iteration 15 중간 bundle이며, 남아 있는 학습 로그도 iteration 0–3, 405 sampled steps, 완료 episode 1개뿐이다.

따라서 현재 SAC는 `주최측 중간 bundle reference baseline`으로는 확정하지만, `우리 팀이 정상 완료한 SAC training baseline`으로는 인정하지 않는다.

## 10회 결과

| 항목 | 결과 |
|---|---:|
| 실행 수 | 10 |
| 정상 로그 세트 | 10 |
| 승 | 0 |
| 패배 판정 | 0 |
| crash | 10 |
| timeout | 0 |
| 가한 피해 | 0% |
| 받은 피해 | 0% |
| WEZ 진입 | 매회 0 tick |
| 종료 시각 | 매회 45.25초 |
| 최단 거리 | 매회 23.08 m |
| 최종 거리 | 매회 24,793.48 m |
| 서로 다른 궤적 | 1 |

Ownship CSV 10개의 SHA-256은 모두 다음 값으로 동일하다.

`0134D78E99E78A0410A744412695C6C59876E5809FE5281767FE4ED728A87A13`

Target CSV 10개의 SHA-256도 모두 다음 값으로 동일하다.

`FDBBF15A09A59370166DDFD96D9A561AD892D39C8A8921E72DD93F9D28A1F47F`

## 문제 1 — 평가 bundle이 학습 완료 모델이 아니다

- bundle 메타데이터: iteration 15
- SAC 학습 시작 기준: `num_steps_sampled_before_learning_starts=1500`
- 남아 있는 training log: iteration 0–3
- training log sampled steps: 405
- training log 완료 episode: 1
- 계획된 runtime: 50 iteration

bundle과 남아 있는 training log의 시점도 서로 일치하지 않는다. 제공 artifact만으로 iteration 15의 실제 학습량과 성능을 완전히 검증할 수 없다.

우선 조치:

1. 새 팀 이름과 실험 태그로 50 iteration 전체를 처음부터 실행한다.
2. 최종 training log, 최종 bundle, experiment YAML을 같은 실행에서 보존한다.
3. SAC의 1,500-step warm-up 이후 actor/critic/alpha loss가 실제로 기록되는지 확인한다.
4. 중간 bundle이 아니라 최종 bundle로 replay를 생성한다.

## 문제 2 — crash가 Reward에서는 draw 수준으로 처리된다

환경 결과 분류는 `ownship altitude below min`을 `crash`로 기록한다. 그러나 기본 Reward는 체력으로만 승패를 구분하고, 체력이 양쪽 모두 1.0인 추락을 `draw_reward=-30`으로 처리한다.

SAC 학습 로그의 유일한 완료 episode도 다음과 같다.

- `crash_rate=1.0`
- `reward_mean=-31.5203`
- `ep_reward_safety=-1.0`
- `ep_reward_pursuit=+3.2197`
- `ep_wez_steps=0`

즉 지면 충돌이 일반 패배 `-100`보다 훨씬 약한 약 `-30` terminal로 처리됐다.

우선 조치:

1. `ownship altitude below min`과 ownship FDM fall을 `ownship_crash_reward=-120`으로 분리한다.
2. `target altitude below min`은 target crash이므로 win으로 분리한다.
3. generic `FDM Update Fail`은 어느 기체 문제인지 구분한 뒤 보상한다.
4. timeout, draw, crash를 하나의 체력 기반 else 분기로 묶지 않는다.

## 문제 3 — 저고도 안전 신호가 너무 늦다

기본 Reward는 고도 600 m 미만에서만 step당 -0.1을 준다. 대표 궤적은 2,000 m에 도달한 38.43초에 이미 Pitch -48.78°였고, 600 m에 도달한 44.05초에는 Pitch -49.12°였다.

안전 페널티 600 m는 경고선이 아니라 사실상 충돌 직전선이다.

우선 조치:

1. 2,000 m부터 연속적인 altitude risk를 계산한다.
2. 저고도이면서 Pitch가 음수면 dive penalty를 추가한다.
3. 과도한 bank가 저고도에서 유지되면 작은 추가 페널티를 준다.
4. 전투 shaping은 altitude safety gate를 곱해 위험한 급강하 중에는 pursuit 보상을 줄인다.

## 문제 4 — 비행 안정화 자체가 학습 과제다

같은 조건에서 추가 대조 실행을 수행했다.

| Ownship 제어 | 결과 | 해석 |
|---|---|---|
| `fixed` 무조작 | `ownship altitude below min` | 중립 입력만으로 trim된 수평 비행이 유지되지 않음 |
| `AIP_BASE_target.dll` BT | 60초 timeout, 양측 체력 1.0 | 조종 정책이 있으면 동일 FDM에서 생존 가능 |

따라서 시뮬레이터가 무조건 추락하는 것은 아니다. 하지만 SAC는 전투 기하를 배우기 전에 F-16 자세·고도 안정화부터 학습해야 한다.

권장 curriculum:

1. `flight_survival`: 60초 생존, 고도 유지, 급강하 억제
2. `pursuit`: 안전 고도를 유지하며 ATA와 거리 개선
3. `wez_entry`: 실제 WEZ 진입과 유지
4. `full_dogfight`: BT 상대와 승패 학습

## 문제 5 — merge 처리와 재획득도 실패한다

대표 리플레이는 8.23초에 상대와 23.08 m까지 접근했지만 ATA는 95.72°였다. 10초에는 ATA가 169°로 상대가 거의 완전히 뒤로 넘어갔다.

즉 SAC도 PPO와 마찬가지로 최초 merge 이후 overshoot와 재획득 실패가 있다. 다만 SAC에서는 이 문제를 다루기 전에 추락 방지가 우선이다.

## 문제 6 — 현재 10회는 독립 표본이 아니다

고정 초기 조건, 결정론적 추론, 동일 bundle, 동일 BT를 사용했으므로 10회가 byte-level로 동일하다.

현재 10회는 실패 재현성 자료로 보존한다. 정상 학습 후에는 고정 seed 10개와 초기 geometry를 기록한 paired evaluation으로 바꾼다.

## 수정 우선순위

1. SAC 50 iteration 전체 학습 및 최종 bundle 확보
2. crash/target crash/timeout terminal 보상 분리
3. 2,000 m부터 altitude risk와 dive penalty 적용
4. flight-survival curriculum 통과
5. pursuit와 WEZ 보상 추가
6. seed 10개로 재평가

확정한 다음 SAC 실험 이름:

`team01/sac_mlp_v2_flight_safety_seeded`
