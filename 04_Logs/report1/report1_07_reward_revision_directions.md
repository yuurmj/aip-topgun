# SAC/PPO baseline v1 Reward 수정 방향

## SAC v1 → v2

목표는 공격이 아니라 `60초 안정 비행`을 먼저 만드는 것이다.

### 1단계: terminal 의미 수정

- ownship 파괴: -100
- ownship 저고도 추락: -120
- target 파괴 또는 target 추락: +100
- timeout: -30
- 단순 draw: -10
- generic FDM failure는 어느 기체 failure인지 분리

### 2단계: 추락 이전의 연속 안전 신호

- 2,000 m부터 altitude risk 적용
- 저고도에서 음수 Pitch이면 dive penalty
- 60°를 넘는 bank에 작은 추가 페널티
- 위험 고도에서는 pursuit 보상에 safety gate 적용

### 3단계: curriculum

`flight_survival → pursuit → WEZ entry → full dogfight` 순서로 진행한다.

진급 기준 시작안:

- 최근 평가 crash rate ≤ 10%
- 평균 생존 시간 ≥ 55초
- 고도 1,500 m 미만 step 비율 ≤ 5%
- action saturation rate ≤ 15%

### 4단계: 학습 완료 조건

- 최소 50 iteration 정상 완료
- warm-up 1,500 steps 이후 actor/critic/alpha loss 존재
- 최종 bundle과 training log iteration 일치
- 최종 bundle로 seed 10개 평가

## PPO v1 → v2

목표는 `timeout 생존 정책`을 `실제 WEZ 진입과 재교전 정책`으로 바꾸는 것이다.

### 1단계: timeout 보상 버그 수정

현재 timeout은 `truncated=True`인데 기본 terminal 보상은 `terminated=True`만 처리한다.

- terminal 조건을 `terminated or truncated`로 처리
- `end_condition == "max time out"`에 -30 적용
- crash와 target crash도 별도 분기

### 2단계: proxy pursuit 축소

현재 pursuit는 최대 3 km, ATA 30° 범위에서 보상되지만 실제 WEZ는 152.4–914.4 m, ATA ±1°다.

- `pursuit_scale`: 0.30 → 0.10
- 실제 WEZ hold: +0.05/step
- damage differential 유지

### 3단계: progress와 overshoot

- `abs(ATA)` 감소를 보상
- 533.4 m WEZ 중앙까지의 거리 오차 감소를 보상
- ATA > 90°이면서 거리가 증가하면 overshoot penalty
- 3 km 밖으로 계속 벌어지면 escape penalty

이 항목은 이전 geometry가 필요하므로 reward contract에 previous distance/ATA 또는 환경 instance의 progress 값을 추가해야 한다.

### 4단계: 평가

- seed 10개를 SAC와 동일하게 사용
- win/loss/crash/timeout 외에 WEZ step, damage, overshoot count 기록
- 기존 고정 시나리오도 회귀 테스트로 한 번 유지

## 확정 이름

- SAC: `team01/sac_mlp_v2_flight_safety_seeded`
- PPO: `team01/ppo_mlp_v2_timeout_wez_seeded`

이름 규칙은 `{team}/{algorithm}_{network}_{version}_{main-change}_{evaluation}`로 고정한다.
