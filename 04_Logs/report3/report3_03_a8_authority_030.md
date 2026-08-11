# A8 — 30% 조종 권한 SAC 재학습

## 목적

A7 합격 bundle 15를 30% 권한으로 무학습 평가했을 때 WEZ 1.9초와 일부 피해는 달성했지만 저고도 추락했다. 이 첫 실패 단계에서 SAC를 재학습한다.

## 무학습 probe

- parent: `sac_mlp_v2_wez_a7_authority_015/bundle_000015`
- seed 1001: crash, 연속 WEZ 1.9초, 누적 피해 0.3225
- 종료: 46.2초, `ownship altitude below min`
- 판단: WEZ 최소 목표는 달성했지만 추락 방지와 격추 기준을 만족하지 못해 재학습 필요

## 재학습 설정

- 실험명: `team01/sac_mlp_v2_wez_a8_authority_030`
- control surface: 30%
- 시작거리: 950m
- ownship/target 속도: 350/200m/s
- target: 직선·수평
- reward: `student.sac_wez_hold_reward_v4`, 변경 없음
- 학습: 최대 50 iteration, bundle 매 5 iteration
- randomization: 없음

## 재학습 결과

50 iteration을 완료하고 bundle 5–50을 seed 1001로 평가했지만 합격 bundle은 없었다.

| bundle | 결과 | 연속 WEZ | 누적 피해 | 종료 |
|---:|---|---:|---:|---|
| 5 | crash | 1.9초 | 0.3225 | 46.2초 저고도 추락 |
| 10 | crash | 1.9초 | 0.3225 | 46.2초 저고도 추락 |
| 15 | crash | 2.7초 | 0.6853 | 58.8초 저고도 추락 |
| 20 | crash | 0.3초 | 0.0670 | 47.9초 저고도 추락 |
| 25 | crash | 1.8초 | 0.2898 | 39.6초 저고도 추락 |
| 30 | timeout | 0.4초 | 0.0090 | 60초 생존, 격추 실패 |
| 35 | crash | 0.4초 | 0.0122 | 41.2초 저고도 추락 |
| 40 | crash | 0.4초 | 0.0090 | 40.6초 저고도 추락 |
| 45 | crash | 2.7초 | 0.6768 | 45.2초 저고도 추락 |
| 50 | crash | 1.0초 | 0.0891 | 38.4초 저고도 추락 |

## 판단

15→30% 직접 점프는 실패했다. bundle 15와 45가 WEZ 2.7초까지 도달했으나 격추 전에 추락했으므로 채택하지 않고 20% 중간 curriculum을 삽입했다.
