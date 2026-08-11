# A10 — 100% 조종 권한 actor 보정 및 SAC 미세학습

## 판단

- A8 30% 직접 학습과 A9 20% 중간 학습 모두 합격 bundle을 만들지 못했다.
- `gamma=0.99`에서는 0.1초 step 기준 40–60초 뒤 추락 penalty의 영향이 매우 작고, 높은 권한에서 기존 actor 출력이 과대 조종으로 이어졌다.
- A7 15% 합격 actor의 roll/pitch/rudder 평균 출력을 `0.15/1.00`로 보정하자 100% 권한 probe에서 A7과 같은 궤적을 재현했다.

## 100% warm-start probe

- 결과: win, crash 없음
- 외부 WEZ 진입 및 연속 WEZ 3.3초
- 누적 target 피해 1.0074, 3.5초 자체 사격 격추
- 의미: 권한을 제한한 것이 아니라 100% action space 안에서 정책이 더 작은 surface 명령을 선택하도록 초기화한 것

## 재학습 설정

- 실험명: `team01/sac_mlp_v2_wez_a10_authority_100_adapted`
- control surface: 100%
- init: A7 bundle 15의 surface actor 평균·탐색 표준편차를 15% 비율로 보정
- `gamma`: 0.99 → 0.999
- learning rate: 3e-4 → 1e-4
- altitude hold: 0.05 → 0.20
- level pitch: 0.02 → 0.20
- level roll: 0.005 → 0.05
- 나머지 terminal·damage·WEZ 설정은 유지
- 20 iteration, bundle 매 5 iteration

## A10 결과

| bundle | 결과 | 연속 WEZ | 누적 피해 | 판단 |
|---:|---|---:|---:|---|
| 5 | win | 3.3초 | 1.0074 | 학습 전 보존점 |
| 10 | win | 3.3초 | 1.0074 | 학습 전 보존점 |
| 15 | crash | 2.5초 | 0.6357 | 업데이트 후 퇴행 |
| 20 | crash | 0.2초 | 0.0028 | 업데이트 후 퇴행 |

`lr=1e-4` 단일 값으로 actor까지 업데이트하면 100% 성공 초기화가 무너졌다. 다음 A11에서는 actor learning rate만 `1e-6`으로 낮추고 critic `3e-4`, alpha `1e-5`로 분리해 보수적으로 미세학습한다.
