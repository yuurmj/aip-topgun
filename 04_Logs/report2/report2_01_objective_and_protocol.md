# 목표와 평가 프로토콜

## 목표

- 최소: 저장된 SAC bundle이 엄격 WEZ에 외부에서 진입해 연속 1초 이상 유지.
- 확장: target health를 감소시키고 격추.
- 최종: 무작위 full dogfight에서도 추락하지 않고 자체 사격으로 격추.

## 네 단계의 차이

| 단계 | 표적/환경 | 배우는 행동 | 통과 기준 |
|---|---|---|---|
| flight_survival | 쉬운 loiter 표적, 교전 보상 제거 | 60초 고도·자세 유지 | crash 0/10, 60초 생존 |
| pursuit | 직선/느린 표적부터 시작 | 안전하게 정렬하고 거리 축소 | 낮은 crash와 접근 거리 개선 |
| WEZ entry | 엄격 사거리·ATA 조건 | 외부 진입 후 연속 유지·damage | 연속 1초 이상 |
| full dogfight | BT/loiter 혼합, 무작위 기하 | 생존·추적·WEZ·격추를 동시에 | 자체 사격 win과 낮은 crash |

## 평가

- seed 1001–1010, deterministic `explore=False`.
- 엄격 WEZ: 152.4–914.4m, ownship ATA 절대값 1도 이하.
- 0.1초/RL step이므로 연속 10 step 이상을 1초 성공으로 판정.
- 내부 시작과 외부 진입을 분리하고, 누적 피해는 `1 - target_health`로 계산.
