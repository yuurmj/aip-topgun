# Terminal과 Reward 변경

## Terminal 표

| 조건 | 값 | 비고 |
|---|---:|---|
| ownship crash/FDM failure | -120 | `ownship altitude below min` 포함 |
| ownship destroyed | -100 | health 기반 loss |
| target crash/FDM failure | +100 | 자체 사격과 구분해 기록 |
| target destroyed | +100 | health 기반 win |
| timeout/step limit | -30 | survival 단계만 목표 정렬을 위해 0 |
| draw/기타 동시 종료 | -10 | crash와 분리 |
| geometry guard fail | -100 | 잘못된 초기/기하 종료 |

## 연속 reward

- 2,000m 이하 고도 risk: `-0.10 * risk²/step`.
- 저고도 pitch down dive: 최대 `-0.05 * risk/step`.
- 60도 초과 bank: 작은 penalty, 저고도 risk와 곱함.
- 위험 고도에서는 pursuit/closure/alignment/range를 `1-risk`로 줄임.
- A2 survival: 7,000m altitude hold와 level pitch/roll.
- pursuit: signed closure, ATA alignment, range.
- WEZ: damage scale 100, strict hold `+2/step`, fine ATA penalty.

## Damage 계산

거리 152.4–914.4m와 ATA ±1도 안에서 native step마다
`damage = ((914.4 - distance) / (914.4 - 152.4)) * (1/60)`이다. 양측에 대칭 적용된다.
