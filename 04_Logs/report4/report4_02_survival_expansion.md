# Scenario 생존 범위 확대

## Scenario 7

- A35 bundle30 + pitch bias `-0.26`으로 seed 1001–1010을 평가했다.
- 결과: 60초 생존 10/10, ownship crash 0/10, 평균 최저고도 6,138.3m.
- 현재 Release의 scenario 5·6·7은 코드상 같은 초기 상태와 loiter 설정이므로 scenario 5와 동일한 결과가 재현됐다.
- 증빙: `05_HW/evaluation/report4/scenario7/formal_pitch_m0p260/summary.json`

## Scenario 4

- 첫 평가는 BT DLL이 빠져 표적이 0.1초에 추락했으므로 무효 처리했다.
- `AIP_BASE_target.dll`을 연결한 재평가는 ownship 생존 10/10, ownship crash 0/10이었다.
- 7회는 BT 표적의 지면충돌로 조기 종료됐고 3회는 60초 timeout이었다.
- 표적 지면충돌은 자체 격추로 계산하지 않는다.
- 증빙: `05_HW/evaluation/report4/scenario4/formal_bt_pitch_m0p260/summary.json`

## Scenario 0–7 혼합

- BT DLL을 연결하고 scenario index 0–7을 seed별 무작위 선택해 평가했다.
- 결과: ownship 생존 10/10, ownship crash 0/10.
- 6회는 BT 표적 지면충돌, 4회는 60초 timeout이며 자체 격추는 0/10이다.
- 평균 최저고도는 4,898.9m였지만 seed 1001은 1,379.4m까지 하강했다.
- pursuit 복원 이후 seed 1001을 저고도 회귀 감시 seed로 사용한다.
- 증빙: `05_HW/evaluation/report4/all_scenarios/formal_pitch_m0p260/summary.json`

## 결론

- 생존 curriculum의 scenario 5 → 7 → 4 → 0–7 혼합 관문을 모두 통과했다.
- 다음 단계는 현재 안전 보정을 보존하면서 pursuit reward를 점진적으로 복원하는 것이다.

