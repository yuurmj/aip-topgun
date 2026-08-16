# WEZ curriculum 실험 기록

## 초기 WEZ 재학습

- A38: scenario 5 WEZ bootstrap을 시도했으나 strict WEZ 1초 진입이 없었다.
- A39: 정렬 reward를 강화했으나 후반 bundle에서 추락이 늘어 폐기했다.
- A40: 15초 단기 curriculum으로 거리는 줄였지만 strict WEZ 진입은 만들지 못했다.
- A41: residual pitch baseline을 완화해 근접했지만 ATA 조건을 충족하지 못했다.
- A42: moving alignment 기준을 180°로 넓혀 첫 strict 외부 진입 0.2초를 만들었다.
- A43: A42 hold 미세학습은 0.2초를 넘기지 못했다.
- A44: WEZ angle 10° curriculum은 strict 기준 0.2초에 그쳤다.
- A45: WEZ angle 5° curriculum도 strict 기준 개선이 없었다.
- A46: level/altitude shaping 완화도 초기 0.2초를 보존하는 데 그쳤다.
- A47: A30에서 heading jitter 5°를 바로 재학습했으나 최대 WEZ 0.3초로 간격이 너무 컸다.

## 세밀한 heading curriculum

- A30 무학습 경계 탐색: 0.5°는 WEZ 9/10, 1°는 8/10, 2°·3°는 2/10, 4°는 1/10이었다.
- A48: 첫 실패 0.5°를 재학습해 bundle 28·30 모두 WEZ 1초 10/10, crash 0/10을 달성했다.
- A49: 1° 재학습은 두 실패 seed를 동시에 해결하지 못했다.
- A50: A49 연속학습에서 `learning_starts=0`을 사용하자 성능이 전부 악화돼 폐기했다.
- 추가 경계 탐색: A48 bundle 28은 0.6° 10/10, 0.7° 9/10으로 확인됐다.
- A51: 0.7° 재학습에서 bundle 24가 9/10, 유일 실패 seed가 0.9초까지 개선됐다.
- A52: 낮은 actor LR 미세학습은 0.8초에서 지속 악화돼 폐기했다.
- A53: 근거리 braking·최소거리 penalty를 추가한 v6 reward를 구현했지만 strict ATA 유지가 악화돼 정책 branch는 폐기했다.
- A54: 학습 WEZ angle을 3°로 완화했으나 strict 2° 평가에서는 최대 0.8초로 미통과했다.
- A51 roll bias 탐색: 정책 action roll `+0.1`이 실패 seed를 1.0초로 만들었다.
- self-contained 변환: residual roll baseline을 `0.12 → 0.13`으로 내장해 0.7° WEZ 10/10을 재현했다.
- 0.8° 확대: residual roll baseline `0.145`에서 WEZ 10/10, crash 0/10, 최대 3.2초를 달성했다.
- 0.9° 확대: WEZ 8/10, 자체 격추 1/10이 확인됐지만 두 실패 seed의 조종 보정 방향이 서로 달랐다.
- A55: surface scale 0.10 재학습은 seed 1001의 WEZ 진입을 없애며 악화됐다.
- A56: surface scale 0.15 재학습은 초기에는 seed 1001만, 후반에는 seed 1008만 통과해 동시 통과 bundle이 없었다.

## 구현·검증 변경

- RLlib 기본 log 경로가 workspace 밖의 `ray_results`를 만들며 멈추던 문제를 수정해 `artifacts/logs/.../rllib`을 명시적으로 사용하도록 했다.
- `AIP_RAY_NUM_CPUS` 선택 옵션을 추가해 실험 시 Ray CPU 수를 제한할 수 있게 했다.
- `sac_wez_range_reward_v6.py`에 close-range braking과 inner-WEZ penalty를 분리 구현했다.
- reward 및 action 관련 표준 `unittest` 25개를 실행해 전부 통과했다.
- bundle clone 도구에 `--heading-jitter-deg`를 추가해 채택 기하를 metadata에 내장할 수 있게 했다.
