# Full-BT 안정화 및 격추 완료 기록 (A57–A103)

## 최종 결과

- 최종 채택 bundle: `team01/sac_mlp_v2_wez_a102_fullbt_heading0p9_branched_qualified/bundle_000001`
- 평가 조건: full BT 100%, 시작 거리 925–1,200m, heading jitter ±0.9°, seeds 1001–1010, 최대 60초.
- 결과: 자체 격추 10/10, WEZ 연속 1초 10/10, crash 0/10, timeout 0/10.
- 평균 격추 시간 4.44초, 평균 누적 target damage 1.033, 최대 연속 WEZ 3.5초.
- 저고도에서는 A35 생존 bundle로 전환하는 recovery wrapper를 함께 검증했으며, 전 episode가 고도 약 6,965m 이상에서 조기 격추되어 실제 recovery 개입은 필요하지 않았다.
- 근거: `01_Work/AIP_LIB/DogFightEnv/Release_260708/05_HW/evaluation/report4/a102_fullbt_heading0p9_10seed_60s_final/summary.json`

## 실험 흐름과 판단

1. A57–A58: 0.9° 실패 seed를 균형 반복하고 WEZ milestone을 추가했지만 두 경계 seed를 동시에 통과하지 못했다.
2. A60–A61: yaw 0/360° 불연속을 없애기 위해 `tactical17`의 sin/cos yaw 관측을 구현하고 warm-start 학습했지만 후반 성능이 퇴행했다.
3. A62–A63: observation feedback을 비교해 LOS azimuth roll feedback으로 0.9° loiter WEZ 1초 10/10, crash 0/10을 달성했다.
4. A64–A67: heading 1° 이상에서는 성능이 급락했고, 60초 생존은 10/10이지만 첫 교차 후 재진입·추가 피해가 없음을 확인했다.
5. A68–A73: SAC→BT/A30 전환은 재교전 중 추락했고, A30 격추 정책에 A35 저고도 recovery를 붙이자 crash를 0/10으로 줄였지만 격추는 2/10이었다.
6. A74–A80: 과거 full-BT 10/10의 실제 주 정책이 A16임을 재확인했다. ±0.2°에서는 격추 10/10을 재현했지만 ±0.5° 이상에서 급락했고 A63→A16 전환도 1/10에 그쳤다.
7. A81–A85: A16 tactical17 actor에 LOS feedback과 pitch/rudder bias를 탐색해 ±0.9° 경계 3-seed에서 2/3 격추까지 개선했다.
8. A86: 10-seed full-BT 미세학습을 38 iteration 수행했다. iteration 1–5는 2/3을 유지했으나 10 이후 급격히 퇴행해 학습 bundle은 모두 미채택했다.
9. A87–A89: self-contained 보정 bundle을 만들고 LOS gain을 50으로 높여 full-BT ±0.9°에서 격추 8/10, WEZ 9/10, crash 0/10을 달성했다.
10. A90–A99: 실패가 음수 heading 두 seed에만 집중됨을 확인했다. 작은 음수와 큰 음수 heading에 필요한 보정이 달라 각각 seed 1002와 1001을 격추하는 구간을 찾았다.
11. A100–A102: 초기 yaw sin 값으로 세 보정 branch를 episode 시작 시 한 번만 선택해 고정하는 latch를 구현했다. A102가 최종 10-seed에서 격추·WEZ·생존 모두 10/10을 달성했다.
12. A103: heading ±1.0° 확장은 격추 9/10, WEZ 10/10, crash 0/10으로 첫 실패했다. 채택 범위는 ±0.9°로 유지하고 2° 이상 평가는 중단했다.

## 구현 변경

- `tactical17` 관측 모드와 tactical16→17 bundle 변환 도구를 추가했다.
- seed-balanced 학습을 위한 `episode_seed_cycle`을 추가했다.
- WEZ 1초 curriculum용 milestone reward를 추가했다.
- direct SAC evaluator에 observation feedback, bundle 전환, mapped recovery 기능을 추가했다.
- 독립 bundle의 관측·action mapping과 policy bias를 지원했다.
- observation feedback에 고정 action bias와 초기 관측 기반 latched branch를 추가했다.
- bundle clone 도구가 full BT, target authority, feedback bias/branch를 metadata에 저장하도록 확장했다.
- 최종 관련 단위 테스트 35개를 모두 통과했다.

## 채택 판단과 다음 계획

- 현재 완료 범위는 “근거리 시작 기하 + heading ±0.9° + full BT”이다. 이 범위에서는 격추와 안정성을 모두 10/10 달성했다.
- 다음 첫 실패점은 heading ±1.0°의 seed 1001이다. 이 범위를 재학습할 때는 A102를 보존하고 실패 branch만 별도 학습해야 한다.
- heading 1.0° 10/10 이후에만 시작 거리 → 위치 randomization → 속도 차 순서로 한 축씩 확대한다.
- 모든 확대 단계는 seeds 1001–1010에서 crash 0/10, WEZ 1초 10/10, 자체 격추 10/10을 통과해야 다음 단계로 이동한다.
- 현재 A102의 세 branch는 좁은 경계 기하에 맞춘 결정적 보정이므로, 넓은 기하 일반화의 최종 해법은 branch 추가가 아니라 tactical17 기반 seed-balanced 재학습이다.
