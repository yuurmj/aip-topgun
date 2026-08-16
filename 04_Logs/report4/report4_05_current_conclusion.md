# Report 4 최종 결론

## 완료 판단

- 최종 채택: `team01/sac_mlp_v2_wez_a102_fullbt_heading0p9_branched_qualified/bundle_000001`
- full BT 100%, 시작 거리 925–1,200m, heading jitter ±0.9°, seeds 1001–1010, 최대 60초에서 평가했다.
- 자체 격추 10/10, WEZ 연속 1초 10/10, crash 0/10, timeout 0/10을 달성했다.
- 평균 격추 시간은 4.44초이며, 10개 episode가 모두 고도 약 6,965m 이상에서 종료됐다.
- 최종 근거는 `05_HW/evaluation/report4/a102_fullbt_heading0p9_10seed_60s_final/summary.json`이다.

## 핵심 판단

- SAC는 연속 조종과 좁은 기하의 정밀 WEZ 유지·격추에 사용하고, A35 생존 SAC는 저고도 recovery 역할로 분리했다.
- full BT 자체는 현재 채택 범위에서 문제가 아니며, 다음 병목은 초기 heading 범위다.
- A86 seed-balanced 미세학습은 iteration 10 이후 격추 능력이 붕괴했으므로 채택하지 않았다.
- A102는 초기 yaw 부호·크기에 따라 세 보정값 중 하나를 episode 시작 시 latch해 tactical16 경계 민감도를 우회했다.
- 이 방식은 현재 범위의 안정화 해법이며 넓은 기하의 일반화 해법은 아니다.

## 다음 실험 계획

1. A102를 보존한 채 heading ±1.0°의 실패 seed 1001 branch만 재학습한다.
2. heading 1.0° 격추·WEZ 10/10, crash 0/10 이후 heading 범위를 조금씩 넓힌다.
3. heading 단계 통과 후 시작 거리 → 위치 randomization → 속도 차 순서로 한 축씩 확대한다.
4. 넓은 기하에서도 동일 기준을 통과하면 legacy scenario 0–7 full-BT 정식 평가로 넘어간다.
5. 최종 통합 평가는 공통 terminal·damage 계약과 동일 seeds 1001–1010을 유지한다.
