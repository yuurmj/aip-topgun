# 최종 요약

## 달성

- flight survival: 10/10에서 60초 생존, crash 0.
- 필수 WEZ: 외부 진입 후 연속 3.3초로 1초 기준 달성.
- acquisition 격추: A6 seed 10/10에서 target destroyed.
- terminal/reward 분리, 누적/연속 WEZ 평가, 고정 seed evaluator와 최종 bundle을 생성했다.

## 미달

- A6는 control surface 0%, randomization off인 쉬운 bootstrap이다.
- full dogfight는 crash 6/10, 자체 사격 격추 0/10, WEZ 1초 0/10이다.
- 따라서 최종 목표인 일반 전투 격추 정책은 아직 달성하지 못했다.

## 보존 bundle

1. 안전 비행: `sac_mlp_v2_flight_safety_seeded_a2_level`
2. 엄격 WEZ/격추 bootstrap: `sac_mlp_v2_wez_a6_qualified_kill`
3. full 기준선: `sac_mlp_v2_full_dogfight_once`

## 다음 권고

상세 실행안은 [다음 실험 방향](report2_09_next_experiment_direction.md)에 정리했다. 핵심은 A6에서 full로 바로 점프하지 않고 조종 권한, 초기 기하, 표적 기동성을 한 축씩 복원하는 것이다.
