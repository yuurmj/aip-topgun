# Report 4 — Full Dogfight 안정화

Report3 이후 생존 능력을 보존하면서 pursuit, WEZ 진입·유지, full-BT 자체 격추까지 확장한 기록이다.

## 현재 결론

- scenario 0–7 생존 10/10을 완료했다.
- strict WEZ 연속 1초 목표를 완료했다.
- full BT 근거리 기하, heading jitter ±0.9°에서 자체 격추 10/10과 crash 0/10을 완료했다.
- 최종 채택 bundle은 `team01/sac_mlp_v2_wez_a102_fullbt_heading0p9_branched_qualified/bundle_000001`이다.
- heading ±1.0°는 격추 9/10으로 첫 실패점이며 다음 재학습 시작점이다.

## 문서 목록

1. [Scenario 5 생존 안정화](report4_01_scenario5_survival.md)
2. [Scenario 생존 범위 확대](report4_02_survival_expansion.md)
3. [Pursuit 복원과 WEZ 초기값 비교](report4_03_pursuit_restore.md)
4. [WEZ curriculum 실험 기록](report4_04_wez_curriculum.md)
5. [최종 결론과 다음 계획](report4_05_current_conclusion.md)
6. [Full-BT 안정화 및 격추 완료 기록](report4_06_full_bt_stabilization.md)
7. [Full geometry WEZ 일반화 진행 기록](report4_07_full_geometry_wez_generalization.md)
