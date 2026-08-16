# Report 3 — SAC Control-Authority Curriculum

Report 2 완료 후 시작한 후속 실험을 기록한다.

## 현재 목표

A6의 제한된 WEZ 격추 능력을 유지하면서 roll/pitch/yaw 조종 권한을 단계적으로 복원한다.

## 문서 순서

1. [A6 조종 권한 sweep](report3_01_authority_sweep.md)
2. [A7 15% 조종 권한 재학습](report3_02_a7_authority_015.md)
3. [A8 30% 직접 재학습](report3_03_a8_authority_030.md)
4. [A9 20% 중간 재학습](report3_04_a9_authority_020.md)
5. [A10 100% actor 보정 및 미세학습](report3_05_a10_authority_100_adapted.md)
6. [A11/A12 warm-start 동기화 수정과 최종 통과](report3_06_a11_a12_warmstart_sync.md)
7. [초기 기하 확대 — 거리·heading·위치](report3_07_initial_geometry_expansion.md)
8. [표적 난이도 확대 — 직선부터 full BT까지](report3_08_target_difficulty_expansion.md)
9. [Residual SAC 표적 추적 확대 — A18부터 A28까지](report3_09_residual_target_tracking.md)
10. [조종 권한·속도 차 확대 결과](report3_10_authority_and_speed_expansion.md)
11. [30° loiter부터 full_dogfight까지 진행 결과](report3_11_loiter_bt_and_full_dogfight.md)
12. [Report 3 결론](report3_12_conclusion.md)

## 기록 원칙

- Report 2에는 이전 flight-survival부터 full-dogfight 1회까지의 결과만 유지한다.
- Report 3에는 authority sweep 이후 실행하는 모든 실험과 판단을 순서대로 추가한다.
- 학습, 평가, 실패, 채택 bundle을 구분해서 기록한다.
