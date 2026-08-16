# 다음 회의 결과물 — 완료본

작성일: 2026-07-24

| 결과물 | 상태 | 파일 |
|---|---|---|
| SAC baseline 문제 정리 | 완료 | `SAC_baseline/sac_baseline_problem_analysis.md` |
| PPO baseline 문제 정리 | 완료 | `PPO_baseline/ppo_baseline_problem_analysis.md` |
| SAC/PPO 통합 기록표 | 완료 | `SAC_PPO_baseline_integrated_record.md` |
| 대표 리플레이 캡처 또는 메모 | 완료 | SAC/PPO 각 `representative_replay.md` |
| Reward에서 사용할 값 목록 | 완료 | `reward_value_catalog.md` |
| SAC v1 Reward 수정 방향 | 완료 | `reward_v1_revision_directions.md` |
| PPO v1 Reward 수정 방향 | 완료 | `reward_v1_revision_directions.md` |
| 다음 실험 이름 확정 | 완료 | SAC/PPO 통합 기록표 및 Reward 수정 방향 |

## 회의에서 먼저 말할 결론

1. SAC는 주최측 iteration 15 중간 bundle reference이며 10회 모두 45.25초 급강하 추락이다.
2. PPO는 우리 팀이 50 iteration 학습한 bundle이지만 10회 모두 무피해 timeout이다.
3. 두 정책 모두 약 8.2초의 최초 merge 이후 overshoot하고 실제 WEZ에는 진입하지 못한다.
4. SAC는 비행 안정성, PPO는 timeout 보상과 재획득이 첫 수정 대상이다.
5. 다음 평가는 같은 seed 10개를 사용한 paired evaluation으로 바꾼다.

## 확정 실험

- `team01/sac_mlp_v2_flight_safety_seeded`
- `team01/ppo_mlp_v2_timeout_wez_seeded`
