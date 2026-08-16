# PPO baseline 10회 리플레이 및 문제 분석

작성일: 2026-07-24

## 결론

`team01/ppo_mlp_v1`은 학습 실행과 bundle 저장까지 끝난 기술적 baseline이다. 그러나 전투 성능 baseline은 10전 0승, 0패, 10 timeout, 피해 0%로 실패다.

10회가 모두 같은 궤적이므로 이 결과는 “고정 초기 조건에서 PPO가 안정적으로 같은 실패를 재현한다”는 것은 확인하지만, 서로 다른 상황에 대한 성능 표본 10개를 의미하지는 않는다.

## baseline 확정 근거

- 학습 시작: 2026-07-20 18:06경
- 학습 기록 생성: 2026-07-20 20:18:09
- 소요 시간: 약 2시간 11분
- 알고리즘: PPO
- 학습 완료: 50 iteration
- 누적 sampled steps: 51,200
- 완료 episode: 94
- 최종 bundle:
  - `artifacts/models/team01/ppo_mlp_v1/metadata.json`
  - `artifacts/models/team01/ppo_mlp_v1/policy_weights.pkl.gz`
- 관측: `tactical16` 16차원
- 행동: 4차원
- 상대: `AIP_BASE_target.dll` behavior tree

최종 학습 iteration 49도 `reward_mean=-1.5769`, `episode_len_mean=600`, `win_rate=0`, `loss_rate=0`, `timeout_rate=1`, `ep_wez_steps=0`이었다. 즉 학습이 중단된 것이 아니라, 전투를 끝내지 않는 정책으로 수렴한 상태다.

## 10회 리플레이 결과

| 항목 | 결과 |
|---|---:|
| 총 실행 | 10 |
| 정상 종료 파일 세트 | 10 |
| 승 | 0 |
| 패 | 0 |
| crash | 0 |
| timeout | 10 |
| PPO가 준 피해 | 0% |
| PPO가 받은 피해 | 0% |
| WEZ 진입 | 매회 0 tick |
| 평균 최단 거리 | 136.78 m |
| 평균 최종 거리 | 32,297.99 m |
| 서로 다른 궤적 수 | 1 |

각 실행은 ownship CSV, target CSV, summary JSON으로 저장되어 총 30개 파일이다. 상세 행은 `ppo_baseline_results.csv`, 대표 진행은 `representative_replay.md`에 정리했다.

PPO ownship CSV 10개의 SHA-256은 모두 아래 값으로 같았다.

`6816C6056C2FABBD10ACB33915CA1AE2A0F4772D0AA7B68BCF6D1723BCB450D7`

BT target CSV 10개의 SHA-256도 모두 아래 값으로 같았다.

`40EE9A802932E0317108745C826AEECDF528781527D77563B519D7A7D917A6B9`

## 문제 1 — timeout 보상이 실제로 적용되지 않는다

설정에는 `draw_reward: -30.0`이 있지만, 제한 시간 도달은 `termination.py`에서 `truncated=True`로 반환된다. 반면 `reward.py`의 terminal reward는 `if terminated:`일 때만 적용된다. 따라서 정상적인 60초 timeout에는 -30점이 들어가지 않고 terminal component가 0으로 남는다.

학습 기록도 이 구조와 일치한다. iteration 18 이후 episode 길이가 대부분 최대치 600으로 고정되면서 reward가 약 -1~-4까지 개선됐다. 승리한 것이 아니라, 조기 종료에 붙는 큰 음수 보상을 피하고 step penalty와 pursuit shaping만 받는 생존/회피 정책이 유리해진 것이다.

우선 수정:

1. terminal 분기를 `terminated or truncated`로 처리한다.
2. timeout을 draw와 구분해 명시적인 `timeout_reward`를 둔다.
3. timeout 시 체력 차가 0이면 충분한 패널티를 주되, 승패 보상보다 절댓값은 작게 유지한다.

## 문제 2 — pursuit 대리 목표와 실제 WEZ 사이 간격이 크다

현재 실제 WEZ는 거리 152.4–914.4 m, 전체 각도 2°이므로 PPO 기수 기준 ±1°다. 반면 pursuit shaping은 거리 3,000 m 이내, ATA 30° 이내에서 연속 양의 보상을 준다.

대표 리플레이에서 pursuit 영역에는 약 4.12초 들어갔지만 실제 WEZ에는 한 틱도 들어가지 못했다. 7초 시점에는 거리 775 m로 사거리 조건을 만족해도 ATA가 20.91°였고, 8.27초 최단거리에는 ATA가 109.60°이면서 최소 사거리 안쪽이었다.

우선 수정:

1. `WEZ 진입`, `WEZ 유지 시간`, `상대에게 준 피해`를 단계적으로 더 직접 보상한다.
2. 단순 현재 pursuit 값뿐 아니라 ATA 감소량, 거리 변화량, aspect 개선량 같은 progress reward를 사용한다.
3. merge 이후 상대가 뒤로 넘어가고 거리가 증가하면 이탈 패널티를 준다.
4. 실제 WEZ 보상이 terminal reward를 압도하지 않도록 component별 episode 합을 계속 기록한다.

## 문제 3 — 10회가 독립 표본이 아니다

`run_local_dogfight.py`는 default 초기 위치를 사용하며 scenario/seed 인자를 받지 않는다. default reset에서는 위치 randomization이 비활성이고, `RLActionProvider`는 기본 `explore=False`다. PPO와 BT 모두 결정론적으로 실행되어 10회 CSV가 byte-level로 동일해졌다.

현재 10회는 재현성 확인용으로 보존한다. 성능 평가용 10회는 다음과 같이 별도로 만들어야 한다.

1. `--seed`와 `--scenario-mode` 또는 초기 geometry 인자를 replay CLI에 추가한다.
2. 고정 seed 10개를 실험표에 기록한다.
3. 초기 거리, ATA, aspect, 고도, 속도를 summary JSON에 저장한다.
4. 동일 seed로 SAC/PPO를 각각 실행해 paired comparison을 한다.

## 문제 4 — 학습량과 제어 품질도 부족하다

50 iteration, 51,200 steps는 좁은 WEZ를 학습하기에 작은 baseline 예산이다. 마지막 iteration의 action saturation rate도 약 25.25%로 높아, 연속 제어가 극단값에 자주 붙었다.

다만 학습 시간을 늘리기 전에 timeout 보상 누락과 평가 다양성부터 고쳐야 한다. 현재 보상 그대로 오래 돌리면 같은 회피 목표를 더 강하게 학습할 가능성이 있다.

## 수정 및 재실험 순서

1. timeout terminal reward 수정과 단위 테스트
2. replay CLI에 seed/scenario 추가
3. summary에 WEZ/ATA/aspect/거리/action 통계 추가
4. PPO v2 reward에 WEZ·progress·이탈 항목 반영
5. 짧은 smoke run으로 보상 component와 종료 보상 확인
6. 10개 고정 seed로 평가
7. timeout rate, win rate, damage, WEZ time, min/final distance, action saturation 비교

권장 다음 실험 이름:

`team01/ppo_mlp_v2_timeout_wez_seeded`

## 이번 실행 중 발견해 바로 고친 replay 도구 문제

- `run_local_dogfight.py`에 `--episodes`를 추가해 bundle을 한 번만 복원하고 여러 episode를 실행하도록 했다.
- 환경 reset 시 Tacview 로그 배열이 비워지지 않아 2회차부터 CSV가 누적되는 문제를 수정했다.
- 누적 로그가 생긴 최초 실행 10세트는 `artifacts/logs/invalid_cumulative_2026_07_24_1953`에 격리했고 분석에서 제외했다.
- 수정 후 생성한 10세트는 각 CSV가 정확히 3,600행이다.

## 재현 명령

```powershell
powershell -ExecutionPolicy Bypass -File C:\dev\aip-topgun\03_Experiments\run_ppo_baseline_replays.ps1 -ReplayCount 10
```
