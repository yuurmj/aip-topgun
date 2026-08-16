# A11/A12 — SAC warm-start 동기화 수정

## A11 실패와 원인

- A11은 actor LR을 `1e-6`으로 낮췄지만 bundle 15가 WEZ 1.0초·피해 0.0796 뒤 55.5초에 추락했다.
- warm-start 대비 encoder 최대 절대 변화 0.4942, actor head weight 0.1226, bias 1.9661로 작은 LR로 설명할 수 없는 변화가 확인됐다.
- 새 RLlib API에서 lightweight bundle 로더가 EnvRunner에만 가중치를 적용하고 LearnerGroup에는 적용하지 않은 채 반환했다.
- 첫 학습 update에서 learner의 무작위 가중치가 env policy를 덮어쓴 것이 직접 원인이다.

## 수정

- `checkpoint_io._apply_policy_weights`가 LearnerGroup, EnvRunner, local RLModule을 모두 동기화하도록 변경했다.
- learner와 env runner 양쪽에 전달되는 회귀 테스트를 추가했다.
- reward/action mapping 테스트를 포함한 `unittest` 13개가 모두 통과했다.

## A12 실행

- A11과 동일한 100% 보수적 미세학습 설정을 동기화 수정 후 재실행한다.
- bundle을 매 iteration 저장해 첫 실제 업데이트 이후의 합격점을 정밀하게 포착한다.
- bundle 10–14는 warm-up이라 actor 변화가 0이었고, bundle 15에서 첫 update가 반영됐다.
- bundle 15의 warm-start 대비 actor encoder/head 최대 변화는 약 `1e-6`으로 설정한 actor LR과 일치했다.

## A12 bundle 15 정식 평가

- seed 1001–1010: win 10/10, crash 0/10.
- started in WEZ 0/10, 외부 WEZ 진입 10/10.
- WEZ 1초 성공 10/10, 최대 연속 WEZ 3.3초.
- 3.5초에 자체 사격으로 target destroyed, 평균 누적 피해 1.0074.
- 평균 최저고도 6,986.57m.
- randomization이 없어 unique trajectory hash는 1개다.

## 최종 판단

- 100% 조종 권한 고정 조건 단계를 통과했다.
- 채택 bundle: `artifacts/models/team01/sac_mlp_v2_wez_a12_authority_100_synced/bundle_000015`.
- 이 모델은 100% action space를 사용하며 surface 명령 크기를 actor가 작게 선택한다.
- 같은 bundle의 중간 권한 호환성 probe도 30%·60% 모두 WEZ 3.3초, 3.5초 자체 격추, crash 0으로 통과했다.
- 아직 시작 기하와 표적이 고정되어 있으므로 다음 실험은 시작거리 randomization을 가장 먼저 확대한다.
