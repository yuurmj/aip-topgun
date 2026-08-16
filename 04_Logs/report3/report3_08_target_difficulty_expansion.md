# 표적 난이도 확대 — 직선부터 full BT까지

## 원칙과 판정 기준

- parent는 초기 기하 채택점 `sac_mlp_v2_wez_a16_geometry_qualified/bundle_000001`로 고정했다.
- 난이도는 `직선 → shallow loiter → loiter → 제한 BT → full BT` 순서로 평가했다.
- 합격은 자체 사격 격추 10/10, 외부 WEZ 진입·연속 1초 10/10, ownship crash 0으로 판정했다.
- `target altitude below min`은 raw win이더라도 자체 격추로 인정하지 않았다.

## 구현 및 평가 보강

- shallow/loiter를 bank 값으로 분리할 수 있도록 평가기에 target loiter bank/pitch override를 추가했다.
- 제한 BT는 native BT의 roll/pitch/rudder에만 `target_control_surface_scale`을 적용하고 throttle은 유지하도록 구현했다.
- 평가 summary에 `ownship_kills`와 `target_crash_wins`를 분리해 표적 추락 승리를 격추로 오인하지 않도록 보강했다.
- 관련 단위 테스트는 총 18개로 늘었다.

## T0 — 직선

- A16의 `loiter(enabled=false)`가 직선 단계다.
- seed 1001–1010 결과: 자체 격추 10/10, 외부 WEZ 진입·1초 10/10, 최대 연속 WEZ 3.4초, crash·timeout 0.
- 판정: 통과, 현재 최종 채택점 유지.

## T1 — shallow loiter 경계 탐색

### 10° 무학습

- 자체 격추 8/10, timeout 2/10, crash 0.
- 외부 WEZ 진입·1초는 10/10, 최대 연속 WEZ 3.4초였다.
- seed 1001·1002가 각각 누적 피해 0.2688·0.7496에서 timeout됐다.
- 판정: 첫 실패 단계, SAC 재학습 대상으로 선정.

### A17 재학습

- 실험명: `team01/sac_mlp_v2_wez_a17_target_shallow_loiter`
- parent: A16, bank 10°, 초기 기하 범위 유지.
- actor/critic/alpha LR: `1e-6 / 3e-4 / 1e-5`, gamma 0.999.
- 40 iteration, 5 iteration 간격 bundle 저장.
- 실제 update 전 bundle 5·10은 parent와 weight hash가 같았고 bundle 15부터 변했다.
- 대표 seed 1001–1003 결과: bundle 15는 격추 1/3·timeout 2, bundle 20은 격추 1/3·crash 2, bundle 25는 격추 0/3·crash 3.
- 판단: 학습이 진행될수록 안전성과 격추가 함께 퇴행해 A17 bundle은 모두 탈락.

### bank 경계 및 actor 보정

- bank 5° 정식 평가: 자체 격추 9/10, timeout 1, crash 0, WEZ 1초 10/10.
- bank 2° seed 1001: timeout, WEZ 2.6초, 누적 피해 0.6127.
- A16 actor mean 2×·4×·8× 보정도 bank 2° seed 1001에서 같은 timeout 결과여서 채택하지 않았다.
- 판정: 비행경로가 조금만 선회해도 격추 정렬을 끝까지 유지하지 못하므로 shallow 단계 미통과.

## T2 — loiter 30° 기준선

- 자체 격추 0/10, timeout 10/10, crash 0.
- 외부 WEZ 진입·연속 1초는 10/10, 최대 연속 WEZ 2.8초.
- 평균 누적 표적 피해 0.5754, 최대 0.7205.
- 판정: 진입 능력은 유지하지만 선회 표적 추적·재조준 능력이 없어 탈락.

## T3 — 제한 BT 30% 기준선

- raw win 10/10이지만 `target destroyed` 자체 격추는 1/10뿐이었다.
- 나머지 9/10은 56.4초에 `target altitude below min`으로 표적이 추락한 false win이다.
- 외부 WEZ 진입·연속 1초 10/10, ownship crash 0.
- 판정: 표적 추락을 제외하면 격추 기준 미달로 탈락.

## T4 — full BT 100% 기준선

- 자체 격추 1/10, timeout 9/10, crash 0.
- 외부 WEZ 진입·연속 1초 10/10, 최대 연속 WEZ 3.4초.
- 평균 누적 표적 피해 0.8590, 최대 1.0095.
- 판정: 최소 WEZ 목표는 전 seed에서 유지했지만 격추 일반화는 미달.

## 최종 판단

- 표적 난이도 전체에서 필수 목표인 외부 WEZ 1초 이상은 유지했다.
- 격추 합격 단계는 직선 A16뿐이며 shallow loiter부터 아직 미통과다.
- A17을 반복하는 방식은 퇴행 가능성이 높으므로 다음 학습은 동일 설정 반복이 아니라 선회 표적 추적용 residual/각속도 학습 구조가 필요하다.
- A16은 덮어쓰지 않고 최종 채택점으로 유지하며 A17 및 이후 결과는 실패 기준선으로 보존한다.

## 다음 실험 계획

- A18은 A16의 안정 action을 baseline으로 고정하고 SAC가 작은 residual 조종만 학습하도록 구성한다.
- 먼저 2° shallow loiter, 고정 950m·고정 heading/position에서 seed 다양성만 주고 추적을 학습한다.
- reward에는 ATA 자체 값뿐 아니라 ATA 감소량 또는 LOS angular-rate 감소 신호를 추가해 움직이는 표적을 따라가는 행동을 직접 보상한다.
- 2° 자체 격추 10/10을 통과한 뒤 bank를 5°→10°→30°로 올리고, 그 뒤 제한 BT 30%→full BT 100%를 다시 평가한다.
- BT 단계에서는 `ownship_kills`만 승리로 인정하고 `target_crash_wins`는 실패로 집계한다.
