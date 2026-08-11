# 30° loiter부터 full_dogfight까지 진행 결과

## 30° loiter 판단

- A26 가중치를 유지하고 residual baseline을 미세 탐색했다.
- `[roll, pitch, rudder, throttle] = [0.12, -0.0065, 0.06, -0.20]` 설정에서 seed 1001–1010 자체 격추 10/10, WEZ 1초 이상 10/10, crash·timeout 0을 달성했다.
- 평균 격추 시간은 3.72초, 평균 최소 고도는 약 6,977m였다.
- 같은 설정을 `sac_mlp_v2_wez_a30_loiter30_qualified/bundle_000001`로 패키징했고, override 없는 재평가에서도 10/10 결과를 동일하게 재현했다.

## 제한 BT·full BT 판단

- A16의 상태 의존 actor 출력을 그대로 사용하고 작은 action bias `[0, 0.0111, -0.005, 0]`를 적용했다.
- 제한 BT 30%는 자체 격추 10/10, WEZ 1초 이상 10/10, crash·timeout 0을 달성했다.
- full BT 100%도 동일한 근거리 초기 기하에서 자체 격추 10/10, WEZ 1초 이상 10/10, crash·timeout 0을 달성했다.
- 평균 격추 시간은 3.83초, 평균 최소 고도는 약 6,980m였다.
- 제한/full BT 결과가 같은 이유는 약 4초 안에 격추되어 BT 장기 기동 차이가 나타나기 전에 episode가 끝났기 때문이다.
- BT action-bias 결과는 아직 self-contained 최종 bundle이 아니라 평가 설정 후보로 보존한다.

## 실제 full_dogfight 판단

- 과거 full 조건과 같은 8종 legacy 기하, BT/loiter 혼합, 위치·자세·속도 randomization, 60초로 평가했다.
- 결과는 자체 격추 0/10, WEZ 1초 0/10, ownship 추락 6/10, 표적 추락 false win 3/10, timeout 1/10이었다.
- 8개 시나리오를 각각 분리해도 약 43–45초 뒤 공통적으로 고도를 잃었다. 특정 상대 기하보다 장기 비행 안전과 재획득 능력의 문제다.
- flight-safety SAC를 고도 3,000m 또는 5,000m에서 전환한 hybrid도 추락 4/10이 남아 채택하지 않았다.

## full 기하 생존 재학습

- A34b: 8종 full 기하 전체에서 50 iteration 생존 학습. bundle40이 최선이었지만 정식 평가 추락 4/10으로 탈락했다.
- A35: 첫 실패 loiter 시나리오 5만 사용하고 고도 경고를 4,000m로 앞당겨 30 iteration 학습. bundle30이 scenario 5 생존 5/10으로 현재 생존 branch 최선이다.
- A36: A35 bundle30에서 30 iteration 연속 학습. bundle5 이후 업데이트가 모두 퇴행해 탈락했다.
- 기존 근거리 격추 후보와 A35 생존 후보는 서로 덮어쓰지 않고 별도 보존한다.

## 현재 위치

- 완료: 조종 권한 100%, 초기 기하, 30° loiter, 제한 BT, 근거리 full BT에서 WEZ·격추 검증.
- 미완료: 실제 full legacy 분포에서 60초 무추락, WEZ 재획득, 자체 격추.
- 현재 첫 실패점: legacy loiter scenario 5의 10-seed 생존 5/10.

## 다음 실험 계획

- A35 bundle30을 보존하고 scenario 5 실패 seed 1001·1003·1004·1006·1007을 중심으로 안전 행동을 직접 학습한다.
- 반복 SAC update가 퇴행했으므로 다음에는 actor learning rate를 더 낮추고, replay를 실패 seed와 안전 seed가 균형 있게 구성되도록 한다.
- scenario 5 생존 10/10 후 scenario 7 → scenario 4 → 전체 0–7 순서로 하나씩 확대한다.
- 전체 생존 10/10을 통과한 뒤 pursuit reward를 작게 복원하고, WEZ 1초 → 자체 격추 순서로 다시 연결한다.
- 최종 안정화 판정은 full legacy seed 1001–1010에서 ownship crash 0, WEZ 1초 10/10, 자체 격추 10/10을 모두 만족할 때만 내린다.
