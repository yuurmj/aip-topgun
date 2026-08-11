# Report 3 결론

## 최종 판단

- A6에서 확보한 WEZ 진입·유지·격추 능력은 조종 권한 100%, 초기 기하 randomization, 30° loiter, 제한 BT, 근거리 full BT까지 확장했다.
- 필수 목표였던 연속 WEZ 1초 이상은 위 단계의 정식 평가에서 모두 달성했다.
- 다만 실제 full_dogfight의 8종 legacy 초기 기하에서는 장시간 추격 중 고도를 잃는 문제가 남아 있어 최종 안정화 단계에는 아직 도달하지 못했다.
- 따라서 현재 상태는 `근거리 교전 능력 확보 완료 → 실제 full_dogfight 생존 안정화 진행 중`으로 판단한다.

## 완료된 범위

### 조종 권한 확대

- 30%, 60%, 100% 권한 모두 seed 1001–1010에서 자체 격추 10/10, WEZ 1초 이상 10/10, ownship 추락 0건을 달성했다.
- 조종 권한 curriculum은 100%까지 완료했다.

### 초기 기하 확대

- 시작 거리, heading jitter, 위치 randomization 확대를 완료했다.
- 표적 속도 180–200m/s randomization은 WEZ 1초 이상 10/10, 추락 0건, 자체 격추 7/10이었다.
- 표적 속도 205m/s는 현재 격추 실패 경계로 남겼다.

### 표적 난이도 확대

- 30° loiter는 자체 격추 10/10, WEZ 1초 이상 10/10, crash·timeout 0으로 통과했다.
- 해당 설정은 `sac_mlp_v2_wez_a30_loiter30_qualified/bundle_000001`로 패키징했으며 override 없는 평가에서도 같은 결과를 재현했다.
- 제한 BT 30%와 근거리 full BT 100%도 자체 격추 10/10, WEZ 1초 이상 10/10, crash·timeout 0을 달성했다.

## 미완료 범위

### 실제 full_dogfight

- 8종 legacy 기하, BT/loiter 혼합, 위치·자세·속도 randomization, 60초 조건에서는 자체 격추 0/10, WEZ 1초 0/10, ownship 추락 6/10이었다.
- 모든 legacy 시나리오에서 표적을 놓친 뒤 장시간 하강하는 공통 실패가 확인됐다.
- 근거리 full BT 통과는 약 4초 안에 교전이 끝나는 조건의 결과이므로 실제 full_dogfight 안정화와 동일하게 해석하지 않는다.

### 생존 재학습

- A34b의 full 기하 생존 학습은 최선 bundle에서도 추락 4/10으로 탈락했다.
- A35는 첫 실패점인 legacy loiter scenario 5에서 생존율을 0/10에서 5/10으로 개선했다.
- A36 연속 학습은 bundle5 이후 성능이 퇴행해 탈락했다.
- 현재 full 생존 branch의 최선점은 `sac_mlp_v2_full_geometry_survival_a35_loiter/bundle_000030`이다.

## 현재 진행 위치

```text
조종 권한 100% 완료
→ 초기 기하 확대 완료
→ 30° loiter 완료
→ 제한 BT 완료
→ 근거리 full BT 완료
→ 실제 full_dogfight 생존 안정화 진행 중  ← 현재
→ WEZ 재획득
→ 실제 격추 안정화
```

전체 진행률은 약 60–70%로 판단한다. 근거리 WEZ·격추 능력은 확보했지만 실제 full 분포의 장기 생존, 표적 재획득, 격추 일반화가 남아 있다.

## 다음 실험 계획

1. A35 bundle30을 보존한 채 scenario 5 실패 seed 중심으로 생존 10/10을 만든다.
2. scenario 5 통과 후 scenario 7 → scenario 4 순서로 실패 기하를 하나씩 추가한다.
3. 이후 legacy scenario 0–7 전체에서 ownship 추락 0/10을 확인한다.
4. 생존 10/10을 유지하면서 pursuit reward를 작게 복원한다.
5. WEZ 1초 이상 10/10을 다시 달성한 뒤 자체 격추 10/10으로 확장한다.

## 최종 안정화 기준

실제 full legacy seed 1001–1010에서 다음 조건을 모두 만족할 때만 최종 안정화로 판정한다.

- ownship crash 0/10
- 외부에서 WEZ 진입 10/10
- 연속 WEZ 1초 이상 10/10
- `target destroyed` 자체 격추 10/10
- 표적 추락에 의한 false win 0/10

