# 초기 기하 확대 — 거리·heading·위치

## 원칙

- A12 100% 합격 bundle 15는 덮어쓰지 않고 parent로 보존한다.
- `시작거리 → heading → 위치` 순서로 한 축씩 추가한다.
- 각 축은 무학습 10-seed 평가를 먼저 하고 실패할 때만 SAC를 재학습한다.
- 통과 기준은 외부 WEZ 진입, 연속 WEZ 1초 이상, 자체 사격 격추, ownship crash 0이다.

## 기반 구현

- 기존 `radius`는 N/E/D를 동시에 바꾸므로 거리 전용 축으로 사용하지 않는다.
- `distance_range_m`을 추가해 baseline LOS 방향의 separation만 episode별로 변경한다.
- 매 reset마다 YAML 기준 위치를 먼저 복원해 randomization 값이 episode 사이에 누적되지 않도록 수정했다.
- 평가 결과에 초기 heading/N/E/D와 unique initial geometry 수를 추가했다.
- reward/action/warm-start/geometry 관련 `unittest` 16개가 통과했다.

## G1 — 시작거리 randomization

- parent: `sac_mlp_v2_wez_a12_authority_100_synced/bundle_000015`
- 첫 범위: 925–1,050m
- heading·position randomization: 비활성
- 무학습 결과: win 4/10, crash 6/10, 외부진입·WEZ 1초 10/10
- 실제 초기거리: 925.11–1,001.57m, unique initial geometry/trajectory 10/10
- 실패 episode도 누적 피해 0.9563–0.9943까지 도달했으나 격추 전에 약 52초 저고도 추락
- 판단: 최소 WEZ 목표는 일반화됐지만 격추·무추락 기준 미달

### A13 재학습

- 실험명: `team01/sac_mlp_v2_wez_a13_geometry_distance_925_1050`
- parent: A12 bundle 15
- 거리 925–1,050m만 randomization, heading·position 고정
- actor/critic/alpha LR: `3e-6 / 3e-4 / 1e-5`
- gamma 0.999, safety/WEZ reward 유지
- 30 iteration, bundle 매 iteration
- 상태: 재학습 시작

### A13 결과 및 거리 경계 탐색

- A13 대표 bundle 15/20/25/30은 합격하지 못했고 학습 후반으로 갈수록 퇴행했다.
- surface mean 14%·13%·12% 후보는 925–1,050m 대표 seed에서 모두 성공했으며, 최소 변경인 14%를 채택했다.
- surface 14% 후보의 925–1,050m 정식 평가: win 10/10, crash 0, 외부진입 10/10, WEZ 3.3–3.4초, unique geometry/trajectory 10.
- 1,100/1,200/1,300/1,500m 고정 경계는 모두 실패했다. 1,100m가 첫 실패 경계다.
- 1,500m에서는 surface 5–25% scaling을 비교했지만 최고 WEZ도 0.4초여서 단순 scaling만으로는 확대할 수 없었다.

### A14 재학습

- 실험명: `team01/sac_mlp_v2_wez_a14_geometry_distance_925_1100`
- parent: surface 14% warm-start
- 거리 범위: 925–1,100m
- actor LR `1e-5`, critic `3e-4`, alpha `1e-5`, gamma 0.999
- 50 iteration, bundle 매 iteration
- 상태: 첫 실패 거리 경계 재학습 시작

### 거리 축 최종 결과

- A14 학습 후 bundle 15가 대표 seed 2/3 승리로 가장 좋았으나, 이후 bundle은 전반적으로 퇴행했다.
- 대신 경계별 surface actor 보정을 통해 1,100m는 11%, 1,200m는 2%에서 격추 임계점을 찾았다.
- 최종 distance 후보 surface 2%의 925–1,200m 정식 평가: win 10/10, crash 0, 외부진입·WEZ 1초 10/10, unique geometry/trajectory 10.
- 별도 1,200m 고정 경계도 win·WEZ 3.3초였고, 1,300m는 timeout·WEZ 1.3초로 실패했다.
- 현재 거리 축 채택 범위: 925–1,200m. 1,300m 이상은 후속 확대 과제로 남긴다.

## G2 — heading randomization

- 거리 925–1,200m를 유지하고 ownship heading만 추가한다.
- 기존 정수 jitter가 1° 미만 값을 0으로 절삭하던 문제를 연속 균등분포로 수정했다.
- 첫 범위: ±0.5°, position randomization 비활성
- 상태: 무학습 seed 1001–1010 평가 시작

### ±0.5° 무학습 결과

- win 5/10, timeout 5/10, crash 0
- 외부진입 10/10, WEZ 1초 9/10, unique geometry/trajectory 10
- 작은 heading 차이에도 정렬 유지시간이 줄어 격추 일반화 기준을 만족하지 못했다.

### A15 재학습

- 실험명: `team01/sac_mlp_v2_wez_a15_geometry_heading_0p5`
- parent: distance surface 2% 후보
- 거리 925–1,200m + heading ±0.5°, position 고정
- fine alignment: scale 0.5→1.0, reference 10°→2°
- actor LR `1e-5`, 50 iteration, bundle 매 iteration
- 상태: heading 적응 재학습 완료, 대표 bundle 전부 탈락

### A15 결과 및 heading 축 채택

- A15 bundle 15/20/30/40/50을 대표 seed로 평가했지만 모두 격추에 실패해 학습 bundle은 채택하지 않았다.
- 같은 parent에서 범위를 ±0.2°로 낮춘 무학습 정식 평가가 win 10/10, crash 0, 외부진입·WEZ 1초 10/10을 기록했다.
- 최대 연속 WEZ는 3.4초였고 unique initial geometry/trajectory는 각각 10개였다.
- 현재 heading 축 채택 범위: ±0.2°. ±0.5°는 후속 재학습 과제로 남긴다.

## G3 — 위치 randomization

- 거리 925–1,200m와 heading ±0.2°를 유지하고 ownship N/E/D 위치 jitter를 추가했다.
- 첫 5m 무학습 평가: win 7/10, timeout 3/10, crash 0, 외부진입 10/10, WEZ 1초 8/10.
- 1m 실패 seed 선별: win 2/3, timeout 1/3, crash 0. seed 1001은 WEZ 3.1초·누적 피해 0.878까지 갔지만 격추하지 못했다.
- 1m 실패 후 더 큰 2m·3m는 효율상 생략하고 0.5m로 경계를 낮췄다.
- 0.5m 실패 seed 선별은 win 3/3, crash 0, WEZ 3.3–3.4초로 통과했다.
- 0.5m 10-seed 정식 평가: win 10/10, crash 0, 외부진입·WEZ 1초 10/10, 최대 연속 WEZ 3.4초, unique geometry/trajectory 10.
- 현재 위치 축 채택 범위: position radius 0.5m. 1m 이상은 후속 재학습 과제로 남긴다.

## A16 — 초기 기하 self-contained 보존 bundle

- 실험명: `team01/sac_mlp_v2_wez_a16_geometry_qualified`
- parent: `sac_mlp_v2_geometry_distance_warmstart_surface_2`
- 저장 설정: 거리 925–1,200m, heading ±0.2°, position radius 0.5m, 조종 권한 100%.
- 학습 시작 전 1 iteration에서 bundle을 저장했고, 압축 해제한 policy weight SHA-256이 parent와 일치해 정책 가중치가 변하지 않았음을 확인했다.
- self-contained bundle: `artifacts/models/team01/sac_mlp_v2_wez_a16_geometry_qualified/bundle_000001`
- override 없는 최종 10-seed 평가: win 10/10, crash·timeout 0, 외부진입 10/10, WEZ 1초 10/10, 최대 연속 WEZ 3.4초.

## 최종 판단

- 초기 기하 확대 채택 범위는 `거리 925–1,200m + heading ±0.2° + 위치 반경 0.5m`다.
- 1,300m 거리, heading ±0.5°, 위치 반경 1m가 각 축의 확인된 첫 실패 경계다.
- A13/A14/A15의 SAC 재학습은 안정적으로 범위를 넓히지 못했고, actor surface 보정과 범위 축소로 얻은 무학습 후보가 최종 채택됐다.
- 다음 단계는 A16을 보존한 채 표적 난이도를 `직선 → shallow loiter → loiter → 제한 BT → full BT` 순서로 한 단계씩 높이는 것이다.
