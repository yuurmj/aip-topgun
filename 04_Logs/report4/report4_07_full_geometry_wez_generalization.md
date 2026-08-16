# Full geometry WEZ 일반화 진행 기록 (A104~A137)

## 목표와 판정 기준

- 기존 A102의 좁은 시작 기하 성과를 보존하되, scenario 5의 넓은 초기 기하에서도 60초 생존과 strict WEZ 연속 1초를 달성한다.
- strict WEZ는 거리 152.4~914.4m, ATA 1도 이하를 연속 10 step(1.0초) 만족한 경우로 판정한다.
- 평가 seed는 1001~1010이며, 시작부터 WEZ였던 경우가 아니라 외부에서 진입한 경우만 추적한다.

## 실험 순서와 핵심 결과

1. A104: 좁은 기하 채택 bundle A102를 scenario 0~7에 직접 이식했으나 crash 6/10, WEZ 0/10으로 일반화되지 않았다.
2. A105: native ownship BT를 같은 범위에 적용했으나 crash 3/10, WEZ 0/10으로 교사 정책으로 부적합했다.
3. A106~A118: flight-safe pure-pursuit 교사를 구현하고 lead, bank, roll/rudder 부호, pitch 부호, 고도 recovery를 탐색했다. 추락 0을 유지하면서 ATA를 약 1도까지 줄였지만 WEZ 진입은 만들지 못했다.
4. A119: seed 1001에서 strict WEZ 외부 진입과 실제 damage를 최초 달성했다. 연속 유지시간은 0.3초였다.
5. A120~A123: 900~1,000m부터 throttle을 줄여 근거리 overshoot를 완화했고 유지시간을 0.7초까지 늘렸다. 진입 후 이탈 원인은 거리 하한이 아니라 heading 오차 증가임을 계측했다.
6. A124~A128: 근거리 bank/rudder 이득과 lead를 비교했다. 최고 0.8초였고 큰 lead는 진입 자체를 악화시켰다.
7. A129~A130: WEZ 진입 직후 native BT 또는 A102로 넘기는 계층형 전환을 검토했다. native BT와 A102 모두 전환 후 추락해 폐기했다. tactical16→17 bundle adapter 오류는 수정하고 단위 테스트를 추가했다.
8. A131~A134: heading-rate 보상과 근거리 heading bias를 추가했다. 작은 음의 rate 보상과 양의 bias가 damage를 늘렸지만 유지시간은 0.8초에 머물렀다.
9. A135: `heading bias 16도 / 적용거리 950m`에서 seed 1001 strict WEZ 연속 1.0초와 누적 damage 0.1376을 최초 달성했다. crash는 없었다.
10. A136: 같은 조건을 60초로 재평가해 생존 60초, crash 0, strict WEZ 1.0초, damage 0.1376을 재현했다.
11. A137: scenario 5 seed 1001~1010으로 확장했다. 생존 10/10, crash 0/10은 유지했으나 strict WEZ 1초는 1/10, 외부 진입은 2/10이었다.

## 현재 판단

- 최소 목표인 “실제 strict WEZ 연속 1초”는 외부 진입·60초 평가에서 재현 완료했다.
- 다만 Report 4의 scenario 5 일반화 관문인 WEZ 1초 10/10은 아직 미통과다.
- 실패는 세 군으로 나뉜다: 조준 가능(1001, 1004), 접근하지만 ATA가 큼(1003, 1005, 1006, 1007), 60초 안에 접근하지 못함(1008~1010).
- 고정 이득을 모든 기하에 적용하는 방식만으로는 부족하므로, 초기 기하별 교사 분기 또는 교사 residual을 포함한 SAC 재학습이 필요하다.
- 좁은 기하 A102나 native BT로 단순 전환하는 방식은 full geometry에서 안전성을 깨뜨리므로 사용하지 않는다.

## 다음 진행

1. 서로 다른 접근 이득 후보를 10개 seed에 교차평가해 기하별 최적 교사와 분기 기준을 만든다.
2. 교사 제어를 baseline으로 두고 SAC가 residual만 학습하도록 scenario 5 재학습을 구성한다.
3. scenario 5에서 생존 10/10, strict WEZ 1초 10/10을 통과한 뒤 scenario 7 → scenario 4 → scenario 0~7로 확대한다.
4. 전체 생존·WEZ 관문 통과 후 자체 격추 10/10을 평가한다.

## 근거 경로

- 최초 1초: `05_HW/evaluation/report4/a135_seed1001_bias16_d950/summary.json`
- 60초 재현: `05_HW/evaluation/report4/a136_seed1001_60s_repro/summary.json`
- scenario 5 10-seed: `05_HW/evaluation/report4/a137_scenario5_guidance_10seed/summary.json`

## A138~A201 추가 진행 결과

### 평가 신뢰성 수정

- A181에서 동일 seed 1008이 단독 평가에서는 WEZ 1.2초였지만, 여러 seed를 같은 환경에서 순차 실행하면 0초가 되는 native simulator 상태 누수를 확인했다.
- evaluator에 `--fresh-process-per-seed`를 추가해 seed마다 Python·시뮬레이터를 완전히 새로 생성하도록 수정했다.
- A182 smoke에서 seed 1008의 1.2초가 독립 프로세스로 정확히 재현됐고, 이후 모든 공식 10-seed 평가는 fresh-process 기준으로 변경했다.

### 유도 teacher 구조 개선

1. A138~A149: 기존 pure-pursuit의 gain·lead·heading bias와 residual SAC를 비교했으나 scenario 5는 WEZ 1초 1/10에 머물렀다. 초기 진단 결과 ownship과 표적은 동일 속도였고, 강한 loiter 선회에서 접근 궤적이 벌어지는 문제가 확인됐다.
2. A150: ownship 초기 속도 우위 25~100m/s를 시험했으나 기존 성공 seed까지 잃어 폐기했다.
3. A151~A162: lag-pursuit, bank 확대, 조종 부호, native autopilot을 비교했다. native autopilot은 최대 28km까지 이탈해 폐기했고, 안정화된 직접 유도에서 lead 6초를 적용해 seed 1008 최소거리 257m까지 접근했다.
4. A163~A180: capture 전용 lead·bank·pitch·rate damping, target-bank feedforward, 조건부 감속을 구현했다. `brake range 1,800m + heading error 20도 이내` 조합이 seed 1008에서 생존 60초, crash 0, strict WEZ 1.2초, damage 0.233을 달성했다.
5. A183: fresh-process scenario 5 10-seed 공식 평가는 생존 10/10·crash 0/10을 유지했지만 strict WEZ 1초는 seed 1008 한 건으로 1/10이었다.
6. A184: 기존 A135도 fresh-process로 재평가했으며 seed 1001만 WEZ 1.0초였다. A135와 A180은 서로 다른 선회 기하에 특화돼 하나의 고정 teacher로 합쳐지지 않았다.
7. A185~A195: positive-bank seed 1005/1007에서 lead, capture pitch, bank feedforward, yaw-rate damping, formation hold를 비교했다. 두 seed 모두 strict WEZ 진입은 만들었지만 최대 연속 유지가 0.3초에 머물렀다.
8. A196: 정렬 후 좁은 기하 격추 bundle A102로 전환했으나 두 seed 모두 고도 260~284m에서 추락해 직접 재사용을 폐기했다.
9. A197~A200: 안전 teacher 주변 residual SAC 35 iteration과 consecutive-WEZ progress reward 20 iteration을 재학습했다. 생존과 진입은 보존됐지만 최대 유지시간은 0.3초로 개선되지 않았다.
10. A201: positive-bank seed의 초기 ownship 속도를 15~45m/s 낮췄으나 접근과 damage가 악화되어 폐기했다.
11. A202~A203: tactical17, residual 권한 30%, actor LR `1e-5`로 30 iteration 재학습했다. bundle 5에서 damage는 약 0.052로 증가했지만 WEZ 연속 유지는 0.2초였고, 이후 bundle도 1초 관문을 넘지 못했다. 20초 episode에서 WEZ 이후 경험 시간이 부족한 것이 다음 병목으로 판단됐다.

### 현재 결론

- flight survival 관문은 scenario 5를 포함해 유지되고 있으며, 이번 구간의 모든 공식 controller 평가는 crash 0을 우선 조건으로 삼았다.
- 최소 목표인 full geometry에서 strict WEZ 1초 이상 진입은 seed 1008에서 1.2초로 달성했다.
- Report 4의 scenario 5 일반화 관문인 WEZ 1초 10/10은 아직 미통과이며 현재 공식 결과는 1/10이다.
- deterministic teacher는 안전 접근과 순간 strict 진입에는 유효하지만, 강한 loiter의 선회율·수직 오차·상대속도를 동시에 유지하는 데 한계가 있다.
- 다음 학습은 tactical17 observation을 사용하고 residual 권한을 확대해 SAC가 미세 유지 조종을 직접 학습하도록 진행한다. scenario 5 10/10을 통과하기 전에는 scenario 7·4·0~7 확대나 full dogfight 격추 관문으로 넘어가지 않는다.

### 추가 근거 경로

- seed 1008 WEZ 1.2초: `05_HW/evaluation/report4/a180_d1800_e20/summary.json`
- fresh-process 검증: `05_HW/evaluation/report4/a182_fresh_process_smoke/summary.json`
- scenario 5 공식 10-seed: `05_HW/evaluation/report4/a183_scenario5_fresh_10seed/summary.json`
- A135 fresh-process 비교: `05_HW/evaluation/report4/a184_a135_fresh_10seed/summary.json`
- A197 residual 재학습 후보: `05_HW/evaluation/report4/a198_bundle_005_screen/summary.json`
- A199 progress reward 후보: `05_HW/evaluation/report4/a200_bundle_005_screen/summary.json`

## A202~A274: scenario 5 적응형 WEZ 유지 제어

### 진행 요약

1. A202~A232: tactical17 SAC bundle과 deterministic hold teacher의 조합을 비교했다. 근접 진입은 가능했지만 상대 선회율과 속도 차 때문에 strict WEZ 유지가 0.2초에 머물렀다.
2. A233~A250: 목표 기수 정렬, 속도 matching, target-bank 추종, pitch/roll damping을 분리 탐색했다. seed 1005에서 유지 시간이 0.4→0.9초로 개선됐다.
3. A251: heading-to-bank 3.31~3.35와 속도 offset -47~-49m/s의 좁은 영역에서 seed 1005 strict WEZ 1.0초 후보 4개를 최초 확보했다.
4. A252: 중앙 후보의 60초 fresh-process 검증에서 seed 1005는 1.0초, seed 1007은 0.5초였다. 둘 다 생존 60초·crash 0이었다.
5. A253~A265: 전환 거리, 근접 LOS 혼합, WEZ latch, roll gain/damping을 비교했다. 불연속 latch는 유지 시간을 0.2~0.3초로 악화시켜 기각했고, 연속형 roll damping 조정으로 seed 1007을 0.9초까지 개선했다.
6. A266: roll gain 0.017~0.019, damping 0.35~0.40 범위에서 seed 1007 strict WEZ 1.0초 후보 5개를 확보했다.
7. A267: seed 1007 고정 후보는 1.0초를 유지했지만 seed 1005 성공을 깨뜨려, 하나의 고정 이득으로 두 기하를 처리할 수 없음을 확인했다.
8. A268~A271: 관측된 표적 bank 크기로 heading gain과 roll damping을 연속 보간하는 적응형 controller를 구현했다. A271의 네 damping slope 모두 seed 1005·1007을 동시에 1.0초, 생존 2/2, crash 0으로 통과했다.
9. A272: scenario 5 공식 10-seed fresh-process 평가에서 생존 10/10·crash 0을 유지했으나 strict WEZ 1초는 seed 1005·1007의 2/10이었다. seed 1006은 0.3초, 나머지는 미진입이었다.
10. A273: 기존 음의-bank teacher A180을 재현했다. seed 1008은 다시 1.2초를 통과했지만 다른 음의-bank 4개는 실패했다.
11. A274: 음의-bank 5개에 lead 2/4/6/8초와 heading gain 1.5~3.0을 확대했으나 최고 결과가 단일 seed 0.5초여서 이 고정 teacher 계열의 추가 확대를 중단했다.

### 현재 판단

- 비행 안전 관문은 계속 충족한다: A272에서 60초 생존 10/10, crash 0/10이다.
- strict WEZ 1초 최소 목표는 여러 full-geometry seed에서 재현됐으며, 현재 scenario 5 공식 일반화 성능은 2/10이다.
- 표적 bank의 크기에 따른 연속 적응은 양의-bank 55~59도 범위를 하나의 controller로 합치는 데 유효했다.
- 남은 실패는 음의-bank 5개, 낮은 양의-bank 44도 1개, 높은 양의-bank 63~69도 2개다. scenario 5가 10/10이 되기 전에는 scenario 7·4·전체 0~7로 넘어가지 않는다.
- 다음 실험은 음의-bank 부호 대칭 controller를 먼저 완성한 뒤, 양의-bank 보간 범위를 44~69도로 확대한다.

### 추가 근거 경로

- seed 1005 첫 1.0초 격자: `05_HW/evaluation/report4/a251_g3p31_om49/summary.json`
- seed 1005/1007 동시 통과: `05_HW/evaluation/report4/a271_adaptive_damp_s0p05/summary.json`
- scenario 5 공식 10-seed: `05_HW/evaluation/report4/a272_scenario5_10seed_formal/summary.json`
- 음의-bank A180 재현: `05_HW/evaluation/report4/a273_negative_teacher_s1008/summary.json`

## A275~A286: scenario 5 미통과 기하별 제어 분해

### 진행 요약

1. A275: 음의-bank 표적의 target-heading 정렬 조합 15개를 비교했으나 WEZ 진입은 없어서 기각했다.
2. A276: `capture → align` 순서의 상태 전환을 추가했다. seed 1010에서 연속 WEZ 1.9초 후보를 확보했다.
3. A277~A278: seed 1009의 pure-pursuit 이득을 미세 탐색해 연속 WEZ 1.0초 후보 3개를 확보했다.
4. A279~A280: seed 1001·1002를 비교했다. A135 계열은 seed 1001에서 1.0초를 재현했지만 표적 속도가 더 빠른 seed 1002에서는 진입하지 못했다.
5. A281: seed 1002에서 capture bias와 range를 바꿔도 궤적이 동일해, capture 이전 외곽 추격이 병목임을 확인했다.
6. A282: 외곽 heading gain 0.8~2.5와 lead -4~6초를 선별했다. `heading gain 1.6 / lead 0`에서 seed 1002 연속 WEZ 1.0초를 최초 달성했다.
7. A283: A282 후보를 60초로 재검증해 생존 60초·crash 0·연속 WEZ 1.0초를 재현했다.
8. A284: 남은 양의-bank seed 1003·1004·1006의 heading gain을 선별했다. 각각 WEZ 0.1·0.4·0.3초까지 진입했으며 추락은 없었다.
9. A285: 속도 offset -20~-80m/s와 trail 800~1600m를 비교했다. 활성 provider 구조상 trail이 적용되지 않는 것을 결과 불변성으로 확인했다.
10. A286: ATA 0.5·1·2·5도에서 실제 hold provider로 latch하도록 전환을 활성화했다. seed 1004는 0.5초로 개선됐지만 1003·1006은 각각 0.1·0.3초에 머물렀다. 6개 병렬 실행은 자원 경합으로 정체돼 중단됐고, 동시 2개로 제한한 재실행 36개는 모두 정상 완료됐다.

### 현재 판단

- scenario 5 안전성 기준은 A272의 생존 10/10·crash 0/10을 계속 기준선으로 유지한다.
- 현재 서로 다른 full-geometry seed에서 확인된 strict WEZ 1초 후보는 1001, 1002, 1005, 1007, 1008, 1009, 1010이다. 다만 아직 하나의 공통 controller로 10/10을 통과한 것은 아니다.
- 공식 단일 controller 성능은 여전히 A272의 2/10이며, 개별 후보를 단순 seed 하드코딩으로 결합하지 않는다. 관측 가능한 표적 bank·속도·기하에 따른 일반화 가능한 전환식으로 합쳐야 한다.
- seed 1003·1004·1006은 WEZ 진입에는 성공하므로 외곽 추격보다 진입 직후 교차·이탈이 남은 병목이다. 다음은 진입 전후 거리율과 기수차 trace를 비교해 감속과 bank 동기화 중 우선 원인을 판별한다.
- scenario 5 공통 controller가 fresh-process 10/10을 통과하기 전에는 scenario 7로 확대하지 않는다.

### 추가 근거 경로

- seed 1010 1.9초: `05_HW/evaluation/report4/a276_s1010_a10_l4/summary.json`
- seed 1009 1.0초 후보: `05_HW/evaluation/report4/a278_s1009_l6p1_h1p25_e25/summary.json`
- seed 1002 60초 재검증: `05_HW/evaluation/report4/a283_s1002_h1p6_l0_60s/summary.json`
- 양의-bank heading 선별: `05_HW/evaluation/report4/a284_s1004_g3/summary.json`
- hold 전환 선별: `05_HW/evaluation/report4/a286_s1004_a1_l0/summary.json`

## A288~A293: 진입 직후 이탈 원인 분석과 규칙 제어 종료

1. A288: seed 1003·1004·1006의 매 step trace를 수집했다. WEZ 진입 시 기수차가 각각 36.6도·22.6도·15.4도로 남았고, bank 차보다 기수 평행화 부족이 공통 병목이었다.
2. A289: full-BT 기준선은 세 seed 모두 WEZ 0초이며 47.5~55.2초에 추락해 기각했다.
3. A290: LOS와 표적 기수를 0.25·0.50·0.75 비율로 혼합했다. 최고 유지가 0.2·0.4·0.4초로 소폭 개선됐지만 1초에는 도달하지 못했다.
4. A291: 근접 ATA에서만 표적 기수 비율을 높이는 24개 전환식을 평가했다. 최고 결과는 동일하게 0.2·0.4·0.4초였다.
5. A292: 근접 rudder 보조 옵션을 구현하고 단위 테스트 17개를 통과한 뒤 ±0.1~0.3 이득을 평가했다. ATA는 일부 개선됐지만 WEZ 유지 시간은 늘지 않아 기각했다.
6. A293: hold 내부에서 거리·ATA 조건으로 target-formation native autopilot에 latch하는 공통 관측 기반 전환을 구현했다. 27개 조합 모두 crash 0이었지만 최고 유지가 0.2·0.4·0.4초여서 기각했다.

### 판단

- seed 1003·1004·1006은 규칙 기반 제어의 작은 이득 변경으로는 0.4초 장벽을 넘지 못했다.
- full-BT는 안전성과 WEZ 모두 악화하므로 대체 수단이 아니다.
- seed별 분기나 하드코딩을 추가하지 않고, A204 tactical17 SAC를 seed 1003·1004·1006에 균등 warm-start 재학습하는 방향으로 전환한다.
- A204는 native checkpoint가 없어 replay buffer까지 복원할 수 없으므로, 기존 bundle 가중치를 초기값으로 사용하고 새 replay에서 학습한다.

### 근거 경로

- 진입 순간 trace: `05_HW/evaluation/report4/a288_trace_s1003/trace.csv`
- full-BT 실패: `05_HW/evaluation/report4/a289_bt_s1003/summary.json`
- 고정 heading 혼합: `05_HW/evaluation/report4/a290_s1006_b0p5_g3/summary.json`
- 근접 전환: `05_HW/evaluation/report4/a291_s1006_a2_n0p75/summary.json`
- rudder 선별: `05_HW/evaluation/report4/a292_s1006_rm0p3/summary.json`
- formation 전환: `05_HW/evaluation/report4/a293_s1006_d600_a3/summary.json`

## A294~A304: hard-3 재학습과 관측 병목 확인

1. A294~A295: A204 bundle 2를 warm-start해 seed 1003·1004·1006을 15 iteration 재학습했다. 다섯 bundle 모두 생존 3/3·crash 0이었지만 strict WEZ는 0/3, 최고 0.3초였다.
2. A296: 개선 teacher 후보를 단독 검증했으나 실제 A290의 2단계 전환과 불일치해 WEZ 진입 0/3이었다.
3. A297~A298: 학습·평가가 동일한 composite guidance builder를 사용하도록 수정하고 9 iteration 재학습했다. seed 1006은 0.4초로 개선됐지만 strict 1초는 0/3이었다.
4. A299: 표적 뒤 가상점을 native autopilot으로 추적하는 trail controller 54개를 평가했다. 전부 crash 0이었지만 최고는 0.3·0.4·0.4초였다.
5. A300: 기존 surface guidance의 trail 연결 누락을 수정하고 15개를 재평가했다. ATA는 개선됐지만 range와 동시에 유지하지 못해 strict 1초는 0개였다.
6. A301~A303: WEZ latch와 `capture → align` 전환을 실제 활성화해 평가했다. 전환 뒤 ATA가 74~104도까지 벌어져 최고 0.3초에 머물렀다.
7. A304: tactical17 관측에 target bank·상대 기수차·상대 속도가 없음을 확인했다. 이를 append한 tactical21과 17→21 actor bundle 변환기를 구현했다. 관련 테스트 21개가 통과했고, 변환 전후 hard-3 WEZ 시간은 동일하며 최소거리 차이는 최대 0.000016m였다.

### 현재 판단

- hard-3의 공통 병목은 진입 직후 상대 기동에 맞는 서로 다른 조종을 선택하지 못하는 것이다.
- tactical17에서는 그 차이를 만드는 target bank, 상대 heading, 상대 speed가 직접 관측되지 않아 reward/teacher 강화만으로 일반화하기 어렵다.
- tactical21은 기존 17채널과 WEZ index를 보존하고 뒤에 4채널만 추가한다. 변환 bundle의 새 입력 가중치는 0으로 초기화해 기존 안전 궤적을 실질적으로 보존했다.
- 다음 A305는 tactical21에서 hard-3를 seed-balanced 재학습하고, 신규 관측이 0.4초 장벽을 넘는지 확인한다.

### 근거 경로

- A294 재학습 선별: `05_HW/evaluation/report4/a295_bundle_015_hard3/summary.json`
- composite teacher 선별: `05_HW/evaluation/report4/a298_bundle_003_hard3/summary.json`
- trail autopilot 선별: `05_HW/evaluation/report4/a299_s1003_s900_a3_t400/summary.json`
- tactical21 동일성 원본: `05_HW/evaluation/report4/a304_original_t17_hard3/summary.json`
- tactical21 동일성 변환본: `05_HW/evaluation/report4/a304_upgraded_t21_hard3/summary.json`

## A305~A306: tactical21 첫 학습

1. A305: tactical21 hard-3 30-iteration 학습을 시작했으나 Windows Ray 초기화가 두 차례 지연됐다. 세 번째 실행은 정상 시작해 bundle 5를 저장했지만 20분 실행 제한에 걸려 이후 iteration은 종료됐다.
2. A306: bundle 5를 seed별 개별 fresh-process로 평가했다. 세 seed 모두 35초 생존·crash 0이었지만 WEZ 유지가 0.2·0.1·0.0초로 초기 baseline보다 퇴행했다.
3. tactical21 direct actor 초기화도 seed당 약 2~3분이 걸려, 이후 평가는 후보 수를 줄이고 개별 프로세스 제한을 충분히 둔다.

### 판단

- 신규 4채널을 학습하는 기술 경로는 정상 동작하며 bundle 생성과 direct 추론까지 검증됐다.
- actor LR `1e-4`의 첫 5 iteration은 안전성은 보존했지만 접근/WEZ를 퇴행시켰다.
- bundle 5에서 actor LR을 `3e-5`로 낮춰 짧게 안정화하고, native checkpoint를 저장해 다음부터 critic/replay까지 이어갈 수 있게 한다.

### 근거 경로

- tactical21 bundle 5 seed 1003: `05_HW/evaluation/report4/a306_a305_bundle005_hard3/seed_1003/summary.json`
- seed 1004: `05_HW/evaluation/report4/a306_a305_bundle005_s1004/summary.json`
- seed 1006: `05_HW/evaluation/report4/a306_a305_bundle005_s1006/summary.json`

## A307~A309: tactical21 안정화 판정과 hold curriculum 전환

1. A307은 A305 bundle 5에서 actor learning rate를 `3e-5`로 낮춰 학습했고 bundle 2·4·6과 native checkpoint 5를 남겼다.
2. A308에서 bundle 2·4·6을 scenario 5 hard seed `1003/1004/1006`에 35초씩 fresh-process 평가했다. 세 bundle 모두 생존 3/3·crash 0이었지만 WEZ 1초 성공은 0/3이었다.
3. 최대 연속 WEZ는 bundle 2 `0.1초`, bundle 4 `0.3초`, bundle 6 `0.2초`였다. 기존 hard-seed 최고 `0.4초`도 넘지 못해 A307 계열은 채택하지 않는다.
4. 진입 순간 ATA는 약 `0.4~1.1도`로 충분했지만 다음 1~2 step 안에 각도 조건을 잃었다. 단순 capture 학습보다 진입 후 표적 선회를 따라가는 hold 경험 부족이 병목이라고 판단했다.
5. A309는 strict 2도 WEZ 내부의 정렬·동속·동일 bank 기하에서 시작해 1초 유지를 먼저 반복 학습한 뒤 scenario 5 acquisition으로 돌아가는 curriculum으로 구성한다.

근거 파일:

- A308 bundle 2: `05_HW/evaluation/report4/a308_a307_bundle002_hard3/summary.json`
- A308 bundle 4: `05_HW/evaluation/report4/a308_a307_bundle004_hard3/summary.json`
- A308 bundle 6: `05_HW/evaluation/report4/a308_a307_bundle006_hard3/summary.json`
- A309 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a309_hold_curriculum.yaml`

### A309~A312 결과

1. A309은 strict 2도 WEZ 내부의 정렬·동속·55도 bank 기하에서 6 iteration 학습했다. 학습 episode에서는 최대 연속 WEZ 3.0초가 관측돼 hold 경험 생성 자체는 성공했다.
2. A310에서 scenario 5 hard-3로 되돌리자 bundle 3은 최대 0.2초, bundle 6은 진입 0회였다. 쉬운 hold 분포를 단독 학습하면 기존 capture 능력이 손상돼 채택할 수 없었다.
3. A311은 A304 capture 뒤 첫 WEZ step에 A309 hold 정책으로 전환했지만 두 후보 모두 최대 0.2초였다. 첫 진입 뒤 전환은 조종 지연상 너무 늦었다.
4. A312는 거리/ATA 전환을 `1200m/15도`, `1000m/10도`, `900m/5도`, `700m/5도`로 앞당겼다. 이른 두 전환은 진입 0/3, 늦은 두 전환은 진입 2/3·최대 0.1초여서 모두 탈락했다.
5. 다음 A313은 실제 hard-3 capture 분포를 유지하고 훈련 WEZ 각도만 10도로 완화한다. 이후 엄격 2도 평가에서 개선될 때만 5도 단계로 좁힌다.

근거 파일:

- A310 bundle 3: `05_HW/evaluation/report4/a310_a309_bundle003_hard3/summary.json`
- A310 bundle 6: `05_HW/evaluation/report4/a310_a309_bundle006_hard3/summary.json`
- A311 전환: `05_HW/evaluation/report4/a311_capture_a304_hold_a309_b003_hard3/summary.json`
- A312 전환 예시: `05_HW/evaluation/report4/a312_switch_d700_a5_hard3/summary.json`
- A313 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a313_angle10_hard3.yaml`

### A313~A315 진행

1. A313 10도 curriculum은 실행 제한으로 6 iteration에서 종료됐고 bundle 3·6을 보존했다. 학습 로그의 hard-seed 연속 WEZ는 최대 0.4·0.9·0.5초였다.
2. A314 strict 2도 평가에서 bundle 3은 진입 3/3·최대 0.2초, bundle 6은 진입 2/3·최대 0.2초였다. 아직 기존 0.4초를 넘지 못했다.
3. 하지만 seed당 경험이 사실상 한 episode뿐이므로 10도 curriculum을 즉시 폐기하지 않는다. 진입 3/3인 bundle 3에서 actor LR을 `5e-6`으로 낮추고 10초 episode 24 iteration을 추가하는 A315로 반복 경험을 확보한다.

근거 파일:

- A314 bundle 3: `05_HW/evaluation/report4/a314_a313_bundle003_strict2_hard3/summary.json`
- A314 bundle 6: `05_HW/evaluation/report4/a314_a313_bundle006_strict2_hard3/summary.json`
- A315 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a315_angle10_continue.yaml`

### A315~A318 장기 학습 전 판정

1. A315는 10초 episode 24 iteration을 완료해 bundle 3~24와 checkpoint 6~24를 보존했다. crash는 없었지만 seed 1004·1006의 진입이 10초 이후라 학습 WEZ 경험이 계속 0이었다.
2. A316에서 bundle 6·15·18을 strict 2도로 평가했으나 모두 최대 0.2초였다. 짧은 horizon 반복은 무효로 판정한다.
3. tactical21 새 입력 열의 actor 가중치 L2 norm은 A304 `0`에서 A315 bundle 18 `0.288`로 증가했다. 신규 관측이 학습되지 않는 문제는 아니었다.
4. A317은 target-bank 동기화 teacher 세 조합을 평가했다. `bank blend 0.50 / hold heading blend 0.25`가 진입 3/3·최대 0.3초·crash 0으로 가장 안전했고, blend를 더 높이면 진입률이 떨어졌다.
5. A318은 이 teacher에 SAC residual 40%를 연결하고 20초 horizon·10도 WEZ를 60 iteration 학습한다. seed별 반복 진입 경험을 확보한 뒤 strict 2도 후보를 선별한다.

근거 파일:

- A316 bundle 18: `05_HW/evaluation/report4/a316_a315_bundle018_strict2_hard3/summary.json`
- A317 최선 teacher: `05_HW/evaluation/report4/a317_teacher_b0p50_h0p25_hard3/summary.json`
- A318 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a318_teacher_angle10_long.yaml`

### A318~A322 residual 권한 재설정

1. A318은 강화 teacher + SAC residual 40%로 60 iteration을 계획했지만, checkpoint 30 시점 최근 8개 완료 episode의 10도 WEZ가 모두 0이었다. capture가 빠르게 손상돼 iteration 32에서 조기 종료했다.
2. A320은 동일 강화 teacher에서 residual을 5%로 낮췄다. 진입 3/3은 보존했지만 strict 2도 최대 유지가 0.2초여서 teacher와 기존 actor의 조합이 맞지 않았다.
3. A321은 A304 guidance를 그대로 두고 residual만 0·2·5·10·20%로 비교했다. 2%와 5%가 진입 3/3·최대 0.3초로 최선이었고, 0%와 10~20%는 최대 0.2초였다.
4. A322는 A304 guidance를 보존하고 residual을 5%로 제한한다. 20초 horizon·10도 hard-3 curriculum을 30 iteration 수행해 미세 보정만 학습시킨다.

근거 파일:

- A320: `05_HW/evaluation/report4/a320_guidance5_baseline_hard3/summary.json`
- A321 residual 2%: `05_HW/evaluation/report4/a321_residual_0p02_hard3/summary.json`
- A321 residual 5%: `05_HW/evaluation/report4/a321_residual_0p05_hard3/summary.json`
- A322 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a322_residual5_angle10.yaml`

### A322~A324 10도에서 5도로 축소

1. A322는 30 iteration을 완료했다. 10도 훈련에서 seed 1004가 1.0초를 세 번 재현했고, seed 1003은 0.4초, seed 1006은 0.5~0.6초였으며 crash는 전부 0이었다.
2. A323은 bundle 5·10·15·20·25·30을 guidance/residual 5% 그대로 strict 2도로 복제해 평가했다. 모든 후보가 진입 3/3·crash 0이었지만 최대 연속 WEZ는 0.2~0.3초로 1초 전이는 없었다.
3. bundle 20이 seed별 `0.2·0.3·0.2초`로 가장 균형적이었다. A324는 이 bundle에서 중간 5도 curriculum을 30 iteration 학습한 뒤 strict 2도로 다시 좁힌다.

근거 파일:

- A323 bundle 20: `05_HW/evaluation/report4/a323_bundle020_strict2_hard3/summary.json`
- A324 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a324_residual5_angle5.yaml`

### A324~A326 동기화 reward 추가

1. A324는 5도 curriculum 30 iteration을 완료했다. 학습 유지 시간은 대체로 seed 1003 `0.2~0.3초`, 1004 `0.5~0.6초`, 1006 `0.3초`였고 crash는 0이었다.
2. A325 strict 2도 평가에서 모든 bundle이 진입 3/3·crash 0이었지만 최대 0.2~0.3초였다. 각도 curriculum만으로 strict 장벽을 넘지는 못했다.
3. reward v6는 ATA·거리·closure를 사용하지만 표적 bank, 상대 heading, 상대 speed 차이를 직접 보상하지 않았다. 진입 직후 이탈 원인과 reward가 불일치했다.
4. `student.sac_wez_sync_reward_v7`을 추가했다. 2,000m·ATA 30도 이내에서만 bank/heading/speed 일치를 shaping하고, 외부 기하에서는 세 항을 모두 0으로 게이트한다.
5. matched/mismatched/gated 세 경우의 단위 테스트를 추가했고 전체 reward 테스트가 통과했다. A326은 A324 bundle 30에서 v7·5도 curriculum을 30 iteration 학습한다.

근거 파일:

- A325 bundle 30: `05_HW/evaluation/report4/a325_bundle030_strict2_hard3/summary.json`
- reward v7: `01_Work/AIP_LIB/DogFightEnv/Release_260708/student/sac_wez_sync_reward_v7.py`
- A326 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a326_sync_v7_angle5.yaml`

### A326~A329 expert 분해 전환

1. A326 v7 학습은 5도에서 최대 0.6초였고, A327 strict 2도에서는 최대 0.3초였다. 새 shaping이 공통 정책의 유지 시간을 올리지 못했다.
2. A328은 A326 bundle 30의 residual 권한을 0~40%로 확대 평가했다. 0~5%는 최대 0.2초, 10% 이상은 진입률까지 감소해 v7 actor 보정 방향이 공통 해법으로 수렴하지 않았음을 확인했다.
3. hard-3의 target loiter bank는 seed 1003 약 69도, 1004 약 63도, 1006 약 44도로 관측상 분리 가능하다. seed를 직접 하드코딩하지 않고 target-bank 구간별 expert를 학습·라우팅하는 방향으로 전환한다.
4. A329는 가장 가까운 seed 1004를 strict 2도에서 단독 학습해 1초 expert 달성 가능성을 먼저 검증한다. 성공하면 동일 절차로 high-bank/low-bank expert를 만든다.

근거 파일:

- A327: `05_HW/evaluation/report4/a327_bundle005_strict2_hard3/summary.json`
- A328: `05_HW/evaluation/report4/a328_b30_scale_0p40_hard3/summary.json`
- A329 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a329_seed1004_strict2.yaml`

### A204~A232: 60초 재학습 및 WEZ 유지 병목 분석

12. A204~A205: tactical17 정책을 60초 episode로 10 iteration 이어 학습하고 bundle 2/6/10을 fresh-process로 평가했다. 세 후보 모두 생존 2/2와 WEZ 진입 2/2는 유지했지만 최대 연속 WEZ는 0.2초로 동일해 채택하지 않았다.
13. A206: close geometry에서 표적 heading·고도·속도를 native autopilot으로 추종하는 formation hold를 시험했으나 최대 0.2초여서, native autopilot 응답 지연이 근접 조준 유지에 부적합하다고 판단했다.
14. A207~A216: seed 1005의 0.1초 trace를 기준으로 target-bank feed-forward, 연속 속도 일치, capture lead/trail, 조기 bank feed-forward, roll-rate damping을 비교했다. 진입 시 ownship bank 19도 대 target bank 56도 및 ownship 속도 약 347m/s 대 target 291m/s의 불일치를 확인했지만 각 단독 보정은 최대 0.3초에 머물렀다.
15. A217: 저장 bundle의 guidance residual이 command-line teacher 위에 다시 적용되는 평가 중첩을 발견해 `--disable-saved-guidance-residual`을 추가했다. 이 중첩 때문에 A208~A216의 일부 teacher-only 비교는 동일 trajectory였으며 채택 근거에서 제외했다.
16. A218~A223: A204 residual의 접근 능력은 보존하고 close geometry에서 별도 surface-level hold controller로 전환했다. bank는 target 56도에 근접했으나 전환이 늦거나 rear-quarter 정렬 없이 교차 진입해 최대 0.3초였다.
17. A224~A227: 조향을 보존한 채 1.8km부터 표적 속도를 연속 추종하고 1.2km부터 bank hold를 결합해 strict WEZ 진입 거리를 약 541m에서 828m까지 확대했다. 생존 60초·crash 0을 유지했고 최대 연속 WEZ는 0.3초였다.
18. A228~A232: target-bank 배율, WEZ 진입 후 bank latch, target-heading blend, rudder nose-pointing을 비교했다. rudder를 포화 수준까지 확대해도 최대 0.3초였으며, 진입 시 ownship/target heading 차 약 35도로 인해 LOS가 약 8.5도/s 회전하는 교차 진입이 근본 병목임을 확인했다.

### 갱신된 판단 및 다음 실행

- scenario 5 생존 10/10·crash 0/10은 유지 중이다.
- scenario 5 strict WEZ 1초는 공식 fresh-process 기준 1/10으로 아직 관문을 통과하지 못했다. 따라서 scenario 7, scenario 4, 전체 0~7 확대는 시작하지 않는다.
- 단순 gain 확대나 짧은 residual 재학습은 더 반복하지 않는다. 다음은 `직접 추격 → rear-quarter heading 정렬 → WEZ capture`의 3단 상태기로 교차 진입을 제거한다.
- 3단 상태기가 scenario 5의 seed 1005/1007에서 1초를 만든 뒤 10-seed fresh-process 검증으로 확대하고, 10/10일 때만 다음 scenario로 이동한다.

### 추가 근거 경로

- A204 60초 재학습 후보: `05_HW/evaluation/report4/a205_bundle_002_screen/summary.json`
- native formation autopilot: `05_HW/evaluation/report4/a206_formation_d900_a1/summary.json`
- seed 1005 원인 trace: `05_HW/evaluation/report4/a207_seed1005_trace/trace.csv`
- 속도 일치+bank hold 최선 trace: `05_HW/evaluation/report4/a227_best_trace/trace.csv`
- rudder 포화 비교: `05_HW/evaluation/report4/a232_rm120/summary.json`

## A330~A362: tactical25 회전율 관측과 WEZ 동기화 재학습

### A330~A335 기존 정책·정적 보정 재검증

1. 과거 near-geometry 10/10 WEZ·격추 bundle인 A30을 hard-3에 직접 적용하거나 A304 capture 뒤 전환했지만 최대 연속 WEZ는 0.2~0.3초였다.
2. tactical21→tactical16 adapter에서 yaw 변환 전에 뒤쪽 feature를 잘못 자를 수 있는 호환 문제를 수정했다. 수정 후에도 A30 전환 성능은 동일해 이 버그가 현재 병목의 원인은 아니었다.
3. target-bank 기반 roll/rudder scale, observation feedback, 작은 surface bias를 각각 비교했으나 최고 0.3초를 넘지 못했다. 정적 action 보정은 종료했다.

### A336~A341 tactical25 도입과 10도 관문 최초 통과

1. `tactical25`를 추가했다. tactical21 뒤에 ownship yaw rate, target yaw rate, relative yaw rate, horizontal LOS rate를 붙였다. 기존 tactical21 bundle의 actor 입력 열을 0으로 확장하는 변환기와 16/17/21/25 adapter를 함께 구현했다.
2. A336 변환 직후 hard-3 출력 동등성 평가에서 원본과 변환본이 모두 `0.2/0.3/0.2초`, 진입 3/3, crash 0을 재현했다. 따라서 신규 관측 도입으로 기존 능력이 깨지지 않았다.
3. A337/A337b는 10도 hard-3에서 tactical25를 학습했다. A338 공식 평가에서 bundle 25가 seed별 `0.4/1.2/0.6초`, 1초 성공 1/3, 진입 3/3, crash 0을 달성했다. 이번 구간에서 평가 기준 1초를 처음 실제 재현했다.
4. A339에서 5도로 직접 축소하자 최고 0.6초였다. A340 5도 전용 30 iteration 후 A341 전체 bundle 평가도 최대 0.6초여서 5도 관문은 통과하지 못했다.

근거 파일:

- A336 출력 동등성: `05_HW/evaluation/report4/a336_tactical25_equivalence/summary.json`
- A338 10도 최선: `05_HW/evaluation/report4/a338_bundle025_angle10_hard3/summary.json`
- A341 5도 후보: `05_HW/evaluation/report4/a341_bundle025_angle5_hard3/summary.json`
- tactical25 변환기: `01_Work/AIP_LIB/DogFightEnv/Release_260708/scripts/upgrade_tactical25_bundle.py`

### A342~A348 hold·혼합 curriculum

1. A342는 strict 2도 내부의 정렬·동속·동일 bank 기하에서 hold를 학습했다. 훈련 episode에서 1.1~2.7초가 반복돼 유지 기술 자체는 학습 가능함을 재확인했다.
2. A343에서 A342 가중치만 hard acquisition 환경에 얹자 최대 0.6초였고, A345~A346의 조기/직전 정책 전환도 최대 0.3초였다. 별도 capture/hold actor 전환은 기각했다.
3. hard acquisition seed와 aligned hold seed를 한 replay buffer에서 번갈아 제공하는 `mixed_ref_old_random_hold` 모드를 추가하고 A347을 40 iteration 학습했다. hold episode는 0.8~2.9초를 유지했지만 A348 별도 strict hard 평가는 최대 0.3초였다.
4. 결론적으로 쉬운 hold 분포와 hard acquisition 분포를 MLP actor 하나에 단순 혼합하는 것만으로는 진입 직후 동역학 전이가 해결되지 않았다.

근거 파일:

- A342 hold 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a342_tactical25_hold.yaml`
- A343 hard 전이: `05_HW/evaluation/report4/a343_hold_bundle003_hard5/summary.json`
- A348 mixed strict 평가: `05_HW/evaluation/report4/a348_bundle005_strict2_hard3/summary.json`

### A349~A360 동적 권한·bank·energy 진단

1. 접근 중 residual 5%를 유지하고 근접 조건에서만 20~100%로 높이는 동적 권한을 구현했다. A349~A351에서 권한을 올려도 최대 0.4초였고 40% 이후에는 악화돼 권한 부족 가설을 기각했다.
2. strict seed 1004 trace에서 진입 시 ownship bank 39도 대 target 62도, 속도 261m/s 대 223m/s였고 heading error가 0.3초 만에 -0.5도에서 +2.45도로 넘어갔다.
3. smooth target-bank ramp는 진입 시 ownship bank를 57~62도로 맞추는 데 성공했지만 속도 차는 남아 최대 0.3초였다. trace상 접근 초반 ownship이 target 반대 방향으로 bank -71도까지 갔다가 +62도로 역전하는 S-turn이 과속의 주원인이었다.
4. 조기 target-bank feed-forward는 S-turn을 줄였지만 intercept 진입률도 떨어뜨렸다. bank 비례 pitch compensation은 고도 손실을 줄였으나 vertical alignment를 깨뜨렸으며, 0.005~0.02의 미세 gain에서도 최대 0.3초였다.

근거 파일:

- strict 원인 trace: `05_HW/evaluation/report4/a352_seed1004_strict_trace/trace.csv`
- smooth bank trace: `05_HW/evaluation/report4/a357r2_a20_seed1004_trace/trace.csv`
- 미세 pitch 보정: `05_HW/evaluation/report4/a360_bank_pitch_g0p01_strict2/summary.json`

### A361~A362 실패 seed 집중 재학습과 현재 결론

1. 10도에서 이미 1.2초를 달성한 seed 1004를 제외하고 seed 1003·1006만 40 iteration 집중 재학습했다.
2. 훈련 중 seed 1003은 0.4초, seed 1006은 최대 0.6초에서 정체됐다. A362 평가 최선 bundle 20도 `0.4/1.2/0.6초`, 진입 3/3, crash 0이었다.
3. 현재 보존할 최선은 A337b bundle 25 또는 동등 성능의 A361 bundle 20이다. 10도 1초 성공 1/3은 달성했지만 3/3·10/10 안정화, 5도·strict 2도 축소, scenario 7/4/0~7 확대는 아직 미완료다.
4. tactical25 MLP는 순간 rate를 보더라도 접근 S-turn에서 진입 후 선회 동기화로 이어지는 시간 문맥을 안정적으로 학습하지 못했다. 다음 실험은 recurrent SAC/LSTM 또는 짧은 observation history stack을 소규모 smoke로 비교한 뒤, 더 나은 구조만 장기 재학습한다.
5. 코드 회귀 테스트는 총 64개가 통과했다. 신규 기능은 tactical25, bundle 변환/adapter, mixed curriculum, 동적 near-WEZ residual, smooth bank ramp, 확장 trace이다.

근거 파일:

- A362 최선: `05_HW/evaluation/report4/a362_bundle020_angle10_hard3/summary.json`
- A361 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a337_tactical25_angle10.yaml`
- 현재 테스트: `01_Work/AIP_LIB/DogFightEnv/Release_260708/tests/test_sac_flight_safety_reward.py`

### 다음 실행 계획

- recurrent SAC/LSTM과 history stack 중 기존 bundle 능력을 가장 적게 훼손하는 구조를 3~5 iteration smoke로 비교한다.
- 10도 hard-3에서 먼저 `1초 3/3·진입 3/3·crash 0`을 만든다.
- 이후 5도 → strict 2도 순서로 축소하고, 각 단계가 통과했을 때만 10-seed 평가로 확대한다.
- strict 2도 10/10 이후에만 scenario 7 → scenario 4 → 전체 0~7, pursuit reward 복원, 격추 10/10을 진행한다.

## A363~A388: recurrent 검증, WEZ 관측 수정, hard-3 1초 관문 통과

### A363~A365 recurrent SAC 검증

1. Ray 2.54 SAC LSTM patch를 적용하고 actor-only LSTM 64·sequence 8·prioritized replay 조합의 3 iteration smoke와 bundle recurrent-state 추론 smoke를 통과시켰다. 적용 전 원본 Ray 파일은 `RLLibLstm/tools/backups/aip_20260814_181514`에 보존했다.
2. A364는 seed 1003·1006에 recurrent SAC를 40 iteration 재학습했다. bundle 10/20/40의 hard-3 결과는 모두 대체로 `0.4/1.0/0.5초`로 기존 MLP 최선보다 낮았다.
3. RLlib Algorithm 전체를 쓰는 recurrent 평가는 3 episode에 10분 이상 걸렸다. Ray cluster 없이 저장 RLModule만 직접 구성하는 `--direct-sac-lstm` 경로를 추가해 같은 평가를 약 40초로 줄였다.
4. recurrent actor-only는 시간 상태를 정상 유지했지만 현재 데이터 양에서는 WEZ 유지 성능을 개선하지 못해 채택하지 않았다.

근거 파일:

- A364 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a363_lstm_smoke.yaml`
- A365 bundle 20: `05_HW/evaluation/report4/a365_lstm_bundle020_angle10_hard3_direct/summary.json`
- 경량 recurrent inference: `01_Work/AIP_LIB/DogFightEnv/Release_260708/src/dogfight/ai/rllib_utils.py`

### A366~A368 tactical25 WEZ flag 연결 수정

1. `tactical21/25` observation 계산에는 WEZ flag가 있었지만 환경이 WEZ 설정을 `tactical16/17`에만 전달해 feature 14가 항상 -1이던 연결 오류를 발견했다.
2. tactical16/17/21/25 모두 WEZ 설정을 전달하고 bounded observation space를 사용하도록 수정했으며, 해당 회귀 테스트를 추가했다. 전체 reward·provider 테스트는 65개가 통과했다.
3. A367은 수정된 WEZ flag로 hard acquisition과 aligned hold를 번갈아 26 iteration까지 재학습했다. hold episode에서는 44~71 step WEZ와 자체 격추가 나타났지만 hard-3 최선은 bundle 15의 기존 동등 성능 `0.4/1.2/0.6초`였다.
4. WEZ flag를 -1에서 +1로 바꾸면 actor 출력이 roll `+0.06`, pitch `-0.48`, rudder `+0.57`만큼 달라져 정책이 flag를 실제로 사용함은 확인했다. 다만 residual 5%에서는 hard seed를 개선하기에 권한이 부족했다.

근거 파일:

- WEZ flag 회귀 평가: `05_HW/evaluation/report4/a366_mlp_b20_wez_observation_fix_zero_shot/summary.json`
- A367 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a367_wez_flag_mixed.yaml`
- A368 bundle 15: `05_HW/evaluation/report4/a368_a367_bundle015_hard3/summary.json`

### A369~A383 접근 energy·capture geometry 진단

1. seed 1003 trace에서 기존 진입은 거리 405m, ownship/target 속도 293/244m/s, bank -16/+70도였다. 진입 직후 ATA가 0.37→6.62→10.32도로 커져 추월과 선회 불일치가 핵심 병목이었다.
2. 초기 ownship 속도를 target보다 40m/s 낮추자 `0.8/1.0/1.1초`로 2/3까지 개선됐다. -60m/s는 진입을 잃어 고정 감속만으로는 공통 해가 없었다.
3. brake heading gate, capture speed loop, trail point, capture lead, pitch blend를 각각 비교했다. 단독 변경은 3/3을 만들지 못했고 과도한 감속·bank·pitch 동기화는 진입률을 낮췄다.
4. `초기 속도 -40m/s + capture lead 1초 + target-bank blend 50%(ATA 15도부터 ramp)` 조합은 `0.9/1.1/1.2초`까지 올라와 다음 학습 curriculum의 시작점으로 채택했다.
5. bundle guidance clone 도구에 brake heading, trail, capture lead, target pitch, speed gain/offset을 선택적으로 보존·변경하는 옵션을 추가했다.

근거 파일:

- seed 1003 원인 trace: `05_HW/evaluation/report4/a369_a367_b15_seed1003_trace/trace.csv`
- 실제 속도 비교: `05_HW/evaluation/report4/a372_a367b15_actual_speed_m40_hard3/summary.json`
- 결합 최선: `05_HW/evaluation/report4/a381_lead1_speedm40_b050s15_hard3/summary.json`

### A384~A388 hard-3 1초 달성과 10-seed 확대

1. A384는 위 2/3 energy curriculum에서 hard-3만 26 iteration까지 재학습했다. bundle 15/20은 `0.8/1.1/1.2초`로 2/3을 유지했으나 seed 1003은 0.8초에 정체됐다.
2. A384 bundle 15에서 near-WEZ 조건(거리 1,200m·ATA 15도 이내)의 residual만 20%와 30%로 높였다. 20%는 `1.0/1.5/3.1초`, 30%는 `1.3/1.4/1.2초`로 두 후보 모두 hard-3 1초 3/3·진입 3/3·crash 0을 달성했다.
3. 최소 여유가 큰 30% 후보를 채택했다. fresh-process-per-seed 평가에서도 `1.3/1.4/1.2초`가 그대로 재현돼 이번 최소 목표인 WEZ 1초 이상 진입·유지를 최초로 공식 통과했다.
4. scenario 5 seed 1001~1010 확대 결과는 WEZ 1초 5/10·진입 5/10·crash 0/10이었다. 성공 seed는 1002/1003/1004/1006/1007, 실패 seed는 1001/1005/1008/1009/1010이며 실패군은 WEZ에 진입하지 못했다.
5. 따라서 현재 단계는 `hard-3 관문 통과, scenario 5 10-seed 안정화 진행 중`이다. 아직 5도·strict 2도, scenario 7/4/0~7, pursuit 복원, 격추 안정화로 넘어가지 않는다.

근거 파일:

- A384 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a384_energy_curriculum.yaml`
- hard-3 최초 통과 후보: `01_Work/AIP_LIB/DogFightEnv/Release_260708/artifacts/models/team01/sac_mlp_v2_full_geometry_wez_a386_near_residual/r030`
- fresh hard-3: `05_HW/evaluation/report4/a387_r030_hard3_fresh/summary.json`
- scenario 5 10-seed: `05_HW/evaluation/report4/a388_r030_scenario5_10seed_screen/summary.json`

### 갱신된 다음 실행 계획

- 실패 seed 1001/1005/1008/1009/1010을 집중 재학습하되 성공 seed도 replay에 섞어 현재 5/10 능력을 보존한다.
- scenario 5 10-seed에서 먼저 WEZ 1초 10/10·진입 10/10·crash 0/10을 달성하고 fresh-process로 재검증한다.
- 이후 초기 속도 차를 -40 → -30 → -20 → 0m/s 순서로 복원한다.
- 속도 차 0에서도 10/10을 유지한 뒤 5도 → strict 2도 → scenario 7 → scenario 4 → 전체 0~7 → pursuit reward 복원 → 자체 격추 10/10 순서로 진행한다.

## A389~A426: scenario 5 10/10, 전체 생존 80/80, 공격 재연결 준비

### A389~A398 canonical 재학습과 보존 판단

1. A389는 scenario 5 실패 seed를 더 자주 섞어 25 iteration 재학습했지만 최고 성능이 4/10으로 기존 5/10보다 나빠 폐기했다.
2. 실패가 음수 target bank에 집중되는 점을 이용해 tactical25 observation과 roll/rudder action을 좌우 반전하는 canonical 계약을 구현했다. 평가 전용 canonical은 5/10을 7/10으로 개선했고, 이를 환경 observation/action 계약에 넣은 self-contained A396에서도 7/10을 재현했다.
3. A397은 A396에서 actor LR을 `1e-5`로 낮추고 실패 seed를 집중 재학습했다. bundle 5/10/15/20/25의 WEZ 1초 결과는 각각 `5/10, 3/10, 6/10, 5/10, 3/10`이어서 전부 미채택했다.
4. 이 결과로 actor 재학습이 기존 성공 궤적을 빠르게 잊는다는 점을 확인했고, A396을 보존한 채 deterministic guidance만 한 축씩 조정하기로 했다.

근거 파일:

- A396 기준: `05_HW/evaluation/report4/a396r2_canonical_env_contract_10seed/summary.json`
- A397 설정: `01_Work/AIP_LIB/DogFightEnv/Release_260708/experiments/team01_sac_mlp_v2_full_geometry_wez_a397_canonical_failures.yaml`
- A397 후보 평가: `05_HW/evaluation/report4/a398_a397_bundle005_scenario5_10seed/summary.json` 외 bundle 10/15/20/25

### A399~A414 scenario 5 60초 WEZ 10/10

1. A399의 CLI brake override는 저장 bundle의 residual teacher에 적용되지 않아 무효 screen으로 처리했다. clone 도구에 brake/capture/speed 관련 guidance override를 정식 연결했다.
2. 실제 brake range와 capture speed controller를 바꾼 A400~A402는 실패 seed를 해결하지 못해 폐기했다.
3. 평가 시간을 원래 비행 안전 목표와 같은 60초로 늘리자 A396에서 seed 1005는 1.5초, 1008은 1.2초 WEZ를 달성했다. 전체는 생존 10/10·crash 0·WEZ 9/10이었고, 35초 실패 둘은 정책 실패가 아니라 관측 시간 부족이었다.
4. 남은 seed 1009 trace에서 1.1km 부근 ownship/target 속도가 약 298/234m/s이고 이후 ownship이 365m/s까지 가속해 ATA가 벌어지는 overshoot를 확인했다.
5. 1.4km 이내 근접 속도 match는 1009를 해결했지만 모든 seed에 적용하면 기존 성공이 5/10으로 퇴행했다. 성공 seed의 초기 접근 과속은 약 54~56m/s, 1009는 60~70m/s인 점을 이용해 `과속 58m/s 이상에서 1회 발동 후 latch`하는 guard를 구현했다.
6. self-contained A413은 scenario 5 seeds 1001~1010에서 60초 생존 10/10, crash 0/10, 외부 WEZ 진입·연속 1초 10/10, damage 10/10을 달성했다. 최대 연속 WEZ는 1.8초였다.
7. 근접 속도 guard와 후속 rear-threat 기능까지 포함한 단위 테스트는 최종 71개가 모두 통과했다.

채택 bundle:

- WEZ 기준 A413: `01_Work/AIP_LIB/DogFightEnv/Release_260708/artifacts/models/team01/sac_mlp_v2_full_geometry_wez_a413_speedguard58_latched`
- 공식 10-seed 평가: `05_HW/evaluation/report4/a414_a413_scenario5_10seed_60s/summary.json`

### A415~A424 scenario 7·4·전체 0~7 안전 확대

1. scenario 7은 60초 생존·WEZ 10/10을 재현했다. 코드상 scenario 5·6·7 초기 상태 정의가 완전히 같으므로 새로운 geometry 일반화가 아니라 재현성 확인으로 분류했다.
2. scenario 4 첫 평가는 target mode가 loiter로 초기화돼 BT 객체가 없었고 0.1초 target crash가 발생한 무효 실험이었다. `target_mode=behavior_tree`, `AIP_BASE_target.dll`로 다시 연결했다.
3. 유효한 scenario 4 A413 평가는 ownship loss 1/10(seed 1004, 3.4초), crash 0, WEZ 1초 0/10이었다. native ownship BT도 같은 seed에서 정확히 3.4초에 사망해 초기 후방 753m 위협이 구조적 난도임을 확인했다.
4. rear-threat에서 roll gain만 10배 올린 A420은 bank 77도까지 만들었지만 heading 변화가 약 9도뿐이라 사망 시간이 그대로였다. bank에 비례한 pitch back-pressure `0.25 × |bank|`를 함께 넣은 A422는 seed 1004를 48.3초까지 생존시키고 target crash 종료로 바꿨다.
5. A422 scenario 4 정식 평가는 ownship loss 0/10, crash 0/10으로 안전 관문을 통과했다. target crash 7, timeout 3, WEZ 1초 1/10, 자체 격추 0/10이었다.
6. A422로 scenario 0~3을 각각 10 seeds 평가한 결과 모든 40 episode에서 ownship loss·crash가 0이었다. scenario별 target crash/timeout은 0~2가 7/3, 3이 8/2였고 자체 격추는 모두 0이었다.
7. scenario 0~4 BT 50 episode와 동일 정의인 scenario 5~7 loiter 30 episode를 합친 안전 범위는 ownship loss 0/80, crash 0/80이다. 안전 curriculum 관문은 통과했지만 BT scenario 공격 성능은 WEZ 1초가 `0,0,0,0,1/10`, 자체 격추 0/50으로 미완료다.

현재 채택 bundle:

- A422: `01_Work/AIP_LIB/DogFightEnv/Release_260708/artifacts/models/team01/sac_mlp_v2_full_geometry_wez_a422_rear_break_pull`
- scenario 4: `05_HW/evaluation/report4/a423_a422_scenario4_bt_10seed_60s/summary.json`
- scenario 0~3: `05_HW/evaluation/report4/a424_a422_scenario0_bt_10seed_60s/summary.json` 외 scenario 1/2/3

### A425~A426 격추 정책 전환 재검토

1. scenario 5 seed 1009에서 WEZ 1초 뒤 과거 근거리 격추 A102로 전환했으나 60초 timeout, damage 0.659로 A413 단독 0.795보다 낮아 미채택했다.
2. 같은 시점 native BT 전환도 timeout·damage 0.659로 동일하게 미채택했다.
3. 계층 전환만으로는 넓은 기하의 장기 WEZ 유지·격추가 복원되지 않는다. A422 안전 controller를 기반으로 SAC pursuit/WEZ 유지 reward를 보수적으로 재학습해야 한다.

### 현재 결론과 다음 실행 계획

- 완료: scenario 5의 60초 생존·WEZ 1초 10/10, 전체 scenario 0~7 ownship loss·crash 0/80.
- 미완료: BT scenario의 WEZ 재획득, 장기 damage 누적, 자체 격추 10/10.
- 다음 실험은 A422를 보존하고 낮은 actor LR의 mixed scenario pursuit 재학습을 짧게 수행한다. 중간 bundle마다 0~4 안전 회귀와 scenario 5 WEZ 10/10을 먼저 확인한다.
- 안전·WEZ 기준을 유지하는 후보에만 damage/kill reward를 단계적으로 올리고, 공식 성공은 `ownship_kills`만 인정한다.

## A427~A441: pursuit 복원, WEZ 10/10과 첫 자체 격추 동시 달성

### A427~A432 저학습률 pursuit 복원

1. A422 안전 controller에서 actor LR `5e-6`로 scenario 4 pursuit를 20 iteration 재학습하고 2 iteration마다 후보를 저장했다.
2. 대표 seed 4개 평가에서 모든 후보가 loss·crash 0을 유지했지만, iteration이 늘수록 평균 damage는 증가하고 연속 WEZ는 짧아졌다. 과도한 연속 학습 대신 조기 후보를 선별했다.
3. bundle 4는 scenario 4 안전 10/10과 평균 damage 소폭 개선을 보였으나 scenario 5가 WEZ 9/10으로 퇴행했다. 대신 seed 1007에서 첫 실제 `target destroyed`가 발생해 공격 능력 복원 가능성을 확인했다.
4. bundle 2는 scenario 5 생존·WEZ·damage 10/10, crash 0을 보존했고 최대 damage 0.991까지 도달했다. scenario 4도 loss·crash 0/10을 유지해 안전한 조기 절충점으로 분류했다.

### A433~A440 조건부 speed guard와 채택 bundle

1. bundle 4의 scenario 5 유일 실패 seed 1009를 복구하기 위해 overspeed latch 임계값을 54/50/45m/s로 비교했다. 45m/s는 seed 1009를 WEZ 1.2초로 복구했지만 seed 1010을 0.9초로 퇴행시켰다.
2. trace상 45m/s latch 진입 순간 ATA가 seed 1009는 1.9도, seed 1010은 8.5도였다. 따라서 speed match를 `overspeed >= 45m/s AND ATA <= 5도`에서만 시작하도록 조건을 추가했다.
3. 새 ATA gate는 기본값 180도로 기존 bundle 호환성을 유지하며, guidance/reward 단위 테스트는 72/72 통과했다.
4. A437은 scenario 5에서 생존·외부 WEZ 진입·연속 WEZ 1초가 모두 10/10, crash 0, damage 10/10, 자체 격추 1/10을 동시에 달성했다. 평균 damage는 0.422, 최대 연속 WEZ는 2.1초였다.
5. A437 scenario 4 회귀 평가도 loss·crash 0/10을 유지했다. 따라서 A437을 현재 공격 복원 채택 bundle로 선정했다.

현재 채택 bundle과 근거:

- A437: `01_Work/AIP_LIB/DogFightEnv/Release_260708/artifacts/models/team01/sac_mlp_v2_full_geometry_wez_a437_b4_speedguard45_ata5`
- scenario 5 공식 평가: `05_HW/evaluation/report4/a439_a437_scenario5_10seed_60s/summary.json`
- scenario 4 안전 회귀: `05_HW/evaluation/report4/a440_a437_scenario4_10seed_60s/summary.json`

### A441 실제 mixed full-legacy 기준선

1. scenario 0~7을 seed별로 무작위 선택하고 BT/loiter를 자동 전환하는 최종 판정 조건으로 A437을 평가했다.
2. 결과는 ownship loss·crash 0/10, WEZ 1초 2/10, 외부 WEZ 진입 3/10, 자체 격추 0/10, target crash false win 6/10, timeout 4/10이었다.
3. 분리 scenario 5에서의 10/10과 달리 mixed 평가에서는 scenario 선택이 RNG 소비 순서를 바꿔 loiter seed 1009·1010도 실패했다. 따라서 분리 평가 통과만으로 full 일반화를 선언할 수 없다.
4. 다음 단계는 A437을 보존한 채 scenario 0~7 mixed pursuit rehearsal을 짧게 수행하고, 중간 후보마다 mixed 10-seed와 scenario 5 회귀를 함께 확인하는 것이다.

근거 파일:

- mixed full-legacy 기준선: `05_HW/evaluation/report4/a441_a437_full_legacy_10seed_60s/summary.json`

## A442~A504: full-BT 표적 안정화와 속도 curriculum 재구성

### A442~A480 기존 pursuit 복원 분기 정리

1. A442 mixed 0~7 pursuit rehearsal은 안전성은 유지했지만 full mixed WEZ가 A437의 2/10에서 1/10으로 줄어 채택하지 않았다.
2. rear-threat pitch 보상, 회전 방향 latch, roll damping, rudder, lead-time을 A445~A468에서 비교했다. 일부 seed의 최소 ATA와 거리는 개선됐지만 사거리와 정렬이 동시에 성립하지 않아 WEZ로 이어지지 않았다.
3. A469/A473/A475 scenario 0 SAC 재학습은 actor 출력이 실제로 변했으나 WEZ·damage가 0이었다. 분석 결과 기존 curriculum의 ownship 초기 속도가 target보다 40m/s 낮아 교전 전에 target이 지면으로 추락하는 물리적 병목이 확인됐다.
4. 제한 BT 평가 중 자동 생성된 BT target provider가 reset 후 loiter scenario에도 남는 routing 오류를 발견했다. provider를 behavior-tree scenario에서만 사용하도록 수정하고 단위 테스트를 추가했다.

### A481~A485 target altitude guard

1. full BT target이 지면으로 자멸하는 false win을 제거하기 위해 3,500m에서 개입하고 5,000m에서 해제하는 target altitude guard를 구현했다. 가드는 roll level, pull-up, full throttle을 사용하며 hysteresis 단위 테스트를 포함한다.
2. 2,500/3,000/3,500/4,000m 개입점을 scenario 0 seed 3개로 비교한 결과 네 설정 모두 target crash를 0/3으로 제거했다. 3,500m 설정이 사거리 안 최소 ATA 5.4°로 가장 유망해 채택했다.
3. A437+guard에서 ownship 속도 우위 +80m/s를 적용한 10-seed 기준선은 ownship/target crash 0, WEZ 1초 1/10, 진입 2/10, damage 2/10이었다.

근거 파일:

- guard sweep: `05_HW/evaluation/report4/a484_a483_guard3500_scenario0_3seed_60s/summary.json`
- +80m/s 기준선: `05_HW/evaluation/report4/a485_a483_guard3500_speedplus80_scenario0_10seed_60s/summary.json`

### A486~A500 +80m/s 재학습과 능력 보존

1. A486은 guarded scenario 0, 속도 우위 +80m/s에서 30 iteration을 학습했다. iteration 2가 최고였고 이후 빠르게 퇴화했다.
2. A486-b2는 +80m/s 10-seed에서 crash 0, WEZ 3/10, 진입 4/10, damage 4/10, 자체 격추 1/10을 달성했다. 다만 scenario 5 WEZ가 0/10으로 무너져 단독 채택하지 않았다.
3. residual 권한만 낮춰서는 scenario 5가 복구되지 않아 A437과 A486-b2 actor 가중치를 선형 보간했다.
4. tuned weight 70%인 A496-blend070은 scenario 5에서 생존·WEZ·진입·damage 10/10, crash 0, 자체 격추 2/10을 달성했다. full-BT +80m/s에서는 WEZ 1/10, 진입 2/10, damage 2/10, target crash 0을 유지해 현재 균형 후보로 선정했다.

근거 파일:

- A486-b2 +80m/s: `05_HW/evaluation/report4/a488_a486_b2_speedplus80_scenario0_10seed_60s/summary.json`
- A486-b2 scenario 5 회귀: `05_HW/evaluation/report4/a489_a486_b2_scenario5_regression_10seed_60s/summary.json`
- A496-blend070 scenario 5: `05_HW/evaluation/report4/a499_a496_blend070_scenario5_10seed_60s/summary.json`
- A496-blend070 +80m/s: `05_HW/evaluation/report4/a500_a496_blend070_scenario0_speedplus80_10seed_60s/summary.json`

### A501~A504 첫 실패 경계 +40m/s

1. A496-blend070을 +40/0/-40m/s에서 평가한 결과 모든 조건에서 ownship/target crash는 0이었지만 WEZ는 0/10이었다. 따라서 첫 실패 curriculum을 +40m/s로 확정했다.
2. A502는 +40m/s에서 10 iteration을 재학습했다. iteration 8이 사거리 안 최소 ATA 5.5°로 가장 좋았지만 10-seed WEZ·damage는 아직 0이었다.
3. +40m/s는 30초 학습 episode 안에 사거리와 정렬을 동시에 경험하기 어려워, 다음 실험은 60초 episode로 연장한다. A496 균형 후보를 보존하고 초기 bundle만 선별한 뒤 다시 보간한다.

근거 파일:

- 속도 경계: `05_HW/evaluation/report4/a501_a496_blend070_scenario0_speedplus40_10seed_60s/summary.json` 및 speed zero/minus40 결과
- A502-b8: `05_HW/evaluation/report4/a504_a502_b8_speedplus40_10seed_60s/summary.json`

현재 판단:

- 안전성 병목이었던 full-BT target 자멸은 altitude guard로 제거했다.
- 현재 보존 bundle은 A437, 균형 확장 후보는 A496-blend070이다.
- 전체 완료 조건인 full geometry WEZ 1초 10/10과 자체 격추 10/10은 아직 미달이다.
- 다음 단계는 +40m/s 60초 재학습 → scenario 5 회귀 보간 → 속도 0/-40 순서로 진행한다.

## A505~A619: +40m/s strict WEZ 유지 병목 분석

### A505~A526 첫 strict 1초와 회귀 확인

1. A505는 +40m/s·60초로 재학습했고 iteration 4가 seed 1001에서 strict WEZ 0.5초를 처음 만들었다.
2. capture roll damping 0.50과 근접 residual 0을 적용한 A521은 seed 1001에서 strict WEZ 1.0초를 달성했다. 이번 과제의 최소 목표를 실제 full-BT 조건에서 처음 충족한 결과다.
3. A521의 +40m/s 10-seed 결과는 crash 0, target crash 0, WEZ 1초 1/10, 진입·damage 1/10이었다. 그러나 scenario 5 WEZ는 0/10으로 회귀해 전역 bundle로 채택하지 않았다.
4. A496/A505 actor 보간은 scenario 5 보존과 BT 진입을 동시에 개선하지 못해 폐기했다.

근거 파일:

- A521 +40m/s 10-seed: `05_HW/evaluation/report4/a523_a521_damp050_speedplus40_10seed_60s/summary.json`
- A521 scenario 5 회귀: `05_HW/evaluation/report4/a524_a521_damp050_scenario5_10seed_60s/summary.json`

### A529~A553 역방향 roll 억제와 두 번째 진입 seed

1. seed 1007/1008 집중 SAC 재학습 A529/A533은 초기 bundle 이후 빠르게 퇴화해 추가 vanilla SAC iteration의 효율이 낮다고 판단했다.
2. 큰 heading 오차에서 policy roll 또는 현재 bank가 guidance와 반대로 작동하면 residual roll을 억제하고 guidance roll을 강화하는 조건부 guard를 구현했다. evaluator wrapper와 training env에 동일하게 연결했고 단위 테스트를 추가했다.
3. 임계각 12°·guidance scale 4.0인 A552는 +40m/s에서 seed 1001 strict WEZ 1.2초, seed 1007 진입 0.3초를 만들었다. seed 1007은 기존 사거리 내 최소 ATA 17.0°에서 4.1°까지 개선됐다.
4. clone 도구의 `--preserve-guidance`가 residual scale과 WEZ angle을 기본값으로 덮어쓰는 결함을 발견해, 옵션 미지정 시 원본 값을 보존하도록 수정했다. 결함이 포함된 A548/A549 결과는 채택 근거에서 제외한다.

근거 파일:

- A552 임계각 sweep: `05_HW/evaluation/report4/a553_a552_t12s4_plus40_2seed_60s/summary.json`
- seed 1007 trace: `05_HW/evaluation/report4/a554_a552_t12s4_seed1007_trace/trace.csv`

### A555~A586 근접 hold 제어 한계

1. seed 1007은 약 440m에서 진입한 뒤 표적 pitch 약 78°의 상승 기동을 따라가지 못해 0.3초 만에 이탈했다.
2. hold latch, target pitch blend, target bank 제거, roll gain, roll-rate damping, heading blend, heading/pitch error-rate 보상을 순차 검토했다. roll gain 0.10은 seed 1007을 0.5초·최소 ATA 0.8°까지 개선했지만 1초를 넘기지 못했다.
3. formation autopilot, trail autopilot, ownship BT 전환은 두 핵심 seed의 유지시간을 모두 줄여 폐기했다.
4. 120초 연장에서도 seed 1007은 재진입하지 않아 단순 시간 부족이 아니라 첫 통과 후 geometry 복구 실패로 판단했다.
5. hold 전용 roll damping과 capture pitch-error rate 항을 구현했으며 기본값 0으로 기존 bundle 호환성을 유지했다.

근거 파일:

- roll gain 개선: `05_HW/evaluation/report4/a573_a572_l16rg100_plus40_seed1007_60s/summary.json`
- 개선 trace: `05_HW/evaluation/report4/a574_a572_l16rg100_seed1007_trace/trace.csv`
- 120초 진단: `05_HW/evaluation/report4/a585_a572_l16rg100_plus40_seed1007_120s/summary.json`

### A588~A600 angle curriculum과 body-axis guidance

1. 학습용 WEZ angle을 20°로 넓히면 동일 정책이 seed 1001 2.2초, seed 1007 0.8초의 연속 경험을 만들었다.
2. A590은 seed 1001/1007, 근접 residual 10%, 20° WEZ로 30 iteration을 재학습했다. strict 10° screening에서 iteration 4가 seed 1007 0.5초를 보존했지만 이후 checkpoint는 모두 퇴화했고 기존 최고를 넘지 못했다.
3. 수직 자세의 Euler heading 특이점을 피하기 위해 선택형 body-axis LOS guidance를 구현했다. capture 전 구간 적용은 최고 0.4초, WEZ 진입 후 latch 적용은 0.2초로 악화돼 기본 동작으로 채택하지 않았다.

근거 파일:

- angle 20° 진단: `05_HW/evaluation/report4/a589_a588_angle20_plus40_3seed_60s/summary.json`
- A590 strict checkpoint screening: `05_HW/evaluation/report4/a592_a591_b04_plus40_seed1007_60s/summary.json` 및 A592 형제 결과
- body-axis screen: `05_HW/evaluation/report4/a596_a595_body_r05p10_plus40_seed1007_60s/summary.json`

### A601~A619 target guard 재검토

1. target altitude guard의 pull-up 강도, release altitude, roll-level gain을 조정했다. 대부분 표적 궤적을 크게 바꿔 접근 자체가 사라졌다.
2. roll-level gain 0.05는 seed 1007에서 WEZ 0.5초·damage 0.258로 국소 개선됐지만 seed 1001 진입을 잃어 전역 후보로 채택하지 않았다.
3. 고도에 따라 pull-up을 줄이는 proportional guard를 선택형으로 구현하고 테스트했다. 추락 방지 동작은 유지했지만 현재 full-BT geometry에서는 WEZ 접근이 사라져 기본값은 비활성 상태로 유지한다.
4. 현재 단위 테스트는 81/81 통과한다.

근거 파일:

- target roll-level 0.05: `05_HW/evaluation/report4/a610_a609_guard_g05_plus40_seed1007_60s/summary.json`
- seed 1001/1008 회귀: `05_HW/evaluation/report4/a617_a609_guard_g05_plus40_seeds1001_1008_60s/summary.json`
- proportional guard screen: `05_HW/evaluation/report4/a619_a618_prop_m00_plus40_seed1007_60s/summary.json` 및 A619 형제 결과

현재 결론:

- full-BT +40m/s에서 strict WEZ 1초 최소 목표는 A521 seed 1001에서 달성했으나 일반화는 1/10이다.
- A552는 seed 1001 1.2초를 보존하면서 seed 1007을 첫 진입 0.3초까지 확장한 현재 BT curriculum 기준 후보다.
- A496-blend070은 scenario 5 WEZ 10/10·격추 2/10을 보존하는 전역 균형 후보이며, A552 계열은 아직 scenario 5 회귀 때문에 대체하지 않는다.
- 다음 학습은 strict WEZ를 우연히 기다리는 방식이 아니라 근접·정렬 상태에서 episode를 시작해 hold/recovery 샘플을 반복 생성하고, 통과 후 원래 시작 geometry로 되돌리는 reverse curriculum이 필요하다.
- 전체 완료 조건인 full geometry strict WEZ 1초 10/10과 자체 격추 10/10은 아직 미달이다.

## A620~A689: 실제 전이 curriculum과 조건부 capture 일반화

### A620~A630 실제 전이상태 재학습

1. seed 1007의 실제 진입 직전 상태를 full-state trace로 저장하고 t=56.8초 상태를 초기 geometry로 옮겨 reverse curriculum을 구성했다.
2. 초기화 인터페이스가 N/E/D·자세·속력만 지원해 실제 선속도 방향과 각속도를 재현하지 못했고, A627 12-iteration 학습의 strict screening 최고는 0.3초로 A552의 0.5초 진단 최고를 넘지 못했다.
3. t=50·54·55.5초처럼 더 이른 상태도 검사했으나 원래 진입 궤적을 복원하지 못하거나 JSBSim 초기 안정화가 지연돼 정적 전이상태 curriculum은 폐기했다.

근거 파일:

- 실제 full-state trace: `05_HW/evaluation/report4/a620_a552_seed1007_fullstate_trace/trace.csv`
- A627 후보 screening: `05_HW/evaluation/report4/a630_a629_b02_plus40_seed1007_60s/summary.json` 및 A630 형제 결과

### A636 실제 60초 궤적 초미세 재학습

1. 정적 상태 대신 seed 1007의 실제 60초 episode를 유지하고 actor LR을 `1e-6`으로 낮춰 8 iteration을 학습했다.
2. bundle 3이 seed 1007을 0.3초에서 0.5초, damage를 0.154에서 0.206으로 개선했다. seed 1001도 1.1초로 1초 성공을 보존했지만 seed 1008은 진입하지 못했다.
3. RLlib Algorithm 복원 평가가 장시간 정지하는 병목을 확인해 이후 평가는 동일 lightweight actor를 직접 실행하는 `--direct-sac-mlp` 경로로 통일했다.

근거 파일:

- A636 checkpoint screening: `05_HW/evaluation/report4/a642_a636_b000003_direct_plus40_seed1007_60s/summary.json`
- seed 1001/1008 교차검증: `05_HW/evaluation/report4/a643_a636_b03_direct_plus40_seeds1001_1008_60s/summary.json`

### A645~A669 근접 hold와 capture 경계 튜닝

1. 진입 후 bank freeze와 hold 전용 pitch damping을 구현했으며 기본값은 비활성으로 두었다. 각각의 sweep은 0.5초 plateau를 넘지 못해 채택하지 않았다.
2. capture lead를 1.0초에서 0.90초로 줄이고 capture 시작 거리를 1,550m로 앞당기자 seed 1007이 0.6초·damage 0.258·최소 ATA 0.48도로 개선됐다.
3. 이 고정 조기 capture는 seed 1001을 1.1초에서 0.8초 이하로 낮춰 전역 설정으로는 채택하지 않았다. A668 초미세 재학습도 0.6초를 넘지 못했다.

근거 파일:

- capture 거리 1,550m: `05_HW/evaluation/report4/a661_a660_capturerange_1550_seed1007_60s/summary.json`
- A668 미세학습 screening: `05_HW/evaluation/report4/a669_a668_b000001_seed1007_60s/summary.json` 및 A669 형제 결과

### A673~A689 조건부 capture와 현재 채택 기준

1. 거리 약 1,600m에서 seed 1001은 target pitch/bank가 -8.5/-5.1도였고 seed 1007은 76.4/105.6도였다. target pitch 45도 또는 bank 80도 이상일 때만 1,550m 조기 capture를 선택하도록 구현했다.
2. A673 10-seed 결과는 ownship/target crash 0/10, strict WEZ 1초 2/10, 진입·damage 3/10이었다. seed 1001은 1.6초, seed 1004는 1.9초, seed 1007은 0.6초였다.
3. seed 1009는 target bank 약 162도로 별도 유형이었다. 처음 1,550m 경계를 통과할 때 maneuver profile을 한 번만 선택하고, extreme-bank profile에는 lead 1.15초를 고정하는 A687을 만들었다.
4. profile latch 적용 후 seed 1001=1.1초, 1004=1.9초 성공을 유지하면서 seed 1007=0.6초, seed 1009=0.8초까지 동시에 확보했다. 현재 full-BT curriculum 기준은 A687이며 전역 scenario 5 기준 A496은 별도 보존한다.
5. 현재 단위 테스트는 85/85 통과한다.

근거 파일:

- A673 10-seed: `05_HW/evaluation/report4/a675_a673_conditional_capture_10seed_60s/summary.json`
- A687 profile latch 4-seed: `05_HW/evaluation/report4/a689_a687_latched_tier_4seed_60s/summary.json`

현재 판단과 다음 순서:

- full-BT strict WEZ 1초는 2/10까지 확대됐고 crash 방지는 유지됐다. 자체 격추는 아직 0/10이다.
- 다음은 seed 1009의 0.8초를 초미세 재학습으로 1초 이상 만든 뒤, ATA 26~34도 실패군 1003/1006/1008/1010에 profile을 하나씩 추가한다.
- 최소거리 자체가 WEZ 밖인 seed 1002/1005는 마지막에 장거리 pursuit profile로 분리한다.
- strict WEZ 10/10 뒤 damage/kill reward를 확대하고 자체 격추 10/10을 검증한다.
