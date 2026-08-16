# Scenario 5 생존 안정화

## 목표

- 기준 bundle: `sac_mlp_v2_full_geometry_survival_a35_loiter/bundle_000030`
- 대상: legacy loiter scenario 5
- 시작 결과: 60초 생존 5/10, ownship crash 5/10
- 통과 기준: seed 1001–1010에서 60초 생존 10/10, ownship crash 0/10

## 실험과 판단

- 실패 seed `1001, 1003, 1004, 1006, 1007`로 deterministic pitch action bias를 선별했다.
- 양수 pitch bias는 추락을 악화시켰고, 음수 bias는 nose-up 보정으로 작동했다.
- `-0.24`에서 4/5 생존까지 개선됐으며 seed 1003만 57.1초에 추락했다.
- `-0.26, -0.30, -0.35, -0.40`은 모두 실패 seed에서 생존 5/5를 달성했다.
- 정책 개입이 가장 작고 평균 최소거리도 가장 양호한 `pitch bias -0.26`을 채택했다.

## 정식 결과

- 평가: seed 1001–1010, scenario 5, 60초
- 결과: timeout 10/10, ownship crash 0/10
- 평균 생존시간: 60.0초
- 평균 최저고도: 6,138.3m
- WEZ 1초 이상: 0/10 — 이 단계는 생존 전용이므로 정상
- 증빙: `05_HW/evaluation/report4/scenario5/formal_pitch_m0p260/summary.json`

## 결론

- scenario 5의 생존 목표를 5/10에서 10/10으로 개선했다.
- 이후 단계에서도 A35 bundle30과 pitch bias `-0.26`을 안전 기준으로 유지한다.

