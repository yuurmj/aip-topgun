# Flight Survival 결과

| 시도 | 결과 | 판정 |
|---|---|---|
| A1 safety, 50 iter | timeout 5/10, crash 5/10, 평균 생존 53.2초 | 미달 |
| A2 level, 100 iter | timeout 10/10, crash 0/10, 평균 최저고도 3,848.03m | 통과 |

A2는 altitude hold와 level pitch/roll을 추가한 뒤 60초 생존을 10개 seed 모두 달성했다.

- bundle: `artifacts/models/team01/sac_mlp_v2_flight_safety_seeded_a2_level`
- 평가: `05_HW/evaluation/flight_survival_a2_level/summary.json`
