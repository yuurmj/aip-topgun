# Full Dogfight 결과

50 iteration 단일 run만 수행했다. A6는 별도 보존하고 full authority, BT/loiter 혼합 target, 무작위 legacy scenario 0–7을 사용했다.

| 지표 | 결과 |
|---|---:|
| win | 3/10 |
| ownship crash | 6/10 |
| timeout | 1/10 |
| 자체 사격 target destroyed | 0/10 |
| 외부 WEZ 진입 | 1/10 |
| WEZ 1초 | 0/10 |
| 최대 연속 WEZ | 0.1초 |
| 최대 누적 피해 | 0.0198 |

win 3회는 모두 `target altitude below min`이었다. full 일반화는 미달이며 이 bundle은 기준선이다.

- bundle: `artifacts/models/team01/sac_mlp_v2_full_dogfight_once`
- 평가: `05_HW/evaluation/full_dogfight_once/summary.json`
