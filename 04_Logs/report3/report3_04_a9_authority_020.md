# A9 — 20% 중간 조종 권한 SAC 재학습

## 판단

- A8의 15→30% 직접 재학습에서는 합격 bundle이 나오지 않았다.
- A7 bundle 15 무학습 probe는 20%에서 WEZ 2.6초·피해 0.5721, 25%에서 WEZ 2.2초·피해 0.4155였지만 둘 다 저고도 추락했다.
- 20%가 격추에 더 가깝기 때문에 reward 변경 전에 작은 권한 curriculum step을 삽입한다.

## 설정

- 실험명: `team01/sac_mlp_v2_wez_a9_authority_020`
- parent: `sac_mlp_v2_wez_a7_authority_015/bundle_000015`
- control surface: 20%
- 나머지 기하·reward·학습 설정: A8과 동일
- 최대 50 iteration, bundle 매 5 iteration

## 결과

50 iteration을 완료하고 대표 bundle을 seed 1001로 평가했다.

| bundle | 결과 | 연속 WEZ | 누적 피해 | 판단 |
|---:|---|---:|---:|---|
| 15 | timeout | 1.3초 | 0.1613 | 60초 생존, 격추 실패 |
| 30 | crash | 1.3초 | 0.1281 | 탈락 |
| 45 | crash | 0.4초 | 0.0086 | 탈락 |
| 50 | crash | 0.3초 | 0.0037 | 탈락 |

## 판단

작은 권한 step만으로도 합격 모델을 만들지 못했다. 이후 분석에서 lightweight `init_bundle`이 새 RLlib API의 LearnerGroup에 적용되지 않아 첫 학습 update 때 warm-start가 사라지는 결함을 발견했다. 따라서 A8/A9는 정상적인 연속 curriculum의 성능 판단 근거로 사용하지 않고 실패 기록으로 보존한다.
