# Pursuit 복원과 WEZ 초기값 비교

## A37 pursuit 복원

- 초기값: A35 bundle30, residual pitch baseline `-0.26`.
- 대상: legacy scenario 0–7 혼합, BT DLL 연결.
- 안전 shaping은 유지하고 closure/alignment/range reward를 기존 pursuit 실험의 1/3 강도로 복원했다.
- 10 iteration을 학습하고 bundle 2·4·6·8·10을 비교했다.
- bundle 8부터 추락이 발생하고 bundle 10은 선별 seed에서 2/5 추락해 기각했다.
- bundle 6은 정식 10-seed에서 ownship crash 0/10, WEZ 외부 진입 1/10, 최대 유지 0.1초였다.
- seed 1001의 최저고도가 584.5m라 최종 안전 bundle이 아닌 다음 학습용 후보로만 채택했다.
- 증빙: `05_HW/evaluation/report4/pursuit_a37/bundle_000006_formal/summary.json`

## A30 WEZ bundle의 full scenario 재평가

- loiter 30도에서 WEZ·격추 10/10인 A30 bundle을 full scenario 0–7에 적용했다.
- 원본 A30은 ownship crash 6/10, WEZ 1초 0/10이었다.
- 첫 pitch baseline override 시도는 scale 옵션 누락으로 실제 적용되지 않아 무효 처리했다.
- scale `0.1/0.05`와 pitch baseline `-0.26`을 실제 적용한 결과도 crash 5/10, WEZ 1초 0/10이었다.
- 따라서 A30 기술을 직접 가져오는 경로는 기각하고 A37에서 WEZ를 다시 학습한다.

## 다음 판단

- A37은 생존을 보존했지만 WEZ 신호가 희소하므로 scenario 5에서 WEZ bootstrap을 먼저 수행한다.
- scenario 5에서 WEZ 1초 신호를 회복한 뒤 scenario 0–7로 다시 확대한다.

