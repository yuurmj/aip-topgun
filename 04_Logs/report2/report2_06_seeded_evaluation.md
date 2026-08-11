# Seed 평가 방식

공통 seed는 1001–1010이며 inference는 `explore=False`다.

각 episode에 outcome, end condition, 생존시간, 최저고도, 최저/최종거리, 최종 ATA, 양측 health/damage, WEZ 누적/최대연속, 외부 진입 여부, trajectory hash를 저장한다.

- survival: 10개 seed 모두 60초, crash 0.
- A6: 10개 seed 모두 win/3.3초 WEZ지만 randomization off라 hash 1개.
- full: hash 10개, win 3/crash 6/timeout 1로 실제 seed 다양성이 반영됨.

원본은 `05_HW/evaluation/*/episodes.csv`와 `summary.json`에 있다.
