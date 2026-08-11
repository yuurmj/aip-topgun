# Pursuit 및 WEZ 결과

## Pursuit

- absolute pursuit: crash 4/10, 평균 최저거리 4,801.51m.
- signed closure: 약 5.1km에서 멀어지는 패턴이 지속돼 53 iteration 조기 종료.
- easy acquisition: 평균 최저거리 2,199.10m로 개선됐으나 crash 10/10.

## WEZ

- strict approach는 324.16m까지 접근했지만 ATA 정렬 실패.
- inside hold는 0.3초, 100-iter hold 학습은 탐색 최고 0.8초로 미달.
- 조종 권한 curriculum과 속도 탐색 후 A6에서 성공.

## A6 공식 결과

- bundle: `artifacts/models/team01/sac_mlp_v2_wez_a6_qualified_kill`
- seed 10개: win 10, crash 0, 외부 진입 10, WEZ 1초 10.
- 연속 WEZ 3.3초, 3.3초에 target destroyed, 평균 누적 피해 1.0619.
- 제한: control surface 0%, randomization off, unique trajectory hash 1.

따라서 ‘엄격 WEZ 외부 진입 1초 이상 및 격추’는 acquisition bootstrap에서 달성했지만 일반 기동 전투 성능은 아니다.
