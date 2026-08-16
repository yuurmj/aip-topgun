# SAC/PPO 역할과 공통 계약

## 권장 역할

- SAC: 연속 조종, replay 재사용, reward/curriculum 탐색, pursuit·WEZ acquisition에 사용.
- PPO: on-policy 안정성이 필요한 최종 full-dogfight fine-tuning과 팀 기준선 비교에 사용.
- 현재 결과만으로 PPO가 더 낫다고 증명된 것은 아니다. 동일 관측·reward·seed 계약으로 비교해야 한다.

## 반드시 공통으로 맞출 것

- terminal: win/loss/crash/timeout/draw 표와 end-condition mapping.
- damage: 동일 WEZ 152.4–914.4m, ATA ±1도, 60Hz 공식.
- safety: 2,000m warning, altitude/dive/bank 정의.
- 평가: seed 1001–1010, deterministic, 같은 target/scenario, 같은 60초 제한.
- 보고: crash, survival, WEZ 최대연속, 외부진입, 누적피해, 자체 사격 win을 분리.

알고리즘 성능 비교 전에 이 공통 계약을 코드 한 곳으로 추출하는 것이 다음 우선순위다.
