# Reward에서 사용할 값 목록

## 현재 Reward 함수에 직접 전달되는 값

| 범주 | 값 | 용도 |
|---|---|---|
| Ownship 위치 | N, E, D | 상대 위치·거리·고도 차 계산 |
| Ownship 자세 | Roll, Pitch, Yaw | 급강하, 과도한 bank, 기수 정렬 평가 |
| Ownship 속도 | KCAS | 실속·과속·에너지 상태 평가 |
| Ownship 자원 | Fuel | 연료 고갈 종료 및 효율성 |
| Ownship 시간 | SIM_TIME | episode 진행 시간 |
| Ownship 지리 | Latitude, Longitude, Altitude | 로그·고도 안전 |
| Ownship 상태 | Health | 패배·피해량 |
| Target state | Ownship과 같은 state 항목 | 상대 자세·고도·속도·체력 |
| 피해 | `ownship_damage`, `target_damage` | step 구간 피해 교환 우위 |
| Geometry | distance | 접근·이탈·유효 사거리 |
| Geometry | ATA | ownship 기수와 target 방향 차이 |
| Geometry | AA | target aspect |
| Geometry | HCA | 두 기체 heading 관계 |
| Geometry | LOS azimuth/elevation | 상대 방향 |
| WEZ 설정 | angle, min range, max range | 실제 공격 가능 조건 |
| 종료 상태 | `terminated`, `truncated` | terminal 처리 |
| 종료 이유 | `end_condition` | 승리·피격·추락·timeout 분리 |

안정적으로 문서화된 state index는 `N/E/D`, `ROLL/PITCH/YAW`, `KCAS`, `FUEL`, `SIM_TIME`, `LAT/LON/ALT`, `HEALTH`다.

## tactical16 관측에 이미 들어 있는 값

1. ownship roll
2. ownship pitch
3. ownship yaw
4. ownship speed
5. ownship altitude
6. ownship health
7. target delta N
8. target delta E
9. target delta D
10. ATA
11. AA
12. LOS azimuth
13. LOS elevation
14. target health
15. in-WEZ flag
16. pursuit score

## 현재 함수에 직접 없는 값

다음 값은 유용하지만 현재 `student.my_reward.compute_reward` 인자로 직접 들어오지 않는다.

| 값 | 필요한 이유 | 확보 방법 |
|---|---|---|
| 이전 step 거리 | 접근/이탈 progress | 환경에서 previous geometry 저장 후 전달 |
| 이전 step ATA/AA | 정렬 개선 progress | 환경에서 previous geometry 저장 후 전달 |
| 이전 step 고도 | 상승·하강 추세 | 이전 state 또는 vertical speed 전달 |
| 현재·이전 action | 행동 크기·smoothness | reward contract에 action/previous_action 추가 |
| episode 누적 WEZ | 최초 진입/유지 구분 | 환경 누적 metric 전달 |
| merge 통과 여부 | overshoot 판정 | 거리 최소점·ATA 변화 상태 저장 |

module global 변수로 이전 값을 저장하면 여러 environment runner에서 섞일 수 있으므로 권장하지 않는다. 환경 instance가 이전 geometry/action을 보관해 Reward 함수로 전달하는 방식이 안전하다.

## v1 수정에 사용할 초기 숫자

아래 값은 첫 smoke experiment용 확정 시작점이다. 최대 600 RL step 기준으로 component 누적값을 확인한 뒤 한 번에 하나씩 조정한다.

### 공통 terminal/damage

| 이름 | 시작값 | 적용 |
|---|---:|---|
| `step_penalty` | -0.01/step | SAC/PPO |
| `win_reward` | +100 | SAC/PPO |
| `loss_reward` | -100 | SAC/PPO |
| `ownship_crash_reward` | -120 | SAC/PPO |
| `target_crash_reward` | +100 | SAC/PPO |
| `timeout_reward` | -30 | SAC/PPO |
| `draw_reward` | -10 | SAC/PPO |
| `guard_fail_penalty` | -100 | SAC/PPO |
| `damage_scale` | 20 | SAC/PPO |

### SAC 비행 안전

| 이름 | 시작값 | 정의 |
|---|---:|---|
| `altitude_warning_m` | 2,000 m | 안전 shaping 시작 고도 |
| `altitude_floor_m` | 300 m | crash 종료 고도 |
| `altitude_risk_scale` | 0.10/step | 낮아질수록 제곱 증가 |
| `dive_penalty_scale` | 0.05/step | 저고도 + 음수 Pitch |
| `excess_bank_deg` | 60° | bank 추가 페널티 시작 |
| `excess_bank_scale` | 0.02/step | 과도 bank 억제 |
| `pursuit_scale` | 0.10/step | safety gate 적용 후 사용 |

권장 altitude risk:

`risk = clip((2000 - altitude) / 1700, 0, 1)`

`r_altitude = -0.10 * risk²`

`r_dive = -0.05 * risk * clip((-pitch) / 45, 0, 1)`

전투 보상에는 `safe_gate = 1 - risk`를 곱해 급강하 중 pursuit 보상으로 안전 페널티를 상쇄하지 못하게 한다.

### PPO 교전·재획득

| 이름 | 시작값 | 정의 |
|---|---:|---|
| `pursuit_scale` | 0.10/step | 기존 0.30에서 축소 |
| `ata_progress_scale` | 0.05/step | 이전 step보다 ATA 개선 |
| `range_progress_scale` | 0.05/step | WEZ 중앙 거리 오차 개선 |
| `wez_hold_bonus` | +0.05/step | 실제 WEZ 조건 만족 |
| `overshoot_penalty` | -0.05/step | ATA > 90°이며 거리 증가 |
| `escape_penalty_scale` | 0.02/step | 3 km 밖 이탈 거리 증가 |
| `timeout_reward` | -30 | truncated timeout 명시 |

WEZ 중앙 목표 거리는 `(152.4 + 914.4) / 2 = 533.4 m`로 둔다. 단순히 거리를 0으로 줄이는 보상 대신 `abs(distance - 533.4)`의 감소를 보상해야 과도한 근접 통과를 줄일 수 있다.

## 반드시 로그로 남길 Reward component

- step
- altitude
- dive
- bank
- pursuit
- ATA progress
- range progress
- WEZ hold
- overshoot
- escape
- damage
- terminal

episode마다 각 component 합계를 비교해 shaping 합계가 win/loss/crash terminal을 압도하지 않는지 확인한다.
