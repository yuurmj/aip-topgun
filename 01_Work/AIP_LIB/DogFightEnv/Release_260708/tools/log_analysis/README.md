# 교전 로그 자동 분석기

SAC/PPO 교전 결과의 `summary.json`과 비행 CSV를 같은 기준으로 분석한다.
시뮬레이터 DLL이나 외부 Python 라이브러리가 필요하지 않아 macOS와 Windows에서 모두 실행할 수 있다.

## 분석 항목

- 자체 사격 격추, 상대 추락, 아군 추락, timeout 구분
- 가한 피해와 받은 피해
- 최소 거리, 최저 고도, 최대 Roll/Pitch
- 공식 WEZ 기록
- 공식 기록이 없을 때 CSV 기반 거리, ATA, WEZ 진입 및 유지 시간 근사 계산

## 한 모델 분석

`01_Work/AIP_LIB/DogFightEnv/Release_260708` 폴더에서 실행한다.

```bash
cd 01_Work/AIP_LIB/DogFightEnv/Release_260708
```

Windows:

```bash
python tools/log_analysis/analyze_dogfight_logs.py "PPO_V14=C:\logs\ppo_mlp_v14_stability_entry" --output "../../../../04_Logs/Analysis/PPO_V14"
```

macOS:

```bash
python3 tools/log_analysis/analyze_dogfight_logs.py \
  "PPO_V14=/Users/name/Downloads/ppo_mlp_v14_stability_entry" \
  --output "../../../../04_Logs/Analysis/PPO_V14"
```

## SAC/PPO 비교

```bash
python tools/log_analysis/analyze_dogfight_logs.py "SAC_A6=C:\logs\sac_a6" "PPO_V14=C:\logs\ppo_v14" --output "../../../../04_Logs/Analysis/SAC_A6_vs_PPO_V14"
```

## 생성 파일

| 파일 | 내용 |
| --- | --- |
| `evaluation_results.csv` | 경기별 상세 결과 |
| `evaluation_summary.json` | 자동 처리용 집계 결과 |
| `evaluation_summary.md` | 노션에 바로 붙일 수 있는 비교표 |

분석 결과는 `04_Logs/Analysis/실험명/`에 저장한다. 소스 코드는 `01_Work`에만 두고 `04_Logs`에는 생성 결과만 보관한다.

## 테스트

`Release_260708` 폴더에서 실행한다.

```bash
python -m unittest discover -s tests -v
```

## 기본 판정 규칙

| 종료 조건 | 분석 결과 |
| --- | --- |
| `target destroyed` | 자체 사격 격추 |
| `target altitude below min` | 상대 추락 |
| `ownship altitude below min` | 아군 추락 |
| `max time out` | 시간 초과 |

CSV로 계산한 ATA와 WEZ는 로그에 기록된 위치와 자세를 이용한 근사값이다. 공식 `ep_wez_steps`와 `wez_entered`가 있으면 최종 판단에서는 공식 값을 우선한다.
