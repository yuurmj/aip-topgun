# 2026 AI Pilot Top Gun Challenge 예선 제출본

로컬 BattleServer에 UDP로 접속하는 SAC MLP 기반 AI 클라이언트 프로젝트입니다. 실행 파일 대신 운영진 제공 Python 환경에서 실행하는 프로젝트 ZIP 형태입니다.

제출 기한은 2026-08-24 24:00이며, 공지의 프로젝트 ZIP 제출 방식과 파일당
1GiB 이하 조건을 따릅니다.

## 실행 순서

1. 운영진 제공 BattleServer를 같은 Windows PC에서 실행하고 `OpenServer`를 누릅니다.
2. 팀명이 확정되면 UTF-8 `team_name.txt`의 한 줄을 실제 팀명으로 바꿉니다.
3. `실행_검증.bat`을 실행해 모델과 추론 계약이 정상 로드되는지 확인합니다.
4. `AI_실행.bat`을 실행합니다.
5. 패킷 문제가 있으면 `명령어.txt`의 `--packet-monitor` 명령으로 수신 좌표와 송신 CMD를 확인합니다.

기본 접속 주소는 `127.0.0.1:9999`입니다. 안내받은 주소가 다르면 `--server-ip`와 `--server-port`만 바꿉니다.
두 배치 파일은 활성 conda 환경과 사용자 폴더의 `anaconda3\envs\aip`,
`miniconda3\envs\aip`를 찾아 NumPy/PyTorch가 있는 Python을 선택합니다.

## 정상 검증 출력

```text
observation_mode=tactical25 action_repeat=1 observation_repeat=6 bundle_wrappers=enabled direct_mlp=True
validation OK
```

`action_repeat=1`은 최종 유도와 CMD를 매 PlaneInfo pair마다 갱신한다는 뜻입니다. `observation_repeat=6`은 학습 당시와 같이 MLP 관측을 6프레임 동안 유지한다는 뜻입니다.

## 구성

- `run_unreal_inference.py`: 경기 실행 진입점
- `model/`: 최종 lightweight SAC bundle
- `src/dogfight/`: UDP 프로토콜, 관측 복원, 정책 및 안전 래퍼
- `GeoMathUtil.py`: 제공 환경의 기하 계산 모듈
- `validate_unreal_udp_roundtrip.py`: 자체 mock server와 CMD 왕복 검사
- `validate_submission_contract.py`: 로컬 정책 행동과 Unreal CMD 프레임별 동일성 검사
- `verify_manifest.py`: 제출 파일 크기와 SHA-256 무결성 검사
- `Rule_forTraining.xml`: 제공 규칙 자산
- `명령어.txt`: 운영진용 실행 명령
- `알고리즘_개요.txt`: 모델 및 제어 주기 설명
- `팀_소개서_팀명미정.txt`: 규정집상 예선 필수 팀 소개서 템플릿
- `검증결과.md`: 제출 전 평가 결과와 알려진 제한
- `BattleServer_실연동_체크리스트.md`: 실제 서버 최종 점검표
- `MANIFEST_SHA256.csv`: 제출 파일 무결성 해시

## 주의

- `team_name.txt`의 첫 줄은 제출 전에 반드시 실제 팀명으로 변경해야 합니다. 서버 필드 제한 때문에 UTF-8 기준 29바이트 이하여야 합니다.
- `팀_소개서_팀명미정.txt`의 팀명·팀번호·소속·팀원 자리표시자도 제출 전에 채워야 합니다.
- 서버 팀명 필드는 UTF-8 29바이트까지이므로 한글만 쓰면 9자 이내를 권장합니다. 공백이 있는 팀명도 따옴표 처리되어 실행됩니다.
- CPU 추론용이며 GPU, Ray/RLlib, Gymnasium은 경기 실행 경로에서 필요하지 않습니다.
- `--no-direct-mlp` 옵션은 제출 실행에서 사용하지 않습니다.
- BattleServer와 AI는 로컬 호스트 IP로 통신하므로 같은 PC에서 실행합니다.
- UDP 상태는 같은 frame index의 양 기체 패킷이 모였을 때만 한 번 처리하므로 지연·중복 패킷을 잘못 결합하지 않습니다.
