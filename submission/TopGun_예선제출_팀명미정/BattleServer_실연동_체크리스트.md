# BattleServer 실연동 체크리스트

1. 같은 PC에서 `01_Work\AIP_LIB\Windows\DogFightViewer.exe`를 실행한다. 이 169KB 파일은 내부의 `DogFightViewer\Binaries\Win64\DogFightViewer-Win64-Shipping.exe`를 부르는 공식 부트스트랩이며, 별도 `BattleServer.exe`는 없다.
2. 뷰어에 표시된 포트가 `9999`인지 확인하고 `OpenServer`를 한 번만 누른다. 다른 포트라면 AI의 `--server-port`도 같은 값으로 바꾼다.
3. `실행_검증.bat`에서 `action_repeat=1`, `observation_repeat=6`, `validation OK`를 확인한다.
4. `명령어.txt`의 `--packet-monitor` 명령으로 AI를 실행한다. 상대 기준 모델 클라이언트도 같은 PC에서 실행한다.
5. 두 클라이언트가 등록되고 plane ID가 배정된 뒤 뷰어의 시나리오 시작 버튼을 누른다.
6. 수신 카운터에서 `MT_SetPlaneID`, 양 기체의 `MT_PlaneInfo`가 계속 증가하는지 확인한다.
7. 송신 `MT_CMD`의 plane_id와 frame index가 수신 프레임에 맞고 throttle이 0~1인지 확인한다.
8. `[Counters]`의 `MT_Damage`가 증가하는지와 `[Last RX]`의 payload byte 수를 기록한다. 공식 payload 구조가 없으므로 임의 해석하지 않는다.
9. 위치 Z가 비행 고도에서 양수인지 확인한다. 음수라면 실행을 중단하고 좌표계 설정을 재확인한다.
10. 기수가 북쪽/동쪽을 향한 장면에서 velocity XYZ가 world-frame인지 확인한다. 현재 제출본 기본값은 Z-up, world-frame velocity이다.
11. 한 라운드를 120초까지 실행해 AI 프로세스 종료·무응답·NaN CMD가 없는지 확인한다.
12. 가능하면 운영진 기준 모델과 번갈아 양 진영에서 최소 3회씩 교전한다.
13. 최종 팀명 반영 후 ZIP을 새 폴더에 풀고 `실행_검증.bat`과 한 라운드를 다시 실행한다.

예상 실행 명령:

```bat
python run_unreal_inference.py --mode rl --bundle-dir model --team-name-file team_name.txt --server-ip 127.0.0.1 --server-port 9999 --packet-monitor
```
