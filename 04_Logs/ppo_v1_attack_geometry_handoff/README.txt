실험명: ppo_v1_attack_geometry
알고리즘: PPO
관측값: tactical16
표적 모드: fixed
반복 횟수: 300 iterations
최대 교전 시간: 120초
에피소드 최대 step: 1200

주요 수정:
- 시간 초과 패널티 추가
- 상대 방향 기수 정렬 보상 추가
- 상대 후방 위치 보상 추가
- Soft WEZ 공격 위치 보상 추가
- 너무 가까이 지나치는 행동 패널티 추가

파일 설명:
- config/ppo_v1_attack_geometry.yaml
  학습에 사용한 전체 설정

- code/my_reward.py
  수정된 PPO 보상 함수

- logs/training_log.csv
  iteration별 학습 결과

- logs/replay_index.csv
  평가 교전 결과

- model/
  최종 모델 실행 및 평가용 bundle

- checkpoint_final/
  PPO 학습 재개용 체크포인트
