"""
[학생 작성 파일] 경진대회 제출 — Unreal 서버 연결
=====================================================
학습한 모델을 경진대회 서버에 연결합니다.
BUNDLE_DIR 경로와 팀 이름을 설정한 뒤 이 파일을 실행하세요.

커맨드라인으로 직접 실행하는 방법 (권장)
-------------------------------------------
  # RL 모델 사용
  python run_unreal_inference.py --mode rl \\
      --bundle-dir artifacts/models/team01/v1 \\
      --team-name team01 \\
      --server-ip <서버IP> --server-port 9999

  # BT만 사용 (모델 없이)
  python run_unreal_inference.py --mode bt \\
      --bt-dll AIP_BASE.dll \\
      --bt-rule-xml Rule_forTraining.xml \\
      --team-name team01 \\
      --server-ip <서버IP>

  # RL + BT 하이브리드
  python run_unreal_inference.py --mode hybrid \\
      --bundle-dir artifacts/models/team01/v1 \\
      --bt-dll AIP_BASE.dll \\
      --bt-rule-xml Rule_팀이름.xml \\
      --hybrid-mode residual --residual-scale 0.35 \\
      --team-name team01 \\
      --server-ip <서버IP>

이 파일에서 직접 실행하려면
----------------------------
  python student/my_submission.py

아래 설정을 수정한 뒤 실행하면 됩니다.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for p in (ROOT, SRC):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.bundle_runtime import build_bundle_runtime
from dogfight.ai.hybrid_action_provider import HybridActionProvider
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient


# =============================================================================
# TODO: 아래 설정을 팀에 맞게 수정하세요.
# =============================================================================

TEAM_NAME = "team01"                               # TODO: 팀 이름
SERVER_IP = "127.0.0.1"                            # 로컬 BattleServer 기본값
SERVER_PORT = 9999

# 사용할 백엔드 모드 선택: "rl" | "bt" | "hybrid"
MODE = "rl"

# RL 모드 설정
BUNDLE_DIR = (
    "artifacts/models/team01/"
    "sac_mlp_v2_submission_a437_actor_a687_runtime"
)                                                   # 최종 선발 후 한 번 더 고정
OBSERVATION_MODE = ""                              # 빈 값이면 bundle metadata 사용
OBSERVATION_MODULE = ""                            # 빈 값이면 bundle metadata 사용
USE_DIRECT_MLP = True                              # Ray 없이 동일 SAC MLP actor 직접 실행

# BT 모드 설정
# - 기본 배포 Rule은 Rule_forTraining.xml입니다.
# - 팀별 BT DLL/XML을 제출하는 경우 파일을 Release 루트에 두고 아래 이름을 바꾸세요.
BT_DLL = "AIP_BASE.dll"
BT_RULE_XML = "Rule_forTraining.xml"  # 예: "Rule_team01.xml"

# Hybrid 모드 설정 (MODE="hybrid" 일 때만 사용)
HYBRID_MODE = "residual"   # "residual" | "blend" | "switch"
RESIDUAL_SCALE = 0.35      # residual 모드 강도 (0~1, 클수록 RL 비중 증가)
ALPHA = 0.5                # blend 모드 비율 (alpha × RL + (1-alpha) × BT)

# 연결 설정
AI_TYPE = AIType.ReinforcementLearning
HEARTBEAT_SEC = 1.0
COMMAND_DELAY_SEC = 0.0
RECV_TIMEOUT_SEC = 0.2
ACTION_REPEAT = 1          # 최종 명령/유도는 매 PlaneInfo pair마다 갱신
OBSERVATION_REPEAT = 6     # 학습과 같이 MLP 관측은 6프레임 간격으로 갱신
DEBUG_ACTION_REPEAT = False


# =============================================================================
# 예시: 학습 결과 확인 (로컬 테스트용 백엔드)
# =============================================================================
# 경진대회 제출 전 로컬에서 결과 확인:
#   python run_local_dogfight.py \\
#       --ownship-backend rl \\
#       --ownship-bundle-dir artifacts/models/team01/v1 \\
#       --target-backend bt \\
#       --save-log


# =============================================================================
# 실행 로직 (수정 불필요)
# =============================================================================

def build_action_provider():
    if MODE == "bt":
        print(f"[{TEAM_NAME}] BT 백엔드 사용: {BT_DLL}")
        return BTActionProvider(dll_name=BT_DLL)

    bundle_path = ROOT / BUNDLE_DIR
    if not bundle_path.exists():
        raise FileNotFoundError(
            f"모델 번들을 찾을 수 없습니다: {bundle_path}\n"
            f"먼저 학습을 완료하고 BUNDLE_DIR 경로를 확인하세요."
        )

    print(f"[{TEAM_NAME}] RL 모델 로드: {bundle_path}")
    runtime = build_bundle_runtime(
        bundle_path,
        for_unreal=True,
        use_direct_mlp=USE_DIRECT_MLP,
    )
    rl_provider = runtime.provider

    if MODE == "rl":
        print(f"[{TEAM_NAME}] RL 전용 모드")
        return rl_provider, runtime

    # hybrid
    bt_provider = BTActionProvider(dll_name=BT_DLL)
    print(f"[{TEAM_NAME}] Hybrid 모드: {HYBRID_MODE} (scale={RESIDUAL_SCALE}, alpha={ALPHA})")
    return (
        HybridActionProvider(
            primary_provider=rl_provider,
            secondary_provider=bt_provider,
            mode=HYBRID_MODE,
            alpha=ALPHA,
            residual_scale=RESIDUAL_SCALE,
        ),
        runtime,
    )


def main():
    print(f"=== {TEAM_NAME} 경진대회 클라이언트 시작 ===")
    print(f"서버: {SERVER_IP}:{SERVER_PORT}")
    print(f"모드: {MODE}")
    if MODE in {"bt", "hybrid"}:
        print(f"BT DLL/XML: {BT_DLL} / {BT_RULE_XML}")

    with activate_rule_xml(BT_RULE_XML, ROOT):
        built = build_action_provider()
        if MODE == "bt":
            action_provider = built
            runtime = None
        else:
            action_provider, runtime = built
        observation_module = OBSERVATION_MODULE or (
            runtime.observation_module if runtime else ""
        )
        observation_hook = (
            load_observation_hook(observation_module)
            if observation_module
            else None
        )
        observation_mode = (
            observation_hook["mode"]
            if observation_hook
            else OBSERVATION_MODE
            or (runtime.observation_mode if runtime else "relative14")
        )
        action_repeat = ACTION_REPEAT
        observation_repeat = (
            runtime.action_repeat if runtime else OBSERVATION_REPEAT
        )
        print(
            f"[{TEAM_NAME}] 관측={observation_mode}, action_repeat={action_repeat}, "
            f"observation_repeat={observation_repeat}, "
            f"direct_mlp={USE_DIRECT_MLP if runtime else False}"
        )
        command_policy = ProviderCommandPolicy(
            action_provider=action_provider,
            observation_mode=observation_mode,
            observation_fn=observation_hook["build_observation"]
            if observation_hook
            else None,
            ownship_force_side=1,
            target_force_side=2,
            action_repeat=action_repeat,
            observation_repeat=observation_repeat,
            debug_action_repeat=DEBUG_ACTION_REPEAT,
            wez_config=runtime.env_config.get("wez") if runtime else None,
        )

        client = UnrealAIPilotUDPClient(
            command_policy=command_policy,
            server_ip=SERVER_IP,
            server_port=SERVER_PORT,
            team_name=TEAM_NAME,
            ai_type=AI_TYPE,
            heartbeat_interval_sec=HEARTBEAT_SEC,
            command_delay_sec=COMMAND_DELAY_SEC,
            recv_timeout_sec=RECV_TIMEOUT_SEC,
            enable_terminal_monitor=True,   # 패킷 모니터 표시
        )

        try:
            client.run()
        finally:
            action_provider.close()
            print(f"[{TEAM_NAME}] 클라이언트 종료")


if __name__ == "__main__":
    main()
