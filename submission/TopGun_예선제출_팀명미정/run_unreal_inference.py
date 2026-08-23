from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent   # Release/ 루트
SRC = ROOT / "src"
RELEASE_ROOT = ROOT
DEFAULT_BT_DLL = RELEASE_ROOT / "AIP_BASE.dll"
DEFAULT_BT_RULE_XML = RELEASE_ROOT / "Rule_forTraining.xml"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from dogfight.ai.bt_action_provider import BTActionProvider
from dogfight.ai.bt_rule_manager import activate_rule_xml
from dogfight.ai.bundle_runtime import build_bundle_runtime
from dogfight.ai.hybrid_action_provider import HybridActionProvider
from dogfight.ai.student_hooks import load_observation_hook
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient

# python run_unreal_inference.py --mode rl --bundle-dir artifacts\models\team01\v1 --team-name team01
# python run_unreal_inference.py --mode bt --team-name team01

def parse_args():
    parser = argparse.ArgumentParser(description="Run RL/BT/Hybrid inference and communicate with the Unreal AI server over UDP.")
    parser.add_argument("--mode", choices=["rl", "bt", "hybrid"], required=True, help="Inference backend to use.")
    parser.add_argument("--server-ip", default="127.0.0.1", help="Unreal server IP address.")
    parser.add_argument("--server-port", type=int, default=9999, help="Unreal server UDP port.")
    parser.add_argument("--team-name", default="FDSA", help="Client team name sent to the Unreal server.")
    parser.add_argument(
        "--team-name-file",
        help=(
            "Optional UTF-8 text file containing the team name. This takes "
            "precedence over --team-name and avoids Windows batch code-page issues."
        ),
    )
    parser.add_argument("--simulation-state", type=int, default=1, help="Heartbeat simulation state value.")
    parser.add_argument("--heartbeat-sec", type=float, default=1.0, help="Heartbeat interval in seconds.")
    parser.add_argument("--command-delay-sec", type=float, default=0.0, help="Delay before replying with CMD after both PlaneInfo packets are ready.")
    parser.add_argument("--recv-timeout-sec", type=float, default=0.2, help="UDP socket receive timeout.")
    parser.add_argument(
        "--action-repeat",
        type=int,
        default=None,
        help=(
            "Number of completed own/enemy PlaneInfo pairs to hold each action. "
            "Defaults to 1 so guidance and CMD update for every pair."
        ),
    )
    parser.add_argument(
        "--debug-action-repeat",
        action="store_true",
        help="Print action-repeat counter, frame indices, update/hold state, and action values.",
    )
    parser.add_argument(
        "--observation-repeat",
        type=int,
        default=None,
        help=(
            "Number of PlaneInfo pairs to reuse each policy observation. "
            "RL default: bundle step_ratio; guidance still updates every pair."
        ),
    )
    parser.add_argument("--packet-monitor", action="store_true", help="Render live RX/TX packet values in the terminal.")
    parser.add_argument(
        "--packet-log",
        default="",
        help="Optional JSONL path that records every received and transmitted UDP packet.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Load and validate the selected policy, print its resolved contract, then exit without opening UDP.",
    )
    parser.add_argument("--packet-monitor-interval-sec", type=float, default=0.2, help="Refresh interval for the live packet monitor.")
    parser.add_argument(
        "--rebase-pair-positions",
        action="store_true",
        help="Align the two BattleServer JSBSim map origins for diagnostic inference.",
    )
    parser.add_argument(
        "--initial-pair-range-m",
        type=float,
        default=762.0,
        help="Initial head-on range used with --rebase-pair-positions (default: 2500 ft).",
    )
    parser.add_argument(
        "--plane-info-velocity-frame",
        choices=["world", "body"],
        default="world",
        help="Interpretation of PlaneInfo velocity; normal legacy path uses world.",
    )
    parser.add_argument(
        "--observation-mode",
        default=None,
        choices=[
            "classic12",
            "relative14",
            "tactical16",
            "tactical17",
            "tactical21",
            "tactical25",
            "custom",
        ],
        help="Override the observation mode saved in the RL bundle.",
    )
    parser.add_argument("--observation-module", default="", help="Optional custom observation module.")
    parser.add_argument("--ownship-force-side", type=int, default=1, help="Force side to use for the ownship in BT inference.")
    parser.add_argument("--target-force-side", type=int, default=2, help="Force side to use for the enemy in BT inference.")
    parser.add_argument(
        "--bt-dll",
        default=str(DEFAULT_BT_DLL),
        help="Behavior tree DLL path for BT inference.",
    )
    parser.add_argument(
        "--bt-rule-xml",
        default=str(DEFAULT_BT_RULE_XML),
        help=(
            "Rule XML source to activate while the client runs. "
            "Use Rule_forTraining.xml by default, or pass a team file such as "
            "Rule_team01.xml."
        ),
    )
    parser.add_argument("--bundle-dir", help="Lightweight RL bundle directory created by train_rllib.py.")
    parser.add_argument("--policy-id", default="default_policy", help="RLlib policy id to load from the lightweight bundle.")
    parser.add_argument("--explore", action="store_true", help="Enable stochastic action sampling for RL inference.")
    parser.add_argument(
        "--direct-mlp",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Use the lightweight Torch-only SAC MLP actor (default). "
            "Pass --no-direct-mlp only for an RLlib compatibility check."
        ),
    )
    parser.add_argument("--hybrid-mode", choices=["residual", "blend", "switch"], default="residual", help="Hybrid action composition strategy.")
    parser.add_argument("--alpha", type=float, default=0.5, help="Blend weight for hybrid blend mode.")
    parser.add_argument("--residual-scale", type=float, default=0.35, help="Residual scaling factor for hybrid residual mode.")
    parser.add_argument(
        "--ai-type",
        choices=["rule", "rl", "sl", "fusion", "etc"],
        default="rl",
        help="AI type announced to the Unreal server in ClientJoinInfo.",
    )
    return parser.parse_args()


def resolve_team_name(args) -> str:
    if args.team_name_file:
        team_name = Path(args.team_name_file).read_text(encoding="utf-8-sig").strip()
    else:
        team_name = str(args.team_name).strip()
    if not team_name:
        raise ValueError("team name must not be empty")
    if "\x00" in team_name:
        raise ValueError("team name must not contain a null character")
    encoded = team_name.encode("utf-8")
    if len(encoded) > 29:
        raise ValueError(
            "team name must fit the server's 29-byte UTF-8 field; "
            f"got {len(encoded)} bytes"
        )
    return team_name


def build_action_provider(args):
    if args.mode == "bt":
        return BTActionProvider(dll_name=args.bt_dll), None

    if args.bundle_dir is None:
        raise ValueError("--bundle-dir is required for rl and hybrid modes")

    runtime = build_bundle_runtime(
        args.bundle_dir,
        for_unreal=True,
        policy_id=args.policy_id,
        explore=args.explore,
        use_direct_mlp=args.direct_mlp,
    )
    rl_provider = runtime.provider

    if args.mode == "rl":
        return rl_provider, runtime

    bt_provider = BTActionProvider(dll_name=args.bt_dll)
    return (
        HybridActionProvider(
            primary_provider=rl_provider,
            secondary_provider=bt_provider,
            mode=args.hybrid_mode,
            alpha=args.alpha,
            residual_scale=args.residual_scale,
        ),
        runtime,
    )


def parse_ai_type(value: str) -> AIType:
    mapping = {
        "rule": AIType.RuleBased,
        "rl": AIType.ReinforcementLearning,
        "sl": AIType.SupervisedLearning,
        "fusion": AIType.Fusion,
        "etc": AIType.etc,
    }
    return mapping[value]


def main():
    args = parse_args()
    team_name = resolve_team_name(args)
    with activate_rule_xml(args.bt_rule_xml, ROOT):
        action_provider, runtime = build_action_provider(args)
        saved_observation_module = runtime.observation_module if runtime else ""
        observation_module = args.observation_module or saved_observation_module
        observation_hook = (
            load_observation_hook(observation_module)
            if observation_module
            else None
        )
        observation_mode = (
            observation_hook["mode"]
            if observation_hook
            else args.observation_mode
            or (runtime.observation_mode if runtime else "relative14")
        )
        action_repeat = (
            args.action_repeat
            if args.action_repeat is not None
            else 1
        )
        observation_repeat = (
            args.observation_repeat
            if args.observation_repeat is not None
            else (runtime.action_repeat if runtime else 1)
        )
        print(
            "[DogFightEnv][Unreal] "
            f"observation_mode={observation_mode} "
            f"action_repeat={action_repeat} "
            f"observation_repeat={observation_repeat} "
            f"bundle_wrappers={'enabled' if runtime else 'n/a'} "
            f"direct_mlp={args.direct_mlp}"
        )
        if args.validate_only:
            print(
                "[DogFightEnv][Unreal] validation OK: "
                f"mode={args.mode} observation={observation_mode} "
                f"action_repeat={action_repeat} simulator_actions="
                f"{runtime.simulator_actions if runtime else True}"
            )
            action_provider.close()
            return
        command_policy = ProviderCommandPolicy(
            action_provider=action_provider,
            observation_mode=observation_mode,
            observation_fn=observation_hook["build_observation"] if observation_hook else None,
            ownship_force_side=args.ownship_force_side,
            target_force_side=args.target_force_side,
            action_repeat=action_repeat,
            observation_repeat=observation_repeat,
            debug_action_repeat=args.debug_action_repeat,
            wez_config=runtime.env_config.get("wez") if runtime else None,
            rebase_pair_positions=args.rebase_pair_positions,
            initial_pair_range_m=args.initial_pair_range_m,
            plane_info_velocity_frame=args.plane_info_velocity_frame,
        )
        client = UnrealAIPilotUDPClient(
            command_policy=command_policy,
            server_ip=args.server_ip,
            server_port=args.server_port,
            team_name=team_name,
            ai_type=parse_ai_type(args.ai_type),
            simulation_state=args.simulation_state,
            heartbeat_interval_sec=args.heartbeat_sec,
            command_delay_sec=args.command_delay_sec,
            recv_timeout_sec=args.recv_timeout_sec,
            enable_terminal_monitor=args.packet_monitor,
            terminal_monitor_interval_sec=args.packet_monitor_interval_sec,
            packet_log_path=args.packet_log or None,
        )

        try:
            client.run()
        finally:
            action_provider.close()


if __name__ == "__main__":
    main()
