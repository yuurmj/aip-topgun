from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import socket
import sys
import threading
import time


SCRIPT_PATH = Path(__file__).resolve()
ROOT = SCRIPT_PATH.parents[1] if SCRIPT_PATH.parent.name == "scripts" else SCRIPT_PATH.parent
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from dogfight.ai.bundle_runtime import build_bundle_runtime
from dogfight.unreal import AIType, ProviderCommandPolicy, UnrealAIPilotUDPClient
from dogfight.unreal.protocol import (
    CLIENT_JOIN_INFO_STRUCT,
    CMD_STRUCT,
    MESSAGE_TYPE_STRUCT,
    PLANE_INFO_STRUCT,
    SET_PLANE_ID_STRUCT,
    MessageType,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run local UDP PlaneInfo-to-CMD submission round trips."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--timeout-sec", type=float, default=10.0)
    parser.add_argument("--pairs", type=int, default=1)
    parser.add_argument("--plane-id", type=int, choices=(0, 1), default=0)
    parser.add_argument("--team-name", default="UDP_VALIDATION")
    parser.add_argument(
        "--reverse-packet-order",
        action="store_true",
        help="Send the opponent PlaneInfo before the assigned plane packet.",
    )
    return parser.parse_args()


def _plane_packet(frame: int, plane_id: int, x: float) -> bytes:
    return PLANE_INFO_STRUCT.pack(
        int(MessageType.MT_PlaneInfo),
        frame,
        plane_id,
        x,
        0.0,
        8000.0,
        0.0,
        0.0,
        0.0,
        250.0,
        0.0,
        0.0,
    )


def main() -> int:
    args = parse_args()
    runtime = build_bundle_runtime(
        args.bundle_dir,
        for_unreal=True,
        use_direct_mlp=True,
    )
    policy = ProviderCommandPolicy(
        action_provider=runtime.provider,
        observation_mode=runtime.observation_mode,
        ownship_force_side=1,
        target_force_side=2,
        action_repeat=1,
        observation_repeat=runtime.action_repeat,
        wez_config=runtime.env_config.get("wez"),
    )

    server = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    server.bind(("127.0.0.1", 0))
    server.settimeout(0.2)
    port = server.getsockname()[1]
    client = UnrealAIPilotUDPClient(
        command_policy=policy,
        server_ip="127.0.0.1",
        server_port=port,
        team_name=args.team_name,
        ai_type=AIType.ReinforcementLearning,
        heartbeat_interval_sec=0.05,
        command_delay_sec=0.0,
        recv_timeout_sec=0.05,
    )
    failures: list[BaseException] = []

    def run_client():
        try:
            client.run()
        except BaseException as exc:  # surfaced in the controlling thread below
            failures.append(exc)

    thread = threading.Thread(target=run_client, daemon=True)
    thread.start()
    deadline = time.monotonic() + args.timeout_sec
    client_addr = None
    join_values = None
    command_values = None
    command_count = 0
    round_trip_ms: list[float] = []
    validation_started = None
    try:
        while time.monotonic() < deadline and join_values is None:
            try:
                packet, remote_addr = server.recvfrom(1024)
            except socket.timeout:
                continue
            if len(packet) < MESSAGE_TYPE_STRUCT.size:
                continue
            client_addr = remote_addr
            if (
                MESSAGE_TYPE_STRUCT.unpack_from(packet)[0]
                == int(MessageType.MT_ClientInfo)
                and len(packet) >= CLIENT_JOIN_INFO_STRUCT.size
            ):
                unpacked_join = CLIENT_JOIN_INFO_STRUCT.unpack(
                    packet[: CLIENT_JOIN_INFO_STRUCT.size]
                )
                join_values = {
                    "team_name": unpacked_join[1]
                    .split(b"\x00", 1)[0]
                    .decode("utf-8"),
                    "ai_type": unpacked_join[2],
                    "plane_id": unpacked_join[3],
                }

        if client_addr is None or join_values is None:
            raise TimeoutError("did not receive the client registration packet")
        expected_team_name = (
            args.team_name.split("\x00", 1)[0]
            .encode("utf-8")[:29]
            .decode("utf-8", errors="ignore")
        )
        if join_values != {
            "team_name": expected_team_name,
            "ai_type": int(AIType.ReinforcementLearning),
            "plane_id": -1,
        }:
            raise ValueError(f"client registration mismatch: {join_values}")

        server.sendto(
            SET_PLANE_ID_STRUCT.pack(
                int(MessageType.MT_SetPlaneID), args.plane_id
            ),
            client_addr,
        )
        if args.pairs <= 0:
            raise ValueError("--pairs must be positive")
        validation_started = time.perf_counter()
        for frame in range(1, args.pairs + 1):
            pair_started = time.perf_counter()
            opponent_id = 1 - args.plane_id
            own_packet = _plane_packet(frame, args.plane_id, 0.0)
            opponent_packet = _plane_packet(frame, opponent_id, 1200.0)
            packets = (
                (opponent_packet, own_packet)
                if args.reverse_packet_order
                else (own_packet, opponent_packet)
            )
            for plane_packet in packets:
                server.sendto(plane_packet, client_addr)
            command_values = None
            while time.monotonic() < deadline and command_values is None:
                if failures:
                    raise RuntimeError("UDP client thread failed") from failures[0]
                try:
                    packet, _ = server.recvfrom(1024)
                except socket.timeout:
                    continue
                if len(packet) < MESSAGE_TYPE_STRUCT.size:
                    continue
                if MESSAGE_TYPE_STRUCT.unpack_from(packet)[0] != int(MessageType.MT_CMD):
                    continue
                unpacked = CMD_STRUCT.unpack(packet[: CMD_STRUCT.size])
                command_values = {
                    "plane_id": unpacked[1],
                    "frame_index": unpacked[2],
                    "roll": unpacked[3],
                    "pitch": unpacked[4],
                    "rudder": unpacked[5],
                    "throttle": unpacked[6],
                }

            if command_values is None:
                if failures:
                    raise RuntimeError("UDP client thread failed") from failures[0]
                raise TimeoutError(
                    f"client did not return CMD packet {frame}/{args.pairs}"
                )
            axes = [
                command_values["roll"],
                command_values["pitch"],
                command_values["rudder"],
                command_values["throttle"],
            ]
            if not all(math.isfinite(value) for value in axes):
                raise ValueError(f"non-finite command: {command_values}")
            if not all(-1.0 <= value <= 1.0 for value in axes[:3]):
                raise ValueError(
                    f"control surface command outside [-1,1]: {command_values}"
                )
            if not 0.0 <= axes[3] <= 1.0:
                raise ValueError(f"throttle outside [0,1]: {command_values}")
            if (
                command_values["plane_id"] != args.plane_id
                or command_values["frame_index"] != frame
            ):
                raise ValueError(f"CMD identity mismatch: {command_values}")
            command_count += 1
            round_trip_ms.append((time.perf_counter() - pair_started) * 1000.0)
    finally:
        client.stop()
        thread.join(timeout=2.0)
        server.close()
        runtime.provider.close()

    if failures:
        raise RuntimeError("UDP client thread failed") from failures[0]
    sorted_latency = sorted(round_trip_ms)

    def percentile(fraction: float) -> float:
        index = min(
            len(sorted_latency) - 1,
            max(0, int(math.ceil(fraction * len(sorted_latency))) - 1),
        )
        return sorted_latency[index]

    elapsed = max(time.perf_counter() - validation_started, 1e-9)
    print(
        json.dumps(
            {
                "status": "ok",
                "observation_mode": runtime.observation_mode,
                "action_repeat": 1,
                "observation_repeat": runtime.action_repeat,
                "assigned_plane_id": args.plane_id,
                "reverse_packet_order": args.reverse_packet_order,
                "team_name": args.team_name,
                "received_registration": join_values,
                "command_count": command_count,
                "elapsed_sec": elapsed,
                "commands_per_sec": command_count / elapsed,
                "round_trip_ms": {
                    "mean": sum(round_trip_ms) / len(round_trip_ms),
                    "p95": percentile(0.95),
                    "p99": percentile(0.99),
                    "max": max(round_trip_ms),
                },
                "command": command_values,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
