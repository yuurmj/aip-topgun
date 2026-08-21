from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
ROOT = (
    SCRIPT_PATH.parents[1]
    if SCRIPT_PATH.parent.name == "scripts"
    else SCRIPT_PATH.parent
)
SRC = ROOT / "src"
for candidate in (ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext
from dogfight.ai.bundle_runtime import (
    PolicyToSimulatorActionProvider,
    build_bundle_runtime,
)
from dogfight.envs.observation import build_observation
from dogfight.unreal.client import RemoteClientContext
from dogfight.unreal.policies import PlaneInfoStateTracker, ProviderCommandPolicy
from dogfight.unreal.protocol import PlaneInfo, Rotation3D, Vector3D


def parse_args():
    parser = argparse.ArgumentParser(
        description="Compare local bundle actions with the Unreal CMD path."
    )
    parser.add_argument("--bundle-dir", required=True)
    parser.add_argument("--frames", type=int, default=240)
    parser.add_argument("--atol", type=float, default=1.0e-6)
    return parser.parse_args()


def packet(frame: int, plane_id: int, *, target: bool) -> PlaneInfo:
    time_s = frame / 60.0
    if target:
        yaw = (180.0 + 8.0 * math.sin(time_s * 0.7)) % 360.0
        roll = 65.0 * math.sin(time_s * 1.1)
        pitch = 4.0 * math.cos(time_s * 0.8)
        speed = 235.0
        position = Vector3D(
            1250.0 - 2.5 * frame,
            420.0 * math.cos(time_s * 0.45),
            8100.0 + 80.0 * math.sin(time_s * 0.3),
        )
    else:
        yaw = (12.0 + 10.0 * math.sin(time_s * 0.6)) % 360.0
        roll = 35.0 * math.sin(time_s * 0.9)
        pitch = 3.0 * math.sin(time_s * 0.5)
        speed = 250.0
        position = Vector3D(
            3.0 * frame,
            100.0 * math.sin(time_s * 0.4),
            8000.0 + 60.0 * math.sin(time_s * 0.25),
        )
    yaw_rad = math.radians(yaw)
    velocity = Vector3D(
        speed * math.cos(yaw_rad),
        speed * math.sin(yaw_rad),
        2.0 * math.sin(time_s * 0.5),
    )
    return PlaneInfo(
        index=frame,
        plane_id=plane_id,
        position=position,
        rotation=Rotation3D(roll, pitch, yaw),
        velocity=velocity,
    )


def main() -> int:
    args = parse_args()
    if args.frames <= 0:
        raise ValueError("--frames must be positive")

    local_runtime = build_bundle_runtime(
        args.bundle_dir,
        for_unreal=False,
        use_direct_mlp=True,
    )
    unreal_runtime = build_bundle_runtime(
        args.bundle_dir,
        for_unreal=True,
        use_direct_mlp=True,
    )
    reference_provider = PolicyToSimulatorActionProvider(
        local_runtime.provider,
        local_runtime.env_config,
    )
    command_policy = ProviderCommandPolicy(
        unreal_runtime.provider,
        observation_mode=unreal_runtime.observation_mode,
        action_repeat=1,
        observation_repeat=unreal_runtime.action_repeat,
        wez_config=unreal_runtime.env_config.get("wez"),
    )
    tracker = PlaneInfoStateTracker()
    geometry = GeometryInfo()
    remote = RemoteClientContext(plane_id=0, frame_index=0)
    reference_provider.reset(None)
    command_policy.reset(remote)
    cached_observation = None
    maximum_error = 0.0
    digest = hashlib.sha256()
    reference_ms = []
    unreal_ms = []

    try:
        for frame in range(args.frames):
            own_packet = packet(frame, 0, target=False)
            target_packet = packet(frame, 1, target=True)
            remote.frame_index = frame
            remote.own_plane.update(own_packet)
            remote.enemy_plane.update(target_packet)

            ownship_state = tracker.convert(own_packet)
            target_state = tracker.convert(target_packet)
            if cached_observation is None or frame % unreal_runtime.action_repeat == 0:
                cached_observation = build_observation(
                    unreal_runtime.observation_mode,
                    ownship_state,
                    target_state,
                    geometry,
                    unreal_runtime.env_config.get("wez"),
                )
            started = time.perf_counter()
            expected = reference_provider.compute_action(
                ActionContext(
                    sim=None,
                    opponent_sim=None,
                    ownship_state=ownship_state,
                    target_state=target_state,
                    observation=np.asarray(cached_observation, dtype=np.float32),
                    info={"frame_index": frame},
                )
            ).action
            reference_ms.append((time.perf_counter() - started) * 1000.0)
            started = time.perf_counter()
            command = command_policy.compute_command(remote)
            unreal_ms.append((time.perf_counter() - started) * 1000.0)
            actual = np.asarray(
                [
                    command.roll_cmd,
                    command.pitch_cmd,
                    command.yaw_cmd,
                    command.throttle_cmd,
                ],
                dtype=np.float32,
            )
            error = float(np.max(np.abs(actual - expected)))
            maximum_error = max(maximum_error, error)
            if not np.allclose(actual, expected, atol=args.atol, rtol=0.0):
                raise AssertionError(
                    f"frame {frame}: expected={expected.tolist()} "
                    f"actual={actual.tolist()} max_error={error}"
                )
            digest.update(actual.tobytes())
    finally:
        reference_provider.close()
        unreal_runtime.provider.close()

    def timing(values: list[float]) -> dict:
        ordered = sorted(values)
        maximum_frame = max(range(len(values)), key=values.__getitem__)

        def percentile(fraction: float) -> float:
            index = min(
                len(ordered) - 1,
                max(0, int(math.ceil(fraction * len(ordered))) - 1),
            )
            return ordered[index]

        return {
            "mean": sum(values) / len(values),
            "p95": percentile(0.95),
            "p99": percentile(0.99),
            "max": max(values),
            "max_frame": maximum_frame,
            "over_16_67_ms": sum(value > 16.67 for value in values),
            "over_33_33_ms": sum(value > 33.33 for value in values),
            "over_100_ms": sum(value > 100.0 for value in values),
        }

    print(json.dumps({
        "status": "ok",
        "frames": args.frames,
        "observation_repeat": unreal_runtime.action_repeat,
        "maximum_absolute_error": maximum_error,
        "reference_command_ms": timing(reference_ms),
        "unreal_command_ms": timing(unreal_ms),
        "command_trajectory_sha256": digest.hexdigest(),
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
