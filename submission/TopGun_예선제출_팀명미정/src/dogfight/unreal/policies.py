from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from dogfight.ai.action_provider import ActionProvider
from dogfight.ai.native_bt import AIPilot
from GeoMathUtil import GeometryInfo
from dogfight.ai.action_provider import ActionContext
from dogfight.ai.rl_action_provider import RLActionProvider
from dogfight.envs.observation import build_observation
from dogfight.sim.state_schema import StateIndex
from dogfight.unreal.client import RemoteClientContext
from dogfight.unreal.protocol import CMD


@dataclass
class ConstantCommandPolicy:
    roll_cmd: float = 0.0
    pitch_cmd: float = 0.0
    yaw_cmd: float = 0.0
    throttle_cmd: float = 1.0

    def reset(self, context: RemoteClientContext) -> None:
        return None

    def compute_command(self, context: RemoteClientContext) -> CMD:
        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=self.roll_cmd,
            pitch_cmd=self.pitch_cmd,
            yaw_cmd=self.yaw_cmd,
            throttle_cmd=self.throttle_cmd,
        )


class RLLightweightCommandPolicy:
    def __init__(
        self,
        action_provider: RLActionProvider,
        observation_mode: str = "relative14",
        observation_fn=None,
        wez_config: dict | None = None,
    ):
        self.action_provider = action_provider
        self.observation_mode = observation_mode
        self.observation_fn = observation_fn
        self.wez_config = dict(wez_config) if wez_config else None
        self.geometry = GeometryInfo()
        self.state_tracker = PlaneInfoStateTracker()

    def reset(self, context: RemoteClientContext) -> None:
        self.action_provider.reset(None)
        self.state_tracker.reset()

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(
                plane_id=context.plane_id,
                index=context.frame_index,
                roll_cmd=0.0,
                pitch_cmd=0.0,
                yaw_cmd=0.0,
                throttle_cmd=1.0,
            )

        ownship_state = self.state_tracker.convert(context.own_plane.plane_info)
        target_state = self.state_tracker.convert(context.enemy_plane.plane_info)
        observation = self._build_observation(ownship_state, target_state)

        action_result = self.action_provider.compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship_state,
                target_state=target_state,
                observation=observation,
                info={"frame_index": context.frame_index},
            )
        )
        action = np.asarray(action_result.action, dtype=np.float32)

        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def _build_observation(self, ownship_state, target_state) -> np.ndarray:
        if self.observation_fn is not None:
            return np.asarray(
                self.observation_fn(
                    ownship_state,
                    target_state,
                    self.geometry,
                    None,
                ),
                dtype=np.float32,
            )
        return build_observation(
            self.observation_mode,
            ownship_state,
            target_state,
            self.geometry,
            self.wez_config,
        )


class ProviderCommandPolicy:
    def __init__(
        self,
        action_provider: ActionProvider,
        observation_mode: str = "relative14",
        observation_fn=None,
        ownship_force_side: int = 1,
        target_force_side: int = 2,
        action_repeat: int = 1,
        observation_repeat: int = 1,
        debug_action_repeat: bool = False,
        wez_config: dict | None = None,
    ):
        self.action_provider = action_provider
        self.observation_mode = observation_mode
        self.observation_fn = observation_fn
        self.ownship_force_side = ownship_force_side
        self.target_force_side = target_force_side
        self.action_repeat = max(1, int(action_repeat))
        self.observation_repeat = max(1, int(observation_repeat))
        self.debug_action_repeat = debug_action_repeat
        self.wez_config = dict(wez_config) if wez_config else None
        self.geometry = GeometryInfo()
        self.state_tracker = PlaneInfoStateTracker()
        self._state_pair_count = 0
        self._cached_action: np.ndarray | None = None
        self._cached_observation: np.ndarray | None = None
        self._last_policy_count: int | None = None
        self._last_policy_frame_index: int | None = None

    def reset(self, context: RemoteClientContext) -> None:
        self.action_provider.reset(None)
        self._state_pair_count = 0
        self._cached_action = None
        self._cached_observation = None
        self._last_policy_count = None
        self._last_policy_frame_index = None
        self.state_tracker.reset()

    def compute_command(self, context: RemoteClientContext) -> CMD:
        if context.own_plane.plane_info is None or context.enemy_plane.plane_info is None:
            return CMD(
                plane_id=context.plane_id,
                index=context.frame_index,
                roll_cmd=0.0,
                pitch_cmd=0.0,
                yaw_cmd=0.0,
                throttle_cmd=1.0,
            )

        own_plane = context.own_plane.plane_info
        enemy_plane = context.enemy_plane.plane_info
        pair_count = self._state_pair_count
        self._state_pair_count += 1

        policy_updated = (
            self._cached_action is None
            or pair_count % self.action_repeat == 0
        )
        if policy_updated:
            action = self._compute_provider_action(
                context, own_plane, enemy_plane, pair_count
            )
            self._cached_action = action
            self._last_policy_count = pair_count
            self._last_policy_frame_index = context.frame_index
        else:
            action = np.asarray(self._cached_action, dtype=np.float32)

        if self.debug_action_repeat:
            self._print_action_repeat_debug(
                context=context,
                pair_count=pair_count,
                policy_updated=policy_updated,
                action=action,
            )

        return CMD(
            plane_id=context.plane_id,
            index=context.frame_index,
            roll_cmd=float(action[0]),
            pitch_cmd=float(action[1]),
            yaw_cmd=float(action[2]),
            throttle_cmd=float(action[3]),
        )

    def _compute_provider_action(
        self, context, own_plane, enemy_plane, pair_count: int
    ) -> np.ndarray:
        ownship_state = self.state_tracker.convert(own_plane)
        target_state = self.state_tracker.convert(enemy_plane)
        if (
            self._cached_observation is None
            or pair_count % self.observation_repeat == 0
        ):
            self._cached_observation = self._build_observation(
                ownship_state, target_state
            )
        observation = np.asarray(self._cached_observation, dtype=np.float32)

        own_speed = float(
            np.linalg.norm([own_plane.velocity.x, own_plane.velocity.y, own_plane.velocity.z])
        )
        target_speed = float(
            np.linalg.norm([enemy_plane.velocity.x, enemy_plane.velocity.y, enemy_plane.velocity.z])
        )
        my_plane_data = AIPilot.BuildPlaneData(
            [own_plane.position.x, own_plane.position.y, own_plane.position.z],
            [own_plane.rotation.roll, own_plane.rotation.pitch, own_plane.rotation.yaw],
            own_speed,
            self.ownship_force_side,
        )
        target_plane_data = AIPilot.BuildPlaneData(
            [enemy_plane.position.x, enemy_plane.position.y, enemy_plane.position.z],
            [enemy_plane.rotation.roll, enemy_plane.rotation.pitch, enemy_plane.rotation.yaw],
            target_speed,
            self.target_force_side,
        )

        action_result = self.action_provider.compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship_state,
                target_state=target_state,
                observation=observation,
                info={
                    "frame_index": context.frame_index,
                    "my_plane_id": context.plane_id,
                    "target_plane_id": enemy_plane.plane_id,
                    "my_force_side": self.ownship_force_side,
                    "target_force_side": self.target_force_side,
                    "my_plane_data": my_plane_data,
                    "target_plane_data": target_plane_data,
                },
            )
        )
        return np.asarray(action_result.action, dtype=np.float32)

    def _print_action_repeat_debug(
        self,
        context: RemoteClientContext,
        pair_count: int,
        policy_updated: bool,
        action: np.ndarray,
    ) -> None:
        repeat_offset = pair_count % self.action_repeat
        print(
            "[DogFightEnv][Unreal][ACTION_REPEAT] "
            f"pair_count={pair_count} repeat={self.action_repeat} "
            f"repeat_offset={repeat_offset} policy_updated={policy_updated} "
            f"cmd_frame={context.frame_index} "
            f"own_frame={context.own_plane.frame_index} "
            f"enemy_frame={context.enemy_plane.frame_index} "
            f"policy_frame={self._last_policy_frame_index} "
            f"policy_count={self._last_policy_count} "
            f"action={np.asarray(action, dtype=np.float32).tolist()}"
        )

    def _build_observation(self, ownship_state, target_state) -> np.ndarray:
        if self.observation_fn is not None:
            return np.asarray(
                self.observation_fn(
                    ownship_state,
                    target_state,
                    self.geometry,
                    None,
                ),
                dtype=np.float32,
            )
        return build_observation(
            self.observation_mode,
            ownship_state,
            target_state,
            self.geometry,
            self.wez_config,
        )


class PlaneInfoStateTracker:
    """Restore the training-state fields available in Unreal PlaneInfo.

    Unreal reports Cartesian XYZ with Z up and a world-frame velocity.  The
    local trainer uses NED (Down positive) and body-axis u/v/w.  Health is not
    present in the public PlaneInfo packet, so it defaults to the intact value
    used through most training observations.
    """

    def __init__(self, frame_rate_hz: float = 60.0) -> None:
        self.frame_rate_hz = float(frame_rate_hz)
        self._previous_rotation: dict[int, tuple[int, np.ndarray]] = {}

    def reset(self) -> None:
        self._previous_rotation.clear()

    @staticmethod
    def _wrap_degrees(value: float) -> float:
        return (float(value) + 180.0) % 360.0 - 180.0

    @staticmethod
    def _ned_to_body_rotation(roll_deg: float, pitch_deg: float, yaw_deg: float) -> np.ndarray:
        roll, pitch, yaw = np.deg2rad([roll_deg, pitch_deg, yaw_deg])
        sr, cr = np.sin(roll), np.cos(roll)
        sp, cp = np.sin(pitch), np.cos(pitch)
        sy, cy = np.sin(yaw), np.cos(yaw)
        body_to_ned = np.asarray(
            [
                [cp * cy, sr * sp * cy - cr * sy, cr * sp * cy + sr * sy],
                [cp * sy, sr * sp * sy + cr * cy, cr * sp * sy - sr * cy],
                [-sp, sr * cp, cr * cp],
            ],
            dtype=np.float64,
        )
        return body_to_ned.T

    def _angular_rates(self, plane_info) -> tuple[float, float, float]:
        plane_id = int(plane_info.plane_id)
        frame = int(plane_info.index)
        rotation = np.asarray(
            [
                plane_info.rotation.roll,
                plane_info.rotation.pitch,
                plane_info.rotation.yaw,
            ],
            dtype=np.float64,
        )
        previous = self._previous_rotation.get(plane_id)
        self._previous_rotation[plane_id] = (frame, rotation)
        if previous is None or frame <= previous[0] or self.frame_rate_hz <= 0.0:
            return 0.0, 0.0, 0.0
        dt = (frame - previous[0]) / self.frame_rate_hz
        derivatives = np.asarray(
            [self._wrap_degrees(rotation[i] - previous[1][i]) / dt for i in range(3)],
            dtype=np.float64,
        )
        phi = math.radians(float(rotation[0]))
        theta = math.radians(float(rotation[1]))
        phi_dot, theta_dot, psi_dot = derivatives
        p = phi_dot - psi_dot * math.sin(theta)
        q = theta_dot * math.cos(phi) + psi_dot * math.sin(phi) * math.cos(theta)
        r = -theta_dot * math.sin(phi) + psi_dot * math.cos(phi) * math.cos(theta)
        return float(p), float(q), float(r)

    def convert(self, plane_info) -> np.ndarray:
        state = np.zeros(51, dtype=np.float32)
        state[StateIndex.N] = plane_info.position.x
        state[StateIndex.E] = plane_info.position.y
        state[StateIndex.D] = -plane_info.position.z
        state[StateIndex.ROLL] = plane_info.rotation.roll
        state[StateIndex.PITCH] = plane_info.rotation.pitch
        state[StateIndex.YAW] = plane_info.rotation.yaw % 360.0

        world_velocity_ned = np.asarray(
            [
                plane_info.velocity.x,
                plane_info.velocity.y,
                -plane_info.velocity.z,
            ],
            dtype=np.float64,
        )
        body_velocity = self._ned_to_body_rotation(
            plane_info.rotation.roll,
            plane_info.rotation.pitch,
            plane_info.rotation.yaw,
        ) @ world_velocity_ned
        state[StateIndex.U : StateIndex.W + 1] = body_velocity
        state[StateIndex.P : StateIndex.R + 1] = self._angular_rates(plane_info)
        state[StateIndex.KCAS] = float(np.linalg.norm(world_velocity_ned))
        state[StateIndex.SIM_TIME] = float(plane_info.index) / self.frame_rate_hz
        state[StateIndex.ALT] = max(0.0, float(plane_info.position.z))
        state[StateIndex.HEALTH] = 1.0
        return state


def plane_info_to_state(plane_info) -> np.ndarray:
    """Backward-compatible one-shot PlaneInfo conversion."""

    return PlaneInfoStateTracker().convert(plane_info)
