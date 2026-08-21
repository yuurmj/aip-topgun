from __future__ import annotations

import json
import gzip
from pathlib import Path
import pickle
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.bundle_runtime import (
    PolicyToSimulatorActionProvider,
    SavedObservationFeedbackActionProvider,
    build_bundle_runtime,
)
from dogfight.ai.rl_action_provider import (
    CanonicalTargetBankActionRestoreProvider,
    DirectPPOMLPActionProvider,
    RLActionProvider,
)
from dogfight.sim.state_schema import StateIndex
from dogfight.envs.observation import build_observation
from dogfight.unreal.policies import PlaneInfoStateTracker, ProviderCommandPolicy
from dogfight.unreal.client import RemoteClientContext
from dogfight.unreal.protocol import PlaneInfo, Rotation3D, Vector3D
from GeoMathUtil import GeometryInfo


class StaticProvider(ActionProvider):
    def __init__(self, action) -> None:
        self.action = np.asarray(action, dtype=np.float32)
        self.reset_count = 0

    def reset(self, context: ActionContext | None = None) -> None:
        self.reset_count += 1

    def compute_action(self, context: ActionContext) -> ActionResult:
        return ActionResult(
            action=np.array(self.action, copy=True),
            source="rl",
            info={"static": True},
        )


class RecordingProvider(StaticProvider):
    def __init__(self) -> None:
        super().__init__([0.0, 0.0, 0.0, 0.0])
        self.last_context = None

    def compute_action(self, context: ActionContext) -> ActionResult:
        self.last_context = context
        return super().compute_action(context)


def context(*, observation=None, target_roll=0.0) -> ActionContext:
    target = np.zeros(51, dtype=np.float32)
    target[StateIndex.ROLL] = target_roll
    return ActionContext(
        sim=None,
        opponent_sim=None,
        ownship_state=np.zeros(51, dtype=np.float32),
        target_state=target,
        observation=np.asarray(
            np.zeros(25, dtype=np.float32) if observation is None else observation,
            dtype=np.float32,
        ),
    )


class BundleRuntimeTests(unittest.TestCase):
    def test_unreal_plane_info_restores_training_state_contract(self) -> None:
        tracker = PlaneInfoStateTracker(frame_rate_hz=60.0)
        packet = PlaneInfo(
            index=1,
            plane_id=0,
            position=Vector3D(100.0, 200.0, 8000.0),
            rotation=Rotation3D(0.0, 0.0, 90.0),
            velocity=Vector3D(0.0, 250.0, 0.0),
        )
        state = tracker.convert(packet)
        np.testing.assert_allclose(state[:3], [100.0, 200.0, -8000.0])
        np.testing.assert_allclose(state[StateIndex.U : StateIndex.W + 1], [250.0, 0.0, 0.0], atol=1e-5)
        self.assertAlmostEqual(float(state[StateIndex.KCAS]), 250.0)
        self.assertAlmostEqual(float(state[StateIndex.ALT]), 8000.0)
        self.assertAlmostEqual(float(state[StateIndex.HEALTH]), 1.0)

        packet.index = 61
        packet.rotation.yaw = 96.0
        rotated = tracker.convert(packet)
        self.assertAlmostEqual(float(rotated[StateIndex.R]), 6.0, places=4)
        tracker.reset()
        reset = tracker.convert(packet)
        self.assertAlmostEqual(float(reset[StateIndex.R]), 0.0)

    def test_unreal_tactical25_matches_reconstructable_local_features(self) -> None:
        tracker = PlaneInfoStateTracker()
        own_packet = PlaneInfo(
            index=1,
            plane_id=0,
            position=Vector3D(0.0, 0.0, 8000.0),
            rotation=Rotation3D(0.0, 0.0, 0.0),
            velocity=Vector3D(250.0, 0.0, 0.0),
        )
        target_packet = PlaneInfo(
            index=1,
            plane_id=1,
            position=Vector3D(1000.0, 100.0, 8100.0),
            rotation=Rotation3D(0.0, 0.0, 0.0),
            velocity=Vector3D(230.0, 40.0, 0.0),
        )
        own = tracker.convert(own_packet)
        target = tracker.convert(target_packet)
        wez = {"angle_deg": 10.0, "min_range_m": 152.4, "max_range_m": 914.4}
        restored = build_observation("tactical25", own, target, GeometryInfo(), wez)

        expected_own = np.zeros(51, dtype=np.float32)
        expected_own[[StateIndex.D, StateIndex.U, StateIndex.KCAS, StateIndex.ALT, StateIndex.HEALTH]] = [
            -8000.0,
            250.0,
            250.0,
            8000.0,
            1.0,
        ]
        expected_target = np.zeros(51, dtype=np.float32)
        expected_target[[StateIndex.N, StateIndex.E, StateIndex.D]] = [1000.0, 100.0, -8100.0]
        expected_target[StateIndex.U : StateIndex.W + 1] = [230.0, 40.0, 0.0]
        expected_target[StateIndex.KCAS] = np.hypot(230.0, 40.0)
        expected_target[StateIndex.ALT] = 8100.0
        expected_target[StateIndex.HEALTH] = 1.0
        expected = build_observation(
            "tactical25", expected_own, expected_target, GeometryInfo(), wez
        )
        np.testing.assert_allclose(restored, expected)
        self.assertGreater(float(restored[4]), -1.0)  # speed is no longer missing
        self.assertGreater(float(restored[5]), -1.0)  # altitude is no longer missing
        self.assertEqual(float(restored[6]), 1.0)  # intact-health default

    def test_provider_packet_observation_matches_local_training_observation(self) -> None:
        wez = {"angle_deg": 10.0, "min_range_m": 152.4, "max_range_m": 914.4}
        own = np.zeros(51, dtype=np.float32)
        own[[StateIndex.N, StateIndex.E, StateIndex.D]] = [100.0, 200.0, -8000.0]
        own[[StateIndex.ROLL, StateIndex.PITCH, StateIndex.YAW]] = [0.0, 0.0, 90.0]
        own[StateIndex.U] = 250.0
        own[StateIndex.KCAS] = 250.0
        own[StateIndex.ALT] = 8000.0
        own[StateIndex.HEALTH] = 1.0

        target = np.zeros(51, dtype=np.float32)
        target[[StateIndex.N, StateIndex.E, StateIndex.D]] = [1000.0, 100.0, -8100.0]
        target[[StateIndex.ROLL, StateIndex.PITCH, StateIndex.YAW]] = [20.0, 0.0, 0.0]
        target[StateIndex.U] = 230.0
        target[StateIndex.KCAS] = 230.0
        target[StateIndex.ALT] = 8100.0
        target[StateIndex.HEALTH] = 1.0
        expected = build_observation("tactical25", own, target, GeometryInfo(), wez)

        own_packet = PlaneInfo(
            1,
            0,
            Vector3D(100.0, 200.0, 8000.0),
            Rotation3D(0.0, 0.0, 90.0),
            Vector3D(0.0, 250.0, 0.0),
        )
        target_packet = PlaneInfo(
            1,
            1,
            Vector3D(1000.0, 100.0, 8100.0),
            Rotation3D(20.0, 0.0, 0.0),
            Vector3D(230.0, 0.0, 0.0),
        )
        recorder = RecordingProvider()
        policy = ProviderCommandPolicy(
            recorder,
            observation_mode="tactical25",
            wez_config=wez,
        )
        remote = policy_context = RemoteClientContext(plane_id=0, frame_index=1)
        policy_context.own_plane.update(own_packet)
        policy_context.enemy_plane.update(target_packet)
        policy.compute_command(remote)
        np.testing.assert_allclose(recorder.last_context.observation, expected, atol=1e-6)

    def test_provider_can_hold_observation_while_updating_action_each_pair(self) -> None:
        recorder = RecordingProvider()
        policy = ProviderCommandPolicy(
            recorder,
            observation_mode="tactical25",
            action_repeat=1,
            observation_repeat=2,
        )
        remote = RemoteClientContext(plane_id=0, frame_index=0)
        own_packet = PlaneInfo(
            0, 0, Vector3D(0.0, 0.0, 8000.0), Rotation3D(), Vector3D(250.0, 0.0, 0.0)
        )
        target_packet = PlaneInfo(
            0, 1, Vector3D(1000.0, 0.0, 8000.0), Rotation3D(), Vector3D(250.0, 0.0, 0.0)
        )
        observations = []
        for frame, target_north in enumerate((1000.0, 1200.0, 1400.0)):
            own_packet.index = target_packet.index = remote.frame_index = frame
            target_packet.position.x = target_north
            remote.own_plane.update(own_packet)
            remote.enemy_plane.update(target_packet)
            policy.compute_command(remote)
            observations.append(recorder.last_context.observation.copy())

        np.testing.assert_allclose(observations[1], observations[0])
        self.assertFalse(np.allclose(observations[2], observations[1]))

    def test_direct_ppo_mlp_uses_deterministic_mean_and_clips(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir)
            metadata = {"algorithm_config": {"model_config": {"fcnet_activation": "relu"}}}
            (bundle / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
            weights = {
                "encoder.encoder.net.mlp.0.weight": np.zeros((2, 2), dtype=np.float32),
                "encoder.encoder.net.mlp.0.bias": np.zeros(2, dtype=np.float32),
                "encoder.encoder.net.mlp.2.weight": np.zeros((2, 2), dtype=np.float32),
                "encoder.encoder.net.mlp.2.bias": np.zeros(2, dtype=np.float32),
                "pi.net.mlp.0.weight": np.zeros((8, 2), dtype=np.float32),
                "pi.net.mlp.0.bias": np.asarray(
                    [0.25, -2.0, 2.0, 0.75, 9.0, 9.0, 9.0, 9.0],
                    dtype=np.float32,
                ),
            }
            with gzip.open(bundle / "policy_weights.pkl.gz", "wb") as handle:
                pickle.dump(weights, handle)
            provider = DirectPPOMLPActionProvider(str(bundle))
            result = provider.compute_action(context(observation=[1.0, -1.0]))
            np.testing.assert_allclose(result.action, [0.25, -1.0, 1.0, 0.75])
            self.assertTrue(result.info["direct_ppo_mlp"])

    def test_unreal_observation_uses_bundle_wez_config(self) -> None:
        ownship = np.zeros(51, dtype=np.float32)
        target = np.zeros(51, dtype=np.float32)
        ownship[StateIndex.ALT] = target[StateIndex.ALT] = 7000.0
        ownship[StateIndex.HEALTH] = target[StateIndex.HEALTH] = 1.0
        ownship[StateIndex.KCAS] = target[StateIndex.KCAS] = 300.0
        ownship[StateIndex.U] = target[StateIndex.U] = 300.0
        target[StateIndex.N] = 500.0
        wez = {"angle_deg": 10.0, "min_range_m": 152.4, "max_range_m": 914.4}
        provider = StaticProvider([0, 0, 0, 0])
        without_wez = ProviderCommandPolicy(
            provider, observation_mode="tactical25"
        )._build_observation(ownship, target)
        with_wez = ProviderCommandPolicy(
            provider,
            observation_mode="tactical25",
            wez_config=wez,
        )._build_observation(ownship, target)
        self.assertEqual(float(without_wez[15]), -1.0)
        self.assertEqual(float(with_wez[15]), 1.0)

    def test_rl_provider_can_defer_throttle_mapping(self) -> None:
        provider = RLActionProvider.__new__(RLActionProvider)
        provider.policy_id = "default_policy"
        provider.explore = False
        provider.obs_builder = None
        provider.confidence = 0.9
        provider.remap_throttle = False
        provider._compute_module_action = lambda observation: np.asarray(
            [0.1, -0.2, 0.3, -0.8], dtype=np.float32
        )

        raw = provider.compute_action(context()).action
        np.testing.assert_allclose(raw, [0.1, -0.2, 0.3, -0.8])

        provider.remap_throttle = True
        mapped = provider.compute_action(context()).action
        np.testing.assert_allclose(mapped, [0.1, -0.2, 0.3, 0.1])

    def test_final_mapping_converts_throttle_exactly_once(self) -> None:
        provider = PolicyToSimulatorActionProvider(
            StaticProvider([0.5, -0.5, 0.25, -0.8]),
            {"control_surface_scale": 0.5},
        )
        result = provider.compute_action(context())
        np.testing.assert_allclose(result.action, [0.25, -0.25, 0.125, 0.1])
        self.assertEqual(result.source, "sim_command")
        self.assertTrue(result.info["simulator_action_mapped"])

    def test_canonical_restore_preserves_negative_policy_throttle(self) -> None:
        provider = CanonicalTargetBankActionRestoreProvider(
            StaticProvider([0.4, 0.2, -0.3, -0.7])
        )
        action = provider.compute_action(context(target_roll=-45.0)).action
        np.testing.assert_allclose(action, [-0.4, 0.2, 0.3, -0.7])

    def test_saved_feedback_latches_and_resets_branch(self) -> None:
        config = {
            "enabled": True,
            "observation_index": 0,
            "branch_observation_index": 1,
            "latch_branch": True,
            "branches": [
                {"max_value": 0.0, "action_gains": [1, 0, 0, 0]},
                {"max_value": None, "action_gains": [2, 0, 0, 0]},
            ],
        }
        base = StaticProvider([0, 0, 0, -1])
        provider = SavedObservationFeedbackActionProvider(base, config)
        first = provider.compute_action(context(observation=[0.25, -1.0])).action
        second = provider.compute_action(context(observation=[0.25, 1.0])).action
        np.testing.assert_allclose(first, [0.25, 0, 0, -1])
        np.testing.assert_allclose(second, [0.25, 0, 0, -1])
        provider.reset()
        third = provider.compute_action(context(observation=[0.25, 1.0])).action
        np.testing.assert_allclose(third, [0.5, 0, 0, -1])

    def test_runtime_resolves_saved_observation_and_step_ratio(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            bundle = Path(temp_dir) / "bundle"
            bundle.mkdir()
            metadata = {
                "algorithm_config": {
                    "env_config": {
                        "observation_mode": "tactical25",
                        "step_ratio": 6,
                        "control_surface_scale": 1.0,
                    }
                },
                "metadata": {"use_lstm_sac": False},
            }
            (bundle / "metadata.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )
            fake = StaticProvider([0, 0, 0, -1])
            with patch(
                "dogfight.ai.bundle_runtime.DirectSACMLPActionProvider",
                return_value=fake,
            ):
                runtime = build_bundle_runtime(
                    bundle,
                    for_unreal=True,
                    use_direct_mlp=True,
                )
            self.assertEqual(runtime.observation_mode, "tactical25")
            self.assertEqual(runtime.action_repeat, 6)
            self.assertTrue(runtime.simulator_actions)
            np.testing.assert_allclose(
                runtime.provider.compute_action(context()).action,
                [0, 0, 0, 0],
            )


if __name__ == "__main__":
    unittest.main()
