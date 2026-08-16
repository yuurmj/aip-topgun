from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from dogfight.sim.state_schema import StateIndex
from dogfight.envs.single_agent_env import DogFightEnv
from dogfight.envs.observation import build_observation, observation_size
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.checkpoint_io import _apply_policy_weights
from dogfight.ai.hybrid_action_provider import (
    PursuitAutopilotActionProvider,
    PursuitGuidanceActionProvider,
    GuidanceResidualActionProvider,
    GeometryThenActionProvider,
    WezThenActionProvider,
    TargetAltitudeGuardActionProvider,
)
from dogfight.ai.rl_action_provider import (
    AltitudeRecoveryActionProvider,
    BundleMappedDirectSACActionProvider,
    ObservationFeedbackActionProvider,
    NegativeTargetBankCanonicalActionProvider,
    TargetBankActionScaleProvider,
)
from scripts.evaluate_seeded_bundle import summarize
from student.sac_flight_safety_reward import MY_REWARD_CONFIG, compute_reward
from student.sac_flight_safety_reward_v2 import (
    MY_REWARD_CONFIG as V2_REWARD_CONFIG,
    compute_reward as compute_reward_v2,
)
from student.sac_pursuit_reward_v3 import (
    _body_velocity_to_ned,
    _closure_rate_mps,
)
from student.sac_wez_hold_reward_v4 import (
    MY_REWARD_CONFIG as V4_REWARD_CONFIG,
    compute_reward as compute_reward_v4,
)
from student.sac_moving_target_reward_v5 import (
    MY_REWARD_CONFIG as V5_REWARD_CONFIG,
    compute_reward as compute_reward_v5,
)
from student.sac_wez_range_reward_v6 import (
    MY_REWARD_CONFIG as V6_REWARD_CONFIG,
    compute_reward as compute_reward_v6,
)
from student.sac_wez_sync_reward_v7 import (
    MY_REWARD_CONFIG as V7_REWARD_CONFIG,
    compute_reward as compute_reward_v7,
)


class DummyGeometry:
    def __init__(self, distance=500.0, ata=0.0):
        self.distance = distance
        self.ata = ata

    def _get_distance(self, ownship_state, target_state):
        return self.distance

    def _get_antenna_train_angle(self, ownship_state, target_state, degrees):
        return self.ata


class ConstantActionProvider(ActionProvider):
    def __init__(self, action):
        self.action = np.asarray(action, dtype=np.float32)
        self.reset_count = 0
        self.close_count = 0

    def reset(self, context=None):
        self.reset_count += 1

    def compute_action(self, context):
        return ActionResult(action=self.action.copy(), source="rl")

    def close(self):
        self.close_count += 1


class RecordingActionProvider(ConstantActionProvider):
    def compute_action(self, context):
        self.last_observation = np.asarray(context.observation).copy()
        return super().compute_action(context)


class NegativeTargetBankCanonicalProviderTests(unittest.TestCase):
    def test_mirrors_tactical25_observation_and_roll_rudder_action(self):
        primary = RecordingActionProvider([0.25, -0.5, 0.75, 0.1])
        provider = NegativeTargetBankCanonicalActionProvider(primary)
        observation = np.arange(25, dtype=np.float32)
        target = np.zeros(46, dtype=np.float64)
        target[StateIndex.ROLL] = -45.0

        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            target_state=target,
            observation=observation,
        ))

        mirrored = observation.copy()
        mirrored[[0, 2, 8, 12, 17, 18, 21, 22, 23, 24]] *= -1.0
        np.testing.assert_allclose(primary.last_observation, mirrored)
        np.testing.assert_allclose(result.action, [-0.25, -0.5, -0.75, 0.1])


class AltitudeRecoveryProviderTests(unittest.TestCase):
    @staticmethod
    def context(altitude):
        ownship = np.zeros(46, dtype=np.float64)
        ownship[StateIndex.ALT] = altitude
        return ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            observation=np.zeros(16),
        )

    def test_switches_with_hysteresis_and_biases_primary_only(self):
        primary = ConstantActionProvider([0.1, 0.2, 0.3, 0.4])
        recovery = ConstantActionProvider([-0.1, -0.2, -0.3, 0.6])
        provider = AltitudeRecoveryActionProvider(
            primary,
            recovery,
            trigger_altitude_m=3000.0,
            release_altitude_m=4500.0,
            primary_bias=np.asarray([0.01, 0.02, -0.03, 0.0]),
        )

        np.testing.assert_allclose(
            provider.compute_action(self.context(5000.0)).action,
            [0.11, 0.22, 0.27, 0.4],
        )
        np.testing.assert_allclose(
            provider.compute_action(self.context(2999.0)).action,
            recovery.action,
        )
        np.testing.assert_allclose(
            provider.compute_action(self.context(4000.0)).action,
            recovery.action,
        )
        np.testing.assert_allclose(
            provider.compute_action(self.context(4500.0)).action,
            [0.11, 0.22, 0.27, 0.4],
        )

        provider.reset()
        provider.close()
        self.assertEqual(primary.reset_count, 1)
        self.assertEqual(recovery.reset_count, 1)
        self.assertEqual(primary.close_count, 1)
        self.assertEqual(recovery.close_count, 1)


class LightweightBundleLoadingTests(unittest.TestCase):
    def test_new_api_warm_start_updates_learner_and_env_runner(self):
        calls = []

        class LearnerGroup:
            def set_weights(self, weights):
                calls.append(("learner", weights))

        class EnvRunner:
            def set_state(self, state):
                calls.append(("env_runner", state))

        class Module:
            def set_state(self, state):
                calls.append(("module", state))

        class Algorithm:
            learner_group = LearnerGroup()
            env_runner = EnvRunner()

            def get_module(self, policy_id=None):
                return Module()

            def get_policy(self, policy_id):
                return None

        weights = {"actor": np.asarray([1.0], dtype=np.float32)}
        self.assertTrue(_apply_policy_weights(Algorithm(), weights))
        self.assertEqual([name for name, _ in calls], [
            "learner", "env_runner", "module"
        ])
        self.assertIs(calls[0][1]["default_policy"], weights)
        self.assertIs(calls[1][1]["rl_module"], weights)
        self.assertIs(calls[2][1], weights)


class InitialGeometryRandomizationTests(unittest.TestCase):
    class Fighter:
        _origin_lat = 37.0
        _origin_lon = 128.0
        _origin_alt = 0.0

        def __init__(self, n, e, d):
            self._init_pos_n = n
            self._init_pos_e = e
            self._init_pos_d = d

    def test_distance_range_changes_only_baseline_separation(self):
        env = object.__new__(DogFightEnv)
        env._sim = self.Fighter(1000.0, 20.0, -7000.0)
        env._target_sim = self.Fighter(1950.0, 20.0, -7000.0)
        env.np_random = np.random.default_rng(7)

        sampled = env.randomize_initial_distance([1000.0, 1000.0])

        self.assertEqual(sampled, 1000.0)
        self.assertAlmostEqual(env._sim._init_pos_n, 950.0)
        self.assertAlmostEqual(env._sim._init_pos_e, 20.0)
        self.assertAlmostEqual(env._sim._init_pos_d, -7000.0)

    def test_distance_range_rejects_invalid_values(self):
        env = object.__new__(DogFightEnv)
        with self.assertRaises(ValueError):
            env.randomize_initial_distance([1000.0, 900.0])

    def test_target_speed_range_changes_only_target_speed(self):
        env = object.__new__(DogFightEnv)
        env._target_sim = self.Fighter(1950.0, 0.0, -7000.0)
        env._target_sim._init_speed = 200.0
        env.np_random = np.random.default_rng(7)

        sampled = env.randomize_target_speed([180.0, 200.0])

        self.assertGreaterEqual(sampled, 180.0)
        self.assertLessEqual(sampled, 200.0)
        self.assertEqual(env._target_sim._init_speed, sampled)

    def test_target_speed_range_rejects_invalid_values(self):
        env = object.__new__(DogFightEnv)
        with self.assertRaises(ValueError):
            env.randomize_target_speed([200.0, 180.0])

    def test_ref_old_speed_advantage_changes_only_ownship_speed_delta(self):
        baseline = object.__new__(DogFightEnv)
        baseline._sim = self.Fighter(1000.0, 20.0, -7000.0)
        baseline._target_sim = self.Fighter(1950.0, 20.0, -7000.0)
        baseline._sim._init_speed = 240.0
        baseline._target_sim._init_speed = 240.0
        baseline.np_random = np.random.default_rng(7)

        advantaged = object.__new__(DogFightEnv)
        advantaged._sim = self.Fighter(1000.0, 20.0, -7000.0)
        advantaged._target_sim = self.Fighter(1950.0, 20.0, -7000.0)
        advantaged._sim._init_speed = 240.0
        advantaged._target_sim._init_speed = 240.0
        advantaged.np_random = np.random.default_rng(7)

        randomization = {
            "shared_n_m": 0.0,
            "shared_e_m": 0.0,
            "shared_d_m": 0.0,
            "target_distance_n_m": 0.0,
            "speed_mps": 50.0,
        }
        baseline._add_ref_old_shared_random_starting_position(randomization)
        advantaged._add_ref_old_shared_random_starting_position({
            **randomization,
            "ownship_speed_advantage_mps": 25.0,
        })

        self.assertAlmostEqual(
            advantaged._sim._init_speed - baseline._sim._init_speed, 25.0
        )
        self.assertAlmostEqual(
            advantaged._target_sim._init_speed,
            baseline._target_sim._init_speed,
        )

    def test_subdegree_heading_jitter_is_not_truncated_to_zero(self):
        env = object.__new__(DogFightEnv)
        env._sim = self.Fighter(1000.0, 0.0, -7000.0)
        env._target_sim = self.Fighter(1950.0, 0.0, -7000.0)
        env._sim._init_roll = 0.0
        env._sim._init_pitch = 0.0
        env._sim._init_heading = 0.0
        env.np_random = np.random.default_rng(7)

        env.add_random_init_position(
            radius=0.0, r_roll=0.0, r_pitch=0.0, r_heading=0.5
        )

        self.assertNotEqual(env._sim._init_heading, 0.0)
        self.assertLessEqual(0.0, env._sim._init_heading)
        self.assertLess(env._sim._init_heading, 360.0)

    def test_episode_seed_cycle_balances_training_geometries(self):
        env = object.__new__(DogFightEnv)
        env.config = {"episode_seed_cycle": [1001, 1008]}

        self.assertEqual(env._resolve_episode_seed(None), 1001)
        self.assertEqual(env._resolve_episode_seed(None), 1008)
        self.assertEqual(env._resolve_episode_seed(None), 1001)

    def test_explicit_episode_seed_overrides_cycle(self):
        env = object.__new__(DogFightEnv)
        env.config = {"episode_seed_cycle": [1001, 1008]}

        self.assertEqual(env._resolve_episode_seed(4242), 4242)
        self.assertEqual(env._resolve_episode_seed(None), 1001)


class DummyActionEnv:
    config = {"control_surface_scale": 0.02}
    _SIM_ACTION_LOW = DogFightEnv._SIM_ACTION_LOW
    _SIM_ACTION_HIGH = DogFightEnv._SIM_ACTION_HIGH


class RecordingSim:
    def step(self, action):
        self.action = np.asarray(action)


class FixedRLProvider:
    def compute_action(self, context):
        return ActionResult(
            action=np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32),
            source="rl",
        )


class DummyProviderEnv(DummyActionEnv):
    _ownship_action_provider = FixedRLProvider()
    _sim = RecordingSim()
    _target_sim = object()
    _ownship_state = _target_state = pre_obs = None

    def _build_action_context(self, *args):
        return None

    def _to_sim_action(self, action):
        return DogFightEnv._to_sim_action(self, action)


class ControlAuthorityTests(unittest.TestCase):
    def test_control_surface_scale_leaves_throttle_mapping_unchanged(self):
        action = DogFightEnv._to_sim_action(
            DummyActionEnv(), np.array([1.0, -1.0, 0.5, 0.0], dtype=np.float32)
        )
        np.testing.assert_allclose(action, [0.02, -0.02, 0.01, 0.5])

    def test_rl_provider_uses_same_action_mapping_as_training(self):
        env = DummyProviderEnv()
        DogFightEnv._step_controlled_aircraft(env, np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(env._sim.action, [0.02, -0.02, 0.01, 0.5])

    def test_target_bt_scale_preserves_throttle(self):
        env = object.__new__(DogFightEnv)
        env.config = {"target_control_surface_scale": 0.3}
        action = env._scale_target_action(
            np.array([1.0, -1.0, 0.5, 0.8], dtype=np.float32)
        )
        np.testing.assert_allclose(action, [0.3, -0.3, 0.15, 0.8])

    def test_residual_action_limits_surface_and_throttle_correction(self):
        env = object.__new__(DogFightEnv)
        env.config = {
            "control_surface_scale": 1.0,
            "residual_action": {
                "enabled": True,
                "baseline": [0.0, 0.0, -0.001, -0.04],
                "scale": [0.1, 0.1, 0.1, 0.05],
            },
        }
        action = env._to_sim_action(
            np.array([1.0, -1.0, 0.5, 1.0], dtype=np.float32)
        )
        np.testing.assert_allclose(action, [0.1, -0.1, 0.049, 0.505])

    def test_observation_feedback_uses_signed_feature(self):
        provider = ObservationFeedbackActionProvider(
            FixedRLProvider(),
            observation_index=2,
            action_gains=np.array([0.0, 2.0, 0.0, 0.0], dtype=np.float32),
        )
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            observation=np.array([0.0, 0.0, -0.1], dtype=np.float32),
        )
        result = provider.compute_action(context)
        np.testing.assert_allclose(result.action, [1.0, -1.0, 0.5, 0.0])
        np.testing.assert_allclose(
            result.info["observation_feedback"],
            [0.0, -0.2, 0.0, 0.0],
        )

    def test_env_bundle_feedback_matches_provider_correction(self):
        env = object.__new__(DogFightEnv)
        env.config = {
            "observation_action_feedback": {
                "enabled": True,
                "observation_index": 2,
                "action_gains": [0.0, -20.0, 0.0, 0.0],
                "action_bias": [0.0, 0.02, -0.005, 0.0],
            }
        }
        env.pre_obs = np.array([0.0, 0.0, 0.01], dtype=np.float32)
        corrected = env._apply_observation_action_feedback(
            np.array([0.1, 0.0, 0.2, -0.3], dtype=np.float32)
        )
        np.testing.assert_allclose(corrected, [0.1, -0.18, 0.195, -0.3])

    def test_env_feedback_latches_initial_feature_branch(self):
        env = object.__new__(DogFightEnv)
        env.config = {
            "observation_action_feedback": {
                "enabled": True,
                "observation_index": 2,
                "latch_branch": True,
                "branches": [
                    {
                        "max_value": -0.01,
                        "action_gains": [0.0, 0.0, 0.0, 0.0],
                        "action_bias": [0.0, 0.0, -0.5, 0.0],
                    },
                    {
                        "max_value": None,
                        "action_gains": [0.0, 0.0, 0.0, 0.0],
                        "action_bias": [0.0, 0.02, 0.0, 0.0],
                    },
                ],
            }
        }
        env._action_feedback_branch_index = None
        env.pre_obs = np.array([0.0, 0.0, -0.02], dtype=np.float32)
        first = env._apply_observation_action_feedback(np.zeros(4, dtype=np.float32))
        env.pre_obs[2] = 0.5
        second = env._apply_observation_action_feedback(np.zeros(4, dtype=np.float32))
        np.testing.assert_allclose(first, [0.0, 0.0, -0.5, 0.0])
        np.testing.assert_allclose(second, first)

    def test_wez_then_provider_counts_each_env_timestep_once(self):
        primary = FixedRLProvider()
        secondary = FixedRLProvider()
        provider = WezThenActionProvider(
            primary,
            secondary,
            wez_observation_index=2,
            required_steps=2,
        )
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            observation=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            info={"timestep": 1},
        )
        provider.compute_action(context)
        provider.compute_action(context)
        self.assertEqual(provider._consecutive_wez_steps, 1)
        context.info["timestep"] = 2
        provider.compute_action(context)
        self.assertTrue(provider._secondary_active)


class EvaluationSummaryTests(unittest.TestCase):
    def test_target_crash_win_is_not_counted_as_ownship_kill(self):
        base = {
            "outcome": "win",
            "survival_time_s": 1.0,
            "min_altitude_m": 1000.0,
            "min_distance_m": 100.0,
            "initial_distance_m": 950.0,
            "wez_1s_achieved": True,
            "started_in_wez": False,
            "entered_wez_from_outside": True,
            "max_consecutive_wez_s": 1.0,
            "cumulative_target_damage": 1.0,
            "trajectory_sha256": "a",
            "initial_heading_deg": 0.0,
            "initial_n_m": 0.0,
            "initial_e_m": 0.0,
            "initial_d_m": -7000.0,
        }
        summary = summarize([
            {**base, "end_condition": "target destroyed"},
            {**base, "end_condition": "target altitude below min"},
        ])
        self.assertEqual(summary["win"], 2)
        self.assertEqual(summary["ownship_kills"], 1)
        self.assertEqual(summary["target_crash_wins"], 1)


class WezMilestoneRewardTests(unittest.TestCase):
    @staticmethod
    def make_env(*, consecutive=0, maximum=0, in_wez=False):
        env = object.__new__(DogFightEnv)
        env._reward_config = {
            "wez_required_steps": 10,
            "wez_progress_scale": 0.0,
            "wez_milestone_bonus": 20.0,
            "wez_miss_penalty": -50.0,
        }
        env._ep_consecutive_wez_steps = consecutive
        env._ep_max_consecutive_wez_steps = maximum
        env._in_wez = in_wez
        return env

    def test_tenth_consecutive_step_awards_bonus_once(self):
        env = self.make_env(consecutive=9, maximum=9, in_wez=True)
        total, components = env._apply_wez_milestone_reward(
            1.0, {"base": 1.0}, terminated=False, truncated=False
        )
        self.assertEqual(total, 21.0)
        self.assertEqual(components["wez_milestone"], 20.0)

        env._ep_consecutive_wez_steps = 10
        env._ep_max_consecutive_wez_steps = 10
        total, components = env._apply_wez_milestone_reward(
            1.0, {"base": 1.0}, terminated=False, truncated=False
        )
        self.assertEqual(total, 1.0)
        self.assertEqual(components["wez_milestone"], 0.0)

    def test_terminal_miss_penalty_only_below_required_duration(self):
        env = self.make_env(consecutive=0, maximum=9, in_wez=False)
        total, components = env._apply_wez_milestone_reward(
            -30.0, {"terminal": -30.0}, terminated=False, truncated=True
        )
        self.assertEqual(total, -80.0)
        self.assertEqual(components["wez_miss"], -50.0)

        env._ep_max_consecutive_wez_steps = 10
        total, components = env._apply_wez_milestone_reward(
            -30.0, {"terminal": -30.0}, terminated=False, truncated=True
        )
        self.assertEqual(total, -30.0)
        self.assertEqual(components["wez_miss"], 0.0)

    def test_wez_progress_increases_with_consecutive_steps(self):
        env = self.make_env(consecutive=2, maximum=2, in_wez=True)
        env._reward_config["wez_progress_scale"] = 10.0
        total, components = env._apply_wez_milestone_reward(
            1.0, {"base": 1.0}, terminated=False, truncated=False
        )

        self.assertEqual(components["wez_progress"], 3.0)
        self.assertEqual(total, 4.0)


class Tactical17ObservationTests(unittest.TestCase):
    def test_environment_passes_wez_config_to_tactical25(self):
        ownship = np.asarray(state(), dtype=np.float64)
        target = np.asarray(state(), dtype=np.float64)
        geometry = DummyGeometry(distance=500.0, ata=0.0)
        geometry._get_aspect_angle = lambda *args: 0.0
        geometry._get_los_angle = lambda *args: (0.0, 0.0)
        env = object.__new__(DogFightEnv)
        env._observation_fn = None
        env._observation_mode = "tactical25"
        env._geo_info = geometry
        env._wez = {
            "min_range_m": 152.4,
            "max_range_m": 914.4,
            "angle_deg": 10.0,
        }
        env.num_observation = 25

        observation = env._build_observation_for(ownship, target)

        self.assertEqual(float(observation[14]), 1.0)

    def test_tactical21_appends_target_maneuver_features(self):
        ownship = np.asarray(state(), dtype=np.float64)
        target = np.asarray(state(), dtype=np.float64)
        ownship[StateIndex.YAW] = 350.0
        target[StateIndex.YAW] = 10.0
        target[StateIndex.ROLL] = 45.0
        ownship[StateIndex.U] = 200.0
        target[StateIndex.U] = 250.0
        geometry = DummyGeometry()
        geometry._get_aspect_angle = lambda *args: 0.0
        geometry._get_los_angle = lambda *args: (0.0, 0.0)

        obs17 = build_observation("tactical17", ownship, target, geometry, None)
        obs21 = build_observation("tactical21", ownship, target, geometry, None)

        self.assertEqual(observation_size("tactical21"), 21)
        np.testing.assert_allclose(obs21[:17], obs17)
        self.assertAlmostEqual(float(obs21[17]), 0.25, places=5)
        self.assertAlmostEqual(float(obs21[18]), np.sin(np.deg2rad(20.0)), places=5)
        self.assertAlmostEqual(float(obs21[19]), np.cos(np.deg2rad(20.0)), places=5)
        self.assertAlmostEqual(float(obs21[20]), 0.25, places=5)

    def test_tactical25_appends_turn_rate_features(self):
        ownship = np.asarray(state(), dtype=np.float64)
        target = np.asarray(state(), dtype=np.float64)
        ownship[StateIndex.R] = 3.0
        target[StateIndex.R] = 9.0
        ownship[StateIndex.N] = 0.0
        ownship[StateIndex.E] = 0.0
        target[StateIndex.N] = 1000.0
        target[StateIndex.E] = 0.0
        ownship[StateIndex.U] = 100.0
        target[StateIndex.U] = 100.0
        ownship[StateIndex.YAW] = 0.0
        target[StateIndex.YAW] = 90.0
        geometry = DummyGeometry()
        geometry._get_aspect_angle = lambda *args: 0.0
        geometry._get_los_angle = lambda *args: (0.0, 0.0)

        obs21 = build_observation("tactical21", ownship, target, geometry, None)
        obs25 = build_observation("tactical25", ownship, target, geometry, None)

        self.assertEqual(observation_size("tactical25"), 25)
        np.testing.assert_allclose(obs25[:21], obs21)
        self.assertAlmostEqual(float(obs25[21]), 0.1, places=5)
        self.assertAlmostEqual(float(obs25[22]), 0.3, places=5)
        self.assertAlmostEqual(float(obs25[23]), 0.2, places=5)
        self.assertAlmostEqual(float(obs25[24]), np.degrees(0.1) / 30.0, places=5)

    def test_yaw_encoding_is_continuous_across_zero_degrees(self):
        ownship_a = np.asarray(state(), dtype=np.float64)
        ownship_b = np.asarray(state(), dtype=np.float64)
        target = np.asarray(state(), dtype=np.float64)
        ownship_a[StateIndex.YAW] = 359.0
        ownship_b[StateIndex.YAW] = 1.0
        geometry = DummyGeometry()
        geometry._get_aspect_angle = lambda *args: 0.0
        geometry._get_los_angle = lambda *args: (0.0, 0.0)

        obs_a = build_observation("tactical17", ownship_a, target, geometry, None)
        obs_b = build_observation("tactical17", ownship_b, target, geometry, None)

        self.assertEqual(observation_size("tactical17"), 17)
        self.assertLess(abs(float(obs_a[2] - obs_b[2])), 0.04)
        self.assertLess(abs(float(obs_a[3] - obs_b[3])), 0.001)
        self.assertTrue(np.all(obs_a >= -1.0))
        self.assertTrue(np.all(obs_a <= 1.0))

    def test_tactical17_to_tactical16_adapter_preserves_periodic_yaw(self):
        provider = object.__new__(BundleMappedDirectSACActionProvider)
        provider.observation_mode = "tactical16"
        observation = np.zeros(17, dtype=np.float32)
        observation[2] = np.sin(np.deg2rad(359.0))
        observation[3] = np.cos(np.deg2rad(359.0))
        observation[4:] = np.arange(13, dtype=np.float32)
        adapted = provider._adapt_observation(observation)
        self.assertEqual(adapted.shape, (16,))
        self.assertGreater(adapted[2], 0.99)
        np.testing.assert_allclose(adapted[3:], observation[4:])

    def test_tactical21_to_tactical16_adapter_drops_appended_features(self):
        provider = object.__new__(BundleMappedDirectSACActionProvider)
        provider.observation_mode = "tactical16"
        observation = np.zeros(21, dtype=np.float32)
        observation[2] = np.sin(np.deg2rad(20.0))
        observation[3] = np.cos(np.deg2rad(20.0))
        observation[4:17] = np.arange(13, dtype=np.float32)
        observation[17:] = 99.0
        adapted = provider._adapt_observation(observation)
        self.assertEqual(adapted.shape, (16,))
        self.assertAlmostEqual(float(adapted[2]), 20.0 / 180.0 - 1.0)
        np.testing.assert_allclose(adapted[3:], observation[4:17])

    def test_tactical25_adapter_drops_or_zero_pads_rate_features(self):
        provider = object.__new__(BundleMappedDirectSACActionProvider)
        provider.observation_mode = "tactical21"
        observation25 = np.arange(25, dtype=np.float32)
        np.testing.assert_allclose(
            provider._adapt_observation(observation25), observation25[:21]
        )

        provider.observation_mode = "tactical25"
        observation21 = np.arange(21, dtype=np.float32)
        adapted = provider._adapt_observation(observation21)
        self.assertEqual(adapted.shape, (25,))
        np.testing.assert_allclose(adapted[:21], observation21)
        np.testing.assert_allclose(adapted[21:], 0.0)

    def test_tactical16_to_tactical17_adapter_encodes_periodic_yaw(self):
        provider = BundleMappedDirectSACActionProvider.__new__(
            BundleMappedDirectSACActionProvider
        )
        provider.observation_mode = "tactical17"
        observation = np.zeros(16, dtype=np.float32)
        observation[2] = -0.5
        adapted = provider._adapt_observation(observation)
        self.assertEqual(adapted.shape, (17,))
        self.assertAlmostEqual(float(adapted[2]), 1.0, places=6)
        self.assertAlmostEqual(float(adapted[3]), 0.0, places=6)
        np.testing.assert_allclose(adapted[4:], observation[3:])

    def test_target_bank_action_scale_adjusts_roll_and_rudder_only(self):
        target = np.asarray(state(roll=60.0), dtype=np.float64)
        provider = TargetBankActionScaleProvider(
            ConstantActionProvider([0.1, 0.2, 0.1, 0.4]),
            reference_bank_deg=30.0,
        )
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            target_state=target,
        ))
        np.testing.assert_allclose(result.action, [0.2, 0.2, 0.2, 0.4])
        self.assertEqual(result.info["target_bank_roll_scale"], 2.0)


class PursuitGuidanceTests(unittest.TestCase):
    def test_wrap_degrees_uses_shortest_turn(self):
        self.assertEqual(PursuitGuidanceActionProvider._wrap_degrees(359.0), -1.0)
        self.assertEqual(PursuitGuidanceActionProvider._wrap_degrees(-359.0), 1.0)

    def test_guidance_turns_toward_right_hand_target(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        ownship[StateIndex.YAW] = 0.0
        target[StateIndex.E] = 1000.0
        result = PursuitGuidanceActionProvider().compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))
        self.assertGreater(result.action[0], 0.0)
        self.assertGreater(result.action[2], 0.0)
        self.assertEqual(result.action[3], 1.0)

    def test_guidance_brakes_inside_configured_range(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        result = PursuitGuidanceActionProvider(
            brake_range_m=600.0,
            near_throttle_action=-0.5,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))
        self.assertEqual(result.action[3], -0.5)
        self.assertEqual(result.info["throttle_action"], -0.5)

    def test_guidance_waits_for_heading_alignment_before_braking(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.E] = 500.0
        result = PursuitGuidanceActionProvider(
            brake_range_m=600.0,
            near_throttle_action=-1.0,
            brake_heading_error_deg=30.0,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertEqual(result.action[3], 1.0)
        self.assertFalse(result.info["braking"])

    def test_capture_uses_separate_pitch_gain(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7100.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        result = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            capture_pitch_gain=0.2,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertEqual(result.info["pitch_gain"], 0.2)

    def test_geometry_then_latches_secondary_when_close_and_aligned(self):
        primary = FixedRLProvider()
        secondary = FixedRLProvider()
        provider = GeometryThenActionProvider(
            primary,
            secondary,
            max_distance_m=900.0,
            max_ata_deg=2.0,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["geometry_then_secondary"])

    def test_capture_can_damp_heading_rate_with_rudder(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            capture_rudder_gain=0.0,
            capture_heading_rate_to_rudder=-1.0,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )
        provider.compute_action(context)
        target[StateIndex.E] = 5.0
        result = provider.compute_action(context)

        self.assertLess(result.action[2], 0.0)
        self.assertEqual(result.info["heading_rate_to_rudder"], -1.0)

    def test_wez_hold_can_follow_target_attitude(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            wez_hold_latch_ata_deg=1.0,
            wez_hold_target_heading_blend=1.0,
            wez_hold_target_pitch_blend=1.0,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        target[StateIndex.YAW] = 10.0
        target[StateIndex.PITCH] = 2.0
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["wez_hold_latched"])
        self.assertAlmostEqual(result.info["control_heading_error_deg"], 10.0)
        self.assertAlmostEqual(result.info["desired_pitch_deg"], 2.0)

    def test_trail_guidance_aims_behind_target_heading(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        ownship[StateIndex.E] = 1000.0
        target[StateIndex.N] = 1000.0
        target[StateIndex.YAW] = 0.0
        target[StateIndex.U] = 250.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )

        pure = PursuitGuidanceActionProvider().compute_action(context)
        trail = PursuitGuidanceActionProvider(
            trail_distance_m=500.0
        ).compute_action(context)

        self.assertLess(
            trail.info["heading_error_deg"],
            pure.info["heading_error_deg"],
        )
        self.assertEqual(trail.info["trail_distance_m"], 500.0)

    def test_capture_target_bank_feedforward_blends_target_roll(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        target[StateIndex.ROLL] = -60.0
        result = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            capture_bank_scale=0.0,
            capture_target_bank_blend=0.5,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertAlmostEqual(result.info["desired_bank_deg"], -30.0)
        self.assertEqual(result.info["capture_target_bank_blend"], 0.5)

    def test_capture_uses_separate_heading_rate_gain(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            heading_rate_to_bank=1.0,
            capture_heading_rate_to_bank=-2.0,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )
        provider.compute_action(context)
        target[StateIndex.E] = 5.0
        result = provider.compute_action(context)

        self.assertEqual(result.info["heading_rate_to_bank"], -2.0)

    def test_capture_uses_separate_roll_gain(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        result = PursuitGuidanceActionProvider(
            roll_gain=0.01,
            capture_roll_gain=0.2,
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertEqual(result.info["roll_gain"], 0.2)

    def test_proximity_speed_match_overrides_throttle_when_close(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        ownship[StateIndex.U] = 300.0
        target[StateIndex.U] = 230.0
        target[StateIndex.N] = 500.0
        result = PursuitGuidanceActionProvider(
            proximity_speed_match_start_distance_m=1400.0,
            proximity_speed_match_gain=0.02,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertEqual(float(result.action[3]), -1.0)
        self.assertEqual(
            result.info["proximity_speed_match_start_distance_m"], 1400.0
        )

    def test_proximity_speed_match_respects_minimum_overspeed(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        ownship[StateIndex.U] = 280.0
        target[StateIndex.U] = 230.0
        target[StateIndex.N] = 500.0
        result = PursuitGuidanceActionProvider(
            proximity_speed_match_start_distance_m=1400.0,
            proximity_speed_match_gain=0.02,
            proximity_speed_match_min_overspeed_mps=58.0,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertEqual(float(result.action[3]), 1.0)

    def test_proximity_speed_match_can_latch_after_overspeed(self):
        provider = PursuitGuidanceActionProvider(
            proximity_speed_match_start_distance_m=1400.0,
            proximity_speed_match_gain=0.02,
            proximity_speed_match_min_overspeed_mps=58.0,
            proximity_speed_match_latch=True,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        target[StateIndex.U] = 230.0
        ownship[StateIndex.U] = 300.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )
        provider.compute_action(context)
        ownship[StateIndex.U] = 250.0
        result = provider.compute_action(context)

        self.assertAlmostEqual(float(result.action[3]), -0.4, places=6)
        self.assertTrue(result.info["proximity_speed_match_latched"])
        provider.reset()
        self.assertFalse(provider._proximity_speed_match_latched)

    def test_proximity_speed_match_waits_for_alignment_gate(self):
        provider = PursuitGuidanceActionProvider(
            proximity_speed_match_start_distance_m=1400.0,
            proximity_speed_match_gain=0.02,
            proximity_speed_match_min_overspeed_mps=45.0,
            proximity_speed_match_max_ata_deg=5.0,
            proximity_speed_match_latch=True,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        target[StateIndex.U] = 230.0
        ownship[StateIndex.U] = 300.0
        ownship[StateIndex.YAW] = 10.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )

        unaligned = provider.compute_action(context)
        self.assertEqual(float(unaligned.action[3]), 1.0)
        self.assertFalse(unaligned.info["proximity_speed_match_latched"])

        ownship[StateIndex.YAW] = 0.0
        aligned = provider.compute_action(context)
        self.assertEqual(float(aligned.action[3]), -1.0)
        self.assertTrue(aligned.info["proximity_speed_match_latched"])

    def test_rear_threat_uses_break_turn_roll_gain(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = -500.0
        result = PursuitGuidanceActionProvider(
            roll_gain=0.002,
            rear_threat_distance_m=1000.0,
            rear_threat_ata_deg=120.0,
            rear_threat_roll_gain=0.02,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["rear_threat_active"])
        self.assertEqual(result.info["roll_gain"], 0.02)

    def test_rear_threat_adds_bank_pitch_compensation(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = -500.0
        result = PursuitGuidanceActionProvider(
            rear_threat_distance_m=1000.0,
            rear_threat_ata_deg=120.0,
            rear_threat_roll_gain=0.02,
            rear_threat_bank_pitch_compensation_gain=0.25,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertLess(float(result.action[1]), -0.5)

    def test_rear_threat_can_latch_turn_direction_across_wrap_boundary(self):
        provider = PursuitGuidanceActionProvider(
            max_bank_deg=70.0,
            rear_threat_distance_m=1000.0,
            rear_threat_ata_deg=120.0,
            rear_threat_roll_gain=0.02,
            rear_threat_turn_latch=True,
            rear_threat_turn_release_ata_deg=90.0,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = -500.0
        target[StateIndex.E] = -1.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )

        first = provider.compute_action(context)
        target[StateIndex.E] = 1.0
        across_wrap = provider.compute_action(context)

        self.assertLess(float(first.action[0]), 0.0)
        self.assertLess(float(across_wrap.action[0]), 0.0)
        self.assertTrue(across_wrap.info["rear_threat_turn_latched"])

        target[StateIndex.N] = 500.0
        released = provider.compute_action(context)
        self.assertFalse(released.info["rear_threat_turn_latched"])
        provider.reset()
        self.assertEqual(provider._rear_threat_turn_sign, 0.0)

    def test_rear_threat_can_apply_roll_delta_damping(self):
        provider = PursuitGuidanceActionProvider(
            max_bank_deg=70.0,
            rear_threat_distance_m=1000.0,
            rear_threat_ata_deg=120.0,
            rear_threat_roll_gain=0.02,
            rear_threat_roll_delta_damping=0.1,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = -500.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )
        provider.compute_action(context)
        ownship[StateIndex.ROLL] = -10.0
        damped = provider.compute_action(context)

        self.assertEqual(damped.info["active_roll_delta_damping"], 0.1)
        self.assertGreater(float(damped.action[0]), -1.0)

    def test_rear_threat_can_add_rudder_turn_assist(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = -500.0
        result = PursuitGuidanceActionProvider(
            rear_threat_distance_m=1000.0,
            rear_threat_ata_deg=120.0,
            rear_threat_roll_gain=0.002,
            rear_threat_rudder_gain=0.005,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertAlmostEqual(float(result.action[2]), -0.01, places=6)
        self.assertEqual(result.info["rear_threat_rudder_gain"], 0.005)

    def test_capture_can_damp_pitch_rate(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            pitch_gain=0.05,
            capture_pitch_delta_damping=0.1,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )
        provider.compute_action(context)
        ownship[StateIndex.PITCH] = 5.0
        damped = provider.compute_action(context)

        self.assertEqual(damped.info["capture_pitch_delta_damping"], 0.1)
        self.assertEqual(damped.info["pitch_delta_deg"], 5.0)
        self.assertGreater(float(damped.action[1]), 0.5)

    def test_target_altitude_guard_recovers_with_hysteresis(self):
        primary = ConstantActionProvider([0.2, 0.3, 0.4, 0.5])
        guard = TargetAltitudeGuardActionProvider(
            primary,
            trigger_altitude_m=2500.0,
            release_altitude_m=4000.0,
        )
        ownship = np.asarray(
            state(altitude=2400.0, roll=30.0), dtype=np.float64
        )
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=np.asarray(state(), dtype=np.float64),
        )

        recovery = guard.compute_action(context)
        self.assertTrue(recovery.info["target_altitude_guard"])
        self.assertLess(float(recovery.action[0]), 0.0)
        self.assertEqual(float(recovery.action[1]), -1.0)
        self.assertEqual(float(recovery.action[3]), 1.0)

        ownship[StateIndex.ALT] = 3500.0
        still_recovering = guard.compute_action(context)
        self.assertTrue(still_recovering.info["target_altitude_guard"])

        ownship[StateIndex.ALT] = 4100.0
        released = guard.compute_action(context)
        self.assertFalse(released.info["target_altitude_guard"])
        np.testing.assert_allclose(released.action, primary.action)

    def test_target_altitude_guard_can_taper_pull_up_with_altitude(self):
        guard = TargetAltitudeGuardActionProvider(
            ConstantActionProvider([0.0, 0.0, 0.0, 0.0]),
            trigger_altitude_m=2500.0,
            release_altitude_m=4000.0,
            proportional_pull_up=True,
        )
        ownship = np.asarray(state(altitude=2400.0), dtype=np.float64)
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=np.asarray(state(), dtype=np.float64),
        )
        guard.compute_action(context)
        ownship[StateIndex.ALT] = 3500.0
        tapered = guard.compute_action(context)

        self.assertAlmostEqual(float(tapered.action[1]), -1.0 / 3.0, places=5)
        self.assertAlmostEqual(
            tapered.info["target_altitude_guard_pull_up_action"],
            -1.0 / 3.0,
            places=5,
        )

    def test_capture_mode_can_latch_after_alignment(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=2.0,
            latch_capture_mode=True,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )
        aligned = provider.compute_action(context)
        target[StateIndex.E] = 500.0
        after = provider.compute_action(context)

        self.assertTrue(aligned.info["capture_latched"])
        self.assertTrue(after.info["capture_mode"])
        provider.reset()
        self.assertFalse(provider._capture_latched)

    def test_wez_hold_latches_on_three_dimensional_ata(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            wez_hold_latch_ata_deg=1.0,
            wez_hold_roll_gain=0.2,
            wez_hold_roll_delta_damping=0.7,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["wez_hold_latched"])
        self.assertAlmostEqual(result.info["ata_deg"], 0.0)
        self.assertEqual(result.info["roll_gain"], 0.2)
        self.assertEqual(result.info["active_roll_delta_damping"], 0.7)

    def test_capture_pitch_error_rate_gain_is_reported(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            capture_pitch_error_rate_gain=0.1,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )
        provider.compute_action(context)
        target[StateIndex.D] = -7100.0
        result = provider.compute_action(context)

        self.assertEqual(result.info["capture_pitch_error_rate_gain"], 0.1)
        self.assertGreater(result.info["pitch_error_rate_deg_s"], 0.0)

    def test_wez_hold_can_freeze_bank_at_latch(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            wez_hold_latch_ata_deg=1.0,
            wez_hold_freeze_bank=True,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        ownship[StateIndex.ROLL] = 30.0
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["wez_hold_latched"])
        self.assertTrue(result.info["wez_hold_freeze_bank"])
        self.assertAlmostEqual(result.info["wez_hold_bank_deg"], 30.0)
        self.assertAlmostEqual(result.info["desired_bank_deg"], 30.0)

        provider.reset()
        self.assertIsNone(provider._wez_hold_bank_deg)

    def test_wez_hold_uses_dedicated_pitch_damping(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            wez_hold_latch_ata_deg=1.0,
            wez_hold_pitch_gain=0.4,
            wez_hold_pitch_delta_damping=0.6,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 500.0
        context = ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        )
        provider.compute_action(context)
        ownship[StateIndex.PITCH] = 2.0
        result = provider.compute_action(context)

        self.assertEqual(result.info["wez_hold_pitch_gain"], 0.4)
        self.assertEqual(result.info["wez_hold_pitch_delta_damping"], 0.6)

    def test_maneuver_capture_uses_early_range_and_lead(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1100.0,
            capture_heading_error_deg=180.0,
            capture_lead_time_s=1.0,
            capture_maneuver_range_m=1550.0,
            capture_maneuver_target_pitch_deg=45.0,
            capture_maneuver_target_bank_deg=80.0,
            capture_maneuver_lead_time_s=0.9,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 1500.0
        target[StateIndex.PITCH] = 60.0
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["maneuver_capture_active"])
        self.assertTrue(result.info["capture_mode"])
        self.assertEqual(result.info["effective_capture_range_m"], 1550.0)
        self.assertEqual(result.info["active_lead_time_s"], 0.9)

    def test_extreme_bank_capture_uses_separate_lead(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1100.0,
            capture_heading_error_deg=180.0,
            capture_maneuver_range_m=1550.0,
            capture_maneuver_target_pitch_deg=45.0,
            capture_maneuver_lead_time_s=0.9,
            capture_extreme_bank_deg=140.0,
            capture_extreme_bank_lead_time_s=1.15,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 1500.0
        target[StateIndex.PITCH] = 60.0
        target[StateIndex.ROLL] = 150.0
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["extreme_bank_capture_active"])
        self.assertEqual(result.info["active_lead_time_s"], 1.15)

    def test_slow_target_capture_uses_separate_profile(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1100.0,
            capture_heading_error_deg=180.0,
            capture_slow_target_speed_mps=220.0,
            capture_slow_target_range_m=1900.0,
            capture_slow_target_lead_time_s=0.9,
        )
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 1800.0
        target[StateIndex.U] = 200.0
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["slow_target_capture_active"])
        self.assertEqual(result.info["effective_capture_range_m"], 1900.0)
        self.assertEqual(result.info["active_lead_time_s"], 0.9)

    def test_capture_body_axis_guidance_uses_three_dimensional_los(self):
        provider = PursuitGuidanceActionProvider(
            capture_range_m=1000.0,
            capture_heading_error_deg=90.0,
            capture_body_axis_guidance=True,
            capture_body_axis_roll_gain=0.1,
            capture_body_axis_pitch_gain=0.1,
        )
        ownship = np.asarray(state(), dtype=np.float64)
        target = np.asarray(state(), dtype=np.float64)
        target[StateIndex.N] = 500.0
        target[StateIndex.E] = 50.0
        target[StateIndex.D] = -100.0
        result = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["capture_body_axis_guidance"])
        self.assertGreater(result.info["body_azimuth_deg"], 0.0)
        self.assertGreater(result.info["body_elevation_deg"], 0.0)
        self.assertGreater(float(result.action[0]), 0.0)
        self.assertLess(float(result.action[1]), 0.0)

    def test_target_bank_feedforward_allows_opposite_sign_mapping(self):
        provider = PursuitGuidanceActionProvider(
            capture_target_bank_blend=-1.0,
            wez_hold_target_bank_blend=-1.0,
        )
        self.assertEqual(provider.capture_target_bank_blend, -1.0)
        self.assertEqual(provider.wez_hold_target_bank_blend, -1.0)

    def test_pursuit_autopilot_points_at_stationary_target(self):
        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7500.0), dtype=np.float64)
        target[StateIndex.E] = 1000.0
        result = PursuitAutopilotActionProvider(lead_time_s=0.0).compute_action(
            ActionContext(
                sim=None,
                opponent_sim=None,
                ownship_state=ownship,
                target_state=target,
            )
        )
        self.assertEqual(result.source, "pursuit_autopilot")
        self.assertAlmostEqual(result.info["heading_cmd"], 90.0)
        self.assertAlmostEqual(result.info["altitude_cmd"], -7500.0)

    def test_guidance_residual_combines_policy_in_rl_space(self):
        class ConstantProvider(ActionProvider):
            def compute_action(self, context):
                return ActionResult(
                    action=np.ones(4, dtype=np.float32),
                    source="rl",
                )

        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 1000.0
        result = GuidanceResidualActionProvider(
            ConstantProvider(),
            PursuitGuidanceActionProvider(),
            residual_scale=[0.2, 0.2, 0.2, 0.1],
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))
        self.assertEqual(result.source, "rl")
        self.assertAlmostEqual(float(result.action[0]), 0.2)
        self.assertEqual(float(result.action[3]), 1.0)

    def test_guidance_residual_raises_authority_only_near_aligned_target(self):
        class ConstantProvider(ActionProvider):
            def compute_action(self, context):
                return ActionResult(
                    action=np.ones(4, dtype=np.float32),
                    source="rl",
                )

        ownship = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 800.0
        provider = GuidanceResidualActionProvider(
            ConstantProvider(),
            PursuitGuidanceActionProvider(),
            residual_scale=[0.05, 0.05, 0.05, 0.05],
            near_residual_scale=[0.20, 0.20, 0.20, 0.20],
            near_distance_m=1200.0,
            near_ata_deg=10.0,
        )

        near = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))
        self.assertTrue(near.info["near_residual_active"])
        self.assertAlmostEqual(float(near.action[0]), 0.20)

        target[StateIndex.N] = 0.0
        target[StateIndex.E] = 800.0
        off_axis = provider.compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))
        self.assertFalse(off_axis.info["near_residual_active"])
        np.testing.assert_allclose(off_axis.info["residual_scale"], [0.05] * 4)

    def test_guidance_residual_suppresses_wrong_way_roll(self):
        class ConstantProvider(ActionProvider):
            def compute_action(self, context):
                return ActionResult(
                    action=np.ones(4, dtype=np.float32),
                    source="rl",
                )

        ownship = np.asarray(
            state(altitude=7000.0, roll=20.0), dtype=np.float64
        )
        target = np.asarray(state(altitude=7000.0), dtype=np.float64)
        target[StateIndex.N] = 1000.0
        target[StateIndex.E] = -1000.0
        result = GuidanceResidualActionProvider(
            ConstantProvider(),
            PursuitGuidanceActionProvider(roll_gain=0.01),
            residual_scale=[0.2, 0.2, 0.2, 0.1],
            suppress_opposing_roll_heading_error_deg=10.0,
            opposing_roll_guidance_scale=2.0,
        ).compute_action(ActionContext(
            sim=None,
            opponent_sim=None,
            ownship_state=ownship,
            target_state=target,
        ))

        self.assertTrue(result.info["opposing_roll_suppressed"])
        self.assertEqual(result.info["residual_scale"][0], 0.0)
        self.assertLess(float(result.action[0]), 0.0)


def state(*, altitude=7000.0, pitch=0.0, roll=0.0, health=1.0):
    result = [0.0] * 46
    result[StateIndex.ALT] = altitude
    result[StateIndex.PITCH] = pitch
    result[StateIndex.ROLL] = roll
    result[StateIndex.HEALTH] = health
    return result


def reward(ownship=None, target=None, **kwargs):
    config = {**MY_REWARD_CONFIG, **kwargs.pop("config", {})}
    return compute_reward(
        ownship or state(),
        target or state(),
        kwargs.pop("ownship_damage", 0.0),
        kwargs.pop("target_damage", 0.0),
        DummyGeometry(),
        {},
        config,
        kwargs.pop("terminated", False),
        kwargs.pop("truncated", False),
        kwargs.pop("end_condition", ""),
    )


class SacFlightSafetyRewardTests(unittest.TestCase):
    def test_component_sum_matches_total(self):
        total, components = reward(ownship=state(altitude=1000, pitch=-30, roll=90))
        self.assertAlmostEqual(total, sum(components.values()))

    def test_altitude_risk_starts_at_warning(self):
        _, at_warning = reward(ownship=state(altitude=2000))
        _, halfway = reward(ownship=state(altitude=1150))
        _, at_floor = reward(ownship=state(altitude=300))
        self.assertEqual(at_warning["altitude"], 0.0)
        self.assertAlmostEqual(halfway["altitude"], -0.025)
        self.assertAlmostEqual(at_floor["altitude"], -0.10)

    def test_dive_requires_low_altitude_and_negative_pitch(self):
        _, high = reward(ownship=state(altitude=2500, pitch=-45))
        _, climbing = reward(ownship=state(altitude=300, pitch=10))
        _, diving = reward(ownship=state(altitude=300, pitch=-45))
        self.assertEqual(high["dive"], 0.0)
        self.assertEqual(climbing["dive"], 0.0)
        self.assertAlmostEqual(diving["dive"], -0.05)

    def test_bank_penalty_is_risk_gated(self):
        _, high = reward(ownship=state(altitude=2500, roll=120))
        _, threshold = reward(ownship=state(altitude=300, roll=60))
        _, excessive = reward(ownship=state(altitude=300, roll=180))
        self.assertEqual(high["bank"], 0.0)
        self.assertEqual(threshold["bank"], 0.0)
        self.assertAlmostEqual(excessive["bank"], -0.02)

    def test_terminal_mapping(self):
        cases = {
            "ownship altitude below min": -120.0,
            "Ownship FDM output Fall": -120.0,
            "FDM Update Fail": -120.0,
            "target altitude below min": 100.0,
            "Target FDM output Fall": 100.0,
            "max time out": -30.0,
            "episode step limit": -30.0,
            "two circle headon guard fail": -100.0,
        }
        for condition, expected in cases.items():
            with self.subTest(condition=condition):
                _, components = reward(
                    terminated=condition not in {"max time out", "episode step limit"},
                    truncated=condition in {"max time out", "episode step limit"},
                    end_condition=condition,
                )
                self.assertEqual(components["terminal"], expected)

    def test_health_terminal_mapping(self):
        _, win = reward(target=state(health=0), terminated=True)
        _, loss = reward(ownship=state(health=0), terminated=True)
        _, draw = reward(terminated=True)
        self.assertEqual(win["terminal"], 100.0)
        self.assertEqual(loss["terminal"], -100.0)
        self.assertEqual(draw["terminal"], -10.0)

    def test_pursuit_is_disabled_for_survival_stage(self):
        _, components = reward()
        self.assertEqual(components["pursuit"], 0.0)

    def test_v2_adds_early_level_flight_signals(self):
        _, components = compute_reward_v2(
            state(altitude=5000, pitch=-30, roll=30),
            state(),
            0.0,
            0.0,
            DummyGeometry(),
            {},
            V2_REWARD_CONFIG,
            False,
            False,
            "",
        )
        self.assertLess(components["altitude_hold"], 0.0)
        self.assertLess(components["level_pitch"], 0.0)
        self.assertLess(components["level_roll"], 0.0)

    def test_body_velocity_rotation_and_closure(self):
        ownship = state()
        target = state()
        ownship[StateIndex.N] = 0.0
        target[StateIndex.N] = 5000.0
        ownship[StateIndex.U] = 300.0
        target[StateIndex.U] = 200.0
        self.assertAlmostEqual(_body_velocity_to_ned(ownship)[0], 300.0)
        self.assertAlmostEqual(_closure_rate_mps(ownship, target), 100.0)

        ownship[StateIndex.YAW] = 180.0
        self.assertAlmostEqual(_body_velocity_to_ned(ownship)[0], -300.0)
        self.assertAlmostEqual(_closure_rate_mps(ownship, target), -500.0)

    def test_v4_strict_wez_hold_bonus(self):
        _, components = compute_reward_v4(
            state(), state(), 0.0, 0.01, DummyGeometry(), {},
            V4_REWARD_CONFIG, False, False, "",
        )
        self.assertEqual(components["wez_hold"], 2.0)
        self.assertEqual(components["fine_alignment"], 0.0)

    def test_v5_moving_alignment_rewards_small_ata(self):
        _, components = compute_reward_v5(
            state(), state(), 0.0, 0.0, DummyGeometry(), {},
            V5_REWARD_CONFIG, False, False, "",
        )
        self.assertEqual(components["moving_alignment"], 1.0)

    def test_v6_brakes_positive_closure_inside_wez(self):
        ownship = state()
        target = state()
        ownship[StateIndex.N] = 0.0
        target[StateIndex.N] = 500.0
        ownship[StateIndex.U] = 300.0
        target[StateIndex.U] = 200.0
        wez = {"min_range_m": 152.4, "max_range_m": 914.4}

        _, components = compute_reward_v6(
            ownship, target, 0.0, 0.0, DummyGeometry(distance=500.0), wez,
            V6_REWARD_CONFIG, False, False, "",
        )
        self.assertLess(components["close_range_brake"], 0.0)
        self.assertEqual(components["inside_min_range"], 0.0)

        target[StateIndex.U] = 400.0
        _, opening = compute_reward_v6(
            ownship, target, 0.0, 0.0, DummyGeometry(distance=500.0), wez,
            V6_REWARD_CONFIG, False, False, "",
        )
        self.assertEqual(opening["close_range_brake"], 0.0)

    def test_v6_penalizes_crossing_inner_wez_range(self):
        wez = {"min_range_m": 152.4, "max_range_m": 914.4}
        _, components = compute_reward_v6(
            state(), state(), 0.0, 0.0, DummyGeometry(distance=100.0), wez,
            V6_REWARD_CONFIG, False, False, "",
        )
        self.assertLess(components["inside_min_range"], 0.0)

    def test_v7_rewards_matching_target_turn_heading_and_speed(self):
        ownship = state(roll=45.0)
        target = state(roll=45.0)
        ownship[StateIndex.YAW] = 30.0
        target[StateIndex.YAW] = 30.0
        ownship[StateIndex.U] = 220.0
        target[StateIndex.U] = 220.0
        wez = {"min_range_m": 152.4, "max_range_m": 914.4}
        _, components = compute_reward_v7(
            ownship, target, 0.0, 0.0, DummyGeometry(distance=500.0), wez,
            V7_REWARD_CONFIG, False, False, "",
        )
        self.assertEqual(components["bank_match"], 1.0)
        self.assertEqual(components["heading_match"], 1.0)
        self.assertEqual(components["speed_match"], 1.0)

    def test_v7_penalizes_near_target_mismatch(self):
        ownship = state(roll=-45.0)
        target = state(roll=45.0)
        ownship[StateIndex.YAW] = 0.0
        target[StateIndex.YAW] = 90.0
        ownship[StateIndex.U] = 320.0
        target[StateIndex.U] = 220.0
        wez = {"min_range_m": 152.4, "max_range_m": 914.4}
        _, components = compute_reward_v7(
            ownship, target, 0.0, 0.0, DummyGeometry(distance=500.0), wez,
            V7_REWARD_CONFIG, False, False, "",
        )
        self.assertEqual(components["bank_match"], -1.0)
        self.assertEqual(components["heading_match"], -1.0)
        self.assertEqual(components["speed_match"], -1.0)

    def test_v7_sync_shaping_is_gated_outside_capture_geometry(self):
        wez = {"min_range_m": 152.4, "max_range_m": 914.4}
        _, components = compute_reward_v7(
            state(), state(), 0.0, 0.0,
            DummyGeometry(distance=2500.0, ata=45.0), wez,
            V7_REWARD_CONFIG, False, False, "",
        )
        self.assertEqual(components["bank_match"], 0.0)
        self.assertEqual(components["heading_match"], 0.0)
        self.assertEqual(components["speed_match"], 0.0)


if __name__ == "__main__":
    unittest.main()
