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
from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.checkpoint_io import _apply_policy_weights
from dogfight.ai.rl_action_provider import AltitudeRecoveryActionProvider
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


class DummyGeometry:
    def _get_distance(self, ownship_state, target_state):
        return 500.0

    def _get_antenna_train_angle(self, ownship_state, target_state, degrees):
        return 0.0


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


if __name__ == "__main__":
    unittest.main()
