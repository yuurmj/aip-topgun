from __future__ import annotations

from pathlib import Path
from typing import Callable

import gzip
import pickle

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle
from dogfight.sim.state_schema import StateIndex
from dogfight.envs.observation import mirror_tactical_observation


class RLActionProvider(ActionProvider):
    def __init__(
        self,
        algorithm=None,
        bundle_dir: str | None = None,
        algorithm_factory: Callable[[dict], object] | None = None,
        policy_id: str = "default_policy",
        explore: bool = False,
        obs_builder: Callable[[ActionContext], np.ndarray] | None = None,
        confidence: float = 0.9,
        remap_throttle: bool = True,
    ):
        self.policy_id = policy_id
        self.explore = explore
        self.obs_builder = obs_builder
        self.confidence = confidence
        self.remap_throttle = bool(remap_throttle)
        self._owns_algorithm = False
        self._module_state = None
        self._debug_lstm_io = False
        self._debug_lstm_prints = 0

        if algorithm is not None:
            self.algorithm = algorithm
            self.metadata = None
        elif bundle_dir is not None:
            if algorithm_factory is None:
                raise ValueError("algorithm_factory is required when loading a lightweight bundle")
            self.metadata, weights = load_lightweight_policy_bundle(Path(bundle_dir))
            self.algorithm = algorithm_factory(self.metadata)
            try:
                # Old API stack: Policy-based weight loading
                self.algorithm.get_policy(self.policy_id).set_weights(weights)
            except AttributeError:
                # New API stack: load directly into the local RLModule state.
                env_runner = getattr(self.algorithm, "env_runner", None)
                if env_runner is not None and hasattr(env_runner, "set_state"):
                    env_runner.set_state({"rl_module": weights})
                elif env_runner is not None and hasattr(env_runner, "module"):
                    env_runner.module.set_state(weights)
                else:
                    raise RuntimeError(
                        "Unable to apply RLModule weights to the RLlib algorithm"
                    )
            self._owns_algorithm = True
        else:
            raise ValueError("Either algorithm or bundle_dir must be provided")

    def reset(self, context: ActionContext | None = None) -> None:
        """Reset recurrent module state at episode boundaries."""
        had_state = self._module_state is not None
        self._module_state = None
        self._debug_lstm_prints = 0
        if self._debug_lstm_io:
            frame_index = None
            if context is not None and isinstance(context.info, dict):
                frame_index = context.info.get("frame_index")
            print(
                "[DogFightEnv][RLActionProvider][LSTM_RESET] "
                f"had_state={had_state} frame_index={frame_index}"
            )

    def compute_action(self, context: ActionContext) -> ActionResult:
        observation = context.observation
        if observation is None:
            if self.obs_builder is None:
                raise ValueError("observation is required when obs_builder is not configured")
            observation = self.obs_builder(context)

        action = np.asarray(
            self._compute_module_action(observation),
            dtype=np.float32,
        ).copy()
        if action.shape != (4,):
            raise ValueError(
                f"RL policy action must have shape (4,), got {action.shape}"
            )
        action = np.clip(action, -1.0, 1.0)
        if self.remap_throttle:
            # Unreal/simulator commands use throttle [0, 1].  Composite
            # policy providers can opt out and perform this conversion once,
            # after guidance/residual processing has finished.
            action[3] = (action[3] + 1.0) / 2.0
            action = clip_action(action)

        return ActionResult(
            action=action,
            source="rl",
            confidence=self.confidence,
            info={
                "policy_id": self.policy_id,
                "explore": self.explore,
                "throttle_mapped": self.remap_throttle,
            },
        )

    def close(self) -> None:
        if self._owns_algorithm and hasattr(self.algorithm, "stop"):
            self.algorithm.stop()

    def _compute_module_action(self, observation: np.ndarray) -> np.ndarray:
        """Compute an action through Ray 2.54 RLModule inference APIs."""
        import os
        import torch
        from ray.rllib.core.columns import Columns

        module = self.algorithm.get_module(self.policy_id)
        if module is None:
            module = self.algorithm.get_module()
        if module is None:
            raise RuntimeError("Unable to find an RLModule for action inference")

        self._debug_lstm_io = os.environ.get("DOGFIGHT_RNNSAC_DEBUG") == "1"
        self._ensure_module_state(module, torch)

        obs = np.asarray(observation, dtype=np.float32)
        if self._module_state:
            # Recurrent RLModules expect obs as (B, T, obs_dim). This explicit
            # time axis is the inference-side mirror of replay sequence batches.
            obs_batch = obs[None, None, :]
        else:
            obs_batch = obs[None, :]
        batch = {Columns.OBS: torch.as_tensor(obs_batch, dtype=torch.float32)}
        if self._module_state:
            batch[Columns.STATE_IN] = self._module_state
            batch[Columns.SEQ_LENS] = torch.as_tensor([1], dtype=torch.int32)

        with torch.no_grad():
            output = module.forward_inference(batch)
            if Columns.STATE_OUT in output:
                self._module_state = _detach_state(output[Columns.STATE_OUT])
                self._print_lstm_io_debug(batch, output)

            if Columns.ACTIONS in output:
                action = output[Columns.ACTIONS]
            else:
                logits = output[Columns.ACTION_DIST_INPUTS]
                if self.explore:
                    dist_class = module.get_exploration_action_dist_cls()
                else:
                    dist_class = module.get_inference_action_dist_cls()
                action_dist = dist_class.from_logits(logits)
                if not self.explore:
                    action_dist = action_dist.to_deterministic()
                action = action_dist.sample()

        return _to_numpy_action(action)

    def _ensure_module_state(self, module, torch_module) -> None:
        """Create a batched recurrent state if the loaded RLModule is stateful."""
        if self._module_state is not None:
            return
        if not hasattr(module, "get_initial_state"):
            return
        initial_state = module.get_initial_state()
        if not initial_state:
            return
        self._module_state = _batch_state(initial_state, torch_module)

    def _print_lstm_io_debug(self, batch: dict, output: dict) -> None:
        """Print the recurrent inference contract for visual inspection."""
        if not self._debug_lstm_io or self._debug_lstm_prints >= 20:
            return
        from ray.rllib.core.columns import Columns

        obs = batch[Columns.OBS]
        state_in = batch.get(Columns.STATE_IN)
        state_out = output.get(Columns.STATE_OUT)
        print(
            "[DogFightEnv][RLActionProvider][LSTM_IO] "
            f"obs_shape={tuple(obs.shape)} "
            f"seq_lens={batch.get(Columns.SEQ_LENS)} "
            f"state_in={_state_shape_summary(state_in)} "
            f"state_out={_state_shape_summary(state_out)}"
        )
        self._debug_lstm_prints += 1


class DirectSACMLPActionProvider(ActionProvider):
    """Run a standard SAC MLP actor directly without constructing RLlib/Ray."""

    _WEIGHT_KEYS = (
        "pi_encoder.net.mlp.0.weight",
        "pi_encoder.net.mlp.0.bias",
        "pi_encoder.net.mlp.2.weight",
        "pi_encoder.net.mlp.2.bias",
        "pi.net.mlp.0.weight",
        "pi.net.mlp.0.bias",
    )

    def __init__(
        self,
        bundle_dir: str,
        policy_id: str = "default_policy",
        confidence: float = 0.9,
    ) -> None:
        import torch

        bundle_path = Path(bundle_dir)
        self.metadata, _ = load_lightweight_policy_bundle(bundle_path)
        with gzip.open(bundle_path / "policy_weights.pkl.gz", "rb") as handle:
            weights = pickle.load(handle)
        missing = [key for key in self._WEIGHT_KEYS if key not in weights]
        if missing:
            raise ValueError(
                "Direct SAC MLP inference requires standard actor weights; "
                f"missing={missing}"
            )

        activation = str(
            self.metadata.get("algorithm_config", {})
            .get("model_config", {})
            .get("fcnet_activation", "relu")
        ).lower()
        if activation != "relu":
            raise ValueError(
                "Direct SAC MLP inference currently supports relu actors only, "
                f"got {activation!r}"
            )

        self.policy_id = policy_id
        self.confidence = confidence
        self._torch = torch
        self._weights = {
            key: torch.as_tensor(weights[key], dtype=torch.float32)
            for key in self._WEIGHT_KEYS
        }

    def reset(self, context: ActionContext | None = None) -> None:
        return None

    def compute_action(self, context: ActionContext) -> ActionResult:
        import torch.nn.functional as functional

        if context.observation is None:
            raise ValueError("observation is required for direct SAC MLP inference")
        tensor = self._torch.as_tensor(
            np.asarray(context.observation, dtype=np.float32),
            dtype=self._torch.float32,
        )
        with self._torch.no_grad():
            tensor = functional.relu(
                functional.linear(
                    tensor,
                    self._weights["pi_encoder.net.mlp.0.weight"],
                    self._weights["pi_encoder.net.mlp.0.bias"],
                )
            )
            tensor = functional.relu(
                functional.linear(
                    tensor,
                    self._weights["pi_encoder.net.mlp.2.weight"],
                    self._weights["pi_encoder.net.mlp.2.bias"],
                )
            )
            logits = functional.linear(
                tensor,
                self._weights["pi.net.mlp.0.weight"],
                self._weights["pi.net.mlp.0.bias"],
            )
            action = self._torch.tanh(logits[:4]).cpu().numpy()
        return ActionResult(
            action=np.clip(action, -1.0, 1.0).astype(np.float32),
            source="rl",
            confidence=self.confidence,
            info={"policy_id": self.policy_id, "explore": False, "direct_sac_mlp": True},
        )

    def close(self) -> None:
        return None


class DirectPPOMLPActionProvider(ActionProvider):
    """Run a standard deterministic PPO MLP actor without RLlib/Ray."""

    _WEIGHT_KEYS = (
        "encoder.encoder.net.mlp.0.weight",
        "encoder.encoder.net.mlp.0.bias",
        "encoder.encoder.net.mlp.2.weight",
        "encoder.encoder.net.mlp.2.bias",
        "pi.net.mlp.0.weight",
        "pi.net.mlp.0.bias",
    )

    def __init__(
        self,
        bundle_dir: str,
        policy_id: str = "default_policy",
        confidence: float = 0.9,
    ) -> None:
        import torch

        bundle_path = Path(bundle_dir)
        self.metadata, _ = load_lightweight_policy_bundle(bundle_path)
        with gzip.open(bundle_path / "policy_weights.pkl.gz", "rb") as handle:
            weights = pickle.load(handle)
        missing = [key for key in self._WEIGHT_KEYS if key not in weights]
        if missing:
            raise ValueError(
                "Direct PPO MLP inference requires standard actor weights; "
                f"missing={missing}"
            )

        activation = str(
            self.metadata.get("algorithm_config", {})
            .get("model_config", {})
            .get("fcnet_activation", "relu")
        ).lower()
        if activation != "relu":
            raise ValueError(
                "Direct PPO MLP inference currently supports relu actors only, "
                f"got {activation!r}"
            )

        head_shape = tuple(np.asarray(weights["pi.net.mlp.0.weight"]).shape)
        if len(head_shape) != 2 or head_shape[0] != 8:
            raise ValueError(
                "Direct PPO MLP inference expects an 8-value diagonal-Gaussian "
                f"policy head for four actions, got shape={head_shape}"
            )
        self.policy_id = policy_id
        self.confidence = confidence
        self._torch = torch
        self._weights = {
            key: torch.as_tensor(weights[key], dtype=torch.float32)
            for key in self._WEIGHT_KEYS
        }

    def reset(self, context: ActionContext | None = None) -> None:
        return None

    def compute_action(self, context: ActionContext) -> ActionResult:
        import torch.nn.functional as functional

        if context.observation is None:
            raise ValueError("observation is required for direct PPO MLP inference")
        tensor = self._torch.as_tensor(
            np.asarray(context.observation, dtype=np.float32),
            dtype=self._torch.float32,
        )
        with self._torch.no_grad():
            tensor = functional.relu(
                functional.linear(
                    tensor,
                    self._weights["encoder.encoder.net.mlp.0.weight"],
                    self._weights["encoder.encoder.net.mlp.0.bias"],
                )
            )
            tensor = functional.relu(
                functional.linear(
                    tensor,
                    self._weights["encoder.encoder.net.mlp.2.weight"],
                    self._weights["encoder.encoder.net.mlp.2.bias"],
                )
            )
            logits = functional.linear(
                tensor,
                self._weights["pi.net.mlp.0.weight"],
                self._weights["pi.net.mlp.0.bias"],
            )
            action = logits[:4].cpu().numpy()
        return ActionResult(
            action=np.clip(action, -1.0, 1.0).astype(np.float32),
            source="rl",
            confidence=self.confidence,
            info={"policy_id": self.policy_id, "explore": False, "direct_ppo_mlp": True},
        )

    def close(self) -> None:
        return None


class DirectPPOLSTMActionProvider(ActionProvider):
    """Run a standard deterministic PPO tokenizer/LSTM actor without Ray."""

    _WEIGHT_KEYS = (
        "encoder.encoder.tokenizer.net.mlp.0.weight",
        "encoder.encoder.tokenizer.net.mlp.0.bias",
        "encoder.encoder.tokenizer.net.mlp.2.weight",
        "encoder.encoder.tokenizer.net.mlp.2.bias",
        "encoder.encoder.lstm.weight_ih_l0",
        "encoder.encoder.lstm.weight_hh_l0",
        "encoder.encoder.lstm.bias_ih_l0",
        "encoder.encoder.lstm.bias_hh_l0",
        "pi.net.mlp.0.weight",
        "pi.net.mlp.0.bias",
    )

    def __init__(
        self,
        bundle_dir: str,
        policy_id: str = "default_policy",
        confidence: float = 0.9,
    ) -> None:
        import torch

        bundle_path = Path(bundle_dir)
        self.metadata, _ = load_lightweight_policy_bundle(bundle_path)
        with gzip.open(bundle_path / "policy_weights.pkl.gz", "rb") as handle:
            weights = pickle.load(handle)
        missing = [key for key in self._WEIGHT_KEYS if key not in weights]
        if missing:
            raise ValueError(
                "Direct PPO LSTM inference requires standard recurrent weights; "
                f"missing={missing}"
            )
        activation = str(
            self.metadata.get("algorithm_config", {})
            .get("model_config", {})
            .get("fcnet_activation", "relu")
        ).lower()
        if activation != "relu":
            raise ValueError(
                "Direct PPO LSTM inference currently supports relu tokenizers only, "
                f"got {activation!r}"
            )

        first_weight = np.asarray(
            weights["encoder.encoder.tokenizer.net.mlp.0.weight"]
        )
        second_weight = np.asarray(
            weights["encoder.encoder.tokenizer.net.mlp.2.weight"]
        )
        recurrent_weight = np.asarray(weights["encoder.encoder.lstm.weight_hh_l0"])
        head_weight = np.asarray(weights["pi.net.mlp.0.weight"])
        hidden_size = int(recurrent_weight.shape[1])
        if recurrent_weight.shape[0] != 4 * hidden_size:
            raise ValueError(f"invalid PPO LSTM recurrent shape={recurrent_weight.shape}")
        if head_weight.shape != (8, hidden_size):
            raise ValueError(
                "Direct PPO LSTM expects an 8-value policy head for four actions, "
                f"got shape={head_weight.shape}"
            )

        self.policy_id = policy_id
        self.confidence = confidence
        self._torch = torch
        self._token1 = torch.nn.Linear(first_weight.shape[1], first_weight.shape[0])
        self._token2 = torch.nn.Linear(second_weight.shape[1], second_weight.shape[0])
        self._lstm = torch.nn.LSTM(second_weight.shape[0], hidden_size, batch_first=True)
        self._head = torch.nn.Linear(hidden_size, 8)
        with torch.no_grad():
            self._token1.weight.copy_(torch.as_tensor(first_weight, dtype=torch.float32))
            self._token1.bias.copy_(torch.as_tensor(weights[self._WEIGHT_KEYS[1]], dtype=torch.float32))
            self._token2.weight.copy_(torch.as_tensor(second_weight, dtype=torch.float32))
            self._token2.bias.copy_(torch.as_tensor(weights[self._WEIGHT_KEYS[3]], dtype=torch.float32))
            self._lstm.weight_ih_l0.copy_(torch.as_tensor(weights[self._WEIGHT_KEYS[4]], dtype=torch.float32))
            self._lstm.weight_hh_l0.copy_(torch.as_tensor(recurrent_weight, dtype=torch.float32))
            self._lstm.bias_ih_l0.copy_(torch.as_tensor(weights[self._WEIGHT_KEYS[6]], dtype=torch.float32))
            self._lstm.bias_hh_l0.copy_(torch.as_tensor(weights[self._WEIGHT_KEYS[7]], dtype=torch.float32))
            self._head.weight.copy_(torch.as_tensor(head_weight, dtype=torch.float32))
            self._head.bias.copy_(torch.as_tensor(weights[self._WEIGHT_KEYS[9]], dtype=torch.float32))
        for module in (self._token1, self._token2, self._lstm, self._head):
            module.eval()
        self._state = None

    def reset(self, context: ActionContext | None = None) -> None:
        self._state = None

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.observation is None:
            raise ValueError("observation is required for direct PPO LSTM inference")
        tensor = self._torch.as_tensor(
            np.asarray(context.observation, dtype=np.float32),
            dtype=self._torch.float32,
        ).reshape(1, -1)
        with self._torch.no_grad():
            token = self._torch.relu(self._token1(tensor))
            token = self._torch.relu(self._token2(token)).unsqueeze(1)
            output, state = self._lstm(token, self._state)
            self._state = tuple(item.detach() for item in state)
            logits = self._head(output[:, -1, :])[0]
            action = logits[:4].cpu().numpy()
        return ActionResult(
            action=np.clip(action, -1.0, 1.0).astype(np.float32),
            source="rl",
            confidence=self.confidence,
            info={"policy_id": self.policy_id, "explore": False, "direct_ppo_lstm": True},
        )

    def close(self) -> None:
        self._state = None


class ObservationFeedbackActionProvider(ActionProvider):
    """Add a bounded linear observation feedback term to a policy action."""

    def __init__(
        self,
        primary: ActionProvider,
        *,
        observation_index: int,
        action_gains: np.ndarray,
    ) -> None:
        self.primary = primary
        self.observation_index = int(observation_index)
        self.action_gains = np.asarray(action_gains, dtype=np.float32)
        if self.action_gains.shape != (4,):
            raise ValueError("action_gains must contain four values")

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.observation is None:
            raise ValueError("observation is required for observation feedback")
        observation = np.asarray(context.observation, dtype=np.float32)
        if not 0 <= self.observation_index < observation.size:
            raise IndexError(
                f"observation index {self.observation_index} is outside "
                f"a {observation.size}-element observation"
            )
        result = self.primary.compute_action(context)
        feedback = float(observation[self.observation_index]) * self.action_gains
        return ActionResult(
            action=np.clip(
                np.asarray(result.action, dtype=np.float32) + feedback,
                -1.0,
                1.0,
            ).astype(np.float32),
            source=result.source,
            confidence=result.confidence,
            info={
                **result.info,
                "observation_feedback": feedback.tolist(),
            },
        )

    def close(self) -> None:
        self.primary.close()


class TargetBankActionScaleProvider(ActionProvider):
    """Scale an existing policy's roll/rudder commands by observed target bank."""

    def __init__(
        self,
        primary: ActionProvider,
        *,
        reference_bank_deg: float = 30.0,
        roll_exponent: float = 1.0,
        rudder_exponent: float = 1.0,
        minimum_scale: float = 0.5,
        maximum_scale: float = 2.5,
    ) -> None:
        if reference_bank_deg <= 0.0:
            raise ValueError("reference_bank_deg must be positive")
        self.primary = primary
        self.reference_bank_deg = float(reference_bank_deg)
        self.roll_exponent = float(roll_exponent)
        self.rudder_exponent = float(rudder_exponent)
        self.minimum_scale = float(minimum_scale)
        self.maximum_scale = float(maximum_scale)

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.primary.compute_action(context)
        if context.target_state is None:
            return result
        target_roll = (
            float(context.target_state[StateIndex.ROLL]) + 180.0
        ) % 360.0 - 180.0
        ratio = max(abs(target_roll) / self.reference_bank_deg, 0.0)
        roll_scale = float(np.clip(
            ratio ** self.roll_exponent,
            self.minimum_scale,
            self.maximum_scale,
        ))
        rudder_scale = float(np.clip(
            ratio ** self.rudder_exponent,
            self.minimum_scale,
            self.maximum_scale,
        ))
        action = np.asarray(result.action, dtype=np.float32).copy()
        action[0] *= roll_scale
        action[2] *= rudder_scale
        return ActionResult(
            action=np.clip(action, -1.0, 1.0).astype(np.float32),
            source=result.source,
            confidence=result.confidence,
            info={
                **result.info,
                "target_bank_action_scale": True,
                "target_bank_deg": target_roll,
                "target_bank_roll_scale": roll_scale,
                "target_bank_rudder_scale": rudder_scale,
            },
        )

    def close(self) -> None:
        self.primary.close()


class NegativeTargetBankCanonicalActionProvider(ActionProvider):
    """Mirror negative-turn observations into the actor's positive-turn frame."""

    def __init__(self, primary: ActionProvider) -> None:
        self.primary = primary

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.target_state is None or context.observation is None:
            return self.primary.compute_action(context)
        target_roll = (
            float(context.target_state[StateIndex.ROLL]) + 180.0
        ) % 360.0 - 180.0
        if target_roll >= 0.0:
            return self.primary.compute_action(context)

        observation = mirror_tactical_observation(context.observation)
        mirrored_context = ActionContext(
            sim=context.sim,
            opponent_sim=context.opponent_sim,
            ownship_state=context.ownship_state,
            target_state=context.target_state,
            observation=observation,
            info=context.info,
        )
        result = self.primary.compute_action(mirrored_context)
        action = np.asarray(result.action, dtype=np.float32).copy()
        action[[0, 2]] *= -1.0
        return ActionResult(
            action=np.clip(action, -1.0, 1.0).astype(np.float32),
            source=result.source,
            confidence=result.confidence,
            info={
                **result.info,
                "negative_target_bank_canonicalized": True,
                "target_bank_deg": target_roll,
            },
        )

    def close(self) -> None:
        self.primary.close()


class CanonicalTargetBankActionRestoreProvider(ActionProvider):
    """Restore actor actions when the environment already mirrors observations."""

    def __init__(self, primary: ActionProvider) -> None:
        self.primary = primary

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.primary.compute_action(context)
        if context.target_state is None:
            return result
        target_roll = (
            float(context.target_state[StateIndex.ROLL]) + 180.0
        ) % 360.0 - 180.0
        if target_roll >= 0.0:
            return result
        action = np.asarray(result.action, dtype=np.float32).copy()
        action[[0, 2]] *= -1.0
        return ActionResult(
            action=np.clip(action, -1.0, 1.0).astype(np.float32),
            source=result.source,
            confidence=result.confidence,
            info={
                **result.info,
                "canonical_target_bank_action_restored": True,
                "target_bank_deg": target_roll,
            },
        )

    def close(self) -> None:
        self.primary.close()


class BundleMappedDirectSACActionProvider(ActionProvider):
    """Run a direct SAC actor and return simulator-mapped bundle actions."""

    def __init__(
        self,
        bundle_dir: str,
        *,
        policy_bias: np.ndarray | None = None,
    ) -> None:
        self.primary = DirectSACMLPActionProvider(bundle_dir=bundle_dir)
        self.env_config = dict(
            self.primary.metadata.get("algorithm_config", {}).get("env_config", {})
        )
        self.observation_mode = str(
            self.env_config.get("observation_mode", "tactical16")
        )
        self.policy_bias = np.asarray(
            np.zeros(4, dtype=np.float32) if policy_bias is None else policy_bias,
            dtype=np.float32,
        )
        if self.policy_bias.shape != (4,):
            raise ValueError("policy_bias must contain four values")

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary.reset(context)

    def _adapt_observation(self, observation: np.ndarray) -> np.ndarray:
        observation = np.asarray(observation, dtype=np.float32)
        # tactical21 is tactical17 plus four target-maneuver features, and
        # tactical25 appends four angular-rate features. Older bundles must not
        # receive columns they were not trained to consume.
        if observation.size == 25 and self.observation_mode in {
            "tactical16",
            "tactical17",
            "tactical21",
        }:
            observation = observation[:21]
        if observation.size == 21 and self.observation_mode in {
            "tactical16",
            "tactical17",
        }:
            observation = observation[:17]
        if self.observation_mode == "tactical16" and observation.size == 17:
            yaw_deg = np.degrees(
                np.arctan2(float(observation[2]), float(observation[3]))
            ) % 360.0
            tactical16 = np.empty(16, dtype=np.float32)
            tactical16[0:2] = observation[0:2]
            tactical16[2] = yaw_deg / 180.0 - 1.0
            tactical16[3:] = observation[4:]
            return tactical16
        if self.observation_mode == "tactical17" and observation.size == 16:
            yaw_rad = (float(observation[2]) + 1.0) * np.pi
            tactical17 = np.empty(17, dtype=np.float32)
            tactical17[0:2] = observation[0:2]
            tactical17[2] = np.sin(yaw_rad)
            tactical17[3] = np.cos(yaw_rad)
            tactical17[4:] = observation[3:]
            return tactical17
        if self.observation_mode == "tactical21" and observation.size in {16, 17}:
            if observation.size == 16:
                yaw_rad = (float(observation[2]) + 1.0) * np.pi
                tactical17 = np.empty(17, dtype=np.float32)
                tactical17[0:2] = observation[0:2]
                tactical17[2] = np.sin(yaw_rad)
                tactical17[3] = np.cos(yaw_rad)
                tactical17[4:] = observation[3:]
                observation = tactical17
            tactical21 = np.zeros(21, dtype=np.float32)
            tactical21[:17] = observation
            return tactical21
        if self.observation_mode == "tactical25" and observation.size in {16, 17, 21}:
            if observation.size in {16, 17}:
                original_mode = self.observation_mode
                self.observation_mode = "tactical21"
                try:
                    observation = self._adapt_observation(observation)
                finally:
                    self.observation_mode = original_mode
            tactical25 = np.zeros(25, dtype=np.float32)
            tactical25[:21] = observation
            return tactical25
        return observation

    def _map_action(self, action: np.ndarray) -> np.ndarray:
        mapped = np.clip(action, -1.0, 1.0).astype(np.float32)
        residual = self.env_config.get("residual_action", {})
        if bool(residual.get("enabled", False)):
            baseline = np.asarray(residual["baseline"], dtype=np.float32)
            scale = np.asarray(residual["scale"], dtype=np.float32)
            mapped = np.clip(baseline + scale * mapped, -1.0, 1.0)
        mapped[:3] *= np.clip(
            float(self.env_config.get("control_surface_scale", 1.0)),
            0.0,
            1.0,
        )
        mapped[3] = (mapped[3] + 1.0) / 2.0
        return clip_action(mapped)

    def compute_action(self, context: ActionContext) -> ActionResult:
        adapted_context = ActionContext(
            sim=context.sim,
            opponent_sim=context.opponent_sim,
            ownship_state=context.ownship_state,
            target_state=context.target_state,
            observation=self._adapt_observation(context.observation),
            info=context.info,
        )
        result = self.primary.compute_action(adapted_context)
        biased_action = np.clip(
            np.asarray(result.action, dtype=np.float32) + self.policy_bias,
            -1.0,
            1.0,
        ).astype(np.float32)
        return ActionResult(
            action=self._map_action(biased_action),
            source="bundle_mapped",
            confidence=result.confidence,
            info={
                **result.info,
                "bundle_action_mapped": True,
                "bundle_policy_bias": self.policy_bias.tolist(),
            },
        )

    def close(self) -> None:
        self.primary.close()


class AltitudeRecoveryActionProvider(ActionProvider):
    """Use a second policy below a guarded altitude with hysteresis."""

    def __init__(
        self,
        primary: ActionProvider,
        recovery: ActionProvider,
        *,
        trigger_altitude_m: float,
        release_altitude_m: float,
        primary_bias: np.ndarray | None = None,
    ) -> None:
        if trigger_altitude_m <= 0.0:
            raise ValueError("trigger_altitude_m must be positive")
        if release_altitude_m <= trigger_altitude_m:
            raise ValueError("release_altitude_m must exceed trigger_altitude_m")
        self.primary = primary
        self.recovery = recovery
        self.trigger_altitude_m = float(trigger_altitude_m)
        self.release_altitude_m = float(release_altitude_m)
        self.primary_bias = np.asarray(
            np.zeros(4, dtype=np.float32) if primary_bias is None else primary_bias,
            dtype=np.float32,
        )
        if self.primary_bias.shape != (4,):
            raise ValueError("primary_bias must contain four values")
        self._recovery_active = False

    def reset(self, context: ActionContext | None = None) -> None:
        self._recovery_active = False
        self.primary.reset(context)
        self.recovery.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.ownship_state is None:
            raise ValueError("ownship_state is required for altitude recovery")
        state = np.asarray(context.ownship_state)
        altitude_m = float(state[44])
        if altitude_m <= 0.0:
            altitude_m = -float(state[2])
        if self._recovery_active:
            self._recovery_active = altitude_m < self.release_altitude_m
        else:
            self._recovery_active = altitude_m <= self.trigger_altitude_m

        if self._recovery_active:
            result = self.recovery.compute_action(context)
            return ActionResult(
                action=result.action,
                source=result.source,
                confidence=result.confidence,
                info={**result.info, "altitude_recovery": True},
            )

        result = self.primary.compute_action(context)
        return ActionResult(
            action=clip_action(np.asarray(result.action) + self.primary_bias),
            source=result.source,
            confidence=result.confidence,
            info={**result.info, "altitude_recovery": False},
        )

    def close(self) -> None:
        self.primary.close()
        self.recovery.close()


def _to_numpy_action(action) -> np.ndarray:
    if hasattr(action, "detach"):
        action = action.detach().cpu().numpy()
    action_array = np.asarray(action, dtype=np.float32)
    while action_array.ndim > 1 and action_array.shape[0] == 1:
        action_array = action_array[0]
    return action_array


def _batch_state(state, torch_module):
    if isinstance(state, dict):
        return {key: _batch_state(value, torch_module) for key, value in state.items()}
    tensor = state if hasattr(state, "detach") else torch_module.as_tensor(state)
    return tensor.detach().clone().float().unsqueeze(0)


def _detach_state(state):
    if isinstance(state, dict):
        return {key: _detach_state(value) for key, value in state.items()}
    return state.detach().clone() if hasattr(state, "detach") else state


def _state_shape_summary(state) -> str:
    if state is None:
        return "None"
    if isinstance(state, dict):
        return "{" + ", ".join(
            f"{key}: {_state_shape_summary(value)}"
            for key, value in state.items()
        ) + "}"
    shape = getattr(state, "shape", None)
    return str(tuple(shape)) if shape is not None else type(state).__name__
