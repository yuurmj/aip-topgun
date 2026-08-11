from __future__ import annotations

from pathlib import Path
from typing import Callable

import gzip
import pickle

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult, clip_action
from dogfight.ai.checkpoint_io import load_lightweight_policy_bundle


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
    ):
        self.policy_id = policy_id
        self.explore = explore
        self.obs_builder = obs_builder
        self.confidence = confidence
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

        action = self._compute_module_action(observation)
        action = clip_action(action)

        return ActionResult(
            action=action,
            source="rl",
            confidence=self.confidence,
            info={"policy_id": self.policy_id, "explore": self.explore},
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
            action=clip_action(action),
            source="rl",
            confidence=self.confidence,
            info={"policy_id": self.policy_id, "explore": False, "direct_sac_mlp": True},
        )

    def close(self) -> None:
        return None


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
