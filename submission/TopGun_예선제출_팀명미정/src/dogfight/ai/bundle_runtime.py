from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Callable

import numpy as np

from dogfight.ai.action_provider import ActionContext, ActionProvider, ActionResult
from dogfight.ai.hybrid_action_provider import (
    GuidanceResidualActionProvider,
    build_pursuit_guidance_provider,
)
from dogfight.ai.rl_action_provider import (
    DirectSACMLPActionProvider,
    NegativeTargetBankCanonicalActionProvider,
    RLActionProvider,
)
@dataclass(frozen=True)
class BundleRuntime:
    """Resolved inference contract for one lightweight policy bundle."""

    provider: ActionProvider
    metadata: dict
    env_config: dict
    observation_mode: str
    observation_module: str
    action_repeat: int
    simulator_actions: bool


class SavedObservationFeedbackActionProvider(ActionProvider):
    """Apply the observation feedback saved in a bundle's env config."""

    def __init__(self, primary: ActionProvider, config: dict) -> None:
        self.primary = primary
        self.config = dict(config)
        self._branch_index: int | None = None

    def reset(self, context: ActionContext | None = None) -> None:
        self._branch_index = None
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        if context.observation is None:
            raise ValueError("observation is required for saved observation feedback")
        observation = np.asarray(context.observation, dtype=np.float32)
        observation_index = int(self.config.get("observation_index", 0))
        if not 0 <= observation_index < observation.size:
            raise IndexError(
                f"observation feedback index {observation_index} is outside "
                f"a {observation.size}-element observation"
            )

        selected = self.config
        branches = self.config.get("branches")
        if branches:
            branch_index = self._branch_index
            if branch_index is None or not bool(self.config.get("latch_branch", False)):
                feature_index = int(
                    self.config.get("branch_observation_index", observation_index)
                )
                if not 0 <= feature_index < observation.size:
                    raise IndexError(
                        f"observation feedback branch index {feature_index} is "
                        f"outside a {observation.size}-element observation"
                    )
                feature_value = float(observation[feature_index])
                branch_index = len(branches) - 1
                for index, branch in enumerate(branches):
                    maximum = branch.get("max_value")
                    if maximum is None or feature_value <= float(maximum):
                        branch_index = index
                        break
                if bool(self.config.get("latch_branch", False)):
                    self._branch_index = int(branch_index)
            selected = branches[int(branch_index)]

        gains = np.asarray(
            selected.get("action_gains", [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        bias = np.asarray(
            selected.get("action_bias", [0.0, 0.0, 0.0, 0.0]),
            dtype=np.float32,
        )
        if gains.shape != (4,) or bias.shape != (4,):
            raise ValueError("saved observation feedback gains and bias must have 4 values")

        result = self.primary.compute_action(context)
        feedback = bias + float(observation[observation_index]) * gains
        action = np.clip(
            np.asarray(result.action, dtype=np.float32) + feedback,
            -1.0,
            1.0,
        ).astype(np.float32)
        return ActionResult(
            action=action,
            source=result.source,
            confidence=result.confidence,
            info={
                **result.info,
                "saved_observation_feedback": feedback.tolist(),
                "saved_observation_feedback_branch": self._branch_index,
            },
        )

    def close(self) -> None:
        self.primary.close()


class PolicyToSimulatorActionProvider(ActionProvider):
    """Convert a final policy-space action to the Unreal/simulator contract.

    All policy composition happens in ``[-1, 1]^4``.  This outermost adapter
    applies any saved residual/control scaling and converts throttle exactly
    once from ``[-1, 1]`` to ``[0, 1]``.
    """

    def __init__(self, primary: ActionProvider, env_config: dict | None = None) -> None:
        self.primary = primary
        self.env_config = dict(env_config or {})

    def reset(self, context: ActionContext | None = None) -> None:
        self.primary.reset(context)

    def compute_action(self, context: ActionContext) -> ActionResult:
        result = self.primary.compute_action(context)
        action = np.clip(result.action, -1.0, 1.0).astype(np.float32)

        residual = self.env_config.get("residual_action", {})
        if bool(residual.get("enabled", False)):
            baseline = np.asarray(
                residual.get("baseline", [0.0, 0.0, 0.0, 0.0]),
                dtype=np.float32,
            )
            scale = np.asarray(
                residual.get("scale", [1.0, 1.0, 1.0, 1.0]),
                dtype=np.float32,
            )
            if baseline.shape != (4,) or scale.shape != (4,):
                raise ValueError("residual_action baseline and scale must have 4 values")
            action = np.clip(baseline + scale * action, -1.0, 1.0)

        action[:3] *= np.clip(
            float(self.env_config.get("control_surface_scale", 1.0)),
            0.0,
            1.0,
        )
        action[3] = (action[3] + 1.0) / 2.0
        action = np.clip(
            action,
            np.asarray([-1.0, -1.0, -1.0, 0.0], dtype=np.float32),
            np.ones(4, dtype=np.float32),
        ).astype(np.float32)
        return ActionResult(
            action=action,
            source="sim_command",
            confidence=result.confidence,
            info={
                **result.info,
                "policy_source": result.source,
                "simulator_action_mapped": True,
            },
        )

    def close(self) -> None:
        self.primary.close()


def load_bundle_metadata(bundle_dir: str | Path) -> tuple[dict, dict]:
    bundle_path = Path(bundle_dir).resolve()
    metadata_path = bundle_path / "metadata.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"bundle metadata not found: {metadata_path}")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    env_config = dict(metadata.get("algorithm_config", {}).get("env_config", {}))
    return metadata, env_config


def build_bundle_runtime(
    bundle_dir: str | Path,
    *,
    for_unreal: bool,
    policy_id: str = "default_policy",
    explore: bool = False,
    use_direct_mlp: bool = False,
    algorithm_factory: Callable[[dict], object] | None = None,
) -> BundleRuntime:
    """Build the wrappers saved by training for local or Unreal inference."""

    bundle_path = Path(bundle_dir).resolve()
    metadata, env_config = load_bundle_metadata(bundle_path)
    bundle_metadata = dict(metadata.get("metadata", {}))
    use_lstm = bool(
        bundle_metadata.get("use_lstm_sac", bundle_metadata.get("use_lstm", False))
    )

    if use_direct_mlp:
        if use_lstm:
            raise ValueError("direct MLP inference cannot load a recurrent bundle")
        provider: ActionProvider = DirectSACMLPActionProvider(
            bundle_dir=str(bundle_path),
            policy_id=policy_id,
        )
    else:
        factory = algorithm_factory
        if factory is None:
            # Keep the normal submission path free of Ray/RLlib imports.  Ray
            # is only required for recurrent bundles or compatibility restore.
            from dogfight.ai.rllib_utils import (
                build_algorithm_from_bundle,
                build_sac_lstm_module_from_bundle,
            )

            factory = (
                build_sac_lstm_module_from_bundle
                if use_lstm
                else build_algorithm_from_bundle
            )
        provider = RLActionProvider(
            bundle_dir=str(bundle_path),
            algorithm_factory=factory,
            policy_id=policy_id,
            explore=explore,
            remap_throttle=False,
        )

    if bool(env_config.get("canonicalize_negative_target_bank", False)):
        provider = NegativeTargetBankCanonicalActionProvider(provider)
        if not for_unreal:
            # DogFightEnv must not mirror/restore again around this provider.
            env_config["canonicalize_negative_target_bank"] = False

    guidance = dict(env_config.get("pursuit_guidance_residual", {}))
    if bool(guidance.get("enabled", False)):
        provider = GuidanceResidualActionProvider(
            provider,
            build_pursuit_guidance_provider(guidance),
            residual_scale=guidance.get("residual_scale", [0.25, 0.25, 0.25, 0.10]),
            near_residual_scale=guidance.get("near_residual_scale"),
            near_distance_m=guidance.get("near_distance_m", 0.0),
            near_ata_deg=guidance.get("near_ata_deg", 180.0),
            suppress_opposing_roll_heading_error_deg=guidance.get(
                "suppress_opposing_roll_heading_error_deg", 0.0
            ),
            opposing_roll_guidance_scale=guidance.get(
                "opposing_roll_guidance_scale", 1.0
            ),
        )

    feedback = dict(env_config.get("observation_action_feedback", {}))
    if for_unreal and bool(feedback.get("enabled", False)):
        # DogFightEnv applies this after its ownship provider; Unreal does not.
        provider = SavedObservationFeedbackActionProvider(provider, feedback)

    if for_unreal:
        provider = PolicyToSimulatorActionProvider(provider, env_config)

    observation_mode = str(
        env_config.get(
            "observation_mode",
            bundle_metadata.get("observation_mode", bundle_metadata.get("obs_mode", "tactical16")),
        )
    )
    observation_module = str(
        env_config.get(
            "observation_module",
            bundle_metadata.get("observation_module", ""),
        )
        or ""
    )
    action_repeat = max(1, int(float(env_config.get("step_ratio", 1))))
    return BundleRuntime(
        provider=provider,
        metadata=metadata,
        env_config=env_config,
        observation_mode=observation_mode,
        observation_module=observation_module,
        action_repeat=action_repeat,
        simulator_actions=for_unreal,
    )


__all__ = [
    "BundleRuntime",
    "PolicyToSimulatorActionProvider",
    "SavedObservationFeedbackActionProvider",
    "build_bundle_runtime",
    "load_bundle_metadata",
]
