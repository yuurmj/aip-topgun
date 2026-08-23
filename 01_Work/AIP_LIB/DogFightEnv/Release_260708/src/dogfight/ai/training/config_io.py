from __future__ import annotations

from pathlib import Path

import yaml


def deep_update(base: dict, updates: dict) -> dict:
    """Recursively merge nested experiment config into base in place."""
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_experiment_env_config(path: str, root: Path) -> dict:
    """Load env_config from an experiment YAML path."""
    if not path:
        return {}

    exp_path = Path(path)
    if not exp_path.is_absolute():
        exp_path = (root / exp_path).resolve()
    data = _load_experiment_mapping(exp_path, set())
    if not isinstance(data, dict):
        return {}
    env_config = data.get("env_config", {})
    if env_config is None:
        return {}
    if not isinstance(env_config, dict):
        raise ValueError("experiment YAML env_config must be a mapping")
    return env_config


def _load_experiment_mapping(path: Path, seen: set[Path]) -> dict:
    """Load a YAML mapping and recursively apply a relative ``extends``."""
    resolved = path.resolve()
    if resolved in seen:
        raise ValueError(f"cyclic experiment extends chain at: {resolved}")
    next_seen = set(seen)
    next_seen.add(resolved)
    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError("experiment YAML root must be a mapping")
    extends = data.pop("extends", None)
    if not extends:
        return data
    base_path = Path(str(extends))
    if not base_path.is_absolute():
        base_path = resolved.parent / base_path
    base = _load_experiment_mapping(base_path, next_seen)
    return deep_update(base, data)
