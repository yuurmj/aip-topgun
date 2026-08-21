"""Public AI helpers with lazy imports for lightweight submission startup."""

from importlib import import_module


_EXPORTS = {
    "ActionContext": ("action_provider", "ActionContext"),
    "ActionProvider": ("action_provider", "ActionProvider"),
    "ActionResult": ("action_provider", "ActionResult"),
    "AIPilot": ("native_bt", "AIPilot"),
    "BTActionProvider": ("bt_action_provider", "BTActionProvider"),
    "activate_rule_xml": ("bt_rule_manager", "activate_rule_xml"),
    "DashboardJsonlLogger": ("dashboard_logger", "DashboardJsonlLogger"),
    "copy_experiment_yaml": ("dashboard_logger", "copy_experiment_yaml"),
    "load_experiment_metadata": ("dashboard_logger", "load_experiment_metadata"),
    "training_row_to_dashboard_metrics": ("dashboard_logger", "training_row_to_dashboard_metrics"),
    "HybridActionProvider": ("hybrid_action_provider", "HybridActionProvider"),
    "RLActionProvider": ("rl_action_provider", "RLActionProvider"),
    "load_lightweight_policy_bundle": ("checkpoint_io", "load_lightweight_policy_bundle"),
    "save_lightweight_policy_bundle": ("checkpoint_io", "save_lightweight_policy_bundle"),
    "build_algorithm_config": ("rllib_utils", "build_algorithm_config"),
    "build_algorithm_from_bundle": ("rllib_utils", "build_algorithm_from_bundle"),
    "build_sac_lstm_module_from_bundle": ("rllib_utils", "build_sac_lstm_module_from_bundle"),
    "normalize_algorithm_name": ("rllib_utils", "normalize_algorithm_name"),
    "save_training_record": ("training_record", "save_training_record"),
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    try:
        module_name, attribute = _EXPORTS[name]
    except KeyError as exc:
        raise AttributeError(name) from exc
    value = getattr(import_module(f"{__name__}.{module_name}"), attribute)
    globals()[name] = value
    return value
