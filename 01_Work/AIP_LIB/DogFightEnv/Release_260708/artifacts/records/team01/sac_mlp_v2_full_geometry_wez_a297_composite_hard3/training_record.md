# Training Record

- Created at: `2026-08-13T13:20:56`
- Python: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`
- Algorithm: `sac`

## Observation

- Mode: `tactical17`
- Size: `17`
- Description: Periodic-yaw tactical observation. Yaw is encoded as sine and cosine so 359 and 0 degrees remain adjacent; all features are bounded to [-1, 1].
- Features:
  - `ownship_roll_norm`
  - `ownship_pitch_norm`
  - `ownship_yaw_sin`
  - `ownship_yaw_cos`
  - `ownship_speed_norm`
  - `ownship_alt_norm`
  - `ownship_health_norm`
  - `delta_n_norm`
  - `delta_e_norm`
  - `delta_d_norm`
  - `ata_norm`
  - `aa_norm`
  - `az_norm`
  - `el_norm`
  - `target_health_norm`
  - `in_wez`
  - `pursuit_score_norm`

## Reward

- Description: Survival bonus (curriculum) + step penalty + pursuit shaping (smooth ATA×range gradient) + damage differential + low altitude penalty + terminal rewards.
- Step penalty: `-0.01`
- Damage scale: `200.0`
- Pursuit scale: `0.3`
- Pursuit half angle (deg): `60.0`
- Pursuit range (m): `8000.0`
- Low altitude penalty: `None`
- Win reward: `0.0`
- Loss reward: `-100.0`
- Draw reward: `-10.0`

## CLI Arguments

```json
{
  "algorithm": "sac",
  "iterations": 9,
  "framework": "torch",
  "num_env_runners": 0,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical17",
  "observation_module": "",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_wez_range_reward_v6",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "lr": 0.0003,
  "actor_lr": 3e-05,
  "critic_lr": 0.0003,
  "alpha_lr": 1e-05,
  "gamma": 0.99,
  "train_batch_size": 256,
  "minibatch_size": 256,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": 30000,
  "learning_starts": 200,
  "model_fcnet_hiddens": "256,256",
  "model_fcnet_activation": "relu",
  "model_head_fcnet_hiddens": "",
  "model_head_fcnet_activation": "relu",
  "model_vf_share_layers": null,
  "network_spec_json": "",
  "use_lstm": false,
  "use_lstm_sac": false,
  "lstm_scope": "actor_only",
  "lstm_cell_size": 64,
  "max_seq_len": 8,
  "debug_io": false,
  "use_lstm_prioritized_replay": false,
  "output_name": "team01",
  "output_tag": "sac_mlp_v2_full_geometry_wez_a297_composite_hard3",
  "notes": "A297 common composite close-guidance teacher for the hard scenario-5 seeds.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 3,
  "save_native_checkpoint": false,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_mlp_v2_full_geometry_wez_a204_tactical17_60s/bundle_000002",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": null,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 3,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\dev\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_mlp_v2_full_geometry_wez_a297_composite_hard3.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical17",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "step_ratio": 6,
  "min_altitude": 300.0,
  "control_surface_scale": 1.0,
  "episode_seed_cycle": [
    1003,
    1004,
    1006
  ],
  "pursuit_guidance_residual": {
    "enabled": true,
    "residual_scale": [
      0.4,
      0.4,
      0.4,
      0.15
    ],
    "max_bank_deg": 70.0,
    "heading_to_bank": 2.5,
    "roll_gain": 0.002,
    "pitch_gain": 0.05,
    "capture_pitch_gain": 0.3,
    "pitch_command_sign": -1.0,
    "rudder_gain": 0.0,
    "lead_time_s": 12.0,
    "brake_range_m": 1800.0,
    "brake_heading_error_deg": 10.0,
    "near_throttle_action": -1.0,
    "capture_range_m": 1100.0,
    "capture_heading_error_deg": 90.0,
    "capture_lead_time_s": 0.0,
    "capture_bank_scale": 0.15,
    "close_guidance": {
      "max_distance_m": 1450.0,
      "max_ata_deg": 180.0,
      "max_bank_deg": 80.0,
      "heading_to_bank": 3.0,
      "target_bank_heading_gain_baseline_deg": 44.0,
      "target_bank_to_heading_gain": -0.06,
      "roll_gain": 0.02,
      "pitch_gain": 0.5,
      "capture_pitch_gain": 0.5,
      "pitch_command_sign": -1.0,
      "rudder_gain": 0.0,
      "capture_rudder_gain": 0.0,
      "lead_time_s": 0.0,
      "capture_range_m": 100000.0,
      "capture_heading_error_deg": 180.0,
      "capture_bank_scale": 1.0,
      "capture_target_heading_blend": 0.5,
      "capture_target_bank_blend": 0.0,
      "capture_roll_gain": 0.02,
      "capture_roll_delta_damping": 0.2,
      "capture_speed_gain": 0.03,
      "capture_speed_offset_mps": -40.0
    }
  },
  "ownship_randomization": {
    "enabled": false
  },
  "initial_scenario": {
    "mode": "ref_old_random",
    "legacy_use_random_scenario": true,
    "legacy_use_first_scenario_only": false,
    "legacy_scenario_indices": [
      5
    ],
    "legacy_randomization": {
      "aircraft_radius_m": 100.0,
      "roll_deg": 5.0,
      "pitch_deg": 5.0,
      "heading_deg": 5.0,
      "shared_n_m": 4000.0,
      "shared_e_m": 4000.0,
      "shared_d_m": 1000.0,
      "target_distance_n_m": 300.0,
      "speed_mps": 50.0,
      "loiter_bank_deg_range": [
        40.0,
        70.0
      ]
    }
  },
  "ownship": [
    1000.0,
    0.0,
    -7000.0,
    0.0,
    0.0,
    0.0,
    300.0
  ],
  "target": [
    6000.0,
    0.0,
    -7000.0,
    0.0,
    0.0,
    180.0,
    300.0
  ],
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4
  },
  "reward": {
    "step_penalty": -0.01,
    "survival_bonus": 0.05,
    "win_reward": 0.0,
    "loss_reward": -100.0,
    "ownship_crash_reward": -120.0,
    "target_crash_reward": 0.0,
    "timeout_reward": 0.0,
    "draw_reward": -10.0,
    "guard_fail_penalty": -100.0,
    "damage_scale": 200.0,
    "altitude_warning_m": 4000.0,
    "altitude_floor_m": 300.0,
    "altitude_risk_scale": 0.5,
    "dive_penalty_scale": 0.2,
    "dive_reference_deg": 45.0,
    "excess_bank_deg": 60.0,
    "excess_bank_scale": 0.05,
    "pursuit_scale": 0.3,
    "pursuit_half_angle_deg": 60.0,
    "pursuit_range_m": 8000.0,
    "altitude_hold_target_m": 7000.0,
    "altitude_hold_range_m": 4000.0,
    "altitude_hold_scale": 0.5,
    "level_pitch_scale": 0.1,
    "level_roll_scale": 0.05,
    "closure_scale": 0.5,
    "closure_reference_mps": 300.0,
    "alignment_scale": 1.0,
    "range_scale": 0.3,
    "range_reference_m": 8000.0,
    "wez_hold_scale": 30.0,
    "fine_alignment_scale": 2.0,
    "fine_alignment_reference_deg": 30.0,
    "moving_alignment_scale": 4.0,
    "moving_alignment_reference_deg": 180.0,
    "close_range_brake_scale": 3.0,
    "inside_min_range_scale": 6.0,
    "wez_progress_scale": 30.0,
    "wez_required_steps": 10,
    "wez_milestone_bonus": 40.0,
    "wez_miss_penalty": -30.0
  },
  "reward_module": "student.sac_wez_range_reward_v6"
}
```

## Training History

- iter `0`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `1`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `2`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `3`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `4`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `5`: reward_mean=`880.6513977250981`, episode_len_mean=`600.0`
- iter `6`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `7`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `8`: reward_mean=`nan`, episode_len_mean=`nan`
