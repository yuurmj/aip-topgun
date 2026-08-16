# Training Record

- Created at: `2026-08-14T14:31:15`
- Python: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`
- Algorithm: `sac`

## Observation

- Mode: `tactical21`
- Size: `21`
- Description: Tactical17 plus target bank, periodic relative heading, and relative speed for moving-target WEZ hold control.
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
  - `target_roll_norm`
  - `relative_heading_sin`
  - `relative_heading_cos`
  - `relative_speed_norm`

## Reward

- Description: Survival bonus (curriculum) + step penalty + pursuit shaping (smooth ATA×range gradient) + damage differential + low altitude penalty + terminal rewards.
- Step penalty: `-0.01`
- Damage scale: `200.0`
- Pursuit scale: `0.0`
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
  "iterations": 6,
  "framework": "torch",
  "num_env_runners": 0,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical21",
  "observation_module": "",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_wez_range_reward_v6",
  "max_engage_time": 10.0,
  "episode_step_limit": 600,
  "lr": 0.0003,
  "actor_lr": 1e-05,
  "critic_lr": 0.0003,
  "alpha_lr": 1e-05,
  "gamma": 0.995,
  "train_batch_size": 256,
  "minibatch_size": 256,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": 20000,
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
  "output_tag": "sac_mlp_v2_full_geometry_wez_a309_hold_curriculum",
  "notes": "A309 strict-2deg hold curriculum: start aligned inside WEZ at matched speed and bank before returning to scenario-5 acquisition.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 3,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_mlp_v2_full_geometry_wez_a304_tactical21_init/bundle_000000",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 3,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 3,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\dev\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_mlp_v2_full_geometry_wez_a309_hold_curriculum.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical21",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 10.0,
  "episode_step_limit": 600,
  "episode_seed_cycle": [
    1101,
    1102,
    1103,
    1104,
    1105,
    1106,
    1107,
    1108,
    1109,
    1110
  ],
  "step_ratio": 6,
  "min_altitude": 300.0,
  "control_surface_scale": 1.0,
  "pursuit_guidance_residual": {
    "enabled": true,
    "residual_scale": [
      0.2,
      0.2,
      0.2,
      0.1
    ],
    "max_bank_deg": 70.0,
    "heading_to_bank": 2.5,
    "heading_rate_to_bank": 0.0,
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
    "capture_bank_scale": 0.15
  },
  "ownship": [
    1000.0,
    0.0,
    -8000.0,
    55.0,
    0.0,
    0.0,
    220.0
  ],
  "target": [
    1800.0,
    0.0,
    -8000.0,
    55.0,
    0.0,
    0.0,
    220.0
  ],
  "target_loiter": {
    "enabled": true,
    "bank": 55.0,
    "pitch": 0.0
  },
  "ownship_randomization": {
    "enabled": true,
    "distance_range_m": [
      600.0,
      850.0
    ],
    "radius": 0.2,
    "r_roll": 2.0,
    "r_pitch": 1.0,
    "r_heading": 1.0
  },
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
    "excess_bank_deg": 70.0,
    "excess_bank_scale": 0.02,
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 60.0,
    "pursuit_range_m": 8000.0,
    "altitude_hold_target_m": 8000.0,
    "altitude_hold_range_m": 4000.0,
    "altitude_hold_scale": 0.1,
    "level_pitch_scale": 0.03,
    "level_roll_scale": 0.0,
    "closure_scale": 0.2,
    "closure_reference_mps": 300.0,
    "alignment_scale": 1.0,
    "range_scale": 0.5,
    "range_reference_m": 8000.0,
    "wez_hold_scale": 80.0,
    "fine_alignment_scale": 10.0,
    "fine_alignment_reference_deg": 10.0,
    "moving_alignment_scale": 10.0,
    "moving_alignment_reference_deg": 90.0,
    "close_range_brake_scale": 8.0,
    "inside_min_range_scale": 12.0,
    "wez_progress_scale": 80.0,
    "wez_required_steps": 10,
    "wez_milestone_bonus": 100.0,
    "wez_miss_penalty": -20.0
  },
  "reward_module": "student.sac_wez_range_reward_v6"
}
```

## Training History

- iter `0`: reward_mean=`4625.5449805025555`, episode_len_mean=`100.0`
- iter `1`: reward_mean=`2397.5547843022396`, episode_len_mean=`100.0`
- iter `2`: reward_mean=`5414.208149467273`, episode_len_mean=`43.0`
- iter `3`: reward_mean=`4613.491420448944`, episode_len_mean=`100.0`
- iter `4`: reward_mean=`5012.796859731652`, episode_len_mean=`100.0`
- iter `5`: reward_mean=`6423.323574148647`, episode_len_mean=`100.0`
