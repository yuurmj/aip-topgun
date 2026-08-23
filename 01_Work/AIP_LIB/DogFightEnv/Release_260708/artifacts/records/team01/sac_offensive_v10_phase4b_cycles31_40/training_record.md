# Training Record

- Created at: `2026-08-23T04:20:44`
- Python: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`
- Algorithm: `sac`

## Observation

- Mode: `tactical25`
- Size: `25`
- Description: Tactical21 plus ownship/target yaw rates, relative yaw rate, and horizontal line-of-sight azimuth rate for moving-target turn synchronization.
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
  - `ownship_yaw_rate_norm`
  - `target_yaw_rate_norm`
  - `relative_yaw_rate_norm`
  - `los_azimuth_rate_norm`

## Reward

- Description: Survival bonus (curriculum) + step penalty + pursuit shaping (smooth ATA×range gradient) + damage differential + low altitude penalty + terminal rewards.
- Step penalty: `-0.03`
- Damage scale: `300.0`
- Pursuit scale: `0.0`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `None`
- Win reward: `3200.0`
- Loss reward: `-3400.0`
- Draw reward: `-600.0`

## CLI Arguments

```json
{
  "algorithm": "sac",
  "iterations": 10,
  "framework": "torch",
  "num_env_runners": 0,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "1024",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical25",
  "observation_module": "",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_offensive_reward_v10",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "lr": 0.0003,
  "actor_lr": 3e-05,
  "critic_lr": 0.0002,
  "alpha_lr": 1e-05,
  "gamma": 0.995,
  "train_batch_size": 256,
  "minibatch_size": 256,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": 60000,
  "learning_starts": 512,
  "training_intensity": 2.0,
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
  "output_tag": "sac_offensive_v10_phase4b_cycles31_40",
  "notes": "Feedback cycles 31-40: stabilize cycle 28 against stronger randomized guarded BT engagements.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 1,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_offensive_v10_phase3_cycles21_30/bundle_000008",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 5,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 1,
  "policy_probe_steps": 4,
  "no_policy_probe_print": true,
  "engagement_log_interval": 1,
  "engagement_log_steps": 900,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_offensive_v10_phase4_cycles31_40.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical25",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "episode_seed_cycle": [
    90431,
    90432,
    90433,
    90434,
    90435,
    90436,
    90437,
    90438,
    90439,
    90440,
    90441,
    90442
  ],
  "canonicalize_negative_target_bank": true,
  "step_ratio": 6,
  "min_altitude": 300.0,
  "control_surface_scale": 1.0,
  "target_altitude_guard": {
    "enabled": true,
    "trigger_altitude_m": 3500.0,
    "release_altitude_m": 5000.0,
    "roll_level_gain": 0.02,
    "pull_up_action": -1.0,
    "throttle_action": 1.0
  },
  "pursuit_guidance_residual": {
    "enabled": true,
    "residual_scale": [
      0.4,
      0.4,
      0.3,
      0.12
    ],
    "near_residual_scale": [
      0.6,
      0.6,
      0.45,
      0.2
    ],
    "near_distance_m": 1500.0,
    "near_ata_deg": 35.0,
    "max_bank_deg": 75.0,
    "heading_to_bank": 2.5,
    "roll_gain": 0.002,
    "pitch_gain": 0.05,
    "capture_pitch_gain": 0.3,
    "pitch_command_sign": -1.0,
    "rudder_gain": 0.0,
    "lead_time_s": 8.0,
    "brake_range_m": 1400.0,
    "brake_heading_error_deg": 8.0,
    "near_throttle_action": -0.7,
    "capture_range_m": 1200.0,
    "capture_heading_error_deg": 100.0,
    "capture_lead_time_s": 1.0,
    "capture_maneuver_range_m": 1800.0,
    "capture_maneuver_target_pitch_deg": 35.0,
    "capture_maneuver_target_bank_deg": 75.0,
    "capture_maneuver_lead_time_s": 1.0,
    "capture_extreme_bank_deg": 145.0,
    "capture_extreme_bank_lead_time_s": 1.2,
    "capture_bank_scale": 0.2,
    "capture_target_bank_blend": 0.55,
    "capture_target_bank_blend_start_ata_deg": 12.0,
    "capture_roll_gain": 0.012,
    "capture_roll_delta_damping": 0.35,
    "latch_capture_mode": true,
    "wez_hold_latch_ata_deg": 5.0,
    "wez_hold_target_bank_blend": 1.0,
    "wez_hold_roll_gain": 0.015,
    "wez_hold_target_heading_blend": 0.25,
    "capture_speed_gain": 0.02,
    "proximity_speed_match_start_distance_m": 1300.0,
    "proximity_speed_match_gain": 0.025,
    "proximity_speed_match_min_overspeed_mps": 15.0,
    "proximity_speed_match_max_ata_deg": 10.0,
    "proximity_speed_match_latch": true,
    "rear_threat_distance_m": 1500.0,
    "rear_threat_ata_deg": 100.0,
    "rear_threat_roll_gain": 0.025,
    "rear_threat_bank_pitch_compensation_gain": 0.25,
    "suppress_opposing_roll_heading_error_deg": 10.0,
    "opposing_roll_guidance_scale": 3.0
  },
  "ownship_randomization": {
    "enabled": false
  },
  "initial_scenario": {
    "mode": "ref_old_random",
    "legacy_use_random_scenario": true,
    "legacy_use_first_scenario_only": false,
    "legacy_scenario_indices": [
      0,
      1,
      2,
      3,
      4,
      5,
      6,
      7
    ],
    "legacy_randomization": {
      "aircraft_radius_m": 150.0,
      "roll_deg": 10.0,
      "pitch_deg": 8.0,
      "heading_deg": 10.0,
      "shared_n_m": 4000.0,
      "shared_e_m": 4000.0,
      "shared_d_m": 1000.0,
      "target_distance_n_m": 400.0,
      "speed_mps": 70.0,
      "ownship_speed_advantage_mps": -20.0,
      "loiter_bank_deg_range": [
        30.0,
        80.0
      ]
    }
  },
  "wez": {
    "angle_deg": 10.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4
  },
  "reward": {
    "step_penalty": -0.03,
    "survival_bonus": 0.01,
    "win_reward": 3200.0,
    "loss_reward": -3400.0,
    "ownship_crash_reward": -3400.0,
    "target_crash_reward": 0.0,
    "timeout_reward": -600.0,
    "draw_reward": -600.0,
    "guard_fail_penalty": -900.0,
    "damage_scale": 300.0,
    "altitude_warning_m": 3800.0,
    "altitude_floor_m": 300.0,
    "altitude_risk_scale": 1.2,
    "dive_penalty_scale": 0.5,
    "dive_reference_deg": 45.0,
    "excess_bank_deg": 85.0,
    "excess_bank_scale": 0.04,
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "altitude_hold_target_m": 6000.0,
    "altitude_hold_range_m": 3500.0,
    "altitude_hold_scale": 0.08,
    "level_pitch_scale": 0.02,
    "level_roll_scale": 0.01,
    "closure_scale": 0.6,
    "closure_reference_mps": 300.0,
    "alignment_scale": 3.0,
    "range_scale": 0.5,
    "range_reference_m": 8000.0,
    "wez_hold_scale": 18.0,
    "fine_alignment_scale": 2.5,
    "fine_alignment_reference_deg": 15.0,
    "moving_alignment_scale": 4.0,
    "moving_alignment_reference_deg": 40.0,
    "close_range_brake_scale": 3.0,
    "inside_min_range_scale": 8.0,
    "sync_range_m": 2000.0,
    "sync_ata_gate_deg": 35.0,
    "bank_match_scale": 0.6,
    "bank_match_reference_deg": 45.0,
    "heading_match_scale": 0.6,
    "heading_match_reference_deg": 80.0,
    "speed_match_scale": 0.6,
    "speed_match_reference_mps": 60.0,
    "target_damage_bonus_scale": 450.0,
    "ownship_damage_penalty_scale": 350.0,
    "mutual_damage_penalty": 18.0,
    "angular_advantage_scale": 18.0,
    "angular_advantage_reference_deg": 90.0,
    "attack_window_scale": 12.0,
    "attack_window_ata_deg": 30.0,
    "attack_window_range_m": 3000.0,
    "threat_penalty_scale": 20.0,
    "threat_ata_deg": 35.0,
    "disengagement_penalty_scale": 8.0,
    "disengagement_range_m": 5000.0,
    "fire_control_scale": 45.0,
    "fire_control_ata_deg": 8.0,
    "fire_control_center_range_m": 560.0,
    "fire_control_half_width_m": 355.0,
    "fire_control_target_blind_deg": 45.0,
    "wez_progress_scale": 120.0,
    "wez_required_steps": 15,
    "wez_milestone_bonus": 450.0,
    "wez_miss_penalty": -120.0
  },
  "reward_module": "student.sac_offensive_reward_v10"
}
```

## Training History

- iter `0`: reward_mean=`-9412.749656384374`, episode_len_mean=`600.0`
- iter `1`: reward_mean=`8137.321716935558`, episode_len_mean=`600.0`
- iter `2`: reward_mean=`8864.147978254587`, episode_len_mean=`600.0`
- iter `3`: reward_mean=`7649.705503699487`, episode_len_mean=`600.0`
- iter `4`: reward_mean=`107.59579146233227`, episode_len_mean=`600.0`
- iter `5`: reward_mean=`-9187.299388235817`, episode_len_mean=`600.0`
- iter `6`: reward_mean=`10769.550236912568`, episode_len_mean=`600.0`
- iter `7`: reward_mean=`-359.3596261662001`, episode_len_mean=`600.0`
- iter `8`: reward_mean=`7320.884063518734`, episode_len_mean=`600.0`
- iter `9`: reward_mean=`7950.75245916843`, episode_len_mean=`600.0`
