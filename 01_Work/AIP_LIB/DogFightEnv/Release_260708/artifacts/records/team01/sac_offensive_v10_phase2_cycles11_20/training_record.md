# Training Record

- Created at: `2026-08-23T03:43:32`
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
- Damage scale: `200.0`
- Pursuit scale: `0.0`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `None`
- Win reward: `3000.0`
- Loss reward: `-3000.0`
- Draw reward: `-500.0`

## CLI Arguments

```json
{
  "algorithm": "sac",
  "iterations": 10,
  "framework": "torch",
  "num_env_runners": 0,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "512",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical25",
  "observation_module": "",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_offensive_reward_v10",
  "max_engage_time": 30.0,
  "episode_step_limit": 1800,
  "lr": 0.0003,
  "actor_lr": 0.0002,
  "critic_lr": 0.0003,
  "alpha_lr": 1e-05,
  "gamma": 0.995,
  "train_batch_size": 256,
  "minibatch_size": 256,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": 50000,
  "learning_starts": 256,
  "training_intensity": 4.0,
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
  "output_tag": "sac_offensive_v10_phase2_cycles11_20",
  "notes": "Feedback cycles 11-20: deep and sustained one-sided WEZ with higher update intensity.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 1,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_offensive_v9_phase1_cycles01_10/bundle_000010",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 5,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 1,
  "policy_probe_steps": 4,
  "no_policy_probe_print": true,
  "engagement_log_interval": 1,
  "engagement_log_steps": 450,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_offensive_v10_phase2_cycles11_20.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical25",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 30.0,
  "episode_step_limit": 1800,
  "episode_seed_cycle": [
    90211,
    90212,
    90213,
    90214,
    90215,
    90216,
    90217,
    90218,
    90219,
    90220,
    90221,
    90222
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
    "mode": "scenario_pool",
    "scenarios": [
      {
        "name": "fire_500_center",
        "weight": 0.16,
        "ownship": [
          0,
          0,
          -6000,
          0,
          0,
          0,
          205
        ],
        "target": [
          500,
          0,
          -6000,
          0,
          0,
          0,
          200
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "fire_650_left",
        "weight": 0.14,
        "ownship": [
          0,
          -35,
          -6000,
          0,
          0,
          2,
          210
        ],
        "target": [
          650,
          0,
          -6000,
          0,
          0,
          0,
          200
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "fire_650_right",
        "weight": 0.14,
        "ownship": [
          0,
          35,
          -6000,
          0,
          0,
          358,
          210
        ],
        "target": [
          650,
          0,
          -6000,
          0,
          0,
          0,
          200
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "tail_900_offset",
        "weight": 0.12,
        "ownship": [
          0,
          80,
          -6200,
          0,
          0,
          357,
          215
        ],
        "target": [
          900,
          0,
          -6200,
          0,
          0,
          0,
          205
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "quarter_left",
        "weight": 0.1,
        "ownship": [
          0,
          0,
          -6500,
          0,
          0,
          20,
          215
        ],
        "target": [
          1400,
          -500,
          -6450,
          0,
          0,
          330,
          205
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "quarter_right",
        "weight": 0.1,
        "ownship": [
          0,
          0,
          -6500,
          0,
          0,
          340,
          215
        ],
        "target": [
          1400,
          500,
          -6550,
          0,
          0,
          30,
          205
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "postmerge_left",
        "weight": 0.08,
        "ownship": [
          900,
          -250,
          -5500,
          0,
          0,
          165,
          220
        ],
        "target": [
          0,
          0,
          -5500,
          0,
          0,
          180,
          205
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "postmerge_right",
        "weight": 0.08,
        "ownship": [
          900,
          250,
          -5500,
          0,
          0,
          195,
          220
        ],
        "target": [
          0,
          0,
          -5500,
          0,
          0,
          180,
          205
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "viewer_headon_762",
        "weight": 0.08,
        "ownship": [
          0,
          0,
          -4572,
          0,
          0,
          0,
          200
        ],
        "target": [
          762,
          0,
          -4572,
          0,
          0,
          180,
          200
        ],
        "target_mode": "behavior_tree"
      }
    ]
  },
  "wez": {
    "angle_deg": 10.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4
  },
  "reward": {
    "step_penalty": -0.03,
    "survival_bonus": 0.01,
    "win_reward": 3000.0,
    "loss_reward": -3000.0,
    "ownship_crash_reward": -3000.0,
    "target_crash_reward": 0.0,
    "timeout_reward": -500.0,
    "draw_reward": -500.0,
    "guard_fail_penalty": -800.0,
    "damage_scale": 200.0,
    "altitude_warning_m": 3500.0,
    "altitude_floor_m": 300.0,
    "altitude_risk_scale": 1.0,
    "dive_penalty_scale": 0.4,
    "dive_reference_deg": 45.0,
    "excess_bank_deg": 85.0,
    "excess_bank_scale": 0.03,
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "altitude_hold_target_m": 6000.0,
    "altitude_hold_range_m": 3000.0,
    "altitude_hold_scale": 0.05,
    "level_pitch_scale": 0.02,
    "level_roll_scale": 0.01,
    "closure_scale": 0.2,
    "closure_reference_mps": 300.0,
    "alignment_scale": 2.0,
    "range_scale": 0.3,
    "range_reference_m": 7000.0,
    "wez_hold_scale": 12.0,
    "fine_alignment_scale": 2.0,
    "fine_alignment_reference_deg": 12.0,
    "moving_alignment_scale": 3.0,
    "moving_alignment_reference_deg": 30.0,
    "close_range_brake_scale": 3.0,
    "inside_min_range_scale": 8.0,
    "sync_range_m": 1800.0,
    "sync_ata_gate_deg": 30.0,
    "bank_match_scale": 0.5,
    "bank_match_reference_deg": 40.0,
    "heading_match_scale": 0.5,
    "heading_match_reference_deg": 70.0,
    "speed_match_scale": 0.5,
    "speed_match_reference_mps": 50.0,
    "target_damage_bonus_scale": 300.0,
    "ownship_damage_penalty_scale": 220.0,
    "mutual_damage_penalty": 12.0,
    "angular_advantage_scale": 10.0,
    "angular_advantage_reference_deg": 90.0,
    "attack_window_scale": 8.0,
    "attack_window_ata_deg": 25.0,
    "attack_window_range_m": 2200.0,
    "threat_penalty_scale": 15.0,
    "threat_ata_deg": 30.0,
    "disengagement_penalty_scale": 5.0,
    "disengagement_range_m": 4000.0,
    "fire_control_scale": 35.0,
    "fire_control_ata_deg": 8.0,
    "fire_control_center_range_m": 560.0,
    "fire_control_half_width_m": 355.0,
    "fire_control_target_blind_deg": 45.0,
    "wez_progress_scale": 80.0,
    "wez_required_steps": 15,
    "wez_milestone_bonus": 300.0,
    "wez_miss_penalty": -100.0
  },
  "reward_module": "student.sac_offensive_reward_v10"
}
```

## Training History

- iter `0`: reward_mean=`5516.734765207064`, episode_len_mean=`94.25`
- iter `1`: reward_mean=`4815.9184484881525`, episode_len_mean=`161.75`
- iter `2`: reward_mean=`1600.7803581596995`, episode_len_mean=`300.0`
- iter `3`: reward_mean=`6393.771748005291`, episode_len_mean=`135.0`
- iter `4`: reward_mean=`4279.3643054487975`, episode_len_mean=`164.0`
- iter `5`: reward_mean=`4507.857549167716`, episode_len_mean=`161.75`
- iter `6`: reward_mean=`5456.306076150513`, episode_len_mean=`300.0`
- iter `7`: reward_mean=`5997.474805296769`, episode_len_mean=`93.75`
- iter `8`: reward_mean=`3342.106437685968`, episode_len_mean=`209.33333333333334`
- iter `9`: reward_mean=`5210.714334815801`, episode_len_mean=`115.66666666666667`
