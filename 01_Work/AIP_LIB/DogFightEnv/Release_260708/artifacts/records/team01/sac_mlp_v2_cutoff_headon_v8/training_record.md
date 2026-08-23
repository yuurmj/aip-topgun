# Training Record

- Created at: `2026-08-23T02:53:21`
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
- Step penalty: `-0.01`
- Damage scale: `40.0`
- Pursuit scale: `0.0`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `None`
- Win reward: `1200.0`
- Loss reward: `-1400.0`
- Draw reward: `-50.0`

## CLI Arguments

```json
{
  "algorithm": "sac",
  "iterations": 12,
  "framework": "torch",
  "num_env_runners": 0,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical25",
  "observation_module": "",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_cutoff_headon_reward_v8",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "lr": 0.0003,
  "actor_lr": 3e-05,
  "critic_lr": 0.0003,
  "alpha_lr": 1e-05,
  "gamma": 0.995,
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
  "output_tag": "sac_mlp_v2_cutoff_headon_v8",
  "notes": "Viewer 2500-ft equal-speed head-on fine-tune. Restores actor authority near the merge and penalizes mutual damage.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 2,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_mlp_v2_submission_a437_actor_a687_runtime",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 6,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 2,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_mlp_v2_cutoff_headon_v8.yaml"
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
    260821,
    260822,
    260823,
    260824,
    260825,
    260826,
    260827,
    260828
  ],
  "canonicalize_negative_target_bank": true,
  "step_ratio": 6,
  "min_altitude": 300.0,
  "control_surface_scale": 1.0,
  "target_altitude_guard": {
    "enabled": false
  },
  "pursuit_guidance_residual": {
    "enabled": true,
    "residual_scale": [
      0.3,
      0.3,
      0.2,
      0.1
    ],
    "near_residual_scale": [
      0.4,
      0.4,
      0.25,
      0.1
    ],
    "near_distance_m": 1200.0,
    "near_ata_deg": 20.0,
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
    "capture_lead_time_s": 1.0,
    "capture_maneuver_range_m": 1550.0,
    "capture_maneuver_target_pitch_deg": 45.0,
    "capture_maneuver_target_bank_deg": 80.0,
    "capture_maneuver_lead_time_s": 0.9,
    "capture_extreme_bank_deg": 140.0,
    "capture_extreme_bank_lead_time_s": 1.15,
    "capture_bank_scale": 0.15,
    "capture_target_bank_blend": 0.5,
    "capture_target_bank_blend_start_ata_deg": 15.0,
    "capture_roll_delta_damping": 0.5,
    "proximity_speed_match_start_distance_m": 1400.0,
    "proximity_speed_match_gain": 0.02,
    "proximity_speed_match_min_overspeed_mps": 30.0,
    "proximity_speed_match_max_ata_deg": 8.0,
    "proximity_speed_match_latch": true,
    "rear_threat_distance_m": 1300.0,
    "rear_threat_ata_deg": 100.0,
    "rear_threat_roll_gain": 0.02,
    "rear_threat_bank_pitch_compensation_gain": 0.25,
    "suppress_opposing_roll_heading_error_deg": 12.0,
    "opposing_roll_guidance_scale": 4.0
  },
  "ownship_randomization": {
    "enabled": false
  },
  "initial_scenario": {
    "mode": "scenario_pool",
    "scenarios": [
      {
        "name": "viewer_exact",
        "weight": 0.3,
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
      },
      {
        "name": "viewer_left_40m",
        "weight": 0.1,
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
          -40,
          -4572,
          0,
          0,
          180,
          200
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "viewer_right_40m",
        "weight": 0.1,
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
          40,
          -4572,
          0,
          0,
          180,
          200
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "viewer_range_900",
        "weight": 0.1,
        "ownship": [
          0,
          0,
          -4572,
          0,
          0,
          0,
          195
        ],
        "target": [
          900,
          0,
          -4572,
          0,
          0,
          180,
          205
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "viewer_range_1050_left",
        "weight": 0.1,
        "ownship": [
          0,
          0,
          -4800,
          0,
          0,
          2,
          205
        ],
        "target": [
          1050,
          -60,
          -4770,
          0,
          0,
          178,
          195
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "viewer_range_1050_right",
        "weight": 0.1,
        "ownship": [
          0,
          0,
          -4800,
          0,
          0,
          358,
          205
        ],
        "target": [
          1050,
          60,
          -4830,
          0,
          0,
          182,
          195
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "post_merge_reengage_left",
        "weight": 0.1,
        "ownship": [
          900,
          -250,
          -4572,
          0,
          0,
          170,
          220
        ],
        "target": [
          0,
          0,
          -4572,
          0,
          0,
          180,
          200
        ],
        "target_mode": "behavior_tree"
      },
      {
        "name": "post_merge_reengage_right",
        "weight": 0.1,
        "ownship": [
          900,
          250,
          -4572,
          0,
          0,
          190,
          220
        ],
        "target": [
          0,
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
    "step_penalty": -0.01,
    "survival_bonus": 0.02,
    "win_reward": 1200.0,
    "loss_reward": -1400.0,
    "ownship_crash_reward": -1400.0,
    "target_crash_reward": 0.0,
    "timeout_reward": -50.0,
    "draw_reward": -50.0,
    "guard_fail_penalty": -100.0,
    "damage_scale": 40.0,
    "altitude_warning_m": 3500.0,
    "altitude_floor_m": 300.0,
    "altitude_risk_scale": 0.8,
    "dive_penalty_scale": 0.3,
    "dive_reference_deg": 45.0,
    "excess_bank_deg": 85.0,
    "excess_bank_scale": 0.02,
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "altitude_hold_target_m": 4572.0,
    "altitude_hold_range_m": 2500.0,
    "altitude_hold_scale": 0.05,
    "level_pitch_scale": 0.02,
    "level_roll_scale": 0.01,
    "closure_scale": 0.5,
    "closure_reference_mps": 300.0,
    "alignment_scale": 2.0,
    "range_scale": 0.5,
    "range_reference_m": 6000.0,
    "wez_hold_scale": 20.0,
    "fine_alignment_scale": 1.0,
    "fine_alignment_reference_deg": 15.0,
    "moving_alignment_scale": 2.0,
    "moving_alignment_reference_deg": 30.0,
    "close_range_brake_scale": 2.0,
    "inside_min_range_scale": 5.0,
    "sync_range_m": 1600.0,
    "sync_ata_gate_deg": 30.0,
    "bank_match_scale": 0.5,
    "bank_match_reference_deg": 45.0,
    "heading_match_scale": 0.5,
    "heading_match_reference_deg": 90.0,
    "speed_match_scale": 0.5,
    "speed_match_reference_mps": 60.0,
    "target_damage_bonus_scale": 60.0,
    "ownship_damage_penalty_scale": 160.0,
    "mutual_damage_penalty": 20.0,
    "angular_advantage_scale": 4.0,
    "angular_advantage_reference_deg": 90.0,
    "wez_progress_scale": 50.0,
    "wez_required_steps": 10,
    "wez_milestone_bonus": 200.0,
    "wez_miss_penalty": -30.0
  },
  "reward_module": "student.sac_cutoff_headon_reward_v8"
}
```

## Training History

- iter `0`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `1`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `2`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `3`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `4`: reward_mean=`-821.3801858046629`, episode_len_mean=`459.0`
- iter `5`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `6`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `7`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `8`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `9`: reward_mean=`-1271.8829046342908`, episode_len_mean=`457.0`
- iter `10`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `11`: reward_mean=`nan`, episode_len_mean=`nan`
