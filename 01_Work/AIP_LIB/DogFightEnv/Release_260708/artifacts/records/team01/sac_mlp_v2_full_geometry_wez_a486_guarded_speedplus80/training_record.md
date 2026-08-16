# Training Record

- Created at: `2026-08-14T23:45:25`
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
- Damage scale: `20.0`
- Pursuit scale: `0.0`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `None`
- Win reward: `100.0`
- Loss reward: `-120.0`
- Draw reward: `-10.0`

## CLI Arguments

```json
{
  "algorithm": "sac",
  "iterations": 30,
  "framework": "torch",
  "num_env_runners": 0,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical25",
  "observation_module": "",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_wez_sync_reward_v7",
  "max_engage_time": 30.0,
  "episode_step_limit": 1800,
  "lr": 0.0003,
  "actor_lr": 2e-05,
  "critic_lr": 0.0003,
  "alpha_lr": 2e-05,
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
  "output_tag": "sac_mlp_v2_full_geometry_wez_a486_guarded_speedplus80",
  "notes": "A486 trains scenario-0 acquisition with a guarded full-BT target and an initial +80 m/s ownship speed advantage before reducing the advantage.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 2,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_mlp_v2_full_geometry_wez_a483_guard3500",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 10,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 2,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\dev\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_mlp_v2_full_geometry_wez_a486_guarded_speedplus80.yaml"
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
    1001,
    1002,
    1003,
    1004,
    1005,
    1006,
    1007,
    1008,
    1009,
    1010
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
      0.2,
      0.2,
      0.2,
      0.05
    ],
    "near_residual_scale": [
      0.3,
      0.3,
      0.3,
      0.05
    ],
    "near_distance_m": 1200.0,
    "near_ata_deg": 15.0,
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
    "capture_bank_scale": 0.15,
    "capture_target_bank_blend": 0.5,
    "capture_target_bank_blend_start_ata_deg": 15.0,
    "proximity_speed_match_start_distance_m": 1400.0,
    "proximity_speed_match_gain": 0.02,
    "proximity_speed_match_min_overspeed_mps": 45.0,
    "proximity_speed_match_max_ata_deg": 5.0,
    "proximity_speed_match_latch": true,
    "rear_threat_distance_m": 1000.0,
    "rear_threat_ata_deg": 120.0,
    "rear_threat_roll_gain": 0.02,
    "rear_threat_bank_pitch_compensation_gain": 0.25
  },
  "ownship_randomization": {
    "enabled": false
  },
  "initial_scenario": {
    "mode": "ref_old_random",
    "legacy_use_random_scenario": true,
    "legacy_use_first_scenario_only": false,
    "legacy_scenario_indices": [
      0
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
      "ownship_speed_advantage_mps": 80.0,
      "loiter_bank_deg_range": [
        40.0,
        70.0
      ]
    }
  },
  "wez": {
    "angle_deg": 10.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4
  },
  "reward": {
    "step_penalty": -0.01,
    "survival_bonus": 0.05,
    "win_reward": 100.0,
    "loss_reward": -120.0,
    "ownship_crash_reward": -120.0,
    "target_crash_reward": 0.0,
    "timeout_reward": -10.0,
    "draw_reward": -10.0,
    "guard_fail_penalty": -100.0,
    "damage_scale": 20.0,
    "altitude_warning_m": 4000.0,
    "altitude_floor_m": 300.0,
    "altitude_risk_scale": 0.5,
    "dive_penalty_scale": 0.2,
    "dive_reference_deg": 45.0,
    "excess_bank_deg": 80.0,
    "excess_bank_scale": 0.01,
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "altitude_hold_target_m": 7000.0,
    "altitude_hold_range_m": 3000.0,
    "altitude_hold_scale": 0.05,
    "level_pitch_scale": 0.03,
    "level_roll_scale": 0.01,
    "closure_scale": 1.0,
    "closure_reference_mps": 300.0,
    "alignment_scale": 2.0,
    "range_scale": 0.5,
    "range_reference_m": 8000.0,
    "wez_hold_scale": 120.0,
    "fine_alignment_scale": 0.0,
    "fine_alignment_reference_deg": 10.0,
    "moving_alignment_scale": 0.0,
    "moving_alignment_reference_deg": 5.0,
    "close_range_brake_scale": 1.0,
    "inside_min_range_scale": 2.0,
    "sync_range_m": 1200.0,
    "sync_ata_gate_deg": 15.0,
    "bank_match_scale": 6.0,
    "bank_match_reference_deg": 30.0,
    "heading_match_scale": 6.0,
    "heading_match_reference_deg": 45.0,
    "speed_match_scale": 4.0,
    "speed_match_reference_mps": 100.0,
    "wez_progress_scale": 120.0,
    "wez_required_steps": 10,
    "wez_milestone_bonus": 200.0,
    "wez_miss_penalty": -30.0
  },
  "reward_module": "student.sac_wez_sync_reward_v7"
}
```

## Training History

- iter `0`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `1`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `2`: reward_mean=`5012.149753241567`, episode_len_mean=`300.0`
- iter `3`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `4`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `5`: reward_mean=`616.8316840335898`, episode_len_mean=`300.0`
- iter `6`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `7`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `8`: reward_mean=`631.8752204079203`, episode_len_mean=`300.0`
- iter `9`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `10`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `11`: reward_mean=`625.8223891394629`, episode_len_mean=`300.0`
- iter `12`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `13`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `14`: reward_mean=`624.5602153663991`, episode_len_mean=`300.0`
- iter `15`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `16`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `17`: reward_mean=`691.0414356936785`, episode_len_mean=`300.0`
- iter `18`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `19`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `20`: reward_mean=`587.225882446224`, episode_len_mean=`300.0`
- iter `21`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `22`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `23`: reward_mean=`481.2693947211829`, episode_len_mean=`300.0`
- iter `24`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `25`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `26`: reward_mean=`624.596751889806`, episode_len_mean=`300.0`
- iter `27`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `28`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `29`: reward_mean=`605.6953626195449`, episode_len_mean=`300.0`
