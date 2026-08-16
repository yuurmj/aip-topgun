# Training Record

- Created at: `2026-08-14T17:29:15`
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
- Damage scale: `200.0`
- Pursuit scale: `0.05`
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
  "iterations": 40,
  "framework": "torch",
  "num_env_runners": 0,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical25",
  "observation_module": "",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_wez_sync_reward_v7",
  "max_engage_time": 20.0,
  "episode_step_limit": 1200,
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
  "output_tag": "sac_mlp_v2_full_geometry_wez_a347_mixed_hold_acquisition",
  "notes": "A347 alternates scenario-5 hard acquisition and aligned strict-2deg hold episodes in one replay buffer so a single tactical25 actor learns both skills.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 5,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_mlp_v2_full_geometry_wez_a342_tactical25_hold/bundle_000003",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 10,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 5,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\dev\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_mlp_v2_full_geometry_wez_a347_mixed_hold_acquisition.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical25",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 20.0,
  "episode_step_limit": 1200,
  "episode_seed_cycle": [
    1003,
    1101,
    1004,
    1102,
    1006,
    1103
  ],
  "step_ratio": 6,
  "min_altitude": 300.0,
  "control_surface_scale": 1.0,
  "pursuit_guidance_residual": {
    "enabled": true,
    "residual_scale": [
      0.05,
      0.05,
      0.05,
      0.05
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
  "initial_scenario": {
    "mode": "mixed_ref_old_random_hold",
    "hard_episode_seeds": [
      1003,
      1004,
      1006
    ],
    "hard_scenario": {
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
    }
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
    "pursuit_scale": 0.05,
    "pursuit_half_angle_deg": 60.0,
    "pursuit_range_m": 8000.0,
    "altitude_hold_target_m": 7500.0,
    "altitude_hold_range_m": 4000.0,
    "altitude_hold_scale": 0.15,
    "level_pitch_scale": 0.04,
    "level_roll_scale": 0.005,
    "closure_scale": 0.3,
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
    "sync_range_m": 1200.0,
    "sync_ata_gate_deg": 15.0,
    "bank_match_scale": 5.0,
    "bank_match_reference_deg": 20.0,
    "heading_match_scale": 5.0,
    "heading_match_reference_deg": 20.0,
    "speed_match_scale": 3.0,
    "speed_match_reference_mps": 50.0,
    "wez_progress_scale": 80.0,
    "wez_required_steps": 10,
    "wez_milestone_bonus": 120.0,
    "wez_miss_penalty": -20.0
  },
  "reward_module": "student.sac_wez_sync_reward_v7"
}
```

## Training History

- iter `0`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `1`: reward_mean=`-2605.587668384653`, episode_len_mean=`200.0`
- iter `2`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `3`: reward_mean=`3252.5552995137286`, episode_len_mean=`200.0`
- iter `4`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `5`: reward_mean=`-1414.2354720969015`, episode_len_mean=`200.0`
- iter `6`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `7`: reward_mean=`694.2990968233222`, episode_len_mean=`200.0`
- iter `8`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `9`: reward_mean=`-2247.7983684620995`, episode_len_mean=`200.0`
- iter `10`: reward_mean=`5596.937267616565`, episode_len_mean=`42.0`
- iter `11`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `12`: reward_mean=`-2507.906320505167`, episode_len_mean=`200.0`
- iter `13`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `14`: reward_mean=`4678.674154292443`, episode_len_mean=`200.0`
- iter `15`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `16`: reward_mean=`-1577.3047874296508`, episode_len_mean=`200.0`
- iter `17`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `18`: reward_mean=`559.6044789530092`, episode_len_mean=`200.0`
- iter `19`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `20`: reward_mean=`-2299.279437891244`, episode_len_mean=`200.0`
- iter `21`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `22`: reward_mean=`3712.8957737715264`, episode_len_mean=`200.0`
- iter `23`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `24`: reward_mean=`-2612.925441068828`, episode_len_mean=`200.0`
- iter `25`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `26`: reward_mean=`3675.3830530725686`, episode_len_mean=`200.0`
- iter `27`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `28`: reward_mean=`-1417.7831185587702`, episode_len_mean=`200.0`
- iter `29`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `30`: reward_mean=`461.57272385096013`, episode_len_mean=`200.0`
- iter `31`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `32`: reward_mean=`1611.5136868674017`, episode_len_mean=`120.5`
- iter `33`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `34`: reward_mean=`-2610.120984778078`, episode_len_mean=`200.0`
- iter `35`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `36`: reward_mean=`3688.721104424733`, episode_len_mean=`200.0`
- iter `37`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `38`: reward_mean=`-1383.7149840257978`, episode_len_mean=`200.0`
- iter `39`: reward_mean=`nan`, episode_len_mean=`nan`
