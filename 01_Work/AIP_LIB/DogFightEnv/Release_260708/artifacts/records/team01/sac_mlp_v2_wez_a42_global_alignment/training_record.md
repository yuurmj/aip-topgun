# Training Record

- Created at: `2026-08-11T15:57:43`
- Python: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`
- Algorithm: `sac`

## Observation

- Mode: `tactical16`
- Size: `16`
- Description: Full tactical observation: ownship attitude + speed + altitude + health, relative geometry (ATA, AA, LOS), target health, WEZ flag, pursuit score. All features normalized to [-1, 1]. Observation space bounds: [-1, 1].
- Features:
  - `ownship_roll_norm`
  - `ownship_pitch_norm`
  - `ownship_yaw_norm`
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
- Damage scale: `100.0`
- Pursuit scale: `0.0`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `None`
- Win reward: `0.0`
- Loss reward: `-100.0`
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
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_moving_target_reward_v5",
  "max_engage_time": 15.0,
  "episode_step_limit": 900,
  "lr": 0.0003,
  "actor_lr": 1e-05,
  "critic_lr": 0.0003,
  "alpha_lr": 1e-05,
  "gamma": 0.99,
  "train_batch_size": 256,
  "minibatch_size": 256,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": 20000,
  "learning_starts": 500,
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
  "output_tag": "sac_mlp_v2_wez_a42_global_alignment",
  "notes": "A42 global ATA shaping: 180-degree moving-alignment reference avoids clipped reward plateau above 20 degrees.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 5,
  "save_native_checkpoint": false,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_mlp_v2_wez_a41_align_at_range/bundle_000020",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": null,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 5,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\dev\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_mlp_v2_wez_a42_global_alignment.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 15.0,
  "episode_step_limit": 900,
  "step_ratio": 6,
  "min_altitude": 300.0,
  "control_surface_scale": 1.0,
  "residual_action": {
    "enabled": true,
    "baseline": [
      0.0,
      -0.15,
      0.0,
      0.0
    ],
    "scale": [
      1.0,
      1.0,
      1.0,
      1.0
    ]
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
    "damage_scale": 100.0,
    "altitude_warning_m": 4000.0,
    "altitude_floor_m": 300.0,
    "altitude_risk_scale": 0.5,
    "dive_penalty_scale": 0.2,
    "dive_reference_deg": 45.0,
    "excess_bank_deg": 60.0,
    "excess_bank_scale": 0.05,
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "altitude_hold_target_m": 7000.0,
    "altitude_hold_range_m": 4000.0,
    "altitude_hold_scale": 0.5,
    "level_pitch_scale": 0.1,
    "level_roll_scale": 0.05,
    "closure_scale": 0.3,
    "closure_reference_mps": 300.0,
    "alignment_scale": 1.0,
    "range_scale": 0.3,
    "range_reference_m": 8000.0,
    "wez_hold_scale": 3.0,
    "fine_alignment_scale": 0.5,
    "fine_alignment_reference_deg": 10.0,
    "moving_alignment_scale": 2.0,
    "moving_alignment_reference_deg": 180.0
  },
  "reward_module": "student.sac_moving_target_reward_v5"
}
```

## Training History

- iter `0`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `1`: reward_mean=`128.98969888888237`, episode_len_mean=`150.0`
- iter `2`: reward_mean=`124.12016456427942`, episode_len_mean=`150.0`
- iter `3`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `4`: reward_mean=`248.49834345786363`, episode_len_mean=`150.0`
- iter `5`: reward_mean=`142.7377478344019`, episode_len_mean=`150.0`
- iter `6`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `7`: reward_mean=`215.31292192578533`, episode_len_mean=`150.0`
- iter `8`: reward_mean=`75.10119938067118`, episode_len_mean=`150.0`
- iter `9`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `10`: reward_mean=`44.68117428219634`, episode_len_mean=`150.0`
- iter `11`: reward_mean=`38.65607719168848`, episode_len_mean=`150.0`
- iter `12`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `13`: reward_mean=`23.394223868995823`, episode_len_mean=`150.0`
- iter `14`: reward_mean=`193.81994824471315`, episode_len_mean=`150.0`
- iter `15`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `16`: reward_mean=`269.7240818431073`, episode_len_mean=`150.0`
- iter `17`: reward_mean=`199.80910631836787`, episode_len_mean=`150.0`
- iter `18`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `19`: reward_mean=`262.352628715529`, episode_len_mean=`150.0`
- iter `20`: reward_mean=`100.35439436651042`, episode_len_mean=`150.0`
- iter `21`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `22`: reward_mean=`34.76779213895157`, episode_len_mean=`150.0`
- iter `23`: reward_mean=`135.44155348962883`, episode_len_mean=`150.0`
- iter `24`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `25`: reward_mean=`148.87522210183175`, episode_len_mean=`150.0`
- iter `26`: reward_mean=`192.3984028850503`, episode_len_mean=`150.0`
- iter `27`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `28`: reward_mean=`44.09790853324563`, episode_len_mean=`150.0`
- iter `29`: reward_mean=`179.7920921445251`, episode_len_mean=`150.0`
