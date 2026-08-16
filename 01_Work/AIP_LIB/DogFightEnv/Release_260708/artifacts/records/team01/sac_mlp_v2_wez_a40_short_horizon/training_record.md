# Training Record

- Created at: `2026-08-11T15:33:42`
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
  "iterations": 50,
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
  "output_tag": "sac_mlp_v2_wez_a40_short_horizon",
  "notes": "A40 15-second scenario-5 bootstrap to increase completed-episode density; formal evaluation remains 60 seconds.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 10,
  "save_native_checkpoint": false,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_mlp_v2_wez_a38_scenario5/bundle_000016",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": null,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 10,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\dev\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_mlp_v2_wez_a40_short_horizon.yaml"
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
      -0.26,
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
    "alignment_scale": 0.3,
    "range_scale": 0.3,
    "range_reference_m": 8000.0,
    "wez_hold_scale": 3.0,
    "fine_alignment_scale": 0.5,
    "fine_alignment_reference_deg": 10.0,
    "moving_alignment_scale": 1.0,
    "moving_alignment_reference_deg": 20.0
  },
  "reward_module": "student.sac_moving_target_reward_v5"
}
```

## Training History

- iter `0`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `1`: reward_mean=`-208.7500123180117`, episode_len_mean=`150.0`
- iter `2`: reward_mean=`-183.8037245274298`, episode_len_mean=`150.0`
- iter `3`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `4`: reward_mean=`-214.56140816478003`, episode_len_mean=`150.0`
- iter `5`: reward_mean=`-165.22842057948918`, episode_len_mean=`150.0`
- iter `6`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `7`: reward_mean=`-181.75491742430216`, episode_len_mean=`150.0`
- iter `8`: reward_mean=`-230.62833844188978`, episode_len_mean=`150.0`
- iter `9`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `10`: reward_mean=`-203.13800155989256`, episode_len_mean=`150.0`
- iter `11`: reward_mean=`-181.6082741425549`, episode_len_mean=`150.0`
- iter `12`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `13`: reward_mean=`-179.69221391780883`, episode_len_mean=`150.0`
- iter `14`: reward_mean=`-182.4342525707801`, episode_len_mean=`150.0`
- iter `15`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `16`: reward_mean=`-192.89780688363697`, episode_len_mean=`150.0`
- iter `17`: reward_mean=`-192.46514576078556`, episode_len_mean=`150.0`
- iter `18`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `19`: reward_mean=`-180.68671695662673`, episode_len_mean=`150.0`
- iter `20`: reward_mean=`-186.90963230065702`, episode_len_mean=`150.0`
- iter `21`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `22`: reward_mean=`-120.80591574130858`, episode_len_mean=`150.0`
- iter `23`: reward_mean=`-199.92331938179578`, episode_len_mean=`150.0`
- iter `24`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `25`: reward_mean=`-185.7432409799783`, episode_len_mean=`150.0`
- iter `26`: reward_mean=`-189.33950578351053`, episode_len_mean=`150.0`
- iter `27`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `28`: reward_mean=`-192.6858983185927`, episode_len_mean=`150.0`
- iter `29`: reward_mean=`-204.66146641917314`, episode_len_mean=`150.0`
- iter `30`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `31`: reward_mean=`-173.18313902917882`, episode_len_mean=`150.0`
- iter `32`: reward_mean=`-195.00197389715092`, episode_len_mean=`150.0`
- iter `33`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `34`: reward_mean=`-186.7872771263346`, episode_len_mean=`150.0`
- iter `35`: reward_mean=`-195.67889249042858`, episode_len_mean=`150.0`
- iter `36`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `37`: reward_mean=`-198.26331041466705`, episode_len_mean=`150.0`
- iter `38`: reward_mean=`-196.850714781127`, episode_len_mean=`150.0`
- iter `39`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `40`: reward_mean=`-205.41027088690285`, episode_len_mean=`150.0`
- iter `41`: reward_mean=`-184.50624380361037`, episode_len_mean=`150.0`
- iter `42`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `43`: reward_mean=`-182.46616161224344`, episode_len_mean=`150.0`
- iter `44`: reward_mean=`-166.40202161510456`, episode_len_mean=`150.0`
- iter `45`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `46`: reward_mean=`-180.76981982209244`, episode_len_mean=`150.0`
- iter `47`: reward_mean=`-179.34736251882242`, episode_len_mean=`150.0`
- iter `48`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `49`: reward_mean=`-202.35970187066056`, episode_len_mean=`150.0`
