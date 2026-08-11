# Training Record

- Created at: `2026-08-04T17:12:02`
- Python: `3.11.15`
- Platform: `Windows-10-10.0.26200-SP0`
- Algorithm: `ppo`

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
- Damage scale: `30.0`
- Pursuit scale: `0.05`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `2500.0`
- Low altitude penalty: `0.5`
- Win reward: `150.0`
- Loss reward: `-150.0`
- Draw reward: `-10.0`

## CLI Arguments

```json
{
  "algorithm": "ppo",
  "iterations": 10,
  "framework": "torch",
  "num_env_runners": 1,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "autopilot",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.my_reward",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "lr": 0.0003,
  "gamma": 0.99,
  "train_batch_size": 1024,
  "minibatch_size": 128,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": null,
  "model_fcnet_hiddens": "256,256",
  "model_fcnet_activation": "relu",
  "model_head_fcnet_hiddens": "",
  "model_head_fcnet_activation": "relu",
  "model_vf_share_layers": "true",
  "network_spec_json": "",
  "use_lstm": false,
  "use_lstm_sac": false,
  "lstm_scope": "actor_only",
  "lstm_cell_size": 64,
  "max_seq_len": 8,
  "debug_io": false,
  "use_lstm_prioritized_replay": false,
  "output_name": "team01",
  "output_tag": "ppo_mlp_v11_wez_hold_roll_stability",
  "notes": "",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 5,
  "save_native_checkpoint": false,
  "restore_checkpoint": "",
  "init_bundle": "",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 0,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 3,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 3,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 10,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_ppo_v11_wez_hold_roll_stability.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "autopilot",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "step_ratio": 6,
  "target_autopilot": {
    "heading_cmd": 180.0,
    "altitude_cmd": -7000.0,
    "speed_cmd": 250.0
  },
  "initial_scenario": {
    "mode": "scenario_pool",
    "scenarios": [
      {
        "name": "wez_hold_700m",
        "weight": 1.0,
        "ownship": [
          6700.0,
          0.0,
          -7000.0,
          0.0,
          0.0,
          180.0,
          250.0
        ],
        "target": [
          6000.0,
          0.0,
          -7000.0,
          0.0,
          0.0,
          180.0,
          250.0
        ],
        "target_mode": "autopilot",
        "target_autopilot": {
          "heading_cmd": 180.0,
          "altitude_cmd": -7000.0,
          "speed_cmd": 250.0
        }
      }
    ]
  },
  "reward_module": "student.my_reward",
  "reward": {
    "survival_bonus": 0.0,
    "damage_scale": 30.0,
    "pursuit_scale": 0.05,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 2500.0,
    "step_penalty": -0.01,
    "alignment_scale": 0.05,
    "pre_wez_outer_margin_m": 1500.0,
    "coarse_aim_angle_deg": 30.0,
    "coarse_aim_bonus": 0.08,
    "fine_aim_angle_deg": 10.0,
    "fine_aim_bonus": 0.12,
    "wez_hold_bonus": 0.75,
    "wez_miss_angle_deg": 10.0,
    "wez_miss_penalty": 0.75,
    "target_damage_scale": 30.0,
    "ownship_damage_scale": 25.0,
    "safe_altitude_min_m": 5000.0,
    "safe_altitude_max_m": 8000.0,
    "safe_altitude_bonus": 0.01,
    "low_altitude_warning_m": 5000.0,
    "low_altitude_m": 3000.0,
    "low_altitude_penalty": 0.5,
    "roll_penalty_full_deg": 60.0,
    "roll_penalty": 0.3,
    "level_pitch_start_deg": 3.0,
    "level_pitch_full_deg": 25.0,
    "level_pitch_penalty": 0.2,
    "dive_pitch_start_deg": -3.0,
    "dive_pitch_full_deg": -25.0,
    "dive_pitch_penalty": 0.3,
    "bad_close_range_m": 300.0,
    "bad_close_ata_deg": 30.0,
    "bad_close_penalty": 0.2,
    "win_reward": 150.0,
    "loss_reward": -150.0,
    "crash_penalty": -200.0,
    "guard_fail_penalty": -50.0,
    "target_crash_reward": -100.0,
    "timeout_penalty": -100.0,
    "draw_reward": -10.0
  },
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4000000000001
  }
}
```

## Training History

- iter `0`: reward_mean=`-556.191537245683`, episode_len_mean=`371.5`
- iter `1`: reward_mean=`-508.69676673681533`, episode_len_mean=`521.0`
- iter `2`: reward_mean=`-584.0176692414769`, episode_len_mean=`418.3333333333333`
- iter `3`: reward_mean=`-603.0444783497996`, episode_len_mean=`397.5`
- iter `4`: reward_mean=`-412.49442182857126`, episode_len_mean=`474.0`
- iter `5`: reward_mean=`-567.630743541694`, episode_len_mean=`367.3333333333333`
- iter `6`: reward_mean=`-571.6308170383875`, episode_len_mean=`393.0`
- iter `7`: reward_mean=`-459.63140526215193`, episode_len_mean=`544.5`
- iter `8`: reward_mean=`-636.3878507167005`, episode_len_mean=`470.6666666666667`
- iter `9`: reward_mean=`-613.0143757957882`, episode_len_mean=`444.0`
