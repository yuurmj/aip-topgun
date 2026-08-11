# Training Record

- Created at: `2026-08-11T18:36:30`
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
- Step penalty: `-0.005`
- Damage scale: `350.0`
- Pursuit scale: `0.0`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `1.0`
- Win reward: `150.0`
- Loss reward: `-200.0`
- Draw reward: `0.0`

## CLI Arguments

```json
{
  "algorithm": "ppo",
  "iterations": 3,
  "framework": "torch",
  "num_env_runners": 1,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "8",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "autopilot",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.my_reward_v29_lstm",
  "max_engage_time": 8.0,
  "episode_step_limit": 480,
  "lr": 5e-06,
  "gamma": 0.99,
  "train_batch_size": 1024,
  "minibatch_size": 128,
  "gae_lambda": 0.95,
  "clip_param": 0.03,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": null,
  "model_fcnet_hiddens": "128,128",
  "model_fcnet_activation": "relu",
  "model_head_fcnet_hiddens": "",
  "model_head_fcnet_activation": "relu",
  "model_vf_share_layers": "true",
  "network_spec_json": "",
  "use_lstm": true,
  "use_lstm_sac": false,
  "lstm_scope": "actor_only",
  "lstm_cell_size": 64,
  "max_seq_len": 8,
  "debug_io": false,
  "use_lstm_prioritized_replay": false,
  "output_name": "team01",
  "output_tag": "ppo_lstm_v29_softroll_precision_refine",
  "notes": "v29 restarts from v26 checkpoint_000012 and keeps the exact v26 900 m center/left/right distribution (20/30/50).  It introduces a global soft roll cap: bank <=45 deg remains free, 45-60 deg receives only a mild penalty, and >60 deg is increasingly suppressed.  Near the real firing cone the allowable bank and pitch shrink smoothly, but posture itself is not rewarded at ATA 3-5 deg.  Precision shaping rewards directional progress 5->3->2->the real firing half-angle, then cone entry/hold and actual target damage dominate. Fine-tuning is intentionally tiny (lr=5e-6, clip=0.03, 3 updates); compare all three checkpoints and keep v26 checkpoint_000012 if damage or ATA<=1 occupancy does not improve.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 1,
  "save_native_checkpoint": true,
  "restore_checkpoint": "C:/Users/Ran/Desktop/2026/aip-topgun/01_Work/AIP_LIB/DogFightEnv/Release_260708/artifacts/checkpoints/team01/ppo_lstm_v26_attack_bootstrap/checkpoint_000012",
  "init_bundle": "",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 1,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 1,
  "policy_probe_steps": 8,
  "no_policy_probe_print": false,
  "engagement_log_interval": 1,
  "engagement_log_steps": 480,
  "engagement_log_episodes": 24,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_ppo_lstm_v29_softroll_precision_refine.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "autopilot",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 8.0,
  "episode_step_limit": 480,
  "step_ratio": 6,
  "min_altitude": 6550.0,
  "geometry_guard": {
    "enabled": false
  },
  "initial_scenario": {
    "mode": "scenario_pool",
    "scenarios": [
      {
        "name": "v29_lstm_900m_auto_center",
        "weight": 0.2,
        "ownship": [
          6900.0,
          0.0,
          -7000.0,
          0.0,
          0.0,
          180.0,
          264.0
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
      },
      {
        "name": "v29_lstm_900m_auto_left",
        "weight": 0.3,
        "ownship": [
          6899.7777503,
          -20.0,
          -7000.0,
          0.0,
          0.0,
          179.5,
          264.0
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
      },
      {
        "name": "v29_lstm_900m_auto_right",
        "weight": 0.5,
        "ownship": [
          6899.7777503,
          20.0,
          -7000.0,
          0.0,
          0.0,
          180.5,
          264.0
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
    ],
    "ownship_randomization": {
      "enabled": false
    },
    "target_randomization": {
      "enabled": false
    },
    "shared_randomization": {
      "enabled": false
    }
  },
  "reward_module": "student.my_reward_v29_lstm",
  "reward": {
    "step_penalty": -0.005,
    "survival_bonus": 0.0,
    "damage_scale": 350.0,
    "target_damage_scale": 350.0,
    "ownship_damage_scale": 120.0,
    "firing_confirmation_bonus": 0.03,
    "firing_cone_bonus_max": 0.26,
    "firing_cone_edge_bonus": 0.18,
    "near_cone_outer_deg": 3.0,
    "near_cone_outer_bonus": 0.008,
    "aim_fade_deg": 5.0,
    "ata_improvement_scale_far": 0.018,
    "ata_improvement_scale_near": 0.045,
    "ata_improvement_cap": 0.12,
    "ata_worsening_scale_far": 0.022,
    "ata_worsening_scale_near": 0.05,
    "ata_worsening_cap": 0.14,
    "precision_outer_deg": 5.0,
    "precision_progress_bonus_scale": 0.14,
    "precision_progress_penalty_scale": 0.16,
    "milestone_3deg_bonus": 0.015,
    "milestone_2deg_bonus": 0.025,
    "cone_entry_bonus": 0.08,
    "cone_hold_bonus": 0.12,
    "cone_exit_penalty": 0.18,
    "stable_firing_cone_bonus": 0.1,
    "stable_cone_posture_bonus": 0.05,
    "wez_misaligned_penalty": 0.05,
    "wez_misaligned_start_deg": 10.0,
    "approach_progress_m": 2.0,
    "good_approach_bonus": 0.035,
    "bad_approach_penalty": 0.055,
    "target_altitude_m": 7000.0,
    "altitude_tolerance_m": 30.0,
    "altitude_recovery_scale": 0.015,
    "altitude_recovery_cap": 0.08,
    "descent_penalty_scale": 0.03,
    "descent_penalty_cap": 0.12,
    "low_altitude_warning_m": 6950.0,
    "low_altitude_strong_m": 6900.0,
    "episode_min_altitude_m": 6550.0,
    "low_altitude_penalty": 1.0,
    "low_altitude_warning_penalty": 0.05,
    "low_altitude_strong_penalty": 1.0,
    "global_roll_free_deg": 45.0,
    "global_roll_soft_deg": 60.0,
    "global_roll_hard_deg": 90.0,
    "global_roll_soft_penalty": 0.1,
    "global_roll_hard_penalty": 0.3,
    "global_roll_extreme_penalty": 0.45,
    "near_roll_limit_outer_deg": 45.0,
    "near_roll_limit_cone_deg": 20.0,
    "near_roll_penalty_max": 0.18,
    "stable_roll_deg": 15.0,
    "global_pitch_free_deg": 15.0,
    "global_pitch_penalty_max": 0.18,
    "near_pitch_limit_outer_deg": 15.0,
    "near_pitch_limit_cone_deg": 7.0,
    "near_pitch_penalty_max": 0.12,
    "stable_pitch_deg": 5.0,
    "dive_pitch_start_deg": -2.0,
    "dive_pitch_full_deg": -10.0,
    "dive_descent_penalty": 0.35,
    "safety_reward_min": -3.0,
    "win_reward": 150.0,
    "loss_reward": -200.0,
    "crash_penalty": -300.0,
    "guard_fail_penalty": -80.0,
    "target_crash_reward": 0.0,
    "timeout_penalty": 0.0,
    "draw_reward": 0.0
  },
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4000000000001
  }
}
```

## Training History

- iter `0`: reward_mean=`-72.5911203979832`, episode_len_mean=`80.0`
- iter `1`: reward_mean=`-78.43324683449507`, episode_len_mean=`80.0`
- iter `2`: reward_mean=`-51.70487358217336`, episode_len_mean=`80.0`
