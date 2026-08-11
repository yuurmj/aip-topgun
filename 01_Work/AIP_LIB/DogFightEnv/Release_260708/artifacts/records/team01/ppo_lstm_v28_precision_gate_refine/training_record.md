# Training Record

- Created at: `2026-08-11T18:12:14`
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
  "reward_module": "student.my_reward_v28_lstm",
  "max_engage_time": 8.0,
  "episode_step_limit": 480,
  "lr": 1e-05,
  "gamma": 0.99,
  "train_batch_size": 1024,
  "minibatch_size": 128,
  "gae_lambda": 0.95,
  "clip_param": 0.05,
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
  "output_tag": "ppo_lstm_v28_precision_gate_refine",
  "notes": "v28 restarts refinement from v26 checkpoint_000012, not from v27. The task remains 100% 900 m autopilot. Reward shaping is precision-gated: when ATA is above 5 deg, posture penalties are largely relaxed so the policy can move the nose; from 5 deg toward the true firing half-angle, roll/pitch stabilization ramps in; inside the real firing cone, cone entry, continued cone hold, stable posture, and actual target damage dominate. A small extreme-roll guard remains outside the precision gate. PPO fine-tuning is deliberately very small (lr=1e-5, clip=0.05, three updates). Compare every checkpoint and stop using later checkpoints if ATA<=1 occupancy or real damage falls.",
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
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_ppo_lstm_v28_precision_gate_refine.yaml"
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
        "name": "v28_lstm_900m_auto_center",
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
        "name": "v28_lstm_900m_auto_left",
        "weight": 0.4,
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
        "name": "v28_lstm_900m_auto_right",
        "weight": 0.4,
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
  "reward_module": "student.my_reward_v28_lstm",
  "reward": {
    "step_penalty": -0.005,
    "survival_bonus": 0.0,
    "damage_scale": 350.0,
    "target_damage_scale": 350.0,
    "ownship_damage_scale": 100.0,
    "firing_confirmation_bonus": 0.03,
    "firing_cone_bonus_max": 0.26,
    "firing_cone_edge_bonus": 0.17,
    "near_cone_outer_deg": 3.0,
    "near_cone_outer_bonus": 0.01,
    "aim_fade_deg": 5.0,
    "ata_improvement_scale": 0.024,
    "ata_improvement_cap": 0.1,
    "ata_worsening_scale": 0.028,
    "ata_worsening_cap": 0.12,
    "stable_firing_cone_bonus": 0.1,
    "cone_entry_bonus": 0.06,
    "cone_hold_bonus": 0.1,
    "cone_exit_penalty": 0.16,
    "precision_gate_outer_deg": 5.0,
    "near_aim_improvement_multiplier": 1.35,
    "stable_cone_posture_bonus": 0.06,
    "wez_misaligned_penalty": 0.05,
    "wez_misaligned_start_deg": 10.0,
    "approach_progress_m": 2.0,
    "good_approach_bonus": 0.04,
    "bad_approach_penalty": 0.06,
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
    "stable_roll_deg": 15.0,
    "roll_penalty_strong_deg": 45.0,
    "roll_penalty_very_strong_deg": 55.0,
    "roll_recovery_scale": 0.015,
    "roll_recovery_cap": 0.08,
    "roll_worsening_scale": 0.02,
    "roll_worsening_cap": 0.1,
    "roll_15_45_penalty": 0.3,
    "roll_45_55_penalty": 0.4,
    "roll_over_55_penalty": 0.9,
    "unfocused_extreme_roll_deg": 120.0,
    "unfocused_extreme_roll_penalty": 0.1,
    "stable_pitch_deg": 5.0,
    "pitch_penalty_strong_deg": 15.0,
    "pitch_recovery_scale": 0.015,
    "pitch_recovery_cap": 0.06,
    "pitch_worsening_scale": 0.02,
    "pitch_worsening_cap": 0.08,
    "pitch_absolute_penalty": 0.25,
    "pitch_extreme_penalty": 0.35,
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

- iter `0`: reward_mean=`-19.55367579146545`, episode_len_mean=`80.0`
- iter `1`: reward_mean=`-34.71348258775314`, episode_len_mean=`80.0`
- iter `2`: reward_mean=`-13.12487461201523`, episode_len_mean=`80.0`
