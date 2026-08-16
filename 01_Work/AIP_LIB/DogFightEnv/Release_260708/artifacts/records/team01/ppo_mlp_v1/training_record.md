# Training Record

- Created at: `2026-07-20T20:18:09`
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
- Damage scale: `20.0`
- Pursuit scale: `0.3`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `0.1`
- Win reward: `100.0`
- Loss reward: `-100.0`
- Draw reward: `-30.0`

## CLI Arguments

```json
{
  "algorithm": "ppo",
  "iterations": 50,
  "framework": "torch",
  "num_env_runners": 1,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "",
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
  "output_tag": "ppo_mlp_v1",
  "notes": "Student PPO MLP baseline.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 0,
  "save_native_checkpoint": false,
  "restore_checkpoint": "",
  "init_bundle": "",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 0,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 10,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\dev\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\student_ppo_mlp.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "behavior_tree",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "step_ratio": 6,
  "reward": {
    "step_penalty": -0.01,
    "damage_scale": 20.0,
    "pursuit_scale": 0.3,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "low_altitude_penalty": 0.1,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "draw_reward": -30.0,
    "guard_fail_penalty": -50.0,
    "mode": "default"
  },
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4
  }
}
```

## Training History

- iter `0`: reward_mean=`-32.66398370939858`, episode_len_mean=`352.0`
- iter `1`: reward_mean=`-31.753422600309126`, episode_len_mean=`396.6666666666667`
- iter `2`: reward_mean=`-30.087076053857007`, episode_len_mean=`381.0`
- iter `3`: reward_mean=`-32.406708737198436`, episode_len_mean=`397.5`
- iter `4`: reward_mean=`-31.612829183725772`, episode_len_mean=`460.5`
- iter `5`: reward_mean=`-32.54660643814814`, episode_len_mean=`481.0`
- iter `6`: reward_mean=`-32.518632893857024`, episode_len_mean=`463.0`
- iter `7`: reward_mean=`-30.756529994104497`, episode_len_mean=`478.5`
- iter `8`: reward_mean=`-30.89281431861513`, episode_len_mean=`405.5`
- iter `9`: reward_mean=`-16.596933947262528`, episode_len_mean=`508.0`
- iter `10`: reward_mean=`-32.45918434665294`, episode_len_mean=`446.3333333333333`
- iter `11`: reward_mean=`-32.151553505325765`, episode_len_mean=`476.0`
- iter `12`: reward_mean=`-33.240989096225306`, episode_len_mean=`464.5`
- iter `13`: reward_mean=`-33.82366483895524`, episode_len_mean=`520.0`
- iter `14`: reward_mean=`-33.592590566212586`, episode_len_mean=`485.0`
- iter `15`: reward_mean=`-33.51680954202296`, episode_len_mean=`514.0`
- iter `16`: reward_mean=`-17.292008445366907`, episode_len_mean=`562.5`
- iter `17`: reward_mean=`-15.93388083110002`, episode_len_mean=`521.5`
- iter `18`: reward_mean=`-1.4951296260014613`, episode_len_mean=`600.0`
- iter `19`: reward_mean=`-3.1082935150600237`, episode_len_mean=`600.0`
- iter `20`: reward_mean=`-3.362362900678976`, episode_len_mean=`600.0`
- iter `21`: reward_mean=`-5.061082152289089`, episode_len_mean=`600.0`
- iter `22`: reward_mean=`-0.34458911442607243`, episode_len_mean=`600.0`
- iter `23`: reward_mean=`-2.4887544833795623`, episode_len_mean=`600.0`
- iter `24`: reward_mean=`-2.6565436246736507`, episode_len_mean=`600.0`
- iter `25`: reward_mean=`-2.959345027345359`, episode_len_mean=`600.0`
- iter `26`: reward_mean=`-2.1013776846733356`, episode_len_mean=`600.0`
- iter `27`: reward_mean=`-3.202088322026892`, episode_len_mean=`600.0`
- iter `28`: reward_mean=`-3.677499789818082`, episode_len_mean=`600.0`
- iter `29`: reward_mean=`-0.9075421023255074`, episode_len_mean=`600.0`
- iter `30`: reward_mean=`-0.8298350132868166`, episode_len_mean=`600.0`
- iter `31`: reward_mean=`-1.257587894433112`, episode_len_mean=`600.0`
- iter `32`: reward_mean=`-2.040005657152121`, episode_len_mean=`600.0`
- iter `33`: reward_mean=`-0.6670538372178098`, episode_len_mean=`600.0`
- iter `34`: reward_mean=`-4.219708973834337`, episode_len_mean=`600.0`
- iter `35`: reward_mean=`-1.352752196488137`, episode_len_mean=`600.0`
- iter `36`: reward_mean=`-2.6354984927560916`, episode_len_mean=`600.0`
- iter `37`: reward_mean=`-4.56161279209606`, episode_len_mean=`600.0`
- iter `38`: reward_mean=`-2.4532879146684126`, episode_len_mean=`600.0`
- iter `39`: reward_mean=`-3.0336737307267674`, episode_len_mean=`600.0`
- iter `40`: reward_mean=`-1.498373053695943`, episode_len_mean=`600.0`
- iter `41`: reward_mean=`-0.7334480916604027`, episode_len_mean=`600.0`
- iter `42`: reward_mean=`-1.749637164403064`, episode_len_mean=`600.0`
- iter `43`: reward_mean=`-1.9106144209517852`, episode_len_mean=`600.0`
- iter `44`: reward_mean=`-1.810728359057808`, episode_len_mean=`600.0`
- iter `45`: reward_mean=`-0.43087395401369033`, episode_len_mean=`600.0`
- iter `46`: reward_mean=`-1.6111121725435658`, episode_len_mean=`600.0`
- iter `47`: reward_mean=`-0.8727537580164232`, episode_len_mean=`600.0`
- iter `48`: reward_mean=`-1.839052587484568`, episode_len_mean=`600.0`
- iter `49`: reward_mean=`-1.5768519912682384`, episode_len_mean=`600.0`
