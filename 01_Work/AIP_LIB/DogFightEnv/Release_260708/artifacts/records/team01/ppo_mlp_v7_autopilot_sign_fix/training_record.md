# Training Record

- Created at: `2026-08-04T11:57:58`
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
- Pursuit scale: `0.08`
- Pursuit half angle (deg): `60.0`
- Pursuit range (m): `6000.0`
- Low altitude penalty: `0.2`
- Win reward: `150.0`
- Loss reward: `-150.0`
- Draw reward: `-10.0`

## CLI Arguments

```json
{
  "algorithm": "ppo",
  "iterations": 30,
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
  "output_tag": "ppo_mlp_v7_autopilot_sign_fix",
  "notes": "",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 10,
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
  "engagement_log_interval": 10,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 10,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_ppo_v7_autopilot_sign_fix.yaml"
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
  "reward_module": "student.my_reward",
  "reward": {
    "survival_bonus": 0.0,
    "damage_scale": 30.0,
    "pursuit_half_angle_deg": 60.0,
    "step_penalty": -0.01,
    "pursuit_scale": 0.08,
    "pursuit_angle_deg": 60.0,
    "pursuit_range_m": 6000.0,
    "attack_min_range_m": 500.0,
    "attack_max_range_m": 920.0,
    "attack_ata_deg": 15.0,
    "attack_position_bonus": 0.1,
    "wez_hold_bonus": 0.15,
    "target_damage_scale": 30.0,
    "ownship_damage_scale": 25.0,
    "low_altitude_m": 600.0,
    "low_altitude_penalty": 0.2,
    "bad_close_range_m": 300.0,
    "bad_close_ata_deg": 30.0,
    "bad_close_penalty": 0.2,
    "win_reward": 150.0,
    "loss_reward": -150.0,
    "crash_penalty": -150.0,
    "guard_fail_penalty": -50.0,
    "target_crash_reward": -100.0,
    "timeout_penalty": -50.0,
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

- iter `0`: reward_mean=`-153.52224024157192`, episode_len_mean=`403.5`
- iter `1`: reward_mean=`-154.23598660124563`, episode_len_mean=`411.0`
- iter `2`: reward_mean=`-152.84892719300228`, episode_len_mean=`339.5`
- iter `3`: reward_mean=`-154.55102543705195`, episode_len_mean=`437.3333333333333`
- iter `4`: reward_mean=`-154.55303274730636`, episode_len_mean=`388.0`
- iter `5`: reward_mean=`-153.8794142604127`, episode_len_mean=`382.0`
- iter `6`: reward_mean=`-154.5598005145224`, episode_len_mean=`391.0`
- iter `7`: reward_mean=`-153.49936156582635`, episode_len_mean=`344.3333333333333`
- iter `8`: reward_mean=`-155.0796241469887`, episode_len_mean=`441.5`
- iter `9`: reward_mean=`-153.3456675728419`, episode_len_mean=`356.6666666666667`
- iter `10`: reward_mean=`-154.47756139631107`, episode_len_mean=`429.0`
- iter `11`: reward_mean=`-153.78406336381042`, episode_len_mean=`361.3333333333333`
- iter `12`: reward_mean=`-154.36642426532708`, episode_len_mean=`417.6666666666667`
- iter `13`: reward_mean=`-154.15233377625577`, episode_len_mean=`390.5`
- iter `14`: reward_mean=`-153.95653097373153`, episode_len_mean=`401.3333333333333`
- iter `15`: reward_mean=`-154.61135493250276`, episode_len_mean=`432.5`
- iter `16`: reward_mean=`-153.77192077562583`, episode_len_mean=`390.3333333333333`
- iter `17`: reward_mean=`-153.8111563705422`, episode_len_mean=`349.3333333333333`
- iter `18`: reward_mean=`-154.3935347404028`, episode_len_mean=`415.5`
- iter `19`: reward_mean=`-154.44725706690218`, episode_len_mean=`430.0`
- iter `20`: reward_mean=`-155.7516269892344`, episode_len_mean=`511.5`
- iter `21`: reward_mean=`-154.98049864310823`, episode_len_mean=`429.6666666666667`
- iter `22`: reward_mean=`-104.50578830246016`, episode_len_mean=`485.5`
- iter `23`: reward_mean=`-155.89798786131982`, episode_len_mean=`477.5`
- iter `24`: reward_mean=`-155.14027616400995`, episode_len_mean=`427.3333333333333`
- iter `25`: reward_mean=`-154.1347903935722`, episode_len_mean=`424.0`
- iter `26`: reward_mean=`-154.27026037267495`, episode_len_mean=`385.0`
- iter `27`: reward_mean=`-121.71796420235295`, episode_len_mean=`477.3333333333333`
- iter `28`: reward_mean=`-155.38318739223723`, episode_len_mean=`459.5`
- iter `29`: reward_mean=`-116.593007726757`, episode_len_mean=`532.0`
