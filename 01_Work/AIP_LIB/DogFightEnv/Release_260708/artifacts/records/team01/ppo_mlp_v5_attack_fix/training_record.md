# Training Record

- Created at: `2026-08-03T22:47:19`
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
  "target_mode": "fixed",
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
  "output_tag": "ppo_mlp_v5_attack_fix",
  "notes": "PPO v5 attack fix: stronger pursuit/WEZ/damage rewards and target-crash penalty, preserving v3 logging fields.",
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
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_ppo_v5_attack_fix.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "fixed",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "step_ratio": 6,
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

- iter `0`: reward_mean=`-153.84567994971368`, episode_len_mean=`444.0`
- iter `1`: reward_mean=`-154.15140507327095`, episode_len_mean=`383.6666666666667`
- iter `2`: reward_mean=`-153.92718694712215`, episode_len_mean=`360.5`
- iter `3`: reward_mean=`-154.6516555984035`, episode_len_mean=`424.0`
- iter `4`: reward_mean=`-155.1782034032366`, episode_len_mean=`418.5`
- iter `5`: reward_mean=`-153.9317368637738`, episode_len_mean=`381.3333333333333`
- iter `6`: reward_mean=`-155.06961799459287`, episode_len_mean=`468.0`
- iter `7`: reward_mean=`-153.83707783333094`, episode_len_mean=`409.6666666666667`
- iter `8`: reward_mean=`-154.60423153269335`, episode_len_mean=`432.0`
- iter `9`: reward_mean=`-154.83873089761286`, episode_len_mean=`389.5`
- iter `10`: reward_mean=`-154.9375587015871`, episode_len_mean=`424.0`
- iter `11`: reward_mean=`-154.16988544743748`, episode_len_mean=`438.0`
- iter `12`: reward_mean=`-154.98738536036498`, episode_len_mean=`402.0`
- iter `13`: reward_mean=`-155.46514171997882`, episode_len_mean=`486.0`
- iter `14`: reward_mean=`-154.58366844893044`, episode_len_mean=`430.5`
- iter `15`: reward_mean=`-154.6594960551258`, episode_len_mean=`418.3333333333333`
- iter `16`: reward_mean=`-154.91393800931166`, episode_len_mean=`467.5`
- iter `17`: reward_mean=`-154.9039645613314`, episode_len_mean=`477.5`
- iter `18`: reward_mean=`-155.13212459443002`, episode_len_mean=`499.5`
- iter `19`: reward_mean=`-154.93826139090862`, episode_len_mean=`436.0`
- iter `20`: reward_mean=`-154.7192579991562`, episode_len_mean=`461.0`
- iter `21`: reward_mean=`-155.46430618660057`, episode_len_mean=`450.0`
- iter `22`: reward_mean=`-156.1520556887326`, episode_len_mean=`476.5`
- iter `23`: reward_mean=`-154.80389830730317`, episode_len_mean=`434.0`
- iter `24`: reward_mean=`-102.72305481640099`, episode_len_mean=`568.0`
- iter `25`: reward_mean=`-156.06677437290483`, episode_len_mean=`525.0`
- iter `26`: reward_mean=`-154.50169337744788`, episode_len_mean=`431.6666666666667`
- iter `27`: reward_mean=`-154.52645892405587`, episode_len_mean=`428.0`
- iter `28`: reward_mean=`-154.29720663642703`, episode_len_mean=`430.6666666666667`
- iter `29`: reward_mean=`-154.71956411529558`, episode_len_mean=`427.0`
