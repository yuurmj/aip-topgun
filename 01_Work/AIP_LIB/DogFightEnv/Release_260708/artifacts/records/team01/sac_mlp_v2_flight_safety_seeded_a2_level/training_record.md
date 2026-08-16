# Training Record

- Created at: `2026-08-03T05:33:32`
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
- Damage scale: `0.0`
- Pursuit scale: `0.0`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `3000.0`
- Low altitude penalty: `None`
- Win reward: `100.0`
- Loss reward: `-100.0`
- Draw reward: `-10.0`

## CLI Arguments

```json
{
  "algorithm": "sac",
  "iterations": 100,
  "framework": "torch",
  "num_env_runners": 0,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.sac_flight_safety_reward_v2",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "lr": 0.0003,
  "gamma": 0.99,
  "train_batch_size": 256,
  "minibatch_size": 256,
  "gae_lambda": 0.95,
  "clip_param": 0.2,
  "tau": 0.005,
  "target_entropy": "auto",
  "replay_buffer_capacity": 10000,
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
  "output_tag": "sac_mlp_v2_flight_safety_seeded_a2_level",
  "notes": "SAC survival attempt 2: warm-start with early altitude-hold and level attitude shaping.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 10,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "artifacts/models/team01/sac_mlp_v2_flight_safety_seeded",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 50,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 10,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 0,
  "engagement_log_steps": 600,
  "engagement_log_episodes": 1,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\dev\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\team01_sac_mlp_v2_flight_safety_seeded_a2_level.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "loiter",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 60.0,
  "episode_step_limit": 3600,
  "step_ratio": 6,
  "min_altitude": 300.0,
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
    0.0,
    300.0
  ],
  "target_loiter": {
    "enabled": true,
    "bank": 30.0,
    "pitch": 0.0
  },
  "ownship_randomization": {
    "enabled": true,
    "radius": 250.0,
    "r_roll": 5.0,
    "r_pitch": 5.0,
    "r_heading": 15.0
  },
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4
  },
  "reward": {
    "step_penalty": -0.01,
    "survival_bonus": 0.05,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "ownship_crash_reward": -120.0,
    "target_crash_reward": 100.0,
    "timeout_reward": 0.0,
    "draw_reward": -10.0,
    "guard_fail_penalty": -100.0,
    "damage_scale": 0.0,
    "altitude_warning_m": 2000.0,
    "altitude_floor_m": 300.0,
    "altitude_risk_scale": 0.1,
    "dive_penalty_scale": 0.05,
    "dive_reference_deg": 45.0,
    "excess_bank_deg": 60.0,
    "excess_bank_scale": 0.02,
    "pursuit_scale": 0.0,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 3000.0,
    "altitude_hold_target_m": 7000.0,
    "altitude_hold_range_m": 3000.0,
    "altitude_hold_scale": 0.05,
    "level_pitch_scale": 0.03,
    "level_roll_scale": 0.01
  },
  "reward_module": "student.sac_flight_safety_reward_v2"
}
```

## Training History

- iter `0`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `1`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `2`: reward_mean=`n/a`, episode_len_mean=`n/a`
- iter `3`: reward_mean=`-126.02525259470266`, episode_len_mean=`375.0`
- iter `4`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `5`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `6`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `7`: reward_mean=`-125.26253555742764`, episode_len_mean=`382.0`
- iter `8`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `9`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `10`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `11`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `12`: reward_mean=`-124.45131511928427`, episode_len_mean=`457.0`
- iter `13`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `14`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `15`: reward_mean=`-128.66947181236398`, episode_len_mean=`374.0`
- iter `16`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `17`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `18`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `19`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `20`: reward_mean=`-125.89480647509563`, episode_len_mean=`435.0`
- iter `21`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `22`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `23`: reward_mean=`-127.2997499315121`, episode_len_mean=`359.0`
- iter `24`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `25`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `26`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `27`: reward_mean=`-127.20220978919603`, episode_len_mean=`361.0`
- iter `28`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `29`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `30`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `31`: reward_mean=`-126.3937144931701`, episode_len_mean=`386.0`
- iter `32`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `33`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `34`: reward_mean=`-126.74082683269569`, episode_len_mean=`355.0`
- iter `35`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `36`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `37`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `38`: reward_mean=`-126.41527461409522`, episode_len_mean=`368.0`
- iter `39`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `40`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `41`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `42`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `43`: reward_mean=`-121.3316253256907`, episode_len_mean=`475.0`
- iter `44`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `45`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `46`: reward_mean=`-126.31004684505879`, episode_len_mean=`354.0`
- iter `47`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `48`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `49`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `50`: reward_mean=`-125.9000080678869`, episode_len_mean=`357.0`
- iter `51`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `52`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `53`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `54`: reward_mean=`-126.57620583081459`, episode_len_mean=`379.0`
- iter `55`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `56`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `57`: reward_mean=`-127.30964655663061`, episode_len_mean=`349.0`
- iter `58`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `59`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `60`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `61`: reward_mean=`-125.27113280112043`, episode_len_mean=`407.0`
- iter `62`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `63`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `64`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `65`: reward_mean=`-124.73532705860039`, episode_len_mean=`419.0`
- iter `66`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `67`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `68`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `69`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `70`: reward_mean=`-125.57421132463566`, episode_len_mean=`432.0`
- iter `71`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `72`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `73`: reward_mean=`-127.08001486271331`, episode_len_mean=`348.0`
- iter `74`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `75`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `76`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `77`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `78`: reward_mean=`-123.97628478481784`, episode_len_mean=`494.0`
- iter `79`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `80`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `81`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `82`: reward_mean=`-126.08743965156329`, episode_len_mean=`390.0`
- iter `83`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `84`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `85`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `86`: reward_mean=`-125.26987284227495`, episode_len_mean=`409.0`
- iter `87`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `88`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `89`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `90`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `91`: reward_mean=`-124.7011179967069`, episode_len_mean=`445.0`
- iter `92`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `93`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `94`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `95`: reward_mean=`-125.31807777105591`, episode_len_mean=`404.0`
- iter `96`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `97`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `98`: reward_mean=`-125.69806382544672`, episode_len_mean=`364.0`
- iter `99`: reward_mean=`nan`, episode_len_mean=`nan`
