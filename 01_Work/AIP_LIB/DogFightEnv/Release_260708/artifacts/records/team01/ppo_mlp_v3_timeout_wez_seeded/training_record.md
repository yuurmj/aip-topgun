# Training Record

- Created at: `2026-08-03T15:51:52`
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
- Pursuit scale: `0.02`
- Pursuit half angle (deg): `30.0`
- Pursuit range (m): `8000.0`
- Low altitude penalty: `0.2`
- Win reward: `100.0`
- Loss reward: `-100.0`
- Draw reward: `-20.0`

## CLI Arguments

```json
{
  "algorithm": "ppo",
  "iterations": 100,
  "framework": "torch",
  "num_env_runners": 2,
  "num_envs_per_env_runner": 1,
  "rollout_fragment_length": "auto",
  "batch_mode": "truncate_episodes",
  "observation_mode": "tactical16",
  "observation_module": "",
  "target_mode": "fixed",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "reward_module": "student.my_reward",
  "max_engage_time": 120.0,
  "episode_step_limit": 1200,
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
  "output_tag": "ppo_mlp_v3_timeout_wez_seeded",
  "notes": "PPO v3: corrected aspect-angle direction, stronger attack geometry and WEZ rewards, two-stage overshoot penalty, small initial-state randomization, far-distance penalty, and timeout handling for terminated/truncated episodes.",
  "save_lightweight_bundle": true,
  "lightweight_bundle_frequency": 50,
  "save_native_checkpoint": true,
  "restore_checkpoint": "",
  "init_bundle": "",
  "use_tune": false,
  "checkpoint_frequency": 0,
  "native_checkpoint_frequency": 50,
  "dashboard_logdir": "artifacts/dashboard",
  "disable_dashboard_log": false,
  "policy_probe_interval": 25,
  "policy_probe_steps": 4,
  "no_policy_probe_print": false,
  "engagement_log_interval": 25,
  "engagement_log_steps": 1200,
  "engagement_log_episodes": 3,
  "no_engagement_log_print": false,
  "experiment_yaml": "C:\\Users\\Ran\\Desktop\\2026\\aip-topgun\\01_Work\\AIP_LIB\\DogFightEnv\\Release_260708\\experiments\\ppo_v3_timeout_wez_seeded.yaml"
}
```

## Environment Config

```json
{
  "observation_mode": "tactical16",
  "target_mode": "fixed",
  "target_behavior_dll": "AIP_BASE_target.dll",
  "ownship_control_mode": "rl",
  "max_engage_time": 120.0,
  "episode_step_limit": 1200,
  "step_ratio": 6,
  "ownship_randomization": {
    "enabled": true,
    "radius": 100.0,
    "r_roll": 3.0,
    "r_pitch": 3.0,
    "r_heading": 5.0
  },
  "reward": {
    "step_penalty": -0.01,
    "low_altitude_threshold_m": 600.0,
    "low_altitude_penalty": 0.2,
    "pursuit_scale": 0.02,
    "pursuit_half_angle_deg": 30.0,
    "pursuit_range_m": 8000.0,
    "attack_geometry_scale": 0.3,
    "soft_ata_limit_deg": 30.0,
    "soft_aa_limit_deg": 45.0,
    "attack_range_min_m": 152.4,
    "attack_range_best_m": 533.4,
    "attack_range_max_m": 1000.0,
    "ata_stage_max_range_m": 1500.0,
    "ata_under_10_bonus": 0.03,
    "ata_under_5_bonus": 0.08,
    "ata_under_2_bonus": 0.15,
    "ata_under_1_bonus": 0.25,
    "wez_bonus": 1.0,
    "too_close_distance_m": 300.0,
    "too_close_ata_deg": 10.0,
    "too_close_penalty": 0.4,
    "escape_start_m": 3000.0,
    "escape_full_m": 8000.0,
    "escape_penalty_scale": 0.02,
    "damage_scale": 20.0,
    "win_reward": 100.0,
    "loss_reward": -100.0,
    "crash_penalty": -100.0,
    "timeout_penalty": -30.0,
    "draw_reward": -20.0,
    "guard_fail_penalty": -50.0
  },
  "reward_module": "student.my_reward",
  "wez": {
    "angle_deg": 2.0,
    "min_range_m": 152.4,
    "max_range_m": 914.4000000000001
  }
}
```

## Training History

- iter `0`: reward_mean=`-109.11246967165806`, episode_len_mean=`358.5`
- iter `1`: reward_mean=`-112.05444600262425`, episode_len_mean=`420.0`
- iter `2`: reward_mean=`-111.9549974526719`, episode_len_mean=`435.0`
- iter `3`: reward_mean=`-111.7541756185451`, episode_len_mean=`424.25`
- iter `4`: reward_mean=`-112.52486323637582`, episode_len_mean=`462.0`
- iter `5`: reward_mean=`-110.21916181684784`, episode_len_mean=`383.75`
- iter `6`: reward_mean=`-111.50476107949493`, episode_len_mean=`399.0`
- iter `7`: reward_mean=`-113.7408150269634`, episode_len_mean=`468.0`
- iter `8`: reward_mean=`-112.75036687945538`, episode_len_mean=`444.0`
- iter `9`: reward_mean=`-113.72090114517758`, episode_len_mean=`474.5`
- iter `10`: reward_mean=`-73.73279766266765`, episode_len_mean=`478.0`
- iter `11`: reward_mean=`-74.77462881320297`, episode_len_mean=`542.5`
- iter `12`: reward_mean=`-111.2122993631668`, episode_len_mean=`426.5`
- iter `13`: reward_mean=`-78.53289836368185`, episode_len_mean=`546.0`
- iter `14`: reward_mean=`-33.20666914485882`, episode_len_mean=`568.0`
- iter `15`: reward_mean=`-113.40628310990427`, episode_len_mean=`466.0`
- iter `16`: reward_mean=`-112.18335960182688`, episode_len_mean=`446.0`
- iter `17`: reward_mean=`-73.08820610899403`, episode_len_mean=`527.0`
- iter `18`: reward_mean=`-111.51882601996067`, episode_len_mean=`423.0`
- iter `19`: reward_mean=`-114.8897470685685`, episode_len_mean=`509.0`
- iter `20`: reward_mean=`-53.86814723499829`, episode_len_mean=`513.25`
- iter `21`: reward_mean=`-33.07778045593406`, episode_len_mean=`568.0`
- iter `22`: reward_mean=`-33.53023699635908`, episode_len_mean=`568.0`
- iter `23`: reward_mean=`-71.53121686716969`, episode_len_mean=`470.5`
- iter `24`: reward_mean=`-71.16522124514889`, episode_len_mean=`445.0`
- iter `25`: reward_mean=`-33.75475769215919`, episode_len_mean=`568.0`
- iter `26`: reward_mean=`-71.72828429464275`, episode_len_mean=`476.5`
- iter `27`: reward_mean=`-115.33030849727562`, episode_len_mean=`476.25`
- iter `28`: reward_mean=`-116.8340316097809`, episode_len_mean=`483.0`
- iter `29`: reward_mean=`-72.63324951070848`, episode_len_mean=`485.0`
- iter `30`: reward_mean=`-71.31427309788293`, episode_len_mean=`455.5`
- iter `31`: reward_mean=`-76.83098281593055`, episode_len_mean=`484.0`
- iter `32`: reward_mean=`-33.671588094930556`, episode_len_mean=`568.0`
- iter `33`: reward_mean=`-114.94527134442254`, episode_len_mean=`530.0`
- iter `34`: reward_mean=`-55.114809272801935`, episode_len_mean=`526.0`
- iter `35`: reward_mean=`-72.81284269695779`, episode_len_mean=`485.5`
- iter `36`: reward_mean=`-33.37437965892343`, episode_len_mean=`568.0`
- iter `37`: reward_mean=`-34.11754187682618`, episode_len_mean=`568.0`
- iter `38`: reward_mean=`-33.23280742668058`, episode_len_mean=`568.0`
- iter `39`: reward_mean=`nan`, episode_len_mean=`nan`
- iter `40`: reward_mean=`-53.955146473324135`, episode_len_mean=`538.25`
- iter `41`: reward_mean=`-35.19469207811816`, episode_len_mean=`568.0`
- iter `42`: reward_mean=`-34.128833647333124`, episode_len_mean=`568.0`
- iter `43`: reward_mean=`-33.20076822219618`, episode_len_mean=`568.0`
- iter `44`: reward_mean=`-34.01723208716929`, episode_len_mean=`568.0`
- iter `45`: reward_mean=`-72.84236463221512`, episode_len_mean=`494.5`
- iter `46`: reward_mean=`-34.42296113360405`, episode_len_mean=`568.0`
- iter `47`: reward_mean=`-33.210195063044026`, episode_len_mean=`568.0`
- iter `48`: reward_mean=`-33.317148969120865`, episode_len_mean=`568.0`
- iter `49`: reward_mean=`-33.311110950112365`, episode_len_mean=`568.0`
- iter `50`: reward_mean=`-75.00947504275419`, episode_len_mean=`529.0`
- iter `51`: reward_mean=`-33.44193731986711`, episode_len_mean=`568.0`
- iter `52`: reward_mean=`-33.309328960485864`, episode_len_mean=`568.0`
- iter `53`: reward_mean=`-33.58034180240783`, episode_len_mean=`568.0`
- iter `54`: reward_mean=`-33.327042651985415`, episode_len_mean=`568.0`
- iter `55`: reward_mean=`-34.02026292528255`, episode_len_mean=`568.0`
- iter `56`: reward_mean=`-74.22805500429581`, episode_len_mean=`525.5`
- iter `57`: reward_mean=`-33.29585982190868`, episode_len_mean=`568.0`
- iter `58`: reward_mean=`-33.18348952538521`, episode_len_mean=`568.0`
- iter `59`: reward_mean=`-33.06799959789607`, episode_len_mean=`568.0`
- iter `60`: reward_mean=`-33.29324396036233`, episode_len_mean=`568.0`
- iter `61`: reward_mean=`-33.631325397491665`, episode_len_mean=`568.0`
- iter `62`: reward_mean=`-33.18144533147449`, episode_len_mean=`568.0`
- iter `63`: reward_mean=`-33.254792223321616`, episode_len_mean=`568.0`
- iter `64`: reward_mean=`-33.332958802622585`, episode_len_mean=`568.0`
- iter `65`: reward_mean=`-72.11280864834563`, episode_len_mean=`496.0`
- iter `66`: reward_mean=`-33.45445459789724`, episode_len_mean=`568.0`
- iter `67`: reward_mean=`-33.34295680337602`, episode_len_mean=`568.0`
- iter `68`: reward_mean=`-72.4641051186919`, episode_len_mean=`494.5`
- iter `69`: reward_mean=`-74.61716693586618`, episode_len_mean=`568.0`
- iter `70`: reward_mean=`-74.10185399584512`, episode_len_mean=`533.0`
- iter `71`: reward_mean=`-71.47856746975103`, episode_len_mean=`458.0`
- iter `72`: reward_mean=`-114.74226251186246`, episode_len_mean=`481.0`
- iter `73`: reward_mean=`-33.5085648703393`, episode_len_mean=`568.0`
- iter `74`: reward_mean=`-33.58033558690771`, episode_len_mean=`568.0`
- iter `75`: reward_mean=`-33.59393505092801`, episode_len_mean=`568.0`
- iter `76`: reward_mean=`-33.66080064150833`, episode_len_mean=`568.0`
- iter `77`: reward_mean=`-33.537319422094306`, episode_len_mean=`568.0`
- iter `78`: reward_mean=`-33.632066594927934`, episode_len_mean=`568.0`
- iter `79`: reward_mean=`-33.70289911980528`, episode_len_mean=`568.0`
- iter `80`: reward_mean=`-33.63171136963007`, episode_len_mean=`568.0`
- iter `81`: reward_mean=`-33.703825068444665`, episode_len_mean=`568.0`
- iter `82`: reward_mean=`-33.70416919586494`, episode_len_mean=`568.0`
- iter `83`: reward_mean=`-46.940978295953315`, episode_len_mean=`568.0`
- iter `84`: reward_mean=`-33.63803462304753`, episode_len_mean=`568.0`
- iter `85`: reward_mean=`-33.56530085631506`, episode_len_mean=`568.0`
- iter `86`: reward_mean=`-33.674689577859375`, episode_len_mean=`568.0`
- iter `87`: reward_mean=`-33.57164410495531`, episode_len_mean=`568.0`
- iter `88`: reward_mean=`-33.56642349039942`, episode_len_mean=`568.0`
- iter `89`: reward_mean=`-33.62049164218516`, episode_len_mean=`568.0`
- iter `90`: reward_mean=`-33.600222484391246`, episode_len_mean=`568.0`
- iter `91`: reward_mean=`-33.593288741430285`, episode_len_mean=`568.0`
- iter `92`: reward_mean=`-33.613926193674295`, episode_len_mean=`568.0`
- iter `93`: reward_mean=`-76.40358708273605`, episode_len_mean=`522.0`
- iter `94`: reward_mean=`-33.52102732137706`, episode_len_mean=`568.0`
- iter `95`: reward_mean=`-33.57920725099101`, episode_len_mean=`568.0`
- iter `96`: reward_mean=`-33.67800797688979`, episode_len_mean=`568.0`
- iter `97`: reward_mean=`-74.47289852836452`, episode_len_mean=`548.0`
- iter `98`: reward_mean=`-33.610690677032`, episode_len_mean=`568.0`
- iter `99`: reward_mean=`-75.10773570341708`, episode_len_mean=`557.0`
